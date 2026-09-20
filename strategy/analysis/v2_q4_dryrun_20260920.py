#!/usr/bin/env python3
"""
Q4标定干跑 + Q3权重OOS加权审计 (2026-09-20, 9/17数据态, 只读)

目的 (9/30前最后一轮前置):
  A. Q4标定干跑: 用9/17数据态预跑 calib_2026Q4_tail.py 的计算链
     (窗口2021-10~2026-09, 因子日截断至9/3) —
     ① 验证Q4脚本机制在真实数据上可跑通 (9/30前排雷)
     ② 出 Q4 vs Q3 权重变化早读 (9/30正式标定会怎么调)
     ③ 看Q3翻负的4个基本面因子(fund_score/fund_roe/fund_revenue_growth/
        fund_profit_growth)在Q4窗口的再选择率
  B. Q3权重OOS加权审计: 把早读(v2_oos_preview_0917.py P2')的等权符号一致率
     升级为按2026Q3.yaml权重的组合级IC, 并按月分解(7月/8月/9月) —
     等权83%的结论在"真正影响收益"的权重口径下是否成立。

只读: 不写quarterly_factors/; 因子缓存仅写 /tmp/v2_factor_df_0917.pkl (复用提速)。
9/30正式执行: calib_2026Q4_tail.py (写盘) + v2_oos_runbook_0930.py。

执行: cd strategy && QUANT_ALT_NO_AUTOREFRESH=1 /mnt/d/quant/.venv/bin/python analysis/v2_q4_dryrun_20260920.py
"""
import os
import sys
import gc
import pickle
import yaml
from collections import defaultdict

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.offline_calibration import (
    prepare_calibration_data, compute_factor_data,
    calibrate_industry_regime, select_best_factors,
    _cross_sectional_ic,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
Q_DIR = os.path.join(BASE_DIR, 'config', 'quarterly_factors')
FACTOR_CACHE = '/tmp/v2_factor_df_0917.pkl'  # pickle保真(带object/list列), 读写各~1-2min
CONCEPT_CACHE = '/tmp/v2_concept_map_0917.pkl'

DATA_END = pd.Timestamp('2026-09-17')
FULL_START = pd.Timestamp('2020-06-01')  # 覆盖Q3窗(2021-07)与Q4窗(2021-10)的250日lookback
Q3_WIN = (pd.Timestamp('2021-07-01'), pd.Timestamp('2026-06-30'))
Q4_WIN = (pd.Timestamp('2021-10-01'), pd.Timestamp('2026-09-30'))


def load_quarter_config(q_id):
    with open(os.path.join(Q_DIR, f'{q_id}.yaml'), 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)['industry_factors']


def factor_total_weights(ind_cfg):
    """每因子在yaml中的总权重 (三regime求和) — 组合级影响口径"""
    tw = defaultdict(float)
    for v in ind_cfg.values():
        for fk, wk in (('factors', 'weights'), ('bull_factors', 'bull_weights'),
                       ('bear_factors', 'bear_weights')):
            for f, w in zip(v.get(fk, []) or [], v.get(wk, []) or []):
                tw[f] += float(w)
    return dict(tw)


def factor_select_counts(ind_cfg):
    """每因子被选中(任意regime)的概念数"""
    sc = defaultdict(int)
    for v in ind_cfg.values():
        seen = set()
        for fk in ('factors', 'bull_factors', 'bear_factors'):
            for f in v.get(fk, []) or []:
                seen.add(f)
        for f in seen:
            sc[f] += 1
    return dict(sc)


def daily_ic_series(factor_df, factor_name):
    ics = {}
    for d, g in factor_df.groupby('date'):
        if len(g) < 10:
            continue
        ic = _cross_sectional_ic(g, factor_name, value_col='future_ret', min_samples=10)
        if ic:
            ics[d] = ic[0]  # _cross_sectional_ic返回列表; 单日group→至多1个元素
    return pd.Series(ics)


# ================= Phase 1: 因子数据 (缓存复用) =================
def phase1():
    print("=" * 70)
    print("Phase 1: 尾部因子值计算 (缓存: %s)" % FACTOR_CACHE)
    print("=" * 70)
    if os.path.exists(FACTOR_CACHE) and os.path.exists(CONCEPT_CACHE):
        try:
            factor_df = pd.read_pickle(FACTOR_CACHE)
            with open(CONCEPT_CACHE, 'rb') as f:
                concept_map = pickle.load(f)
            print(f"  缓存命中: {len(factor_df)} 行, "
                  f"{factor_df['date'].min().date()}~{factor_df['date'].max().date()}")
            return factor_df, concept_map
        except Exception as e:
            print(f"  缓存读取失败({e}), 删除重建")
            os.remove(FACTOR_CACHE)
            if os.path.exists(CONCEPT_CACHE):
                os.remove(CONCEPT_CACHE)
    stock_file_map, fundamental_path, regime_lookup, stock_codes, all_dates, concept_map = \
        prepare_calibration_data(start_date=FULL_START, end_date=DATA_END)
    print(f"  股票: {len(stock_codes)}, 交易日: {len(all_dates)}")
    factor_df = compute_factor_data(
        stock_file_map, fundamental_path, regime_lookup, stock_codes, all_dates,
        concept_map=concept_map)
    factor_df['date'] = pd.to_datetime(factor_df['date'])
    factor_df.to_pickle(FACTOR_CACHE)
    with open(CONCEPT_CACHE, 'wb') as f:
        pickle.dump(concept_map, f)
    print(f"  因子数据: {len(factor_df)} 行, {factor_df['code'].nunique()} 只, "
          f"{factor_df['date'].min().date()}~{factor_df['date'].max().date()} "
          f"(已缓存 {FACTOR_CACHE})")
    del stock_file_map, regime_lookup, all_dates
    gc.collect()
    return factor_df, concept_map


# ================= Phase A: Q3权重OOS加权审计 =================
def phase_a(factor_df):
    print("=" * 70)
    print("Phase A: Q3权重OOS加权审计 (yaml权重口径 + 月度分解)")
    print("=" * 70)
    q3_cfg = load_quarter_config('2026Q3')
    tw = factor_total_weights(q3_cfg)
    facts = sorted(tw.keys())
    print(f"  yaml选中因子 {len(facts)} 个; 权重口径下组合级影响 = Σ(行业权重)加权IC")

    calib_df = factor_df[(factor_df['date'] >= Q3_WIN[0]) & (factor_df['date'] <= Q3_WIN[1])]
    q3_df = factor_df[factor_df['date'] >= pd.Timestamp('2026-07-01')]

    rows = []
    for fn in facts:
        ic_calib = daily_ic_series(calib_df, fn)
        ic_q3 = daily_ic_series(q3_df, fn)
        # 月度分解 (Q3内)
        jul = ic_q3[(ic_q3.index >= '2026-07-01') & (ic_q3.index <= '2026-07-31')].mean()
        aug = ic_q3[(ic_q3.index >= '2026-08-01') & (ic_q3.index <= '2026-08-31')].mean()
        sep = ic_q3[ic_q3.index >= '2026-09-01'].mean()
        rows.append({
            'factor': fn, 'tot_w': tw[fn],
            'ic_calib': ic_calib.mean(), 'ic_q3': ic_q3.mean(),
            'jul': jul, 'aug': aug, 'sep': sep,
        })
    r = pd.DataFrame(rows).sort_values('tot_w', ascending=False)
    r['w_share'] = r['tot_w'] / r['tot_w'].sum() * 100
    pd.set_option('display.width', 200)
    print(r.to_string(index=False,
                      float_format=lambda x: f'{x:+.4f}',
                      formatters={'tot_w': '{:.1f}'.format, 'w_share': '{:.1f}%'.format}))

    # 加权组合级IC
    w = r['tot_w'].values
    wsum = w.sum()
    w_ic_calib = (r['ic_calib'] * w).sum() / wsum
    w_ic_q3 = (r['ic_q3'] * w).sum() / wsum
    print(f"\n  加权组合级IC: 标定窗 {w_ic_calib:+.4f} | Q3 {w_ic_q3:+.4f}")
    # 加权符号一致率: 按权重统计"Q3符号==标定窗符号"的因子占比
    pos_w = r.loc[(r['ic_q3'] > 0) & (r['ic_calib'] > 0), 'tot_w'].sum()
    neg_w = r.loc[(r['ic_q3'] < 0) & (r['ic_calib'] < 0), 'tot_w'].sum()
    flip_w = r.loc[(r['ic_q3'] * r['ic_calib'] < 0), 'tot_w'].sum()
    print(f"  加权符号一致: {(pos_w + neg_w) / wsum * 100:.1f}% 权重一致 | "
          f"翻号权重 {flip_w / wsum * 100:.1f}%")
    return r


# ================= Phase B: Q4标定干跑 =================
def phase_b(factor_df, concept_map):
    print("=" * 70)
    print("Phase B: Q4标定干跑 (窗口 %s~%s, 因子日截断9/3, 不写盘)" %
          (Q4_WIN[0].date(), Q4_WIN[1].date()))
    print("=" * 70)
    with open(os.path.join(BASE_DIR, 'config', 'factor_config.yaml'), 'r', encoding='utf-8') as f:
        candidate_factors = yaml.safe_load(f).get('backtest_factors', [])
    window_df = factor_df[(factor_df['date'] >= Q4_WIN[0]) & (factor_df['date'] <= Q4_WIN[1])]
    print(f"  窗口数据: {len(window_df)} 行, {window_df['code'].nunique()} 只, "
          f"{window_df['date'].nunique()} 天, 候选因子 {len(candidate_factors)} 个")
    # concept_map与正式Q4脚本同口径 (prepare_calibration_data返回值透传)
    calib_results = calibrate_industry_regime(window_df, candidate_factors, concept_map=concept_map)
    q4_cfg = select_best_factors(calib_results, window_df, concept_map=concept_map)
    q3_cfg = load_quarter_config('2026Q3')
    print(f"  Q4干跑标定: {len(q4_cfg)} 个概念 | Q3现文件: {len(q3_cfg)} 个概念")

    # 概念级对比: 因子集合(忽略顺序/权重)异同
    def factor_set(cfg, ind):
        v = cfg.get(ind, {})
        s = set()
        for k in ('factors', 'bull_factors', 'bear_factors'):
            s.update(v.get(k, []) or [])
        return frozenset(s)

    common = set(q4_cfg) & set(q3_cfg)
    same_fs = sum(1 for c in common if factor_set(q4_cfg, c) == factor_set(q3_cfg, c))
    diff_fs = len(common) - same_fs
    only_q4 = set(q4_cfg) - set(q3_cfg)
    only_q3 = set(q3_cfg) - set(q4_cfg)
    print(f"  共有概念 {len(common)}: 因子集相同 {same_fs} | 因子集有变 {diff_fs}")
    print(f"  仅Q4 {len(only_q4)} | 仅Q3 {len(only_q3)}")

    # 权重变化幅度
    wdeltas = []
    for c in common:
        for fk, wk in (('factors', 'weights'), ('bull_factors', 'bull_weights'),
                       ('bear_factors', 'bear_weights')):
            old_w = dict(zip(q3_cfg[c].get(fk, []) or [], q3_cfg[c].get(wk, []) or []))
            new_w = dict(zip(q4_cfg[c].get(fk, []) or [], q4_cfg[c].get(wk, []) or []))
            for f in set(old_w) & set(new_w):
                wdeltas.append(abs(float(new_w[f]) - float(old_w[f])))
    wdeltas = pd.Series(wdeltas)
    print(f"  共有因子对权重变化: 中位 {wdeltas.median():.4f} | "
          f"P90 {wdeltas.quantile(0.9):.4f} | >1e-2占 {(wdeltas > 1e-2).mean() * 100:.0f}% | "
          f"最大 {wdeltas.max():.4f}")

    # 选择频次变化: 每因子被选概念数 Q3 vs Q4
    sc3 = factor_select_counts(q3_cfg)
    sc4 = factor_select_counts(q4_cfg)
    allf = sorted(set(sc3) | set(sc4))
    freq = pd.DataFrame({
        'factor': allf,
        'sel_Q3': [sc3.get(f, 0) for f in allf],
        'sel_Q4': [sc4.get(f, 0) for f in allf],
    })
    freq['delta'] = freq['sel_Q4'] - freq['sel_Q3']
    freq = freq.sort_values('delta', ascending=False)
    print("\n  选择频次变化 (Q4干跑 vs Q3现文件, 全候选因子):")
    print(freq.to_string(index=False))
    flipped = ['fund_score', 'fund_roe', 'fund_revenue_growth', 'fund_profit_growth']
    print(f"\n  Q3翻负4因子在Q4干跑的选择频次: " +
          ", ".join(f"{f}: Q3 {sc3.get(f, 0)}→Q4 {sc4.get(f, 0)}" for f in flipped))
    print("  注意: 干跑因子日截断至9/3(缺9/4~9/30), concept_map与正式脚本同口径透传; "
          "9/30正式标定可能有差异, 本干跑=机制验证+方向早读")


def main():
    print(f"Q4标定干跑 + Q3权重OOS加权审计 — 数据态 as-of {DATA_END.date()} (只读)")
    factor_df, concept_map = phase1()
    phase_a(factor_df)
    phase_b(factor_df, concept_map)
    del factor_df
    gc.collect()
    print("=" * 70)
    print("干跑完成。9/30正式: calib_2026Q4_tail.py 写盘 → 全链裁决")


if __name__ == '__main__':
    main()
