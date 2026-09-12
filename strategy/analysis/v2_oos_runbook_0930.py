#!/usr/bin/env python3
"""
V2 runbook: 2026Q3 as-of标定 OOS测试 (2026-09-12 预写, **2026-09-30收盘数据落地后执行**)

背景: 2026Q3滚动权重系9/6补标定(窗口2021Q3~2026Q2, calib_2026Q3_tail.py)。
Q3是该套权重第一个实际执行季度 → Q3实现 = 标定程序的首个OOS持有期。
本脚本按"探针先行、串行执行、不自动跑全链"原则, 分5个Phase:

  Phase 0  前提自检 (秒级, 无计算)
  Phase 1  尾部因子值计算 2020-06-01~2026-09-30 (一次, Phase 2/3共享, ~10-30min)
  Phase 2  Q3 OOS IC审计 (轻量): 2026Q3.yaml选中因子在Q3的截面IC vs 标定窗IC
  Phase 3  标定权重重核 gate态 (轻量): 重跑2026Q3标定程序, diff vs 现2026Q3.yaml
           (E-K1先例: 采纳后重核确认当前代码+数据态下程序仍选出同套权重)
  Phase 4  Q3实现审计 (秒级): 读现有equity_curve, Q3季度收益分解 (若曲线未刷新到9/30则提示)
  Phase 5  决策点: 打印全链清单 (不自动跑! 全链一次~90min, 需用户批准)

只读审计: 本脚本**不写任何文件** (不写quarterly_factors/, 不刷新净值曲线)。
2026Q4滚动权重标定是独立任务, 另行镜像 calib_2026Q3_tail.py 执行。

执行: cd strategy && python analysis/v2_oos_runbook_0930.py
"""
import os
import sys
import gc
import yaml
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.offline_calibration import (
    prepare_calibration_data, compute_factor_data,
    calibrate_industry_regime, select_best_factors,
    _cross_sectional_ic,
)
from core.config_loader import load_config

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
Q_DIR = os.path.join(BASE_DIR, 'config', 'quarterly_factors')
RVR = os.path.join(BASE_DIR, 'rolling_validation_results')
Q3_END = pd.Timestamp('2026-09-30')

CALIB_WINDOW = (pd.Timestamp('2021-07-01'), pd.Timestamp('2026-06-30'))  # 20季度, PIT止于季度前一天
FULL_START, FULL_END = pd.Timestamp('2020-06-01'), Q3_END  # lookback=250交易日 → 起点前推~12个月


def load_quarter_config(q_id):
    with open(os.path.join(Q_DIR, f'{q_id}.yaml'), 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)['industry_factors']


def selected_factors(ind_cfg):
    """2026Q3.yaml中所有被选中的因子名 (去重)"""
    out = set()
    for v in ind_cfg.values():
        for key in ('factors', 'bull_factors', 'bear_factors'):
            out.update(v.get(key, []) or [])
    return sorted(out)


def daily_ic_series(factor_df, factor_name):
    """按日截面IC序列 (Spearman)"""
    ics = {}
    for d, g in factor_df.groupby('date'):
        if len(g) < 10:
            continue
        ic = _cross_sectional_ic(g, factor_name, value_col='future_ret', min_samples=10)
        if ic is not None:
            ics[d] = ic
    return pd.Series(ics)


# ================= Phase 0 =================
def phase0():
    print("=" * 70)
    print("Phase 0: 前提自检")
    print("=" * 70)
    ok = True

    # 1) Q3数据完整: 任一普通股票表有2026-09-30 bar + 指数表
    for probe in ('sh000001', 'sh600519'):
        p = os.path.join(BASE_DIR, '..', 'data/stock_data/backtrader_data', f'{probe}_qfq.csv')
        if os.path.exists(p):
            df = pd.read_csv(p, parse_dates=['datetime'], usecols=['datetime'])
            last = df['datetime'].max()
            print(f"  {probe} 最后bar: {last.date()}")
            if last < Q3_END:
                print("  ✗ 数据未到2026-09-30, 先完成周五收盘下载+refresh_all")
                ok = False
            break
    # 2) index.yaml含2026Q3
    with open(os.path.join(Q_DIR, 'index.yaml'), 'r', encoding='utf-8') as f:
        index = yaml.safe_load(f)
    has_q3 = '2026Q3' in index.get('quarters', {})
    print(f"  index.yaml含2026Q3: {has_q3} | 季度总数 {len(index.get('quarters', {}))}")
    if not has_q3:
        ok = False
    # 3) 2026Q3.yaml存在
    q3_path = os.path.join(Q_DIR, '2026Q3.yaml')
    print(f"  2026Q3.yaml存在: {os.path.exists(q3_path)}")
    ok = ok and os.path.exists(q3_path)
    # 4) 当前净值曲线状态
    eq_path = os.path.join(RVR, 'equity_curve.csv')
    if os.path.exists(eq_path):
        eq = pd.read_csv(eq_path, parse_dates=['date'])['date'].max()
        print(f"  equity_curve.csv最后日期: {eq.date()} (预期9/30前为9/10态)")
    else:
        print("  ✗ equity_curve.csv缺失")
        ok = False
    # 5) sidecar指纹 (gate+#54态应为 ae67c9ee|0)
    fp_path = os.path.join(BASE_DIR, '.signal_code_fp')
    if os.path.exists(fp_path):
        with open(fp_path) as f:
            print(f"  sidecar指纹: {f.read().strip()} (gate+#54态预期 ae67c9ee|0)")
    print("  Phase 0 结论:", "通过" if ok else "阻塞 — 先补数据再跑")
    return ok


# ================= Phase 1 =================
def phase1():
    print("=" * 70)
    print("Phase 1: 尾部因子值计算 (一次, Phase 2/3共享)")
    print("=" * 70)
    stock_file_map, fundamental_path, regime_lookup, stock_codes, all_dates, concept_map = \
        prepare_calibration_data(start_date=FULL_START, end_date=FULL_END)
    print(f"  股票: {len(stock_codes)}, 交易日: {len(all_dates)} "
          f"({all_dates[0].date()}~{all_dates[-1].date()})")
    factor_df = compute_factor_data(
        stock_file_map, fundamental_path, regime_lookup, stock_codes, all_dates,
        concept_map=concept_map)
    if factor_df.empty:
        print("  ✗ 无因子数据")
        return None
    factor_df['date'] = pd.to_datetime(factor_df['date'])
    print(f"  因子数据: {len(factor_df)} 行, {factor_df['code'].nunique()} 只, "
          f"{factor_df['date'].min().date()}~{factor_df['date'].max().date()}")
    del stock_file_map, regime_lookup, all_dates
    gc.collect()
    return factor_df, concept_map


# ================= Phase 2 =================
def phase2(factor_df):
    print("=" * 70)
    print("Phase 2: Q3 OOS IC审计 (2026Q3.yaml选中因子)")
    print("=" * 70)
    ind_cfg = load_quarter_config('2026Q3')
    facts = selected_factors(ind_cfg)
    print(f"  选中因子 {len(facts)} 个 (中性/bull/bear并集)")
    # future_ret只到 9/30 - forward_period(20交易日) ≈ 9/2; Q3内可审计段=7/1~9/2
    q3_lo, q3_hi = pd.Timestamp('2026-07-01'), factor_df['future_ret'].notna() & \
        (factor_df['date'] >= pd.Timestamp('2026-07-01'))
    q3_df = factor_df[factor_df['date'] >= q3_lo].copy()
    q3_max = q3_df[q3_df['future_ret'].notna()]['date'].max()
    print(f"  Q3内可算IC段: 2026-07-01 ~ {q3_max.date()} (future_ret=20日, 尾部无IC)")
    calib_df = factor_df[(factor_df['date'] >= CALIB_WINDOW[0]) &
                         (factor_df['date'] <= CALIB_WINDOW[1])]

    rows = []
    for fn in facts:
        ic_calib = daily_ic_series(calib_df, fn)
        ic_q3 = daily_ic_series(q3_df, fn)
        rows.append({
            'factor': fn,
            'ic_calib_mean': ic_calib.mean(), 'ic_calib_ir': ic_calib.mean() / (ic_calib.std() + 1e-10),
            'ic_q3_mean': ic_q3.mean(), 'ic_q3_ir': ic_q3.mean() / (ic_q3.std() + 1e-10),
            'q3_days': len(ic_q3),
        })
    r = pd.DataFrame(rows).sort_values('ic_q3_mean', ascending=False)
    print(r.to_string(index=False,
                      float_format=lambda x: f'{x:+.4f}',
                      formatters={'q3_days': '{:.0f}'.format}))
    pos = (r['ic_q3_mean'] > 0).sum()
    print(f"\n  Q3 IC>0: {pos}/{len(r)} ({pos/len(r)*100:.0f}%) | "
          f"标定窗IC>0: {(r['ic_calib_mean']>0).sum()}/{len(r)}")
    print("  判定口径: Q3 IC符号与标定窗一致率≥70% 且 均值不崩塌(<标定窗一半) → 标定程序OOS成立")
    return r


# ================= Phase 3 =================
def phase3(factor_df, concept_map):
    print("=" * 70)
    print("Phase 3: 标定权重重核 gate态 (只读diff, 不写文件)")
    print("=" * 70)
    with open(os.path.join(BASE_DIR, 'config', 'factor_config.yaml'), 'r', encoding='utf-8') as f:
        candidate_factors = yaml.safe_load(f).get('backtest_factors', [])
    window_df = factor_df[(factor_df['date'] >= CALIB_WINDOW[0]) &
                          (factor_df['date'] <= CALIB_WINDOW[1])]
    print(f"  窗口 {CALIB_WINDOW[0].date()}~{CALIB_WINDOW[1].date()}: "
          f"{len(window_df)} 行, {window_df['code'].nunique()} 只, 候选因子 {len(candidate_factors)} 个")
    calib_results = calibrate_industry_regime(window_df, candidate_factors, concept_map=concept_map)
    new_cfg = select_best_factors(calib_results, window_df, concept_map=concept_map)
    old_cfg = load_quarter_config('2026Q3')

    # diff: 行业集合 + 每行业因子/权重
    only_new = set(new_cfg) - set(old_cfg)
    only_old = set(old_cfg) - set(new_cfg)
    print(f"  行业: 新标定 {len(new_cfg)} | 现文件 {len(old_cfg)} | "
          f"仅新 {len(only_new)} | 仅旧 {len(only_old)}")
    n_same = n_diff = 0
    for ind in set(new_cfg) & set(old_cfg):
        a, b = new_cfg[ind], old_cfg[ind]
        same = True
        for key in ('factors', 'weights', 'bull_factors', 'bull_weights',
                    'bear_factors', 'bear_weights'):
            va, vb = (a.get(key) or []), (b.get(key) or [])
            if len(va) != len(vb):
                same = False
            else:
                for x, y in zip(va, vb):
                    if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                        if abs(x - y) > 1e-4:
                            same = False
                    elif x != y:
                        same = False
        if same:
            n_same += 1
        else:
            n_diff += 1
            if n_diff <= 5:
                print(f"  差异: {ind}: 新 {a.get('factors')} vs 旧 {b.get('factors')}")
    print(f"  共有行业 {len(set(new_cfg) & set(old_cfg))}: 一致 {n_same} | 有差异 {n_diff}")
    print("  判定口径: 差异=0 → gate态/数据态下程序仍选出同套Q3权重 (E-K1先例通过); "
          "差异>0 → 逐行业核对, 差异来源需归因(数据重建/gate/程序变动)")


# ================= Phase 4 =================
def phase4():
    print("=" * 70)
    print("Phase 4: Q3实现审计 (读现有净值曲线)")
    print("=" * 70)
    eq_path = os.path.join(RVR, 'equity_curve.csv')
    eq = pd.read_csv(eq_path, parse_dates=['date']).set_index('date').iloc[:, 0]
    last = eq.index.max()
    if last < Q3_END:
        print(f"  ✗ 曲线最后日期 {last.date()} < 9/30 — 需先全链刷新 (Phase 5清单), 本Phase跳过")
        return
    q3 = eq[(eq.index >= '2026-07-01') & (eq.index <= '2026-09-30')]
    start_v = eq[eq.index <= '2026-06-30'].iloc[-1]
    q3_ret = (q3.iloc[-1] / start_v - 1) * 100
    q3_dd = (q3 / q3.cummax() - 1).min() * 100
    ytd_ret = (eq.loc[Q3_END] / eq[eq.index <= '2025-12-31'].iloc[-1] - 1) * 100
    print(f"  2026Q3季度收益: {q3_ret:+.2f}% (6/30 {start_v:,.0f} → 9/30 {q3.iloc[-1]:,.0f})")
    print(f"  2026Q3内最大回撤: {q3_dd:.2f}%")
    print(f"  2026YTD收益: {ytd_ret:+.2f}%")
    print(f"  9/10锚点参考: 1,143,938 (gate态) — 9/30全链后与四指标基线比对")


# ================= Phase 5 =================
def phase5():
    print("=" * 70)
    print("Phase 5: 决策点 — 全链刷新清单 (不自动跑)")
    print("=" * 70)
    print("""
  若Phase 2/3判定通过, 且用户批准, 才执行:
    1. cd strategy && python bt_execution.py          # 全链~90min, 串行
       → 产出至9/30的四指标 + backtest_signals.csv + equity_curve.csv
    2. 四指标 vs 当前基线 1,143,938/357.58%/1.5555/27.41% (9/10锚)
       → 增量部分仅是Q3尾部20个交易日的延伸, 期待数字温和外推, 不期待大幅变动
    3. Phase 4 重跑 → 出Q3季度收益分解 (OOS持有期结论)
    4. 首份V2验证报告: Phase 2 IC表 + Phase 3重核结论 + Phase 4 Q3实现
       → 写入 strategy/docs/ 或 rolling_validation_results/
  注意: 全链前不要动 quarterly_factors/ 任何文件; 2026Q4标定是独立任务(镜像
        calib_2026Q3_tail.py, 窗口2021Q4~2026Q3), Q4权重写盘后必须再跑一次
        全链裁决(quarterly_factors不在_signals_stale根集内, 不会自动触发信号重生成)。
  """)


def main():
    if not phase0():
        print("\n阻塞: Phase 0未通过, 退出 (先补9/30数据+refresh_all)")
        return
    factor_df, concept_map = phase1()
    if factor_df is None:
        return
    phase2(factor_df)
    phase3(factor_df, concept_map)
    phase4()
    phase5()


if __name__ == '__main__':
    main()
