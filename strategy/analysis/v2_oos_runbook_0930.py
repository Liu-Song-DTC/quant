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
2026Q4滚动权重标定是独立任务, 镜像已备: analysis/calib_2026Q4_tail.py
(窗口2021Q4~2026Q3, 写盘后必须再跑一次全链裁决 — 见Phase 5)。

只读保证强化 (2026-09-20): alternative_data.py 新增 QUANT_ALT_NO_AUTOREFRESH=1
开关(8处age闸) — 本脚本Phase 1-3本就零alt-data import, 但Phase 5全链前若
alt pkl age>24h, 全链运行会静默重拉改写pkl。9/30全链用:
  QUANT_ALT_NO_AUTOREFRESH=1 python bt_execution.py
(数据由用户下载器+refresh_all受控更新, 运行中零自动刷新, 数据态确定)。

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
        if ic:
            ics[d] = ic[0]  # _cross_sectional_ic返回列表; 单日group→至多1个元素
    return pd.Series(ics)


# ================= Phase 0 =================
def phase0():
    print("=" * 70)
    print("Phase 0: 前提自检")
    print("=" * 70)
    ok = True

    # 1) Q3数据完整: 指数表 + 普通股票表 都须有2026-09-30 bar
    #    (2026-09-21加固: 原实现只查首个存在的文件即break, 若指数已刷新而个股
    #    K线未刷新(部分刷新), Phase 0会误放行且Phase 2的Q3 IC窗口被静默截短。
    #    命名注意: 指数=sh000001_qfq.csv带前缀, 个股=600519_qfq.csv无前缀)
    for probe in ('sh000001', '600519'):
        p = os.path.join(BASE_DIR, '..', 'data/stock_data/backtrader_data', f'{probe}_qfq.csv')
        if os.path.exists(p):
            df = pd.read_csv(p, parse_dates=['datetime'], usecols=['datetime'])
            last = df['datetime'].max()
            print(f"  {probe} 最后bar: {last.date()}")
            if last < Q3_END:
                print("  ✗ 数据未到2026-09-30, 先完成周五收盘下载+refresh_all")
                ok = False
        else:
            print(f"  ✗ {probe} K线文件缺失")
            ok = False
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
        print(f"  equity_curve.csv最后日期: {eq.date()} (预期9/30前为9/15态)")
    else:
        print("  ✗ equity_curve.csv缺失")
        ok = False
    # 5) sidecar指纹 (9/20重锚后生产态应为 6f1cb6b1|0; sidecar在signals.csv同目录)
    fp_path = os.path.join(RVR, '.signal_code_fp')
    if os.path.exists(fp_path):
        with open(fp_path) as f:
            print(f"  sidecar指纹: {f.read().strip()} (C5c态预期 6f1cb6b1|0)")
    # 6) 生产配置快检 (0f池+C5c+0g关闭 三键)
    with open(os.path.join(BASE_DIR, 'config', 'factor_config.yaml'), 'r', encoding='utf-8') as f:
        ytxt = f.read()
    checks = {
        'bp2_score_boost=0.45': 'bp2_score_boost: 0.45' in ytxt,
        'replacement_buffer=0.05': 'replacement_buffer: 0.05' in ytxt,
        'dragon_tiger关闭': 'dragon_tiger_enabled: false' in ytxt or 'dragon_tiger_enabled: False' in ytxt,
    }
    for k, v in checks.items():
        print(f"  yaml {k}: {'✓' if v else '✗ 异常 — 生产配置已漂移, 先对账'}")
        ok = ok and v
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
    # future_ret只到 9/30 - forward_period(yaml=10交易日) ≈ 9/16; Q3内可审计段=7/1~9/16
    q3_df = factor_df[factor_df['date'] >= pd.Timestamp('2026-07-01')].copy()
    q3_max = q3_df[q3_df['future_ret'].notna()]['date'].max()
    print(f"  Q3内可算IC段: 2026-07-01 ~ {q3_max.date()} (future_ret=10日, 尾部无IC)")
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
    print("  判定口径 (9/20早读已预分析): 9/7后已知历史数据修订=9/11基本面更正+"
          "9/12概念PIT/map+9/14 K线重建 → 差异>0为预期, 不自动等于程序/gate破裂。")
    print("    良性签名: ①权重小抖动(≤1e-2量级) ②同因子集重排序(并列IC平局) "
          "③个别概念进出(min_codes阈值边缘); 警报签名: 整组因子替换/行业大换血。")
    print("    9/20实测(9/17态): 248/399一致, 151差异中weights抖动141(87个>1e-2, "
          "最大0.084), factors重排序58, 概念进出5 — 良性为主; 9/30报告按此签名分级")


# ================= Phase 4 =================
def phase4():
    print("=" * 70)
    print("Phase 4: Q3实现审计 — 曲线 + 指数对照 + 信号层命中 + 桥接裁决")
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

    # 4a) 月度分解 vs 指数 (桥接: 正IC≠正P&L时归因市场beta)
    try:
        idx = pd.read_csv(os.path.join(BASE_DIR, '..',
                                       'data/stock_data/backtrader_data/sh000001_qfq.csv'),
                          parse_dates=['datetime'], usecols=['datetime', 'close']).set_index('datetime')['close']
        print("\n  [4a] 月度分解 vs 上证指数:")
        for a, b in (('2026-07-01', '2026-07-31'), ('2026-08-01', '2026-08-31'),
                     ('2026-09-01', '2026-09-30')):
            seg = eq[(eq.index >= a) & (eq.index <= b)]
            if len(seg) == 0:
                continue
            s0 = eq[eq.index < seg.index.min()].iloc[-1]
            s_ret = (seg.iloc[-1] / s0 - 1) * 100
            iseg = idx[(idx.index >= a) & (idx.index <= b)]
            i0 = idx[idx.index < iseg.index.min()].iloc[-1]
            i_ret = (iseg.iloc[-1] / i0 - 1) * 100
            print(f"    {a[:7]}: 策略 {s_ret:+6.2f}% | 指数 {i_ret:+6.2f}% | 相对 {s_ret - i_ret:+6.2f}pp")
        iq3 = idx[idx.index >= '2026-07-01']
        iq3_ret = (iq3.iloc[-1] / idx[idx.index <= '2026-06-30'].iloc[-1] - 1) * 100
        print(f"    Q3累计: 策略 {q3_ret:+.2f}% | 指数 {iq3_ret:+.2f}% | 相对 {q3_ret - iq3_ret:+.2f}pp")
    except Exception as e:
        print(f"  [4a] 指数对照失败: {e}")

    # 4b) 买入信号因子构成 (桥接: 买入是否由Q3正IC因子驱动)
    try:
        sig = pd.read_csv(os.path.join(RVR, 'backtest_signals.csv'),
                          usecols=['date', 'buy', 'factor_name'], low_memory=False)
        sig['date'] = pd.to_datetime(sig['date'])
        q3b = sig[(sig['date'] >= '2026-07-01') & (sig['date'] <= '2026-09-30') & (sig['buy'] == 1)]
        with open(os.path.join(Q_DIR, '2026Q3.yaml'), 'r', encoding='utf-8') as f:
            ind_cfg = yaml.safe_load(f)['industry_factors']
        facts = {f for v in ind_cfg.values()
                 for k in ('factors', 'bull_factors', 'bear_factors')
                 for f in (v.get(k) or [])}
        print(f"\n  [4b] Q3买入 {len(q3b)} 笔的因子构成 (与Phase 2 Q3 IC对照):")
        rows = [(f, q3b['factor_name'].str.contains(f, regex=False).sum()) for f in facts]
        for f, c in sorted(rows, key=lambda x: -x[1])[:8]:
            print(f"    {f:<26} {c:>6} ({c / max(len(q3b), 1) * 100:4.1f}%)")
        if rows:
            print("    对照Phase 2: 买入密集因子是否Q3 IC为正 (翻负因子密集=警报; 9/20实测7/8密集")
            print("    因子全正, fund_profit_growth 9.6%微负 — 买入选择与IC一致)")
    except Exception as e:
        print(f"  [4b] 因子构成失败: {e}")

    # 4c) 买入命中率 (信号层实现, future_ret覆盖内)
    try:
        vr = pd.read_csv(os.path.join(RVR, 'validation_results.csv'),
                         usecols=['date', 'buy', 'future_ret'], low_memory=False)
        vr['date'] = pd.to_datetime(vr['date'])
        q3v = vr[(vr['date'] >= '2026-07-01') & (vr['date'] <= '2026-09-30') & (vr['buy'] == 1)]
        q2v = vr[(vr['date'] >= '2026-04-01') & (vr['date'] <= '2026-06-30') & (vr['buy'] == 1)]
        if len(q3v) and len(q2v):
            f3, f2 = q3v['future_ret'].dropna(), q2v['future_ret'].dropna()
            print(f"\n  [4c] 买入信号命中率: Q3 {len(f3)}笔 命中{(f3 > 0).mean() * 100:.1f}% "
                  f"mean {f3.mean() * 100:+.2f}% | Q2 {len(f2)}笔 命中{(f2 > 0).mean() * 100:.1f}% "
                  f"mean {f2.mean() * 100:+.2f}%")
    except Exception as e:
        print(f"  [4c] 命中率失败: {e}")

    print(f"\n  C5c态参考: 732,689/193.08%/1.1961/17.92% (commit 8a1d650) — "
          f"9/30全链后与四指标基线比对, 按年度分解(2026年内增量单独看)")
    print("  桥接裁决口径 (9/20实测入档): Q3实现不看绝对收益, 看 ①月度相对指数 "
          "(动量组合在V型反弹月结构性落后=已知画像, 非标定失败) ②买入因子构成与Q3 IC一致 "
          "(翻负因子不密集=信号层无故障) ③命中率vs Q2 (约49%持平=信号质量未退化)。"
          "Phase 2 IC通过 + 4b/4c无故障 ⇒ V2 OOS成立; 绝对收益负但相对为正属市场beta, "
          "不否决标定程序 (2022全熊年 −5.94%相对更强已先例)。")


# ================= Phase 5 =================
def phase5():
    print("=" * 70)
    print("Phase 5: 决策点 — 全链刷新清单 (不自动跑)")
    print("=" * 70)
    print("""
  若Phase 2/3判定通过, 且用户批准, 才执行:
    1. cd strategy && QUANT_ALT_NO_AUTOREFRESH=1 python bt_execution.py
       # 全链~90min, 串行; env开关保证运行中零自动刷新(数据态确定)
       → 产出至9/30的四指标 + backtest_signals.csv + equity_curve.csv
    2. 四指标 vs 当前基线 732,689/193.08%/1.1961/17.92% (C5c态, 8a1d650)
       → 增量=9/16~9/30数据刷新+15交易日延伸; 刷新效应与延伸效应不可混读:
         先跑 pool_flip_report.py 出池翻转清单(0f后按日历批量生效+翻转报告),
         再按年度分解对账(2021-2025应逐位一致, 分歧=数据态漂移; 2026内
         增量=Q3尾部延伸) — 9/15危机教训: 刷新锚点必须机制归因后采信。
    3. Phase 4 重跑 → 出Q3季度收益分解 (OOS持有期结论)
    4. 首份V2验证报告: Phase 2 IC表 + Phase 3重核结论 + Phase 4 Q3实现
       → 写入 strategy/docs/ 或 rolling_validation_results/
  注意: 全链前不要动 quarterly_factors/ 任何文件; 2026Q4标定是独立任务
        (analysis/calib_2026Q4_tail.py 已备, 窗口2021Q4~2026Q3), Q4权重写盘后
        必须再跑一次全链裁决(quarterly_factors/*.yaml在信号指纹内, 信号必重生成)。
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
