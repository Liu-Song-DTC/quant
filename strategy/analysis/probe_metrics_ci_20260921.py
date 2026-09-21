#!/usr/bin/env python3
"""度量层CI探针 — 四指标块bootstrap置信区间 + 配对ΔNAV可辨性 (2026-09-21)

测量层是诚实审计唯一空白(无CI)。两个互补口径:

A. 边际CI: 单条曲线自身路径依赖 — 同策略同分布重跑一次的离散度。
   块长10日, 2000次moving-block bootstrap (简单日收益, nan→0, 与驱动Sharpe口径一致)。
   读法: 单路径NAV期末值的抽样散布极宽(右尾赢家主导), 说明"单路径4-0"统计权重低,
   教义的多层校验(年分解/CSCV/响应面/flat-top)才是承重结构。

B. 配对ΔNAV CI: 臂vs生产的期末NAV差, 两路径共用同一bootstrap块索引(保块内相关)。
   读法: 这是臂裁决的正确噪声尺度 — 若ΔNAV的95%CI含0, 则该臂与生产在统计上
   不可辨(无论铁律符号如何); 若排除0, 差异真实, 铁律裁决承载了真实取舍。

9/30锚点重对账±3万阈值的统计地位: 配对CI是"重跑噪声"下界; 真实锚点漂移来自
数据态变化(池滑动+历史更正), 其经验分布见锚点史(每次刷新−5万~−13万级,
9/15 ON/OFF臂差−69万)。阈值应参照经验分布而非bootstrap。

单进程轻量, 不与在跑驱动争资源。
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

A20 = '/mnt/d/quant/strategy/arms_20260920'
TRADING_DAYS_PER_YEAR = 252
N_BOOT = 2000
BLOCK = 10
SEED = 42

# 生产逐位曲线 + 已裁决近失臂(教义补刀对象)
BASE = f'{A20}/C26_deviat0/post_equity_curve.csv'
REFS = [
    # (标签, 路径, 裁决备注)
    ('T1_rb07',      'T1_rb07/post_equity_curve.csv',      '3-1否决(MDD), comb牙'),
    ('T3_rb20',      'T3_rb20/post_equity_curve.csv',      '4-0但2026崩塌'),
    ('AG2_phase05',  'AG2_phase05/post_equity_curve.csv',  '意外5d网格, 2026 −14.6pp'),
    ('AB1_indw02',   'AB1_indw02/post_equity_curve.csv',   '3-1否决(+60.7k), 2026零贡献'),
    ('AF2_exhr075',  'AF2_exhr075/post_equity_curve.csv',  '3-1否决(MDD −0.54pp)'),
    ('AA1_rank02',   'AA1_rank02/post_equity_curve.csv',   '孤立齿28 stock-days彩票'),
    ('C28_reward_x15', 'C28_reward_x15/post_equity_curve.csv', '机会成本103,769换0.36pp MDD'),
    ('C29_reward_x2',  'C29_reward_x2/post_equity_curve.csv',  '奖励×2, 4-0被4-0≠充分否决'),
]


def load(path):
    df = pd.read_csv(path)
    nav = np.array(df['nav'].values, dtype=float)
    ret = np.nan_to_num(np.array(df['daily_ret'].values, dtype=float), 0.0)
    return nav, ret


def sharpe(ret):
    return np.mean(ret) / max(np.std(ret), 1e-10) * np.sqrt(TRADING_DAYS_PER_YEAR)


def mdd(nav):
    cummax = np.maximum.accumulate(nav)
    return np.max((cummax - nav) / cummax) * 100


def block_starts(rng, T, n_blocks):
    return rng.integers(0, T - BLOCK + 1, size=n_blocks)


def path_from_blocks(ret, starts):
    sample = np.concatenate([ret[s:s + BLOCK] for s in starts])
    return 250000.0 * np.cumprod(1.0 + sample)


def ci95(x):
    return np.percentile(x, 2.5), np.percentile(x, 97.5)


def main():
    print('度量层CI探针 — 块bootstrap (块长%d, %d次, 驱动口径: 简单日收益)' % (BLOCK, N_BOOT))
    base_nav, base_ret = load(BASE)
    T = len(base_ret)
    n_blocks = int(np.ceil(T / BLOCK))
    rng = np.random.default_rng(SEED)
    starts_all = [block_starts(rng, T, n_blocks) for _ in range(N_BOOT)]
    print(f'生产曲线: {T}日, NAV期末 {base_nav[-1]:,.2f}, '
          f'Sharpe {sharpe(base_ret):.4f}, MDD {mdd(base_nav):.4f}%\n')

    print('== A. 边际CI (单曲线路径依赖, 半宽=重跑自身离散) ==')
    obs = {'NAV期末': base_nav[-1],
           '总收益%': (base_nav[-1] / base_nav[0] - 1) * 100,
           'Sharpe': sharpe(base_ret),
           'MDD%': mdd(base_nav)}
    boots = np.empty((N_BOOT, 4))
    for b, starts in enumerate(starts_all):
        path = path_from_blocks(base_ret, starts)
        boots[b, 0] = path[-1]
        boots[b, 1] = (path[-1] / path[0] - 1) * 100
        boots[b, 2] = sharpe(np.nan_to_num(
            np.concatenate([base_ret[s:s + BLOCK] for s in starts]), 0.0))
        cummax = np.maximum.accumulate(path)
        boots[b, 3] = np.max((cummax - path) / cummax) * 100
    for i, nm in enumerate(['NAV期末', '总收益%', 'Sharpe', 'MDD%']):
        lo, hi = ci95(boots[:, i])
        print(f'  {nm}: 观测 {obs[nm]:,.4f}  95%CI [{lo:,.4f}, {hi:,.4f}]  半宽 {(hi-lo)/2:,.4f}')
    nav_lo, nav_hi = ci95(boots[:, 0])
    half = (nav_hi - nav_lo) / 2
    print(f'  → NAV半宽 {half:,.0f} ({half/base_nav[-1]*100:.1f}%): '
          f'单路径末端财富由右尾赢家主导, 边际口径下±3万=0.05半宽=噪声下限')

    print('\n== B. 配对ΔNAV CI (臂vs生产共用块索引, 臂裁决的正确噪声尺度) ==')
    base_paths = [path_from_blocks(base_ret, s) for s in starts_all]
    for lab, rel, note in REFS:
        p = os.path.join(A20, rel)
        if not os.path.exists(p):
            print(f'  {lab}: 缺文件, 跳过')
            continue
        nav, ret = load(p)
        n = min(T, len(ret))
        base_n = np.array([bp[-1] for bp in base_paths])  # 全窗口末端
        diffs = np.empty(N_BOOT)
        for b, starts in enumerate(starts_all):
            ap = path_from_blocks(ret[:n], starts)
            diffs[b] = ap[-1] - base_paths[b][-1]
        lo, hi = ci95(diffs)
        obs_diff = nav[-1] - base_nav[-1]
        verdict = '统计可辨' if (lo > 0 or hi < 0) else '不可辨(噪声级)'
        print(f'  {lab}: ΔNAV观测 {obs_diff:+,.0f} ({obs_diff/base_nav[-1]*100:+.2f}%)  '
              f'95%CI [{lo:+,.0f}, {hi:+,.0f}] → {verdict}  [{note}]')

    print('\n读法: ①配对CI排除0=差异真实(铁律承载真实取舍), 含0=噪声级(教义多层校验'
          '才是承重结构); ②边际半宽≈自身离散度下界; ③9/30锚点漂移阈值应参照'
          '数据态变化经验分布(±5万~13万级), bootstrap是下界不是标尺。')


if __name__ == '__main__':
    main()
