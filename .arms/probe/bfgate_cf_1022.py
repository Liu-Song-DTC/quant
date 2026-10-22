#!/usr/bin/env python3
"""conceptbf_1022 机制探针: 回补 → G3 档位 → gate 硬门槛 的**逐格反事实** (2026-10-22).

背景 (读码确认, 决定本臂语义):
  · fixed 模式下 `backtest_factors` **不含** concept_heat; ML whitelist 也不含
    ⇒ 2021-22 回补后的真实 heat **不进入评分**, 只经 `gate_concept_heat` 进 gate_q.
  · G3 grade: 生产 heat 恒 0.5 ⇒ `>0.5` 假, `>0.35` 真 ⇒ **0.6**;
    回补后 heat≈0.01 ⇒ **0.4** (98.7% 股票) ⇒ gate_q = clip(mean/0.55) 降 0.2/4/0.55=0.0909.
  · portfolio: NEW 硬门槛 0.65 / HOLD 0.50 ⇒ 反事实 = 哪些买点被推到门槛下.
本探针从**生产信号 CSV**(在 prodguard 里)直接算这个反事实, 不需要等 80min 臂:
  对每个 (code,date):
    gq_old = CSV gate_quality
    g3_new = grade3(heat_bf)
    gq_new = clip(gq_old + (g3_new-0.6)/4/0.55, 0.5, 2.0)
  统计 2021-22 / 2023Q1 的买点跨越 0.65 / 0.50 的数量与占比.
只读; 输出到 stdout.

用法: cd /mnt/d/quant/strategy && ../.venv/bin/python \
        /mnt/d/quant/.arms/probe/bfgate_cf_1022.py [SIGNALS_CSV]
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, '/mnt/d/quant/strategy')
from core.concept_heat import ConceptHeatCalculator  # noqa: E402

CSV = sys.argv[1] if len(sys.argv) > 1 else \
    '/mnt/d/quant/.arms/prodguard/conceptbf_1022/backtest_signals.csv'
BF = '/mnt/d/quant/.arms/concept_hist_bf1022.pkl'
USE = ['code', 'date', 'buy', 'sell', 'gate_quality']


def grade3(h):
    return np.where(h > 0.7, 1.0, np.where(h > 0.5, 0.8, np.where(h > 0.35, 0.6, 0.4)))


def main():
    calc = ConceptHeatCalculator()
    calc.load(hist_path=BF)
    rows = []
    for ch in pd.read_csv(CSV, usecols=USE, chunksize=1_000_000,
                          dtype={'code': str, 'date': str}):
        ch['date'] = ch['date'].str[:10]
        ch = ch[ch['date'] <= '2023-04-28']
        for d, g in ch.groupby('date', sort=True):
            calc.set_daily_data(d)
            codes = g['code'].tolist()
            h = np.array([calc.get_concept_heat(c) for c in codes])
            g3 = grade3(h)
            gq0 = g['gate_quality'].to_numpy(float)
            gq1 = np.clip(gq0 + (g3 - 0.6) / 4.0 / 0.55, 0.5, 2.0)
            buy = g['buy'].to_numpy(bool)
            rows.append(pd.DataFrame(dict(
                date=d, code=codes, buy=buy, gq0=gq0, gq1=gq1, h=h, g3=g3)))
    d = pd.concat(rows, ignore_index=True)
    d['year'] = d['date'].str[:4]
    d['bucket'] = np.where(d['date'] < '2023-01-03', '1_2021-22',
                           np.where(d['date'] <= '2023-04-03', '2_2023Q1', '3_after'))
    print(f'信号格 {len(d):,} (到 2023-04-28) | heat 中位 {d.h.median():.4f} '
          f'| G3=0.4 占 {np.mean(d.g3 == 0.4):.1%}')

    for b, g in d.groupby('bucket'):
        b_all = g
        b_buy = g[g.buy]
        print(f'\n=== {b}: 格 {len(b_all):,} / buy {len(b_buy):,} ===')
        print(f'  gq_old: p10={b_all.gq0.quantile(.1):.3f} p50={b_all.gq0.median():.3f} '
              f'p90={b_all.gq0.quantile(.9):.3f}  Δgq 均值 {(b_all.gq1-b_all.gq0).mean():+.4f}')
        for lab, th in (('NEW 0.65', 0.65), ('HOLD 0.50', 0.50)):
            o_ok = b_all.gq0 >= th
            n_ok = b_all.gq1 >= th
            cross = o_ok & ~n_ok
            print(f'  {lab}: old 过 {o_ok.sum():,} ({o_ok.mean():.1%}) → '
                  f'new 过 {n_ok.sum():,} ({n_ok.mean():.1%}); 新被拦 {cross.sum():,} 格 '
                  f'({cross.mean():.2%} of all)')
            if lab.startswith('NEW'):
                bb = b_buy
                o2 = bb.gq0 >= th
                n2 = bb.gq1 >= th
                cr = o2 & ~n2
                print(f'    buy 行: old 过 {o2.sum():,}/{len(bb):,} → new 过 {n2.sum():,}; '
                      f'买点新被拦 {cr.sum():,} ({cr.mean():.2%} of buy)')
                if cr.sum():
                    print(f'    被拦买点 gq_old p50={bb.gq0[cr].median():.3f} '
                          f'p90={bb.gq0[cr].quantile(.9):.3f}; heat p50={bb.h[cr].median():.3f}')

    # 反事实对 buy 数量的净影响 (上行不可能: Δgq<=0)
    up = d.gq1 > d.gq0 + 1e-12
    print(f'\n反向变化(grade 上调)格数: {int(up.sum()):,} '
          f'(概念 heat>0.5 的少数情形: {np.mean(d.h[up]):.3f} 均值)' if up.any() else
          '\n反向变化(grade 上调)格数: 0')
    print('注: gate_q 只在 portfolio 作二元门槛(score 不含 gate_quality, signal_engine:1613)')
    print('    ⇒ 本臂 2021-22 的全部效应 = 买点被门槛拦掉的个数, 与 heat 的 IC 无关.')


if __name__ == '__main__':
    main()
