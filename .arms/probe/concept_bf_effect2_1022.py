#!/usr/bin/env python3
"""#366 回补影响探针 2: 尾部幅度 + gate 档位跨越 + 2023+ IC 对照 (2026-10-22).

探针 1 发现: 回补后 2021-22 heat 中位 ~0.01, 98.7% 落 G3=0.4 档; 生产原态
恒 0.5 ⇒ G3=0.6 档. 这不是"加了一点信息", 还是一次**水平移动**:
  gate_q = mean(4 grades)/0.55; 单门降 0.2 ⇒ gate_q 降 ~0.09 ⇒ 硬门槛 0.65 更难过.
本探针量化:
  1. prod vs bf 的 heat 差 |Δ| 分布 (2021-22 / 2023-01..04 / 2023-06+);
  2. G3 档位跨越数 (0.6→0.4 等);
  3. 2023-2025 的 heat IC (prod 态) — 判断 +3.6% 是不是该因子的"正常水平".
只读, 不改生产.
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, '/mnt/d/quant/strategy')
from core.concept_heat import ConceptHeatCalculator  # noqa: E402

BF = '/mnt/d/quant/.arms/concept_hist_bf1022.pkl'
PX = '/mnt/d/quant/data/research/qfq_kline.parquet'


def grade3(h):
    return np.where(h > 0.7, 1.0, np.where(h > 0.5, 0.8, np.where(h > 0.35, 0.6, 0.4)))


def main():
    prod = ConceptHeatCalculator(); prod.load()
    bf = ConceptHeatCalculator(); bf.load(hist_path=BF)
    codes = sorted(prod._stock_concepts)

    print('=== 1. |Δheat| 与 G3 档位跨越 ===')
    print(f'{"date":12s} {"|d|mean":>8s} {"|d|p99":>8s} {"|d|max":>8s} '
          f'{"grade_chg":>9s} {"prod G3":>16s} {"bf G3":>16s}')
    for d in ('2021-06-30', '2021-12-31', '2022-06-30', '2022-12-30',
              '2023-01-03', '2023-01-20', '2023-03-01', '2023-04-03',
              '2023-06-30', '2024-06-28', '2025-06-30'):
        prod.set_daily_data(d)
        hp = np.array([prod.get_concept_heat(c) for c in codes])
        bf.set_daily_data(d)
        hb = np.array([bf.get_concept_heat(c) for c in codes])
        ad = np.abs(hb - hp)
        gp, gb = grade3(hp), grade3(hb)
        chg = np.mean(gp != gb)
        up = {v: f'{np.mean(gb == v):.0%}' for v in (0.4, 0.6, 0.8)}
        upp = {v: f'{np.mean(gp == v):.0%}' for v in (0.4, 0.6, 0.8)}
        print(f'{d:12s} {ad.mean():8.4f} {np.quantile(ad,.99):8.4f} {ad.max():8.4f} '
              f'{chg:9.1%} '
              f'{"4:"+upp[0.4]+"/6:"+upp[0.6]+"/8:"+upp[0.8]:>16s} '
              f'{"4:"+up[0.4]+"/6:"+up[0.6]+"/8:"+up[0.8]:>16s}')

    print('\n=== 2. heat IC 对照 (fwd10, 每5日) ===')
    px = pd.read_parquet(PX, columns=['code', 'date', 'open', 'close'])
    px['date'] = pd.to_datetime(px['date'])
    px = px.sort_values(['code', 'date'])
    g = px.groupby('code', sort=False)
    px['open_next'] = g['open'].shift(-1)
    px['fwd10'] = g['close'].shift(-10) / px['open_next'] - 1
    px = px[['code', 'date', 'fwd10']].dropna()
    pxmap = {d: gd.set_index('code')['fwd10'] for d, gd in px.groupby('date')}
    cal = sorted(pxmap)

    def ic_series(calc, d0, d1, step=5):
        ds = [d for d in cal if pd.Timestamp(d0) <= d <= pd.Timestamp(d1)][::step]
        out = []
        for d in ds:
            ret = pxmap.get(d)
            if ret is None or len(ret) < 500:
                continue
            calc.set_daily_data(d)
            h = pd.Series({c: calc.get_concept_heat(c) for c in codes})
            j = pd.concat([h.rename('h'), ret.rename('r')], axis=1, join='inner').dropna()
            if len(j) < 500 or j['h'].std() < 1e-9:
                continue
            out.append((pd.Timestamp(d), j['h'].rank().corr(j['r'].rank()), len(j)))
        return pd.DataFrame(out, columns=['date', 'ic', 'n'])

    for lab, calc, d0, d1 in (
            ('backfill 2021-22', bf, '2021-01-04', '2022-12-30'),
            ('backfill 2023-25', bf, '2023-01-04', '2025-12-31'),
            ('PROD     2023-25', prod, '2023-01-04', '2025-12-31'),
            ('PROD     2024-25', prod, '2024-01-02', '2025-12-31')):
        s = ic_series(calc, d0, d1)
        if len(s) < 3:
            print(f'{lab}: n<3, skip'); continue
        mu, se = s.ic.mean(), s.ic.std(ddof=1) / np.sqrt(len(s))
        s['y'] = s.date.dt.year
        yy = ' '.join(f'{y}:{x:+.4f}({n})' for y, x, n in
                      s.groupby('y').ic.agg(['mean', 'count']).itertuples())
        print(f'{lab}: n={len(s)} mean={mu:+.4f} t={mu/se:+.2f} pos={np.mean(s.ic>0):.0%} | {yy}')


if __name__ == '__main__':
    main()
