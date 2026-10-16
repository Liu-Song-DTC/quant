#!/usr/bin/env python
"""选股通道 α 复测 (固化自 /tmp/selalpha2.py, 2026-10-16) — 新数据态(anchor_1015d) vs 旧数据态(anchor_1015c)

问题(用户10/16): "修正历史, 回测降低, 是不是也意味着当天选出的股票可信度不高"
⇒ 直接测: 实际选出的股票, 前向10日收益 比"同日同规模的随机选股地板"高多少?
   这个差值(α)才是"选股有没有用"的度量, 且是**同日内配对**的 ⇒ 免疫数据态.
   若两个数据态下 α 同号同量级 ⇒ 回测数字漂移与选股质量无关.
"""
import numpy as np
import pandas as pd
import duckdb

PQ = '/mnt/d/quant/data/research/qfq_kline.parquet'
RVD = '/mnt/d/quant/strategy/rolling_validation_results'
H = 10
SEED = 20261016
TAGS = ['anchor_1015c', 'anchor_1015d']
NAMES = {'anchor_1015c': '旧态(2,461,946)', 'anchor_1015d': '新态(1,518,087)'}

print('载入行情...', flush=True)
con = duckdb.connect()
px = con.execute(f"SELECT code, date, close FROM '{PQ}'").df()
px['date'] = pd.to_datetime(px['date'])
px = px.sort_values(['code', 'date'])
px['fwd'] = px.groupby('code')['close'].shift(-H) / px['close'] - 1.0
px = px.dropna(subset=['fwd'])
print(f'  {len(px):,} 行, {px["date"].min().date()} ~ {px["date"].max().date()}', flush=True)

# 每日 universe: code -> fwd
by_day = {}
for d, g in px.groupby('date'):
    by_day[d] = (g['code'].values, g['fwd'].values)
# 加速: 用 dict 查 fwd
fwd_map = {}
for d, (codes, fw) in by_day.items():
    fwd_map[d] = dict(zip(codes, fw))

res = {}
for tag in TAGS:
    sel = pd.read_csv(f'{RVD}/portfolio_selections.csv.{tag}', parse_dates=['date'])
    sel['date'] = pd.to_datetime(sel['date'])
    diffs = []
    for d, g in sel.groupby('date'):
        if d not in fwd_map:
            continue
        fm = fwd_map[d]
        codes = [c for c in g['code'].astype(str).str.zfill(6)]
        act = [fm[c] for c in codes if c in fm]
        if not act:
            continue
        uni = np.fromiter(fm.values(), dtype=float)
        diffs.append((d, np.mean(act), uni.mean(), len(act), len(uni)))
    df = pd.DataFrame(diffs, columns=['date', 'act', 'uni', 'n_act', 'n_uni'])
    df['d'] = df['act'] - df['uni']
    res[tag] = df
    a = df['act'].mean()
    u = df['uni'].mean()
    # 配对 bootstrap over days
    rng = np.random.default_rng(SEED)
    dv = df['d'].values
    m = len(dv)
    bs = np.array([dv[rng.integers(0, m, m)].mean() for _ in range(5000)])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    p = float((bs > 0).mean())
    print(f'\n=== {NAMES[tag]} ({tag}) ===')
    print(f'  调仓日 {m}  实际均 {a*100:+.2f}%  地板均 {u*100:+.2f}%')
    print(f'  α = {df["d"].mean()*100:+.2f}pp/10日  CI[{lo*100:+.2f},{hi*100:+.2f}]  P(α>0)={p*100:.1f}%')

# 共同调仓日: 两态 α 的配对比较
d1, d2 = res[TAGS[0]], res[TAGS[1]]
mg = d1.merge(d2, on='date', suffixes=('_c', '_d'))
print(f'\n=== 共同调仓日 {len(mg)} ===')
print(f'  α_旧 {mg["d_c"].mean()*100:+.2f}pp   α_新 {mg["d_d"].mean()*100:+.2f}pp   '
      f'差 {(mg["d_d"]-mg["d_c"]).mean()*100:+.2f}pp')
rng = np.random.default_rng(SEED)
dd = (mg['d_d'] - mg['d_c']).values
m = len(dd)
bs = np.array([dd[rng.integers(0, m, m)].mean() for _ in range(5000)])
lo, hi = np.percentile(bs, [2.5, 97.5])
print(f'  Δα 的 95%CI [{lo*100:+.2f},{hi*100:+.2f}]  P(Δα>0)={float((bs>0).mean())*100:.1f}%')
print(f'  共同日实际收益: 旧 {mg["act_c"].mean()*100:+.2f}%  新 {mg["act_d"].mean()*100:+.2f}%')
print(f'  选股重叠: 完全相同日 {int((mg["act_c"].round(6)==mg["act_d"].round(6)).sum())}/{m}')
