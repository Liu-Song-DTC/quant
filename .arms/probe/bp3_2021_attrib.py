#!/usr/bin/env python3
"""bp3 采纳后 2021 年 −25.3pp 的持仓级归因 (10-21).

问题: bp3 (min=3) 全期 +218,881, 但 2021 年 −25.3pp — 闸紧在 2021 趋势市
砍掉了哪些票? 缺的是哪一源确认 (chan_buy / bi_buy / signal_level / divergence)?

方法:
  1. 选股集合 diff (base1020fix vs bp3) 按年;
  2. 差异票 fwd10 收益 (逐股 bt CSV close→close);
  3. 单遍扫描信号 CSV, 取差异 (code,date) 的 bp_confirm 四分量
     (chan_buy_point>0且≠5 / bi_buy_point>0 / signal_level>=1 /
      chan_divergence_type∈{bottom,bottom_fx,bottom_fx_3x,B2});
  4. 分量缺口分布 + 各切面平均收益.
两臂仅 portfolio 层不同 ⇒ 信号 CSV 同一份, 分量逐位可比.
"""
import numpy as np
import pandas as pd

RVD = '/mnt/d/quant/strategy/rolling_validation_results'
SIG = f'{RVD}/backtest_signals.csv'
BT = '/mnt/d/quant/data/stock_data/backtrader_data/{}_qfq.csv'
FWD = 10
DIV_OK = {'bottom', 'bottom_fx', 'bottom_fx_3x', 'B2'}


def load(tag):
    df = pd.read_csv(f'{RVD}/portfolio_selections.csv.{tag}', dtype={'code': str})
    df['date'] = pd.to_datetime(df['date'].astype(str)).dt.strftime('%Y-%m-%d')
    return df


def fwd_ret(code, date):
    try:
        d = pd.read_csv(BT.format(code), usecols=['datetime', 'close'])
    except Exception:
        return np.nan
    idx = d.index[d['datetime'] == date]
    if len(idx) == 0:
        return np.nan
    j = idx[0]
    if j + FWD >= len(d):
        return np.nan
    return d['close'].iloc[j + FWD] / d['close'].iloc[j] - 1


base = load('base1020fix')
bp3 = load('bp3')
bk = set(zip(base['code'], base['date']))
pk = set(zip(bp3['code'], bp3['date']))
only_b = bk - pk
only_p = pk - bk
print(f'选股总数 base={len(bk)} bp3={len(pk)} 共有={len(bk & pk)}')
for y in [2021, 2022, 2023, 2024, 2025, 2026]:
    print(f'  {y}: base {sum(1 for _, d in bk if d.startswith(str(y)))} | '
          f'bp3 {sum(1 for _, d in pk if d.startswith(str(y)))} | '
          f'仅base丢失 {sum(1 for _, d in only_b if d.startswith(str(y)))} | '
          f'仅bp3新增 {sum(1 for _, d in only_p if d.startswith(str(y)))}')

want = only_b | only_p
print(f'\n差异票 {len(want)} 个, 扫描信号 CSV 取四分量…', flush=True)
rows = []
for ch in pd.read_csv(SIG, chunksize=2_000_000, dtype={'code': str},
                      usecols=['code', 'date', 'chan_buy_point', 'bi_buy_point',
                               'signal_level', 'chan_divergence_type']):
    ch['date'] = ch['date'].astype(str).str[:10]
    m = ch.set_index(['code', 'date']).index.isin(want)
    if m.any():
        rows.append(ch[m])
sig = pd.concat(rows) if rows else pd.DataFrame()
sig['date'] = sig['date'].astype(str).str[:10]
sig = sig.drop_duplicates(['code', 'date']).set_index(['code', 'date'])
print(f'信号命中 {len(sig)}/{len(want)}')


def comps(code, date):
    if (code, date) not in sig.index:
        return None
    r = sig.loc[(code, date)]
    c1 = int(r['chan_buy_point'] > 0 and r['chan_buy_point'] != 5)
    c2 = int(float(r['bi_buy_point'] or 0) > 0)
    c3 = int(float(r['signal_level'] or 0) >= 1)
    c4 = int(r['chan_divergence_type'] in DIV_OK)
    return c1, c2, c3, c4


recs = []
for side, keys in [('lost(仅base)', only_b), ('new(仅bp3)', only_p)]:
    for c, d in sorted(keys):
        cp = comps(c, d)
        r = fwd_ret(c, d)
        rec = {'side': side, 'code': c, 'date': d, 'ret': r}
        if cp is None:
            rec.update(c1=np.nan, c2=np.nan, c3=np.nan, c4=np.nan, total=np.nan, missing='')
        else:
            miss = [n for n, v in zip(['chan_buy', 'bi_buy', 'signal_level', 'div'], cp) if not v]
            rec.update(c1=cp[0], c2=cp[1], c3=cp[2], c4=cp[3],
                       total=sum(cp), missing='+'.join(miss) if miss else '(4/4)')
        recs.append(rec)
df = pd.DataFrame(recs)
df['year'] = df['date'].str[:4]
pd.set_option('display.width', 200)
print('\n=== 2021 差异明细 ===')
print(df[df.year == '2021'].sort_values(['side', 'ret'], ascending=[True, False])
      .to_string(index=False, float_format=lambda x: f'{x:.3f}'))

print('\n=== 各年差异票平均 fwd10 ===')
print(df.groupby(['year', 'side'])['ret'].agg(['count', 'mean', 'median'])
      .to_string(float_format=lambda x: f'{x:.4f}'))

print('\n=== 分量缺口 (仅 2021) ===')
d21 = df[df.year == '2021']
for side, g in d21.groupby('side'):
    print(f'-- {side} n={len(g)} 平均ret {g.ret.mean():+.4f}')
    print(g.groupby(['total', 'missing'])['ret'].agg(['count', 'mean'])
          .to_string(float_format=lambda x: f'{x:.4f}'))

print('\n=== 各年 × 缺口画像 fwd10 (lost=被bp3丢掉; new=bp3新增) ===')
prof = df.groupby(['year', 'side', 'total', 'missing'])['ret'].agg(['count', 'mean'])
print(prof.to_string(float_format=lambda x: f'{x:.4f}'))

print('\n=== 各年共有票数 (路径分叉度) ===')
for y in [2021, 2022, 2023, 2024, 2025, 2026]:
    b = {k for k in bk if k[1].startswith(str(y))}
    p = {k for k in pk if k[1].startswith(str(y))}
    print(f'  {y}: base {len(b)} bp3 {len(p)} 共有 {len(b & p)} '
          f'(共有占比 {len(b & p) / max(1, len(b)):.0%} of base)')

print('\n=== 分量逐源命中率 (2021) ===')
for side, g in d21.groupby('side'):
    print(f'-- {side}: chan_buy {g.c1.mean():.2f} bi_buy {g.c2.mean():.2f} '
          f'signal_level {g.c3.mean():.2f} div {g.c4.mean():.2f}')

print('\n=== 计算 2021 年收益 (equity_curve) ===')
for tag in ['base1020fix', 'bp3']:
    e = pd.read_csv(f'{RVD}/equity_curve.csv.{tag}', parse_dates=['date']).set_index('date')['nav']
    y = e[e.index.year == 2021]
    # 2021 起点=首个nav; 年度收益=年末/年初前一日
    prev = e[e.index < pd.Timestamp('2021-01-01')]
    start = prev.iloc[-1] if len(prev) else e.iloc[0]
    print(f'{tag}: 2021 {y.iloc[-1] / start - 1:+.2%}  (start {start:,.0f} end {y.iloc[-1]:,.0f})')
