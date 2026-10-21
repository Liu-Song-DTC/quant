#!/usr/bin/env python3
"""买点确认源级归因探针 (2026-10-21 新信号).

问题: bp_confirm_min=3 的四源 (S1 chan_buy>0且≠5 / S2 bi_buy_point>0 /
S3 signal_level>=1 / S4 底背离) 在 #365 修复后, 各源的边际信息是什么?
k=0..4 的短/长前瞻收益与胜率形状? 组合层实际成交按源的分布?

口径:
  - 信号层 = buy=True 的全部行 (1,016,089 行); 组合层 = portfolio_selections.
  - 前瞻收益用 data/research/qfq_kline.parquet (至 2026-10-16) 的 qfq close,
    fwd_h = close[t+h]/close[t]-1, 按股票自身交易日; 同时给截面 demean α.
  - 组合层另用 trade_realized.csv 的真实成交收益 (ret).
  - 仅聚合, 不跑回测; 信号 CSV 只读.
"""
import os
import duckdb
import numpy as np
import pandas as pd

R = '/mnt/d/quant/strategy/rolling_validation_results'
OUT = '/mnt/d/quant/.arms/probe/bp_deep_1021'
PQ = '/mnt/d/quant/data/research/qfq_kline.parquet'
os.makedirs(OUT, exist_ok=True)
os.makedirs(OUT + '/tmp', exist_ok=True)

DIV = {'bottom', 'bottom_fx', 'bottom_fx_3x', 'B2'}
USE = ['code', 'date', 'buy', 'score', 'pre_discount_score', 'adjusted_score',
       'ml_score', 'chan_buy_point', 'chan_sell_point', 'bi_buy_point',
       'signal_level', 'chan_divergence_type']

# ---------- A. 信号层 cell 表 (buy rows only) ----------
cells_path = f'{OUT}/cells.parquet'
if not os.path.exists(cells_path):
    parts = []
    for ch in pd.read_csv(f'{R}/backtest_signals.csv', usecols=USE,
                          chunksize=1_000_000, dtype={'code': str}):
        ch = ch[ch['buy'].astype(bool)]
        if not len(ch):
            continue
        for c in ('chan_buy_point', 'bi_buy_point', 'signal_level'):
            ch[c] = pd.to_numeric(ch[c], errors='coerce').fillna(0)
        div = ch['chan_divergence_type'].fillna('').astype(str).str.strip()
        ch['S1'] = ((ch.chan_buy_point > 0) & (ch.chan_buy_point != 5)).astype('int8')
        ch['S2'] = (ch.bi_buy_point > 0).astype('int8')
        ch['S3'] = (ch.signal_level >= 1).astype('int8')
        ch['S4'] = div.isin(DIV).astype('int8')
        ch['k'] = ch[['S1', 'S2', 'S3', 'S4']].sum(axis=1).astype('int8')
        parts.append(ch[['code', 'date', 'score', 'pre_discount_score',
                         'adjusted_score', 'ml_score', 'chan_buy_point',
                         'chan_sell_point', 'bi_buy_point', 'signal_level',
                         'chan_divergence_type', 'S1', 'S2', 'S3', 'S4', 'k']])
    cells = pd.concat(parts, ignore_index=True)
    cells['date'] = pd.to_datetime(cells['date'])
    cells.to_parquet(cells_path, index=False)
    print(f'cells: {len(cells):,} buy rows -> {cells_path}', flush=True)
else:
    print(f'cells exists: {cells_path}', flush=True)

# ---------- B. 前瞻收益 + 截面 demean ----------
full_path = f'{OUT}/cells_full.parquet'
if not os.path.exists(full_path):
    con = duckdb.connect()
    con.execute("SET memory_limit='6GB'; SET threads=4;")
    con.execute(f"SET temp_directory='{OUT}/tmp';")
    con.execute(f"CREATE TABLE cells AS SELECT * FROM read_parquet('{cells_path}')")
    con.execute(f"""CREATE TABLE px AS
        SELECT code, date, close FROM read_parquet('{PQ}')
        WHERE date >= DATE '2020-06-01'""")
    con.execute("""CREATE TABLE fwd AS
        SELECT code, date,
               lead(close,1)  OVER w / close - 1 AS f1,
               lead(close,5)  OVER w / close - 1 AS f5,
               lead(close,10) OVER w / close - 1 AS f10,
               lead(close,20) OVER w / close - 1 AS f20,
               lead(close,60) OVER w / close - 1 AS f60
        FROM px WINDOW w AS (PARTITION BY code ORDER BY date)""")
    con.execute("""CREATE TABLE mkt AS
        SELECT date, avg(f1) m1, avg(f5) m5, avg(f10) m10,
               avg(f20) m20, avg(f60) m60 FROM fwd GROUP BY date""")
    con.execute("""CREATE TABLE cf AS
        SELECT c.*,
               f.f1, f.f5, f.f10, f.f20, f.f60,
               f.f1-m.m1 AS a1, f.f5-m.m5 AS a5, f.f10-m.m10 AS a10,
               f.f20-m.m20 AS a20, f.f60-m.m60 AS a60
        FROM cells c LEFT JOIN fwd f USING(code,date)
        LEFT JOIN mkt m USING(date)""")
    reg = f'{R}/regime_state.csv'
    con.execute(f"CREATE TABLE reg AS SELECT CAST(date AS DATE) date, regime FROM read_csv_auto('{reg}')")
    con.execute("""CREATE TABLE cf2 AS
        SELECT c.*, r.regime FROM cf c LEFT JOIN reg r USING(date)""")
    con.execute(f"COPY cf2 TO '{full_path}' (FORMAT PARQUET)")
    con.close()
    print(f'cells_full -> {full_path}', flush=True)

df = pd.read_parquet(full_path)
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
print(f'joined cells: {len(df):,}; regime null={df.regime.isna().sum():,}', flush=True)


def agg(g, h):
    x = g[f'f{h}'].dropna()
    a = g[f'a{h}'].dropna()
    n = len(x)
    if n == 0:
        return pd.Series({'n': 0})
    sd = x.std(ddof=1)
    return pd.Series({
        'n': n,
        'mean': x.mean(), 'med': x.median(), 'win': float((x > 0).mean()),
        'ex_mean': a.mean() if len(a) else np.nan,
        't': x.mean() / sd * np.sqrt(n) if sd and n > 1 else np.nan,
    })


def table(keys, hs=(1, 5, 10, 20, 60)):
    rows = []
    for k, g in df.groupby(keys, dropna=False):
        r = {'key': k if not isinstance(k, tuple) else '|'.join(map(str, k))}
        for h in hs:
            s = agg(g, h)
            for c, v in s.items():
                r[f'{c}{h}'] = v
        rows.append(r)
    return pd.DataFrame(rows)


# ---------- C. 聚合 ----------
by_k = table('k')
by_k.to_csv(f'{OUT}/by_k.csv', index=False)
by_k[['key', 'n1', 'mean1', 'win1', 'mean5', 'win5', 'ex_mean5',
      'mean10', 'win10', 'ex_mean10', 't10', 'mean20', 'win20', 'ex_mean20',
      'mean60', 'win60', 'ex_mean60', 't60']].to_csv(f'{OUT}/by_k_readable.csv', index=False)

df['subset'] = df[['S1', 'S2', 'S3', 'S4']].astype(str).agg(''.join, axis=1)
by_subset = table('subset')
by_subset.to_csv(f'{OUT}/by_subset.csv', index=False)

marg = []
for s in ['S1', 'S2', 'S3', 'S4']:
    for val, g in df.groupby(s):
        r = {'source': s, 'present': int(val), **agg(g, 10).to_dict(),
             **{f'f{h}_win': agg(g, h).get('win', np.nan) for h in (1, 5, 20, 60)}}
        marg.append(r)
pd.DataFrame(marg).to_csv(f'{OUT}/by_source_marginal.csv', index=False)

by_chan = table('chan_buy_point')
by_chan.to_csv(f'{OUT}/by_chanbuy.csv', index=False)

by_year_k = table(['year', 'k'])
by_year_k.to_csv(f'{OUT}/by_year_k.csv', index=False)

by_reg_k = table(['regime', 'k'])
by_reg_k.to_csv(f'{OUT}/by_regime_k.csv', index=False)

# ---------- D. 组合层: 实际选股 + 成交 ----------
sel = pd.read_csv(f'{R}/portfolio_selections.csv', dtype={'code': str})
sel['date'] = pd.to_datetime(sel['date'])
selj = sel.merge(df.drop(columns=['score'], errors='ignore'),
                 on=['date', 'code'], how='left', suffixes=('_sel', ''))
selj.to_csv(f'{OUT}/selections_enriched.csv', index=False)
sel_sum = selj.groupby('k', dropna=False).apply(
    lambda g: pd.Series({'n': len(g), 'has_signal': int(g.S1.notna().sum()),
                         'mean_f10': g.f10.mean(), 'win_f10': (g.f10 > 0).mean(),
                         'mean_f20': g.f20.mean(), 'win_f20': (g.f20 > 0).mean()}))
sel_sum.to_csv(f'{OUT}/selections_by_k.csv')

tr = pd.read_csv(f'{R}/trade_realized.csv', dtype={'code': str})
tr['entry_date'] = pd.to_datetime(tr['entry_date'])
tr['exit_date'] = pd.to_datetime(tr['exit_date'])
# merge_asof 要求 on-key 全局有序 (by= 只做分组匹配); 不能只按 [code,date] 排.
tr = tr.sort_values(['entry_date', 'code'])
sel2 = sel.sort_values(['date', 'code'])[['code', 'date', 'weight', 'score']]
trm = pd.merge_asof(tr, sel2.rename(columns={'date': 'sel_date'}),
                    left_on='entry_date', right_on='sel_date', by='code',
                    direction='backward', tolerance=pd.Timedelta('10D'))
trm = trm.merge(df[['date', 'code', 'S1', 'S2', 'S3', 'S4', 'k',
                    'chan_buy_point', 'bi_buy_point', 'signal_level',
                    'chan_divergence_type']],
                left_on=['sel_date', 'code'], right_on=['date', 'code'],
                how='left')
trm.to_csv(f'{OUT}/trades_enriched.csv', index=False)


def tstats(g):
    n = len(g)
    if n == 0:
        return pd.Series({'n': 0})
    wins = g.loc[g.ret > 0, 'ret'].sum()
    losses = -g.loc[g.ret < 0, 'ret'].sum()
    return pd.Series({
        'n': n, 'win': float((g.ret > 0).mean()),
        'mean': g.ret.mean(), 'med': g.ret.median(),
        'pf': wins / losses if losses > 0 else np.inf,
        'hold': g.hold_days.mean(),
        'sum': g.ret.sum(),
    })


for keys, name in [(['k'], 'trades_by_k'), (['S1'], 'trades_by_S1'),
                   (['S2'], 'trades_by_S2'), (['S3'], 'trades_by_S3'),
                   (['S4'], 'trades_by_S4'), (['year'], None)]:
    if name is None:
        continue
    trm.groupby(keys, dropna=False).apply(tstats).to_csv(f'{OUT}/{name}.csv')

# ---------- E. 控制台摘要 ----------
print('\n===== k 分布 (buy rows) =====')
print(df['k'].value_counts().sort_index().to_string())
print('\n===== by_k fwd10 =====')
print(by_k[['key', 'n10', 'mean10', 'win10', 'ex_mean10', 't10']].to_string(index=False))
print('\n===== 源边际 (fwd10) =====')
print(pd.DataFrame(marg)[['source', 'present', 'n', 'mean', 'win', 'ex_mean', 't']].to_string(index=False))
print('\n===== 组合层实际成交 by k =====')
print(trm.groupby('k', dropna=False).apply(tstats).to_string())
print('\n===== 组合层 by chan_buy_point =====')
print(trm.groupby('chan_buy_point', dropna=False).apply(tstats).to_string())
print(f'\n输出目录: {OUT}')
