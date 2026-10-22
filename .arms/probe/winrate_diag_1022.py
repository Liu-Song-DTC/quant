#!/usr/bin/env python3
"""选股胜率诊断 (2026-10-22) — 只读, 不改生产.

用户裁定: "调优方向终点放在如何提高选股胜率上".
本探针回答: 胜率在哪个维度上有**单调、跨时段稳定**的区分度.

口径:
  已实现 tW  = trade_realized.ret > 0
  前瞻 wH    = 次日开盘入场 → +H 日收盘 > 0 (H=1/3/5/10/20)
             —— 与回测执行同构(T+1 open), 不扣成本, 衡量"选股"本身
  分数/ML/买点/行业 来自生产信号备份 backtest_signals.csv.pre_w730_1022
  (当前正在跑暖机臂, 生产信号 CSV 被重生成, 故读备份)

用法:
  python .arms/probe/winrate_diag_1022.py a     # D1/D2 (价格层), 轻
  python .arms/probe/winrate_diag_1022.py b     # +信号层 ml/level/industry (重)
  python .arms/probe/winrate_diag_1022.py all
"""
import os
import sys
import numpy as np
import pandas as pd

ROOT = '/mnt/d/quant'
RVD = f'{ROOT}/strategy/rolling_validation_results'
BT = f'{ROOT}/data/stock_data/backtrader_data'
SIG = f'{RVD}/backtest_signals.csv.pre_w730_1022'
SEL = f'{RVD}/portfolio_selections.csv.base_tr1022'
TRD = f'{RVD}/trade_realized.csv.base_tr1022'
REG = f'{RVD}/regime_state.csv.base_tr1022'
EQ = f'{RVD}/equity_curve.csv.base_tr1022'
HORIZONS = [1, 3, 5, 10, 20]


def wilson(k, n, z=1.96):
    if n <= 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - m) / d * 100, (c + m) / d * 100)


def show(df, by, label, sort_by=None):
    print(f'\n--- {label} ---')
    if df.empty:
        print('  (空)')
        return
    rows = []
    for key, g in df.groupby(by, observed=True, dropna=False):
        n = len(g)
        if n < 3:
            continue
        r = {'grp': str(key), 'n': n}
        rr = g['ret'].dropna() if 'ret' in g else pd.Series(dtype=float)
        if len(rr) >= 3:
            k = int((rr > 0).sum())
            lo, hi = wilson(k, len(rr))
            r['nRet'] = len(rr)
            r['tW%'] = round(k / len(rr) * 100, 2)
            r['CI'] = f'{lo:.0f}-{hi:.0f}'
        for h in HORIZONS:
            col = f'w{h}'
            if col in g:
                v = g[col].dropna()
                if len(v):
                    r[f'w{h}%'] = round(float(v.mean() * 100), 1)
                    r[f'm{h}%'] = round(float(g[f'r{h}'].dropna().mean() * 100), 2)
        rows.append(r)
    out = pd.DataFrame(rows)
    if sort_by and sort_by in out:
        out = out.sort_values(sort_by)
    print(out.to_string(index=False))


def load_panel():
    eq = pd.read_csv(EQ, parse_dates=['date'])
    cal = list(eq['date'])
    nxt = {cal[i]: cal[i + 1] for i in range(len(cal) - 1)}
    sel = pd.read_csv(SEL)
    sel['code'] = sel['code'].astype(str).str.zfill(6)
    sel['date'] = pd.to_datetime(sel['date'])
    sel['exec_date'] = sel['date'].map(nxt)
    trd = pd.read_csv(TRD, parse_dates=['entry_date', 'exit_date'])
    trd['code'] = trd['code'].astype(str).str.zfill(6)
    m = sel.merge(trd[['entry_date', 'code', 'ret', 'hold_days']],
                  left_on=['exec_date', 'code'], right_on=['entry_date', 'code'],
                  how='left')
    reg = pd.read_csv(REG, parse_dates=['date'])
    reg = reg[['date', 'regime']].drop_duplicates('date')
    m = m.merge(reg, on='date', how='left')
    m['twin'] = (m['ret'] > 0).astype(int)
    m['year'] = m['date'].dt.year
    m['half'] = np.where(m['year'] <= 2023, 'H1:2021-23', 'H2:2024-26')
    m['srank'] = m.groupby('date')['score'].rank(ascending=False, method='first')
    return m, eq


def add_forward(m):
    px = {}
    need = set(m['code'])
    bad = []
    for code in need:
        f = f'{BT}/{code}_qfq.csv'
        if not os.path.exists(f):
            bad.append(code)
            continue
        try:
            d = pd.read_csv(f, usecols=['datetime', 'open', 'close'])
            d['datetime'] = pd.to_datetime(d['datetime'])
            d = d.sort_values('datetime').reset_index(drop=True)
            px[code] = d
        except Exception:
            bad.append(code)
    if bad:
        print(f'⚠ 缺/坏 qfq: {len(bad)} 只 (例 {bad[:5]})')
    recs = []
    for _, r in m.iterrows():
        d = px.get(r['code'])
        if d is None or pd.isna(r['exec_date']):
            recs.append({})
            continue
        dt = d['datetime'].values
        # 信号日索引 / 执行日索引 (停牌 → 取 >= 执行日首个 bar)
        si = np.searchsorted(dt, np.datetime64(r['date']))
        ei = np.searchsorted(dt, np.datetime64(r['exec_date']))
        if si >= len(d) or ei >= len(d):
            recs.append({})
            continue
        # 必须有信号日或执行日 bar (容忍停牌: 执行取下一可交易 bar)
        entry = float(d['open'].iloc[ei])
        close_sig = float(d['close'].iloc[si])
        out = {'entry_open': entry, 'sig_close': close_sig}
        out['gap'] = entry / close_sig - 1 if close_sig > 0 else np.nan
        if not np.isfinite(out['gap']):
            out['gap'] = np.nan
        if si >= 60:
            out['mom60'] = close_sig / float(d['close'].iloc[si - 60]) - 1
        if si >= 21:
            rr = d['close'].iloc[si - 20:si + 1].pct_change().dropna()
            out['vol20'] = float(rr.std() * np.sqrt(252))
        out['price'] = close_sig
        for h in HORIZONS:
            j = ei + h
            if j < len(d):
                rr = float(d['close'].iloc[j]) / entry - 1 if entry > 0 else np.nan
                out[f'r{h}'] = rr
                out[f'w{h}'] = int(rr > 0) if np.isfinite(rr) else np.nan
        recs.append(out)
    fwd = pd.DataFrame(recs, index=m.index)
    return pd.concat([m, fwd], axis=1)


def part_a(m, with_fwd=True):
    print('=' * 78)
    print(f'D1 已实现交易胜率 (n={len(m)}, 有成交={m["ret"].notna().sum()})')
    print(f'总体: tW={m["twin"].mean()*100:.2f}%  mean_ret={m["ret"].mean()*100:+.2f}%')
    mm = m[m['ret'].notna()].copy()
    mm['score_q'] = pd.qcut(mm['score'], 5, duplicates='drop')
    mm['weight_q'] = pd.qcut(mm['weight'], 5, duplicates='drop')
    mm['srank_b'] = mm['srank'].clip(upper=4).astype(int)
    show(mm, 'score_q', '按入选分数五分位')
    show(mm, 'srank_b', '按当日分数名次 (1=最高)')
    show(mm, 'weight_q', '按权重五分位')
    show(mm, 'year', '按年份')
    show(mm, 'regime', '按入场日 regime')
    mm['hold_b'] = pd.cut(mm['hold_days'], [0, 10, 20, 40, 80, 999],
                          labels=['≤10', '11-20', '21-40', '41-80', '>80'])
    show(mm, 'hold_b', '按持有天数 (内生, 仅描述)')
    if not with_fwd:
        return

    print('\n' + '=' * 78)
    print('D2 逐笔前瞻胜率 (次日开盘入场, 收盘计; 概率>0)')
    for h in HORIZONS:
        v = m[f'w{h}'].dropna()
        n = len(v)
        lo, hi = wilson(int(v.sum()), n)
        print(f'  w{h}: n={n} win={v.mean()*100:.2f}% CI[{lo:.0f},{hi:.0f}] '
              f'mean={m[f"r{h}"].mean()*100:+.2f}%')
    mm = m[m['w5'].notna()].copy()
    mm['score_q'] = pd.qcut(mm['score'], 5, duplicates='drop')
    show(mm, 'score_q', '前瞻胜率: 分数五分位')
    mm['gap_q'] = pd.qcut(mm['gap'], 5, duplicates='drop')
    show(mm, 'gap_q', '前瞻胜率: 次日高开幅度五分位')
    mm['vol_q'] = pd.qcut(mm['vol20'], 5, duplicates='drop')
    show(mm, 'vol_q', '前瞻胜率: 20日波动率五分位')
    mm['mom_q'] = pd.qcut(mm['mom60'], 5, duplicates='drop')
    show(mm, 'mom_q', '前瞻胜率: 60日动量五分位')
    mm['price_q'] = pd.qcut(mm['price'], 5, duplicates='drop')
    show(mm, 'price_q', '前瞻胜率: 股价五分位')

    print('\n' + '=' * 78)
    print('D3 时间稳定性 (2021-23 vs 2024-26)')
    for by in ['score_q', 'gap_q', 'vol_q', 'mom_q']:
        print(f'\n[{by}]')
        for half, g in mm.groupby('half'):
            if by not in g:
                continue
            t = g.groupby(by, observed=True).agg(
                n=('w5', 'size'), w5=('w5', 'mean'),
                w20=('w20', 'mean'), r20=('r20', 'mean'))
            t['w5'] = (t['w5'] * 100).round(1)
            t['w20'] = (t['w20'] * 100).round(1)
            t['r20'] = (t['r20'] * 100).round(2)
            print(f'  {half}:'); print(t.to_string())


def part_b(m):
    print('\n' + '=' * 78)
    print('D0 信号层: ml_score / signal_level / 买点源 / 行业')
    keys = set(m['code'] + '|' + m['date'].dt.strftime('%Y-%m-%d'))
    cols = ['code', 'date', 'score', 'ml_score', 'adjusted_score', 'signal_level',
            'bi_buy_point', 'chan_buy_point', 'industry', 'factor_name',
            'buy', 'gate_quality']
    parts, nrows = [], 0
    for ch in pd.read_csv(SIG, usecols=cols, chunksize=500_000):
        ch['code'] = ch['code'].astype(str).str.zfill(6)
        ch = ch[(ch['code'] + '|' + ch['date'].astype(str)).isin(keys)]
        if len(ch):
            parts.append(ch)
        nrows += 500_000
        if nrows % 5_000_000 == 0:
            print(f'  ... 扫描 {nrows/1e6:.0f}M 行, 命中 {sum(len(p) for p in parts)}')
    if not parts:
        print('⚠ 信号层未命中 (code/date 对齐问题?)'); return
    sig = pd.concat(parts, ignore_index=True).drop_duplicates(['code', 'date'])
    print(f'信号层命中 {len(sig)}/{len(m)}')
    m2 = m.merge(sig, on=['code', 'date'], how='left', suffixes=('', '_sig'))
    mm = m2[m2['w5'].notna()].copy()
    for c in ['ml_score', 'adjusted_score', 'score_sig']:
        if c in mm and mm[c].notna().sum() > 50:
            mm[f'{c}_q'] = pd.qcut(mm[c], 5, duplicates='drop')
            show(mm, f'{c}_q', f'前瞻胜率: {c} 五分位')
    for c in ['signal_level', 'bi_buy_point', 'chan_buy_point']:
        if c in mm:
            show(mm, c, f'前瞻胜率: {c}')
    ind = mm.groupby('industry').agg(n=('w5', 'size'), w5=('w5', 'mean'),
                                     w20=('w20', 'mean'), r20=('r20', 'mean'))
    ind = ind[ind['n'] >= 15].sort_values('w5', ascending=False)
    ind['w5'] = (ind['w5'] * 100).round(1); ind['w20'] = (ind['w20'] * 100).round(1)
    ind['r20'] = (ind['r20'] * 100).round(2)
    print('\n--- 前瞻胜率: 行业 (n≥15) ---'); print(ind.to_string())


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else 'a'
    print(f'# 输入: {SEL} / {TRD} / 信号备份 {SIG}')
    m, eq = load_panel()
    print(f'入选 {len(m)} 行; 有成交 {m["ret"].notna().sum()}; '
          f'日期 {m["date"].min().date()}..{m["date"].max().date()}')
    if mode == 't':
        part_a(m, with_fwd=False)
        return
    m = add_forward(m)
    part_a(m)
    if mode in ('b', 'all'):
        part_b(m)
    print('\n判定提示: 只信 n≥50、跨两段同号、且 w5/w20 同向的维度; '
          '单组差异 <5pp 视为噪声.')


if __name__ == '__main__':
    main()
