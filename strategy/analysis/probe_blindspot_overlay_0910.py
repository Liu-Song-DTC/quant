#!/usr/bin/env python3
"""2026-09-10 阶段4终裁: ML盲区前5强因子的信号日叠加 (组合层可兑现性)
因子: lu_prox20(涨停邻近) amplitude_5d(振幅) fund_pe(估值) vol_price_res(量价共振)
      overnight_gap_5d(隔夜跳空) — 全部按probe_ml_blindspots_0910.py同口径
测度: buy信号日因子四分桶 × fwd5/10/20(市场调整) + 胜率 + 桶0-3差
"""
import os
import numpy as np
import pandas as pd

BT = '/mnt/d/quant/data/stock_data/backtrader_data'
FUND = '/mnt/d/quant/data/stock_data/fundamental_data'
SIG = '/mnt/d/quant/strategy/rolling_validation_results/backtest_signals.csv'
CHG_DATE = pd.Timestamp('2020-08-24')


def load_fund_pe():
    rows = []
    for fn in sorted(os.listdir(FUND)):
        if not fn.endswith('.csv'):
            continue
        try:
            f = pd.read_csv(os.path.join(FUND, fn),
                            usecols=['股票代码', '报告期', '最新公告日期', '每股收益'])
        except Exception:
            continue
        f['code'] = f['股票代码'].astype(str).str.zfill(6)
        f['报告期'] = pd.to_datetime(f['报告期'].astype(str), format='%Y%m%d', errors='coerce')
        f['公告日'] = pd.to_datetime(f['最新公告日期'], errors='coerce')
        f['eps'] = pd.to_numeric(f['每股收益'], errors='coerce')
        rows.append(f[['code', '报告期', '公告日', 'eps']])
    return pd.concat(rows, ignore_index=True).dropna(subset=['code', '公告日', 'eps'])


def main():
    # ---- 价格因子 (每股日频5个) ----
    sig = pd.read_csv(SIG, usecols=['code', 'date', 'buy'], dtype={'code': str})
    sig = sig[sig['buy'] == True].copy()
    sig['code'] = sig['code'].str.zfill(6)
    sig['date'] = pd.to_datetime(sig['date'])
    sig = sig[sig['date'] >= '2021-01-01']
    grp = {c: g['date'].values for c, g in sig.groupby('code')}

    # ---- fund_pe 携带 (公告日收盘/eps, tanh) ----
    fund = load_fund_pe()
    fund = fund[fund['报告期'].dt.month == 12]
    fund = fund[fund['eps'] > 0]
    pe_carry = {}
    for code, g in fund.groupby('code'):
        g = g.sort_values('公告日')
        pe_carry[code] = (g['公告日'].values, g['eps'].values)

    rows = []
    n_miss = 0
    for code, gdates in grp.items():
        p = os.path.join(BT, f'{code}_qfq.csv')
        if not os.path.exists(p):
            continue
        try:
            df = pd.read_csv(p, usecols=['datetime', 'open', 'high', 'low', 'close',
                                         'volume', 'change_percent'])
        except Exception:
            continue
        df['datetime'] = pd.to_datetime(df['datetime'])
        chg = pd.to_numeric(df['change_percent'], errors='coerce').fillna(0)
        if code.startswith(('300', '301', '688')):
            before = df['datetime'] < CHG_DATE
            thr = np.where(before, 9.8, 19.8)
            thr = np.where(code.startswith('688'), 19.8, thr)
        else:
            thr = 9.8
        lu_prox = (chg / thr).clip(0, 1.2).rolling(20).max().values
        amp = ((df['high'] - df['low']) / df['close'].shift(1)).fillna(0)
        amp5 = amp.rolling(5).mean().values
        vma5 = df['volume'].rolling(5).mean()
        vma20 = df['volume'].rolling(20).mean()
        vt = (vma5 / vma20 - 1).clip(-1, 3) / 3
        mom10 = df['close'].pct_change(10).fillna(0)
        vpr = (mom10 * vt).values
        og = (df['open'] / df['close'].shift(1) - 1).fillna(0)
        og5 = og.rolling(5).sum().values
        t = df['datetime'].values
        idxs = np.searchsorted(t, gdates.astype('datetime64[D]'))
        # fund_pe at signal date
        pe_vals = np.full(len(idxs), np.nan)
        if code in pe_carry:
            pdates, peps = pe_carry[code]
            pdates_np = pdates.astype('datetime64[D]')
            for i, d in enumerate(gdates):
                k = np.searchsorted(pdates_np, np.datetime64(d), side='right') - 1
                if k >= 0:
                    c0 = df['close'].iloc[idxs[i]] if idxs[i] < len(t) else np.nan
                    if c0 is not np.nan and not np.isnan(c0):
                        pe = c0 / peps[k]
                        if 1 <= pe <= 200:
                            pe_vals[i] = np.tanh((15 - pe) / 30)
        for i, idx in enumerate(idxs):
            if idx >= len(t) or idx + 21 > len(t):
                continue
            if abs((pd.Timestamp(t[idx]) - gdates[i]).days) > 10:
                continue
            c0 = df['close'].iloc[idx]
            rows.append((code, gdates[i], lu_prox[idx], amp5[idx], vpr[idx], og5[idx],
                         pe_vals[i],
                         df['close'].iloc[idx + 5] / c0 - 1,
                         df['close'].iloc[idx + 10] / c0 - 1,
                         df['close'].iloc[idx + 20] / c0 - 1))
    r = pd.DataFrame(rows, columns=['code', 'date', 'lu_prox20', 'amplitude_5d',
                                    'vol_price_res', 'overnight_gap_5d', 'fund_pe',
                                    'f5', 'f10', 'f20'])
    print(f'配对: {len(r)} 条')
    for c in ('f5', 'f10', 'f20'):
        r[c + 'a'] = r[c] - r.groupby('date')[c].transform('median')

    for fac in ('lu_prox20', 'amplitude_5d', 'fund_pe', 'vol_price_res', 'overnight_gap_5d'):
        sub = r.dropna(subset=[fac]).copy()
        if len(sub) < 10000:
            print(f'\n[{fac}] 有效n不足: {len(sub)}')
            continue
        sub['qb'] = pd.qcut(sub[fac].rank(method='first'), 4, labels=False)
        print(f'\n[{fac}] n={len(sub)}:')
        for q in range(4):
            s = sub[sub['qb'] == q]
            print(f'  桶{q}(x[{s[fac].min():.3f},{s[fac].max():.3f}]): n={len(s)} '
                  f'fwd5a={s.f5a.mean()*100:+.2f}% fwd10a={s.f10a.mean()*100:+.2f}% '
                  f'fwd20a={s.f20a.mean()*100:+.2f}% 胜率={100*(s.f20 > 0).mean():.0f}%')
        print(f'  桶0-桶3 fwd20a差: {(sub[sub.qb==0].f20a.mean()-sub[sub.qb==3].f20a.mean())*100:+.2f}pp')


if __name__ == '__main__':
    main()
