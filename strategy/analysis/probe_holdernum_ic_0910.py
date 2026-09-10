#!/usr/bin/env python3
"""2026-09-10 阶段4探针: 股东户数筹码集中度 (RPT_HOLDERNUM_DET)
因子: hn_chg = HOLDER_NUM_CHANGE/PRE_HOLDER_NUM (季频, 公告日HOLD_NOTICE_DATE携带)
      hn_conc = -hn_chg (集中度, 户数减少为正)
      hn_chg_4q = 4期(一年)累计变动率
测度: 月度截面Spearman IC vs fwd10/20市场调整(中位数), 2021-2026, n≥50, 逐年
      + 信号日叠加(4桶×fwd5/10/20a + 胜率) — 全按既定协议
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PKL = '/mnt/d/quant/data/alternative_data/holdernum_det.pkl'
BT = '/mnt/d/quant/data/stock_data/backtrader_data'
SIG = '/mnt/d/quant/strategy/rolling_validation_results/backtest_signals.csv'


def build_carry():
    """每只股票: 公告日后可用的 hn_chg / hn_conc / hn_chg_4q 携带序列"""
    d = pd.read_pickle(PKL)
    d = d[d.code.str.match(r'^[0368]') & ~d.code.str.startswith(('399', '8', '43', '92'))]
    # 只用季末标准披露行 (排除季中"户数异动"公告, 其RATIO列语义不同)
    d = d[d.end_date.dt.is_quarter_end]
    carry_chg, carry_conc, carry_4q = {}, {}, {}
    for code, g in d.groupby('code'):
        g = g.drop_duplicates('end_date').sort_values('hold_notice_date')
        chg = g['holder_num_change'] / g['pre_holder_num'].where(
            g['pre_holder_num'] > 0)
        chg = chg.clip(-0.5, 0.5)
        idx = g['hold_notice_date'].values
        s = pd.Series(chg.values, index=idx).dropna()
        if len(s) >= 2:
            carry_chg[code] = s
            carry_conc[code] = -s
            carry_4q[code] = s.rolling(4, min_periods=2).sum().dropna()
    return carry_chg, carry_conc, carry_4q


def ic_batch(fac, prices, months, label):
    rows = []
    for month in months:
        xs, y10, y20 = [], [], []
        for code, t in fac.items():
            if code not in prices:
                continue
            past = t[t.index <= month]
            if past.empty:
                continue
            s = prices[code]
            idx = s.index.searchsorted(month)
            if idx + 21 > len(s.index):
                continue
            if pd.isna(past.iloc[-1]):
                continue
            xs.append(past.iloc[-1])
            y10.append(s.iloc[idx + 10] / s.iloc[idx] - 1)
            y20.append(s.iloc[idx + 20] / s.iloc[idx] - 1)
        if len(xs) < 50:
            continue
        y10 = np.array(y10) - np.median(y10)
        y20 = np.array(y20) - np.median(y20)
        rows.append((month, spearmanr(xs, y10)[0], spearmanr(xs, y20)[0], len(xs)))
    df = pd.DataFrame(rows, columns=['month', 'ic10', 'ic20', 'n'])
    if len(df) < 6:
        print(f'  [{label}] 月数不足({len(df)})')
        return df
    ir20 = df.ic20.mean() / df.ic20.std() if df.ic20.std() > 0 else 0
    ir10 = df.ic10.mean() / df.ic10.std() if df.ic10.std() > 0 else 0
    print(f'  [{label}] ic10={df.ic10.mean():+.4f}(IR{ir10:+.2f}) ic20={df.ic20.mean():+.4f}'
          f'(IR{ir20:+.2f}) |IC20|>5%={100*(df.ic20.abs()>0.05).mean():.0f}% n均={df.n.mean():.0f}')
    yr = df.copy(); yr['year'] = yr.month.dt.year
    print(f'    逐年: ' + '  '.join(f'{i}:{v:+.3f}' for i, v in yr.groupby("year").ic20.mean().items()))
    return df


def overlay(fac, prices, label):
    sig = pd.read_csv(SIG, usecols=['code', 'date', 'buy'], dtype={'code': str})
    sig = sig[sig['buy'] == True].copy()
    sig['code'] = sig['code'].str.zfill(6)
    sig['date'] = pd.to_datetime(sig['date'])
    sig = sig[sig['date'] >= '2021-01-01']
    grp = {c: g['date'].values for c, g in sig.groupby('code')}
    rows = []
    for code, t in fac.items():
        if code not in prices or code not in grp:
            continue
        s = prices[code]
        for d in grp[code]:
            past = t[t.index <= d]
            if past.empty:
                continue
            idx = s.index.searchsorted(d)
            if idx + 21 > len(s.index):
                continue
            c0 = s.iloc[idx]
            rows.append((d, past.iloc[-1], s.iloc[idx + 5] / c0 - 1,
                         s.iloc[idx + 10] / c0 - 1, s.iloc[idx + 20] / c0 - 1))
    if len(rows) < 10000:
        print(f'  [{label}] 叠加配对不足: {len(rows)}')
        return
    r = pd.DataFrame(rows, columns=['date', 'x', 'f5', 'f10', 'f20'])
    for c in ('f5', 'f10', 'f20'):
        r[c + 'a'] = r[c] - r.groupby('date')[c].transform('median')
    r['qb'] = pd.qcut(r['x'].rank(method='first'), 4, labels=False)
    print(f'  [{label}] 信号日叠加 (n={len(r)}):')
    for q in range(4):
        sub = r[r['qb'] == q]
        print(f'    桶{q}(x[{sub.x.min():+.3f},{sub.x.max():+.3f}]): n={len(sub)} '
              f'fwd5a={sub.f5a.mean()*100:+.2f}% fwd10a={sub.f10a.mean()*100:+.2f}% '
              f'fwd20a={sub.f20a.mean()*100:+.2f}% 胜率={100*(sub.f20 > 0).mean():.0f}%')
    print(f'    桶0-桶3 fwd20a差: {(r[r.qb==0].f20a.mean()-r[r.qb==3].f20a.mean())*100:+.2f}pp')


def main():
    carry_chg, carry_conc, carry_4q = build_carry()
    print(f'携带序列: chg={len(carry_chg)}只 conc={len(carry_conc)}只 4q={len(carry_4q)}只')
    prices = {}
    for code in set(carry_chg):
        p = os.path.join(BT, f'{code}_qfq.csv')
        if not os.path.exists(p):
            continue
        try:
            df = pd.read_csv(p, usecols=['datetime', 'close'])
        except Exception:
            continue
        prices[code] = pd.Series(df['close'].values, index=pd.to_datetime(df['datetime']))
    print(f'价格覆盖: {len(prices)} 只')
    months = pd.date_range('2021-01-01', '2026-08-01', freq='MS')
    print('\n[IC批量] 月度截面 fwd10/20 市场调整:')
    ic_batch(carry_chg, prices, months, 'hn_chg 户数变动率')
    ic_batch(carry_conc, prices, months, 'hn_conc 集中度(-变动)')
    ic_batch(carry_4q, prices, months, 'hn_chg_4q 一年累计')
    print('\n[信号日叠加]')
    overlay(carry_conc, prices, 'hn_conc 集中度')
    overlay(carry_chg, prices, 'hn_chg 户数变动率')


if __name__ == '__main__':
    main()
