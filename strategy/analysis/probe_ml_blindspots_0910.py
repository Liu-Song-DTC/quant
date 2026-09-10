#!/usr/bin/env python3
"""2026-09-10 阶段4探针: ML盲区12因子批量IC (白名单声明但factor_df缺失/幽灵)
价格量能族10个(标准定义, qfq可算) + 基本面族2个(公告日PIT携带):
  vol_divergence    = -corr20(close, volume)            量价背离
  rel_vol_ratio     = vol/vol_ma20 - 1                  相对量比
  volume_trend      = vol_ma5/vol_ma20 - 1              量能趋势
  volume_dry_up     = vol_ma5/vol_ma20                  地量(小=缩量)
  vol_price_res     = mom10 * clip(vol_trend,-1,3)/3    量价共振
  lu_prox20         = max20(chg/涨停幅度) clip0-1.2      涨停邻近
  ld_prox20         = max20(-chg/涨停幅度) clip0-1.2     跌停邻近
  overnight_gap_5d  = Σ5(open/prev_close-1)             隔夜跳空
  amplitude_5d      = mean5((high-low)/prev_close)       振幅
  amplitude_exp     = amplitude_5d/amplitude_ma20        振幅扩张
  fund_pe           = close(公告日)/eps年度, tanh压缩      估值
  pg_accel          = Δ(净利润同比增长) 相邻报告期        成长加速度
测度: 月度截面Spearman IC vs fwd10/20市场调整(中位数), 2021-2026 → 前2强做信号日叠加
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

BT = '/mnt/d/quant/data/stock_data/backtrader_data'
FUND = '/mnt/d/quant/data/stock_data/fundamental_data'
SIG = '/mnt/d/quant/strategy/rolling_validation_results/backtest_signals.csv'
CHG_DATE = pd.Timestamp('2020-08-24')


def limit_thr(code, dates):
    if code.startswith(('300', '301', '688')):
        t = np.where(dates < CHG_DATE, 9.8, 19.8)
        return np.where(code.startswith('688'), 19.8, t)
    return 9.8


def price_factors(df, code):
    """df: datetime/open/high/low/close/volume/chg → 每因子日频Series"""
    df = df.set_index('datetime')
    c, v = df['close'], df['volume']
    thr = limit_thr(code, df.index)
    chg = df['chg'].fillna(0)
    out = {}
    out['vol_divergence'] = np.tanh(-c.rolling(20).corr(v).fillna(0))
    vma5, vma20 = v.rolling(5).mean(), v.rolling(20).mean()
    out['rel_vol_ratio'] = v / vma20 - 1
    out['volume_trend'] = vma5 / vma20 - 1
    out['volume_dry_up'] = vma5 / vma20
    mom10 = c.pct_change(10).fillna(0)
    out['vol_price_res'] = mom10 * (out['volume_trend'].clip(-1, 3) / 3)
    out['lu_prox20'] = (chg / thr).clip(0, 1.2).rolling(20).max()
    out['ld_prox20'] = (-chg / thr).clip(0, 1.2).rolling(20).max()
    og = (df['open'] / c.shift(1) - 1).fillna(0)
    out['overnight_gap_5d'] = og.rolling(5).sum()
    amp = ((df['high'] - df['low']) / c.shift(1)).fillna(0)
    out['amplitude_5d'] = amp.rolling(5).mean()
    out['amplitude_exp'] = out['amplitude_5d'] / out['amplitude_5d'].rolling(20).mean()
    return out


def fund_factors():
    """fund_pe + pg_accel 携带序列 (公告日后)"""
    rows = []
    for fn in sorted(os.listdir(FUND)):
        if not fn.endswith('.csv'):
            continue
        try:
            f = pd.read_csv(os.path.join(FUND, fn),
                            usecols=['股票代码', '报告期', '最新公告日期', '每股收益', '净利润-同比增长'])
        except Exception:
            continue
        f['code'] = f['股票代码'].astype(str).str.zfill(6)
        f['报告期'] = pd.to_datetime(f['报告期'].astype(str), format='%Y%m%d', errors='coerce')
        f['公告日'] = pd.to_datetime(f['最新公告日期'], errors='coerce')
        f['eps'] = pd.to_numeric(f['每股收益'], errors='coerce')
        f['g'] = pd.to_numeric(f['净利润-同比增长'], errors='coerce')
        rows.append(f[['code', '报告期', '公告日', 'eps', 'g']])
    d = pd.concat(rows, ignore_index=True).dropna(subset=['code', '公告日'])
    pe_carry, acc_carry = {}, {}
    ann = d[d['报告期'].dt.month == 12].dropna(subset=['eps'])
    ann = ann[ann['eps'] > 0]
    for code, g in ann.groupby('code'):
        g = g.sort_values('公告日')
        pe_carry[code] = (g[['公告日', 'eps']])
    for code, g in d.dropna(subset=['g']).groupby('code'):
        g = g.drop_duplicates('报告期').sort_values('报告期')
        acc = g['g'].diff()
        t = pd.Series(acc.values, index=g['公告日'].values).dropna()
        acc_carry[code] = t
    return pe_carry, acc_carry


def pe_at_announce(pe_carry, prices):
    """公告日收盘/eps → tanh压缩PE, 携带"""
    out = {}
    for code, g in pe_carry.items():
        if code not in prices:
            continue
        df = prices[code]
        t = df['datetime'].values
        vals, idxs = [], []
        for d, eps in zip(g['公告日'], g['eps']):
            i = np.searchsorted(t, np.datetime64(pd.Timestamp(d)))
            if i >= len(t):
                continue
            pe = df['close'].iloc[i] / eps
            if 1 <= pe <= 200:
                vals.append(np.tanh((15 - pe) / 30))
                idxs.append(d)
        if vals:
            out[code] = pd.Series(vals, index=idxs)
    return out


def ic_batch(factors, prices, months):
    """factors: dict[name][code]->Series; 返回IC表"""
    res = {}
    for name, fac in factors.items():
        rows = []
        for month in months:
            xs, y10, y20 = [], [], []
            for code, t in fac.items():
                if code not in prices:
                    continue
                df = prices[code]
                td = df['datetime'].values
                idx = np.searchsorted(td, np.datetime64(month))
                if idx + 21 > len(td):
                    continue
                d_anchor = pd.Timestamp(td[idx])
                if d_anchor in t.index:
                    xv = t.loc[d_anchor]
                    if isinstance(xv, pd.Series):
                        xv = xv.iloc[-1]
                else:
                    past = t[t.index <= d_anchor]  # 携带型序列: 取最后已知值
                    if past.empty:
                        continue
                    xv = past.iloc[-1]
                if pd.isna(xv):
                    continue
                c0 = df['close'].iloc[idx]
                xs.append(xv)
                y10.append(df['close'].iloc[idx + 10] / c0 - 1)
                y20.append(df['close'].iloc[idx + 20] / c0 - 1)
            if len(xs) < 50:
                continue
            y10 = np.array(y10) - np.median(y10)
            y20 = np.array(y20) - np.median(y20)
            rows.append((month, spearmanr(xs, y10)[0], spearmanr(xs, y20)[0], len(xs)))
        df = pd.DataFrame(rows, columns=['month', 'ic10', 'ic20', 'n'])
        res[name] = df
    return res


def main():
    print('加载价格...', flush=True)
    prices = {}
    for fn in sorted(os.listdir(BT)):
        if not fn.endswith('_qfq.csv'):
            continue
        code = fn.split('_')[0]
        if not code.isdigit() or code.startswith(('399', '8', '43', '92')):
            continue
        try:
            df = pd.read_csv(os.path.join(BT, fn),
                             usecols=['datetime', 'open', 'high', 'low', 'close', 'volume', 'change_percent'])
        except Exception:
            continue
        df['datetime'] = pd.to_datetime(df['datetime'])
        df['chg'] = pd.to_numeric(df['change_percent'], errors='coerce')
        prices[code] = df
    print(f'价格: {len(prices)} 只', flush=True)

    months = pd.date_range('2021-01-01', '2026-08-01', freq='MS')
    facs = {}
    names = ['vol_divergence', 'rel_vol_ratio', 'volume_trend', 'volume_dry_up',
             'vol_price_res', 'lu_prox20', 'ld_prox20', 'overnight_gap_5d',
             'amplitude_5d', 'amplitude_exp']
    for n in names:
        facs[n] = {}
    for code, df in prices.items():
        out = price_factors(df, code)
        for n in names:
            s = out[n].dropna()
            facs[n][code] = s

    pe_carry, acc_carry = fund_factors()
    facs['fund_pe'] = pe_at_announce(pe_carry, prices)
    facs['pg_accel'] = acc_carry

    print('\n[IC批量] 月度截面 fwd10/20 市场调整:')
    res = ic_batch(facs, prices, months)
    summary = []
    for n, df in res.items():
        if len(df) < 6:
            print(f'  {n}: 月数不足({len(df)})')
            continue
        ir20 = df.ic20.mean() / df.ic20.std() if df.ic20.std() > 0 else 0
        ir10 = df.ic10.mean() / df.ic10.std() if df.ic10.std() > 0 else 0
        print(f'  {n}: ic10={df.ic10.mean():+.4f}(IR{ir10:+.2f}) ic20={df.ic20.mean():+.4f}'
              f'(IR{ir20:+.2f}) |IC20|>5%={100*(df.ic20.abs()>0.05).mean():.0f}% n均={df.n.mean():.0f}')
        summary.append((n, abs(ir20), df))
    # 前2强逐年
    summary.sort(key=lambda x: -x[1])
    for n, ir, df in summary[:2]:
        yr = df.groupby(df.month.dt.year)['ic20'].mean()
        print(f'  [{n}] ic20逐年: ' + '  '.join(f'{i}:{v:+.3f}' for i, v in yr.items()))


if __name__ == '__main__':
    main()
