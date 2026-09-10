#!/usr/bin/env python3
"""2026-09-10 阶段3探针: YJS预告差因子 (业绩预告 vs 实际财报) 两侧测量
数据全在本地, 无网络。回答两个问题:
Q1 预告公告漂移: notice_date后5/10/20日收益 × 预告类型/预告方向
Q2 惊喜因子截面: surprise=(实际-预告)/|预告| 在财报公告后已知, 作为持续因子
   月度截面Spearman IC vs 前向10/20日收益 (2021-2026), 分桶收益, 逐年IC
口径: 预告取归母净利润indicator; 预测值优先predict_value>0, 否则base_value*(1+change_pct/100)
      实际值来自fundamental_data 净利润-净利润, 公告日=最新公告日期
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

FUND = '/mnt/d/quant/data/stock_data/fundamental_data'
YJYG = '/mnt/d/quant/data/alternative_data/yjyg_records.pkl'
BT = '/mnt/d/quant/data/stock_data/backtrader_data'


def load_yjyg():
    d = pd.read_pickle(YJYG)
    d = d[d['indicator'] == '归属于上市公司股东的净利润'].copy()
    d['notice_date'] = pd.to_datetime(d['notice_date'], errors='coerce')
    d['report_period'] = pd.to_datetime(d['report_period'], errors='coerce')
    d['predict_value'] = pd.to_numeric(d['predict_value'], errors='coerce')
    d['base_value'] = pd.to_numeric(d['base_value'], errors='coerce')
    d['change_pct'] = pd.to_numeric(d['change_pct'], errors='coerce')
    # 预测值: predict_value>0 直接取; 否则 base*(1+pct/100); 两者都无 -> NaN
    pred = d['predict_value'].where(d['predict_value'] > 0)
    derived = d['base_value'] * (1 + d['change_pct'] / 100)
    d['pred'] = pred.fillna(derived)
    d = d.dropna(subset=['pred', 'notice_date', 'report_period'])
    return d


def load_actuals():
    rows = []
    for fn in sorted(os.listdir(FUND)):
        if not fn.endswith('.csv'):
            continue
        try:
            f = pd.read_csv(os.path.join(FUND, fn))
        except Exception:
            continue
        need = ['股票代码', '报告期', '净利润-净利润', '最新公告日期']
        if not all(c in f.columns for c in need):
            continue
        sub = f[need].copy()
        sub['code'] = sub['股票代码'].astype(str).str.zfill(6)
        sub['报告期'] = pd.to_datetime(sub['报告期'].astype(str), format='%Y%m%d', errors='coerce')
        sub['最新公告日期'] = pd.to_datetime(sub['最新公告日期'], errors='coerce')
        sub['actual'] = pd.to_numeric(sub['净利润-净利润'], errors='coerce')
        rows.append(sub[['code', '报告期', '最新公告日期', 'actual']])
    return pd.concat(rows, ignore_index=True).dropna(subset=['actual', '报告期'])


def load_prices(codes):
    """交易日历 + 每只收盘价, 返回 dict[code]->Series(datetime->close) 和 全局交易日"""
    import glob
    cal = None
    prices = {}
    for c in codes:
        p = os.path.join(BT, f'{c}_qfq.csv')
        if not os.path.exists(p):
            continue
        try:
            df = pd.read_csv(p, usecols=['datetime', 'close'])
        except Exception:
            continue
        s = pd.Series(df['close'].values, index=pd.to_datetime(df['datetime']))
        prices[c] = s
        if cal is None:
            cal = s.index
    return prices, cal


def main():
    yj = load_yjyg()
    act = load_actuals()
    print(f'YJYG事件(净利润): {len(yj)} 条, {yj.code.nunique() if "code" in yj else "?"} 只')
    print(f'实际财报行: {len(act)} 行')

    # ---- Q1 预告公告漂移 ----
    print('\n' + '=' * 70)
    print('Q1 预告公告漂移 (notice_date 次日~T+N 收益)')
    print('=' * 70)
    codes_need = set(yj['code'].astype(str).str.zfill(6).unique()) if 'code' in yj.columns else set()
    prices, cal = load_prices(sorted(codes_need))
    print(f'价格覆盖: {len(prices)}/{len(codes_need)} 只, 交易日 {len(cal)}')

    yj = yj.copy()
    yj['code'] = yj['code'].astype(str).str.zfill(6)
    yj = yj[yj['code'].isin(prices.keys())]
    yj = yj[(yj['notice_date'] >= '2021-01-01') & (yj['notice_date'] <= '2026-09-01')]
    print(f'Q1窗口内事件: {len(yj)}')

    pos = yj['change_pct'] > 0
    results = {}
    for label, mask in [('预增/略增/续盈', yj['forecast_type'].isin(['预增', '略增', '续盈', '扭亏'])),
                        ('预减/略减', yj['forecast_type'].isin(['预减', '略减'])),
                        ('首亏/增亏/续亏', yj['forecast_type'].isin(['首亏', '增亏', '续亏', '减亏']))]:
        ev = yj[mask]
        fwds = []
        for N in (5, 10, 20):
            vals = []
            for _, r in ev.iterrows():
                s = prices[r['code']]
                idx = s.index.searchsorted(r['notice_date'])
                if idx + 1 + N > len(s.index):
                    continue
                start = s.index[idx + 1]
                end = s.index[idx + N]
                if (end - start).days > N * 2 + 10:  # 停牌保护
                    continue
                vals.append(s[end] / s[start] - 1)
            fwds.append(np.mean(vals) * 100)
            results.setdefault(label, {})[f'next{N}d'] = (np.mean(vals) * 100, len(vals))
        print(f'  {label}: ' + '  '.join(f'{k}={v[0]:+.2f}% (n={v[1]})' for k, v in results[label].items()))

    # ---- Q2 惊喜因子截面IC ----
    print('\n' + '=' * 70)
    print('Q2 预告差惊喜因子 (surprise 财报公告后已知, 持续携带)')
    print('=' * 70)
    m = yj.merge(act, left_on=['code', 'report_period'], right_on=['code', '报告期'],
                 how='inner', suffixes=('_yj', '_act'))
    m = m.dropna(subset=['actual', 'pred'])
    m['surprise'] = (m['actual'] - m['pred']) / m['pred'].abs()
    m['surprise'] = m['surprise'].clip(-3, 3)
    m['known'] = m[['notice_date', '最新公告日期']].max(axis=1)
    m = m[m['known'] >= '2020-01-01']
    print(f'配对成功: {len(m)} 条 (yjyg预告 × 同报告期实际值)')
    print(f'surprise分布: mean={m.surprise.mean():.3f} median={m.surprise.median():.3f} '
          f'|>0占比={100*(m.surprise>0).mean():.1f}%')

    # 每只股票构建 surprise 时间序列: known日之后携带
    carry = {}
    for code, g in m.groupby('code'):
        g = g.sort_values('known')
        t = pd.Series(g['surprise'].values, index=g['known'])
        carry[code] = t
    # 也做60日半衰期变体: 因子值 = 最近surprise (raw) vs 时间加权
    print('  [IC] 月度截面 Spearman(最新surprise, 前向10/20日收益), 2021-2026:')
    rows_ic = []
    for month in pd.date_range('2021-01-01', '2026-08-01', freq='MS'):
        xs, y10, y20 = [], [], []
        for code, t in carry.items():
            past = t[t.index <= month]
            if past.empty or code not in prices:
                continue
            s = prices[code]
            idx = s.index.searchsorted(month)
            if idx + 21 > len(s.index):
                continue
            xs.append(past.iloc[-1])
            y10.append(s.iloc[idx + 10] / s.iloc[idx] - 1)
            y20.append(s.iloc[idx + 20] / s.iloc[idx] - 1)
        if len(xs) < 30:
            continue
        ic10 = spearmanr(xs, y10)[0]
        ic20 = spearmanr(xs, y20)[0]
        rows_ic.append((month, ic10, ic20, len(xs)))
    icdf = pd.DataFrame(rows_ic, columns=['month', 'ic10', 'ic20', 'n'])
    icdf['year'] = icdf['month'].dt.year
    for N in ('ic10', 'ic20'):
        print(f'  {N}: 全期 mean={icdf[N].mean():+.4f} std={icdf[N].std():.4f} '
              f'IR={icdf[N].mean()/icdf[N].std():+.2f} |IC|>0.05占比={100*(icdf[N].abs()>0.05).mean():.0f}% '
              f'平均截面n={icdf["n"].mean():.0f}')
        yr = icdf.groupby('year')[N].agg(['mean', 'count'])
        print(f'    逐年: ' + '  '.join(f'{i}:{r["mean"]:+.3f}' for i, r in yr.iterrows()))

    # 分桶: 按surprise符号×大小
    print('\n  [分桶] surprise 5桶 × 财报公告后20日收益:')
    m2 = m[(m['known'] >= '2021-01-01') & (m['code'].isin(prices.keys()))].copy()
    m2['fwd20'] = np.nan
    for i, r in m2.iterrows():
        s = prices[r['code']]
        idx = s.index.searchsorted(r['known'])
        if idx + 21 <= len(s.index):
            m2.loc[i, 'fwd20'] = s.iloc[idx + 20] / s.iloc[idx + 1] - 1
    m2 = m2.dropna(subset=['fwd20'])
    m2['bucket'] = pd.qcut(m2['surprise'].rank(method='first'), 5, labels=False)
    for b in range(5):
        sub = m2[m2['bucket'] == b]
        print(f'   桶{b}: surprise[{sub.surprise.min():+.2f},{sub.surprise.max():+.2f}] '
              f'n={len(sub)} fwd20均值={sub.fwd20.mean()*100:+.2f}% 正收益占比={100*(sub.fwd20>0).mean():.0f}%')


if __name__ == '__main__':
    main()
