#!/usr/bin/env python
"""E-N1前置探针 (2026-09-09): bp2快车道的收益侧/成本侧

问题: rebalance_days=10 → 信号触发到入场最多等9个自然日。bp2(hit1 81.8%/
mean5 +3.02%)的edge是否随时间衰减? 快车道(信号次日建仓)能抢回多少?

识别优势: 入场年龄不由个股决定, 由信号日 vs 调仓网格的相位决定(准随机) →
年龄桶近似可交换, 无E-N15式选择混杂。

收益侧: bp2 × 年龄桶 → hit1/mean5/实际ret; 非bp2对照; 逐年稳定性。
成本侧: 快车道会把入场提前age-1天 → 短期持有(≤5天)占比↑; 调仓日排挤:
  提前入场的bp2在下一个调仓日已是持仓, 若卖出→churn; 统计现在bp2中
  hold_days≤7的占比作为churn上界。
输入: rolling_validation_results/trade_realized.csv (基线834,560)
      rolling_validation_results/backtest_signals.csv (9/8态信号)
"""
import os

import numpy as np
import pandas as pd

BASE = '/mnt/d/quant/strategy/rolling_validation_results'
DATA_DIR = '/mnt/d/quant/data/stock_data/backtrader_data'
TRADES = f'{BASE}/trade_realized.csv'
SIG_COLS = ['date', 'code', 'buy', 'chan_buy_point', 'chan_sell_point', 'signal_level']

_qfq_cache = {}


def load_qfq(code):
    if code in _qfq_cache:
        return _qfq_cache[code]
    p = os.path.join(DATA_DIR, f'{code}_qfq.csv')
    if not os.path.exists(p):
        _qfq_cache[code] = None
        return None
    df = pd.read_csv(p, usecols=['datetime', 'close'])
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)
    _qfq_cache[code] = df
    return df


def px_on(df_price, day):
    idx = df_price['datetime'].searchsorted(pd.Timestamp(day))
    if idx >= len(df_price):
        return np.nan
    return float(df_price['close'].iloc[idx])


def fwd_ret(df_price, entry_date, horizon):
    idx = df_price['datetime'].searchsorted(pd.Timestamp(entry_date))
    if idx + horizon >= len(df_price) or idx < 0:
        return np.nan
    base = df_price['close'].iloc[idx]
    if base <= 0:
        return np.nan
    return float(df_price['close'].iloc[idx + horizon] / base - 1)


def main():
    trades = pd.read_csv(TRADES)
    trades['entry_date'] = pd.to_datetime(trades['entry_date'])
    trades['code'] = trades['code'].astype(str).str.zfill(6)
    print(f"trades: {len(trades)} | {trades['entry_date'].min().date()} → {trades['entry_date'].max().date()}")

    sig = pd.read_csv(f'{BASE}/backtest_signals.csv', usecols=SIG_COLS, low_memory=False)
    sig['code'] = sig['code'].astype(str).str.zfill(6)
    sig['d'] = pd.to_datetime(sig['date'])
    sig = sig[sig['buy']].sort_values('d')
    print(f"signals(buy=True): {len(sig):,}")

    m = trades.merge(sig, on='code', how='left', suffixes=('', '_sig'))
    m = m[(m['d'] <= m['entry_date']) & (m['d'] >= m['entry_date'] - pd.Timedelta(days=20))]
    m = m.sort_values('d').groupby(['entry_date', 'code'], as_index=False).tail(1)
    m['age'] = (m['entry_date'] - m['d']).dt.days
    matched = m['chan_buy_point'].notna().sum()
    print(f"信号匹配(20天窗): {matched}/{len(trades)} ({matched/len(trades)*100:.1f}%)")
    m = m[m['chan_buy_point'].notna()].copy()

    f1, f5, drift = [], [], []
    for _, t in m.iterrows():
        dfp = load_qfq(t['code'])
        if dfp is None:
            f1.append(np.nan); f5.append(np.nan); drift.append(np.nan)
            continue
        f1.append(fwd_ret(dfp, t['entry_date'], 1))
        f5.append(fwd_ret(dfp, t['entry_date'], 5))
        sig_px = px_on(dfp, t['d'])
        drift.append(t['avg_cost'] / sig_px - 1 if sig_px and sig_px > 0 else np.nan)
    m['fwd1'] = f1
    m['fwd5'] = f5
    m['drift'] = drift
    m['year'] = m['entry_date'].dt.year
    m['bp2'] = m['chan_buy_point'] == 2
    m['bp2clean'] = (m['chan_buy_point'] == 2) & (m['chan_sell_point'] == 0)
    m['age_b'] = pd.cut(m['age'], [-1, 2, 5, 9, 20],
                        labels=['0-2天', '3-5天', '6-9天', '10-20天'])

    print("\n=== 年龄分布 (全部/bp2) ===")
    print(m['age'].value_counts().sort_index().to_string())
    print(m[m['bp2']]['age'].value_counts().sort_index().to_string())

    print("\n=== 收益侧: 年龄桶 × 实际ret/hit1/mean5 ===")
    for name, sub in [('全部', m), ('bp2', m[m['bp2']]), ('bp2无卖点', m[m['bp2clean']]),
                      ('非bp2', m[~m['bp2']])]:
        g = sub.groupby('age_b', observed=True).agg(
            n=('fwd5', 'size'),
            ret=('ret', 'mean'), med_ret=('ret', 'median'),
            hit1=('fwd1', lambda x: (x > 0).mean()),
            mean5=('fwd5', 'mean'), med5=('fwd5', 'median'),
            drift=('drift', 'mean'))
        print(f"\n--- {name} (n={len(sub)}) ---")
        print(g.round(4).to_string())

    print("\n=== 收益侧: bp2 逐年 年轻(≤5天) vs 年长(>5天) ===")
    b2 = m[m['bp2']].copy()
    b2['young'] = b2['age'] <= 5
    y = b2.groupby(['year', 'young'], observed=True).agg(
        n=('ret', 'size'), ret=('ret', 'mean'))
    print(y.round(4).to_string())
    yw = b2[b2['young']]['ret'].mean()
    yo = b2[~b2['young']]['ret'].mean()
    print(f"bp2年轻桶全期ret {yw*100:+.2f}% (n={b2['young'].sum()}) vs "
          f"年长桶 {yo*100:+.2f}% (n={(~b2['young']).sum()})")

    print("\n=== 成本侧: 快车道会制造多少短持有churn ===")
    # 提前age-1天入场 → 持有期延长age-1天; 短持有(≤7天)占bp2比例=churn风险上界
    for name, sub in [('bp2', m[m['bp2']]), ('全部', m)]:
        short = (sub['hold_days'] <= 7).mean()
        very_short = (sub['hold_days'] <= 3).mean()
        print(f"{name}: hold≤7d {short*100:.1f}% (n={(sub['hold_days']<=7).sum()}), "
              f"hold≤3d {very_short*100:.1f}%")

    print("\n=== 年龄 vs 持有期: 年轻入场是否本身就短命 ===")
    g2 = m.groupby('age_b', observed=True)['hold_days'].agg(['size', 'mean', 'median'])
    print(g2.round(1).to_string())

    print("\n=== 漂移 vs 年龄: 等调仓日是否在追高 ===")
    g3 = m.groupby('age_b', observed=True)['drift'].agg(['mean', 'median'])
    print(g3.round(4).to_string())

    part2(m)


def part2(m_trades):
    """Part 2: 中周期bp2信号普查 — 非调仓日的bp2信号从不入场, 值得抓吗?"""
    print("\n" + "=" * 60)
    print("Part 2: 中周期bp2信号普查 (快车道新解: 非调仓日bp2从未入场)")
    print("=" * 60)

    # ── 参考交易日历 ──
    cal = pd.read_csv(os.path.join(DATA_DIR, 'sh000001_qfq.csv'),
                      usecols=['datetime'])['datetime']
    cal = pd.to_datetime(cal).sort_values().reset_index(drop=True)
    cal_set = set(cal.values)
    # 网格: 每10个交易日一次, 锚定首笔交易入场日
    anchor = m_trades['entry_date'].min()
    a_idx = cal.searchsorted(anchor)
    grid_idx = set(range(a_idx, len(cal), 10))
    grid_dates = set(cal.iloc[list(grid_idx)].values)
    # 验证: 550笔入场日有多少落在网格±1交易日
    def near_grid(d):
        i = cal.searchsorted(pd.Timestamp(d))
        return any(cal.iloc[max(0, i - 1):i + 2].isin(grid_dates))
    on_grid = m_trades['entry_date'].map(near_grid).mean()
    print(f"网格验证: {on_grid*100:.1f}% 的入场日落在每10交易日网格±1内 "
          f"(锚={anchor.date()})")

    # ── 信号全集 bp2无卖点 ──
    sig = pd.read_csv(f'{BASE}/backtest_signals.csv', usecols=SIG_COLS, low_memory=False)
    sig['code'] = sig['code'].astype(str).str.zfill(6)
    sig['d'] = pd.to_datetime(sig['date'])
    bp2c = sig[sig['buy'] & (sig['chan_buy_point'] == 2) & (sig['chan_sell_point'] == 0)].copy()
    bp2c = bp2c[bp2c['d'].isin(cal_set)]
    print(f"bp2无卖点信号总数: {len(bp2c):,}")
    bp2c['year'] = bp2c['d'].dt.year
    bp2c['in_grid'] = [d in grid_dates for d in bp2c['d'].values]
    print(f"调仓日命中: {bp2c['in_grid'].sum():,} | 中周期(从未入场): {(~bp2c['in_grid']).sum():,}")

    # ── 全部收盘价载入内存 (一次性) ──
    print("载入全市场收盘价...")
    closes = {}  # code -> (dates np.ndarray, closes np.ndarray)
    import glob
    n_files = 0
    for p in glob.glob(os.path.join(DATA_DIR, '*_qfq.csv')):
        code = os.path.basename(p).split('_')[0]
        if code == 'sh000001':
            continue
        df = pd.read_csv(p, usecols=['datetime', 'close'])
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values('datetime')
        closes[code] = (df['datetime'].values, df['close'].values.astype(float))
        n_files += 1
    print(f"载入 {n_files} 只")

    def fwd_from_sig(code, d, horizon):
        if code not in closes:
            return np.nan
        dts, cl = closes[code]
        i = np.searchsorted(dts, np.datetime64(d))
        if i + horizon >= len(cl) or i < 0:
            return np.nan
        base = cl[i]
        return float(cl[i + horizon] / base - 1) if base > 0 else np.nan

    f1s, f5s = [], []
    for code, d in zip(bp2c['code'], bp2c['d']):
        f1s.append(fwd_from_sig(code, d, 1))
        f5s.append(fwd_from_sig(code, d, 5))
    bp2c['fwd1'] = f1s
    bp2c['fwd5'] = f5s
    valid = bp2c['fwd5'].notna()
    print(f"fwd5有效: {valid.sum():,}/{len(bp2c)}")

    print("\n=== 调仓日bp2 vs 中周期bp2: 信号日fwd5/hit1 ===")
    g = bp2c[valid].groupby('in_grid').agg(
        n=('fwd5', 'size'), hit1=('fwd1', lambda x: (x > 0).mean()),
        mean5=('fwd5', 'mean'), med5=('fwd5', 'median'))
    g.index = ['调仓日(现系统已入场)', '中周期(快车道对象)']
    print(g.round(4).to_string())

    print("\n=== 中周期bp2逐年: n / mean5 (奖品规模) ===")
    mc = bp2c[valid & ~bp2c['in_grid']]
    gy = mc.groupby('year').agg(n=('fwd5', 'size'), hit1=('fwd1', lambda x: (x > 0).mean()),
                                mean5=('fwd5', 'mean'))
    print(gy.round(4).to_string())

    print("\n=== 中周期bp2: 前5日已大涨(追高风险)占比 ===")
    # 信号日已在5日高位跑的占比 — 快车道买的是不是已经跑掉的票
    mc2 = mc.dropna(subset=['fwd5']).copy()
    print(f"n={len(mc2)}")
    # 信号日前5日涨幅 (需要前5收盘)
    pre5 = []
    for code, d in zip(mc2['code'], mc2['d']):
        if code not in closes:
            pre5.append(np.nan)
            continue
        dts, cl = closes[code]
        i = np.searchsorted(dts, np.datetime64(d))
        if i - 5 < 0:
            pre5.append(np.nan)
            continue
        pre5.append(float(cl[i] / cl[i - 5] - 1))
    mc2['pre5'] = pre5
    print(f"信号日前5日涨幅: 中位 {mc2['pre5'].median()*100:+.2f}% | "
          f">5%占比 {(mc2['pre5'] > 0.05).mean()*100:.1f}% | "
          f">10%占比 {(mc2['pre5'] > 0.10).mean()*100:.1f}%")

    print("\n=== 中周期bp2: score分档 × mean5 (奖品可选择性) ===")
    sig_all = pd.read_csv(f'{BASE}/backtest_signals.csv',
                          usecols=['date', 'code', 'score'], low_memory=False)
    sig_all['code'] = sig_all['code'].astype(str).str.zfill(6)
    sig_all['d'] = pd.to_datetime(sig_all['date'])
    mc3 = mc2.merge(sig_all, on=['code', 'd'], how='left')
    mc3['score_q'] = pd.qcut(mc3['score'], 4, labels=['Q1低', 'Q2', 'Q3', 'Q4高'])
    gq = mc3.groupby('score_q', observed=True).agg(
        n=('fwd5', 'size'), hit1=('fwd1', lambda x: (x > 0).mean()),
        mean5=('fwd5', 'mean'), med5=('fwd5', 'median'))
    print(gq.round(4).to_string())
    # 调仓日bp2同口径对照
    rg = bp2c[valid & bp2c['in_grid']].merge(sig_all, on=['code', 'd'], how='left')
    rg['score_q'] = pd.qcut(rg['score'], 4, labels=['Q1低', 'Q2', 'Q3', 'Q4高'])
    print("\n调仓日bp2同口径 (对照):")
    gq2 = rg.groupby('score_q', observed=True).agg(
        n=('fwd5', 'size'), hit1=('fwd1', lambda x: (x > 0).mean()),
        mean5=('fwd5', 'mean'), med5=('fwd5', 'median'))
    print(gq2.round(4).to_string())


if __name__ == '__main__':
    main()
