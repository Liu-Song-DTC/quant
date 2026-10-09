#!/usr/bin/env python
"""回测富报告 — 2026-10-09
对一次回测产物做只读后处理, 产出丰富指标 (不碰仿真, 零身份风险):
  输入: rolling_validation_results/equity_curve.csv (date,nav,daily_ret)
        portfolio_selections.csv (date,code,weight,score)   [可选]
        backtest_signals.csv (信号层统计)                    [可选]
        backtest_diagnostics.json (ML/门控指标)              [可选]
        data/stock_data/raw_data/sh000001/qfq.csv (基准)     [可选]
  输出: 控制台富报告 + rolling_validation_results/backtest_report.json
报告含: 总览/年度分解/月度热力/回撤事件/基准对比/组合换手/信号摘要
用法: .venv/bin/python analysis/backtest_report.py [--no-json]
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

RES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       '..', 'rolling_validation_results')
RES_DIR = os.path.normpath(RES_DIR)
BENCH_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '..', '..', 'data', 'stock_data', 'raw_data',
                          'sh000001', 'qfq.csv')
BENCH_PATH = os.path.normpath(BENCH_PATH)
TRADING_DAYS = 252


def load_nav():
    path = os.path.join(RES_DIR, 'equity_curve.csv')
    if not os.path.exists(path):
        print(f"[report] 缺equity_curve.csv: {path}")
        sys.exit(1)
    c = pd.read_csv(path, parse_dates=['date'])
    c = c[c['nav'] > 0].reset_index(drop=True)
    c['ret'] = c['nav'].pct_change()
    return c


def metrics_of(nav: pd.Series, ret: pd.Series) -> dict:
    m = {}
    m['days'] = len(nav)
    m['years'] = m['days'] / TRADING_DAYS
    m['total_return'] = nav.iloc[-1] / nav.iloc[0] - 1
    m['annual_return'] = (nav.iloc[-1] / nav.iloc[0]) ** (1 / m['years']) - 1
    m['annual_vol'] = ret.std(ddof=1) * np.sqrt(TRADING_DAYS)
    m['sharpe'] = ret.mean() / ret.std(ddof=1) * np.sqrt(TRADING_DAYS) \
        if ret.std(ddof=1) > 0 else np.nan
    downside = ret[ret < 0]
    m['sortino'] = ret.mean() / downside.std(ddof=1) * np.sqrt(TRADING_DAYS) \
        if len(downside) > 1 and downside.std(ddof=1) > 0 else np.nan
    m['mdd'] = (nav / nav.cummax() - 1).min()
    m['calmar'] = m['annual_return'] / abs(m['mdd']) if m['mdd'] < 0 else np.nan
    m['daily_win_rate'] = (ret > 0).mean()
    m['best_day'] = ret.max()
    m['worst_day'] = ret.min()
    from scipy import stats
    m['skew'] = float(stats.skew(ret.dropna()))
    m['kurtosis'] = float(stats.kurtosis(ret.dropna()))
    return m


def drawdown_events(nav: pd.Series, dates: pd.Series) -> list:
    """回撤事件: 峰→谷(深度/时长) → 恢复(日期/时长)。"""
    cummax = nav.cummax()
    dd = nav / cummax - 1
    events = []
    in_dd = False
    peak_val = peak_date = None
    trough_val = trough_date = None
    for i in range(len(nav)):
        if dd.iloc[i] < 0:
            if not in_dd:
                in_dd = True
                peak_idx = int(np.where(nav.iloc[:i + 1].values == cummax.iloc[i])[0][-1]) \
                    if i > 0 else 0
                peak_val = nav.iloc[peak_idx]
                peak_date = dates.iloc[peak_idx]
                trough_val = nav.iloc[i]
                trough_date = dates.iloc[i]
            else:
                if nav.iloc[i] < trough_val:
                    trough_val = nav.iloc[i]
                    trough_date = dates.iloc[i]
        else:
            if in_dd:
                in_dd = False
                events.append(_mk_event(peak_date, peak_val, trough_date,
                                        trough_val, dates.iloc[i], nav.iloc[i]))
    if in_dd:
        events.append(_mk_event(peak_date, peak_val, trough_date, trough_val,
                                None, None))
    events.sort(key=lambda e: e['depth'])   # depth负值, 升序=最深在前
    return events


def _mk_event(pd_, pv, td_, tv, rd, rv):
    depth = tv / pv - 1
    duration = int((td_ - pd_).days)
    rec_days = int((rd - td_).days) if rd is not None else None
    return {'peak': str(pd_.date()), 'trough': str(td_.date()),
            'recover': str(rd.date()) if rd is not None else None,
            'depth': float(depth), 'duration_days': duration,
            'recover_days': rec_days}


def annual_table(c: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year, g in c.groupby(c['date'].dt.year):
        nav = g['nav']
        ret = g['ret']
        m = metrics_of(nav, ret)
        rows.append({'year': year, 'ret': m['total_return'],
                     'mdd': m['mdd'], 'vol': m['annual_vol'],
                     'sharpe': m['sharpe']})
    return pd.DataFrame(rows)


def monthly_table(c: pd.DataFrame) -> pd.DataFrame:
    m = c.assign(ym=c['date'].dt.to_period('M')).groupby('ym')['nav'].last()
    r = m.pct_change().dropna()
    return r


def benchmark(c: pd.DataFrame) -> dict:
    if not os.path.exists(BENCH_PATH):
        return {'available': False}
    b = pd.read_csv(BENCH_PATH, parse_dates=['date'])
    b = b[['date', 'close']].sort_values('date')
    lo, hi = c['date'].min(), c['date'].max()
    b = b[(b['date'] >= lo) & (b['date'] <= hi)].reset_index(drop=True)
    if len(b) < 100:
        return {'available': False, 'rows': len(b)}
    bret = b['close'].pct_change()
    aligned = pd.DataFrame({'date': c['date']}).merge(
        b.assign(iret=bret)[['date', 'iret']], on='date', how='left')
    sret = c['ret']
    bm = metrics_of(b['close'], bret)
    joint = pd.DataFrame({'s': sret.values, 'b': aligned['iret'].values}).dropna()
    cov = np.cov(joint['s'], joint['b'])
    beta = cov[0, 1] / cov[1, 1] if cov[1, 1] > 0 else np.nan
    corr = np.corrcoef(joint['s'], joint['b'])[0, 1]
    excess = joint['s'] - joint['b']
    ir = excess.mean() / excess.std(ddof=1) * np.sqrt(TRADING_DAYS) \
        if excess.std(ddof=1) > 0 else np.nan
    return {'available': True, 'index_total': bm['total_return'],
            'index_annual': bm['annual_return'], 'index_mdd': bm['mdd'],
            'excess_annual': metrics_of(nav=c['nav'], ret=c['ret'])['annual_return'] - bm['annual_return'],
            'beta': float(beta), 'corr': float(corr),
            'info_ratio': float(ir),
            'excess_win_rate': float((excess > 0).mean())}


def portfolio_stats(equity_dates: pd.Series) -> dict:
    path = os.path.join(RES_DIR, 'portfolio_selections.csv')
    if not os.path.exists(path):
        return {'available': False}
    s = pd.read_csv(path, parse_dates=['date'], dtype={'code': str}) \
        .sort_values(['date', 'code'])
    n_dates = s['date'].nunique()
    avg_pos = len(s) / n_dates
    top1 = s.groupby('date')['weight'].max().mean()
    top5 = s.groupby('date')['weight'].nlargest(5).groupby('date').sum().mean()
    # 单边换手: 1 - Σ min(w_new, w_prev) over 共同码; 全新建仓=1
    turnover = []
    dates = sorted(s['date'].unique())
    prev = None
    for d in dates:
        w = dict(zip(s.loc[s['date'] == d, 'code'], s.loc[s['date'] == d, 'weight']))
        if prev is None:
            turnover.append(1.0)
        else:
            shared = set(w) & set(prev)
            turnover.append(1 - sum(min(w[c], prev[c]) for c in shared))
        prev = w
    # 年化换手: 平均单边换手 × (252 / 平均调仓间隔交易日) — 间隔用equity日历折算,
    # 不直接用日期diff (含周末)
    eq_days = pd.DatetimeIndex(equity_dates).values
    pos_idx = eq_days.searchsorted(pd.DatetimeIndex(dates).values)
    gaps = np.diff(pos_idx)
    gap_td = float(np.mean(gaps)) if len(gaps) else 0.0
    ann_turnover = np.mean(turnover) * TRADING_DAYS / max(gap_td, 1) \
        if len(dates) > 1 else float(np.mean(turnover))
    return {'available': True, 'rebalance_dates': int(n_dates),
            'avg_positions': float(avg_pos), 'top1_avg': float(top1),
            'top5_avg': float(top5),
            'avg_gap_days': gap_td,
            'annual_turnover': float(ann_turnover)}


def signal_stats() -> dict:
    path = os.path.join(RES_DIR, 'backtest_signals.csv')
    out = {'available': False}
    if os.path.exists(path):
        cols = pd.read_csv(path, nrows=1).columns
        usecols = [c for c in ('code', 'date', 'buy', 'sell', 'industry',
                               'factor_name', 'ml_score')
                   if c in cols]
        s = pd.read_csv(path, usecols=usecols, parse_dates=['date'],
                        dtype={'code': str})
        out = {'available': True, 'rows': len(s)}
        if 'buy' in s.columns:
            out['buy_count'] = int((s['buy'] == True).sum())
            out['sell_count'] = int((s['sell'] == True).sum())
            out['buy_share'] = float((s['buy'] == True).mean())
        if 'factor_name' in s.columns:
            top_f = s['factor_name'].value_counts().head(5)
            out['top_factors'] = {k: int(v) for k, v in top_f.items()}
        if 'industry' in s.columns and s['industry'].notna().any():
            ind = s.loc[s['industry'].notna() & (s['industry'] != '')]
            out['top_industries'] = {k: int(v) for k, v in
                                     ind['industry'].value_counts().head(5).items()}
        if 'ml_score' in s.columns:
            m = s['ml_score']
            out['ml_active_share'] = float((m.abs() > 0.01).mean())
            out['ml_mean'] = float(m.mean())
    return out


def diagnostics_stats() -> dict:
    path = os.path.join(RES_DIR, 'backtest_diagnostics.json')
    if not os.path.exists(path):
        return {'available': False}
    with open(path) as f:
        d = json.load(f)
    keys = ['ml_predictions_total', 'ml_predictions_active', 'ml_validation_ic',
            'alt_northbound_hits', 'alt_margin_hits', 'alt_dragon_tiger_hits']
    return {'available': True,
            **{k: d.get(k) for k in keys if k in d}}


def fmt_pct(x):
    return f"{x*100:+.2f}%" if x is not None and not np.isnan(x) else "n/a"


def print_report(c, m, ann, mon, events, bench, port, sig, diag):
    w = 70
    print("=" * w)
    print("回测富报告")
    print("=" * w)
    print(f"区间: {c['date'].min().date()} ~ {c['date'].max().date()} "
          f"({m['days']}交易日 / {m['years']:.2f}年)")
    print(f"期末净值: {c['nav'].iloc[-1]:,.2f} (起始 {c['nav'].iloc[0]:,.2f})")
    print(f"总收益: {fmt_pct(m['total_return'])}  年化: {fmt_pct(m['annual_return'])}")
    print(f"年化波动: {fmt_pct(m['annual_vol'])}  Sharpe: {m['sharpe']:.4f}  "
          f"Sortino: {m['sortino']:.4f}")
    print(f"MDD: {fmt_pct(m['mdd'])}  Calmar: {m['calmar']:.3f}")
    print(f"日胜率: {fmt_pct(m['daily_win_rate'])}  最佳日: {fmt_pct(m['best_day'])}  "
          f"最差日: {fmt_pct(m['worst_day'])}")
    print(f"收益偏度: {m['skew']:+.3f}  峰度: {m['kurtosis']:+.3f}")
    print()
    print("-" * w)
    print("年度分解:")
    for _, r in ann.iterrows():
        print(f"  {int(r['year'])}: 收益{fmt_pct(r['ret'])}  MDD{fmt_pct(r['mdd'])}  "
              f"波动{fmt_pct(r['vol'])}  Sharpe {r['sharpe']:.2f}")
    print()
    print("-" * w)
    print("月度收益热力 (%):")
    years = sorted(set(mon.index.year))
    print("  " + "".join(f"{y:>8d}" for y in years))
    for month in range(1, 13):
        row = ""
        for y in years:
            v = mon.get(pd.Period(f"{y}-{month:02d}", freq='M'))
            row += f"{v*100:>+7.1f} " if v is not None and not np.isnan(v) else f"{'·':>8s}"
        print(f"  {month:>2d}月 {row}")
    print()
    print("-" * w)
    print("回撤事件 Top6:")
    for e in events[:6]:
        rec = f"恢复于{e['recover']} (+{e['recover_days']}d)" \
            if e['recover'] else "**未恢复**"
        print(f"  {e['peak']} → {e['trough']}: {fmt_pct(e['depth'])} "
              f"({e['duration_days']}d) → {rec}")
    print()
    if bench.get('available'):
        print("-" * w)
        print("基准对比 (sh000001 同期):")
        print(f"  指数收益: {fmt_pct(bench['index_total'])}  年化: "
              f"{fmt_pct(bench['index_annual'])}  MDD: {fmt_pct(bench['index_mdd'])}")
        print(f"  超额年化: {fmt_pct(bench['excess_annual'])}  β: {bench['beta']:.2f}  "
              f"相关: {bench['corr']:.2f}  IR: {bench['info_ratio']:.3f}")
        print(f"  超额日胜率: {fmt_pct(bench['excess_win_rate'])}")
        print()
    if port.get('available'):
        print("-" * w)
        print("组合层:")
        print(f"  调仓次数: {port['rebalance_dates']}  平均持仓: "
              f"{port['avg_positions']:.1f}只  平均间隔: {port['avg_gap_days']:.0f}交易日")
        print(f"  集中度: Top1均权 {fmt_pct(port['top1_avg'])}  "
              f"Top5均权 {fmt_pct(port['top5_avg'])}")
        print(f"  年化换手: {port['annual_turnover']*100:.0f}%")
        print()
    if sig.get('available'):
        print("-" * w)
        print("信号层:")
        print(f"  信号记录: {sig['rows']:,}  买入: {sig.get('buy_count', 'n/a'):,}  "
              f"卖出: {sig.get('sell_count', 'n/a'):,}  "
              f"买入占比: {fmt_pct(sig.get('buy_share'))}")
        if sig.get('top_factors'):
            print(f"  因子Top3: " + ", ".join(
                f"{k}:{v:,}" for k, v in list(sig['top_factors'].items())[:3]))
        if sig.get('top_industries'):
            print(f"  行业Top3: " + ", ".join(
                f"{k}:{v:,}" for k, v in list(sig['top_industries'].items())[:3]))
        print(f"  ML: 有效预测占比 {fmt_pct(sig.get('ml_active_share'))}  "
              f"均值 {sig.get('ml_mean', 0):.3f}")
        print()
    if diag.get('available'):
        print("-" * w)
        print("诊断指标:")
        for k, v in diag.items():
            if k != 'available':
                print(f"  {k}: {v}")
        print()
    print("=" * w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--no-json', action='store_true', help='不落盘JSON')
    args = ap.parse_args()
    c = load_nav()
    nav = c['nav']
    m = metrics_of(nav, c['ret'])
    ann = annual_table(c)
    mon = monthly_table(c)
    events = drawdown_events(nav, c['date'])
    bench = benchmark(c)
    port = portfolio_stats(c['date'])
    sig = signal_stats()
    diag = diagnostics_stats()
    print_report(c, m, ann, mon, events, bench, port, sig, diag)
    if not args.no_json:
        payload = {
            'generated': str(pd.Timestamp.now()),
            'metrics': {k: (float(v) if isinstance(v, (int, float, np.floating))
                            else v) for k, v in m.items()},
            'annual': ann.to_dict('records'),
            'monthly': {str(k): float(v) for k, v in mon.items()},
            'drawdowns': events[-6:],
            'benchmark': bench,
            'portfolio': port,
            'signal': sig,
            'diagnostics': diag,
        }
        out = os.path.join(RES_DIR, 'backtest_report.json')
        with open(out, 'w') as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
        print(f"[report] 已落盘: {out}")


if __name__ == '__main__':
    main()
