"""臂对比报告 (2026-10-25): 基线 vs 臂 的净值/胜率/回撤/集中度差异。

用法: compare_arm_1025.py <基线目录> <臂目录> [臂名]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def load(d):
    d = Path(d)
    curve = pd.read_csv(d / 'equity_curve.csv', parse_dates=['date'])
    nav = curve['nav'].to_numpy(float)
    r = pd.Series(nav).pct_change().dropna()
    peak = np.maximum.accumulate(nav)
    out = {
        'nav': float(nav[-1]),
        'cum': float(nav[-1] / nav[0] - 1),
        'sharpe': float(r.mean() / r.std(ddof=0) * np.sqrt(252)) if r.std(ddof=0) > 0 else 0.0,
        'mdd': float((nav / peak - 1).min()),
        'days': len(nav),
        'start': str(curve['date'].min().date()),
        'end': str(curve['date'].max().date()),
    }
    tr = d / 'trade_realized.csv'
    if tr.exists():
        t = pd.read_csv(tr)
        if len(t):
            t['entry_date'] = pd.to_datetime(t['entry_date'])
            t['ret'] = pd.to_numeric(t['ret'], errors='coerce')
            out['ntr'] = len(t)
            out['twr'] = float((t['ret'] > 0).mean())
            out['tmean'] = float(t['ret'].mean())
            out['top1'] = float(t['ret'].max())
            out['top5pct'] = float(t.nlargest(5, 'ret')['ret'].sum() / t['ret'].sum())
            out['hold_med'] = float(t['hold_days'].median())
            yr = t.groupby(t['entry_date'].dt.year)['ret'].agg(['count', 'mean'])
            out['years'] = {int(y): (int(r_['count']), float(r_['mean'])) for y, r_ in yr.iterrows()}
    sel = d / 'portfolio_selections.csv'
    if sel.exists():
        s = pd.read_csv(sel)
        out['nsel'] = len(s)
        out['nsel_codes'] = int(s['code'].nunique())
    return out


def main():
    base_dir, arm_dir = sys.argv[1], sys.argv[2]
    name = sys.argv[3] if len(sys.argv) > 3 else Path(arm_dir).name
    b, a = load(base_dir), load(arm_dir)
    print(f'=== {name} ===')
    rows = [('NAV', b['nav'], a['nav'], a['nav'] - b['nav']),
            ('累计收益', b['cum'], a['cum'], a['cum'] - b['cum']),
            ('Sharpe', b['sharpe'], a['sharpe'], a['sharpe'] - b['sharpe']),
            ('MDD', b['mdd'], a['mdd'], a['mdd'] - b['mdd']),
            ('成交笔数', b.get('ntr'), a.get('ntr'), (a.get('ntr') or 0) - (b.get('ntr') or 0)),
            ('已平仓胜率', b.get('twr'), a.get('twr'), (a.get('twr') or 0) - (b.get('twr') or 0)),
            ('均收益', b.get('tmean'), a.get('tmean'), (a.get('tmean') or 0) - (b.get('tmean') or 0)),
            ('top1', b.get('top1'), a.get('top1'), (a.get('top1') or 0) - (b.get('top1') or 0)),
            ('top5占Σ%', b.get('top5pct'), a.get('top5pct'), (a.get('top5pct') or 0) - (b.get('top5pct') or 0)),
            ('持有期中位', b.get('hold_med'), a.get('hold_med'), (a.get('hold_med') or 0) - (b.get('hold_med') or 0)),
            ('选股次数', b.get('nsel'), a.get('nsel'), (a.get('nsel') or 0) - (b.get('nsel') or 0))]
    print(f'{"指标":<14}{"基线":>14}{"臂":>14}{"Δ":>14}')
    for k, bv, av, dv in rows:
        f = (lambda x: f'{x:,.4f}') if isinstance(bv, float) else (lambda x: str(x))
        print(f'{k:<14}{f(bv):>14}{f(av):>14}{f(dv):>14}')
    if 'years' in b and 'years' in a:
        print('\n分年 (笔数/均收益):')
        for y in sorted(set(b['years']) | set(a['years'])):
            bc, bm = b['years'].get(y, (0, 0.0))
            ac, am = a['years'].get(y, (0, 0.0))
            print(f'  {y}: 基线 {bc:>3}笔 {bm*100:+7.2f}%   臂 {ac:>3}笔 {am*100:+7.2f}%')
    print(f'\n基线区间 {b["start"]}~{b["end"]} ({b["days"]}日)  臂区间 {a["start"]}~{a["end"]} ({a["days"]}日)')


if __name__ == '__main__':
    sys.exit(main())
