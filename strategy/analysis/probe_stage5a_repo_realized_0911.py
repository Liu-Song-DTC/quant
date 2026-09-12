#!/usr/bin/env python3
"""2026-09-11 阶段5a: 回购flag × 已实现成交首读(基线bak0911人群, n=506)
flag: entry_date前30d内有回购公告(NOTICEDATE, 严格<入场日=盘后公告PIT)。
子桶: 金额<1亿(小回购, 事件研究最强桶) vs >=1亿。
bp2交叉: factor_df的bp2分位与回购flag的关系(边际价值检查)。
"""
import numpy as np
import pandas as pd

PKL_REPO = '/mnt/d/quant/data/alternative_data/repurchase_plans.pkl'
TR = '/mnt/d/quant/strategy/rolling_validation_results/trade_realized.csv'
SIG = '/mnt/d/quant/strategy/rolling_validation_results/backtest_signals.csv'


def main():
    tr = pd.read_csv(TR, dtype={'code': str})
    tr['code'] = tr['code'].str.zfill(6)
    tr['entry_date'] = pd.to_datetime(tr['entry_date'])

    plans = pd.read_pickle(PKL_REPO)
    plans = plans[plans['NOTICEDATE'] >= '2020-12-01'].copy()
    plans['code'] = plans['code'].astype(str).str.zfill(6)
    plans = plans[~plans['code'].str.startswith(('4', '8', '92'))]
    ev = plans[['code', 'NOTICEDATE', 'JESX']].sort_values('NOTICEDATE')
    ev['JESX'] = pd.to_numeric(ev['JESX'], errors='coerce')

    tr = tr.sort_values('entry_date')
    m = pd.merge_asof(tr, ev.rename(columns={'NOTICEDATE': 'ev_date'}), left_on='entry_date',
                      right_on='ev_date', by='code', direction='backward',
                      tolerance=pd.Timedelta('30d'))
    m['days_since'] = (m['entry_date'] - m['ev_date']).dt.days
    m['flag'] = m['ev_date'].notna() & (m['days_since'] > 0)  # 严格<入场日(PIT)
    m['flag_small'] = m['flag'] & (m['JESX'] < 1e8)
    m['flag_big'] = m['flag'] & (m['JESX'] >= 1e8)

    print(f'已实现成交 {len(m)} 笔')
    for name, s in [('全部', m), ('回购flag(30d内, 严格<入场)', m[m.flag]),
                    ('  小回购<1亿', m[m.flag_small]), ('  大回购>=1亿', m[m.flag_big]),
                    ('无flag', m[~m.flag])]:
        print(f'  [{name}] n={len(s):4d} 平均ret={s.ret.mean()*100:+.2f}% '
              f'中位={s.ret.median()*100:+.2f}% 胜率={100*(s.ret > 0).mean():.0f}% '
              f'平均持有={s.hold_days.mean():.1f}d')
    print(f'  差(有flag-无flag): {(m[m.flag].ret.mean()-m[~m.flag].ret.mean())*100:+.2f}pp')
    print(f'  差(小回购-无flag): {(m[m.flag_small].ret.mean()-m[~m.flag].ret.mean())*100:+.2f}pp')

    # bp2交叉: 从signals CSV配对chan_buy_point (bp2类=2, E-K1加成对象)
    # 配对: 信号日→次日开盘买入, 入场日=信号日+1, 用backward 3d容差
    sig = pd.read_csv(SIG, usecols=['code', 'date', 'buy', 'chan_buy_point'],
                      dtype={'code': str})
    sig = sig[sig.buy == True].copy()
    sig['code'] = sig['code'].str.zfill(6)
    sig['date'] = pd.to_datetime(sig['date'])
    sig = sig.sort_values('date')
    mm = pd.merge_asof(m.sort_values('entry_date'), sig[['code', 'date', 'chan_buy_point']],
                       left_on='entry_date', right_on='date', by='code',
                       direction='backward', tolerance=pd.Timedelta('3d'))
    print(f'\nbp2交叉 (信号配对 {mm.chan_buy_point.notna().sum()}/{len(mm)}):')
    mm['is_bp2'] = (mm['chan_buy_point'] == 2)
    for fn, s in [('flag且bp2', mm[mm.flag & mm.is_bp2]),
                  ('flag且非bp2', mm[mm.flag & ~mm.is_bp2]),
                  ('无flag且bp2', mm[~mm.flag & mm.is_bp2]),
                  ('无flag且非bp2', mm[~mm.flag & ~mm.is_bp2])]:
        print(f'  {fn:>12s}: n={len(s):4d} ret={s.ret.mean()*100:+.2f}% '
              f'胜率={100*(s.ret > 0).mean():.0f}%')
    print(f'  全人群bp2占比: {mm.is_bp2.sum()}/{len(mm)} = {mm.is_bp2.mean()*100:.1f}%'
          f' (基线652笔bp2=13≈2%)')


if __name__ == '__main__':
    main()
