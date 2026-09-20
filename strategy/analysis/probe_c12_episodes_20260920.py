#!/usr/bin/env python3
"""
C12硬止损触发事件史探针 (2026-09-20, 只读, 曲线级判读辅助)

机制 (portfolio.py:1042-1058, 已钉死): 每调仓日(rebalance_days=10交易日)检查
drawdown(相对peak_equity) > trigger(0.10) → 敞口cap 0.65 + 记录触发日; 恢复期按
自然日进度从0.65线性回到1.0 (recovery_days=10); 触发期内再破阈值→重置触发日。
本探针从生产equity_curve计算: 各trigger(0.08/0.10/0.12)下的事件次数/时长/深度 —
纯曲线级反事实, 仅用于判读臂结果 (臂运行=真值, E-N13教训: 探针≠机制可实现)。

用法: .venv/bin/python analysis/probe_c12_episodes_20260920.py <equity_curve.csv>
"""
import sys
import pandas as pd

REBAL = 10       # rebalance_days
RECOVERY = 10    # 自然日恢复窗口
CAP = 0.65


def episodes(nav, dates, trigger):
    """调仓日序列上的触发事件: 连续dd>trigger的调仓日run (每个触发日重置恢复窗)"""
    peak = nav.cummax()
    dd = (nav - peak) / peak
    evs = []
    in_ep = False
    for i in range(0, len(nav), REBAL):
        if dd.iloc[i] < -trigger:
            if not in_ep:
                evs.append({'start': dates.iloc[i], 'start_i': i, 'trig_dates': [dates.iloc[i]],
                            'rebal_days': 1, 'trough': dd.iloc[i], 'trough_date': dates.iloc[i]})
                in_ep = True
            else:
                evs[-1]['rebal_days'] += 1
                evs[-1]['trig_dates'].append(dates.iloc[i])
                if dd.iloc[i] < evs[-1]['trough']:
                    evs[-1]['trough'] = dd.iloc[i]
                    evs[-1]['trough_date'] = dates.iloc[i]
        else:
            in_ep = False
    return evs


def main():
    df = pd.read_csv(sys.argv[1], parse_dates=['date']).sort_values('date').reset_index(drop=True)
    nav, dates = df['nav'], df['date']
    peak = nav.cummax()
    dd = (nav - peak) / peak
    print(f"曲线: {dates.iloc[0].date()} ~ {dates.iloc[-1].date()}, {len(df)}行, "
          f"调仓日样本 {len(range(0, len(nav), REBAL))}个")
    print(f"全史最大回撤 {dd.min()*100:.2f}% @ {dates[dd.idxmin()].date()}")
    for trig in (0.08, 0.10, 0.12):
        evs = episodes(nav, dates, trig)
        print(f"\n[trigger {trig:.2f}] 触发事件 {len(evs)}次:")
        for e in evs:
            gaps = [int((b - a).days) for a, b in zip(e['trig_dates'], e['trig_dates'][1:])]
            maxgap = max(gaps) if gaps else 0
            complete = "恢复窗可走完" if maxgap >= RECOVERY else "恢复窗被连续重置"
            print(f"  {e['start'].date()} 连续{e['rebal_days']}调仓日(≈{e['rebal_days']*REBAL}交易日), "
                  f"最深{e['trough']*100:.2f}% @ {e['trough_date'].date()}, 触发日最大间隔{maxgap}自然日 → {complete}")


if __name__ == '__main__':
    main()
