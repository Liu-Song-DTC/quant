#!/usr/bin/env python3
"""成交价口径审计 — 回测当日收盘成交 vs 实盘次日成交 (2026-09-21)

回测fill模型: bt_execution.py:1344/1389 price=d.close[0] — 信号/调仓**当日收盘价**
成交。实盘: 收盘后出单 → **次日**成交(开盘或盘中), 至少T+1开盘价。
口径差距=结构性的实盘/回测错配(阶段0b已修涨停不可成交, 此探针量化的是
可成交但价格不同的一般情况)。

方法: 生产559笔realized交易重定价 — entry@open(entry+1) vs avg_cost,
exit@open(exit+1) vs exit_px → 每笔滑点差 → 组合级NAV一阶影响。
零成本探针, 只读K线与交易日志, 不动生产。数据态9/17冻结。
"""
import numpy as np
import pandas as pd

BT = '/mnt/d/quant/data/stock_data/backtrader_data'
TRD = '/mnt/d/quant/strategy/arms_20260920/C26_deviat0/post_trade_realized.csv'


def main():
    trd = pd.read_csv(TRD, dtype={'code': str})
    trd['entry_date'] = pd.to_datetime(trd['entry_date'])
    trd['exit_date'] = pd.to_datetime(trd['exit_date'])

    rows = []
    miss = 0
    for _, t in trd.iterrows():
        fp = f'{BT}/{t.code}_qfq.csv'
        try:
            k = pd.read_csv(fp, usecols=['datetime', 'open', 'close'])
        except Exception:
            miss += 1
            continue
        k['datetime'] = pd.to_datetime(k['datetime'])
        by_date = dict(zip(k['datetime'].dt.date, zip(k['open'], k['close'])))

        ed = t.entry_date.date()
        xd = t.exit_date.date()
        eo = by_date.get(ed)
        xo = by_date.get(xd)
        if eo is None or xo is None:
            miss += 1
            continue
        entry_open_same, entry_close = eo
        exit_open_same, exit_close = xo
        # 次日开盘: 找entry_date之后的首个K线
        nxt = k[k['datetime'].dt.date > ed]
        nxtx = k[k['datetime'].dt.date > xd]
        if nxt.empty or nxtx.empty:
            miss += 1
            continue
        entry_next_open = nxt['open'].iloc[0]
        exit_next_open = nxtx['open'].iloc[0]

        rows.append({
            'code': t.code,
            'entry_slip_next': entry_next_open / entry_close - 1.0,
            'exit_slip_next': exit_next_open / exit_close - 1.0,
            # 实盘卖出方向: 次日开盘卖出, 滑点对净值影响取负
        })
    df = pd.DataFrame(rows)
    print(f'可重定价: {len(df)}/{len(trd)} (缺K线 {miss})')
    print(f'\n入场滑点(次日开盘 vs 当日收盘): mean={df.entry_slip_next.mean()*100:+.3f}%  '
          f'med={df.entry_slip_next.median()*100:+.3f}%  '
          f'P5={df.entry_slip_next.quantile(.05)*100:+.3f}%  '
          f'P95={df.entry_slip_next.quantile(.95)*100:+.3f}%')
    print(f'出场滑点(次日开盘 vs 当日收盘): mean={df.exit_slip_next.mean()*100:+.3f}%  '
          f'med={df.exit_slip_next.median()*100:+.3f}%  '
          f'P5={df.exit_slip_next.quantile(.05)*100:+.3f}%  '
          f'P95={df.exit_slip_next.quantile(.95)*100:+.3f}%')

    # 一阶NAV影响: 每笔买入金额×(入场滑点) + 卖出金额×(出场滑点)
    # 用avg_cost×股数近似金额 — 股数 = 金额/avg_cost, 用名义仓位(回测中位持仓价值)
    # 简化: 假设每笔交易名义本金相同(等权近似) → 平均每笔总滑点
    total_slip = (df.entry_slip_next + df.exit_slip_next)
    print(f'\n每笔总滑点(买+卖): mean={total_slip.mean()*100:+.3f}%  '
          f'med={total_slip.median()*100:+.3f}%')
    # 精确一阶: 用realized ret重构 — 生产ret vs 重定价ret的差
    trd2 = trd.merge(df, on='code', how='inner')
    n = len(trd2)
    # 生产ret = exit_px/avg_cost - 1 (含手续费影响); 重定价ret近似:
    # (exit_close*(1+exit_slip_next)) / (entry_close*(1+entry_slip_next)) - 1
    entry_close_px = trd2['avg_cost']  # avg_cost≈当日收盘价(+冲击)
    exit_close_px = trd2['exit_px']
    repriced_ret = (exit_close_px * (1 + trd2['exit_slip_next'])) / \
                   (entry_close_px * (1 + trd2['entry_slip_next'])) - 1
    delta_ret = repriced_ret - trd2['ret']
    print(f'重定价ret差(次日开盘成交口径): mean={delta_ret.mean()*100:+.3f}%  '
          f'med={delta_ret.median()*100:+.3f}%  P5={delta_ret.quantile(.05)*100:+.3f}%  '
          f'P95={delta_ret.quantile(.95)*100:+.3f}%')
    # 组合一阶: Σ 仓位×(delta_ret) — 仓位未知, 用全组合平均仓位=总持仓价值/股票数近似
    # 更稳口径: 每笔delta_ret的mean × 平均每笔仓位占比。平均持仓10-15只 → 每笔≈8%仓位
    print(f'  组合一阶影响(假设每笔≈7%仓位): {delta_ret.mean()*0.07*100:+.3f}% NAV')
    print(f'  559笔×7%仓位×mean_delta = 全史NAV一阶 {delta_ret.sum()*0.07*100:+.2f}%')


if __name__ == '__main__':
    main()
