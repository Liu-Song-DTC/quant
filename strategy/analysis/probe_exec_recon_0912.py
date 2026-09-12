#!/usr/bin/env python3
"""2026-09-12 实盘vs回测执行对账探针: 19只实盘持仓的入场口径三重核对

背景: 实盘无本地成交回报(xtquant侧), 可对账的是三个本地可测口径:
  A. 入场价来源: current_positions entry_price vs 信号日close(qfq) — 出单脚本
     load_prices_for_date读target_date收盘 → 预期Δ≈0, 验证实盘记账锚=信号日收盘
  B. 回测成本模型差: 向量化回测按D+1开盘×1.001成交, 实盘记账锚=D收盘 —
     隔夜跳空=实盘账本相对回测模型的系统性成本偏移(每个仓位的幻影盈亏)
  C. 信号忠实度: 每只实盘入场在entry_date是否有buy=True信号(含bp2快车道路径)
只读。轻量。执行: cd strategy && /mnt/d/quant/.venv/bin/python
  analysis/probe_exec_recon_0912.py > logs/probe_exec_recon_0912.log 2>&1
"""
import json
import os
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BT = '/mnt/d/quant/data/stock_data/backtrader_data'
POS = '/mnt/d/quant/current_positions.json'
SIG = os.path.join(BASE, 'rolling_validation_results', 'backtest_signals.csv')
SLIP = 0.001


def main():
    pos = json.load(open(POS, encoding='utf-8'))
    rows = []
    for code, p in pos.items():
        rows.append(dict(code=code, entry_date=p['entry_date'],
                         entry_price=float(p['entry_price']),
                         shares=int(p['shares']), amount=float(p['amount'])))
    pos_df = pd.DataFrame(rows)
    print(f'[0] 实盘持仓 {len(pos_df)} 只', flush=True)

    # 信号buy掩码 (D日是否有buy信号)
    sig = pd.read_csv(SIG, usecols=['code', 'date', 'buy'], dtype={'code': str})
    sig['code'] = sig['code'].str.zfill(6)
    sig['date'] = pd.to_datetime(sig['date'])
    buy_set = set(zip(sig[sig.buy == True].code, sig[sig.buy == True].date))
    del sig
    print('[0] buy信号集就绪', flush=True)

    out = []
    for _, r in pos_df.iterrows():
        fn = os.path.join(BT, f"{r.code}_qfq.csv")
        if not os.path.exists(fn):
            out.append({**r.to_dict(), 'close_D': np.nan, 'open_D1': np.nan,
                        'has_buy': (r.code, pd.Timestamp(r.entry_date)) in buy_set})
            continue
        df = pd.read_csv(fn, parse_dates=['datetime'],
                         usecols=['datetime', 'open', 'close'])
        d = pd.Timestamp(r.entry_date)
        cD = df.loc[df.datetime == d, 'close'].values
        oD1 = df.loc[df.datetime > d, 'open'].values
        close_D = float(cD[0]) if len(cD) else np.nan
        open_D1 = float(oD1[0]) if len(oD1) else np.nan
        out.append({**r.to_dict(), 'close_D': close_D, 'open_D1': open_D1,
                    'has_buy': (r.code, d) in buy_set})
    r = pd.DataFrame(out)
    r['entry_vs_close'] = r.entry_price / r.close_D - 1
    r['overnight_gap'] = r.open_D1 / r.close_D - 1
    r['backtest_cost_vs_entry'] = r.open_D1 * (1 + SLIP) / r.entry_price - 1

    print(f'\n[1] 口径A: 入场价 vs 信号日收盘(qfq)')
    print(f'  entry_vs_close: mean={r.entry_vs_close.mean()*100:+.3f}% '
          f'中位={r.entry_vs_close.median()*100:+.3f}% '
          f'|Δ|>0.5%的只数={(r.entry_vs_close.abs()>0.005).sum()}/{len(r)}')
    print(f'\n[2] 口径B: 隔夜跳空与回测成本模型 (D+1开盘×1.001 vs 实盘记账锚D收盘)')
    v = r.overnight_gap.dropna()
    print(f'  overnight_gap: n={len(v)} mean={v.mean()*100:+.3f}% '
          f'中位={v.median()*100:+.3f}% 最小={v.min()*100:+.2f}% 最大={v.max()*100:+.2f}%')
    v2 = r.backtest_cost_vs_entry.dropna()
    print(f'  回测成本vs实盘锚: mean={v2.mean()*100:+.3f}% (正=实盘账本相对回测高估入场)')
    print(f'  18只(有D+1数据)仓位合计={r[r.open_D1.notna()].amount.sum()/10000:.1f}万, '
          f'系统性偏移≈{v2.mean()*r[r.open_D1.notna()].amount.sum()/10000:.2f}万')
    print(f'\n[3] 口径C: entry_date有buy信号的比例')
    print(f'  {r.has_buy.sum()}/{len(r)} = {r.has_buy.mean()*100:.0f}%')
    if not r.has_buy.all():
        print('  无信号入场:')
        for _, x in r[~r.has_buy].iterrows():
            print(f'    {x.code} {x.entry_date} 价={x.entry_price} 仓位={x.amount}')
    print(f'\n[4] 明细')
    for _, x in r.iterrows():
        print(f'  {x.code} {x.entry_date} 入={x.entry_price:8.2f} '
              f'收={x.close_D:8.2f} Δ={x.entry_vs_close*100:+5.2f}% '
              f'D+1开={x.open_D1 if np.isfinite(x.open_D1) else float("nan"):8.2f} '
              f'隔夜={x.overnight_gap*100:+5.2f}% 信号={"Y" if x.has_buy else "N"} '
              f'仓={x.amount/10000:5.2f}万')


if __name__ == '__main__':
    main()
