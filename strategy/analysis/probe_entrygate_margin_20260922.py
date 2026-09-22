#!/usr/bin/env python3
"""买入初始门控边界裕量探针 (2026-09-22, 只读, 零生产写入)

signal_engine.py:738-748 初始买入条件 (硬编码, 非yaml):
  buy = 无hard_reject & score≥−0.05 & (结构 OR score>0) & dist_ma20<max_dist
        & (无结构时 price>MA20)
  max_dist按结构: b3+SL≥2=0.40 / b3=0.35 / b1强=0.30 / b1弱=0.25 / b6=0.12 / 其余0.30
设计意图: "极低门槛, 让portfolio判断" — 门控是候选过滤器, 不是选择器。

本探针量化两问 (只读, /tmp输出):
  A) 门控对执行集(570行)的绑定程度 — 执行buy距各边界有多近;
     score<0.05 / score<0 / dist_ma20分布 (dist从BTD qfq CSV逐code复算ma20)。
     若全部裕量宽 → 门控不直接切执行buy (直接效应关闭)。
  B) 门控对候选池的修剪率 — 全buy行(36k)距边界10%内的比例 + 被门控拒掉的
     近似比例。修剪率≈0 → 门控惰性; 修剪率高 → 门控承重(改它=实质变池,
     间接改变rank_pct分位, 需冷跑裁决)。

零写入: 仅 /tmp/probe_entrygate_20260922.csv
"""
import os
import sys

import numpy as np
import pandas as pd

VAL_CSV = '/mnt/d/quant/strategy/rolling_validation_results/validation_results.csv'
SEL_CSV = '/mnt/d/quant/strategy/rolling_validation_results/portfolio_selections.csv'
BTD = '/mnt/d/quant/data/stock_data/backtrader_data'

FILE_CACHE = {}


def close_series(code):
    if code not in FILE_CACHE:
        p = os.path.join(BTD, f'{code}_qfq.csv')
        if not os.path.exists(p):
            FILE_CACHE[code] = None
        else:
            df = pd.read_csv(p, usecols=['datetime', 'close'], parse_dates=['datetime'])
            df = df.sort_values('datetime').drop_duplicates('datetime', keep='last')
            FILE_CACHE[code] = df.set_index('datetime')['close']
    return FILE_CACHE[code]


def dist_at(code, dt):
    s = close_series(code)
    if s is None:
        return np.nan
    if not isinstance(dt, pd.Timestamp):
        dt = pd.Timestamp(dt)  # int64 ns epoch → Timestamp (dt64列)
    pos = s.index.searchsorted(dt)
    if pos >= len(s) or s.index[pos] != dt:
        return np.nan
    # 生产口径: 信号bar当日close, ma20含当日 (指标在bar上计算)
    c = s.iloc[pos]
    win = s.iloc[max(0, pos - 19):pos + 1]
    if len(win) < 5:
        return np.nan
    ma20 = win.mean()
    return float(c / ma20 - 1.0) if ma20 > 0 else np.nan


def main():
    v = pd.read_csv(VAL_CSV, usecols=['date', 'code', 'score', 'buy', 'future_ret'],
                    low_memory=False, dtype={'code': str})
    v['date'] = pd.to_datetime(v['date'])
    v = v.sort_values(['code', 'date']).reset_index(drop=True)
    v['dt64'] = v['date'].values.astype('datetime64[ns]').astype('int64')
    print(f'panel: {len(v)}行, buy行 {int(v["buy"].sum())}')

    codes = v['code'].values
    dts = v['dt64'].values
    scores = v['score'].values.astype(np.float64)
    buys = v['buy'].values.astype(bool)
    buy_rows = np.flatnonzero(buys)
    buy_key = {(codes[r], dts[r]): r for r in buy_rows}
    buy_by_code = {}
    for r in buy_rows:
        buy_by_code.setdefault(codes[r], []).append(r)

    sel = pd.read_csv(SEL_CSV, dtype={'code': str})
    sel['date'] = pd.to_datetime(sel['date'])
    sel['dt64'] = sel['date'].values.astype('datetime64[ns]').astype('int64')

    recs = []
    for _, s in sel.iterrows():
        r = buy_key.get((s['code'], s['dt64']))
        if r is None:
            # 陈信号: 前15日内最近buy
            rows = buy_by_code.get(s['code'])
            if rows:
                sub = [rr for rr in rows
                       if s['dt64'] - 15 * 86400_000_000_000 <= dts[rr] <= s['dt64']]
                if sub:
                    r = sub[-1]
        if r is not None:
            recs.append((s['code'], s['dt64'], scores[r]))
    ex = pd.DataFrame(recs, columns=['code', 'dt64', 'score'])
    print(f'执行buy匹配: {len(ex)}行')

    # ---- A) 执行buy裕量 ----
    ex['dist'] = ex.apply(lambda r: dist_at(r['code'], r['dt64']), axis=1)
    d_ok = ex[ex['dist'].notna()]
    print(f'\n=== A) 执行buy裕量 ({len(d_ok)}行可算dist) ===')
    print('score:')
    print(ex['score'].describe().round(3).to_string())
    print(f'score<0.05: {int((ex["score"] < 0.05).sum())}行 '
          f'(含score<0: {int((ex["score"] < 0).sum())}行 — 必靠结构过门)')
    print(f'score<−0.05(初始门槛下不可能): {int((ex["score"] < -0.05).sum())} (自检应=0)')
    print('\ndist_ma20 (close/ma20−1):')
    print(d_ok['dist'].describe().round(3).to_string())
    print(f'dist<0(低于MA20, 必靠结构豁免): {int((d_ok["dist"] < 0).sum())}行')
    print(f'dist>0.30(超默认max_dist, 必为b1/b3结构容差): '
          f'{int((d_ok["dist"] > 0.30).sum())}行')
    print(f'dist>0.40(超最大容差): {int((d_ok["dist"] > 0.40).sum())}行 '
          f'(自检应≈0, b3+SL2容差0.40)')

    # ---- B) 全buy行距边界 (候选池修剪率) ----
    print('\n=== B) 全buy行门控修剪 ===')
    b = pd.DataFrame({'code': codes[buy_rows], 'dt64': dts[buy_rows],
                      'score': scores[buy_rows]})
    n_low = int((b['score'] < 0.05).sum())
    n_neg = int((b['score'] < 0).sum())
    print(f'全buy {len(b)}行: score<0.05 {n_low} ({100*n_low/len(b):.1f}%), '
          f'score<0 {n_neg} ({100*n_neg/len(b):.1f}%)')
    b['dist'] = b.apply(lambda r: dist_at(r['code'], r['dt64']), axis=1)
    bd = b[b['dist'].notna()]
    print(f'dist可算 {len(bd)}行:')
    print(bd['dist'].describe().round(3).to_string())
    print(f'dist<0: {int((bd["dist"] < 0).sum())} ({100*(bd["dist"]<0).mean():.1f}%) '
          f'— 需结构豁免才可买')
    print(f'dist>0.30: {int((bd["dist"] > 0.30).sum())} ({100*(bd["dist"]>0.30).mean():.1f}%)')
    print(f'dist∈[0.12,0.30](b6容差内其他类): {int(((bd["dist"]>=0.12)&(bd["dist"]<=0.30)).sum())}行')

    # 拒绝侧近似: 有多少stock-days是"接近买入"但被门控拒? 不可得(需全链复刻) —
    # 用buy行的边界邻近度代理门控作用力: 门控切的是score<−0.05或dist>max_dist
    # 或(无结构&dist<0)的候选 — 从buy行一侧看不到拒绝行。仅报通过侧。

    ex.to_csv('/tmp/probe_entrygate_20260922.csv', index=False)
    print('\n细节 → /tmp/probe_entrygate_20260922.csv (零生产写入)')


if __name__ == '__main__':
    main()
