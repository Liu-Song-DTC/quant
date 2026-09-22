#!/usr/bin/env python3
"""买入侧动态阈值绑定探针 (2026-09-22, 只读, 零生产写入)

背景: sell_threshold已被末班八项量化(执行层仅8+11笔落带, 旋转+C5c天然惰性),
但**买入侧**从未bracket过。机制 (signal_engine.py):
  - buy_th = per-stock rolling-400bar p30 的自身adjusted_score (shift(1),
    min_periods=100), floor = buy_threshold×0.15 (yaml: 0.0 → floor=0)
  - 唯一消费: fqg块 — premium标签 cut if score < buy_th×0.80 (行805);
    非premium cut if score < buy_th×max(restricted_mult,1.0) (行813)
  - 初始买入判定(行747)门槛极低(score≥−0.05), "让portfolio判断"

旋钮: buy_threshold_pct=0.3 (yaml signal节)。提pct → buy_th上移 → fqg切点
上移 → 更多买入被切。本探针给执行集(portfolio_selections 570行)的敏感性:
  1) 每个执行buy在pct∈{0.3,0.4,0.5,0.6,0.7}下的buy_th(生产口径逐位复刻);
  2) 边界b∈{0.8(premium),1.0(plain),1.25,1.5(restricted多档)}敏感性表:
     各(pct,b)组合切掉多少执行buy + 其weight×future_ret合计;
  3) 边界邻近带(ratio<1.3@p30)执行行的fwd画像 vs 远离带;
  4) 全buy行(36k)同口径 ratio分桶fwd — 与末班八项边际+0.44%互检。

裁决逻辑: 若全sweep切不到执行行 → 旋钮对执行集惰性, 免冷跑关闭(与sell_threshold
同签名); 若某档切到且切掉集合weight×fwd为负 → 候选臂(提pct, 冷跑四指标)。

零写入: 仅 /tmp/probe_buyth_pct_20260922.csv
"""
import os
import sys

import numpy as np
import pandas as pd

VAL_CSV = '/mnt/d/quant/strategy/rolling_validation_results/validation_results.csv'
SEL_CSV = '/mnt/d/quant/strategy/rolling_validation_results/portfolio_selections.csv'

PCTS = [0.3, 0.4, 0.5, 0.6, 0.7]
# 生产fqg精确边界乘数: premium=0.80, plain=1.0, _F=1.20, _FBA/_FV=1.25
BOUNDS = [0.8, 1.0, 1.2, 1.25]
WINDOW = 400
MIN_OBS = 100


def main():
    # ---- 全panel加载 (所有stock-days的adjusted_score) ----
    v = pd.read_csv(VAL_CSV, usecols=['date', 'code', 'score', 'buy', 'future_ret'],
                    low_memory=False, dtype={'code': str})
    v['date'] = pd.to_datetime(v['date'])
    v = v.sort_values(['code', 'date']).reset_index(drop=True)
    v['dt64'] = v['date'].values.astype('datetime64[ns]').astype('int64')
    print(f'panel: {len(v)}行, {v["code"].nunique()}股, '
          f'{v["date"].nunique()}日, buy行 {int(v["buy"].sum())}')

    # per-code numpy数组 (一次切分, 后续O(log n)定位)
    codes = v['code'].values
    dts = v['dt64'].values
    scores = v['score'].values.astype(np.float64)
    fwd = v['future_ret'].values.astype(np.float64)
    buys = v['buy'].values.astype(bool)

    starts = np.r_[0, np.flatnonzero(codes[1:] != codes[:-1]) + 1, len(codes)]
    code_arr = codes[starts[:-1]]
    code_pos = {c: i for i, c in enumerate(code_arr)}
    n_stocks = len(code_arr)
    print(f'per-code切分: {n_stocks}股')

    def buyth_at(code, dt64, pct):
        """生产口径: 该bar前400根自身score(排除当前bar)的p分位, <100有效→NaN"""
        i = code_pos.get(code)
        if i is None:
            return np.nan
        s0, s1 = starts[i], starts[i + 1]
        cd = dts[s0:s1]
        pos = np.searchsorted(cd, dt64)
        if pos >= s1 or cd[pos] != dt64:
            return np.nan  # 该bar无记录
        win = scores[max(s0, pos - WINDOW):pos]
        win = win[~np.isnan(win)]
        if len(win) < MIN_OBS:
            return np.nan
        return float(np.quantile(win, pct))

    # ---- 执行集: 精确(当日buy) + 陈信号(前15日最近buy) ----
    sel = pd.read_csv(SEL_CSV, dtype={'code': str})
    sel['date'] = pd.to_datetime(sel['date'])
    sel['dt64'] = sel['date'].values.astype('datetime64[ns]').astype('int64')
    print(f'\n执行记录: {len(sel)}行, {sel["date"].nunique()}日, {sel["code"].nunique()}股')

    # 全buy行index: (code, dt64) → panel行号
    buy_rows = np.flatnonzero(buys)
    buy_key = {}
    for r in buy_rows:
        buy_key[(codes[r], dts[r])] = r

    recs = []
    for _, s in sel.iterrows():
        r = buy_key.get((s['code'], s['dt64']))
        if r is not None:
            recs.append((s['date'], s['code'], s['weight'], s['dt64'], r,
                         scores[r], fwd[r], 'exact'))
            continue
        # 陈信号: 同code且 dt64 <= sel dt64 且 >= sel dt64 - 15d 的最近buy
        i = code_pos.get(s['code'])
        if i is not None:
            s0, s1 = starts[i], starts[i + 1]
            lo = s['dt64'] - 15 * 86400_000_000_000
            sub = buy_rows[(buy_rows >= s0) & (buy_rows < s1) &
                           (dts[buy_rows] <= s['dt64']) & (dts[buy_rows] >= lo)]
            if len(sub):
                r = sub[-1]
                recs.append((s['date'], s['code'], s['weight'], dts[r], r,
                             scores[r], fwd[r], 'stale'))
                continue
        recs.append((s['date'], s['code'], s['weight'], s['dt64'], -1,
                     np.nan, np.nan, 'none'))

    ex = pd.DataFrame(recs, columns=['sel_date', 'code', 'weight', 'buy_dt64',
                                     'row', 'score', 'future_ret', 'match'])
    print('执行行匹配: exact', (ex['match'] == 'exact').sum(),
          '| stale', (ex['match'] == 'stale').sum(),
          '| none', (ex['match'] == 'none').sum())

    # ---- 逐执行buy复算buy_th (生产口径) ----
    m = ex[ex['match'] != 'none'].copy()
    for p in PCTS:
        m[f'bt_{p}'] = m.apply(
            lambda r: buyth_at(r['code'], r['buy_dt64'], p), axis=1)
    n_th = int(m[f'bt_{PCTS[0]}'].notna().sum())
    print(f'\n执行buy行: {len(m)}, buy_th(p30)可复算: {n_th} '
          f'(不可算={len(m)-n_th}: 早期bar<100观测, 生产用静态floor 0.0)')

    # ---- 敏感性表: (pct, bound) → 切掉执行行数 + weight×fwd ----
    print('\n=== 敏感性表 (每格: 切掉执行buy数 | weight×future_ret合计) ===')
    print(f'{"pct":>4} | ' + ' | '.join(f'b={b}' for b in BOUNDS))
    table_rows = []
    for p in PCTS:
        bt = m[f'bt_{p}']
        cells = []
        for b in BOUNDS:
            cut = m[bt.notna() & (m['score'] < bt * b)]
            wfr = (cut['weight'] * cut['future_ret']).sum()
            cells.append(f'{len(cut):>2} | {wfr:+.4f}')
            table_rows.append({'pct': p, 'bound': b, 'n_cut': len(cut),
                               'w_fwd': float(wfr)})
        print(f'{p:>4} | ' + ' | '.join(cells))
    print('(p30生产行: b=0.8应=0自检[最松边界]; b≥1.0被切行=premium/plain标签合法过境, 非误差)')

    # ---- ratio分布: 执行buy的 score/buy_th(p30) ----
    r30 = m[m['bt_0.3'].notna()].copy()
    r30['ratio'] = r30['score'] / r30['bt_0.3']
    print('\n=== 执行buy score/buy_th(p30) ratio ===')
    print(r30['ratio'].describe().round(2).to_string())
    print(f'ratio<1.0: {int((r30["ratio"] < 1.0).sum())} (必为premium标签0.8边界过)')
    print(f'ratio<0.8: {int((r30["ratio"] < 0.8).sum())} (生产口径下不可能被执行, 自检应=0)')
    for lo, hi in [(1.0, 1.3), (1.3, 2.0), (2.0, np.inf)]:
        band = r30[(r30['ratio'] >= lo) & (r30['ratio'] < hi)]
        if len(band):
            print(f'  ratio[{lo},{hi}): {len(band)}行, '
                  f'weight×fwd={(band["weight"]*band["future_ret"]).sum():+.4f}, '
                  f'fwd均值{band["future_ret"].mean():+.5f}')
        else:
            print(f'  ratio[{lo},{hi}): 0行')

    # ---- 全buy行同口径 (互检末班八项边际发现 + 池收缩间接效应) ----
    print('\n=== 全buy行 p30边界邻近带fwd (互检末班八项) ===')
    allb = []
    for r in buy_rows:
        if r < starts[code_pos[codes[r]]] + MIN_OBS:  # 该股前100根内 → NaN阈值
            continue
        p = buyth_at(codes[r], dts[r], 0.3)
        if not np.isnan(p):
            allb.append((scores[r] / p, fwd[r]))
    ab = pd.DataFrame(allb, columns=['ratio', 'fwd'])
    print(f'可算: {len(ab)}/{len(buy_rows)} buy行')
    for lo, hi in [(0, 1.0), (1.0, 1.3), (1.3, 2.0), (2.0, np.inf)]:
        band = ab[(ab['ratio'] >= lo) & (ab['ratio'] < hi)]
        if len(band):
            print(f'  ratio[{lo},{hi}): {len(band)}行, fwd均值{band["fwd"].mean():+.5f}')

    # ---- 池收缩表: 全buy行在(pct,bound)下被切比例 (间接效应: 候选池→rank分位) ----
    print('\n=== 池收缩表 (全buy行被切%, 间接效应代理) ===')
    allb_t = []
    for r in buy_rows:
        s0, s1 = starts[code_pos[codes[r]]], starts[code_pos[codes[r]]] + 1
        p30v = buyth_at(codes[r], dts[r], 0.3)
        if not np.isnan(p30v):
            allb_t.append((codes[r], dts[r], scores[r]))
    abt = pd.DataFrame(allb_t, columns=['code', 'dt64', 'score'])
    print(f'{"pct":>4} | ' + ' | '.join(f'b={b}' for b in BOUNDS))
    for p in PCTS[1:]:  # p30行必≈0 (生产边界)
        bt = abt.apply(lambda r: buyth_at(r['code'], r['dt64'], p), axis=1)
        cells = []
        for b in BOUNDS:
            cut_pct = 100 * float((abt['score'] < bt * b).mean())
            cells.append(f'{cut_pct:5.1f}%')
        print(f'{p:>4} | ' + ' | '.join(cells))

    # 静态floor检查: 执行buy中 buy_th(p30)==0 (floor绑定) 的数量
    n_floor = int((r30['bt_0.3'] == 0).sum())
    print(f'\n静态floor检查: 执行buy中buy_th(p30)==0(floor绑定): {n_floor}行 '
          f'(提静态buy_threshold只影响这些+早期不可算行)')

    m.to_csv('/tmp/probe_buyth_pct_20260922.csv', index=False)
    print('细节 → /tmp/probe_buyth_pct_20260922.csv (零生产写入)')


if __name__ == '__main__':
    main()
