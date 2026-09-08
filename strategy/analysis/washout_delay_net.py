#!/usr/bin/env python
"""延迟入场机制净效果对账探针 (P1+P2, E-N13前置, 2026-09-09)

用户要求: 先仔细分析再跑机制 (跑一次很耗时)。
E-N11b反事实只测了55.7%洗盘笔的B-A收益(+4.25pp mean5), 没测另外44.3%
无洗盘笔(全场最赚: hit1 84.8%/实际平仓+8.23%)被延迟10天后追高接盘的成本。

本探针把"延迟入场机制"对全部519笔逐笔对账:
  机制 = 入场日锚定, 其后10个交易日内:
    - 收盘下探≥3%入场价 → 等坑后首次收盘>MA5且>MA10, 以收复日收盘价接
    - 无下探 → 第10个交易日收盘价接
    - 下探但坑后10日内未收复 → 跳过不接
  每笔延迟收益 = (1+实际ret) × avg_cost/延迟接价 - 1  (固定实际出场价, 一阶近似)
  出场日早于延迟接价日的笔单独报出(机制下出场时点会整体平移, 此近似失效)

P2: 入场日(信号日)可事前观测特征 × 洗盘率 × 实际/延迟平仓收益,
  找"可事前识别的洗盘前兆组" — 选择性延迟的目标池候选。

输出: 分桶对账表 + 特征×洗盘率表 + NAV一阶净效果估计。
"""
import os

import numpy as np
import pandas as pd

BASE = '/mnt/d/quant/strategy/rolling_validation_results'
DATA_DIR = '/mnt/d/quant/data/stock_data/backtrader_data'
TRADES = os.environ.get('WASH_TRADES') or f'{BASE}/trade_realized.csv'

_qfq_cache = {}


def load_qfq(code):
    if code in _qfq_cache:
        return _qfq_cache[code]
    p = os.path.join(DATA_DIR, f'{code}_qfq.csv')
    if not os.path.exists(p):
        _qfq_cache[code] = None
        return None
    df = pd.read_csv(p, usecols=['datetime', 'close', 'volume'])
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)
    _qfq_cache[code] = df
    return df


def main():
    trades = pd.read_csv(TRADES)
    trades['entry_date'] = pd.to_datetime(trades['entry_date'])
    trades['exit_date'] = pd.to_datetime(trades['exit_date'])
    trades['code'] = trades['code'].astype(str).str.zfill(6)

    sig = pd.read_csv(f'{BASE}/backtest_signals.csv',
                      usecols=['date', 'code', 'buy', 'chan_buy_point'], low_memory=False)
    sig['code'] = sig['code'].astype(str).str.zfill(6)
    sig['d'] = pd.to_datetime(sig['date'])
    sig = sig[sig['buy']].sort_values('d')
    t2 = trades.merge(sig, on='code', how='left', suffixes=('', '_sig'))
    t2 = t2[(t2['d'] <= t2['entry_date']) & (t2['d'] >= t2['entry_date'] - pd.Timedelta(days=3))]
    t2 = t2.sort_values('d').groupby(['entry_date', 'code'], as_index=False).tail(1)
    trades = t2
    trades['bp'] = trades['chan_buy_point'].map(
        lambda x: 'bp0' if x == 0 else ('bp1' if x == 1 else ('bp2' if x == 2 else f'bp{x}')))

    rows = []
    for _, t in trades.iterrows():
        dfp = load_qfq(t['code'])
        if dfp is None:
            continue
        i = dfp['datetime'].searchsorted(t['entry_date'])
        if i < 25 or i + 16 >= len(dfp):
            continue
        cl = dfp['close'].astype(float)
        entry_close = float(cl.iloc[i])
        # 洗盘窗口: 入场后10个交易日
        win = cl.iloc[i + 1:i + 11].values.astype(float)
        trough_rel = win.min() / entry_close - 1
        trough_i = i + 1 + int(np.argmin(win))
        has_washout = trough_rel <= -0.03
        reclaim_i = None
        if has_washout:
            for j in range(trough_i + 1, min(trough_i + 11, len(dfp) - 1)):
                ma5 = float(cl.iloc[j - 4:j + 1].mean())
                ma10 = float(cl.iloc[j - 9:j + 1].mean())
                if float(cl.iloc[j]) > ma5 and float(cl.iloc[j]) > ma10:
                    reclaim_i = j
                    break
        # 延迟接价
        if has_washout and reclaim_i is not None:
            d_i, d_px = reclaim_i, float(cl.iloc[reclaim_i])
            grp = 'washout+reclaim'
        elif has_washout:
            d_i, d_px = None, np.nan
            grp = 'washout_noreclaim'
        else:
            d_i, d_px = i + 10, float(cl.iloc[i + 10])
            grp = 'nodip'
        # 延迟收益 (固定实际出场价, 一阶近似)
        exit_i = dfp['datetime'].searchsorted(t['exit_date'])
        exited_before_entry = grp != 'washout_noreclaim' and exit_i <= d_i
        if grp == 'washout_noreclaim':
            d_ret = 0.0          # 机制跳过, 资金让给其他信号(机会成本未建模)
            exited_before_entry = False
        elif exited_before_entry:
            d_ret = np.nan        # 近似失效, 单独报出
        else:
            d_ret = (1.0 + t['ret']) * t['avg_cost'] / d_px - 1.0
        # 入场日特征 (信号日, 可事前观测)
        pct = float(cl.iloc[i] / cl.iloc[i - 1] - 1)
        is_star = t['code'].startswith(('300', '301', '688'))  # 创业板/科创板 20%板
        limup = pct >= (0.195 if is_star else 0.095)
        lim_thr = 0.195 if is_star else 0.095
        pre_lim = (float(cl.iloc[i - 1] / cl.iloc[i - 2] - 1) >= lim_thr
                   or float(cl.iloc[i - 2] / cl.iloc[i - 3] - 1) >= lim_thr)
        runup5 = float(cl.iloc[i] / cl.iloc[i - 5] - 1)
        vol5 = float(dfp['volume'].iloc[i - 4:i + 1].mean())
        vol20 = float(dfp['volume'].iloc[i - 19:i + 1].mean())
        vol_ratio = vol5 / vol20 if vol20 > 0 else np.nan
        high20 = float(cl.iloc[i - 20:i].max())
        dist_high = entry_close / high20 - 1 if high20 > 0 else np.nan
        rows.append({
            'grp': grp, 'ret': t['ret'], 'd_ret': d_ret, 'hold_days': t['hold_days'],
            'avg_cost': t['avg_cost'], 'entry_date': t['entry_date'],
            'bp': t['bp'], 'trough_rel': trough_rel,
            'chase': (d_px / t['avg_cost'] - 1) if d_i is not None else np.nan,
            'delay_days': (d_i - i) if d_i is not None else np.nan,
            'pct': pct, 'limup': limup, 'runup5': runup5,
            'vol_ratio': vol_ratio, 'dist_high': dist_high,
            'pre_lim': pre_lim, 'exited_before_entry': exited_before_entry,
        })
    m = pd.DataFrame(rows)
    print(f"可用: {len(m)} 笔")

    print("\n=== P1 逐笔对账: 实际 vs 延迟机制 (固定出场价一阶近似) ===")
    v = m[m['d_ret'].notna()].copy()
    print(f"有效n={len(v)} (排除出场早于延迟接价日{int(m['exited_before_entry'].sum())}笔)")
    a, d = v['ret'], v['d_ret']
    print(f"实际:   mean={a.mean()*100:+.2f}%  med={a.median()*100:+.2f}%  sum={a.sum()*100:+.0f}%")
    print(f"延迟:   mean={d.mean()*100:+.2f}%  med={d.median()*100:+.2f}%  sum={d.sum()*100:+.0f}%")
    print(f"差:     mean={(d-a).mean()*100:+.2f}pp  | 延迟更优占比={(d>a).mean()*100:.0f}%")
    try:
        eq = pd.read_csv(f'{BASE}/equity_curve.csv', usecols=['nav'])
        avg_conc = v['hold_days'].sum() / len(eq)
        nav_pp = (d.sum() - a.sum()) / avg_conc * 100
        print(f"NAV一阶换算: 平均并发持仓={avg_conc:.1f}只 | 总差={nav_pp:+.1f}pp NAV")
    except Exception:
        pass

    print("\n=== 分组对账 ===")
    for grp, label in [('nodip', '无洗盘→T+10接'), ('washout+reclaim', '洗盘→收复价接'),
                       ('washout_noreclaim', '未收复→跳过')]:
        sub = m[m['grp'] == grp]
        if len(sub) == 0:
            continue
        sv = sub[sub['d_ret'].notna()]
        print(f"{label} (n={len(sub)}"
              f"{'' if len(sv)==len(sub) else f', 有效{len(sv)}'}"
              f"{', 出场早于接价'+str(int(sub['exited_before_entry'].sum()))+'笔' if grp!='washout_noreclaim' else ''}):")
        print(f"    实际 mean={sub['ret'].mean()*100:+.2f}% | 延迟 mean={sv['d_ret'].mean()*100:+.2f}%"
              f" | 追高/折价 {label.split('→')[1]}: p10={sub['chase'].quantile(0.1)*100:+.1f}% "
              f"p50={sub['chase'].median()*100:+.1f}% p90={sub['chase'].quantile(0.9)*100:+.1f}%")
        if grp == 'nodip':
            print(f"    接价延迟天数: p50={sub['delay_days'].median():.0f}")

    print("\n=== 出场早于延迟接价日的笔 (机制下出场时点整体平移, 一阶近似失效) ===")
    eb = m[m['exited_before_entry']]
    if len(eb):
        print(f"n={len(eb)}: 实际 mean={eb['ret'].mean()*100:+.2f}% | "
              f"hold_days中位={eb['hold_days'].median():.0f} | 分组: "
              f"{eb['grp'].value_counts().to_dict()}")

    print("\n=== P2 入场日特征 × 洗盘率 × 收益 (可事前观测) ===")
    for col, buckets in [
        ('limup', {True: '信号日涨停', False: '非涨停'}),
    ]:
        for k, lab in buckets.items():
            sub = m[m[col] == k]
            wr = (sub['grp'].str.startswith('washout')).mean()
            sv = sub[sub['d_ret'].notna()]
            print(f"{lab} (n={len(sub)}): 洗盘率={wr*100:.0f}% | 实际={sub['ret'].mean()*100:+.2f}%"
                  f" | 延迟={sv['d_ret'].mean()*100:+.2f}%")
    for col, labs in [
        ('pct', [('单日≥5%', lambda x: x >= 0.05), ('2~5%', lambda x: 0.02 <= x < 0.05),
                 ('<2%', lambda x: x < 0.02)]),
        ('runup5', [('5日≥10%', lambda x: x >= 0.10), ('5~10%', lambda x: 0.05 <= x < 0.10),
                    ('<5%', lambda x: x < 0.05)]),
        ('vol_ratio', [('量比≥1.5', lambda x: x >= 1.5), ('1~1.5', lambda x: 1 <= x < 1.5),
                       ('<1', lambda x: x < 1)]),
        ('dist_high', [('贴新高>-1%', lambda x: x > -0.01), ('-1~-10%', lambda x: -0.10 < x <= -0.01),
                       ('≤-10%', lambda x: x <= -0.10)]),
    ]:
        for lab, f in labs:
            sub = m[m[col].apply(f)]
            if len(sub) < 15:
                continue
            wr = (sub['grp'].str.startswith('washout')).mean()
            sv = sub[sub['d_ret'].notna()]
            print(f"{lab} (n={len(sub)}): 洗盘率={wr*100:.0f}% | 实际={sub['ret'].mean()*100:+.2f}%"
                  f" | 延迟={sv['d_ret'].mean()*100:+.2f}%")
    print("\n=== 买点类 × 洗盘率 × 收益 ===")
    for bp in ['bp0', 'bp1', 'bp2']:
        sub = m[m['bp'] == bp]
        if len(sub) == 0:
            continue
        wr = (sub['grp'].str.startswith('washout')).mean()
        sv = sub[sub['d_ret'].notna()]
        print(f"{bp} (n={len(sub)}): 洗盘率={wr*100:.0f}% | 实际={sub['ret'].mean()*100:+.2f}%"
              f" | 延迟={sv['d_ret'].mean()*100:+.2f}%")

    print("\n=== P2b 配对有效样本 (同一笔的actual vs delayed, 消除配对偏差) ===")
    v = m[m['d_ret'].notna()].copy()
    v['diff'] = v['d_ret'] - v['ret']

    def paired(sub, lab):
        sv = sub[sub['d_ret'].notna()]
        if len(sv) < 12:
            return
        df = sv['d_ret'] - sv['ret']
        print(f"{lab} (n={len(sub)}, 有效{len(sv)}): 实际={sv['ret'].mean()*100:+.2f}% "
              f"延迟={sv['d_ret'].mean()*100:+.2f}% 差={df.mean()*100:+.2f}pp "
              f"延迟更优占比={(df>0).mean()*100:.0f}%")

    print("\n-- 单日涨幅桶 × 洗盘状态交叉 --")
    for lab, f in [('单日<2%', lambda x: x < 0.02), ('单日2~5%', lambda x: 0.02 <= x < 0.05),
                   ('单日≥5%', lambda x: x >= 0.05)]:
        sub = m[m['pct'].apply(f)]
        print(f"{lab}: 洗盘率={(sub['grp'].str.startswith('washout')).mean()*100:.0f}%")
        for grp in ['nodip', 'washout+reclaim', 'washout_noreclaim']:
            paired(sub[sub['grp'] == grp], f'    {grp}')

    print("\n-- 入场前2日内涨停 × 洗盘率 × 收益 (E-F1重新定位) --")
    for k, lab in [(True, '入场前2日有涨停'), (False, '前2日无涨停')]:
        sub = m[m['pre_lim'] == k]
        wr = (sub['grp'].str.startswith('washout')).mean()
        sv = sub[sub['d_ret'].notna()]
        print(f"{lab} (n={len(sub)}): 洗盘率={wr*100:.0f}% | 实际={sub['ret'].mean()*100:+.2f}% "
              f"| 延迟={sv['d_ret'].mean()*100:+.2f}%")

    print("\n-- bp2 × 单日涨幅 交叉 --")
    for lab, f in [('bp2+单日<2%', lambda r: r['bp'] == 'bp2' and r['pct'] < 0.02),
                   ('bp2+单日≥2%', lambda r: r['bp'] == 'bp2' and r['pct'] >= 0.02)]:
        paired(m[m.apply(f, axis=1)], lab)

    print("\n=== E-N14前奏: 洗盘收复加仓 一阶估计 (固定出场, 加仓=原仓50%) ===")
    # 与延迟入场不同: 保留原入场(赢家不受损), 收复日追加50%仓位, 出场价固定
    # 前提: 出场晚于收复日(否则无从加仓); 加仓资金另有来源(一阶假设可及)
    g = m.copy()
    g['add_ret'] = np.nan
    for idx, r in g.iterrows():
        if r['grp'] != 'washout+reclaim':
            g.loc[idx, 'add_ret'] = r['ret']          # 无洗盘/未收复: 不加仓, 原收益
            continue
        if not r['d_ret'] == r['d_ret']:              # 出场早于接价 → 不加仓
            g.loc[idx, 'add_ret'] = r['ret']
            continue
        # exit_px = (1+ret)*avg_cost; 收复日买入价 = d_px (chase列) — 用 avg_cost*(1+chase)
        px = r['avg_cost'] * (1 + r['chase'])
        if px <= 0:
            g.loc[idx, 'add_ret'] = r['ret']
            continue
        # 加仓臂收益 = (1+ret)*avg_cost/px - 1, 加仓50% → 组合收益 = (ret + 0.5*臂)/1.5
        arm = (1 + r['ret']) * r['avg_cost'] / px - 1
        g.loc[idx, 'add_ret'] = (r['ret'] + 0.5 * arm) / 1.5
    ga = g.dropna(subset=['add_ret'])
    print(f"n={len(ga)}: 实际 mean={ga['ret'].mean()*100:+.2f}% | "
          f"加仓 mean={ga['add_ret'].mean()*100:+.2f}% | "
          f"差={(ga['add_ret']-ga['ret']).mean()*100:+.2f}pp")
    for grp, lab in [('nodip', '无洗盘(不动)'), ('washout+reclaim', '洗盘收复(加50%)'),
                     ('washout_noreclaim', '未收复(不动)')]:
        sub = ga[ga['grp'] == grp]
        if len(sub) == 0:
            continue
        print(f"  {lab} (n={len(sub)}): 实际={sub['ret'].mean()*100:+.2f}% "
              f"加仓={sub['add_ret'].mean()*100:+.2f}%")
    for bp in ['bp0', 'bp1', 'bp2']:
        sub = ga[ga['bp'] == bp]
        if len(sub) == 0:
            continue
        print(f"  {bp} (n={len(sub)}): 实际={sub['ret'].mean()*100:+.2f}% "
              f"加仓={sub['add_ret'].mean()*100:+.2f}%")

    print("\n=== E-N15前置: 入场位置桶(相对20日高点) × 年度 × 实际平仓 ===")
    m2 = m.copy()
    m2['pos_bucket'] = m2['dist_high'].apply(
        lambda x: '高位(>-3%)' if x > -0.03 else ('深回调(≤-15%)' if x <= -0.15 else '回调中'))
    m2['year'] = pd.to_datetime(m2['entry_date']).dt.year
    piv = m2.pivot_table(index='pos_bucket', columns='year', values='ret',
                         aggfunc=['count', 'mean'])
    print(piv.round(4).to_string())
    for bkt in ['高位(>-3%)', '回调中', '深回调(≤-15%)']:
        sub = m2[m2['pos_bucket'] == bkt]
        wr = (sub['grp'].str.startswith('washout')).mean()
        print(f"{bkt} (n={len(sub)}): 洗盘率={wr*100:.0f}% 实际={sub['ret'].mean()*100:+.2f}%")


if __name__ == '__main__':
    main()
