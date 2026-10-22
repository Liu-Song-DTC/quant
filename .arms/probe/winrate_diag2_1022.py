#!/usr/bin/env python3
"""选股胜率诊断 v2 (2026-10-22) — 只读.

v1 的 D1 已完成; 本脚本补:
  - D2 前瞻胜率 + 涨幅捕获 (MFE / 大牛股命中率)
  - D3 两时段稳定性
  - D0' 信号层 (ml_score/adjusted_score/signal_level/bi_buy_point/chan_buy_point/industry)
        信号行已用 grep 从 3.25GB 备份抽出 → /tmp/wr_sig_rows.csv (247 行)
  - E1 资金利用率/周转 (持仓数分布/持有天数/交易频率)
  - X1 分数地板离线反事实: 丢弃低分入选后, 胜率/涨幅如何变
"""
import io
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, '/mnt/d/quant/.arms/probe')
import winrate_diag_1022 as W

RVD = W.RVD
SIGROWS = '/tmp/wr_sig_rows.csv'
SIGHDR = '/tmp/wr_sig_hdr.csv'
HORIZONS = [1, 3, 5, 10, 20]


def wilson(k, n, z=1.96):
    return W.wilson(k, n)


def ci(k, n):
    lo, hi = wilson(k, n)
    return f'{lo:.0f}-{hi:.0f}'


def add_extras(m):
    """MFE/大牛命中: 用 high/low, 从执行日开盘入场算."""
    recs = []
    for _, r in m.iterrows():
        d = None
        f = f'{W.BT}/{r["code"]}_qfq.csv'
        try:
            d = pd.read_csv(f, usecols=['datetime', 'open', 'high', 'low', 'close'])
            d['datetime'] = pd.to_datetime(d['datetime'])
            d = d.sort_values('datetime').reset_index(drop=True)
        except Exception:
            recs.append({})
            continue
        if pd.isna(r['exec_date']):
            recs.append({})
            continue
        dt = d['datetime'].values
        ei = int(np.searchsorted(dt, np.datetime64(r['exec_date'])))
        if ei >= len(d):
            recs.append({})
            continue
        entry = float(d['open'].iloc[ei])
        out = {'entry_open2': entry}
        if entry <= 0:
            recs.append(out)
            continue
        for h in (3, 5, 10, 20):
            j = min(ei + h, len(d) - 1)
            seg_h = d['high'].iloc[ei:j + 1]
            seg_l = d['low'].iloc[ei:j + 1]
            if len(seg_h):
                out[f'mfe{h}'] = float(seg_h.max()) / entry - 1
                out[f'mae{h}'] = float(seg_l.min()) / entry - 1
        for lvl in (0.05, 0.10, 0.20):
            out[f'hit{int(lvl*100)}_20'] = int(out.get('mfe20', np.nan) >= lvl)
        recs.append(out)
    return pd.concat([m, pd.DataFrame(recs, index=m.index)], axis=1)


def show2(df, by, label, cols=None):
    print(f'\n--- {label} ---')
    if df.empty:
        print('  (空)')
        return
    rows = []
    for key, g in df.groupby(by, observed=True, dropna=False):
        n = len(g)
        if n < 3:
            continue
        r = {'grp': str(key), 'n': n}
        rr = g['ret'].dropna() if 'ret' in g else pd.Series(dtype=float)
        if len(rr) >= 3:
            r['tW%'] = round((rr > 0).mean() * 100, 1)
            r['tCI'] = ci(int((rr > 0).sum()), len(rr))
        for h in HORIZONS:
            c = f'w{h}'
            if c in g:
                v = g[c].dropna()
                if len(v):
                    r[f'w{h}%'] = round(float(v.mean() * 100), 1)
                    r[f'r{h}%'] = round(float(g[f'r{h}'].dropna().mean() * 100), 2)
        for c in ('mfe5', 'mfe20', 'hit10_20', 'hit20_20'):
            if c in g:
                v = g[c].dropna()
                if len(v):
                    r[c] = round(float(v.mean()) * (100 if c.startswith('hit') else 1), 1)
        rows.append(r)
    out = pd.DataFrame(rows)
    if cols:
        out = out[[c for c in cols if c in out]]
    print(out.to_string(index=False))


def part_signal(m):
    hdr = pd.read_csv(SIGHDR, nrows=0).columns.tolist()
    sig = pd.read_csv(SIGROWS, header=None, names=hdr)
    sig['code'] = sig['code'].astype(str).str.zfill(6)
    sig['date'] = pd.to_datetime(sig['date']).dt.strftime('%Y-%m-%d')
    m2 = m.copy()
    m2['date_s'] = m2['date'].dt.strftime('%Y-%m-%d')
    m2 = m2.merge(sig.drop(columns=['score'], errors='ignore'),
                  left_on=['code', 'date_s'], right_on=['code', 'date'], how='left')
    print(f'\n信号层命中 {m2["ml_score"].notna().sum()}/{len(m2)}')
    mm = m2[m2['w5'].notna()].copy()
    for c in ['ml_score', 'adjusted_score', 'pre_discount_score', 'factor_value']:
        if c in mm and mm[c].notna().sum() >= 50:
            mm[f'{c}_q'] = pd.qcut(mm[c], 5, duplicates='drop')
            show2(mm, f'{c}_q', f'前瞻胜率/涨幅: {c} 五分位',
                  ['grp', 'n', 'tW%', 'tCI', 'w5%', 'w20%', 'r20%', 'mfe20', 'hit10_20', 'hit20_20'])
    for c in ['signal_level', 'bi_buy_point', 'chan_buy_point', 'gate_quality', 'trend_type']:
        if c in mm and mm[c].notna().sum() >= 20:
            show2(mm, c, f'前瞻胜率/涨幅: {c}',
                  ['grp', 'n', 'tW%', 'tCI', 'w5%', 'w20%', 'r20%', 'mfe20', 'hit10_20', 'hit20_20'])
    if 'industry' in mm:
        ind = mm.groupby('industry').agg(n=('w5', 'size'), w5=('w5', 'mean'),
                                         w20=('w20', 'mean'), r20=('r20', 'mean'),
                                         mfe20=('mfe20', 'mean'), tW=('twin', 'mean'))
        ind = ind[ind['n'] >= 15].sort_values('w5', ascending=False)
        for c in ('w5', 'w20', 'tW'):
            ind[c] = (ind[c] * 100).round(1)
        ind['r20'] = (ind['r20'] * 100).round(2)
        ind['mfe20'] = (ind['mfe20'] * 100).round(1)
        print('\n--- 前瞻胜率: 行业 (n≥15) ---')
        print(ind.to_string())


def part_util(m):
    print('\n' + '=' * 78)
    print('E1 资金利用率 / 周转 (全部入选, 含未成交)')
    trd = m[m['ret'].notna()].copy()
    print(f'入选 {len(m)}; 成交 {len(trd)}; 调仓日 {m["date"].nunique()}')
    print(f'年均成交 {len(trd) / (m["date"].max() - m["date"].min()).days * 365:.1f} 笔; '
          f'平均持有 {trd["hold_days"].mean():.1f} 天, 中位 {trd["hold_days"].median():.0f}')
    hb = pd.cut(trd['hold_days'], [0, 5, 10, 20, 40, 80, 999],
                labels=['≤5', '6-10', '11-20', '21-40', '41-80', '>80'])
    show2(trd.assign(hold_b=hb), 'hold_b', '按持有天数',
          ['grp', 'n', 'tW%', 'tCI', 'r20%', 'mfe20', 'hit10_20', 'hit20_20'])
    # 持仓数分布 (调仓日)
    reg = pd.read_csv(W.REG, parse_dates=['date'])
    vc = reg['n_pos'].value_counts().sort_index()
    tot = len(reg)
    print('\n调仓日持仓数分布 (regime_state):')
    for k, v in vc.items():
        print(f'  n_pos={k}: {v} 天 ({v/tot*100:.1f}%)')
    print(f'  平均持仓数 {reg["n_pos"].mean():.2f}; 平均目标敞口 {reg["exposure"].mean():.3f}')
    # 实际资金利用: nav 日收益=0 的天数占比 (空仓/满仓现金)
    eq = pd.read_csv(W.EQ, parse_dates=['date'])
    r = eq['daily_ret'].dropna()
    print(f'日收益=0 占比 {float((r == 0).mean())*100:.1f}%; |日收益|<1e-9 占比 {float((r.abs() < 1e-9).mean())*100:.1f}%')
    print(f'年化日波动 {float(r.std()*np.sqrt(252))*100:.1f}%')
    # 交易收益分布
    print(f'\n胜笔均值 {trd.loc[trd.ret>0,"ret"].mean()*100:+.2f}% (n={int((trd.ret>0).sum())}); '
          f'负笔均值 {trd.loc[trd.ret<=0,"ret"].mean()*100:+.2f}% (n={int((trd.ret<=0).sum())})')
    pf = trd.loc[trd.ret > 0, 'ret'].sum() / abs(trd.loc[trd.ret <= 0, 'ret'].sum())
    print(f'PF={pf:.2f}; 收益 std={trd.ret.std()*100:.1f}%')


def part_floor(m):
    print('\n' + '=' * 78)
    print('X1 分数地板离线反事实 (只看保留样本的 tW/w5/w20; 不含账簿路径)')
    mm = m[(m['ret'].notna()) & m['w5'].notna()].copy()
    print(f'样本 {len(mm)}')
    for q in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40):
        thr = mm['score'].quantile(q)
        keep = mm[mm['score'] > thr]
        drop = mm[mm['score'] <= thr]
        if len(keep) < 20 or len(drop) < 5:
            continue
        kw, dw = int((keep['ret'] > 0).sum()), int((drop['ret'] > 0).sum())
        print(f'  地板 q{q:.2f} (score>{thr:.3f}): 保留 {len(keep)} tW={kw/len(keep)*100:.1f}% '
              f'w5={keep.w5.mean()*100:.1f}% w20={keep.w20.mean()*100:.1f}% r20={keep.r20.mean()*100:+.2f}% '
              f'| 剔除 {len(drop)} tW={dw/len(drop)*100:.1f}% w5={drop.w5.mean()*100:.1f}%')
    print('\n绝对分数地板 (对照 min_absolute_score 直接语义):')
    for thr in (0.20, 0.30, 0.40, 0.45, 0.50, 0.55, 0.60):
        keep = mm[mm['score'] > thr]
        drop = mm[mm['score'] <= thr]
        if len(keep) < 20 or len(drop) < 3:
            continue
        print(f'  score>{thr:.2f}: 保留 {len(keep)} tW={(keep.ret>0).mean()*100:.1f}% '
              f'w5={keep.w5.mean()*100:.1f}% w20={keep.w20.mean()*100:.1f}% r20={keep.r20.mean()*100:+.2f}% '
              f'| 剔除 {len(drop)} tW={(drop.ret>0).mean()*100:.1f}% w5={drop.w5.mean()*100:.1f}%')
    print('\n名次离线反事实 (同日按 score 排名, 只丢弃某名次):')
    for k in (1, 2, 3):
        keep = mm[mm['srank'] <= k]
        drop = mm[mm['srank'] > k]
        print(f'  只留前{k}名: 保留 {len(keep)} tW={(keep.ret>0).mean()*100:.1f}% '
              f'w5={keep.w5.mean()*100:.1f}% w20={keep.w20.mean()*100:.1f}% r20={keep.r20.mean()*100:+.2f}%'
              + (f' | 剔除 {len(drop)} tW={(drop.ret>0).mean()*100:.1f}% w5={drop.w5.mean()*100:.1f}%' if len(drop) else ''))


def main():
    m, eq = W.load_panel()
    print(f'# 入选 {len(m)} 行; 有成交 {m["ret"].notna().sum()}; '
          f'{m["date"].min().date()}..{m["date"].max().date()}')
    m = W.add_forward(m)
    m = add_extras(m)
    W.part_a(m)          # D1 + D2 + D3
    part_signal(m)       # 信号层
    part_util(m)         # 资金利用率/周转
    part_floor(m)        # 分数地板离线
    print('\n判定提示: n≥50、两段同号、w5/w20 同向、差异≥5pp 才算信号.')


if __name__ == '__main__':
    main()
