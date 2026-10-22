#!/usr/bin/env python3
"""选股胜率臂判读 (2026-10-22). 读 *.$tag 产物, 报逐笔前瞻胜率/资金利用率/分年.

用法: python winrate_arm_1022.py base_tr1022 id0 new2 ah1 new2ah score45 ttno2
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, '/mnt/d/quant/.arms/probe')
import winrate_diag_1022 as W
import winrate_diag2_1022 as W2

R = '/mnt/d/quant/strategy/rolling_validation_results'


def load_arm(tag):
    eq = pd.read_csv(f'{R}/equity_curve.csv.{tag}', parse_dates=['date'])
    eq = eq.set_index('date')['nav'].astype(float).sort_index()
    cal = list(eq.index)
    nxt = {cal[i]: cal[i + 1] for i in range(len(cal) - 1)}
    sel = pd.read_csv(f'{R}/portfolio_selections.csv.{tag}', dtype={'code': str}, parse_dates=['date'])
    sel['code'] = sel['code'].str.zfill(6)
    sel['exec_date'] = sel['date'].map(nxt)
    trd = pd.read_csv(f'{R}/trade_realized.csv.{tag}', dtype={'code': str},
                      parse_dates=['entry_date', 'exit_date'])
    trd['code'] = trd['code'].str.zfill(6)
    m = sel.merge(trd[['entry_date', 'code', 'ret', 'hold_days']],
                  left_on=['exec_date', 'code'], right_on=['entry_date', 'code'], how='left')
    reg = pd.read_csv(f'{R}/regime_state.csv.{tag}', parse_dates=['date'])
    m['year'] = m['date'].dt.year
    m['half'] = np.where(m['year'] <= 2023, 'H1', 'H2')
    return eq, m, trd, reg


def metrics(nav, trd):
    r = nav.pct_change().dropna()
    peak = nav.cummax(); mdd = float((nav / peak - 1).min())
    yrs = len(nav) / 252.0
    cagr = (nav.iloc[-1] / nav.iloc[0]) ** (1 / yrs) - 1
    sharpe = r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else np.nan
    def pw(k):
        v = nav.values; n = len(v) // k
        ends = v[k - 1::k][:n]; starts = np.r_[v[0], ends[:-1]]
        return float(((ends / starts - 1) > 0).mean())
    mr = nav.resample('ME').last().pct_change()
    mr.iloc[0] = nav.resample('ME').last().iloc[0] / nav.iloc[0] - 1
    yr = nav.resample('YE').last().pct_change()
    yr.iloc[0] = nav.resample('YE').last().iloc[0] / nav.iloc[0] - 1
    w = trd.ret[trd.ret > 0]; l = -trd.ret[trd.ret < 0]
    return dict(nav=nav.iloc[-1], cagr=cagr, sharpe=sharpe, mdd=mdd,
                dW=float((r > 0).mean()), w5=pw(5), w20=pw(20),
                mW=float((mr > 0).mean()), yW=float((yr > 0).mean()),
                ntr=len(trd), tW=float((trd.ret > 0).mean()) if len(trd) else np.nan,
                tMean=float(trd.ret.mean()) if len(trd) else np.nan,
                PF=float(w.sum() / l.sum()) if len(l) and l.sum() > 0 else np.nan,
                hold=float(trd.hold_days.mean()) if len(trd) else np.nan,
                holdmed=float(trd.hold_days.median()) if len(trd) else np.nan)


def fwd(m):
    """逐笔前瞻 (次日开盘→+H收盘), 复用诊断函数."""
    m2 = W.add_forward(m)
    m2 = W2.add_extras(m2)
    out = {'nSel': len(m2)}
    for h in (1, 3, 5, 10, 20):
        c = f'w{h}'
        if c in m2:
            v = m2[c].dropna()
            out[f'w{h}'] = float(v.mean()) if len(v) else np.nan
            out[f'r{h}'] = float(m2[f'r{h}'].dropna().mean()) if f'r{h}' in m2 else np.nan
    for c in ('mfe20', 'hit10_20', 'hit20_20'):
        if c in m2:
            out[c] = float(m2[c].dropna().mean())
    return m2, out


def main():
    base = sys.argv[1]
    tags = sys.argv[2:]
    allm = {}
    print(f'# base={base}')
    for tag in [base] + tags:
        if tag in allm:
            continue
        eq, m, trd, reg = load_arm(tag)
        mt = metrics(eq, trd)
        m2, fw = fwd(m)
        allm[tag] = (m2, mt, trd, reg)
        print(f'\n=== {tag} ===')
        print(f'  NAV={mt["nav"]:.2f} CAGR={mt["cagr"]:.2%} Sharpe={mt["sharpe"]:.4f} '
              f'MDD={mt["mdd"]:.2%} dW={mt["dW"]:.2%} w5={mt["w5"]:.2%} w20={mt["w20"]:.2%} '
              f'mW={mt["mW"]:.2%} yW={mt["yW"]:.0%}')
        print(f'  nSel={fw["nSel"]} nTr={mt["ntr"]} tW={mt["tW"]:.2%} tMean={mt["tMean"]:+.2%} '
              f'PF={mt["PF"]:.2f} hold={mt["hold"]:.1f}d(med {mt["holdmed"]:.0f}) '
              f'笔/年={mt["ntr"]/(len(eq)/252):.1f}')
        print(f'  前瞻: ' + ' '.join(
            f'w{h}={fw.get(f"w{h}", np.nan)*100:.1f}%/r{h}={fw.get(f"r{h}", np.nan)*100:+.2f}%'
            for h in (1, 3, 5, 10, 20)))
        print(f'  mfe20={fw.get("mfe20", np.nan)*100:.1f}% hit10={fw.get("hit10_20", np.nan)*100:.1f}% '
              f'hit20={fw.get("hit20_20", np.nan)*100:.1f}%')
        # 资金利用率
        zpos = float((reg['n_pos'] == 0).mean())
        zret = float((eq.pct_change().dropna().abs() < 1e-9).mean())
        print(f'  利用率: 持仓={reg["n_pos"].mean():.2f}只 目标敞口={reg["exposure"].mean():.3f} '
              f'n_pos=0日={zpos:.1%} 日收益=0={zret:.1%}')
        # 分年 tW / w5 / w20
        rows = []
        for y, g in m2.groupby('year'):
            rr = g['ret'].dropna()
            rows.append((y, len(g), float((rr > 0).mean()) if len(rr) else np.nan,
                         float(g['w5'].dropna().mean()) if g['w5'].notna().any() else np.nan,
                         float(g['w20'].dropna().mean()) if g['w20'].notna().any() else np.nan,
                         float(g['r20'].dropna().mean()) if g['r20'].notna().any() else np.nan))
        print('  分年: ' + ' | '.join(
            f'{y}:n{n}/tW{t*100:.0f}/w5{w5*100:.0f}/w20{w20*100:.0f}/r20{r*100:+.1f}'
            for y, n, t, w5, w20, r in rows))
        # 分段
        for half, g in m2.groupby('half'):
            rr = g['ret'].dropna()
            print(f'    {half}: n={len(g)} tW={(rr>0).mean()*100:.1f}% '
                  f'w5={g["w5"].dropna().mean()*100:.1f}% w20={g["w20"].dropna().mean()*100:.1f}% '
                  f'r20={g["r20"].dropna().mean()*100:+.2f}%')

    # 对比表 + Jaccard
    if base in allm:
        bm = allm[base][0]
        bs = set(zip(bm['date'].dt.strftime('%Y-%m-%d'), bm['code']))
        print('\n\n=== 汇总 (vs base) ===')
        print(f'{"tag":10s} {"NAV":>12s} {"ΔNAV":>10s} {"Sharpe":>7s} {"MDD":>7s} '
              f'{"nTr":>4s} {"tW":>6s} {"Δtw":>6s} {"w5":>6s} {"w20":>6s} {"r20":>7s} '
              f'{"hold":>5s} {"expo":>5s} {"Jacc":>5s}')
        bnav = allm[base][1]['nav']; btw = allm[base][1]['tW']
        for tag, (m2, mt, trd, reg) in allm.items():
            s = set(zip(m2['date'].dt.strftime('%Y-%m-%d'), m2['code']))
            j = len(s & bs) / len(s | bs) if (s | bs) else np.nan
            print(f'{tag:10s} {mt["nav"]:12.2f} {mt["nav"]-bnav:10.2f} {mt["sharpe"]:7.4f} '
                  f'{mt["mdd"]:7.2%} {mt["ntr"]:4d} {mt["tW"]:6.2%} {mt["tW"]-btw:+6.2%} '
                  f'{float(m2["w5"].dropna().mean()):6.2%} {float(m2["w20"].dropna().mean()):6.2%} '
                  f'{float(m2["r20"].dropna().mean()):+7.2%} {mt["hold"]:5.1f} '
                  f'{reg["exposure"].mean():5.3f} {j:5.2f}')


if __name__ == '__main__':
    main()
