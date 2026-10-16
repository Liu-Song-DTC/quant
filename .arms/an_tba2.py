#!/usr/bin/env python
"""tba2 家族(换手保护/E-A2 强度比)评估 — 2026-10-15
用法: /mnt/d/quant/.venv/bin/python /mnt/d/quant/.arms/an_tba2.py [face]
读 $RVD/equity_curve.csv.<tag>, 与 ident(基线) 配对比较.
配对铁律: 月度块 bootstrap 同一月份抽样同时喂两臂.
同族校准: 对已采纳臂 mp4/fec1/fec0/fec3/fec4 同跑"剔最大N月", 无判别力的判据不作否决依据.
"""
import sys, os
import numpy as np
import pandas as pd

RVD = '/mnt/d/quant/strategy/rolling_validation_results'
FACE = sys.argv[1] if len(sys.argv) > 1 else 'tba2'

FACES = {
    'tba2': ['ident', 'a2_inf', 'a2_500', 'a2_300', 'a2_200', 'a2_300_tb_30'],
    'tba2b': ['ident', 'a2_150', 'a2_180', 'a2_200', 'a2_250', 'a2_300'],
    'icap': ['ident', 'icap30', 'icap50', 'icap60', 'icap100'],
    'tb2': ['ident', 'tb05', 'tb15', 'tb20', 'tb30'],
    'rw': ['ident', 'rw00', 'rw50', 'rw70', 'rw100'],
    'mhd': ['ident', 'mhd20', 'mhd30', 'mhd45', 'mhd60'],
    'mrp': ['ident', 'mrp20', 'mrp40', 'mrp50', 'mrp60'],
    'rwc': ['ident', 'rwc20', 'rwc30', 'rwc375', 'rwc100'],
    'slk': ['ident', 'slk100', 'slk175', 'slk2125', 'slkfull'],
    'slk2': ['ident', 'slk100', 'slk160', 'slk170', 'slk175', 'slk180', 'slk200', 'slk2125', 'slkfull'],
    'esp': ['ident', 'esp60', 'esp70', 'esp90', 'esp100'],
    'esp2': ['ident', 'esp85', 'esp90', 'esp95', 'esp100'],
    # esp3: 基线换成 slk180 采纳后的新生产态(_SC=1.80) — 两旋钮抢同一块地盘(新腿资金)
    'esp3': ['ident2', 'esp3_85', 'esp3_90', 'esp3_95', 'esp3_100'],
    'esp3b': ['ident2', 'esp3_85', 'esp3_90', 'esp3_92', 'esp3_95', 'esp3_97', 'esp3_100'],
    # tvol: 敞口时序(target_volatility)首次bracket; 基线=10-15数据态生产态
    'tvol': ['ident3', 'tvol_off', 'tvol_20', 'tvol_25', 'tvol_32', 'tvol_36', 'tvol_45'],
    # rean: 重锚归因 — 配置回退阶梯, 全部跑在同一 10-15 数据态
    'rean': ['r_prod', 'r_slk', 'r_ea2', 'r_fec', 'r_pool14'],
    # rebr(10/16): 已采纳臂面在新数据态+新信号集下复扫 (基线 b_ident=anchor_1015d dca8678b)
    'rebr': ['b_ident', 'b_tel', 'b_fec0', 'b_fec2', 'b_fec3', 'b_fec4',
             'b_a2_150', 'b_a2_250', 'b_a2_300',
             'b_slk170', 'b_slk175', 'b_slk200'],
    # mp2(10/16): max_positions 左缘 — 旧面 {3,4,5,6,8} 从未探到 3 以下.
    #   左坡浅于右坡(mp3 −27,085 vs mp5 −76,781) ⇒ 峰可能更左. mp=2 时 slack=0.5 ⇒
    #   天花板 0.4375(vs mp4 0.45, 差 2.8% 可接受); mp=1 因 slack=0 天花板塌回 0.25 属混淆点, 不跑.
    'mp2': ['c_ident', 'c_mp2', 'c_mp3', 'c_mp5', 'c_mp6'],
    # esp4(10/16): entry_speed 面在**新数据态**复扫 — 旧否决(10/15)的叙述引用了 es=1.00 坍缩端点的
    #   数字, 近区 es=0.90 实为 六年全正/剔3月+1.34(全族最强)/Sharpe 仅 −0.013. 判据=能否复现.
    'esp4': ['d_ident', 'd_esp85', 'd_esp90', 'd_esp95'],
}
LABEL = {
    'ident': 'ident(基线 r=1.15 tb=0.10)',
    'a2_inf': 'r=99(保护恒生效)',
    'a2_500': 'r=5.0',
    'a2_300': 'r=3.0',
    'a2_200': 'r=2.0',
    'a2_300_tb_30': 'r=3.0 + tb=0.30',
    'a2_150': 'r=1.50', 'a2_180': 'r=1.80', 'a2_250': 'r=2.50',
    'icap30': '行业上限0.30', 'icap50': '行业上限0.50(=mp6语义)',
    'icap60': '行业上限0.60', 'icap100': '行业上限1.00(无闸)',
    'tb05': 'tb=0.05 (s≈0.020)', 'tb15': 'tb=0.15 (s≈0.059)',
    'tb20': 'tb=0.20 (s≈0.078)', 'tb30': 'tb=0.30 (s≈0.117)',
    'rw00': '排名ww=0.0(纯截面)', 'rw50': '排名ww=0.5',
    'rw70': '排名ww=0.7', 'rw100': '排名ww=1.0(纯行业内)',
    'mhd20': '最小持仓20日', 'mhd30': '最小持仓30日',
    'mhd45': '最小持仓45日', 'mhd60': '最小持仓60日',
    'mrp20': 'min_rank_pct=0.20', 'mrp40': 'min_rank_pct=0.40',
    'mrp50': 'min_rank_pct=0.50', 'mrp60': 'min_rank_pct=0.60',
    'rwc20': '单票上限0.20(收紧)', 'rwc30': '单票上限0.30',
    'rwc375': '单票上限0.375', 'rwc100': '单票上限1.00(无上限)',
    'slk100': 'slack_cap=1.00', 'slk175': 'slack_cap=1.75',
    'slk2125': 'slack_cap=2.125', 'slkfull': 'slack全开(M=3,C=3)',
    'slk160': 'slack_cap=1.60', 'slk170': 'slack_cap=1.70',
    'slk180': 'slack_cap=1.80', 'slk200': 'slack_cap=2.00',
    'esp60': 'entry_speed=0.60', 'esp70': 'entry_speed=0.70',
    'esp85': 'entry_speed=0.85', 'esp90': 'entry_speed=0.90', 'esp95': 'entry_speed=0.95', 'esp100': 'entry_speed=1.00(入场日足额)',
    'ident2': 'slk180基线(_SC=1.80, 新生产)',
    'esp3_85': 'slk180+entry_speed=0.85', 'esp3_90': 'slk180+entry_speed=0.90',
    'esp3_95': 'slk180+entry_speed=0.95', 'esp3_100': 'slk180+entry_speed=1.00',
    'esp3_92': 'slk180+entry_speed=0.92', 'esp3_97': 'slk180+entry_speed=0.97',
    'ident3': '10-15数据态生产态(基线)',
    'tvol_off': 'vol_scale关闭(地板=1.0)', 'tvol_20': 'target_vol=0.20(绑定40.5%)',
    'tvol_25': 'target_vol=0.25(绑定22.9%)',
    'tvol_32': 'target_vol=0.32', 'tvol_36': 'target_vol=0.36',
    'tvol_45': 'target_vol=0.45(≈关闭)',
    # rean: 重锚归因回退阶梯 (10-15 数据态)
    'r_prod': '现生产 fec1+ea2_2.0+slk180',
    'r_slk': '撤 slk180(_SC=1.5)',
    'r_ea2': '再撤 ea2(r=1.15)',
    'r_fec': '再撤 fec1(=anchor_1014b配置)',
    'r_pool14': 'POOL_TODATE=2026-10-14',
    # rebr
    'b_ident': '基线=anchor_1015d(生产)', 'b_tel': '遥测臂(应=基线)',
    'b_fec0': 'fast_entry_cap=0', 'b_fec2': 'fast_entry_cap=2',
    'b_fec3': 'fast_entry_cap=3', 'b_fec4': 'fast_entry_cap=4',
    'b_a2_150': 'ea2_ratio=1.50', 'b_a2_250': 'ea2_ratio=2.50', 'b_a2_300': 'ea2_ratio=3.00',
    'b_slk170': 'slack_cap=1.70', 'b_slk175': 'slack_cap=1.75', 'b_slk200': 'slack_cap=2.00',
    # mp2
    'c_ident': '基线=anchor_1015d(生产)', 'c_mp2': 'max_positions=2', 'c_mp3': 'max_positions=3',
    'c_mp5': 'max_positions=5', 'c_mp6': 'max_positions=6',
    # esp4
    'd_ident': '基线=anchor_1015d(生产)', 'd_esp85': 'entry_speed=0.85',
    'd_esp90': 'entry_speed=0.90', 'd_esp95': 'entry_speed=0.95',
}
TAGS = FACES[FACE]
# tba2/tba2b 的臂是相对**旧锚**跑出来的; icap 相对新锚(anchor_1015b, ea2_ratio=2.0)
BASE_TAG = {'tba2': 'ident_1015', 'tba2b': 'ident_1015', 'esp3': 'ident2', 'esp3b': 'ident2',
            'tvol': 'ident3', 'rean': 'r_prod', 'rebr': 'b_ident', 'mp2': 'c_ident',
            'esp4': 'd_ident'}.get(FACE, 'ident')
# 同族已采纳对照(用于校准否决判据)
CALIB = ['mp4', 'fec0', 'fec1', 'fec3', 'fec4', 'mp3', 'mp5']


def load(tag):
    p = f'{RVD}/equity_curve.csv.{tag}'
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p, parse_dates=['date'])
    df = df.dropna(subset=['nav']).reset_index(drop=True)
    return df


def monthly(df):
    s = df.set_index('date')['nav']
    m = s.resample('ME').last()
    r = m.pct_change()
    r.iloc[0] = m.iloc[0] / 250000.0 - 1.0
    return r.dropna()


def metrics(df):
    s = df['nav'].values
    r = np.diff(s) / s[:-1]
    n = len(s)
    yrs = n / 244.0
    nav = s[-1]
    tot = nav / s[0] - 1.0
    ann = (nav / s[0]) ** (1.0 / yrs) - 1.0
    sd = r.std(ddof=1)
    sharpe = (r.mean() / sd) * np.sqrt(244) if sd > 0 else 0.0
    peak = np.maximum.accumulate(s)
    mdd = ((s - peak) / peak).min()
    return dict(nav=nav, tot=tot, ann=ann, sharpe=sharpe, mdd=mdd)


def ann_from_ratio(ratio, yrs):
    return ratio ** (1.0 / yrs) - 1.0


def boot_pair(rA, rB, n_iter=5000, seed=20261015):
    """配对月度块 bootstrap: 同一月份抽样喂两臂. 返回 Δann 的 CI 与 P(A>B)."""
    idx = rA.index.intersection(rB.index)
    a = (1.0 + rA.loc[idx]).values
    b = (1.0 + rB.loc[idx]).values
    m = len(idx)
    yrs = m / 12.0
    rng = np.random.default_rng(seed)
    d = np.empty(n_iter)
    ratio = np.empty(n_iter)
    for i in range(n_iter):
        j = rng.integers(0, m, m)
        na = a[j].prod()
        nb = b[j].prod()
        d[i] = ann_from_ratio(na, yrs) - ann_from_ratio(nb, yrs)
        ratio[i] = na / nb
    return d, ratio


def drop_months(rA, rB, months, idx):
    keep = [d for d in idx if d not in set(months)]
    yrs = len(keep) / 12.0
    na = (1.0 + rA.loc[keep]).prod()
    nb = (1.0 + rB.loc[keep]).prod()
    return ann_from_ratio(na, yrs) - ann_from_ratio(nb, yrs)


def trades(tag):
    p = f'{RVD}/trade_realized.csv.{tag}'
    if not os.path.exists(p):
        return None
    try:
        t = pd.read_csv(p)
    except Exception:
        return None
    return t


def main():
    base = load(BASE_TAG)
    if base is None:
        print('ident 产物缺失'); return
    mb = monthly(base)
    mb_idx = mb.index
    m_base = metrics(base)
    print(f'=== 面[{FACE}] 基线 ident NAV={m_base["nav"]:,.0f} '
          f'收益{m_base["tot"]*100:.2f}% Sharpe={m_base["sharpe"]:.4f} MDD={m_base["mdd"]*100:.2f}% ===\n')

    rows = []
    for tag in TAGS:
        df = load(tag)
        if df is None:
            print(f'  [缺] {tag}'); continue
        mm = metrics(df)
        if tag == 'ident':
            rows.append((tag, mm, 0.0, None, None))
            continue
        ma = monthly(df)
        d, ratio = boot_pair(ma, mb)
        lo, hi = np.percentile(d, [2.5, 97.5])
        p_gt = float((ratio > 1.0).mean())
        rows.append((tag, mm, float(d.mean()), (lo, hi), p_gt))

    print(f'{"臂":<26} {"NAV":>13} {"总收益":>9} {"年化":>8} {"Sharpe":>8} {"MDD":>8} '
          f'{"Δ年化":>9} {"95%CI":>18} {"P(>基线)":>8}')
    for tag, mm, d, ci, p in rows:
        cis = f'[{ci[0]*100:+.2f},{ci[1]*100:+.2f}]' if ci else '—'
        ps = f'{p*100:.1f}%' if p is not None else '—'
        ds = f'{d*100:+.2f}pp' if ci else '—'
        print(f'{LABEL.get(tag,tag):<26} {mm["nav"]:>13,.0f} {mm["tot"]*100:>8.2f}% '
              f'{mm["ann"]*100:>7.2f}% {mm["sharpe"]:>8.4f} {mm["mdd"]*100:>7.2f}% '
              f'{ds:>9} {cis:>18} {ps:>8}')

    # ---- 逐年分解 ----
    print('\n--- 逐年 Δ年化(pp) vs ident ---')
    yrs = sorted(set(mb_idx.year))
    print(f'{"臂":<26} ' + ' '.join(f'{y:>8}' for y in yrs))
    for tag, mm, d, ci, p in rows:
        if tag == 'ident':
            continue
        ma = monthly(load(tag))
        line = []
        for y in yrs:
            ia = ma[ma.index.year == y]
            ib = mb[mb.index.year == y]
            k = ia.index.intersection(ib.index)
            if len(k) == 0:
                line.append(f'{"—":>8}'); continue
            ra = (1 + ia.loc[k]).prod(); rb = (1 + ib.loc[k]).prod()
            line.append(f'{(ra-rb)*100:>+7.2f}')
        print(f'{LABEL.get(tag,tag):<26} ' + ' '.join(line))

    # ---- 尾载: 剔最大N月 (配对, 同月份两臂同剔) ----
    print('\n--- 尾载检验(配对剔月, Δ年化pp) ---')
    print('  A=按基线自身最大月剔;  B=按各臂自身最大月剔;  校准列=同族已采纳臂 ---')
    def tail_table(tags, labels):
        print(f'{"臂":<26} {"剔3月(A)":>10} {"剔5月(A)":>10} {"剔5月(B)":>10} {"剔10月(A)":>11}')
        for tag in tags:
            df = load(tag)
            if df is None:
                continue
            ma = monthly(df)
            idx = ma.index.intersection(mb_idx)
            top_b = mb.loc[idx].nlargest(10).index
            top_a = ma.loc[idx].nlargest(10).index
            v3 = drop_months(ma, mb, top_b[:3], idx)
            v5 = drop_months(ma, mb, top_b[:5], idx)
            w5 = drop_months(ma, mb, top_a[:5], idx)
            v10 = drop_months(ma, mb, top_b[:10], idx)
            nm = LABEL.get(tag, labels.get(tag, tag))
            print(f'{nm:<26} {v3*100:>+9.2f} {v5*100:>+9.2f} {w5*100:>+9.2f} {v10*100:>+10.2f}')
    tail_table([t for t, *_ in rows if t != 'ident'], LABEL)
    print('  --- 校准: 同族已采纳臂 ---')
    tail_table([t for t in CALIB if os.path.exists(f'{RVD}/equity_curve.csv.{t}')], {})

    # ---- 换手/笔级画像 ----
    print('\n--- 笔级画像(trade_realized.csv) ---')
    print(f'{"臂":<26} {"笔数":>6} {"均收益":>8} {"胜率":>7} {"中位持有":>8} {"Σret":>9}')
    for tag, *_ in rows:
        t = trades(tag)
        if t is None:
            continue
        cols = {c.lower(): c for c in t.columns}
        pnl = None
        for cand in ('pnl', 'profit', 'realized_pnl', 'ret'):
            if cand in cols:
                pnl = cols[cand]; break
        if pnl is None:
            print(f'{LABEL.get(tag,tag):<26} (列: {list(t.columns)[:8]})'); continue
        v = pd.to_numeric(t[pnl], errors='coerce').dropna()
        hd = None
        for cand in ('hold_days', 'days', 'holding_days'):
            if cand in cols:
                hd = pd.to_numeric(t[cols[cand]], errors='coerce').dropna(); break
        hds = f'{hd.median():.0f}' if hd is not None and len(hd) else '—'
        wr = f'{(v > 0).mean()*100:.1f}%'
        print(f'{LABEL.get(tag,tag):<26} {len(v):>6} {v.mean()*100:>7.2f}% {wr:>7} {hds:>8} {v.sum():>9.2f}')

    # ---- 选股重叠(彩票探测器) ----
    print('\n--- 选股重叠 vs ident (Jaccard, 按调仓日) ---')
    try:
        sb = pd.read_csv(f'{RVD}/portfolio_selections.csv.{BASE_TAG}', parse_dates=['date'])
        gb = {d: set(g['code']) for d, g in sb.groupby('date')}
    except Exception as e:
        print(f'  跳过: {e}'); gb = None
    if gb:
        print(f'{"臂":<26} {"共同日":>7} {"Jaccard中位":>11} {"Jaccard均值":>11} {"完全同批%":>10} {"新入场笔数":>10}')
        for tag, *_ in rows:
            if tag == 'ident':
                continue
            try:
                sa = pd.read_csv(f'{RVD}/portfolio_selections.csv.{tag}', parse_dates=['date'])
            except Exception:
                continue
            ga = {d: set(g['code']) for d, g in sa.groupby('date')}
            ks = sorted(set(ga) & set(gb))
            js = []
            same = 0
            for d in ks:
                A, B = ga[d], gb[d]
                u = len(A | B)
                j = len(A & B) / u if u else 1.0
                js.append(j)
                if A == B:
                    same += 1
            if not js:
                continue
            print(f'{LABEL.get(tag,tag):<26} {len(ks):>7} {np.median(js):>11.3f} {np.mean(js):>11.3f} '
                  f'{same/len(ks)*100:>9.1f}% {len(sa):>10}')

    # ---- 诊断行(来自 bt log) ----
    print('\n--- 诊断(bt log) ---')
    for tag, *_ in rows:
        lp = f'/mnt/d/quant/.arms/logs/bt_{tag}.log'
        if not os.path.exists(lp):
            continue
        out = []
        for line in open(lp, errors='ignore'):
            if any(k in line for k in ('A1赢家修剪', '行业上限(', '最小持仓锁(', 'E-A2')):
                out.append(line.strip())
        if out:
            print(f'  [{tag}] ' + ' | '.join(out[:4]))


if __name__ == '__main__':
    main()
