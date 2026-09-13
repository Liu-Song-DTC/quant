#!/usr/bin/env python3
"""probe_pit_materiality_vector_0913.py — P0-1 PIT审计 · 第三刀: 泄漏实质影响(向量化, 生产语义忠实复刻)

背景: 第一刀已证 数据可用日期≡报告期(100%) → 回测窗口58.6%基本面可见股票-日
用了未公开财报(法定截止保守代理)。第二刀P1公告日列不可靠(最新公告日期=快照泄漏/
zcfz下期串行) → 毒化影子废弃。本刀用**法定截止影子**(fundamental_data_deadline/,
与第一刀同口径, 紧上界: 真实公告早于截止 → 本刀变化率为真实影响的上界)。

三段(全部只读, 生产语义逐位复刻):
  A 信号层 fund_score: FundamentalData getter语义(_get_available_data快路径=
     报告期降序+avail≤D掩码取首) → compute_fundamental_score(roe,pg,rg,eps)
     — 含单元退化bug忠实复刻(str'%'→/100分数, 数值float原样=百分比单位)
  B 排雷 _check_profit_decline: 数据可用日期升序 tail(3), pg单位一致百分比
  C factor_df 6个fund_列: factor_preparer._preload_fundamental_cache指针语义
     (数据可用日期_str升序推进) + compress_fundamental_factor — 先identity对账
     生产parquet(syst), 再算honest, 再逐日截面Spearman秩相关(IC消费的是秩)

执行: cd strategy && /mnt/d/quant/.venv/bin/python
  analysis/probe_pit_materiality_vector_0913.py > logs/probe_pit_materiality_vector_0913.log 2>&1
"""
import os
import time
import resource

import numpy as np
import pandas as pd

ROOT = '/mnt/d/quant'
SRC_DIR = os.path.join(ROOT, 'data', 'stock_data', 'fundamental_data')
HON_DIR = os.path.join(ROOT, 'data', 'stock_data', 'fundamental_data_deadline')
PARQUET = os.path.join(ROOT, 'strategy', 'cache',
                       'factor_df_2718s_809d_d814a206.parquet')

FUND_COLS = ['fund_score', 'fund_roe', 'fund_profit_growth',
             'fund_revenue_growth', 'fund_gross_margin', 'fund_cf_to_profit']


def to_obj(v) -> np.ndarray:
    """pandas3 pyarrow字符串列 → numpy object数组 (支持[np索引])"""
    return np.asarray(pd.Series(v).astype(str), dtype=object)


# ---------- 生产语义复刻 ----------

def parse_pct_col(col: pd.Series) -> np.ndarray:
    """_parse_pct/getter语义: str带% → /100分数; 数值 → float原样(百分比单位)"""
    s = col.astype(str)
    out = np.full(len(col), np.nan)
    has_pct = s.str.contains('%', na=False)
    if has_pct.any():
        out[has_pct.values] = pd.to_numeric(
            s[has_pct].str.replace('%', '', regex=False),
            errors='coerce').values / 100.0
    no_pct = ~has_pct.values
    if no_pct.any():
        out[no_pct] = pd.to_numeric(s[no_pct], errors='coerce').values
    return out


def parse_eps_col(col: pd.Series) -> np.ndarray:
    """get_eps+compute_fundamental_score语义: float(原值), 不带%剥离"""
    return pd.to_numeric(col.astype(str), errors='coerce').values


def parse_decline_col(col: pd.Series) -> np.ndarray:
    """_check_profit_decline语义: float(str(v).replace('%','')), 不/100"""
    s = col.astype(str).str.replace('%', '', regex=False)
    return pd.to_numeric(s, errors='coerce').values


def row_signal_score(roe, pg, rg, eps):
    """compute_fundamental_score(roe,pg,rg,eps) 向量化 (NaN→该族+0, 与float(NaN)比较False等价)"""
    pts = np.zeros(len(roe))
    pts += np.where(roe > 0.15, 0.35, np.where(roe > 0.10, 0.25,
                                               np.where(roe > 0.05, 0.15, 0.0)))
    pts += np.where(pg > 0.50, 0.30, np.where(pg > 0.20, 0.20,
                                              np.where(pg > 0.0, 0.10, 0.0)))
    pts += np.where(rg > 0.30, 0.20, np.where(rg > 0.15, 0.12,
                                              np.where(rg > 0.0, 0.05, 0.0)))
    pts += np.where(eps > 1.0, 0.20, np.where(eps > 0.5, 0.12, 0.0))
    return np.minimum(pts, 1.0)


def row_preload_columns(df):
    """factor_preparer._preload_fundamental_cache 行级计算, 返回压缩后6列 (None→0.0)"""
    roe = parse_pct_col(df['净资产收益率'])
    pg = parse_pct_col(df['净利润-同比增长'])
    rg = parse_pct_col(df['营业总收入-同比增长'])
    gm = parse_pct_col(df['销售毛利率'])
    eps = parse_eps_col(df['每股收益'])

    score = np.zeros(len(df))
    score += np.where(np.isnan(roe), 0.0, np.minimum(roe * 100, 30.0))
    score += np.where(pg > 0.5, 25.0, np.where(pg > 0.2, 15.0,
                                               np.where(pg > 0.0, 5.0, 0.0)))
    score += np.where(eps > 0.0, np.minimum(eps * 10.0, 20.0), 0.0)
    score += np.where(rg > 0.3, 15.0, np.where(rg > 0.1, 10.0, 0.0))

    oc = pd.to_numeric(df['xjll_经营性现金流-现金流量净额'].astype(str),
                       errors='coerce').values
    pr = pd.to_numeric(df['lrb_净利润'].astype(str), errors='coerce').values
    cf = np.full(len(df), np.nan)
    m = (~np.isnan(oc)) & (~np.isnan(pr)) & (pr > 0)
    cf[m] = oc[m] / pr[m]

    def _nan0(v):
        return np.where(np.isnan(v), 0.0, v)

    out = {}
    out['fund_score'] = np.tanh((np.clip(score, 0.0, 1.0) - 0.5) * 3)
    out['fund_roe'] = np.tanh((np.clip(_nan0(roe), -50, 50) - 10) / 20)
    out['fund_profit_growth'] = np.tanh(np.clip(_nan0(pg), -100, 100))
    out['fund_revenue_growth'] = np.tanh(np.clip(_nan0(rg), -100, 100))
    out['fund_gross_margin'] = np.tanh((np.clip(_nan0(gm), -20, 80) - 30) / 30)
    out['fund_cf_to_profit'] = np.tanh(np.clip(_nan0(cf), -5, 5) - 1)
    return out, roe, pg, rg, eps


def main():
    t0 = time.time()
    print('=' * 70, flush=True)
    print('P0-1 PIT审计第三刀: 泄漏实质影响 (法定截止影子, 向量化)', flush=True)

    fdf = pd.read_parquet(PARQUET, columns=['code', 'date'])
    bad = fdf['code'].astype(str).str.startswith(('8', '43', '92', '399'))
    fdf = fdf[~bad]
    codes = sorted(fdf['code'].astype(str).str.zfill(6).unique())
    dates = pd.DatetimeIndex(sorted(fdf['date'].unique()))
    dvals = dates.values.astype('datetime64[D]')
    dstr = to_obj(dates.strftime('%Y%m%d'))
    years = dates.year.to_numpy()
    w = dates >= pd.Timestamp('2021-01-04')
    print(f'[0] 面板 {len(codes)}只 × {len(dates)}日, 2021+窗口 {w.sum()}日', flush=True)

    # ---------- A/B: 信号层 ----------
    a_chg = np.zeros(len(dates), dtype=np.int64)
    a_vis = np.zeros(len(dates), dtype=np.int64)
    a_flip = np.zeros(len(dates), dtype=np.int64)
    a_blind = np.zeros(len(dates), dtype=np.int64)  # sys可见hon盲
    a_delta = []
    b_sys_true = np.zeros(len(dates), dtype=np.int64)
    b_hon_true = np.zeros(len(dates), dtype=np.int64)
    b_vis = np.zeros(len(dates), dtype=np.int64)
    b_flip = np.zeros(len(dates), dtype=np.int64)

    n_files_used = 0
    for i, code in enumerate(codes):
        hp = os.path.join(HON_DIR, code + '.csv')
        if not os.path.exists(hp):
            continue
        n_files_used += 1
        df = pd.read_csv(hp, dtype={'报告期': str})
        df['数据可用日期'] = df['数据可用日期'].astype(str)
        r = pd.to_datetime(df['报告期'].astype(str).str.split('.').str[0],
                           format='%Y%m%d', errors='coerce')
        ok = r.notna()
        if ok.sum() == 0:
            continue
        df = df[ok].reset_index(drop=True)
        r = r[ok].values.astype('datetime64[D]')
        avail_hon = to_obj(df['数据可用日期'])  # str object数组

        # A: sys=报告期(单调) → searchsorted; hon → mask+max(报告期升序最后可见位)
        sys_idx = np.searchsorted(r, dvals, side='right') - 1
        M = avail_hon[None, :] <= dstr[:, None]
        idx = np.where(M, np.arange(len(r))[None, :], -1)
        hon_idx = np.where(M.any(axis=1), idx.max(axis=1), -1)
        vis = (sys_idx >= 0) | (hon_idx >= 0)
        # 值(逐行, 与getter语义一致)
        roe = parse_pct_col(df['净资产收益率'])
        pg = parse_pct_col(df['净利润-同比增长'])
        rg = parse_pct_col(df['营业总收入-同比增长'])
        eps = parse_eps_col(df['每股收益'])
        sc = row_signal_score(roe, pg, rg, eps)
        vis_sys = sys_idx >= 0
        vis_hon = hon_idx >= 0
        # 不可见→getter None→score 0 (不能clip到首行)
        sc_sys = np.where(vis_sys, sc[np.clip(sys_idx, 0, None)], 0.0)
        sc_hon = np.where(vis_hon, sc[np.clip(hon_idx, 0, None)], 0.0)
        delta = np.abs(sc_sys - sc_hon)
        a_vis += vis
        a_chg += vis & (delta > 1e-9)
        a_flip += vis & ((sc_sys > 0) != (sc_hon > 0))
        a_blind += vis_sys & (~vis_hon)
        a_delta.append(delta[vis])

        # B: decline tail(3) — 数据可用日期_str升序(稳定), m=前缀计数, 末3位
        a_sys = to_obj(pd.DatetimeIndex(r).strftime('%Y%m%d'))  # sys可用日=报告期8位串
        order_sys = np.argsort(a_sys, kind='stable')
        m_sys = np.searchsorted(a_sys[order_sys], dstr, side='right')
        order_hon = np.argsort(avail_hon, kind='stable')
        m_hon = np.searchsorted(avail_hon[order_hon], dstr, side='right')
        pg_d = parse_decline_col(df['净利润-同比增长'])
        rg_d = parse_decline_col(df['营业总收入-同比增长'])

        def _decline_flag(m_, order_):
            ok3 = m_ >= 3
            i3 = np.clip(m_[:, None] - np.array([3, 2, 1])[None, :], 0, None)
            v = np.take(pg_d, order_[i3])
            u = np.take(rg_d, order_[i3])
            real3 = np.all(~np.isnan(v), axis=1)
            neg3 = np.all(v < 0, axis=1)
            trend = v[:, -1] - v[:, 0]
            rg_ok = np.all(~np.isnan(u), axis=1) & np.all(u < 0, axis=1)
            return ok3 & real3 & neg3 & ((trend < -5) | ((trend > 5) & rg_ok))

        flag_sys = _decline_flag(m_sys, order_sys)
        flag_hon = _decline_flag(m_hon, order_hon)
        b_sys_true += flag_sys
        b_hon_true += flag_hon
        b_vis += vis
        b_flip += vis & (flag_sys != flag_hon)

        if (i + 1) % 800 == 0:
            print(f'  [A/B] {i+1}/{len(codes)} (elapsed {time.time()-t0:.0f}s)',
                  flush=True)

    print(f'[A] 信号层 fund_score (2021+, 法定截止上界):', flush=True)
    vis_tot = a_vis[w].sum()
    print(f'    可见股票-日 {vis_tot:,} | 评分变化 {a_chg[w].sum():,} = '
          f'{100*a_chg[w].sum()/vis_tot:.2f}% | 开关翻转 {a_flip[w].sum():,} = '
          f'{100*a_flip[w].sum()/vis_tot:.2f}% | sys见hon盲 {a_blind[w].sum():,} = '
          f'{100*a_blind[w].sum()/vis_tot:.2f}%', flush=True)
    dd = np.concatenate([a for a in a_delta if len(a)])
    print(f'    |Δ|分布(变化日): 中位 {np.median(dd):.3f}, P75 {np.percentile(dd,75):.3f}, '
          f'P90 {np.percentile(dd,90):.3f}, max {dd.max():.3f}', flush=True)
    print(f'    逐年评分变化%: ' + ' | '.join(
        f'{y}: {100*a_chg[(years==y)&w].sum()/max(a_vis[(years==y)&w].sum(),1):.1f}%'
        for y in sorted(set(years[w]))), flush=True)
    print(f'    逐年开关翻转%: ' + ' | '.join(
        f'{y}: {100*a_flip[(years==y)&w].sum()/max(a_vis[(years==y)&w].sum(),1):.1f}%'
        for y in sorted(set(years[w]))), flush=True)

    print(f'[B] 排雷 decline (2021+): 可见 {b_vis[w].sum():,} | sys True '
          f'{b_sys_true[w].sum():,} | hon True {b_hon_true[w].sum():,} | 翻转 '
          f'{b_flip[w].sum():,} = {100*b_flip[w].sum()/b_vis[w].sum():.2f}%', flush=True)

    # ---------- C: factor_df 6列 ----------
    print(f'\n[C] factor_df fund列 (指针语义):', flush=True)
    pf = pd.read_parquet(PARQUET, columns=['code', 'date'] + FUND_COLS)
    pf['code'] = pf['code'].astype(str).str.zfill(6)
    pf = pf[~pf['code'].str.startswith(('8', '43', '92', '399'))]
    pf['dstr'] = pd.to_datetime(pf['date']).dt.strftime('%Y%m%d')
    print(f'    parquet行 {len(pf):,} × {len(FUND_COLS)}列, dtype: '
          f'{str(pf[FUND_COLS].dtypes.unique())[:80]}', flush=True)

    # identity: 抽100只, 复刻syst vs parquet逐位对账
    rng = np.random.default_rng(42)
    id_codes = sorted(rng.choice(pf['code'].unique(), size=100, replace=False))
    pf_by_code = {c: g.sort_values('dstr') for c, g in pf.groupby('code')}
    id_max = 0.0
    id_n = 0
    id_rows = 0
    for code in id_codes:
        sp = os.path.join(SRC_DIR, code + '.csv')
        if not os.path.exists(sp):
            continue
        df = pd.read_csv(sp, dtype={'报告期': str})
        if '数据可用日期' not in df.columns or '报告期' not in df.columns:
            continue
        df['数据可用日期'] = df['数据可用日期'].astype(str)
        r = pd.to_datetime(df['报告期'].astype(str).str.split('.').str[0],
                           format='%Y%m%d', errors='coerce')
        df = df[r.notna()].reset_index(drop=True)
        sys_avail = to_obj(r[r.notna()].dt.strftime('%Y%m%d'))
        sub = pf_by_code.get(code)
        if sub is None or len(sub) == 0:
            continue
        cols, _, _, _, _ = row_preload_columns(df)
        # 指针推进 (syst = 源目录可用日)
        order = np.argsort(sys_avail, kind='stable')
        av_sorted = sys_avail[order]
        ptr = np.searchsorted(av_sorted, sub['dstr'].values, side='right') - 1
        for c in FUND_COLS:
            mine = cols[c][order[np.clip(ptr, 0, None)]]
            theirs = sub[c].values
            m2 = ptr >= 0
            if m2.sum() == 0:
                continue
            d = np.abs(mine[m2] - theirs[m2])
            id_max = max(id_max, np.nanmax(d) if not np.isnan(d).all() else 0.0)
            id_n += int(np.nanmax(d) > 1e-6)
            id_rows += m2.sum()
    print(f'    identity(100只×{id_rows:,}行): max|Δ|={id_max:.3e}, '
          f'超1e-6行组数 {id_n}', flush=True)

    # 全量: sys vs hon 六列 + 逐日Spearman
    n_chg = {c: 0 for c in FUND_COLS}
    n_vis = 0
    per_date = {c: {} for c in FUND_COLS}  # date -> (sys, hon) list
    for i, code in enumerate(codes):
        hp = os.path.join(HON_DIR, code + '.csv')
        if not os.path.exists(hp):
            continue
        df = pd.read_csv(hp, dtype={'报告期': str})
        df['数据可用日期'] = df['数据可用日期'].astype(str)
        r = pd.to_datetime(df['报告期'].astype(str).str.split('.').str[0],
                           format='%Y%m%d', errors='coerce')
        df = df[r.notna()].reset_index(drop=True)
        sys_avail = to_obj(r[r.notna()].dt.strftime('%Y%m%d'))
        hon_avail = to_obj(df['数据可用日期'])
        sub = pf_by_code.get(code)
        if sub is None or len(sub) == 0:
            continue
        cols, _, _, _, _ = row_preload_columns(df)
        order_s = np.argsort(sys_avail, kind='stable')
        order_h = np.argsort(hon_avail, kind='stable')
        ptr_s = np.searchsorted(sys_avail[order_s], sub['dstr'].values,
                                side='right') - 1
        ptr_h = np.searchsorted(hon_avail[order_h], sub['dstr'].values,
                                side='right') - 1
        for c in FUND_COLS:
            v_s = cols[c][order_s[np.clip(ptr_s, 0, None)]]
            v_h = cols[c][order_h[np.clip(ptr_h, 0, None)]]
            chg = np.abs(v_s - v_h) > 1e-6
            n_chg[c] += int(chg.sum())
            n_vis += 0 if c != 'fund_score' else int((ptr_s >= 0).sum())
            if c == 'fund_score':
                for j, dte in enumerate(sub['date'].values):
                    if ptr_s[j] >= 0 and ptr_h[j] >= 0:
                        per_date[c].setdefault(pd.Timestamp(dte).strftime('%Y-%m-%d'),
                                               ([], []) )
                        per_date[c][pd.Timestamp(dte).strftime('%Y-%m-%d')][0].append(v_s[j])
                        per_date[c][pd.Timestamp(dte).strftime('%Y-%m-%d')][1].append(v_h[j])
        if (i + 1) % 800 == 0:
            print(f'  [C] {i+1}/{len(codes)} (elapsed {time.time()-t0:.0f}s)',
                  flush=True)

    print(f'    fund_score 可见行 {n_vis:,}, 变化 {n_chg["fund_score"]:,} = '
          f'{100*n_chg["fund_score"]/max(n_vis,1):.2f}%', flush=True)
    for c in FUND_COLS[1:]:
        print(f'    {c}: 变化 {n_chg[c]:,}', flush=True)

    from scipy.stats import rankdata
    rho = []
    for dte, (a, b) in per_date['fund_score'].items():
        if len(a) < 30:
            continue
        ra, rb = rankdata(a), rankdata(b)
        rho.append((dte, float(np.corrcoef(ra, rb)[0, 1])))
    if rho:
        rh = pd.Series({d: v for d, v in rho}).sort_index()
        rh.index = pd.DatetimeIndex(rh.index)
        rh_y = rh.groupby(rh.index.year).median()
        print(f'    fund_score 逐日截面Spearman(sys vs hon): 全体中位 '
              f'{rh.median():.4f}', flush=True)
        print('    逐年中位: ' + ' | '.join(
            f'{y}: {v:.4f}' for y, v in rh_y.items()), flush=True)

    print(f'\n总耗时 {time.time()-t0:.0f}s, rss='
          f'{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6:.1f}GB', flush=True)


if __name__ == '__main__':
    main()
