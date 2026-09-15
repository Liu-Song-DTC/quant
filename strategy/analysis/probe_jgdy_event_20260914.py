"""阶段8a: 机构调研事件研究探针 (2026-09-14夜)

数据源: 东财 数据中心-机构调研明细 (EM datacenter API 直连, 见 fetch_jgdy_direct.py;
akshare.stock_jgdy_detail_em 有整表分页陷阱已弃用)
原始CSV: rolling_validation_results/jgdy_raw_2021_2026.csv (fetch_jgdy_direct.py 产出,
RPT_ORG_SURVEYNEW columns=ALL; 事件分析本地裁剪 NOTICE_DATE<=2024-12-31)

PIT: 调研公告盘后发布 → 事件日=公告日, 收益从公告日收盘起算 (与stage5a回购探针同约定)

测度:
  1. 事件研究: fwd5/10/20/60 原始 + 指数调整(sh000001同期) 超额, 分桶:
     接待机构数量(NUMBERNEW) / 调研→公告滞后天数 / 逐年稳定性
  2. 月度截面IC: 近60日机构调研强度(接待机构数合计) vs fwd10/20市场调整收益
     判据对标: bp2类 realized mean5 +3.02%/hit1 81.8%; 增持事件 +1.56%/20d(已否决)

产出: rolling_validation_results/jgdy_event_excess.pkl + jgdy_monthly_ic.csv
"""
import os
import sys
import numpy as np
import pandas as pd
from functools import lru_cache

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BT = '/mnt/d/quant/data/stock_data/backtrader_data'
OUT = '/mnt/d/quant/strategy/rolling_validation_results'
os.makedirs(OUT, exist_ok=True)
CSV_TMP = os.path.join(OUT, 'jgdy_raw_2021_2026.csv')

_idx = pd.read_csv(f'{BT}/sh000001_qfq.csv', parse_dates=['datetime'])
_idx = _idx.set_index('datetime')['close']


@lru_cache(maxsize=1200)
def _prices(code):
    p = f'{BT}/{code}_qfq.csv'
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p, parse_dates=['datetime'])
    s = df.set_index('datetime')['close']
    s.index = pd.to_datetime(s.index)
    return s


def fwd_series(code, t0, horizons=(5, 10, 20, 60)):
    """t0=事件日(公告日)收盘为基准, 返回各horizon的raw+指数调整收益."""
    s = _prices(code)
    if s is None:
        return None
    if t0 not in s.index:
        pos = s.index.searchsorted(t0)
        if pos == 0:
            return None
        t0 = s.index[pos - 1]  # 取最近交易日
    i0 = s.index.get_loc(t0)
    out = {}
    for h in horizons:
        if i0 + h >= len(s):
            out[f'r{h}'] = np.nan
            out[f'x{h}'] = np.nan
            continue
        r = s.iloc[i0 + h] / s.iloc[i0] - 1
        x = r - (_idx.loc[s.index[i0 + h]] / _idx.loc[t0] - 1)
        out[f'r{h}'] = r
        out[f'x{h}'] = x
    return out


def rep(name, ev):
    print(f'\n--- {name} (n={len(ev)}) ---', flush=True)
    for h in (5, 10, 20, 60):
        v = ev[f'x{h}'].dropna()
        print(f'  fwd{h:>2d}a 均值{100*v.mean():+.2f}% 中位{100*v.median():+.2f}% '
              f'胜率{100*(v > 0).mean():.0f}%', flush=True)


def main():
    if not os.path.exists(CSV_TMP):
        print(f'缺原始CSV {CSV_TMP}, 先跑 fetch_jgdy_direct.py', flush=True)
        sys.exit(1)
    start = sys.argv[1] if len(sys.argv) > 1 else '2021-01-01'
    end = sys.argv[2] if len(sys.argv) > 2 else '2024-12-31'
    print(f'事件窗口: {start} ~ {end}', flush=True)
    raw = pd.read_csv(CSV_TMP)
    print(f'原始 {len(raw)} 行', flush=True)

    # 直连API列(columns=ALL): SECURITY_CODE/NOTICE_DATE/RECEIVE_START_DATE/RECEIVE_OBJECT/
    #   NUMBERNEW/SUM... 行=单一机构接待记录, SUM=该事件接待机构总数(同行内一致)
    raw['code'] = raw['SECURITY_CODE'].astype(str).str.zfill(6)
    raw['ann_date'] = pd.to_datetime(raw['NOTICE_DATE'])
    raw = raw[(raw['ann_date'] >= start) & (raw['ann_date'] <= end)]
    raw = raw[~raw['code'].str.startswith(('4', '8', '92'))]  # 北交所排除
    raw = raw.drop_duplicates(subset=['code', 'ann_date', 'RECEIVE_OBJECT'])
    if 'RECEIVE_START_DATE' in raw.columns:
        raw['visit_date'] = pd.to_datetime(raw['RECEIVE_START_DATE'], errors='coerce')
        raw['lag_d'] = (raw['ann_date'] - raw['visit_date']).dt.days
    else:
        raw['lag_d'] = np.nan
    raw['sum_n'] = pd.to_numeric(raw.get('SUM'), errors='coerce')
    raw['nnew_n'] = pd.to_numeric(raw.get('NUMBERNEW'), errors='coerce')
    print(f'清洗后: {len(raw)} 行, {raw.code.nunique()} 只', flush=True)
    print('SUM非空率', round(raw['sum_n'].notna().mean(), 3),
          '| SUM样本', raw['sum_n'].dropna().head(5).tolist(), flush=True)

    # 同(code,公告日)合并: 总接待机构数 = SUM(每行同值取max), 行数=机构明细条数
    g = raw.groupby(['code', 'ann_date'], as_index=False).agg(
        inst_tot=('sum_n', 'max'), visits=('sum_n', 'count'),
        lag_min=('lag_d', 'min'), lag_max=('lag_d', 'max'))
    # SUM缺失的事件退回行计数
    g['inst_tot'] = g['inst_tot'].fillna(g['visits']).clip(lower=1).astype(int)
    g = g.sort_values('ann_date')
    print(f'事件数(按公告日合并): {len(g)}', flush=True)

    print('\n[事件研究] 公告日收盘为基准...', flush=True)
    rows = []
    for code, nd in zip(g['code'], g['ann_date']):
        f = fwd_series(code, nd)
        if f is None or np.isnan(f.get('x20', np.nan)):
            continue
        rows.append((code, nd, f['r5'], f['r10'], f['r20'], f['r60'],
                     f['x5'], f['x10'], f['x20'], f['x60']))
    ev = pd.DataFrame(rows, columns=['code', 'date', 'r5', 'r10', 'r20', 'r60',
                                     'x5', 'x10', 'x20', 'x60'])
    ev = ev.merge(g.rename(columns={'ann_date': 'date'}), on=['code', 'date'])
    ev.to_pickle(os.path.join(OUT, 'jgdy_event_excess.pkl'))
    print(f'事件收益计算完成: {len(ev)}', flush=True)

    rep('全部', ev)
    ev['b'] = pd.cut(ev['inst_tot'], [-np.inf, 5, 15, 40, np.inf],
                     labels=['1-5家', '6-15家', '16-40家', '>40家'])
    for b in ['1-5家', '6-15家', '16-40家', '>40家']:
        rep(f'机构数={b}', ev[ev.b == b])
    ev['lb'] = pd.cut(ev['lag_min'], [-np.inf, 3, 8, np.inf],
                      labels=['≤3天', '4-8天', '>8天'])
    for b in ['≤3天', '4-8天', '>8天']:
        rep(f'滞后={b}', ev[ev.lb == b])

    print('\n逐年 fwd20a:')
    for y in sorted(ev.date.dt.year.unique()):
        sub = ev[ev.date.dt.year == y]
        v = sub['x20'].dropna()
        print(f'  {y}: n={len(sub):5d} fwd20a={100*v.mean():+.2f}% 胜率={100*(v > 0).mean():.0f}%',
              flush=True)
    print('\n逐年 >40家桶 fwd20a:')
    for y in sorted(ev.date.dt.year.unique()):
        sub = ev[(ev.date.dt.year == y) & (ev.b == '>40家')]
        v = sub['x20'].dropna()
        if len(v):
            print(f'  {y}: n={len(v):4d} fwd20a={100*v.mean():+.2f}% '
                  f'胜率={100*(v > 0).mean():.0f}%', flush=True)

    # 月度截面IC: 近60日调研强度
    print('\n[月度IC] 近60日接待机构数 vs fwd10/20市场调整...', flush=True)
    months = pd.date_range(ev.date.min().strftime('%Y-%m-01'), ev.date.max(), freq='MS')
    ic_rows = []
    for m in months:
        m_end = m + pd.DateOffset(months=1) - pd.DateOffset(days=1)
        if m_end not in _idx.index:
            continue
        w = g[(g['ann_date'] >= m_end - pd.Timedelta(days=60)) & (g['ann_date'] <= m_end)]
        if len(w) < 30:
            continue
        strength = w.groupby('code')['inst_tot'].sum()
        st = []
        for code, sval in strength.items():
            f = fwd_series(code, m_end)
            if f is None or np.isnan(f.get('x20', np.nan)):
                continue
            st.append((code, sval, f['x10'], f['x20']))
        if len(st) < 30:
            continue
        df2 = pd.DataFrame(st, columns=['code', 's60', 'x10', 'x20'])
        from scipy import stats as _st
        ic10 = _st.spearmanr(df2['s60'], df2['x10'])[0]
        ic20 = _st.spearmanr(df2['s60'], df2['x20'])[0]
        ic_rows.append((m.strftime('%Y-%m'), len(df2), ic10, ic20))
        print(f'  {m:%Y-%m}: n={len(df2):4d} IC10={ic10:+.3f} IC20={ic20:+.3f}', flush=True)
    icdf = pd.DataFrame(ic_rows, columns=['month', 'n', 'ic10', 'ic20'])
    icdf.to_csv(os.path.join(OUT, 'jgdy_monthly_ic.csv'), index=False)
    print(f'\nIC汇总: IC10 mean={icdf.ic10.mean():+.3f} ir={icdf.ic10.mean()/icdf.ic10.std():+.2f} | '
          f'IC20 mean={icdf.ic20.mean():+.3f} ir={icdf.ic20.mean()/icdf.ic20.std():+.2f}', flush=True)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
