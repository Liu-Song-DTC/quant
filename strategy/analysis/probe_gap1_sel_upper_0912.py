#!/usr/bin/env python3
"""2026-09-12 差距1上界探针: 组合层贪婪构造 vs 前视最优 — 选股日层面的损失量化

问题: 阶段6已证信号层alpha真而大(+3.57pp残差), 组合层贪婪构造是转化瓶颈。
本探针在"选股日×候选池"层面量化四件事:
  A. 贪婪实际选择的 fwd20 (portfolio_selections.csv 各选股日持仓)
  B. 候选池整体质量 (当日buy信号池的fwd20中位)
  C. 前视上界: 同池top-N by fwd20 (完美预知, max_per_industry=0无行业约束,
     唯一结构性约束=槽数) — 组合层理论上能拿到的天花板
  D. 机制对照: 同池top-N by score / adjusted_score / ml_score —
     纯分数排序能拿多少(量化chan调整/产业链/换手保护等启发式的净贡献)
输出: 整体+逐年 + 池内score↔fwd20秩相关(池内IC, 决定"分数排序"路线的上限)。

口径(与阶段6一致): fwd20=close-to-close 20交易日; 信号日=选股日当天buy=True;
北交所(4/8/92)排除; 停牌ffill; 选股日<=2026-08-13(保证20日前视期)。
只读。串行。.venv。执行: cd strategy && /mnt/d/quant/.venv/bin/python
  analysis/probe_gap1_sel_upper_0912.py > logs/probe_gap1_sel_upper_0912.log 2>&1
"""
import os
import time
import numpy as np
import pandas as pd

BT = '/mnt/d/quant/data/stock_data/backtrader_data'
SIG = '/mnt/d/quant/strategy/rolling_validation_results/backtest_signals.csv'
PS = '/mnt/d/quant/strategy/rolling_validation_results/portfolio_selections.csv'
IDX = os.path.join(BT, 'sh000001_qfq.csv')
FWD = 20
END_OK = '2026-08-13'  # fwd20需完整前视期


def build_close():
    idx = pd.read_csv(IDX, usecols=['datetime'], parse_dates=['datetime'])
    idx = idx[(idx.datetime >= '2020-12-01') & (idx.datetime <= '2026-10-15')]
    D = idx['datetime'].values.astype('datetime64[ns]')
    T = len(D)
    codes = []
    for fn in sorted(os.listdir(BT)):
        if not fn.endswith('_qfq.csv'):
            continue
        c = fn[:-len('_qfq.csv')]
        if c.startswith(('sh', 'sz')) or c.startswith(('4', '8', '92')):
            continue
        codes.append(c)
    N = len(codes)
    colmap = {c: i for i, c in enumerate(codes)}
    print(f'[0] 日期 {T} 天 x 股票 {N} 只', flush=True)
    close = np.full((T, N), np.nan, dtype=np.float32)
    t0 = time.time()
    for i, c in enumerate(codes):
        try:
            df = pd.read_csv(os.path.join(BT, f'{c}_qfq.csv'),
                             usecols=['datetime', 'close'])
        except Exception:
            continue
        dt = df['datetime'].values.astype('datetime64[ns]')
        pos = np.searchsorted(D, dt)
        pos = pos[pos < T]
        if len(pos) == 0:
            continue
        close[pos, i] = df['close'].values[:len(pos)].astype(np.float32)
        if (i + 1) % 1000 == 0:
            print(f'  加载 {i+1}/{N} ({time.time()-t0:.0f}s)', flush=True)
    print(f'[0] 矩阵加载完成 {time.time()-t0:.0f}s, ffill...', flush=True)
    close = pd.DataFrame(close).ffill(axis=0).values
    return D, codes, colmap, close


def load_pool_and_picks():
    sig = pd.read_csv(SIG, usecols=['code', 'date', 'buy', 'score',
                                    'adjusted_score', 'ml_score'],
                      dtype={'code': str})
    sig = sig[sig.buy == True].copy()
    sig['code'] = sig['code'].str.zfill(6)
    sig = sig[~sig['code'].str.startswith(('4', '8', '92'))]
    sig['date'] = pd.to_datetime(sig['date'])
    sig = sig[(sig.date >= '2021-04-01') & (sig.date <= END_OK)]
    print(f'[1] buy池行数: {len(sig)}', flush=True)

    ps = pd.read_csv(PS, dtype={'code': str})
    ps['code'] = ps['code'].str.zfill(6)
    ps['date'] = pd.to_datetime(ps['date'])
    ps = ps[ps.date <= END_OK]
    print(f'[1] 选股行数: {len(ps)}, 唯一选股日: {ps["date"].nunique()}', flush=True)
    return sig, ps


def top_n_by(vals, n, ascending=False):
    """取vals最大的n个索引 (n可能>有效数)"""
    v = np.asarray(vals, dtype=float)
    ok = np.isfinite(v)
    if ok.sum() == 0:
        return np.array([], dtype=int)
    order = np.argsort(-v[ok], kind='stable')
    idx_ok = np.where(ok)[0]
    return idx_ok[order[:min(n, len(order))]]


def main():
    t0 = time.time()
    D, codes, colmap, close = build_close()
    f20 = (pd.DataFrame(close).shift(-FWD) / pd.DataFrame(close) - 1).values
    print(f'[0] fwd20完成 ({time.time()-t0:.0f}s)', flush=True)

    sig, ps = load_pool_and_picks()
    tmap = {d: i for i, d in enumerate(pd.to_datetime(D))}
    sig = sig[sig.date.isin(tmap)]
    sig['ti'] = sig['date'].map(tmap)
    sig = sig[sig.code.isin(colmap)]
    sig['ci'] = sig['code'].map(colmap)
    f20sig = f20[sig['ti'].values, sig['ci'].values]
    sig['f20'] = f20sig
    print(f'[2] 池内f20命中: {np.isfinite(sig.f20).mean()*100:.0f}% '
          f'(停牌前视缺失)', flush=True)

    rows = []
    for d, g in sig.groupby('date'):
        ti = tmap[d]
        pick = ps[ps.date == d]
        if len(pick) == 0:
            continue
        g_ok = g[np.isfinite(g.f20)]
        if len(g_ok) == 0:
            continue
        n_slot = len(pick)
        pick_c = pick.code.map(colmap).values
        pick_c = pick_c[~pd.isna(pick_c)].astype(int)
        pick_f = f20[ti, pick_c]
        pick_f = pick_f[np.isfinite(pick_f)]
        n_pick_f = len(pick_f)

        # 前视上界: 同池top-N by f20 (N=实际持仓数, 无其他约束)
        top = top_n_by(g_ok.f20.values, n_slot)
        top_codes = set(g_ok.code.iloc[top].values)
        actual_codes = set(pick.code.values)
        top_f = g_ok.f20.iloc[top].values
        # 分数排序对照 (纯score / adjusted_score / ml_score)
        s_score = top_n_by(g_ok.score.values, n_slot)
        s_adj = top_n_by(g_ok.adjusted_score.values, n_slot)
        s_ml = top_n_by(g_ok.ml_score.values, n_slot)

        def _mean(idx):
            if len(idx) == 0:
                return np.nan
            return float(np.nanmean(g_ok.f20.iloc[idx].values))

        rows.append({
            'date': d, 'pool': len(g_ok), 'slots': n_slot,
            'pool_med': float(np.nanmedian(g_ok.f20)),
            'actual': float(np.nanmean(pick_f)) if n_pick_f else np.nan,
            'top_f20': float(np.nanmean(top_f)),
            'top_score': _mean(s_score),
            'top_adj': _mean(s_adj),
            'top_ml': _mean(s_ml),
            'overlap_actual_top': len(actual_codes & top_codes) / max(n_slot, 1),
        })
    r = pd.DataFrame(rows)
    r['yr'] = r.date.dt.year

    print(f'\n[3] 选股日汇总 (n={len(r)}, 池中位, 槽位中位 {r.slots.median():.0f})')
    cols = ['pool_med', 'actual', 'top_f20', 'top_score', 'top_adj', 'top_ml']
    print(f'  {"口径":>10s} {"mean":>9s} {"中位":>9s}')
    for c in cols:
        print(f'  {c:>10s} {r[c].mean()*100:>+8.2f}% {r[c].median()*100:>+8.2f}%')
    print(f'\n[4] 关键差距 (按日配对平均, 逐日算差再平均)')
    r['greedy_vs_pool'] = r.actual - r.pool_med
    r['bound_total'] = r.top_f20 - r.actual
    r['machinery'] = r.top_score - r.actual          # 纯score能否超过实际(启发式净贡献)
    r['machinery_adj'] = r.top_adj - r.actual
    r['machinery_ml'] = r.top_ml - r.actual
    r['noise_part'] = r.top_f20 - r.top_score        # 分数解释不了的部分(前视-纯分数)
    for c in ['greedy_vs_pool', 'bound_total', 'machinery', 'machinery_adj',
              'machinery_ml', 'noise_part']:
        print(f'  {c:>15s}: {r[c].mean()*100:+7.2f}pp  '
              f'(中位{r[c].median()*100:+.2f}pp, 正比例{(r[c]>0).mean()*100:.0f}%)')
    print(f'  实际∩前视topN 重叠率: {r.overlap_actual_top.mean()*100:.0f}%')
    print(f'\n[5] 逐年 (mean fwd20)')
    for y in sorted(r.yr.unique()):
        m = r.yr == y
        print(f'  {y}: n={m.sum():3d} 池{r.pool_med[m].mean()*100:+.2f}% '
              f'实际{r.actual[m].mean()*100:+.2f}% '
              f'前视{r.top_f20[m].mean()*100:+.2f}% '
              f'pure_score{r.top_score[m].mean()*100:+.2f}% '
              f'上界差{(r.top_f20[m]-r.actual[m]).mean()*100:+.2f}pp')
    # 池内IC: score/adjusted/ml 与 f20 的秩相关 (逐日spearman, 平均)
    print(f'\n[6] 池内秩相关 (逐日spearman平均, n={len(r)})')
    ics = {'score': [], 'adjusted_score': [], 'ml_score': []}
    for d, g in sig.groupby('date'):
        if d not in set(r.date):
            continue
        g_ok = g[np.isfinite(g.f20)]
        for k in ics:
            if g_ok[k].nunique() >= 5:
                ics[k].append(g_ok[k].corr(g_ok.f20, method='spearman'))
    for k, v in ics.items():
        vv = [x for x in v if np.isfinite(x)]
        print(f'  {k:>14s}: mean={np.mean(vv):+.4f} 正比例='
              f'{(np.array(vv)>0).mean()*100:.0f}% n={len(vv)}')
    print(f'\n[总耗时 {time.time()-t0:.0f}s]')


if __name__ == '__main__':
    main()
