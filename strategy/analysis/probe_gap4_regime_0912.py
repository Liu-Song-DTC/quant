#!/usr/bin/env python3
"""2026-09-13 差距4 regime探针(E-GAP4-B): ML regime交叉特征激活 — 现网潜伏bug修复估值

背景: bt_execution 691-695行的_regime_map用regime_df.iterrows()的RangeIndex当键
(str(_idx)="0","1"…), 而737/783行按'YYYY-MM-DD'查询 → 恒miss → _train_regime/
_regime_val恒0 → prepare_features的regime_*交叉特征全为常数0列(死特征, chunk10
17位吻合已证)。这是ML层唯一未兑现的设计元素(regime交互×前10白名单特征)。
本探针在23-chunk purged walk-forward同构harness上(与pool_rerank探针同数据同
路径)对照三臂, 隔离"regime特征激活"的净效应:
  arm0 现网复刻: 训练/预测regime=0标量 (val_ic须与rerank探针父IC一致)
  armA 最小修复: 训练标量=median(修复后map的训练日regime), 预测=逐行当日regime
  armB 完全修复: 训练+预测均逐行当日regime (ml_predictor支持逐行, 95-100行)
评测: val_ic/chunk; 池IC_f10(标签口径)+池IC_f20(经济口径); top-6池内选股f20
(逐日配对, 池行≥20)。
判定(跑前定):
  sanity: arm0 val_ic vs rerank探针日志父IC 最大|Δ|≤1e-3。
  GO(→生产修复+冷跑四指标): 池IC_f20 armB ≥ arm0+0.005 且 top6_f20 armB−arm0
    ≥ +0.20pp 且 配对正比例≥52%。
  否则 NO: 潜伏bug定性为行为保持(死特征对现网零影响), 不修(修了只会加重训方差),
    差距4全面关闭。
PIT安全: regime在日d来自d日前指数数据(检测器无前视), walk-forward同现网。
只读。串行。.venv。预期~20min。
执行: cd strategy && /mnt/d/quant/.venv/bin/python
  analysis/probe_gap4_regime_0912.py > logs/probe_gap4_regime_0912.log 2>&1
"""
import os
import re
import sys
import time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config_loader import load_config
from core.ml_predictor import MLFactorPredictor
from core.market_regime_detector import MarketRegimeDetector

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(BASE, '..', 'data')
BT = os.path.join(DATA_ROOT, 'stock_data', 'backtrader_data')
CACHE = os.path.join(BASE, 'cache', 'factor_df_2718s_809d_d814a206.parquet')
SIG = os.path.join(BASE, 'rolling_validation_results', 'backtest_signals.csv')
RERANK_LOG = os.path.join(BASE, 'logs', 'probe_gap4_pool_rerank_0912.log')

FWD = 20
MIN_POOL_ROWS = 20
TOP_N = 6


def build_close():
    idx = pd.read_csv(os.path.join(BT, 'sh000001_qfq.csv'),
                      usecols=['datetime'], parse_dates=['datetime'])
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
    colmap = {c: i for i, c in enumerate(codes)}
    print(f'[close] 日期 {T} 天 x 股票 {len(codes)} 只', flush=True)
    close = np.full((T, len(codes)), np.nan, dtype=np.float32)
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
    print(f'[close] 加载 {time.time()-t0:.0f}s', flush=True)
    close_m = pd.DataFrame(close).ffill(axis=0).values
    f20 = (pd.DataFrame(close_m).shift(-FWD) / pd.DataFrame(close_m) - 1).values
    return D, codes, colmap, close_m, f20


def main():
    t_start = time.time()
    config = load_config()
    ml_config = config.config.get('ml', {})

    # === 数据加载 (与pool_rerank探针同路径) ===
    factor_df = pd.read_parquet(CACHE)
    if factor_df['code'].dtype != object:
        factor_df['code'] = factor_df['code'].astype(str).str.zfill(6)
    print(f'[load] factor_df {len(factor_df)} 行 '
          f'{factor_df.date.min().date()}~{factor_df.date.max().date()} '
          f'({time.time()-t_start:.0f}s)', flush=True)

    sig = pd.read_csv(SIG, usecols=['code', 'date', 'buy'], dtype={'code': str})
    sig = sig[sig.buy == True].copy()
    sig['code'] = sig['code'].str.zfill(6)
    sig = sig[~sig['code'].str.startswith(('4', '8', '92'))]
    sig['date'] = pd.to_datetime(sig['date'])
    buy_set = set(zip(sig.code, sig.date))
    del sig
    print(f'[load] buy池 {len(buy_set)} 行', flush=True)

    fromdate = pd.Timestamp(config.get('backtest.fromdate', '2021-01-01'))
    idx = pd.read_csv(os.path.join(BT, 'sh000001_qfq.csv'), parse_dates=['datetime'])
    calendar = sorted(pd.Timestamp(t) for t in idx['datetime'])
    raw_path = os.path.join(DATA_ROOT, 'stock_data', 'raw_data', '000001', 'qfq.csv')
    raw_dates = pd.to_datetime(pd.read_csv(raw_path, encoding='utf-8-sig',
                                           usecols=['日期'])['日期'])
    early = sorted(d for d in raw_dates
                   if fromdate - pd.Timedelta(days=730) <= d < fromdate)
    all_dates = sorted(set(early) | set(calendar))

    # === regime序列 (与管线同源: MarketRegimeDetector + 主/辅指数) ===
    idx_f = pd.read_csv(os.path.join(BT, 'sh000001_qfq.csv'), parse_dates=['datetime'])
    if fromdate:
        idx_f = idx_f[idx_f['datetime'] >= fromdate]
    tod = pd.Timestamp(config.get('backtest.todate', '2026-09-10'))
    if tod:
        idx_f = idx_f[idx_f['datetime'] <= tod]
    sc = pd.read_csv(os.path.join(BT, 'sh000852_qfq.csv'), parse_dates=['datetime']) \
        if os.path.exists(os.path.join(BT, 'sh000852_qfq.csv')) else None
    gr = pd.read_csv(os.path.join(BT, '399006_qfq.csv'), parse_dates=['datetime']) \
        if os.path.exists(os.path.join(BT, '399006_qfq.csv')) else None
    regime_df = MarketRegimeDetector().generate(idx_f, small_cap_df=sc, growth_df=gr)
    # 现网buggy map (RangeIndex键) — 验证lookup恒miss
    buggy_map = {}
    for _idx, _row in regime_df.iterrows():
        k = _idx.strftime('%Y-%m-%d') if isinstance(_idx, pd.Timestamp) else str(_idx)
        buggy_map[k] = int(_row['regime'])
    # 修复后map (按datetime列键)
    _dt = pd.to_datetime(regime_df['datetime'])
    fixed_map = dict(zip(_dt.dt.strftime('%Y-%m-%d'),
                         regime_df['regime'].astype(int).values))
    _reg_vals = pd.Series(list(fixed_map.values()))
    print(f'[regime] 检测器输出 {len(regime_df)} 日, 分布: '
          f'{-1:+.0f}={( _reg_vals==-1).mean()*100:.0f}% '
          f'{0:+.0f}={( _reg_vals==0).mean()*100:.0f}% '
          f'{1:+.0f}={( _reg_vals==1).mean()*100:.0f}%', flush=True)
    _sample_dates = pd.to_datetime(regime_df['datetime']).iloc[:5].dt.strftime('%Y-%m-%d')
    _buggy_hits = sum(1 for d in _sample_dates if d in buggy_map)
    print(f'[regime] buggy map键样例={list(buggy_map.keys())[:5]} '
          f'(日期串命中={_buggy_hits}/5, 预期0)', flush=True)
    del idx_f, sc, gr, regime_df

    # === close矩阵/f20 ===
    D, codes, colmap, close, f20 = build_close()
    tmap = {d: i for i, d in enumerate(pd.to_datetime(D))}

    # === rerank探针日志父IC解析 (arm0 sanity锚) ===
    _rtext = open(RERANK_LOG, encoding='utf-8', errors='ignore').read()
    ref_ic = {}
    for ci, v in re.findall(r'chunk\s+(\d+):[^\n]*?父IC=([+-]?\d+\.\d+)', _rtext):
        ref_ic[int(ci)] = float(v)
    print(f'[sanity] rerank日志父IC {len(ref_ic)}条', flush=True)

    # === 23-chunk walk-forward ===
    _train_window = ml_config.get('train_window_days', 750)
    _retrain_freq = ml_config.get('retrain_frequency', 30)
    _pred_start = pd.Timestamp(ml_config.get('pred_start_date', '2021-01-01'))
    _fp_days = int(config.get('dynamic_factor.forward_period', 10))

    _all_dates = sorted(factor_df['date'].unique())
    _pred_dates = [d for d in _all_dates if d >= _pred_start]
    assert len(_pred_dates) == 690, f'pred_dates={len(_pred_dates)}'
    chunk_starts = list(range(0, len(_pred_dates), _retrain_freq))
    print(f'\n[loop] {len(chunk_starts)} chunks, train_window={_train_window}日 '
          f'retrain={_retrain_freq} purge={_fp_days}日', flush=True)

    rows = []
    t6_global = {'pred0': [], 'predA': [], 'predB': []}  # 跨chunk逐日top-6配对
    med_nonzero = 0
    for chunk_idx, chunk_start in enumerate(chunk_starts):
        t0 = time.time()
        chunk_end = min(chunk_start + _retrain_freq, len(_pred_dates))
        chunk_dates = _pred_dates[chunk_start:chunk_end]
        first_pred_date = chunk_dates[0]

        _d0_ts = pd.Timestamp(first_pred_date)
        _d0_pos = int(np.searchsorted(np.asarray(all_dates), _d0_ts))
        _purge_end = all_dates[max(0, _d0_pos - _fp_days)]
        train_start = first_pred_date - pd.Timedelta(days=_train_window)
        train_mask = (factor_df['date'] >= train_start) & \
                     (factor_df['date'] < _purge_end)
        train_df = factor_df[train_mask].copy()
        if len(train_df) < 50000:
            print(f'  chunk {chunk_idx}: 训练样本不足({len(train_df)}), 跳过')
            continue

        # 修复后训练regime: 标量median (armA) + 逐行 (armB, 预排以对齐train内sort)
        _train_dates = train_df['date'].unique()
        _train_regs = [fixed_map.get(d.strftime('%Y-%m-%d'), 0) for d in _train_dates]
        _train_med = int(np.median(_train_regs)) if _train_regs else 0
        med_nonzero += (1 if _train_med != 0 else 0)
        train_sorted = train_df.sort_values(['date', 'code'], kind='stable')
        reg_arr_train = (train_sorted['date'].dt.strftime('%Y-%m-%d')
                         .map(fixed_map).fillna(0).astype(float).values)

        # arm0 现网复刻 (regime=0标量)
        p0 = MLFactorPredictor(config.config)
        vic0 = p0.train(train_df, regime_info={'regime': 0})
        # armA 最小修复 (训练median标量)
        pA = MLFactorPredictor(config.config)
        vicA = pA.train(train_df, regime_info={'regime': _train_med})
        # armB 完全修复 (训练逐行)
        pB = MLFactorPredictor(config.config)
        vicB = pB.train(train_sorted, regime_info={'regime': reg_arr_train})

        # === 预测 (chunk全部因子行; 逐行regime与pred_rows行序对齐, predict不排序) ===
        pred_rows = factor_df[factor_df['date'].isin(set(chunk_dates))]
        if len(pred_rows) == 0:
            continue
        reg_arr_pred = (pred_rows['date'].dt.strftime('%Y-%m-%d')
                        .map(fixed_map).fillna(0).astype(float).values)
        pred0 = p0.predict(pred_rows, regime_info={'regime': 0})
        predA = pA.predict(pred_rows, regime_info={'regime': reg_arr_pred})
        predB = pB.predict(pred_rows, regime_info={'regime': reg_arr_pred})

        pv = pred_rows[['date', 'code', 'future_ret']].copy()
        pv['pred0'] = pred0
        pv['predA'] = predA
        pv['predB'] = predB
        # 池行 + f20 join
        pool_mask = pd.MultiIndex.from_arrays(
            [pv['code'], pv['date']]).isin(buy_set)
        pv = pv[pool_mask].copy()
        ti = pv['date'].map(tmap).values
        ci = pv['code'].map(colmap).values
        ok = ~pd.isna(ti) & ~pd.isna(ci)
        f20v = np.full(len(pv), np.nan)
        f20v[ok] = f20[ti[ok].astype(int), ci[ok].astype(int)]
        pv['f20'] = f20v
        n_pool_pred = len(pv)

        def _ic_by_date(col, label='future_ret'):
            vals = []
            for d, g in pv.groupby('date'):
                g2 = g[g[col].notna() & g[label].notna()]
                if len(g2) >= MIN_POOL_ROWS:
                    v = g2[col].corr(g2[label], method='spearman')
                    if np.isfinite(v):
                        vals.append(v)
            return np.mean(vals) if vals else np.nan, len(vals)

        ic10 = {k: _ic_by_date(k, 'future_ret') for k in
                ['pred0', 'predA', 'predB']}
        ic20 = {k: _ic_by_date(k, 'f20') for k in ['pred0', 'predA', 'predB']}

        # top-6池内选股f20 (逐日配对, 池行≥20; 跨chunk累积)
        for d, g in pv.groupby('date'):
            g2 = g[g.f20.notna()]
            if len(g2) >= MIN_POOL_ROWS:
                for k in t6_global:
                    v = g2.nlargest(TOP_N, k).f20.mean()
                    if np.isfinite(v):
                        t6_global[k].append(v)

        logged_ic = ref_ic.get(chunk_idx, np.nan)
        delta = (vic0 - logged_ic) if np.isfinite(logged_ic) else np.nan
        rows.append(dict(chunk=chunk_idx, first=first_pred_date.date(),
                         n_train=len(train_df), train_med=_train_med,
                         vic0=vic0, vicA=vicA, vicB=vicB,
                         **{f'ic10_{k}': ic10[k][0] for k in ic10},
                         **{f'ic20_{k}': ic20[k][0] for k in ic20},
                         n_pool_pred=n_pool_pred, delta=delta))

        def _fmt(x):
            return f'{x:+.4f}' if np.isfinite(x) else '  ---  '

        d_s = f' Δref={delta:+.5f}' if np.isfinite(delta) else ''
        print(f'  chunk {chunk_idx:2d}: 训练={len(train_df):,} med={_train_med:+d} '
              f'父IC={_fmt(vic0)}/{_fmt(vicA)}/{_fmt(vicB)} '
              f'池IC_f10 {_fmt(ic10["pred0"][0])}/{_fmt(ic10["predA"][0])}/'
              f'{_fmt(ic10["predB"][0])} f20 {_fmt(ic20["pred0"][0])}/'
              f'{_fmt(ic20["predA"][0])}/{_fmt(ic20["predB"][0])} '
              f'池行={n_pool_pred:,}{d_s} ({time.time()-t0:.0f}s)',
              flush=True)
        del p0, pA, pB, pv

    # === 汇总 ===
    r = pd.DataFrame(rows)
    r = r[r.n_pool_pred > 0].copy()
    valid = r[r.ic10_pred0.notna()]
    print(f'\n[sum] 有效chunk {len(valid)}/{len(r)}')
    for c in ['vic0', 'vicA', 'vicB']:
        v = valid[c].dropna()
        print(f'  {c:>6s}: mean={v.mean():+.4f} n={len(v)}')
    for c in ['ic10_pred0', 'ic10_predA', 'ic10_predB',
              'ic20_pred0', 'ic20_predA', 'ic20_predB']:
        v = valid[c].dropna()
        if len(v):
            print(f'  {c:>9s}: mean={v.mean():+.4f} 中位={v.median():+.4f} '
                  f'正比例={(v>0).mean()*100:.0f}% n={len(v)}')
    if valid.delta.notna().any():
        dmax = valid.delta.abs().max()
        print(f'[sanity] arm0 val_ic vs rerank父IC 最大|Δ|={dmax:.3g} '
              f'({"PASS(≤1e-3)" if dmax <= 1e-3 else "FAIL — 排查harness"})')
    print(f'[regime] _train_med非零chunk数={med_nonzero}/23 '
          f'(现网buggy=0; 修复后应>0)')

    # === 判定 (跑前定; 配对=逐日) ===
    t0 = np.array(t6_global['pred0'])
    tA = np.array(t6_global['predA'])
    tB = np.array(t6_global['predB'])
    n_days = min(len(t0), len(tA), len(tB))
    t0, tA, tB = t0[:n_days], tA[:n_days], tB[:n_days]
    print(f'\n[sum] top-6逐日配对 n={n_days}日: 现网={t0.mean()*100:+.2f}% '
          f'armA={tA.mean()*100:+.2f}% armB={tB.mean()*100:+.2f}%')
    base20 = valid.ic20_pred0.mean()
    b20 = valid.ic20_predB.mean()
    d_t6 = tB - t0
    pos_share = (d_t6 > 0).mean() if len(d_t6) else 0.0
    go = (b20 >= base20 + 0.005) and (d_t6.mean() >= 0.20 / 100) and (pos_share >= 0.52)
    print(f'\n[判定] 池IC_f20: 现网={base20:+.4f} → armB={b20:+.4f} '
          f'(Δ={b20-base20:+.4f}, 须≥+0.005{"✓" if b20>=base20+0.005 else "✗"})')
    print(f'  top-6 f20配对: Δ={d_t6.mean()*100:+.2f}pp (须≥+0.20pp'
          f'{"✓" if d_t6.mean()>=0.20/100 else "✗"}) '
          f'正比例={pos_share*100:.0f}% (须≥52%{"✓" if pos_share>=0.52 else "✗"})')
    print(f'  => 判定: {"GO 生产修复+冷跑四指标" if go else "NO 行为保持不修, 差距4关闭"}')
    print(f'\n[总耗时 {(time.time()-t_start)/60:.1f}min]')


if __name__ == '__main__':
    main()
