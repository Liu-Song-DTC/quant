#!/usr/bin/env python3
"""2026-09-12 差距4主力探针: 池内重排名器 — 阶段2模型用buy池内秩标签, 池IC vs 基线

背景: gap1探针证组合层贪婪机制无罪(净胜纯分数排序+0.89pp), 真瓶颈=池内IC
(score 0.036 / ml_score 0.065)。gap4敏感性模拟立GO线: 池IC 0.065→0.10-0.12
= +1.6~2.5pp/调仓日(f20口径, 校准锚命中)。本探针复刻bt_execution的23-chunk
purged walk-forward, 同chunk同特征下对照三种模型:
  A. 基线parent: 全市场截面秩标签 (=现网ML, val_ic须与verify日志逐位一致)
  B. 混合hybrid: 池内行→池内秩标签, 池外行→全市场秩标签(保留上下文体积)
  C. 纯池pool_only: 只在池内行上训练, 标签=池内秩(直接学目标函数)
  B/C的cross特征对冻结=parent同chunk的top-8选择(隔离标签效应, 唯一变量=标签)。
评测: 每chunk预测行∩buy池 → 逐日spearman(pred, future_ret=fwd10)平均 = 池IC_f10;
另用close矩阵算池IC_f20(经济口径, 与gap4校准曲线对齐)。判定(须同时满足):
  池IC_f10 ≥ 0.10 且 正比例≥70% 且 投影f20 top-N增益 ≥ +1pp vs 基线 → GO接线。
PIT安全性: buy掩码在日期d来自d日信号(非ML决定项之外无前视), 标签=d→d+10收益,
walk-forward下训练窗止于purge_end, 与现网ML同构。只读。串行。.venv。
执行: cd strategy && /mnt/d/quant/.venv/bin/python
  analysis/probe_gap4_pool_rerank_0912.py > logs/probe_gap4_pool_rerank_0912.log 2>&1
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
VERIFY_LOG = os.path.join(BASE, 'logs', 'bt_execution_20260912_venvA_verify.log')

FWD = 20                    # 经济口径前视期(与gap1/gap4探针一致)
MIN_POOL_ROWS = 20          # 池IC按日spearman的最小池行数
MIN_POOL_TRAIN = 500        # 纯池训练的最小池行数(与train()的500门槛一致)
MIN_POOL_DATES = 3          # chunk内有效池日数下限
# gap4敏感性曲线 (目标池ICρ → top-N fwd20 均值%)
PROJ_RHO = [0.0, 0.036, 0.05, 0.065, 0.08, 0.10, 0.12, 0.15, 0.20]
PROJ_F20 = [1.79, 3.28, 3.84, 4.43, 4.80, 5.74, 6.66, 8.03, 10.47]


class PoolRankPredictor(MLFactorPredictor):
    """阶段2池内重排名器: train()与父类逐行一致, 唯一差异=标签与cross对冻结"""

    def train(self, factor_df, regime_info=None, cross_pairs=None):
        """cross_pairs: 冻结父模型同chunk的交叉特征对(隔离标签效应)。
        标签: _in_pool==1的行→date内池中秩; 其余行→date内全市场秩。
        _in_pool为int8: 不进numeric_cols(int8不在白名单)也不进meta之外的任何特征。"""
        try:
            from xgboost import XGBRegressor
        except ImportError:
            print("[ML] xgboost未安装")
            return None

        exclude_set = {'code', 'date', 'future_ret', 'industry', '_in_pool'}
        numeric_cols = [c for c in factor_df.columns if c not in exclude_set
                        and factor_df[c].dtype in ('float64', 'float32', 'int64', 'int32')]
        meta_cols = [c for c in ['code', 'date', 'future_ret', 'industry', '_in_pool']
                     if c in factor_df.columns]
        df = factor_df[meta_cols + numeric_cols].copy()
        if 'date' in df.columns and 'code' in df.columns:
            df = df.sort_values(['date', 'code'], kind='stable').reset_index(drop=True)

        df = self._cross_sectional_zscore(df, numeric_cols)

        # === 混合池标签: 池内行→池内秩, 池外行→全市场秩 ===
        if 'future_ret' in df.columns and 'date' in df.columns:
            uni = df.groupby('date')['future_ret'].rank(pct=True) - 0.5
            if '_in_pool' in df.columns:
                pool_mask = df['_in_pool'].astype(bool)
                if pool_mask.any():
                    pr = (df.loc[pool_mask]
                          .groupby('date')['future_ret'].rank(pct=True) - 0.5)
                    uni.loc[pool_mask] = pr
                df.drop(columns=['_in_pool'], inplace=True)
            df['future_ret'] = uni

        if cross_pairs is not None:
            self._cross_feature_pairs = list(cross_pairs)
        df = self.prepare_features(df, regime_info, is_train=(cross_pairs is None))
        if 'future_ret' not in df.columns:
            print("[ML] 训练数据缺少future_ret列")
            return None

        valid_features = [c for c in self.feature_cols if c in df.columns]
        if len(valid_features) < 3:
            print(f"[ML] 有效特征不足: {len(valid_features)}")
            return None

        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df = df.dropna(subset=['future_ret'])
        num_cols = df.select_dtypes(include=['float64', 'float32', 'int64', 'int32']).columns
        df[num_cols] = df[num_cols].fillna(0.0)
        train_df = df
        if len(train_df) < 500:
            base_only = [c for c in self.feature_cols
                         if not c.startswith('cross_') and not c.startswith('regime_')]
            base_only = [c for c in base_only if c in df.columns]
            if len(base_only) >= 3:
                train_df = df.dropna(subset=base_only + ['future_ret'])
                valid_features = base_only
            if len(train_df) < 500:
                print(f"[ML] 训练样本不足: {len(train_df)}")
                return None

        dates = sorted(train_df['date'].unique())
        split_idx = int(len(dates) * 0.8)
        train_dates = set(dates[:split_idx])
        val_dates = set(dates[split_idx:])

        tr_mask = train_df['date'].isin(train_dates)
        va_mask = train_df['date'].isin(val_dates)

        X_train = train_df.loc[tr_mask, valid_features].values
        y_train = train_df.loc[tr_mask, 'future_ret'].values
        X_val = train_df.loc[va_mask, valid_features].values
        y_val = train_df.loc[va_mask, 'future_ret'].values

        if len(y_train) < 200 or len(y_val) < 50:
            print(f"[ML] 训练/验证集太小: {len(y_train)}/{len(y_val)}")
            return None

        _max_date = train_df.loc[tr_mask, 'date'].max()
        _days_diff = (_max_date - train_df.loc[tr_mask, 'date']).dt.days.values
        sample_weight = np.exp(_days_diff / 500.0)

        self._ensemble_models = []
        _ensemble_preds = np.zeros(len(y_val))
        _ensemble_seeds = [42, 123, 777]
        for _seed in _ensemble_seeds:
            _params = {**self.xgb_params, 'random_state': _seed}
            _m = XGBRegressor(**_params)
            _m.fit(X_train, y_train, sample_weight=sample_weight,
                   eval_set=[(X_val, y_val)], verbose=False)
            self._ensemble_models.append(_m)
            _ensemble_preds += _m.predict(X_val) / len(_ensemble_seeds)

        self.model = self._ensemble_models[-1]
        self._last_train_date = dates[-1]
        self._trained_features = valid_features

        val_ic = np.corrcoef(_ensemble_preds, y_val)[0, 1] if len(y_val) > 1 else 0
        return val_ic


def build_close():
    """close矩阵 + f20: 与gap1/gap4探针同口径 (65s)"""
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
    close = pd.DataFrame(close).ffill(axis=0).values
    f20 = (pd.DataFrame(close).shift(-FWD) / pd.DataFrame(close) - 1).values
    return D, codes, colmap, close, f20


def main():
    t_start = time.time()
    config = load_config()
    ml_config = config.config.get('ml', {})

    # === 数据加载 ===
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
    print(f'[load] buy池 {len(buy_set)} 行 (2021-04-06起)', flush=True)

    # === 密集交易日历 (复刻probe_train_repeat, chunk10 17位验证过) ===
    fromdate = pd.Timestamp(config.get('backtest.fromdate', '2021-01-01'))
    idx = pd.read_csv(os.path.join(BT, 'sh000001_qfq.csv'), parse_dates=['datetime'])
    calendar = sorted(pd.Timestamp(t) for t in idx['datetime'])
    raw_path = os.path.join(DATA_ROOT, 'stock_data', 'raw_data', '000001', 'qfq.csv')
    raw_dates = pd.to_datetime(pd.read_csv(raw_path, encoding='utf-8-sig',
                                           usecols=['日期'])['日期'])
    early = sorted(d for d in raw_dates
                   if fromdate - pd.Timedelta(days=730) <= d < fromdate)
    all_dates = sorted(set(early) | set(calendar))

    # === regime地图 (复刻管线构造; 索引为RangeIndex→lookup全miss→恒0, 与chunk10 17位吻合) ===
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
    _regime_map = {}
    for _idx, _row in regime_df.iterrows():
        k = _idx.strftime('%Y-%m-%d') if isinstance(_idx, pd.Timestamp) else str(_idx)
        _regime_map[k] = int(_row['regime'])
    del idx_f, sc, gr, regime_df

    # === close矩阵/f20 (经济口径) ===
    D, codes, colmap, close, f20 = build_close()
    tmap = {d: i for i, d in enumerate(pd.to_datetime(D))}

    # === 日志IC解析 (sanity锚): 管线train()按chunk顺序打印"[ML] 训练完成 ... 验证IC=x",
    #     "chunk N: 验证IC="前缀仅失败/复用分支出现(chunk22)。训练完成行序↔chunk 0..22 ===
    _logtext = open(VERIFY_LOG, encoding='utf-8', errors='ignore').read()
    logged = {}
    for i, v in enumerate(re.findall(r'\[ML\] 训练完成: [^\n]*?验证IC=([\d.eE+-]+)', _logtext)):
        logged[i] = float(v)
    for ci, v in re.findall(r'chunk (\d+): 验证IC=([\d.eE+-]+)', _logtext):
        logged[int(ci)] = float(v)  # 17位精度覆盖4dp
    print(f'[sanity] verify日志chunk IC {len(logged)}条 (训练完成行序↔chunk)',
          flush=True)

    # === 23-chunk walk-forward (与bt_execution ML块逐行同构) ===
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
    regime_hits = 0
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

        _train_regime = 0
        if _regime_map:
            _train_regs = [_regime_map.get(d.strftime('%Y-%m-%d'), 0)
                           for d in train_df['date'].unique()]
            _train_regime = int(np.median(_train_regs)) if _train_regs else 0
        regime_hits += (1 if _train_regime != 0 else 0)

        # _in_pool: int8不进numeric_cols, parent的exclude不含它但被meta选择丢弃 → 无影响
        _keys = pd.MultiIndex.from_arrays(
            [train_df['code'], train_df['date']])
        train_df['_in_pool'] = _keys.isin(buy_set).astype('int8')
        n_pool_train = int(train_df['_in_pool'].sum())
        del _keys

        # A. 基线 (现网ML)
        parent = MLFactorPredictor(config.config)
        vic_a = parent.train(train_df, regime_info={'regime': _train_regime})
        # B. 混合池标签
        hyb = PoolRankPredictor(config.config)
        vic_b = hyb.train(train_df, regime_info={'regime': _train_regime},
                          cross_pairs=parent._cross_feature_pairs)
        # C. 纯池
        vic_c = None
        pool_only = None
        if n_pool_train >= MIN_POOL_TRAIN:
            pool_train = train_df[train_df['_in_pool'] == 1].copy()
            pool_only = PoolRankPredictor(config.config)
            vic_c = pool_only.train(pool_train, regime_info={'regime': _train_regime},
                                    cross_pairs=parent._cross_feature_pairs)

        # === 预测 (chunk全部因子行, 向量化≡逐日: zscore/rank均按date分组) ===
        pred_rows = factor_df[factor_df['date'].isin(set(chunk_dates))]
        if len(pred_rows) == 0:
            continue
        pred_a = parent.predict(pred_rows, regime_info={'regime': _train_regime})
        pred_b = hyb.predict(pred_rows, regime_info={'regime': _train_regime})
        pred_c = (pool_only.predict(pred_rows, regime_info={'regime': _train_regime})
                  if pool_only is not None else np.full(len(pred_rows), np.nan))

        pv = pred_rows[['date', 'code', 'future_ret']].copy()
        pv['pred_a'] = pred_a
        pv['pred_b'] = pred_b
        pv['pred_c'] = pred_c
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

        def _ic_by_date(col):
            vals = []
            for d, g in pv.groupby('date'):
                g2 = g[g[col].notna() & g.future_ret.notna()]
                if len(g2) >= MIN_POOL_ROWS:
                    v = g2[col].corr(g2.future_ret, method='spearman')
                    if np.isfinite(v):
                        vals.append(v)
            return np.mean(vals) if vals else np.nan, len(vals)

        def _ic20_by_date(col):
            vals = []
            for d, g in pv.groupby('date'):
                g2 = g[g[col].notna() & g.f20.notna()]
                if len(g2) >= MIN_POOL_ROWS:
                    v = g2[col].corr(g2.f20, method='spearman')
                    if np.isfinite(v):
                        vals.append(v)
            return np.mean(vals) if vals else np.nan, len(vals)

        ic10_a, nd = _ic_by_date('pred_a')
        ic10_b, _ = _ic_by_date('pred_b')
        ic10_c, _ = _ic_by_date('pred_c')
        ic20_a, nd20 = _ic20_by_date('pred_a')
        ic20_b, _ = _ic20_by_date('pred_b')
        ic20_c, _ = _ic20_by_date('pred_c')

        logged_ic = logged.get(chunk_idx, np.nan)
        delta = (vic_a - logged_ic) if np.isfinite(logged_ic) else np.nan
        rows.append(dict(chunk=chunk_idx, first=first_pred_date.date(),
                         n_train=len(train_df), n_pool_train=n_pool_train,
                         vic_a=vic_a, vic_b=vic_b, vic_c=vic_c,
                         ic10_a=ic10_a, ic10_b=ic10_b, ic10_c=ic10_c,
                         ic20_a=ic20_a, ic20_b=ic20_b, ic20_c=ic20_c,
                         n_dates=nd, n_dates20=nd20, n_pool_pred=n_pool_pred,
                         delta=delta))

        def _fmt(x):
            return f'{x:+.4f}' if np.isfinite(x) else '  ---  '

        d_s = f' Δ日志={delta:+.5f}' if np.isfinite(delta) else ''
        print(f'  chunk {chunk_idx:2d}: 训练={len(train_df):,}(池={n_pool_train:,}) '
              f'父IC={_fmt(vic_a)} 池IC_f10 父={_fmt(ic10_a)} 混={_fmt(ic10_b)} '
              f'纯={_fmt(ic10_c)} | f20 父={_fmt(ic20_a)} 混={_fmt(ic20_b)} '
              f'纯={_fmt(ic20_c)} 池行={n_pool_pred:,}{d_s} ({time.time()-t0:.0f}s)',
              flush=True)
        del parent, hyb, pool_only, pv

    # === 汇总 ===
    r = pd.DataFrame(rows)
    r = r[r.n_pool_pred > 0].copy()
    valid = r[r.ic10_a.notna() & r.ic10_b.notna()]
    print(f'\n[sum] 有效chunk {len(valid)}/{len(r)} (池日数≥{MIN_POOL_DATES}按行另标)')
    for c in ['ic10_a', 'ic10_b', 'ic10_c', 'ic20_a', 'ic20_b', 'ic20_c']:
        v = valid[c].dropna()
        if len(v):
            print(f'  {c:>7s}: mean={v.mean():+.4f} 中位={v.median():+.4f} '
                  f'正比例={(v>0).mean()*100:.0f}% n={len(v)}')
    print(f'  n_pool_train均值={valid.n_pool_train.mean():,.0f} '
          f'n_pool_pred均值={valid.n_pool_pred.mean():,.0f}')
    # sanity: 父val_ic vs 日志
    if valid.delta.notna().any():
        dmax = valid.delta.abs().max()
        print(f'[sanity] 父val_ic vs 日志 最大|Δ|={dmax:.3g} '
              f'({"PASS(≤1e-3)" if dmax <= 1e-3 else "FAIL — 排查harness"})')
    print(f'[regime] _train_regime非零chunk数={regime_hits} (预期0, 复刻管线lookup行为)')

    # === 判定 ===
    base10 = valid.ic10_a.mean()
    base20 = valid.ic20_a.mean()
    best10 = max(valid.ic10_b.mean(), valid.ic10_c.dropna().mean()
                 if valid.ic10_c.notna().any() else -9)
    best20 = max(valid.ic20_b.mean(), valid.ic20_c.dropna().mean()
                 if valid.ic20_c.notna().any() else -9)
    best_name = 'hyb' if valid.ic10_b.mean() >= (valid.ic10_c.dropna().mean() if valid.ic10_c.notna().any() else -9) else 'pool_only'
    proj_base = float(np.interp(base20, PROJ_RHO, PROJ_F20))
    proj_best = float(np.interp(best20, PROJ_RHO, PROJ_F20))
    # 正比例(逐chunk)
    pos_b = (valid.ic10_b > 0).mean()
    pos_c = (valid.ic10_c.dropna() > 0).mean() if valid.ic10_c.notna().any() else 0.0
    pos_best = max(pos_b, pos_c)
    print(f'\n[判定] 基线池IC_f10={base10:+.4f} ({best_name}最佳={best10:+.4f}, Δ={best10-base10:+.4f})')
    print(f'  池IC_f20: 基线={base20:+.4f} 最佳={best20:+.4f}')
    print(f'  gap4曲线投影 top-N fwd20: 基线≈{proj_base:+.2f}% → 最佳≈{proj_best:+.2f}% '
          f'(Δ={proj_best-proj_base:+.2f}pp)')
    go = (best10 >= 0.10) and (pos_best >= 0.70) and (proj_best - proj_base >= 1.0)
    soft = (best10 >= 0.08) and not go
    print(f'  GO条件: 池IC_f10≥0.10({best10:+.3f}{"✓" if best10>=0.10 else "✗"}) '
          f'正比例≥70%({pos_best*100:.0f}%{"✓" if pos_best>=0.70 else "✗"}) '
          f'投影增益≥+1pp({proj_best-proj_base:+.2f}pp{"✓" if proj_best-proj_base>=1.0 else "✗"})')
    print(f'  => 判定: {"GO 接线" if go else ("SOFT 加样本/调参再看" if soft else "NO 否决")}')
    print(f'\n[总耗时 {(time.time()-t_start)/60:.1f}min]')


if __name__ == '__main__':
    main()
