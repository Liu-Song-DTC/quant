# core/ml_predictor.py
"""
XGBoost ML预测层 - 捕获因子间的非线性交互

设计原则:
1. 浅树(max_depth=4)防过拟合，金融数据信噪比极低
2. 时序验证集早停（非随机划分），防止look-ahead
3. 特征工程：top-8因子两两交叉 + 市场状态交互
4. 与线性因子得分按blend_weight混合，不替代

用法:
    predictor = MLFactorPredictor(config)
    predictor.train(factor_df)          # 训练
    preds = predictor.predict(date_df)  # 截面预测
"""

import numpy as np
import pandas as pd
import warnings
from typing import Dict, List, Optional

# 常数特征在金融数据中常见(如概念热度在大部分股票为0), 不影响训练
warnings.filterwarnings('ignore', message='An input array is constant')


class MLFactorPredictor:
    """XGBoost因子预测器 - 非线性因子组合"""

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.model = None
        self.feature_cols = []
        self.feature_contract = {}
        self._fitted_preprocessing = None
        self._last_train_date = None

        ml_cfg = self.config.get('ml', {})
        market_cols = ml_cfg.get('market_feature_cols') or []
        if (not isinstance(market_cols, (list, tuple))
                or any(not isinstance(c, str) or not c for c in market_cols)
                or set(market_cols) & {'code', 'date', 'industry', 'future_ret', 'label_end_date'}):
            raise ValueError('market_feature_cols must list feature names, not labels or identifiers')
        self._market_feature_cols = list(dict.fromkeys(market_cols))
        self.xgb_params = {
            'n_estimators': ml_cfg.get('n_estimators', 200),
            'max_depth': ml_cfg.get('max_depth', 4),
            'learning_rate': ml_cfg.get('learning_rate', 0.05),
            'subsample': ml_cfg.get('subsample', 0.7),
            'colsample_bytree': ml_cfg.get('colsample_bytree', 0.6),
            'reg_lambda': ml_cfg.get('reg_lambda', 2.0),
            'reg_alpha': ml_cfg.get('reg_alpha', 1.0),
            'min_child_weight': ml_cfg.get('min_child_weight', 5),
            'random_state': 42,
            'n_jobs': min(16, __import__('os').cpu_count() or 4),
            # 2026-10-09: 原min(4,...)在20核机器只用4线程; 探针证实hist树法
            # n_jobs=4 vs 16预测逐位一致(0/363,600), 拟合1.38×加速, 免费取用
            'verbosity': 0,
            'tree_method': 'hist',      # 节省训练内存
            'device': 'cpu',            # 强制 CPU（避免 GPU OOM）
            'early_stopping_rounds': 30,  # XGBoost 3.x: 在构造时传入
        }

        self._use_cross_features = ml_cfg.get('use_cross_features', True)
        self._use_regime_features = ml_cfg.get('use_regime_features', True)
        # 缺失指示位 (2026-10-26): 让ML知道哪些因子值是中性化填出来的, 并剔除
        # "全部特征都被填"的池外掩码行。默认关 = 逐位零改动。
        self.use_fill_indicator = bool(ml_cfg.get('use_fill_indicator', False))
        self.fill_tol = float(ml_cfg.get('fill_tol', 1e-5))
        self.fill_drop_pool_rows = bool(ml_cfg.get('fill_drop_pool_rows', True))
        self.fill_indicator_max_frac = float(ml_cfg.get('fill_indicator_max_frac', 0.35))
        self._fill_indicator_src_cols: List[str] = []
        self.blend_weight = ml_cfg.get('blend_weight', 0.50)
        self._cross_feature_pairs = []  # 训练时固化，预测时复用
        self._regime_feature_sources = None

    def prepare_features(self, df: pd.DataFrame,
                         regime_info: dict = None,
                         is_train: bool = False,
                         selection_df: pd.DataFrame = None) -> pd.DataFrame:
        exclude = {'code', 'date', 'future_ret', 'industry'}
        base_features = [c for c in df.columns if c not in exclude
                         and df[c].dtype in ('float64', 'float32', 'int64', 'int32')]

        # 白名单过滤: 只用已验证的因子, 避免噪声因子混入特征空间
        whitelist = self.config.get('ml', {}).get('feature_whitelist', None)
        if not is_train and self._fitted_preprocessing is not None:
            whitelist = self._fitted_preprocessing['feature_whitelist']
        if whitelist:
            base_features = [f for f in base_features if f in whitelist]

        if len(base_features) < 2:
            self.feature_cols = base_features
            return df

        if self._use_cross_features:
            if is_train or not self._cross_feature_pairs:
                # A shared market observation has no within-date cross-sectional
                # rank. Preserve it as a time-series input, not a constant rank.
                crossable = [f for f in base_features if f not in self._market_feature_cols]
                top_k = min(8, len(crossable))
                top_features = self._select_top_features(
                    selection_df if selection_df is not None else df, crossable, k=top_k)
                self._cross_feature_pairs = []
                for i, f1 in enumerate(top_features):
                    for f2 in top_features[i+1:]:
                        self._cross_feature_pairs.append((f1, f2))
            # Rank-based交叉: 先转截面百分位再相乘, 对异常值和分布漂移更鲁棒
            # 性能(2026-10-09): 全部pair一次polars窗口函数完成, 替代逐pair
            # pandas groupby.rank (每chunk最多56次全帧groupby) — /tmp/probe_polars_mlprep.py
            # 证实与pandas逐位一致 (rank/count均只计非空, 空rank填0.5)
            _pend_pairs = []
            for f1, f2 in self._cross_feature_pairs:
                col = f'cross_{f1}_{f2}'
                if col not in df.columns:
                    if f1 in df.columns and f2 in df.columns:
                        _pend_pairs.append((f1, f2, col))
                        if col not in base_features:
                            base_features.append(col)
            if _pend_pairs:
                import polars as pl
                _pf = pl.from_pandas(
                    df[['date'] + sorted({f for p in _pend_pairs for f in p[:2]})])
                _exprs = []
                for _f1, _f2, _col in _pend_pairs:
                    _r1 = pl.col(_f1).rank('average').over('date') \
                        / pl.col(_f1).count().over('date')
                    _r2 = pl.col(_f2).rank('average').over('date') \
                        / pl.col(_f2).count().over('date')
                    _exprs.append((_r1.fill_null(0.5) * _r2.fill_null(0.5)).alias(_col))
                _pf = _pf.with_columns(_exprs)
                for _f1, _f2, _col in _pend_pairs:
                    df[_col] = _pf[_col].to_numpy()

        if self._use_regime_features and regime_info:
            if 'by_date' in regime_info:
                regime_val = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d').map(
                    regime_info['by_date']).fillna(0).to_numpy()
            else:
                regime_val = regime_info.get('regime', 0)
            # 支持逐行regime: list/array/Series按行使用, 标量广播
            if isinstance(regime_val, (list, np.ndarray)):
                regime_arr = np.asarray(regime_val, dtype=float)
                if len(regime_arr) != len(df):
                    regime_arr = np.full(len(df), float(np.mean(regime_arr)))
            elif isinstance(regime_val, pd.Series):
                regime_arr = regime_val.values.astype(float)
                if len(regime_arr) != len(df):
                    regime_arr = np.full(len(df), float(regime_arr.mean()))
            else:
                regime_arr = np.full(len(df), float(regime_val))
            if is_train or self._regime_feature_sources is None:
                self._regime_feature_sources = list(base_features[:10])
            for f in self._regime_feature_sources:
                col = f'regime_{f}'
                if col not in df.columns and f in df.columns:
                    df[col] = df[f].fillna(0).values * regime_arr
                    base_features.append(col)

        if self.use_fill_indicator:
            for fc in self._fill_indicator_src_cols:
                col = f'{fc}__filled'
                if col in df.columns and col not in base_features:
                    base_features.append(col)

        self.feature_cols = [c for c in base_features
                             if c in df.columns and c not in exclude]
        return df

    def _fill_flags(self, df: pd.DataFrame, feat_cols: List[str]) -> Dict[str, np.ndarray]:
        """逐格识别"被中性化填成当日均值"的格子 (df 必须已做截面 z-score)。

        中性化把无效值(池外掩码/自然缺失)填成该列当日均值 ⇒ 之后的截面 z-score
        让被填的格子恰好落在 0 附近。探针实测(.arms/probe/fill_detect_probe_1026.py,
        真实 raw 缓存 + 真实池掩码 + 真实 neutralize_factor_df):
            被填格最大 |z| = 2.4e-6, 自然格最小 |z| ≈ 1e-5 (非退化列)
            阈值 1e-5 ⇒ 行级(全填行=池外掩码行) precision/recall = 100% / 100%
        """
        flags: Dict[str, np.ndarray] = {}
        for c in feat_cols:
            if c not in df.columns:
                continue
            v = pd.to_numeric(df[c], errors='coerce').to_numpy(dtype='float64', na_value=np.nan)
            flags[c] = np.isfinite(v) & (np.abs(v) <= self.fill_tol)
        return flags

    def _mark_filled_train(self, df: pd.DataFrame, feat_cols: List[str]) -> pd.DataFrame:
        """训练期: 剔除池外掩码行 + 给被填的格子加 0/1 指示位。

        池外掩码行 = 池成员之外的所有因子列被置NaN ⇒ 中性化后全部等于当日均值 ⇒
        "所有有值的特征列都在当日均值上"。这些行不是可交易标的, 让它们进训练集
        等于把成员资格(池)泄漏给模型 ⇒ 整行剔除(标签的截面排名分母也随之只算成员)。
        指示位: 让模型知道"这个基本面值是假的"(均值占位), 而不是把它当中性信号。
        """
        if not self.use_fill_indicator:
            return df
        flags = self._fill_flags(df, feat_cols)
        if not flags:
            return df
        n = len(df)
        notna = np.zeros(n, dtype=np.int32)
        filled = np.zeros(n, dtype=np.int32)
        for c, fl in flags.items():
            v = pd.to_numeric(df[c], errors='coerce').to_numpy(dtype='float64', na_value=np.nan)
            ok = np.isfinite(v)
            notna += ok
            filled += fl & ok
        keep = np.ones(n, dtype=bool)
        if self.fill_drop_pool_rows:
            pool_rows = (notna >= 2) & (filled >= notna)
            if pool_rows.any():
                keep = ~pool_rows
                df = df.loc[keep].copy()
        src_cols: List[str] = []
        for c, fl in flags.items():
            fl_k = fl[keep]
            if fl_k.mean() > self.fill_indicator_max_frac:
                # 退化列: 当日均值撞上自然众数(eg concept_heat 大量0) ⇒ 指示位失真
                continue
            df[f'{c}__filled'] = fl_k.astype('float32')
            src_cols.append(c)
        self._fill_indicator_src_cols = src_cols
        print(f"[ML][FILL_INDICATOR] 池外掩码行 {int((~keep).sum())}/{n} 剔除, "
              f"指示位列 {len(src_cols)}/{len(flags)}")
        return df

    def _mark_filled_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """预测期: 按训练期固化的列集合加指示位 (不剔行, 预测要覆盖全市场)。"""
        if not self._fill_indicator_src_cols:
            return df
        flags = self._fill_flags(df, self._fill_indicator_src_cols)
        for c in self._fill_indicator_src_cols:
            if c not in flags:
                continue
            df[f'{c}__filled'] = flags[c].astype('float32')
        return df

    @staticmethod
    def describe_feature_contract(frame: pd.DataFrame, whitelist=None):
        """Report configured versus usable inputs without changing selection."""
        meta = {'code', 'date', 'future_ret', 'industry'}
        numeric = [c for c in frame.columns if c not in meta
                   and frame[c].dtype in ('float64', 'float32', 'int64', 'int32')]
        requested = list(whitelist) if whitelist else numeric.copy()
        requested_set = set(requested)
        return {
            'status': 'observed_training_input',
            'requested': requested,
            'available_base': [c for c in numeric if c in requested_set],
            'absent': [c for c in requested if c not in frame.columns],
            'non_numeric': [c for c in requested if c in frame.columns and c not in numeric],
            'generated_but_not_requested': [c for c in numeric if c not in requested_set],
            'scope': 'columns and numeric dtypes; does not prove source freshness or nonmissing coverage',
        }

    @staticmethod
    def _cross_sectional_zscore(df: pd.DataFrame, cols: List[str], preserve_cols=()):
        """截面标准化: 每个日期内 (x - mean) / std, 消除市场量级干扰"""
        # 注意(2026-10-09): 勿换polars — numpy在float32列上保持f32累加,
        # polars窗口mean/std升f64累加, 结果差~1e-7相对(探针实测378k单元
        # 不一致), XGBoost hist分箱边界翻转会破逐位恒等。此路径保留pandas。
        if 'date' not in df.columns:
            return df
        for col in cols:
            if col not in df.columns:
                continue
            if col in preserve_cols:
                if (df.groupby('date')[col].nunique(dropna=True) > 1).any():
                    raise ValueError(f'market feature differs across stocks on the same date: {col}')
                continue
            grouped = df.groupby('date')[col]
            mean_s = grouped.transform('mean')
            std_s = grouped.transform('std').replace(0, 1.0)
            df[col] = (df[col] - mean_s) / std_s
        return df

    def train(self, factor_df: pd.DataFrame,
              regime_info: dict = None) -> Optional[float]:
        """Build a candidate; preserve the current model on failure or no fit."""
        import copy
        candidate = copy.copy(self)
        candidate.__dict__ = self.__dict__.copy()
        result = candidate._train_impl(factor_df, regime_info)
        if result is not None:
            self.__dict__ = candidate.__dict__
        return result

    def _train_impl(self, factor_df: pd.DataFrame,
                    regime_info: dict = None) -> Optional[float]:
        """训练XGBoost模型，返回验证集IC"""
        try:
            from xgboost import XGBRegressor
        except ImportError:
            print("[ML] xgboost未安装")
            return None

        self.feature_contract = self.describe_feature_contract(
            factor_df, self.config.get('ml', {}).get('feature_whitelist'))
        self.feature_contract['market_feature_cols'] = list(self._market_feature_cols)
        if self.feature_contract['absent'] or self.feature_contract['non_numeric']:
            print('[ML][FEATURE_CONTRACT] '
                  f"配置={len(self.feature_contract['requested'])} "
                  f"可用={len(self.feature_contract['available_base'])} "
                  f"缺列={self.feature_contract['absent']} "
                  f"非数值={self.feature_contract['non_numeric']}")

        # 只复制数值型特征列
        exclude_set = {'code', 'date', 'future_ret', 'industry'}
        numeric_cols = [c for c in factor_df.columns if c not in exclude_set
                       and factor_df[c].dtype in ('float64', 'float32', 'int64', 'int32')]
        meta_cols = [c for c in ['code', 'date', 'future_ret', 'industry', 'label_end_date'] if c in factor_df.columns]
        df = factor_df[meta_cols + numeric_cols].copy()
        availability_cols = self.config.get('ml', {}).get('feature_whitelist') or numeric_cols
        availability_cols = [c for c in availability_cols if c in numeric_cols]
        if not availability_cols:
            return None
        # 股票池外的全缺失行不参与标签截面和模型训练。
        df = df.loc[df[availability_cols].notna().any(axis=1)].copy()
        # 行序无关性: XGBoost subsample按行位置抽样, merge_asof等上游重排序会改变
        # 抽样子集导致同数据不同模型. 固定按(date, code)排序保证确定性.
        if 'date' in df.columns and 'code' in df.columns:
            df = df.sort_values(['date', 'code'], kind='stable').reset_index(drop=True)

        # 截面标准化: 消除跨日期量级差异, 模型学习相对排名而非绝对量级
        df = self._cross_sectional_zscore(df, numeric_cols, self._market_feature_cols)

        # 缺失指示位 (可选, 默认关): 识别被中性化填成当日均值的格子 ⇒ 剔除池外掩码行
        # (成员资格泄漏) + 给模型 0/1 指示位。必须在标签排名之前, 否则标签的截面
        # 分母仍把非成员算进去。
        if self.use_fill_indicator:
            _feat = [c for c in availability_cols if c not in exclude_set]
            df = self._mark_filled_train(df, _feat)

        # Rank标签: 截面百分位排名 → 中心化到[-0.5, 0.5], 预测排序而非绝对收益
        # 性能(2026-10-09): polars窗口rank替代pandas groupby.rank, 探针逐位一致
        if 'future_ret' in df.columns and 'date' in df.columns:
            import polars as pl
            _pf = pl.from_pandas(df[['date', 'future_ret']])
            _fr = pl.col('future_ret')
            df['future_ret'] = _pf.select(
                (_fr.rank('average').over('date')
                 / _fr.count().over('date') - 0.5).alias('_r')
            )['_r'].to_numpy()

        if 'future_ret' not in df.columns:
            print("[ML] 训练数据缺少future_ret列")
            return None

        # 先划分并隔离标签窗口，再选择交叉特征；验证标签绝不能参与选特征。
        df = df.replace([np.inf, -np.inf], np.nan)
        dates = sorted(df.loc[df['future_ret'].notna(), 'date'].unique())
        split_idx = int(len(dates) * 0.8)
        if split_idx < 1 or split_idx >= len(dates):
            return None
        val_start = pd.Timestamp(dates[split_idx])
        train_mask, val_mask = self.split_masks(df, val_start)
        # 构造交叉因子的截面必须包含当时全部可预测股票。标签不可得（例如
        # 次日涨停买不入）只能排除监督样本，不能提前改变当日特征的排名分母。
        df = self.prepare_features(df, regime_info, is_train=True,
                                   selection_df=df.loc[train_mask & df['future_ret'].notna()])

        valid_features = [c for c in self.feature_cols if c in df.columns]
        if len(valid_features) < 3:
            print(f"[ML] 有效特征不足: {len(valid_features)}")
            return None

        # 特征过滤已关闭: XGBoost树模型自带方向学习 + L1/L2正则 + sample_weight时变加权
        # 负IC特征不删也不翻转 — 树分裂方向不依赖特征符号, regime交叉特征处理多空不对称

        # 处理Inf/NaN: 先丢无标签行(尾部日期future_ret=NaN), 再fill特征NaN
        # (顺序不能反: fillna会把NaN标签变成0.0=中位数排名, 污染训练)
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

        train_mask, val_mask = self.split_masks(train_df, val_start)

        X_train = train_df.loc[train_mask, valid_features].values
        y_train = train_df.loc[train_mask, 'future_ret'].values
        X_val = train_df.loc[val_mask, valid_features].values
        y_val = train_df.loc[val_mask, 'future_ret'].values

        if len(y_train) < 200 or len(y_val) < 50:
            print(f"[ML] 训练/验证集太小: {len(y_train)}/{len(y_val)}")
            return None

        # 时序样本权重: 越靠近训练期末的数据权重越大(近期市场规律更相关)
        _max_date = train_df.loc[train_mask, 'date'].max()
        _days_diff = (_max_date - train_df.loc[train_mask, 'date']).dt.days.values
        sample_weight = self.time_weights(_days_diff)

        # Ensemble: 3个不同种子模型，预测取均值 → 降方差5-10%
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

        # 主模型保留最后一个(用于特征重要性), predict时用ensemble
        self.model = self._ensemble_models[-1]
        self._last_train_date = dates[-1]

        # 固化训练特征: predict时复用, 避免prepare_features覆盖导致shape mismatch
        self._trained_features = valid_features
        self.feature_contract['trained_features'] = list(valid_features)
        import copy
        self._fitted_preprocessing = {
            'version': 1,
            'feature_whitelist': copy.deepcopy(self.config.get('ml', {}).get('feature_whitelist')),
            'use_cross_features': self._use_cross_features,
            'use_regime_features': self._use_regime_features,
            'use_fill_indicator': self.use_fill_indicator,
            'fill_tol': self.fill_tol,
            'fill_drop_pool_rows': self.fill_drop_pool_rows,
            'fill_indicator_max_frac': self.fill_indicator_max_frac,
        }

        # 每日截面 Rank IC，避免池大小和跨日分布主导一个混合 Pearson IC。
        scored = pd.DataFrame({'date': train_df.loc[val_mask, 'date'].values,
                               'pred': _ensemble_preds, 'target': y_val})
        daily_ics = [g['pred'].corr(g['target'], method='spearman')
                     for _, g in scored.groupby('date') if len(g) >= 5]
        finite_ics = np.asarray(daily_ics, dtype=float)
        finite_ics = finite_ics[np.isfinite(finite_ics)]
        val_ic = float(finite_ics.mean()) if len(finite_ics) else 0.0
        self.validation_summary = {'daily_rank_ic': val_ic, 'dates': int(len(finite_ics)),
                                   'train_rows': int(train_mask.sum()),
                                   'validation_rows': int(val_mask.sum()),
                                   'validation_start': str(val_start.date())}

        # 特征重要性: ensemble均值
        _all_imp = {}
        for _m in self._ensemble_models:
            for fn, imp in zip(valid_features, _m.feature_importances_):
                _all_imp[fn] = _all_imp.get(fn, 0) + imp / len(_ensemble_seeds)
        self._feature_importances = dict(sorted(_all_imp.items(), key=lambda x: -x[1]))

        log_context = self.config.get('ml', {}).get('log_context', 'WALK_FORWARD')
        print(f"[ML][{log_context}] 训练完成: 样本={len(train_df)}, "
              f"特征={len(valid_features)}, 验证IC={val_ic:.4f}")
        return val_ic

    @staticmethod
    def time_weights(days_old):
        return np.exp(-np.maximum(np.asarray(days_old, dtype=float), 0.0) / 500.0)

    def split_masks(self, df, validation_start):
        dates = pd.to_datetime(df['date'])
        val_start = pd.Timestamp(validation_start)
        train = dates < val_start
        if 'label_end_date' in df:
            train &= pd.to_datetime(df['label_end_date']) < val_start
        else:
            # 兼容旧的无标签终止日帧：宁可多隔离，不让相邻收益窗口重叠。
            horizon = int(self.config.get('dynamic_factor', {}).get('forward_period', 10))
            train &= dates < val_start - pd.offsets.BDay(horizon)
        return train, dates >= val_start

    def predict(self, df: pd.DataFrame,
                regime_info: dict = None) -> np.ndarray:
        if self.model is None:
            return np.zeros(len(df))
        if 'date' not in df or (df.groupby('date').size() < 2).any():
            raise ValueError('ML截面模型必须联合输入同日多只股票；禁止按单只股票标准化预测')

        exclude_set = {'code', 'date', 'future_ret', 'industry'}
        numeric_cols = [c for c in df.columns if c not in exclude_set
                       and df[c].dtype in ('float64', 'float32', 'int64', 'int32')]
        meta_cols = [c for c in ['code', 'date', 'future_ret', 'industry'] if c in df.columns]
        df = df[meta_cols + numeric_cols].copy()

        # 截面标准化 (与训练一致)
        df = self._cross_sectional_zscore(df, numeric_cols, self._market_feature_cols)

        # 缺失指示位 (与训练同口径: 同一批列, 同一阈值)
        if self.use_fill_indicator:
            df = self._mark_filled_predict(df)

        df = self.prepare_features(df, regime_info)
        # 使用训练时固化的特征列表, 避免prepare_features重建的feature_cols与模型不匹配
        trained_cols = getattr(self, '_trained_features', self.feature_cols)
        valid_features = [c for c in trained_cols if c in df.columns]
        if not valid_features:
            return np.zeros(len(df))

        X = df[valid_features].replace([np.inf, -np.inf], np.nan).fillna(0).values
        # 补齐缺失特征列: 训练时63特征, 预测时可能只有32个基础因子 → 缺失列填0
        if len(valid_features) < len(trained_cols):
            full_X = np.zeros((len(df), len(trained_cols)))
            col_to_idx = {c: i for i, c in enumerate(trained_cols)}
            for j, c in enumerate(valid_features):
                full_X[:, col_to_idx[c]] = X[:, j]
            X = full_X
        # Ensemble预测: 3个模型取均值
        if hasattr(self, '_ensemble_models') and self._ensemble_models:
            preds = np.zeros(len(df))
            for _m in self._ensemble_models:
                preds += _m.predict(X) / len(self._ensemble_models)
            return preds
        elif self.model is not None:
            return self.model.predict(X)
        return np.zeros(len(df))

    def get_feature_importance(self) -> Dict[str, float]:
        if not hasattr(self, '_feature_importances'):
            return {}
        return self._feature_importances

    def is_trained(self) -> bool:
        return hasattr(self, '_ensemble_models') and len(self._ensemble_models) > 0

    def save_model(self, path: str):
        """保存ensemble模型到文件"""
        if not hasattr(self, '_ensemble_models') or not self._ensemble_models:
            return
        import hashlib
        import os
        from pathlib import Path
        import tempfile
        from .live_state import atomic_bytes
        checksums = {}
        for i, _m in enumerate(self._ensemble_models):
            member = Path(f"{path}.{i}.json")
            fd, temp = tempfile.mkstemp(dir=member.parent, prefix=member.name + '.', suffix='.json')
            os.close(fd)
            try:
                _m.save_model(temp)
                payload = Path(temp).read_bytes()
                checksums[str(i)] = hashlib.sha256(payload).hexdigest()
                atomic_bytes(member, payload)
            finally:
                if os.path.exists(temp):
                    os.unlink(temp)
        import pickle
        atomic_bytes(path + '.meta', pickle.dumps({
                'feature_cols': self.feature_cols,
                'trained_features': getattr(self, '_trained_features', self.feature_cols),
                'cross_feature_pairs': self._cross_feature_pairs,
                'feature_importances': getattr(self, '_feature_importances', {}),
                'n_models': len(self._ensemble_models),
                'model_sha256': checksums,
                'feature_contract': self.feature_contract,
                'validation_summary': getattr(self, 'validation_summary', None),
                'market_feature_cols': list(self._market_feature_cols),
                'regime_feature_sources': self._regime_feature_sources,
                'preprocessing': self._fitted_preprocessing,
                # 缺失指示位(2026-10-26): 训练期固化的被填列集合 — 预测期必须用同一批列
                # 加指示位, 否则特征矩阵与模型不匹配(载入模型后重算预测的路径会走错)。
                'fill_indicator_src_cols': list(self._fill_indicator_src_cols),
            }))

    def load_model(self, path: str):
        """从文件加载ensemble模型"""
        from xgboost import XGBRegressor
        import hashlib
        from pathlib import Path
        import pickle
        meta_path = path + '.meta'
        with open(meta_path, 'rb') as f:
            data = pickle.load(f)
        n_models = data.get('n_models', 3)
        if not isinstance(n_models, int) or n_models < 1 or not data.get('feature_cols'):
            raise ValueError('invalid ensemble metadata')
        preprocessing = data.get('preprocessing')
        if preprocessing is not None:
            required = {'version', 'feature_whitelist', 'use_cross_features', 'use_regime_features',
                        'use_fill_indicator', 'fill_tol', 'fill_drop_pool_rows', 'fill_indicator_max_frac'}
            if (not isinstance(preprocessing, dict) or preprocessing.get('version') != 1
                    or not required.issubset(preprocessing)):
                raise ValueError('invalid fitted preprocessing metadata')
        members = []
        for i in range(n_models):
            _m = XGBRegressor()
            _mp = f"{path}.{i}.json"
            payload = Path(_mp).read_bytes()  # Any missing member invalidates the entire ensemble.
            expected = data.get('model_sha256')
            if expected is not None and hashlib.sha256(payload).hexdigest() != expected.get(str(i)):
                raise ValueError(f'ensemble member checksum mismatch: {_mp}')
            _m.load_model(bytearray(payload))
            members.append(_m)
        self.feature_cols = data['feature_cols']
        self._trained_features = data.get('trained_features', data['feature_cols'])
        self._fill_indicator_src_cols = list(data.get('fill_indicator_src_cols', []) or [])
        self._cross_feature_pairs = data['cross_feature_pairs']
        self._ensemble_models = members
        self.model = members[-1]
        self._feature_importances = data.get('feature_importances', {})
        self.feature_contract = data.get('feature_contract') or {'status': 'legacy_metadata_unavailable'}
        self.validation_summary = data.get('validation_summary')
        # Preprocessing belongs to the fitted model, not the caller's new config.
        self._market_feature_cols = list(data.get('market_feature_cols') or [])
        self._regime_feature_sources = data.get('regime_feature_sources')
        if self._regime_feature_sources is None:
            self._regime_feature_sources = [c[len('regime_'):] for c in self._trained_features
                                            if c.startswith('regime_')]
        self._fitted_preprocessing = preprocessing
        if preprocessing is not None:
            import copy
            self.config = copy.deepcopy(self.config)
            self.config.setdefault('ml', {})['feature_whitelist'] = preprocessing['feature_whitelist']
            self._use_cross_features = preprocessing['use_cross_features']
            self._use_regime_features = preprocessing['use_regime_features']
            self.use_fill_indicator = preprocessing['use_fill_indicator']
            self.fill_tol = preprocessing['fill_tol']
            self.fill_drop_pool_rows = preprocessing['fill_drop_pool_rows']
            self.fill_indicator_max_frac = preprocessing['fill_indicator_max_frac']

    @staticmethod
    def _filter_stable_features(df: pd.DataFrame, features: List[str],
                                 min_ic_sharpe: float = 0.2,
                                 n_windows: int = 5) -> List[str]:
        """过滤IC不稳定的噪声特征: 分n_windows时间段, IC_sharpe < min_ic_sharpe 则移除"""
        if 'future_ret' not in df.columns or 'date' not in df.columns:
            return features
        dates = sorted(df['date'].unique())
        if len(dates) < n_windows * 2:
            return features
        window_size = len(dates) // n_windows
        stable = []
        for f in features:
            if f not in df.columns:
                continue
            window_ics = []
            for w in range(n_windows):
                w_start = w * window_size
                w_end = (w + 1) * window_size if w < n_windows - 1 else len(dates)
                w_dates = set(dates[w_start:w_end])
                valid = df[df['date'].isin(w_dates)][[f, 'future_ret']].dropna()
                if len(valid) >= 30:
                    ic = valid[f].corr(valid['future_ret'], method='spearman')
                    window_ics.append(ic)
            if len(window_ics) >= 3:
                ic_sharpe = np.mean(window_ics) / (np.std(window_ics) + 1e-10)
                if ic_sharpe >= min_ic_sharpe:
                    stable.append(f)
        # 至少保留8个特征 — 特征太少反而过拟合
        if len(stable) < 8:
            return features
        return stable

    def _select_top_features(self, df: pd.DataFrame,
                              features: List[str], k: int = 8) -> List[str]:
        """选IC稳定且高的特征，而非仅按|IC|排序。IC_sharpe = mean(IC) / std(IC)"""
        if 'future_ret' not in df.columns:
            return features[:k]
        dates = sorted(df['date'].unique())
        if len(dates) < 5:
            return features[:k]
        # 分5段时间窗口计算IC，取IC_sharpe最高的k个
        n_windows = min(5, len(dates))
        window_size = len(dates) // n_windows
        scores = []
        for f in features:
            if f not in df.columns:
                continue
            window_ics = []
            for w in range(n_windows):
                w_start = w * window_size
                w_end = (w + 1) * window_size if w < n_windows - 1 else len(dates)
                w_dates = set(dates[w_start:w_end])
                valid = df[df['date'].isin(w_dates)][[f, 'future_ret']].dropna()
                if len(valid) >= 30:
                    ic = valid[f].corr(valid['future_ret'], method='spearman')
                    window_ics.append(ic)
            if len(window_ics) >= 3:
                ic_mean = np.mean(window_ics)
                ic_std = np.std(window_ics) + 1e-10
                ic_sharpe = ic_mean / ic_std
                # 综合得分: IC_sharpe为主 + |IC|为辅
                score = ic_sharpe * 0.7 + abs(ic_mean) * 0.3
                scores.append((f, score, ic_mean, ic_sharpe))
        scores.sort(key=lambda x: -x[1])
        return [f for f, _, _, _ in scores[:k]]
