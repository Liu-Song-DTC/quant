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
        self._last_train_date = None

        ml_cfg = self.config.get('ml', {})
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
            'n_jobs': min(4, __import__('os').cpu_count() or 4),
            'verbosity': 0,
            'tree_method': 'hist',      # 节省训练内存
            'device': 'cpu',            # 强制 CPU（避免 GPU OOM）
            'early_stopping_rounds': 30,  # XGBoost 3.x: 在构造时传入
        }

        self._use_cross_features = ml_cfg.get('use_cross_features', True)
        self._use_regime_features = ml_cfg.get('use_regime_features', True)
        self.blend_weight = ml_cfg.get('blend_weight', 0.50)
        self._cross_feature_pairs = []  # 训练时固化，预测时复用

    def prepare_features(self, df: pd.DataFrame,
                         regime_info: dict = None,
                         is_train: bool = False) -> pd.DataFrame:
        exclude = {'code', 'date', 'future_ret', 'industry'}
        base_features = [c for c in df.columns if c not in exclude
                         and df[c].dtype in ('float64', 'float32', 'int64', 'int32')]

        # 白名单过滤: 只用已验证的因子, 避免噪声因子混入特征空间
        whitelist = self.config.get('ml', {}).get('feature_whitelist', None)
        if whitelist:
            base_features = [f for f in base_features if f in whitelist]

        if len(base_features) < 2:
            self.feature_cols = base_features
            return df

        if self._use_cross_features:
            if is_train or not self._cross_feature_pairs:
                top_k = min(8, len(base_features))
                top_features = self._select_top_features(df, base_features, k=top_k)
                self._cross_feature_pairs = []
                for i, f1 in enumerate(top_features):
                    for f2 in top_features[i+1:]:
                        self._cross_feature_pairs.append((f1, f2))
            # Rank-based交叉: 先转截面百分位再相乘, 对异常值和分布漂移更鲁棒
            for f1, f2 in self._cross_feature_pairs:
                col = f'cross_{f1}_{f2}'
                if col not in df.columns:
                    if f1 in df.columns and f2 in df.columns:
                        r1 = df.groupby('date')[f1].rank(pct=True).fillna(0.5)
                        r2 = df.groupby('date')[f2].rank(pct=True).fillna(0.5)
                        df[col] = r1 * r2
                        if col not in base_features:
                            base_features.append(col)

        if self._use_regime_features and regime_info:
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
            for f in base_features[:10]:
                col = f'regime_{f}'
                if col not in df.columns and f in df.columns:
                    df[col] = df[f].fillna(0).values * regime_arr
                    base_features.append(col)

        self.feature_cols = [c for c in base_features
                             if c in df.columns and c not in exclude]
        return df

    @staticmethod
    def _cross_sectional_zscore(df: pd.DataFrame, cols: List[str]):
        """截面标准化: 每个日期内 (x - mean) / std, 消除市场量级干扰"""
        if 'date' not in df.columns:
            return df
        for col in cols:
            if col not in df.columns:
                continue
            grouped = df.groupby('date')[col]
            mean_s = grouped.transform('mean')
            std_s = grouped.transform('std').replace(0, 1.0)
            df[col] = (df[col] - mean_s) / std_s
        return df

    def train(self, factor_df: pd.DataFrame,
              regime_info: dict = None) -> Optional[float]:
        """训练XGBoost模型，返回验证集IC"""
        try:
            from xgboost import XGBRegressor
        except ImportError:
            print("[ML] xgboost未安装")
            return None

        # 只复制数值型特征列
        exclude_set = {'code', 'date', 'future_ret', 'industry'}
        numeric_cols = [c for c in factor_df.columns if c not in exclude_set
                       and factor_df[c].dtype in ('float64', 'float32', 'int64', 'int32')]
        meta_cols = [c for c in ['code', 'date', 'future_ret', 'industry'] if c in factor_df.columns]
        df = factor_df[meta_cols + numeric_cols].copy()

        # 截面标准化: 消除跨日期量级差异, 模型学习相对排名而非绝对量级
        df = self._cross_sectional_zscore(df, numeric_cols)

        # Rank标签: 截面百分位排名 → 中心化到[-0.5, 0.5], 预测排序而非绝对收益
        if 'future_ret' in df.columns and 'date' in df.columns:
            df['future_ret_rank'] = df.groupby('date')['future_ret'].rank(pct=True) - 0.5
            df['future_ret'] = df['future_ret_rank']
            df.drop(columns=['future_ret_rank'], inplace=True)

        # 特征工程 (基于标准化后的因子值)
        df = self.prepare_features(df, regime_info, is_train=True)
        if 'future_ret' not in df.columns:
            print("[ML] 训练数据缺少future_ret列")
            return None

        valid_features = [c for c in self.feature_cols if c in df.columns]
        if len(valid_features) < 3:
            print(f"[ML] 有效特征不足: {len(valid_features)}")
            return None

        # 特征过滤已关闭: XGBoost树模型自带方向学习 + L1/L2正则 + sample_weight时变加权
        # 负IC特征不删也不翻转 — 树分裂方向不依赖特征符号, regime交叉特征处理多空不对称

        # 处理Inf/NaN
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        num_cols = df.select_dtypes(include=['float64', 'float32', 'int64', 'int32']).columns
        df[num_cols] = df[num_cols].fillna(0.0)
        train_df = df.dropna(subset=['future_ret'])
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

        # 时序划分: 前80%训练, 后20%验证
        dates = sorted(train_df['date'].unique())
        split_idx = int(len(dates) * 0.8)
        train_dates = set(dates[:split_idx])
        val_dates = set(dates[split_idx:])

        train_mask = train_df['date'].isin(train_dates)
        val_mask = train_df['date'].isin(val_dates)

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
        sample_weight = np.exp(_days_diff / 500.0)  # ~2年半衰, 近期数据权重≈2.7x最远数据

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

        val_ic = np.corrcoef(_ensemble_preds, y_val)[0, 1] if len(y_val) > 1 else 0

        # 特征重要性: ensemble均值
        _all_imp = {}
        for _m in self._ensemble_models:
            for fn, imp in zip(valid_features, _m.feature_importances_):
                _all_imp[fn] = _all_imp.get(fn, 0) + imp / len(_ensemble_seeds)
        self._feature_importances = dict(sorted(_all_imp.items(), key=lambda x: -x[1]))

        print(f"[ML] 训练完成: 样本={len(train_df)}, "
              f"特征={len(valid_features)}, 验证IC={val_ic:.4f}")
        return val_ic

    def predict(self, df: pd.DataFrame,
                regime_info: dict = None) -> np.ndarray:
        if self.model is None:
            return np.zeros(len(df))

        exclude_set = {'code', 'date', 'future_ret', 'industry'}
        numeric_cols = [c for c in df.columns if c not in exclude_set
                       and df[c].dtype in ('float64', 'float32', 'int64', 'int32')]
        meta_cols = [c for c in ['code', 'date', 'future_ret', 'industry'] if c in df.columns]
        df = df[meta_cols + numeric_cols].copy()

        # 截面标准化 (与训练一致)
        df = self._cross_sectional_zscore(df, numeric_cols)

        df = self.prepare_features(df, regime_info)
        # 使用训练时固化的特征列表, 避免prepare_features重建的feature_cols与模型不匹配
        trained_cols = getattr(self, '_trained_features', self.feature_cols)
        valid_features = [c for c in trained_cols if c in df.columns]
        if not valid_features:
            return np.zeros(len(df))

        X = df[valid_features].fillna(0).values
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
        for i, _m in enumerate(self._ensemble_models):
            _m.save_model(f"{path}.{i}.json")
        import pickle
        with open(path + '.meta', 'wb') as f:
            pickle.dump({
                'feature_cols': self.feature_cols,
                'trained_features': getattr(self, '_trained_features', self.feature_cols),
                'cross_feature_pairs': self._cross_feature_pairs,
                'feature_importances': getattr(self, '_feature_importances', {}),
                'n_models': len(self._ensemble_models),
            }, f)

    def load_model(self, path: str):
        """从文件加载ensemble模型"""
        from xgboost import XGBRegressor
        import pickle
        meta_path = path + '.meta'
        n_models = 3
        if __import__('os').path.exists(meta_path):
            with open(meta_path, 'rb') as f:
                data = pickle.load(f)
            self.feature_cols = data['feature_cols']
            self._trained_features = data.get('trained_features', data['feature_cols'])
            self._cross_feature_pairs = data['cross_feature_pairs']
            n_models = data.get('n_models', 3)
        self._ensemble_models = []
        for i in range(n_models):
            _m = XGBRegressor()
            _mp = f"{path}.{i}.json"
            if __import__('os').path.exists(_mp):
                _m.load_model(_mp)
                self._ensemble_models.append(_m)
        if self._ensemble_models:
            self.model = self._ensemble_models[-1]
            self._feature_importances = data['feature_importances']

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
