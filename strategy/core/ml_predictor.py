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
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')


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
            'n_jobs': -1,
            'verbosity': 0,
            'early_stopping_rounds': 30,  # XGBoost 3.x: 在构造时传入
        }

        self._use_cross_features = ml_cfg.get('use_cross_features', True)
        self._use_regime_features = ml_cfg.get('use_regime_features', True)
        self.blend_weight = ml_cfg.get('blend_weight', 0.30)
        self._cross_feature_pairs = []  # 训练时固化，预测时复用

    def prepare_features(self, df: pd.DataFrame,
                         regime_info: dict = None,
                         is_train: bool = False) -> pd.DataFrame:
        exclude = {'code', 'date', 'future_ret', 'industry'}
        base_features = [c for c in df.columns if c not in exclude
                         and df[c].dtype in ('float64', 'float32', 'int64', 'int32')]

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
            for f1, f2 in self._cross_feature_pairs:
                col = f'cross_{f1}_{f2}'
                if col not in df.columns:
                    if f1 in df.columns and f2 in df.columns:
                        df[col] = df[f1].fillna(0) * df[f2].fillna(0)
                        if col not in base_features:
                            base_features.append(col)

        if self._use_regime_features and regime_info:
            regime_val = regime_info.get('regime', 0)
            for f in base_features[:10]:
                col = f'regime_{f}'
                if col not in df.columns and f in df.columns:
                    df[col] = df[f].fillna(0) * regime_val
                    base_features.append(col)

        self.feature_cols = [c for c in base_features
                             if c in df.columns and c not in exclude]
        return df

    def train(self, factor_df: pd.DataFrame,
              regime_info: dict = None) -> Optional[float]:
        """训练XGBoost模型，返回验证集IC"""
        try:
            from xgboost import XGBRegressor
        except ImportError:
            print("[ML] xgboost未安装（需 libomp: brew install libomp）")
            return None

        df = self.prepare_features(factor_df.copy(), regime_info, is_train=True)
        if 'future_ret' not in df.columns:
            print("[ML] 训练数据缺少future_ret列")
            return None

        valid_features = [c for c in self.feature_cols if c in df.columns]
        if len(valid_features) < 3:
            print(f"[ML] 有效特征不足: {len(valid_features)}")
            return None

        train_df = df.dropna(subset=valid_features + ['future_ret'])
        if len(train_df) < 500:
            print(f"[ML] 训练样本不足: {len(train_df)}")
            return None

        # 时序划分：前80%训练，后20%验证（防look-ahead）
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

        self.model = XGBRegressor(**self.xgb_params)
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        self._last_train_date = dates[-1]

        # 验证集IC
        y_pred = self.model.predict(X_val)
        val_ic = np.corrcoef(y_pred, y_val)[0, 1] if len(y_val) > 1 else 0

        # 特征重要性
        importances = self.model.feature_importances_
        self._feature_importances = dict(sorted(
            zip(valid_features, importances), key=lambda x: -x[1]
        ))

        print(f"[ML] 训练完成: 样本={len(train_df)}, "
              f"特征={len(valid_features)}, 验证IC={val_ic:.4f}")

        return val_ic

    def predict(self, df: pd.DataFrame,
                regime_info: dict = None) -> np.ndarray:
        if self.model is None:
            return np.zeros(len(df))

        df = self.prepare_features(df.copy(), regime_info)
        valid_features = [c for c in self.feature_cols if c in df.columns]

        if not valid_features:
            return np.zeros(len(df))

        X = df[valid_features].fillna(0).values
        return self.model.predict(X)

    def get_feature_importance(self) -> Dict[str, float]:
        if not hasattr(self, '_feature_importances'):
            return {}
        return self._feature_importances

    def is_trained(self) -> bool:
        return self.model is not None

    def save_model(self, path: str):
        """保存模型到文件（XGBoost原生JSON格式）"""
        if self.model is None:
            return
        self.model.save_model(path)
        # 同时保存元数据
        import pickle
        meta_path = path + '.meta'
        with open(meta_path, 'wb') as f:
            pickle.dump({
                'feature_cols': self.feature_cols,
                'cross_feature_pairs': self._cross_feature_pairs,
                'feature_importances': getattr(self, '_feature_importances', {}),
            }, f)

    def load_model(self, path: str):
        """从文件加载模型"""
        from xgboost import XGBRegressor
        self.model = XGBRegressor()
        self.model.load_model(path)
        # 加载元数据
        import pickle
        meta_path = path + '.meta'
        if __import__('os').path.exists(meta_path):
            with open(meta_path, 'rb') as f:
                data = pickle.load(f)
            self.feature_cols = data['feature_cols']
            self._cross_feature_pairs = data['cross_feature_pairs']
            self._feature_importances = data['feature_importances']

    def _select_top_features(self, df: pd.DataFrame,
                              features: List[str], k: int = 8) -> List[str]:
        if 'future_ret' not in df.columns:
            return features[:k]
        ics = []
        for f in features:
            if f not in df.columns:
                continue
            valid = df[[f, 'future_ret']].dropna()
            if len(valid) >= 30:
                ic = valid[f].corr(valid['future_ret'], method='spearman')
                ics.append((f, abs(ic)))
        ics.sort(key=lambda x: -x[1])
        return [f for f, _ in ics[:k]]
