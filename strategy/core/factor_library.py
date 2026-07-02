# core/factor_library.py
"""
因子库 — 因子评估持久化 + 时变质量追踪 + 生命周期管理

核心理念: 因子有时效性。5年回测中因子的IC不是常数，随时间衰减/恢复。
因子库按时间窗口存储IC评估结果，追踪因子质量的时间序列，支持:
- 时间加权因子评分 (近期IC权重 > 远期IC权重)
- 因子衰减检测 (IC trend斜率下降 -> 标记衰减)
- 因子生命周期 (candidate -> active -> decaying -> retired)
- 多窗口评估 (60d/120d/250d, 捕捉不同时间尺度的alpha)

使用方式:
    store = FactorStore("strategy/cache/factor_library.parquet")
    library = FactorLibrary(store)

    # 注册因子
    library.register("mom_quality", family="momentum")

    # 保存评估结果
    store.save_batch(records)

    # 查询时变质量
    quality = library.get_quality("mom_quality", "机械设备", pd.Timestamp("2024-06-15"))

    # 按时间加权评分选因子
    selected = library.select("机械设备", pd.Timestamp("2024-06-15"), top_n=3)
"""

import json
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from collections import defaultdict

# ============================================================
# Factor Store - 持久化层
# ============================================================

_FACTOR_METRICS_DTYPE = {
    'eval_date': 'datetime64[ns]',
    'factor_name': 'str',
    'window_len': 'int32',
    'industry': 'str',
    'ic_mean': 'float32',
    'ir': 'float32',
    'ic_stability': 'float32',
    'ret_spread': 'float32',
    'direction': 'int8',
    'combined_ir': 'float32',
    'coverage': 'float32',
    'n_dates': 'int16',
}

_FACTOR_METRICS_COLS = list(_FACTOR_METRICS_DTYPE.keys())


class FactorStore:
    """因子评估结果持久化 (parquet)."""

    def __init__(self, store_path: str):
        self.store_path = store_path
        self._df: Optional[pd.DataFrame] = None

    @property
    def df(self) -> pd.DataFrame:
        if self._df is None:
            self._df = self._load()
        return self._df

    def _load(self) -> pd.DataFrame:
        if os.path.exists(self.store_path):
            df = pd.read_parquet(self.store_path)
            for col, dtype in _FACTOR_METRICS_DTYPE.items():
                if col in df.columns:
                    try:
                        df[col] = df[col].astype(dtype)
                    except (ValueError, TypeError):
                        pass
            return df
        return pd.DataFrame(columns=_FACTOR_METRICS_COLS)

    def save_batch(self, records: List[dict]):
        """批量保存评估记录. 同 (eval_date, factor_name, window_len, industry) 去重."""
        if not records:
            return
        new_df = pd.DataFrame(records)
        for col, dtype in _FACTOR_METRICS_DTYPE.items():
            if col in new_df.columns:
                try:
                    new_df[col] = new_df[col].astype(dtype)
                except (ValueError, TypeError):
                    pass
        if self._df is not None and len(self._df) > 0:
            self._df = pd.concat([self._df, new_df], ignore_index=True)
            self._df = self._df.drop_duplicates(
                subset=['eval_date', 'factor_name', 'window_len', 'industry'], keep='last')
        else:
            self._df = new_df
        self._df.reset_index(drop=True, inplace=True)
        os.makedirs(os.path.dirname(self.store_path) or '.', exist_ok=True)
        self._df.to_parquet(self.store_path, index=False)

    def query(self, factor_name: str = None, industry: str = None,
              start_date=None, end_date=None, window_len: int = None) -> pd.DataFrame:
        df = self.df
        if df.empty:
            return df
        mask = pd.Series(True, index=df.index)
        if factor_name:
            mask &= df['factor_name'] == factor_name
        if industry:
            mask &= df['industry'] == industry
        if window_len is not None:
            mask &= df['window_len'] == window_len
        if start_date:
            mask &= df['eval_date'] >= pd.Timestamp(start_date)
        if end_date:
            mask &= df['eval_date'] <= pd.Timestamp(end_date)
        return df[mask].sort_values('eval_date')

    def get_timeline(self, factor_name: str, industry: str,
                     window_len: int = 250) -> List[dict]:
        df = self.query(factor_name=factor_name, industry=industry, window_len=window_len)
        if df.empty:
            return []
        return df.sort_values('eval_date').to_dict('records')

    def get_all_factors(self) -> List[str]:
        if self.df.empty:
            return []
        return sorted(self.df['factor_name'].unique().tolist())

    def get_industries(self) -> List[str]:
        if self.df.empty:
            return []
        return sorted(self.df['industry'].unique().tolist())

    def get_latest_eval_date(self) -> Optional[pd.Timestamp]:
        if self.df.empty:
            return None
        return self.df['eval_date'].max()

    @property
    def size(self) -> int:
        return len(self.df) if self._df is not None else (
            len(self._load()) if os.path.exists(self.store_path) else 0)


# ============================================================
# Factor Library - 时变质量分析 + 生命周期管理
# ============================================================

class FactorLibrary:
    """因子知识库: 时间加权评分 / 衰减检测 / 因子选择."""

    def __init__(self, store: FactorStore, config: dict = None):
        self.store = store
        self.config = config or {}
        self.registry: Dict[str, dict] = {}
        self._load_registry()

        # 可配置参数
        self.decay_lookback = self.config.get('decay_lookback', 5)
        self.decay_slope_threshold = self.config.get('decay_slope_threshold', -0.003)
        self.time_weight_halflife = self.config.get('time_weight_halflife', 4)
        self.min_active_ic = self.config.get('min_active_ic', 0.015)
        self.promotion_windows = self.config.get('promotion_windows', 3)
        self.retirement_windows = self.config.get('retirement_windows', 2)

    # -- Registry --

    def _registry_path(self) -> str:
        return self.store.store_path.replace('.parquet', '_registry.json')

    def _load_registry(self):
        path = self._registry_path()
        if os.path.exists(path):
            with open(path, 'r') as f:
                self.registry = json.load(f)

    def _save_registry(self):
        path = self._registry_path()
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.registry, f, indent=2, ensure_ascii=False)

    def register(self, name: str, family: str = 'other'):
        if name in self.registry:
            return
        self.registry[name] = {
            'family': family,
            'status': 'candidate',
            'registered_at': str(pd.Timestamp.now()),
            'promoted_at': None,
            'decay_detected_at': None,
            'retired_at': None,
        }
        self._save_registry()

    def register_batch(self, factor_names: List[str], family_map: dict = None):
        family_map = family_map or {}
        changed = False
        for name in factor_names:
            if name not in self.registry:
                self.registry[name] = {
                    'family': family_map.get(name, 'other'),
                    'status': 'candidate',
                    'registered_at': str(pd.Timestamp.now()),
                    'promoted_at': None,
                    'decay_detected_at': None,
                    'retired_at': None,
                }
                changed = True
        if changed:
            self._save_registry()

    def get_status(self, name: str) -> str:
        return self.registry.get(name, {}).get('status', 'candidate')

    # -- Time-Weighted Quality --

    def get_quality(self, factor_name: str, industry: str,
                    as_of_date, window_len: int = 250) -> Optional[dict]:
        """计算因子在指定日期的时变质量评分.

        时间权重: 指数衰减, 近期窗口权重高.
        时变IC能区分 "IC=0.04且在上升" vs "IC=0.04但在下降" 的因子.
        """
        as_of_ts = pd.Timestamp(as_of_date)
        timeline = self.store.get_timeline(factor_name, industry, window_len)
        if not timeline:
            return None

        past = [t for t in timeline if pd.Timestamp(t['eval_date']) < as_of_ts]
        if not past:
            return None

        recent = past[-self.decay_lookback:]
        n = len(recent)

        weights = np.array([0.5 ** ((n - 1 - i) / self.time_weight_halflife)
                           for i in range(n)])
        weights = weights / weights.sum()

        ic_values = np.array([r['ic_mean'] for r in recent])
        ir_values = np.array([r['ir'] for r in recent])
        spread_values = np.array([r['ret_spread'] for r in recent])

        time_weighted_ic = float(np.sum(ic_values * weights))
        time_weighted_ir = float(np.sum(ir_values * weights))
        time_weighted_spread = float(np.sum(spread_values * weights))
        raw_latest_ic = float(ic_values[-1])
        raw_latest_ir = float(ir_values[-1])

        if n >= 3:
            slope = float(np.polyfit(np.arange(n), ic_values, 1)[0])
        else:
            slope = 0.0

        ic_volatility = float(np.std(ic_values)) if n >= 2 else 0.0

        # 衰减分数: 综合斜率+近端vs历史+波动
        decay_score = float(
            0.5 * (slope / max(abs(raw_latest_ic), 0.001)) +
            0.3 * (1.0 if raw_latest_ic < time_weighted_ic else -0.5) +
            0.2 * (1.0 if ic_volatility < 0.02 else -0.5)
        )

        # 衰减判断用IC量级: |IC|持续下降=预测力衰减
        # 负IC因子(如IC=-0.08)方向取反即可, 量级大说明预测力强
        abs_ic = abs(raw_latest_ic)
        abs_prev_ic = abs(ic_values[-2]) if n >= 2 else 0
        abs_slope = float(np.polyfit(np.arange(n), np.abs(ic_values), 1)[0]) if n >= 3 else 0.0

        if n >= 3 and abs_slope < -0.003 and abs_ic < 0.02:
            status = 'decaying'
        elif n >= 2 and abs_ic < abs_prev_ic * 0.5:
            status = 'decaying'
        elif decay_score < -0.2 and abs_ic < 0.03:
            status = 'decaying'
        elif abs_ic > self.min_active_ic and abs_slope >= -0.001 and n >= 3:
            status = 'active'
        else:
            status = 'candidate'

        return {
            'time_weighted_ic': time_weighted_ic,
            'time_weighted_ir': time_weighted_ir,
            'time_weighted_spread': time_weighted_spread,
            'raw_latest_ic': raw_latest_ic,
            'raw_latest_ir': raw_latest_ir,
            'ic_trend_slope': slope,
            'ic_volatility': ic_volatility,
            'decay_score': decay_score,
            'status': status,
            'n_windows': n,
        }

    # -- Factor Selection --

    def select(self, industry: str, as_of_date, top_n: int = 3,
               window_len: int = 250, min_ic: float = 0.015,
               exclude_decaying: bool = True) -> List[dict]:
        """为指定日期+行业选择最优因子, 使用时间加权评分.

        使用绝对值IC排名(保留符号做方向), 负IC因子通过取反同样有效.
        """
        all_factors = self.store.get_all_factors()
        as_of_ts = pd.Timestamp(as_of_date)
        if not all_factors:
            return []

        candidates = []
        for fn in all_factors:
            q = self.get_quality(fn, industry, as_of_ts, window_len)
            if q is None:
                continue
            if abs(q['raw_latest_ic']) < min_ic:
                continue
            if exclude_decaying and q['status'] in ('decaying', 'retired'):
                continue

            # 用绝对值排名(量级=预测力), 保留符号做方向
            direction = 1 if q['raw_latest_ic'] > 0 else -1
            if q['n_windows'] >= 3:
                score = abs(q['time_weighted_ic']) * 0.6 + abs(q['raw_latest_ic']) * 0.4
                if q['decay_score'] < -0.2:
                    score *= 0.7
            else:
                score = abs(q['raw_latest_ic'])

            candidates.append({
                'factor_name': fn,
                'score': round(score, 5),
                'direction': direction,
                'time_weighted_ic': round(q['time_weighted_ic'], 5),
                'raw_latest_ic': round(q['raw_latest_ic'], 5),
                'ic_trend_slope': round(q['ic_trend_slope'], 5),
                'status': q['status'],
                'decay_score': round(q['decay_score'], 3),
                'family': self.registry.get(fn, {}).get('family', 'other'),
            })

        candidates.sort(key=lambda x: x['score'], reverse=True)

        # 家族分散化
        selected = []
        used_families = set()
        for c in candidates:
            if c['family'] not in used_families and len(selected) < top_n:
                used_families.add(c['family'])
                selected.append(c)
            elif len(selected) >= top_n:
                break
        return selected

    # -- 统一因子选择入口 (YAML种子 → 实盘IC渐进替代) --

    def get_scoring_factors(self, industry: str, as_of_date=None,
                            top_n: int = 5, fallback_config: dict = None,
                            ic_weighted: bool = True
                            ) -> List[tuple]:
        """为评分公式提供最优因子列表 [(factor_name, weight, direction), ...]

        优先级: 实盘IC数据 > YAML种子数据 > 通用因子兜底
        ic_weighted=True: 权重=因子IC得分(不归一化), 好因子自然高权重
        """
        # 1) 尝试从 store 取实盘IC数据
        if as_of_date is not None and self.store.get_all_factors():
            selected = self.select(industry, as_of_date, top_n=top_n)
            if selected:
                if ic_weighted:
                    return [(s['factor_name'], s['score'], s.get('direction', 1))
                            for s in selected]
                else:
                    total_score = sum(s['score'] for s in selected) + 1e-10
                    return [(s['factor_name'], s['score'] / total_score, s.get('direction', 1))
                            for s in selected]

        # 2) YAML种子数据
        if fallback_config and industry in fallback_config:
            cfg = fallback_config[industry]
            factors = cfg.get('factors', [])
            weights = cfg.get('weights', [])
            if factors and weights and len(factors) == len(weights):
                total_w = sum(abs(w) for w in weights) + 1e-10
                return [(fn, abs(w) / total_w, 1) for fn, w in zip(factors, weights)]

        # 3) 通用因子兜底
        default_factors = [
            ('trend_lowvol', 0.30, 1),
            ('relative_strength', 0.25, 1),
            ('low_downside', 0.25, 1),
            ('momentum_reversal', 0.20, 1),
        ]
        return default_factors

    # -- Lifecycle Management --

    def update_lifecycles(self, industry: str = None):
        """扫描所有因子, 更新生命周期状态."""
        all_factors = self.store.get_all_factors()
        industries = [industry] if industry else self.store.get_industries()
        if not all_factors or not industries:
            return

        latest_date = self.store.get_latest_eval_date()
        if latest_date is None:
            return

        for fn in all_factors:
            for ind in industries:
                q = self.get_quality(fn, ind, latest_date + pd.Timedelta(days=1))
                if q is None:
                    continue

                current = self.get_status(fn)
                new_status = current
                recent_ics = self._get_recent_ics(fn, ind, max(self.promotion_windows, self.retirement_windows))

                if current == 'candidate':
                    if len(recent_ics) >= self.promotion_windows:
                        if all(ic > self.min_active_ic for ic in recent_ics[-self.promotion_windows:]) \
                           and q['ic_trend_slope'] >= -0.001:
                            new_status = 'active'

                elif current == 'active':
                    if q['decay_score'] < -0.3:
                        new_status = 'decaying'
                    elif q['raw_latest_ic'] < 0:
                        if len(recent_ics) >= self.retirement_windows \
                           and all(ic < 0 for ic in recent_ics[-self.retirement_windows:]):
                            new_status = 'retired'

                elif current == 'decaying':
                    if q['decay_score'] > 0.1 and q['raw_latest_ic'] > self.min_active_ic:
                        new_status = 'active'
                    elif q['raw_latest_ic'] < 0:
                        if len(recent_ics) >= self.retirement_windows \
                           and all(ic < 0 for ic in recent_ics[-self.retirement_windows:]):
                            new_status = 'retired'

                elif current == 'retired':
                    retired_at = self.registry.get(fn, {}).get('retired_at')
                    if retired_at:
                        try:
                            if latest_date - pd.Timestamp(retired_at) > pd.Timedelta(days=180):
                                if q['raw_latest_ic'] > self.min_active_ic:
                                    new_status = 'candidate'
                        except (ValueError, TypeError):
                            pass

                if new_status != current:
                    self.registry[fn]['status'] = new_status
                    if new_status == 'active':
                        self.registry[fn]['promoted_at'] = str(latest_date)
                    elif new_status == 'decaying':
                        self.registry[fn]['decay_detected_at'] = str(latest_date)
                    elif new_status == 'retired':
                        self.registry[fn]['retired_at'] = str(latest_date)

        self._save_registry()

    def _get_recent_ics(self, factor_name: str, industry: str, n: int) -> List[float]:
        timeline = self.store.get_timeline(factor_name, industry)
        if not timeline:
            return []
        return [r['ic_mean'] for r in timeline[-n:]]

    # -- Summary --

    def get_summary(self, industry: str = None) -> pd.DataFrame:
        """因子库概览: 所有因子的当前状态和评分."""
        latest_date = self.store.get_latest_eval_date()
        if latest_date is None:
            return pd.DataFrame()

        all_factors = self.store.get_all_factors()
        industries = [industry] if industry else self.store.get_industries()
        if not all_factors or not industries:
            return pd.DataFrame()

        rows = []
        for fn in all_factors:
            for ind in industries:
                q = self.get_quality(fn, ind, latest_date + pd.Timedelta(days=1))
                if q is None:
                    continue
                rows.append({
                    'factor_name': fn,
                    'industry': ind,
                    'status': self.get_status(fn),
                    'family': self.registry.get(fn, {}).get('family', 'other'),
                    'time_weighted_ic': q['time_weighted_ic'],
                    'raw_latest_ic': q['raw_latest_ic'],
                    'ic_trend_slope': q['ic_trend_slope'],
                    'decay_score': q['decay_score'],
                    'n_windows': q['n_windows'],
                })

        return pd.DataFrame(rows).sort_values('time_weighted_ic', ascending=False)


# ============================================================
# Factory
# ============================================================

def create_factor_library(store_path: str = None, config: dict = None) -> FactorLibrary:
    if store_path is None:
        strategy_dir = os.path.dirname(os.path.abspath(__file__))
        store_path = os.path.normpath(os.path.join(strategy_dir, '..', 'cache', 'factor_library.parquet'))

    store = FactorStore(store_path)
    library = FactorLibrary(store, config or {})

    if store.size > 0:
        all_factors = store.get_all_factors()
        from .dynamic_factor_selector import get_factor_family
        library.register_batch(all_factors, {fn: get_factor_family(fn) for fn in all_factors})

    return library
