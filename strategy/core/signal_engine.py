# core/signal_engine.py
"""
信号生成引擎

基于行业验证结果的因子配置，支持:
- 行业自适应因子选择
- 市场状态动态权重
- 风格因子调整
- 动态因子选择（Walk-Forward）
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, List, Optional, Tuple
from scipy import stats
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

try:
    from numba import njit
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False
    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]):
            return args[0]
        def wrapper(fn):
            return fn
        return wrapper
    logger.warning("numba 不可用，_ema 等函数将退化为纯Python执行（性能降低）")

from .signal import Signal
from .signal_store import SignalStore
from .config_loader import load_config
from .industry_mapping import INDUSTRY_KEYWORDS, get_industry_category
from .factor_calculator import calculate_indicators as calc_indicators, compute_composite_factors, compress_fundamental_factor
from .multi_timeframe import MultiTimeframeAnalyzer
from .gate_scorer import compute_all_gates, compute_gate_quality
from .bom_chain import compute_stock_bom_score
import yaml
import os

import warnings


from .dynamic_factor_selector import (
    DynamicFactorSelector,
    FACTOR_FAMILIES,
    get_factor_family,
    _compute_date_chunk,
    _compute_date_chunks_worker,
)
# 向后兼容别名
_get_factor_family = get_factor_family

# 行业因子配置（从YAML加载）
def _load_industry_factors():
    """加载行业因子配置"""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base_dir, 'config', 'factor_config.yaml')
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            return config.get('industry_factors', {})
    return {}


def _load_concept_map():
    """加载概念板块映射（与标定对齐，用于因子配置查找）"""
    import pickle
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    map_path = os.path.join(project_root, 'data', 'stock_concept_map.pkl')
    if not os.path.exists(map_path):
        return {}
    with open(map_path, 'rb') as f:
        raw = pickle.load(f)
    # 过滤宽泛标签概念（与标定一致）
    STYLE_KW = ['融资融券', '深股通', '沪股通', '富时罗素', '标准普尔', 'MSCI',
                 '创业板综', '机构重仓', 'QFII', '破增发', '破发股', '昨日高',
                 '中证500', '深成500', '中盘股', '小盘股', '央国企改革',
                 '西部大开发', '年报预增', '专精特新', '上证380', 'HS300',
                 '微盘股', '百元股', '大盘股', '小盘成长', '小盘价值',
                 '转债标的', '长江三角', '深圳特区', '破净股', '创投']
    result = {}
    for code, concepts in raw.items():
        filtered = [c for c in concepts if not any(kw in c for kw in STYLE_KW)]
        if filtered:
            result[code] = filtered
    return result


INDUSTRY_FACTOR_CONFIG = _load_industry_factors()
STOCK_CONCEPT_MAP = _load_concept_map()  # 概念板块→因子配置查找（与标定对齐）


def _safe_get_arr(ind: dict, key: str, n: int, default):
    """向量化版 _safe_get：从ind字典取key对应的numpy数组，不存在时返回填充默认值的数组。"""
    arr = ind.get(key)
    if arr is not None:
        arr = np.asarray(arr)
        if len(arr) == n:
            return arr
    # key不存在或长度不匹配 → 返回填充默认值的数组
    if isinstance(default, (int, float)):
        return np.full(n, default, dtype=np.float64)
    elif isinstance(default, bool):
        return np.full(n, default, dtype=bool)
    return np.full(n, default)


@njit
def _ema(arr, span):
    """指数移动平均 — 模块级 Numba JIT 编译函数"""
    result = np.zeros_like(arr, dtype=float)
    result[0] = arr[0]
    alpha = 2 / (span + 1)
    for i in range(1, len(arr)):
        result[i] = alpha * arr[i] + (1 - alpha) * result[i-1]
    return result


class SignalEngine:
    """信号生成引擎 - 使用行业验证后的高质量因子"""

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.min_score = 0.35

        # 加载配置
        self._load_config()

        # 市场状态信息
        self.market_regime_data = None
        self.current_idx = 0

        # 因子选择统计
        self._stats = {
            'dynamic_success': 0,
            'dynamic_skip_low_ic': 0,  # 低IC行业跳过动态因子
            'dynamic_fallback_fixed': 0,
            'dynamic_fallback_default': 0,
            'dynamic_fallback_none': 0,  # 无高质量因子时不产生信号
            'fixed_industry': 0,
            'fixed_default': 0,
            'ic_values': [],
        }
        # DYN失败原因调试计数器
        self._dyn_fail = {
            'no_code_or_date': 0,
            'no_industry': 0,
            'no_cache_data': 0,
            'no_dates': 0,
            'lookup_fail': 0,       # select_factors_for_date 返回空/行业不匹配
            'low_quality': 0,
            'no_factor_score': 0,   # 因子名解析失败
            'total_calls': 0,
        }

        # 行业内因子值缓存：{industry: deque(maxlen=500)}
        self._industry_factor_cache = {}
        # BOM命中率诊断
        self._bom_diag: Dict[str, Any] = {'total': 0, 'hit': 0, 'miss': 0, 'moat': 0,
                                           'sum_score': 0.0, '_unique_codes': set(),
                                           '_moat_codes': set()}

    def _load_config(self):
        """从配置文件加载参数"""
        config_loader = load_config()

        # 信号阈值（从配置文件加载）
        signal_config = config_loader.get('signal', {})
        self.buy_threshold = signal_config.get('buy_threshold', 0.12)
        self.sell_threshold = signal_config.get('sell_threshold', -0.35)
        self.composite_bias = signal_config.get('composite_bias', 0.0)

        # 卖出趋势保护阈值（防止趋势中途虚假卖出）
        self.trend_sell_threshold_strong = signal_config.get('trend_sell_threshold_strong', -0.15)
        self.trend_sell_threshold_weak = signal_config.get('trend_sell_threshold_weak', -0.10)
        self.trend_sell_ti_threshold = signal_config.get('trend_sell_ti_threshold', 0.05)
        self.trend_sell_ti_relax = signal_config.get('trend_sell_ti_relax', -0.20)

        # Fix#4: 动态阈值百分位 — 从YAML正确读取 (self.config可能为{})
        self._buy_threshold_pct_map = {
            1: signal_config.get('buy_threshold_pct_bull', 0.50),
            0: signal_config.get('buy_threshold_pct', 0.45),  # YAML默认0.45
            -1: signal_config.get('buy_threshold_pct_bear', 0.70),
        }
        self._buy_threshold_pct = self._buy_threshold_pct_map[0]

        # === 三系统共振阈值 ===
        resonance_cfg = signal_config.get('resonance', {})
        self.resonance_sys1_buy_mult = resonance_cfg.get('sys1_buy_threshold_mult', 0.7)
        self.resonance_sys2_cf_threshold = resonance_cfg.get('sys2_cf_score_threshold', 0.5)
        self.resonance_sys3_ns_threshold = resonance_cfg.get('sys3_ns_score_threshold', 0.3)
        self.resonance_sys2_cf_sell_threshold = resonance_cfg.get('sys2_cf_sell_threshold', 0.35)
        self.resonance_sys3_ns_sell_threshold = resonance_cfg.get('sys3_ns_sell_threshold', 0.2)

        # === TI增强 & 自适应融合参数 ===
        self.ti_boost_scale = signal_config.get('ti_boost_scale', 2.0)
        self.ti_boost_magnitude = signal_config.get('ti_boost_magnitude', 0.12)
        self.ti_adaptive_scale = signal_config.get('ti_adaptive_scale', 0.3)
        self.ti_adaptive_max_adjust = signal_config.get('ti_adaptive_max_adjust', 0.12)
        self.signal_confidence_baseline = signal_config.get('signal_confidence_baseline', 0.2)

        # === 因子质量门控 ===
        fqg = signal_config.get('factor_quality_gate', {})
        self.fqg_enabled = fqg.get('enabled', True)
        self.fqg_banned = set(fqg.get('banned_suffixes', ['_T']))
        self.fqg_restricted = fqg.get('restricted_suffixes', {'_FBA': 1.25, '_FV': 1.25, '_F': 1.20})
        self.fqg_premium = set(fqg.get('premium_suffixes', ['_FLA', '_FGR', '_FGRV', '_FLAV', '_FSM', '_FD']))

        # 基本面因子配置
        self.fundamental_enabled = True
        self.fundamental_weight = config_loader.get('fundamental_weight', 0.3)

        # 缠论融合权重: 因子分 vs 缠论分的比例, 0.35 = chan占35%
        self.chan_fusion_weight = config_loader.get('signal.chan_fusion_weight', 0.35)

        # 风格因子开关
        self.style_enabled = config_loader.get('style_factor_enabled', True)

        # 行业因子开关
        industry_config = config_loader.get_industry_factor_config()
        self.industry_factor_enabled = industry_config.get('enabled', True)

        # 技术指标参数
        self.indicator_params = config_loader.get_indicator_params()

        # 动态因子选择器
        self.dynamic_factor_selector = DynamicFactorSelector()

        # === ML预测层 ===
        ml_config = config_loader.get('ml', {})
        self.ml_enabled = ml_config.get('enabled', False)
        self.ml_blend_weight = ml_config.get('blend_weight', 0.30)
        self.ml_predictor = None
        self._ml_predictions = {}  # {(code, date): float}

        # === 缓存 ===
        self._alt_data = None  # 延迟加载
        self._fd_sorted_cache = {}  # code → 按日期排序的基本面DataFrame

        # === 回测诊断 ===
        self._diag = None  # 延迟加载

        # === 缠论增强配置 ===
        chan_config = config_loader.get('chan_theory', {})
        div_config = chan_config.get('divergence', {})
        self.chan_bottom_div_threshold = div_config.get('bottom_div_threshold', 0.3)
        self.chan_top_div_threshold = div_config.get('top_div_threshold', 0.3)

        # 动态因子模式配置：dynamic(仅动态) / fixed(仅固定) / both(动态优先+固定兜底)
        self.factor_mode = config_loader.get('factor_mode', 'both')
        # 兼容两种配置key: fallback_to_fixed 和 fallback_to_static
        self.factor_fallback_to_fixed = config_loader.get('dynamic_factor.fallback_to_fixed',
                                                          config_loader.get('dynamic_factor.fallback_to_static', True))

        # === 动态因子质量阈值 ===
        dyn_cfg = config_loader.get('dynamic_factor', {})
        # DYN质量阈值：对齐缓存层 combined_ir 最低门槛(0.015)→阈值0.015
        # 原0.04*0.5=0.02仍高于缓存层最低门槛0.015, 导致部分合格DYN因子被丢弃
        # DYN质量阈值对齐IC筛选的combined_ir最低门槛(0.007)，避免合格因子被误杀
        self.dyn_quality_threshold = min(
            dyn_cfg.get('min_quality_threshold', 0.04) * 0.5,
            0.007  # 对齐缓存层 combined_ir 最低门槛
        )

        # === Chan增强门控阈值（B1/B2买点）===
        chan_enh_cfg = config_loader.get('chan_theory_enhanced', {})
        b1_cfg = chan_enh_cfg.get('b1_gate', {})
        b2_cfg = chan_enh_cfg.get('b2_gate', {})
        b4_cfg = chan_enh_cfg.get('b4_gate', {})
        bp8_cfg = chan_enh_cfg.get('bp8_gate', {})
        bp7_cfg = chan_enh_cfg.get('bp7_gate', {})
        bp6_cfg = chan_enh_cfg.get('bp6_gate', {})
        bp5_cfg = chan_enh_cfg.get('bp5_gate', {})
        self.chan_b1_min_bottom_div = b1_cfg.get('min_bottom_div', 0.15)
        self.chan_b1_min_fx_vol_spike = b1_cfg.get('min_fx_vol_spike', 1.5)
        self.b1_gate_enabled = b1_cfg.get('enabled', False)
        self.chan_b1_min_signal_level = b1_cfg.get('min_signal_level', 1)
        self.chan_b2_min_div_strength = b2_cfg.get('min_div_strength', 0.10)
        # B4质量门控 (60%信号, 数据验证后开启)
        self.b4_gate_enabled = b4_cfg.get('enabled', False)
        self.chan_b4_min_signal_level = b4_cfg.get('min_signal_level', 1)
        self.chan_b4_min_confidence = b4_cfg.get('min_confidence', 0.30)
        # BP8质量门控
        self.bp8_gate_enabled = bp8_cfg.get('enabled', True)
        self.chan_bp8_min_signal_level = bp8_cfg.get('min_signal_level', 2)
        # BP7质量门控 (质量最高买点但SL=0占17%未过滤)
        self.bp7_gate_enabled = bp7_cfg.get('enabled', True)
        self.chan_bp7_min_signal_level = bp7_cfg.get('min_signal_level', 2)
        # BP6质量门控 (均线回踩, 占10.7%信号, WR=49.6%低于平均)
        self.bp6_gate_enabled = bp6_cfg.get('enabled', True)
        self.chan_bp6_min_signal_level = bp6_cfg.get('min_signal_level', 1)
        self.chan_bp6_min_trend_type = bp6_cfg.get('min_trend_type', 0)  # TT>=0: 盘整或上涨才允许均线回踩
        # B5质量门控 (趋势启动, 占4.3%信号, 待数据验证)
        self.b5_gate_enabled = bp5_cfg.get('enabled', True)
        self.chan_b5_min_signal_level = bp5_cfg.get('min_signal_level', 1)

        # === B3门控配置: 启停控制, 质量条件已改为数据驱动(signal_level+trend_type) ===
        b3_cfg = chan_enh_cfg.get('b3_filter', {})
        self.b3_gate_enabled = b3_cfg.get('enabled', True)

        # === 统一多时间框架分析器 ===
        self.mtf_analyzer = MultiTimeframeAnalyzer(config_loader.config if config_loader.config else {})
        mtf_cfg = config_loader.get('multi_timeframe', {})
        self.mtf_blend_strength = mtf_cfg.get('discount', {}).get('blend_strength', 0.4)

    def set_factor_data(self, factor_df: pd.DataFrame):
        """设置因子数据（用于动态因子选择）

        Args:
            factor_df: DataFrame with columns [code, date, factor1, factor2, ..., future_ret]
        """
        self.dynamic_factor_selector.set_factor_data(factor_df)

    def set_industry_mapping(self, industry_codes: Dict[str, List[str]]):
        """设置行业映射（用于动态因子选择）

        Args:
            industry_codes: {category: [stock_codes]}
        """
        self.dynamic_factor_selector.set_industry_mapping(industry_codes)
        # 同时保存到 SignalEngine 自身，方便 _select_factor_dynamic 访问
        self.industry_codes = industry_codes

    def set_fundamental_data(self, fundamental_data):
        """设置基本面数据"""
        self.fundamental_data = fundamental_data

    def set_ml_predictor(self, predictor):
        """设置ML预测器"""
        self.ml_predictor = predictor

    def set_ml_predictions(self, predictions: dict):
        """设置预计算的ML预测值 {(code, date): float}，构建前向填充缓存"""
        self._ml_predictions = predictions
        # 构建前向填充缓存: code → (sorted_dates_array, values_array)
        from collections import defaultdict
        code_groups = defaultdict(list)
        for (c, d), v in predictions.items():
            code_groups[str(c).zfill(6)].append((d, v))
        self._ml_cache = {}
        for c, items in code_groups.items():
            items.sort(key=lambda x: x[0])
            self._ml_cache[c] = (
                np.array([d for d, _ in items]),
                np.array([v for _, v in items])
            )

    def set_market_regime(self, regime_df: pd.DataFrame):
        """设置市场状态数据"""
        if 'datetime' in regime_df.columns:
            regime_df = regime_df.copy()
            regime_df['datetime'] = pd.to_datetime(regime_df['datetime'])
            self.market_regime_data = regime_df.set_index('datetime')

    def print_factor_stats(self):
        """打印因子选择统计"""
        stats = self._stats
        total = sum(v for k, v in stats.items() if k != 'ic_values')
        if total == 0:
            print("\n因子选择统计: 无数据（worker 统计未合并）")
        else:
            print("\n========== 因子选择统计 ==========")
            print(f"动态因子成功:     {stats['dynamic_success']:6d} ({100*stats['dynamic_success']/total:.1f}%)")
            print(f"动态跳过(低IC):   {stats['dynamic_skip_low_ic']:6d} ({100*stats['dynamic_skip_low_ic']/total:.1f}%)")
            print(f"动态->固定fallback: {stats['dynamic_fallback_fixed']:6d} ({100*stats['dynamic_fallback_fixed']/total:.1f}%)")
            print(f"动态->默认fallback: {stats['dynamic_fallback_default']:6d} ({100*stats['dynamic_fallback_default']/total:.1f}%)")
            print(f"动态->无信号: {stats['dynamic_fallback_none']:6d} ({100*stats['dynamic_fallback_none']/total:.1f}%)")
            print(f"固定行业因子:    {stats['fixed_industry']:6d} ({100*stats['fixed_industry']/total:.1f}%)")
            print(f"固定默认因子:    {stats['fixed_default']:6d} ({100*stats['fixed_default']/total:.1f}%)")
            print(f"总计:            {total}")
        # 打印 _dyn_fail 诊断
        df = self._dyn_fail
        if df.get('total_calls', 0) > 0:
            tc = df['total_calls']
            print(f"\n--- 动态因子失败诊断 (总调用 {tc}) ---")
            for key in ['no_code_or_date', 'no_industry', 'no_cache_data', 'no_dates',
                         'lookup_fail', 'low_quality', 'no_factor_score']:
                v = df.get(key, 0)
                print(f"  {key}: {v:6d} ({100*v/max(tc,1):.1f}%)")
        print("==================================\n")
        if df.get('total_calls', 0) > 0:
            tc = df['total_calls']
            success = tc - df['no_code_or_date'] - df['no_industry'] - df['no_cache_data'] - df['no_dates'] - df['lookup_fail'] - df['low_quality'] - df['no_factor_score']
            print(f"  成功:            {success:5d} ({100*success/max(tc,1):.1f}%)")
        # DYN lookup 失败行业分布
        if hasattr(self, '_dyn_lookup_miss_by_ind') and self._dyn_lookup_miss_by_ind:
            print(f"\n--- DYN lookup_fail 按行业 (前15) ---")
            for ind, cnt in sorted(self._dyn_lookup_miss_by_ind.items(), key=lambda x: -x[1])[:15]:
                print(f"  {ind}: {cnt:>8,}")
        print("==================================\n")

    def print_bom_stats(self):
        """打印BOM产业链命中率诊断"""
        d = self._bom_diag
        total = d['total']
        if total == 0:
            print("[BOM] 无数据")
            return
        n_codes = len(d['_unique_codes'])
        hit_pct = 100 * d['hit'] / max(total, 1)
        moat_pct = 100 * d['moat'] / max(total, 1)
        moat_codes_n = len(d['_moat_codes'])
        avg_score = d['sum_score'] / max(total, 1)
        print(f"\n========== BOM产业链命中率诊断 ==========")
        print(f"BOM总调用:    {total:6d} (去重股票: {n_codes})")
        print(f"命中(>0.30):  {d['hit']:6d} ({hit_pct:.1f}%)")
        print(f"未命中(0.30): {d['miss']:6d} ({100-hit_pct:.1f}%)")
        print(f"moat命中:     {d['moat']:6d} ({moat_pct:.1f}%, 去重: {moat_codes_n})")
        print(f"平均BOM分:    {avg_score:.4f}")
        print("==========================================\n")

    def generate(self, code: str, market_data: pd.DataFrame, signal_store: SignalStore,
                 latest_only: bool = False):
        """生成信号（向量化批处理：收集标量→数组装配→向量化阈值→买卖判定）

        latest_only=True 时采用混合路径：
        - 所有 bar 用快速默认因子计算分数（用于动态阈值 quantile 估计）
        - 仅最新 bar 走完整 _select_factor 链
        - 信号构造也只处理最新 bar
        用于实盘选股，回测不可用（需要全部历史信号）。
        Chan结构增强由 gate_scorer Gate 1 在向量化装配中处理。
        """
        dates = market_data["datetime"].values
        close = market_data['close'].values

        if len(market_data) < 60:
            return

        indicators = self._calculate_indicators(market_data)
        # 注入真实题材热度 (逐bar计算，避免前视偏差)
        try:
            from .concept_heat import get_calculator
            calc = get_calculator()
            n = len(indicators.get('close', []))
            heat_arr = np.full(n, 0.5)
            last_date = None
            for i in range(60, n):  # 从第60根bar开始(最小技术指标窗口)
                bar_date = dates[i]
                if bar_date != last_date:
                    calc.set_daily_data(bar_date)
                    last_date = bar_date
                heat_arr[i] = calc.get_concept_heat(code)
            indicators['concept_heat'] = heat_arr
        except Exception as e:
            logger.warning(f"[ConceptHeat] {code}: 题材热度注入失败 ({e}), 使用默认0.5")
        self._preload_stock_fundamentals(code, dates)

        n = len(close)

        # 向量化预计算市场状态（避免逐 bar _get_market_info）
        regimes = self._precompute_regimes(dates, n)

        # ===== Phase 1: 逐bar收集复杂方法调用的标量结果 =====
        scalars = self._collect_bar_scalars(indicators, code, dates, n,
                                            regimes=regimes, latest_only=latest_only)

        # ===== Phase 2: 向量化分数装配 =====
        result = self._vectorized_score_assembly(scalars, indicators, n, code, dates)
        del scalars

        # ===== Phase 3: 动态阈值（用 adjusted_score 校准入阈值分布，与买入判定一致）=====
        buy_thresholds, sell_thresholds = self._compute_dynamic_thresholds(
            result['adjusted_score'], result['risk_regime']
        )

        # ===== Phase 4: 买卖判定 + 构造Signal对象 =====
        valid = result['valid']
        close_arr = _safe_get_arr(indicators, 'close', n, 0.0)
        ma20_arr = _safe_get_arr(indicators, 'ma20', n, 0.0)
        bottom_div_arr = _safe_get_arr(indicators, 'bottom_divergence', n, 0.0)
        top_div_arr = _safe_get_arr(indicators, 'top_divergence', n, 0.0)

        bar_indices = [n - 1] if latest_only else range(n)
        for i in bar_indices:
            date = pd.to_datetime(dates[i]).date()

            if i < 60 or not valid[i]:
                # 无效bar返回空信号
                sig = Signal(
                    buy=False, sell=False, score=0.0, factor_value=0.0,
                    factor_name='V41' if i < 60 else 'NONE',
                    risk_vol=0.03, risk_regime=int(result['risk_regime'][i]),
                    risk_confidence=0.0, risk_extreme=bool(result['risk_extreme'][i]),
                    adjusted_score=0.0, pre_discount_score=0.0,
                    industry=result['industry'][i] if code else '',
                    exhaustion_risk=0.0, gap_breakout_confirm=0.0,
                    vol_opening_confirm=0.0, vol_opening_strength=0.0,
                    bom_quality_score=0.3,
                    stroke_phase=0.0, top_fractal_volume=0.0,
                    ma_trend_up=False, profit_declining=False,
                    chan_buy_strength=0.0, chan_sell_strength=0.0,
                    b3_trend_confirmed=False,
                    mom_60d=0.0, dist_ma60=0.0, max_dd_20d=0.0, vol_regime=1.0,
                    weekly_trend_up=False, monthly_trend_up=False,
                    weekly_trend_strength=0.0, monthly_trend_strength=0.0,
                    mtf_alignment_score=0.0, mtf_discount_factor=1.0,
                    weekly_pattern_signal=0.0, nearest_resistance_pct=0.0,
                    nearest_support_pct=0.0,
                    _chan_buy_signal=False, _chan_sell_signal=False, _dist_ma20=0.0,
                    _gate_quality=0.5,
                )
                signal_store.set(code, date, sig)
                continue

            # === 买入判定（Gate主导 + Factor排序，无绝对阈值） ===
            buy_th = float(buy_thresholds[i])
            sell_th = float(sell_thresholds[i])
            chan_buy_sig = bool(result['_chan_buy_signal'][i])
            chan_sell_sig = bool(result['_chan_sell_signal'][i])
            bp_buy = int(result['chan_buy_point'][i])
            sl = int(result['signal_level'][i])
            dist_ma20 = float(result['_dist_ma20'][i])
            hard_reject = bool(result['_hard_rejects'][i])

            # 使用门控调整后的分数
            gate_score = float(result['adjusted_score'][i])
            score = gate_score

            vol_oc = float(_safe_get_arr(indicators, 'vol_opening_confirm', n, 0.0)[i])
            vol_os = float(_safe_get_arr(indicators, 'vol_opening_strength', n, 0.0)[i])
            close_p = float(close_arr[i])
            ma20_v = float(ma20_arr[i])

            ma60_v = float(_safe_get_arr(indicators, 'ma60', n, 0.0)[i])
            price_above_ma60 = close_p > 0 and ma60_v > 0 and close_p > ma60_v
            price_above_ma20 = close_p > 0 and ma20_v > 0 and close_p > ma20_v

            if code.startswith('688'):
                price_ok = price_above_ma20
            else:
                # 原三重条件(>MA20+>MA60+MA20>MA60)淘汰太多股票
                # 仅要求>MA60防止深熊, 均值回归分析: 低动量+距MA60近=最佳
                price_ok = price_above_ma60

            # B1/B3 乖离容忍度（结构越好，允许离MA20越远）
            b1_strong = (bp_buy == 1 and chan_buy_sig and sl >= 2)
            b1_weak = (bp_buy == 1 and chan_buy_sig and sl >= 1)
            is_b3 = (bp_buy == 3 and chan_buy_sig)
            is_b6 = (bp_buy == 6)  # 均线回踩, 必须在均线附近
            if is_b3 and sl >= 2:
                max_dist = 0.40
            elif is_b3:
                max_dist = 0.35
            elif b1_strong:
                max_dist = 0.30
            elif b1_weak:
                max_dist = 0.25
            elif is_b6:
                max_dist = 0.12  # 均线回踩必须贴近均线
            else:
                max_dist = 0.30
            price_not_extended = dist_ma20 < max_dist

            # 买入条件：无硬拒绝 · 分数有效 · 价格OK · 未过度乖离 · 有缠论结构
            # Gate保证质量 → 因子评分在池内做区分
            _regime_i = int(result['risk_regime'][i])
            # 纯因子评分范围~[-0.5,0.2], mean=-0.17, threshold设为截面中上水平
            _abs_score_floor = {1: -0.05, 0: -0.02, -1: 0.0}.get(_regime_i, -0.05)
            _score_ok = (score >= _abs_score_floor)
            _dt_sig = float(result['_dt_signal'][i]) if '_dt_signal' in result else 0.0
            struct_ok = (bp_buy >= 1) or (_dt_sig > 0.3)
            buy = (not hard_reject and not np.isnan(score) and _score_ok and
                   price_ok and price_not_extended and struct_ok)

            # 回测诊断: 记录买入信号
            if buy:
                try:
                    if self._diag is None:
                        from analysis.backtest_diagnostics import get_diagnostics
                        self._diag = get_diagnostics()
                    fn = str(result['factor_name'][i]) if 'factor_name' in result else ''
                    self._diag.record_buy_signal(bp_buy, fn)
                except Exception as e:
                    import traceback; print(f"[ERR] " + __file__ + ":" + str(e)); traceback.print_exc()

            # === 下跌趋势硬过滤: trend_type==-2禁止新买入 ===
            _trend_type = int(result['trend_type'][i])
            _is_b1 = (bp_buy == 1 and chan_buy_sig and sl >= 1)
            if buy and _trend_type == -2 and not _is_b1:
                buy = False

            # === 因子质量门控: 禁止低质量因子生成买入信号 ===
            if buy and self.fqg_enabled:
                is_star = code.startswith('688')

                _raw_fn = str(result['factor_name'][i])
                if not is_star and _raw_fn in ('V41', 'NONE'):
                    buy = False
                elif not is_star:
                    _has_fund = bool(result['has_fundamental'][i])
                    _has_style = bool(float(result['style_confidence'][i]) > 0.3)
                    _style_code = str(result['style_regime'][i]).replace('_','')[:2].upper() if _has_style else ''
                    _has_sm = bool(float(result['smart_money'][i]) > 0.15)
                    _has_div = bool(float(bottom_div_arr[i]) > 0.3 or float(top_div_arr[i]) > 0.3)
                    _n_resonance = int(result['n_buy_systems'][i])

                    _tag = ''
                    if _has_fund:
                        _tag += '_F'
                    if _has_style and _style_code:
                        _tag += _style_code
                    if _has_sm:
                        _tag += 'V'
                    if _has_div:
                        _tag += 'D'
                    if _n_resonance >= 2:
                        _tag += f'R{_n_resonance}'
                    if not _tag:
                        _tag = '_T'

                    _is_banned = _tag in self.fqg_banned
                    if _is_banned:
                        buy = False
                    else:
                        _is_premium = _tag in self.fqg_premium
                        if _is_premium:
                            # premium因子: 门槛打8折
                            if score < buy_th * 0.80:
                                buy = False
                        else:
                            _restricted_mult = 1.0
                            for _rs, _rm in self.fqg_restricted.items():
                                if _rs in _tag:
                                    _restricted_mult = max(_restricted_mult, _rm)
                            if _restricted_mult > 1.0:
                                if score < buy_th * _restricted_mult:
                                    buy = False

            # 趋势确认已由 Gate 4 (Trend Direction) 处理

            # 妖股保护: 涨停股忽略缠论卖出信号（暴力拉升中的顶背离是假信号）
            _daily_ret = float(result['daily_return'][i])
            _limit_pct = 0.195 if code.startswith('688') or code.startswith('300') else 0.095
            _is_limit_up_stock = (_daily_ret >= _limit_pct)
            _is_limit_down_stock = (_daily_ret <= -_limit_pct)
            chan_force_sell = chan_sell_sig and sl <= -2 and not _is_limit_up_stock
            # 跌停保护: 跌停股禁止买入
            if _is_limit_down_stock:
                buy = False

            # 龙虎榜独立买点: 机构大买(>0.3) + SL>=2门控(需清晰结构确认)
            if _dt_sig > 0.3 and not hard_reject and not _is_limit_down_stock:
                _bp0_sl = int(result['signal_level'][i])
                if _bp0_sl >= 2:
                    buy = True

            # === B3 质量门控: SL=0 WR=32%, TT=-2 WR=8% (数据驱动) ===
            if buy and bp_buy == 3 and self.b3_gate_enabled:
                _b3_sl = int(result['signal_level'][i])
                _b3_tt = int(result['trend_type'][i])
                if _b3_sl < 1 or _b3_tt < 0:
                    buy = False

            # === B2 质量门控 (基于divergence_strength, IC=+0.031) ===
            if buy and bp_buy == 2:
                _b2_div = float(result['chan_divergence_strength'][i])
                if _b2_div < self.chan_b2_min_div_strength:
                    buy = False

            # === BP8 质量门控: signal_level=0胜率仅39.1%, 过滤无结构突破 ===
            if buy and bp_buy == 8 and self.bp8_gate_enabled:
                _b8_sl = int(result['signal_level'][i])
                if _b8_sl < self.chan_bp8_min_signal_level:
                    buy = False

            # === B4 质量门控: SL>=2 + trend_type>=0 (TT=-2的670个信号WR=37%污染) ===
            if buy and bp_buy == 4 and self.b4_gate_enabled:
                _b4_sl = int(result['signal_level'][i])
                _b4_conf = float(result['buy_confidence'][i]) if 'buy_confidence' in result else 0.35
                _b4_tt = int(result['trend_type'][i])
                if _b4_sl < self.chan_b4_min_signal_level or _b4_conf < self.chan_b4_min_confidence or _b4_tt < 0:
                    buy = False

            # === B1 质量门控: SL=0 WR=40% MR=-1.07%, SL>=2 WR=50.8% (数据驱动) ===
            if buy and bp_buy == 1 and self.b1_gate_enabled:
                _b1_sl = int(result['signal_level'][i])
                if _b1_sl < self.chan_b1_min_signal_level:
                    buy = False

            # === BP7 质量门控: 质量最高买点(WR=55.6%)但SL=0占17%(WR=42%/MR=-1.19%)未过滤 ===
            if buy and bp_buy == 7 and self.bp7_gate_enabled:
                _b7_sl = int(result['signal_level'][i])
                if _b7_sl < self.chan_bp7_min_signal_level:
                    buy = False

            # === BP6 质量门控: 均线回踩, 占10.7%信号, WR=49.6%低于平均 ===
            if buy and bp_buy == 6 and self.bp6_gate_enabled:
                _b6_sl = int(result['signal_level'][i])
                _b6_tt = int(result['trend_type'][i])
                _b6_reject = _b6_sl < self.chan_bp6_min_signal_level
                if self.chan_bp6_min_trend_type > -2:
                    _b6_reject = _b6_reject or _b6_tt < self.chan_bp6_min_trend_type
                if _b6_reject:
                    buy = False

            # === B5 质量门控: 趋势启动, 占4.3%信号, 待数据验证 ===
            if buy and bp_buy == 5 and self.b5_gate_enabled:
                _b5_sl = int(result['signal_level'][i])
                if _b5_sl < self.chan_b5_min_signal_level:
                    buy = False

            # MA60止损: 跌破MA60且score转负 → 强制卖出
            ma60_stop = (close_p < ma60_v) and score < 0 and close_p > 0 and ma60_v > 0

            # 卖出：分数低于动态阈值强制卖, 或分数转负+确认信号可卖
            gate_q = float(result['_gate_quality'][i])
            sell_confirmed = (chan_sell_sig and sl <= -1) or ma60_stop
            sell_confirmed = sell_confirmed or (gate_q < 0.75 and score < 0)
            sell_confirmed = sell_confirmed or (score < -0.10 and gate_q < 1.0)
            sell = not np.isnan(score) and (score < sell_th or (score < 0 and sell_confirmed))

            # 因子标签
            factor_tags = []
            if result['has_fundamental'][i]:
                factor_tags.append('F')
            if float(result['style_confidence'][i]) > 0.3:
                factor_tags.append(str(result['style_regime'][i])[:2].upper())
            if float(result['smart_money'][i]) > 0.15:
                factor_tags.append('V')
            if float(bottom_div_arr[i]) > 0.3 or float(top_div_arr[i]) > 0.3:
                factor_tags.append('D')
            n_buy = int(result['n_buy_systems'][i])
            if n_buy >= 2:
                factor_tags.append(f'R{n_buy}')
            fn = result['factor_name'][i]
            fn = fn + ('_' + ''.join(factor_tags) if factor_tags else '_T')

            sig = Signal(
                buy=buy, sell=sell, score=score,
                factor_value=float(result['factor_value'][i]),
                factor_name=fn,
                industry=str(result['industry'][i]) if result['industry'][i] else '',
                risk_vol=float(result['risk_vol'][i]),
                risk_regime=int(result['risk_regime'][i]),
                risk_confidence=float(result['risk_confidence'][i]),
                risk_extreme=bool(result['risk_extreme'][i]),
                adjusted_score=float(result['adjusted_score'][i]),
                pre_discount_score=float(result['pre_discount_score'][i]),
                ml_score=float(result['ml_score'][i]) if 'ml_score' in result else 0.0,
                factor_score=float(result['factor_score'][i]) if 'factor_score' in result else float(result['pre_discount_score'][i]),
                factor_quality=float(result['factor_quality'][i]),
                signal_confidence=float(result['signal_confidence'][i]),
                chan_divergence_type=str(result['chan_divergence_type'][i]),
                chan_divergence_strength=float(result['chan_divergence_strength'][i]),
                chan_structure_score=float(result['chan_structure_score'][i]),
                chan_buy_point=int(result['chan_buy_point'][i]),
                chan_sell_point=int(result['chan_sell_point'][i]),
                signal_level=int(result['signal_level'][i]),
                chan_buy_strength=float(result['chan_buy_strength'][i]),
                chan_sell_strength=float(result['chan_sell_strength'][i]),
                b3_trend_confirmed=bool(result['b3_trend_confirmed'][i]),
                resonance_systems=int(result['resonance_systems'][i]),
                capital_flow_score=float(result['capital_flow_score'][i]),
                news_sentiment_score=float(result['news_sentiment_score'][i]),
                trend_type=int(result['trend_type'][i]),
                chan_pivot_zg=float(result['chan_pivot_zg'][i]),
                chan_pivot_zd=float(result['chan_pivot_zd'][i]),
                chan_pivot_zz=float(result['chan_pivot_zz'][i]),
                daily_return=float(result['daily_return'][i]),
                volume_ratio=float(result['volume_ratio'][i]),
                volume_ratio_raw=float(result['volume_ratio_raw'][i]),
                exhaustion_risk=float(result['exhaustion_risk'][i]),
                gap_breakout_confirm=float(result['gap_breakout_confirm'][i]),
                vol_opening_confirm=float(vol_oc),
                vol_opening_strength=float(vol_os),
                bom_quality_score=float(self._get_bom_score(code)),
                stroke_phase=float(result['stroke_phase'][i]),
                top_fractal_volume=float(result['top_fractal_volume'][i]),
                ma_trend_up=bool(result['ma_trend_up'][i]),
                profit_declining=bool(result['profit_declining'][i]),
                mom_60d=float(result['mom_60d'][i]),
                dist_ma60=float(result['dist_ma60'][i]),
                max_dd_20d=float(result['max_dd_20d'][i]),
                vol_regime=float(result['vol_regime'][i]),
                weekly_trend_up=bool(result['weekly_trend_up'][i]),
                monthly_trend_up=bool(result['monthly_trend_up'][i]),
                weekly_trend_strength=float(result['weekly_trend_strength'][i]),
                monthly_trend_strength=float(result['monthly_trend_strength'][i]),
                mtf_alignment_score=float(result['mtf_alignment_score'][i]),
                mtf_discount_factor=float(result['mtf_discount_factor'][i]),
                weekly_pattern_signal=float(result['weekly_pattern_signal'][i]),
                nearest_resistance_pct=float(result['nearest_resistance_pct'][i]),
                nearest_support_pct=float(result['nearest_support_pct'][i]),
                _chan_buy_signal=chan_buy_sig,
                _chan_sell_signal=chan_sell_sig,
                _dist_ma20=dist_ma20,
                _gate_quality=float(result['_gate_quality'][i]),
            )
            signal_store.set(code, date, sig)

        # 显式释放中间数组，避免大规模循环中GC延迟导致内存堆积
        del result, close_arr, ma20_arr, bottom_div_arr, top_div_arr
        del buy_thresholds, sell_thresholds, valid
        del indicators, regimes
        self._gc_counter = getattr(self, '_gc_counter', 0) + 1
        if self._gc_counter % 100 == 0:
            import gc
            gc.collect()

    def _calculate_indicators(self, data: pd.DataFrame) -> dict:
        """计算技术指标（委托给factor_calculator）"""
        params = self.indicator_params
        dates = data['datetime'].values
        close = data['close'].values
        high = data['high'].values
        low = data['low'].values
        volume = data['volume'].values
        turnover_rate = data['turnover_rate'].values if 'turnover_rate' in data.columns else None
        open_price = data['open'].values if 'open' in data.columns else None
        amplitude = data['amplitude'].values if 'amplitude' in data.columns else None

        # 使用统一的因子计算器
        result = calc_indicators(close, high, low, volume, params,
                                 turnover_rate=turnover_rate, open_arr=open_price,
                                 amplitude=amplitude)

        # 收益率
        result['ret_30'] = close / self._shift(close, 30) - 1

        # ATR比率
        result['atr_ratio_20'] = result['atr_20'] / (close + 1e-10)

        # ── 预信号价格特征（均值回归过滤） ──
        # 1. 60日动量
        result['mom_60d'] = close / self._shift(close, 60) - 1

        # 2. 距MA60偏离
        ma60 = self._rolling_mean(close, 60)
        result['dist_ma60'] = (close - ma60) / (ma60 + 1e-10)

        # 3. 20日最大回撤
        result['max_dd_20d'] = self._compute_max_dd(close, 20)

        # 4. 波动率区间 (10日vol / 60日vol)
        ret_1d = close / self._shift(close, 1) - 1
        vol_10d = self._rolling_std(ret_1d, 10)
        vol_60d = self._rolling_std(ret_1d, 60)
        result['vol_regime'] = vol_10d / (vol_60d + 1e-10)

        # 5. 统一多时间框架分析（替代旧版固定步长降采样）
        open_arr = open_price if open_price is not None else close
        mtf_result = self.mtf_analyzer.analyze(
            close, high, low, open_arr, volume, dates
        )
        result['weekly_trend_up'] = mtf_result.weekly_trend_up
        result['monthly_trend_up'] = mtf_result.monthly_trend_up
        result['weekly_trend_strength'] = mtf_result.weekly_trend_strength
        result['monthly_trend_strength'] = mtf_result.monthly_trend_strength
        result['mtf_alignment_score'] = mtf_result.alignment_score
        result['mtf_discount_factor'] = mtf_result.discount_factor
        result['weekly_pattern_signal'] = mtf_result.weekly_pattern_signal
        result['nearest_resistance_pct'] = mtf_result.nearest_resistance_pct
        result['nearest_support_pct'] = mtf_result.nearest_support_pct

        # 保存close数组（供信号生成中的均线位置过滤使用）
        result['close'] = close

        return result

    def _precompute_regimes(self, dates: np.ndarray, n: int) -> dict:
        """向量化批量获取市场状态，替代逐 bar _get_market_info

        Returns dict of arrays with keys: regime, confidence, trend_score, volatility,
        is_extreme, style_regime, style_score, style_confidence, bear_risk.
        Optimized for speed: single reindex pass + fallback defaults.
        """
        result = {
            'regime': np.zeros(n, dtype=int),
            'confidence': np.zeros(n),
            'trend_score': np.zeros(n),
            'volatility': np.full(n, 0.15),
            'is_extreme': np.zeros(n, dtype=bool),
            'style_regime': np.full(n, 'balanced'),
            'style_score': np.zeros(n),
            'style_confidence': np.zeros(n),
            'bear_risk': np.zeros(n, dtype=bool),
        }
        if self.market_regime_data is None or len(self.market_regime_data) == 0:
            return result

        dt_index = pd.DatetimeIndex(dates)
        mrd = self.market_regime_data
        for col, default in [('regime', 0), ('confidence', 0.0), ('trend_score', 0.0),
                              ('volatility', 0.15), ('style_score', 0.0), ('style_confidence', 0.0)]:
            if col in mrd.columns:
                aligned = mrd[col].reindex(dt_index, method='ffill')
                result[col] = aligned.fillna(default).values

        if 'is_extreme' in mrd.columns:
            result['is_extreme'] = mrd['is_extreme'].reindex(dt_index, method='ffill').fillna(False).astype(bool).values
        if 'bear_risk' in mrd.columns:
            result['bear_risk'] = mrd['bear_risk'].reindex(dt_index, method='ffill').fillna(False).astype(bool).values
        if 'style_regime' in mrd.columns:
            result['style_regime'] = mrd['style_regime'].reindex(dt_index, method='ffill').fillna('balanced').values

        return result

    def _get_market_info(self, date) -> Dict[str, Any]:
        """获取指定日期的市场状态信息"""
        if self.market_regime_data is None:
            return {
                'regime': 0, 'confidence': 0.0, 'momentum_score': 0.0,
                'trend_score': 0.0, 'volatility': 0.15, 'is_extreme': False,
                'style_regime': 'balanced', 'style_score': 0.0,
                'size_score': 0.0, 'style_confidence': 0.0,
                'bear_risk': False,
            }

        dt = pd.to_datetime(date)
        if dt in self.market_regime_data.index:
            row = self.market_regime_data.loc[dt]
            return {
                'regime': int(row.get('regime', 0)),
                'confidence': float(row.get('confidence', 0.0)),
                'momentum_score': float(row.get('momentum_score', 0.0)),
                'trend_score': float(row.get('trend_score', 0.0)),
                'volatility': float(row.get('volatility', 0.15)),
                'is_extreme': bool(row.get('is_extreme', False)),
                'style_regime': str(row.get('style_regime', 'balanced')),
                'style_score': float(row.get('style_score', 0.0)),
                'size_score': float(row.get('size_score', 0.0)),
                'style_confidence': float(row.get('style_confidence', 0.0)),
                'bear_risk': bool(row.get('bear_risk', False)),
            }
        return {
            'regime': 0, 'confidence': 0.0, 'momentum_score': 0.0,
            'trend_score': 0.0, 'volatility': 0.15, 'is_extreme': False,
            'style_regime': 'balanced', 'style_score': 0.0,
            'size_score': 0.0, 'style_confidence': 0.0,
            'bear_risk': False,
        }

    # _get_chan_boost removed — replaced by gate_scorer Gate 1 + vectorized assembly

    # _calc_b3_multiplier removed — was only called by deleted _get_chan_boost

    def _get_bom_score(self, code: str) -> float:
        """获取股票的BOM质量分(带缓存，含基本面对齐)"""
        if not hasattr(self, '_bom_cache'):
            self._bom_cache = {}
        if code in self._bom_cache:
            return self._bom_cache[code]
        d = self._bom_diag
        d['total'] += 1
        d['_unique_codes'].add(code)
        try:
            from .concept_heat import get_calculator
            calc = get_calculator()
            # 构建基本面查询（roe/gross_margin/market_cap）
            fund = None
            if hasattr(self, 'fundamental_data') and self.fundamental_data:
                try:
                    fd = self.fundamental_data
                    # fundamental_data may be {code: DataFrame} or {code: dict}
                    if code in fd:
                        row = fd[code]
                        if hasattr(row, 'iloc'):
                            row = row.iloc[-1] if len(row) > 0 else row
                        fund = {
                            'roe': float(getattr(row, 'roe', 0.10) or 0.10),
                            'gross_margin': float(getattr(row, 'gross_margin', 0.30) or 0.30),
                            'market_cap': float(getattr(row, 'market_cap', 0) or 0),
                        }
                except Exception:
                    fund = None
            scores = compute_stock_bom_score(code, calc._stock_concepts, fundamentals={code: fund} if fund else None)
            result = scores.get('bom_quality_score', 0.3)
            if result > 0.30:
                d['hit'] += 1
            else:
                d['miss'] += 1
            if scores.get('bom_moat_segments', 0) > 0:
                d['moat'] += 1
                d['_moat_codes'].add(code)
            d['sum_score'] += result
        except Exception:
            result = 0.3
            d['miss'] += 1
        self._bom_cache[code] = result
        return result

    def _is_industry_strong(self, industry: str) -> bool:
        """判断行业是否强势 (IC > 0.06 或 配置中标注为主线)"""
        if not industry:
            return False
        cfg = INDUSTRY_FACTOR_CONFIG.get(industry, {})
        if not cfg:
            return False
        combined_ic = cfg.get('combined_ic', 0)
        base_ic = cfg.get('ic', 0)
        return combined_ic > 0.06 or base_ic > 0.06

    # _generate_signal removed — replaced by generate() + vectorized score assembly

    # ========================================================================
    # 向量化批处理：标量收集 + 数组级联装配
    # ========================================================================

    def _collect_bar_scalars(self, ind: dict, code: str, dates: np.ndarray, n: int,
                              regimes: np.ndarray = None, latest_only: bool = False) -> dict:
        """逐bar收集复杂方法调用的标量结果（仅"硬"部分，不含算术）。

        latest_only=True 时采用混合路径：
        - 所有 i>=60 的 bar 用 _calculate_default_factor 快速生成分数（用于动态阈值）
        - 仅 i=n-1 走完整 _select_factor 链（用于最终信号）；Chan结构由 gate_scorer 处理
        - 预期提速 >95%，同时保留动态阈值的自适应能力

        Returns:
            dict of arrays, 每个array长度为n。valid_mask标记有效bar。
        """
        # 预分配数组
        valid = np.zeros(n, dtype=bool)
        mkt_regime = np.zeros(n, dtype=int)
        mkt_extreme = np.zeros(n, dtype=bool)
        mkt_style_regime = np.full(n, 'balanced', dtype=object)
        mkt_style_score = np.zeros(n)
        mkt_style_conf = np.zeros(n)
        mkt_conf = np.zeros(n)
        ind_cat = np.full(n, 'default', dtype=object)
        fname = np.full(n, 'V41', dtype=object)
        fval = np.zeros(n)
        risk_qual = np.zeros(n)
        risk_hvol = np.zeros(n, dtype=bool)
        is_ind = np.zeros(n, dtype=bool)
        fund_score = np.zeros(n)
        has_fund = np.zeros(n, dtype=bool)
        profit_dec = np.zeros(n, dtype=bool)
        style_score = np.zeros(n)
        spec_ind = np.full(n, '', dtype=object)

        # 始终使用逐bar完整因子选择, 回测和实盘逻辑一致
        if False:  # 原 latest_only 快速路径已禁用, 保留代码供参考
            r = regimes['regime']
            idx = slice(60, n)
            r_idx = r[idx].astype(int)
            valid[idx] = True
            mkt_regime[idx] = r_idx

            sl_arr = _safe_get_arr(ind, 'signal_level', n, 0).astype(int)
            tt_arr = _safe_get_arr(ind, 'trend_type', n, 0).astype(int)
            bp_arr = _safe_get_arr(ind, 'buy_point', n, 0).astype(int)

            # 纯因子评分: SL/趋势仅做门控, 分数由因子IC驱动
            _industry = ''
            try:
                if code and hasattr(self, 'dynamic_factor_selector') and \
                   self.dynamic_factor_selector is not None and \
                   self.dynamic_factor_selector.factor_library is not None:
                    lib = self.dynamic_factor_selector.factor_library
                    _mid_date = dates[min(n-1, max(60, n//2))]
                    _industry = self._get_specific_industry(code, _mid_date) if code else ''
                    _scoring = lib.get_scoring_factors(
                        _industry or '', as_of_date=_mid_date,
                        fallback_config=INDUSTRY_FACTOR_CONFIG)
                else:
                    raise Exception("no lib")
            except Exception:
                _scoring = [
                    ('trend_lowvol', 0.30), ('relative_strength', 0.25),
                    ('low_downside', 0.25), ('momentum_reversal', 0.20),
                ]

            factor_contribution = np.zeros(n)
            bull = r_idx == 1
            bear = r_idx == -1
            for fn, w in _scoring[:5]:
                farr = _safe_get_arr(ind, fn, n, 0.0)
                if fn == 'momentum_reversal':
                    contrib_bear = -farr[idx] * w
                    contrib_other = farr[idx] * w
                    factor_contribution[idx] += contrib_bear * bear.astype(np.float64)
                    factor_contribution[idx] += contrib_other * (~bear).astype(np.float64)
                else:
                    factor_contribution[idx] += farr[idx] * w

            fval[idx] = factor_contribution

            # Layer 5: 因子名
            fname[idx][bull] = 'MOM'
            fname[idx][bear] = 'REV'
            fname[idx][neutral] = 'REV'
            # BP8: 横盘突破, 保持专用因子名
            bp_raw2 = _safe_get_arr(ind, 'buy_point', n, 0).astype(int)
            bp8 = (bp_raw2[idx] == 8)
            fname[idx][bp8] = 'REV60'

            # 最新 bar：完整链覆盖
            last = n - 1
            current_date = dates[last]
            market_info = self._get_market_info(current_date)
            industry_category = self._get_industry_category(code, current_date)

            factor_result = self._select_factor(
                ind, last, market_info['regime'], industry_category,
                code=code, current_date=current_date,
                trend_score=market_info.get('trend_score', 0.0),
                volatility=market_info.get('volatility', 0.15)
            )
            if factor_result is not None:
                fn, fv, risk_info, is_ind_f = factor_result
                # 覆盖快速路径的值
                mkt_regime[last] = market_info['regime']
                mkt_extreme[last] = market_info['is_extreme']
                mkt_style_regime[last] = market_info.get('style_regime', 'balanced')
                mkt_style_score[last] = market_info.get('style_score', 0.0)
                mkt_style_conf[last] = market_info.get('style_confidence', 0.0)
                mkt_conf[last] = market_info.get('confidence', 0.0)
                ind_cat[last] = industry_category
                # BP8: 横盘突破用 REV60 (均值回归方向)
                if bp_raw2[last] == 8:
                    fname[last] = 'REV60'
                    _mom60 = _safe_get_arr(ind, 'mom_60d', n, 0.0)[last]
                    fval[last] = np.tanh(-_mom60 * 2)
                else:
                    fname[last] = fn
                    fval[last] = fv
                risk_qual[last] = risk_info.get('dyn_quality', 0.0) if risk_info else 0.0
                risk_hvol[last] = risk_info.get('is_high_vol', False) if risk_info else False
                is_ind[last] = is_ind_f

                if self.fundamental_enabled and code:
                    fs = self._get_fundamental_score(code, current_date)
                    fund_score[last] = fs
                    has_fund[last] = fs > 0
                    profit_dec[last] = self._check_profit_decline(code, current_date)

                style_score[last] = self._get_style_score(ind, last, market_info)
                spec_ind[last] = self._get_specific_industry(code, current_date) if code else ''
                # _get_chan_boost 输出已被 gate_scorer Gate 1 替代
        else:
            # ── 回测路径：预计算数组 + 因子缓存（避免每bar做DataFrame查找）──
            _mkt_regime_arr = regimes['regime'] if regimes is not None else np.zeros(n, dtype=int)
            _mkt_ext_arr = regimes.get('is_extreme', np.zeros(n, dtype=bool)) if regimes else np.zeros(n, dtype=bool)
            _mkt_style_arr = regimes.get('style_regime', np.full(n, 'balanced')) if regimes else np.full(n, 'balanced')
            _mkt_style_score_arr = regimes.get('style_score', np.zeros(n)) if regimes else np.zeros(n)
            _mkt_style_conf_arr = regimes.get('style_confidence', np.zeros(n)) if regimes else np.zeros(n)
            _mkt_conf_arr = regimes.get('confidence', np.zeros(n)) if regimes else np.zeros(n)

            # 预计算行业分类（同股票所有 bar 一致，取中点日期）
            _mid_date = dates[min(n-1, max(60, n//2))]
            _ind_cat = self._get_industry_category(code, _mid_date)
            _spec_ind = self._get_specific_industry(code, _mid_date) if code else ''

            _mi_simple = {'style_score': 0.0, 'confidence': 0.0, 'regime': 0,
                          'is_extreme': False, 'style_regime': 'balanced',
                          'style_confidence': 0.0, 'trend_score': 0.0}

            # BP8因子需要在循环内覆盖(fast path已做, slow path补上)
            _bp_arr = _safe_get_arr(ind, 'buy_point', n, 0).astype(int)
            _close_arr = _safe_get_arr(ind, 'close', n, 0.0)
            _mom60_arr = np.zeros(n)
            _c60 = np.roll(_close_arr, 60)
            _c60[:60] = 0
            _mask = _c60 > 0
            _mom60_arr[_mask] = (_close_arr[_mask] - _c60[_mask]) / _c60[_mask]
            for i in range(60, n):
                _mkt_regime_i = int(_mkt_regime_arr[i])
                _trend_i = float(regimes['trend_score'][i]) if regimes and 'trend_score' in regimes else 0.0
                _vol_i = float(regimes['volatility'][i]) if regimes and 'volatility' in regimes else 0.15
                factor_result = self._select_factor(
                    ind, i, _mkt_regime_i, _ind_cat, code=code, current_date=dates[i],
                    trend_score=_trend_i, volatility=_vol_i
                )
                if factor_result is None:
                    continue

                fn, fv, risk_info, is_ind_f = factor_result
                valid[i] = True
                mkt_regime[i] = _mkt_regime_i
                mkt_extreme[i] = _mkt_ext_arr[i]
                mkt_style_regime[i] = _mkt_style_arr[i]
                mkt_style_score[i] = _mkt_style_score_arr[i]
                mkt_style_conf[i] = _mkt_style_conf_arr[i]
                mkt_conf[i] = _mkt_conf_arr[i]
                ind_cat[i] = _ind_cat
                # BP8: 60日动量反相关(IC=-0.092), 高动量突破=力竭
                if _bp_arr[i] == 8:
                    fname[i] = 'REV60'
                    fval[i] = np.tanh(-_mom60_arr[i] * 2)
                else:
                    fname[i] = fn
                    fval[i] = fv
                risk_qual[i] = risk_info.get('dyn_quality', 0.0) if risk_info else 0.0
                risk_hvol[i] = risk_info.get('is_high_vol', False) if risk_info else False
                is_ind[i] = is_ind_f

                current_date = dates[i]
                if self.fundamental_enabled and code:
                    fs = self._get_fundamental_score(code, current_date)
                    fund_score[i] = fs
                    has_fund[i] = fs > 0
                    profit_dec[i] = self._check_profit_decline(code, current_date)

                # 用数组构建简易 market_info 供 Chan boost
                _mi_simple['regime'] = _mkt_regime_i
                _mi_simple['style_score'] = _mkt_style_score_arr[i]
                _mi_simple['confidence'] = _mkt_conf_arr[i]
                _mi_simple['is_extreme'] = _mkt_ext_arr[i]
                _mi_simple['style_regime'] = _mkt_style_arr[i]
                _mi_simple['style_confidence'] = _mkt_style_conf_arr[i]
                style_score[i] = self._get_style_score(ind, i, _mi_simple)

                spec_ind[i] = _spec_ind
                # _get_chan_boost 已由 gate_scorer Gate 1 替代

        return {
            'valid': valid,
            'market_regime': mkt_regime, 'market_extreme': mkt_extreme,
            'market_style_regime': mkt_style_regime, 'market_style_score': mkt_style_score,
            'market_style_confidence': mkt_style_conf, 'market_confidence': mkt_conf,
            'industry_category': ind_cat,
            'factor_name': fname, 'factor_value': fval,
            'risk_dyn_quality': risk_qual, 'risk_is_high_vol': risk_hvol,
            'is_industry': is_ind,
            'fundamental_score': fund_score, 'has_fundamental': has_fund,
            'profit_declining': profit_dec, 'style_factor_score': style_score,
            'specific_industry': spec_ind,
        }

    def _vectorized_score_assembly(self, s: dict, ind: dict, n: int, code: str = '',
                                    dates: np.ndarray = None) -> dict:
        """向量化分数装配（门控版）：factor × gate_quality 替代6层乘法叠加。

        新架构:
          1. 纯因子值归一化 → score [-1, 1]
          2. 4门控Grade → gate_quality [0, 1]
          3. adjusted_score = score * gate_quality
          4. ML预测融合 → adjusted_score = (1-w)*score + w*ml_pred
          5. 不再做 Chan/MTF/共振 逐层乘法调整

        输入 s = _collect_bar_scalars 的输出 + ind指标字典。
        """
        valid = s['valid']

        # === 1. 纯因子分数（因子值已在[-1,1]，无需/10压缩） ===
        score = np.clip(s['factor_value'], -10.0, 10.0)
        pre_discount_score = score.copy()  # 纯因子分快照, 不含基本面加成
        fund_mask = s['fundamental_score'] > 0
        score[fund_mask] += s['fundamental_score'][fund_mask] * self.fundamental_weight

        # === 2. 门控质量系数 ===
        limit_pct = 0.195 if (code and (code.startswith('688') or code.startswith('300'))) else 0.095
        gate_grades, hard_rejects = compute_all_gates(ind, n)
        gate_quality = compute_gate_quality(gate_grades)

        # === 诊断: 记录门控质量分布（采样每只股票最新bar） ===
        if self._diag is None:
            try:
                from analysis.backtest_diagnostics import get_diagnostics
                self._diag = get_diagnostics()
            except Exception as e:
                import traceback; print(f"[ERR] " + __file__ + ":" + str(e)); traceback.print_exc()
        if self._diag is not None:
            self._diag.record_gate(float(gate_quality[-1]), bool(hard_rejects[-1]))

        # === 3. Gate仅做二元准入, gate_quality(IC≈0)不参与评分 ===
        adjusted_score = score

        # === 3.4 ML预测融合：XGBoost非线性因子组合 ===
        ml_normalized = np.zeros(n)  # 默认无ML
        if self.ml_enabled and self._ml_predictions and dates is not None and code:
            # 归一化code为6位字符串，确保与ML预测字典key格式一致
            code_str = str(code).zfill(6)
            bar_dates = pd.to_datetime(dates)

            # 前向填充：ML预测仅在采样日期生成（date_step=3），需扩展到每日
            ml_pred_series = np.full(len(bar_dates), 0.0)
            if hasattr(self, '_ml_cache') and code_str in self._ml_cache:
                ml_dates_arr, ml_vals_arr = self._ml_cache[code_str]
                if len(ml_dates_arr) > 0:
                    # 统一转为datetime64确保searchsorted类型一致
                    ml_dt64 = np.asarray(ml_dates_arr, dtype='datetime64[ns]')
                    bar_dt64 = np.asarray(bar_dates, dtype='datetime64[ns]')
                    idx = np.searchsorted(ml_dt64, bar_dt64, side='right') - 1
                    valid = (idx >= 0) & (idx < len(ml_vals_arr))
                    ml_pred_series[valid] = ml_vals_arr[idx[valid]]
            else:
                # 回退：逐bar查找（首次调用或无缓存时）
                ml_pred_series = np.array([self._ml_predictions.get((code_str, d), 0.0) for d in bar_dates])

            # ML预测值归一化到score量级（ML输出≈future_ret，需压缩到[-1,1]）
            ml_normalized = np.tanh(ml_pred_series * 15)
            # 仅在ML有非零预测时融合
            ml_active = np.abs(ml_normalized) > 0.01
            adjusted_score[ml_active] = (
                (1 - self.ml_blend_weight) * adjusted_score[ml_active] +
                self.ml_blend_weight * ml_normalized[ml_active]
            )

        # === 3.6 另类数据调整：龙虎榜个股 + 北向/融资市场级 ===
        if self._alt_data is None:
            try:
                from .alternative_data import get_provider
                self._alt_data = get_provider()
            except Exception as e:
                import traceback; print(f"[ERR] " + __file__ + ":" + str(e)); traceback.print_exc()
        if self._alt_data is not None and dates is not None:
            try:
                # 市场级: 北向资金+融资融券信号，逐bar直接查询
                alt_market = np.zeros(n)
                nb = mg = 0.0
                for i in range(60, n):
                    bar_date = pd.to_datetime(dates[i]).date()
                    nb = self._alt_data.get_northbound_signal(bar_date)
                    mg = self._alt_data.get_margin_signal(bar_date)
                    alt_market[i] = (nb * 0.6 + mg * 0.4) * 0.15
                # NaN 兜底：replace NaN with 0 before adding to adjusted_score
                alt_market = np.nan_to_num(alt_market, nan=0.0)
                adjusted_score += alt_market

                # 记录另类数据命中（北向/融资市场级信号）
                if self._diag is not None and abs(alt_market[-1]) > 0.001:
                    self._diag.record_alt_data(northbound=(nb != 0), margin=(mg != 0))

                # 个股级: 龙虎榜独立买点信号 — 机构大买不经过因子筛选, 直接强化
                dt_signal = np.zeros(n)
                if code:
                    for i in range(60, n):
                        dt_sig = self._alt_data.get_dragon_tiger_signal(code, pd.to_datetime(dates[i]).date())
                        if abs(dt_sig) > 0.01:
                            dt_signal[i] = dt_sig
                            adjusted_score[i] += dt_sig * 0.30
                            if self._diag is not None and i == n - 1:
                                self._diag.record_alt_data(dragon_tiger=True)
            except Exception as e:
                import traceback; print(f"[ERR] " + __file__ + ":" + str(e)); traceback.print_exc()  # 另类数据失败不影响主流程

        # === 4. 常用字段 ===
        chan_sl = _safe_get_arr(ind, 'signal_level', n, 0).astype(int)
        buy_point_raw = _safe_get_arr(ind, 'buy_point', n, 0).astype(int)
        sell_point_raw = _safe_get_arr(ind, 'sell_point', n, 0).astype(int)
        trend_type_raw = _safe_get_arr(ind, 'trend_type', n, 0).astype(int)
        bottom_div = _safe_get_arr(ind, 'bottom_divergence', n, 0.0)
        top_div = _safe_get_arr(ind, 'top_divergence', n, 0.0)
        bottom_fx_q = _safe_get_arr(ind, 'bottom_fractal_quality', n, 0.0)
        risk_vol = _safe_get_arr(ind, 'volatility_10', n, 0.02)
        cf_score = _safe_get_arr(ind, 'capital_flow_score', n, 0.0)
        ns_score = _safe_get_arr(ind, 'news_sentiment_score', n, 0.0)
        cf_dir = _safe_get_arr(ind, 'capital_flow_direction', n, 0).astype(int)
        ns_dir = _safe_get_arr(ind, 'news_sentiment_direction', n, 0).astype(int)
        mtf_discount = _safe_get_arr(ind, 'mtf_discount_factor', n, 1.0)

        is_chan_buy = (chan_sl >= 1) & _safe_get_arr(ind, 'confirmed_buy', n, 0).astype(bool)
        is_chan_sell = (chan_sl <= -1) & _safe_get_arr(ind, 'confirmed_sell', n, 0).astype(bool)

        n_buy = ((score > self.buy_threshold * self.resonance_sys1_buy_mult).astype(np.int8) +
                 ((cf_score > self.resonance_sys2_cf_threshold) & (cf_dir == 1)).astype(np.int8) +
                 ((ns_score > self.resonance_sys3_ns_threshold) & (ns_dir == 1)).astype(np.int8))
        # 卖出侧使用独立阈值（流出量级通常小于流入，需更低门槛）
        n_sell = ((score < self.sell_threshold).astype(np.int8) +
                  ((cf_score > self.resonance_sys2_cf_sell_threshold) & (cf_dir == -1)).astype(np.int8) +
                  ((ns_score > self.resonance_sys3_ns_sell_threshold) & (ns_dir == -1)).astype(np.int8))

        # === 5. 信号置信度 ===
        close_price = _safe_get_arr(ind, 'close', n, 0.0)
        ma20_val = _safe_get_arr(ind, 'ma20', n, 0.0)
        price_above_ma20 = (close_price > 0) & (ma20_val > 0) & (close_price > ma20_val)
        dist_ma20 = np.where(ma20_val > 0, (close_price - ma20_val) / ma20_val, 1.0)
        smart_money = _safe_get_arr(ind, 'smart_money_flow', n, 0.0)
        bottom_fx_qual = _safe_get_arr(ind, 'bottom_fractal_quality', n, 0.0)
        vol_ratio_raw = _safe_get_arr(ind, 'volume_ratio_raw', n, 1.0)

        signal_confidence = np.full(n, 0.2)
        signal_confidence += gate_quality * 0.4
        signal_confidence += np.clip(s['risk_dyn_quality'] * 2, 0, 0.2)
        signal_confidence += np.where(np.abs(smart_money) > 0.15, 0.10, 0)
        signal_confidence += np.where((bottom_div > 0.3) | (top_div > 0.3), 0.08, 0)
        signal_confidence += np.where(bottom_fx_qual > 0.35, 0.06, 0)
        signal_confidence += np.where((vol_ratio_raw > 1.2) & price_above_ma20, 0.06, 0)
        signal_confidence = np.clip(signal_confidence, 0.0, 1.0)

        # === 6. 缠论背离类型判定 ===
        confirmed_buy = _safe_get_arr(ind, 'confirmed_buy', n, 0).astype(bool)
        confirmed_sell = _safe_get_arr(ind, 'confirmed_sell', n, 0).astype(bool)
        bi_buy = _safe_get_arr(ind, 'bi_buy_point', n, 0).astype(int)
        bi_sell = _safe_get_arr(ind, 'bi_sell_point', n, 0).astype(int)
        hidden_bottom = _safe_get_arr(ind, 'hidden_bottom_divergence', n, 0.0)
        hidden_top = _safe_get_arr(ind, 'hidden_top_divergence', n, 0.0)
        second_buy = _safe_get_arr(ind, 'second_buy_point', n, 0).astype(int)
        bottom_fx_spike = _safe_get_arr(ind, 'bottom_fractal_vol_spike', n, 0.0)

        div_type = np.full(n, 'none', dtype=object)
        mask = valid & (bottom_fx_qual > 0.35)
        div_type[mask] = 'bottom_fx'
        mask = valid & (bottom_fx_spike >= 3.0)
        div_type[mask] = 'bottom_fx_3x'
        mask = valid & (hidden_top > 0.15)
        div_type[mask] = 'hidden_top'
        mask = valid & (hidden_bottom > 0.15)
        div_type[mask] = 'hidden_bottom'
        mask = valid & (top_div > bottom_div) & (top_div > 0.3)
        div_type[mask] = 'top'
        mask = valid & (bottom_div > top_div) & (bottom_div > 0.3)
        div_type[mask] = 'bottom'
        mask = valid & (second_buy > 0) & (div_type == 'none')
        div_type[mask] = 'B2'

        mask = valid & (chan_sl == -1) & confirmed_sell
        if mask.any():
            for i in np.where(mask)[0]:
                div_type[i] = f'bi{bi_sell[i]}'
        mask = valid & (chan_sl == 1) & confirmed_buy
        if mask.any():
            for i in np.where(mask)[0]:
                div_type[i] = f'bi{bi_buy[i]}'
        mask = valid & (chan_sl == -2) & confirmed_sell
        if mask.any():
            for i in np.where(mask)[0]:
                div_type[i] = f'sell{sell_point_raw[i]}'
        mask = valid & (chan_sl == 2) & confirmed_buy
        if mask.any():
            for i in np.where(mask)[0]:
                div_type[i] = f'buy{buy_point_raw[i]}'
        mask = valid & (chan_sl == -3) & confirmed_sell
        if mask.any():
            for i in np.where(mask)[0]:
                div_type[i] = f'bi{bi_sell[i]}_seg{sell_point_raw[i]}_sell'
        mask = valid & (chan_sl == 3) & confirmed_buy
        if mask.any():
            for i in np.where(mask)[0]:
                div_type[i] = f'bi{bi_buy[i]}_seg{buy_point_raw[i]}_buy'

        # === 7. 组装返回 ===
        return {
            'valid': valid,
            'score': score,
            'adjusted_score': adjusted_score,
            'pre_discount_score': pre_discount_score,
            'ml_score': ml_normalized,          # 独立ML信号(IC=0.28)
            'factor_score': pre_discount_score,  # 独立因子信号(IC=0.07)
            'factor_value': s['factor_value'],
            'factor_name': s['factor_name'],
            'industry': s['specific_industry'],
            'risk_vol': risk_vol,
            'risk_regime': s['market_regime'],
            'risk_confidence': s['market_confidence'],
            'risk_extreme': s['market_extreme'] | s['risk_is_high_vol'],
            'factor_quality': s['risk_dyn_quality'],
            'signal_confidence': signal_confidence,
            'chan_divergence_type': div_type,
            'chan_divergence_strength': np.maximum(bottom_div, top_div),
            'chan_structure_score': _safe_get_arr(ind, 'alignment_score', n, 0.0),
            'chan_buy_point': buy_point_raw,
            'chan_sell_point': sell_point_raw,
            'signal_level': chan_sl,
            'chan_buy_strength': _safe_get_arr(ind, 'buy_strength', n, 0.0),
            'chan_sell_strength': _safe_get_arr(ind, 'sell_strength', n, 0.0),
            'b3_trend_confirmed': _safe_get_arr(ind, 'b3_trend_confirmed', n, 0).astype(bool),
            'b3_trend_rank': _safe_get_arr(ind, 'b3_trend_rank', n, 0).astype(int),
            'b3_breakout_vol_ratio': _safe_get_arr(ind, 'b3_breakout_vol_ratio', n, 0.0),
            'b3_pullback_vol_ratio': _safe_get_arr(ind, 'b3_pullback_vol_ratio', n, 0.0),
            'b3_pullback_shallowness': _safe_get_arr(ind, 'b3_pullback_shallowness', n, 0.0),
            'second_buy_confidence': _safe_get_arr(ind, 'second_buy_confidence', n, 0.0),
            'buy_confidence': _safe_get_arr(ind, 'buy_confidence', n, 0.35),  # B4门控用
            'resonance_systems': n_buy,  # 买入侧系统共振数（与factor_tag R{n}一致）
            'capital_flow_score': cf_score,
            'news_sentiment_score': ns_score,
            'trend_type': trend_type_raw,
            'chan_pivot_zg': _safe_get_arr(ind, 'chan_pivot_zg', n, np.nan),
            'chan_pivot_zd': _safe_get_arr(ind, 'chan_pivot_zd', n, np.nan),
            'chan_pivot_zz': _safe_get_arr(ind, 'chan_pivot_zz', n, np.nan),
            'daily_return': _safe_get_arr(ind, 'ret', n, 0.0),
            'volume_ratio': _safe_get_arr(ind, 'volume_ratio', n, 0.0),
            'volume_ratio_raw': vol_ratio_raw,
            'exhaustion_risk': _safe_get_arr(ind, 'exhaustion_risk', n, 0.0),
            'gap_breakout_confirm': _safe_get_arr(ind, 'gap_breakout_confirm', n, 0.0),
            'stroke_phase': _safe_get_arr(ind, 'stroke_phase', n, 0.0),
            'top_fractal_volume': _safe_get_arr(ind, 'top_fractal_volume', n, 0.0),
            'ma_trend_up': _safe_get_arr(ind, 'ema20_above_60', n, 0).astype(bool),
            'profit_declining': s['profit_declining'],
            'mom_60d': _safe_get_arr(s, 'mom_60d', n, 0.0),
            'dist_ma60': _safe_get_arr(s, 'dist_ma60', n, 0.0),
            'max_dd_20d': _safe_get_arr(ind, 'max_dd_20d', n, 0.0),
            'vol_regime': _safe_get_arr(ind, 'vol_regime', n, 1.0),
            'weekly_trend_up': _safe_get_arr(ind, 'weekly_trend_up', n, 0).astype(bool),
            'monthly_trend_up': _safe_get_arr(ind, 'monthly_trend_up', n, 0).astype(bool),
            'weekly_trend_strength': _safe_get_arr(ind, 'weekly_trend_strength', n, 0.0),
            'monthly_trend_strength': _safe_get_arr(ind, 'monthly_trend_strength', n, 0.0),
            'mtf_alignment_score': _safe_get_arr(ind, 'mtf_alignment_score', n, 0.0),
            'mtf_discount_factor': mtf_discount,
            'weekly_pattern_signal': _safe_get_arr(ind, 'weekly_pattern_signal', n, 0.0),
            'nearest_resistance_pct': _safe_get_arr(ind, 'nearest_resistance_pct', n, 0.0),
            'nearest_support_pct': _safe_get_arr(ind, 'nearest_support_pct', n, 0.0),
            '_chan_buy_signal': is_chan_buy,
            '_chan_sell_signal': is_chan_sell,
            '_dist_ma20': dist_ma20,
            '_hard_rejects': hard_rejects,
            '_gate_quality': gate_quality,
            '_dt_signal': dt_signal,
            'has_fundamental': s['has_fundamental'],
            'style_regime': s['market_style_regime'],
            'style_confidence': s['market_style_confidence'],
            'smart_money': smart_money,
            'n_buy_systems': n_buy,
        }

    def _compute_dynamic_thresholds(self, scores: np.ndarray, regimes: np.ndarray) -> tuple:
        """向量化动态阈值：rolling quantile替代逐bar buffer/counter。

        原逻辑等价转换：
        - deque(maxlen=800) → rolling(window=800, min_periods=100)
        - 每20根K线重算 → 每根K线直接重算（原为性能折衷，向量化后无此必要）
        - 按市场状态选择百分位 → 按regime分组计算不同分位数

        Args:
            scores: shape (n,), 含NaN表示无有效分数
            regimes: shape (n,), 市场状态编码 {1:bull, 0:neutral, -1:bear}

        Returns:
            buy_thresholds: shape (n,), 各bar的买入阈值
            sell_thresholds: shape (n,), 各bar的卖出阈值
        """
        n = len(scores)
        # 默认阈值（rolling quantile不足100根时使用）
        buy_thresholds = np.full(n, self.buy_threshold)
        sell_thresholds = np.full(n, self.sell_threshold)

        # 逐regime计算rolling quantile（regime种类少，循环开销可忽略）
        for regime, pct_raw in self._buy_threshold_pct_map.items():
            mask = regimes == regime
            if not mask.any():
                continue

            pct = pct_raw  # e.g. 0.45
            # NaN-mask non-matching bars so rolling quantile only sees this regime
            s_regime = pd.Series(np.where(mask, scores, np.nan))
            # rolling quantile: shift(1)排除当前bar，避免未来函数
            # 只用当前bar之前的数据计算阈值，与实盘行为一致
            # min_periods=100 等价于 len(buffer) > 100
            s_shifted = s_regime.shift(1)
            roll_buy = s_shifted.rolling(window=800, min_periods=100).quantile(pct)
            roll_sell = s_shifted.rolling(window=800, min_periods=100).quantile(1.0 - pct)

            # 只覆写rolling quantile有效的bar（>=100根），其余保留默认阈值
            buy_vals = roll_buy.values
            sell_vals = roll_sell.values
            valid = mask & ~np.isnan(buy_vals)
            if valid.any():
                buy_thresholds[valid] = np.maximum(buy_vals[valid], self.buy_threshold * 0.15)  # 纯因子评分范围小, 降下限
                sell_thresholds[valid] = np.minimum(sell_vals[valid], self.sell_threshold)
            # else: 该regime样本不足100，静默回退全局阈值

        return buy_thresholds, sell_thresholds

    def _recompute_factor_value(self, ind: dict, idx: int, cached_result: tuple) -> tuple:
        """用缓存的因子选择配置，在指定 bar 重新计算 factor_value。

        缓存存储的是 (factor_name_template, factor_value_at_cache_time, risk_info, is_industry_factor)。
        DYN 因子名格式: DYN_{industry}_{n}F → 提取 industry + valid_factors 由来，
        但无法从缓存中完全恢复 selected_factors。因此对 DYN 因子简单重新执行完整选择。
        对 IND/MOM/REV/SHARPE 因子则调用对应的计算函数重新算值。

        性能权衡: DYN 命中率 88%，cache miss 时才完整计算；cache hit 时只需重新执行便宜的重算。
        """
        fn, fv_cached, risk_info, is_ind_f = cached_result
        if fn is None:
            return None

        # DYN 因子: 缓存键命中时完整重新计算（需要 selected_factors 列表，缓存里没有）
        # 简单回退: cache miss 即可，不重复 _select_factor 的复杂逻辑
        if isinstance(fn, str) and fn.startswith('DYN_'):
            # DYN 因子名包含 industry 前缀，但无法从名字恢复完整的 selected_factors
            # 返回缓存的原始结果（小幅妥协：DYN 因子值在相同 regime/trend/vol 下复用）
            return cached_result

        # IND 因子 (IND_xxx): 重新调用 _calculate_industry_factor_score
        if isinstance(fn, str) and fn.startswith('IND_'):
            industry = '_'.join(fn.split('_')[1:3]) if len(fn.split('_')) >= 3 else ''
            if industry:
                result = self._calculate_industry_factor_score(ind, idx, industry, regime=0)
                if result:
                    new_fn, new_fv, new_risk = result
                    return (new_fn, new_fv, new_risk, is_ind_f)

        # MOM/REV/SHARPE/default: 重新调用 _calculate_default_factor
        if fn in ('MOM', 'REV', 'SHARPE'):
            _mi = _mi_simple if '_mi_simple' in dir() else {}
            regime = _mi.get('regime', 0) if isinstance(_mi := {}, dict) else 0
            new_fn, new_fv, new_risk = self._calculate_default_factor(ind, idx, regime, '')
            return (new_fn, new_fv, new_risk, False)

        # 兜底: 返回缓存结果
        return cached_result

    def _select_factor(self, ind: dict, idx: int, regime: int, industry_category: str = 'default',
                       code=None, current_date=None, trend_score: float = 0.0,
                       volatility: float = 0.15) -> tuple:
        """根据行业选择因子 — 支持因子择时连续状态

        mode配置:
            - dynamic: 只用动态因子（不用固定因子）
            - fixed: 只用固定因子（跳过动态选择）
            - both: 动态优先，失败则用固定因子

        Returns:
            (factor_name, factor_value, risk_info, is_industry_factor)
        """
        specific_industry = self._get_specific_industry(code, current_date) if code else ''

        # fixed模式：直接使用默认因子（跳过行业配置）
        if self.factor_mode == 'fixed':
            self._stats['fixed_default'] += 1
            factor_name, factor_value, risk_info = self._calculate_default_factor(
                ind, idx, regime, industry_category, code=code, current_date=current_date,
                trend_score=trend_score, volatility=volatility)
            return factor_name, factor_value, risk_info, False

        # 动态因子优先 (仅dynamic/both模式)
        if self.factor_mode in ['dynamic', 'both']:
            if self.dynamic_factor_selector.enabled and code and current_date:
                result = self._select_factor_dynamic(ind, idx, regime, code, current_date)
                if result:
                    self._stats['dynamic_success'] += 1
                    self._record_factor_selection(result[0], True, True)
                    return result
                # 动态选择失败
                if self.factor_mode == 'dynamic' and not self.factor_fallback_to_fixed:
                    self._stats['dynamic_fallback_none'] += 1
                    return None

        # reweight模式: 走静态因子选择路径, 但使用walk-forward IC调整权重
        # fixed/both/reweight模式: 走标准静态路径

        # reweight模式: 获取walk-forward IC权重
        dyn_ic_weights = None
        if self.factor_mode == 'reweight' and self.dynamic_factor_selector.enabled and code and current_date:
            dyn_ic_weights = self._get_dynamic_ic_weights(specific_industry, current_date)

        # 固定因子（行业特定或默认）
        # 注意：当factor_mode='dynamic'时，只有fallback允许时才会到达这里
        if self.factor_mode in ['fixed', 'both', 'reweight'] or (self.factor_mode == 'dynamic' and self.factor_fallback_to_fixed):
            if self.industry_factor_enabled and code and current_date:
                # 使用行业特定因子（已按市场状态优化）
                if specific_industry and specific_industry in INDUSTRY_FACTOR_CONFIG:
                    result = self._calculate_industry_factor_score(ind, idx, specific_industry,
                                                                   code=code, current_date=current_date,
                                                                   regime=regime,
                                                                   dynamic_ic_weights=dyn_ic_weights,
                                                                   trend_score=trend_score,
                                                                   volatility=volatility)
                    if result:
                        self._stats['fixed_industry'] += 1
                        factor_name, factor_value, risk_info = result
                        # 不做熊市折扣：组合层用截面rank_pct排序，均匀缩放不改变排名
                        factor_name = factor_name + f'_{specific_industry[:2]}'
                        self._record_factor_selection(factor_name, True, False)
                        return factor_name, factor_value, risk_info, True

        # Fix#13: 最终兜底 — 使用行业category的配置（优先）
        # 如果specific_industry不在配置中,尝试用industry_category匹配
        if self.industry_factor_enabled:
            fallback_ind = specific_industry if specific_industry in INDUSTRY_FACTOR_CONFIG else industry_category
            if fallback_ind in INDUSTRY_FACTOR_CONFIG:
                result = self._calculate_industry_factor_score(
                    ind, idx, fallback_ind, code=code, current_date=current_date,
                    regime=0, dynamic_ic_weights=None,
                    trend_score=trend_score, volatility=volatility
                )
                if result:
                    self._stats['fixed_default'] += 1
                    factor_name, factor_value, risk_info = result
                    return factor_name, factor_value, risk_info, True

        # 绝对兜底: 原始市场状态信号(MOM/REV/SHARPE)
        self._stats['fixed_default'] += 1
        factor_name, factor_value, risk_info = self._calculate_default_factor(ind, idx, regime, industry_category)
        self._record_factor_selection(factor_name, False, False)
        return factor_name, factor_value, risk_info, False

    def _get_dynamic_ic_weights(self, industry: str, current_date) -> Optional[dict]:
        """获取walk-forward IC权重（reweight模式）

        Returns:
            {factor_name: ic_weight} or None
        """
        if not industry or self.dynamic_factor_selector.factor_df is None:
            return None

        if hasattr(current_date, 'date'):
            current_date_str = str(current_date.date())
        else:
            current_date_str = str(current_date)

        all_dates = self.dynamic_factor_selector._all_dates_cache
        if not all_dates:
            return None

        try:
            industry_factors = self.dynamic_factor_selector.select_factors_for_date(current_date_str, all_dates)
        except Exception:
            logger.warning(f"动态因子选择失败 date={current_date_str}", exc_info=True)
            return None

        if not industry_factors or industry not in industry_factors:
            return None

        selected_info = industry_factors[industry]
        if not selected_info or 'factors' not in selected_info:
            return None

        factors = selected_info['factors']
        weights = selected_info.get('weights', None)
        if not weights or len(weights) != len(factors):
            return None

        return {f: w for f, w in zip(factors, weights)}

    def _select_factor_dynamic(self, ind: dict, idx: int, regime: int,
                                code=None, current_date=None) -> Optional[tuple]:
        """动态因子选择

        使用DynamicFactorSelector在每个时点动态选择最优因子

        Returns:
            (factor_name, factor_value, risk_info, is_industry_factor) or None
        """
        self._dyn_fail['total_calls'] += 1
        if not code or not current_date:
            self._dyn_fail['no_code_or_date'] += 1
            return None

        # P1修复: 统一用 pd.to_datetime 归一化 → YYYY-MM-DD 格式
        current_date_str = pd.to_datetime(current_date).strftime('%Y-%m-%d')

        # 获取股票所属行业
        specific_industry = self._get_specific_industry(code, current_date)
        if not specific_industry:
            self._dyn_fail['no_industry'] += 1
            return None

        # 检查因子数据是否存在（factor_df 或预计算的 factor_cache 至少有一个）
        if self.dynamic_factor_selector.factor_df is None and not self.dynamic_factor_selector._factor_cache:
            self._dyn_fail['no_cache_data'] += 1
            return None

        # 获取动态选择的因子
        try:
            all_dates = self.dynamic_factor_selector._all_dates_cache
            if not all_dates:
                self._dyn_fail['no_dates'] += 1
                return None
            industry_factors = self.dynamic_factor_selector.select_factors_for_date(current_date_str, all_dates)
        except Exception:
            self._dyn_fail['lookup_fail'] += 1
            return None

        if not industry_factors or specific_industry not in industry_factors:
            self._dyn_fail['lookup_fail'] += 1
            if not hasattr(self, '_dyn_lookup_miss_by_ind'):
                self._dyn_lookup_miss_by_ind = defaultdict(int)
            self._dyn_lookup_miss_by_ind[specific_industry] += 1
            return None

        # 提取因子列表和质量指标（新返回格式）
        selected_info = industry_factors[specific_industry]
        if not selected_info or 'factors' not in selected_info:
            return None
        selected_factors = selected_info['factors']
        factor_weights = selected_info.get('weights', None)  # IC权重列表
        factor_directions = selected_info.get('directions', None)  # 方向符号（+1正向/-1负向）
        dyn_quality = selected_info.get('quality', 0)

        # 条件fallback: DYN质量过低时返回None，触发fallback到FIXED
        # dyn_quality = avg combined_IR, <0.08 ≈ 因子几乎无预测力
        if dyn_quality < self.dyn_quality_threshold:
            self._dyn_fail['low_quality'] += 1
            return None

        # 只要有因子通过IC验证就使用
        n_factors = len(selected_factors)
        if n_factors < 1:
            return None

        # 计算动态因子得分
        factor_scores = []
        valid_weights = []
        valid_factors = []

        for i, factor_name in enumerate(selected_factors):
            # 基本面因子
            if factor_name.startswith('fund_'):
                # PE/PB 需要当前价格，从 ind 获取 close
                if factor_name == 'fund_pe':
                    close_p = self._safe_get(ind, 'close', idx, None)
                    if close_p and close_p > 0 and hasattr(self, 'fundamental_data') and self.fundamental_data:
                        eps_val = self.fundamental_data.get_eps(code, current_date)
                        if eps_val and eps_val > 0:
                            factor_val = compress_fundamental_factor(close_p / eps_val, 'fund_pe')
                        else:
                            factor_val = None
                    else:
                        factor_val = None
                elif factor_name == 'fund_pb':
                    close_p = self._safe_get(ind, 'close', idx, None)
                    if close_p and close_p > 0 and hasattr(self, 'fundamental_data') and self.fundamental_data:
                        bps_val = self.fundamental_data.get_bps(code, current_date)
                        if bps_val and bps_val > 0:
                            factor_val = compress_fundamental_factor(close_p / bps_val, 'fund_pb')
                        else:
                            factor_val = None
                    else:
                        factor_val = None
                else:
                    factor_val = self._get_fundamental_factor_value(code, current_date, factor_name)
            else:
                # 技术因子
                factor_val = self._safe_get(ind, factor_name, idx, None)

            # 验证因子值的合理性
            if factor_val is not None and not np.isnan(factor_val) and not np.isinf(factor_val):
                # 因子已压缩到(-1, 1)范围，检查是否有超出范围的异常
                if abs(factor_val) > 5:
                    import warnings
                    warnings.warn(f'Extreme factor value after compression: {factor_name}={factor_val:.2e} for {code} on {current_date}')
                    factor_val = np.sign(factor_val) * np.tanh(abs(factor_val))
                # 方向感知：负IC因子取反，使高值始终预测高收益
                sign = factor_directions[i] if factor_directions and i < len(factor_directions) else 1
                factor_scores.append(factor_val * sign)
                w = factor_weights[i] if factor_weights and i < len(factor_weights) else 1.0
                valid_weights.append(w)
                valid_factors.append(factor_name)

        if not factor_scores:
            self._dyn_fail['no_factor_score'] += 1
            return None

        # IC加权平均（使用带符号的权重，保留因子方向）
        if len(valid_weights) > 0 and sum(abs(w) for w in valid_weights) > 0:
            weights_arr = np.array(valid_weights)
            # 对于带符号的权重，用绝对值之和归一化（保留方向信息）
            weights_arr = weights_arr / sum(abs(w) for w in valid_weights)
            factor_value = np.sum(np.array(factor_scores) * weights_arr)
        else:
            factor_value = np.mean(factor_scores)

        # 最终安全检查：因子值应该在(-1, 1)范围
        # 多因子加权平均后可能略微超出，使用tanh再次压缩
        if abs(factor_value) > 1.5:
            factor_value = np.tanh(factor_value)

        # 不做熊市折扣：组合层用截面rank_pct排序，均匀缩放不改变排名

        factor_name = f'DYN_{specific_industry[:4]}_{len(valid_factors)}F'
        risk_info = {'is_high_vol': False, 'dynamic_factor': True, 'n_factors': len(valid_factors),
                     'dyn_quality': dyn_quality, 'selected_factors': valid_factors}

        return factor_name, factor_value, risk_info, True

    def _get_fundamental_factor_value(self, code, current_date, factor_name: str) -> Optional[float]:
        """获取基本面因子值 - 使用统一压缩函数"""
        if not hasattr(self, 'fundamental_data') or not self.fundamental_data:
            return None

        raw_value = self._get_raw_fundamental_value(code, current_date, factor_name)
        if raw_value is None or (isinstance(raw_value, float) and np.isnan(raw_value)):
            return None
        return compress_fundamental_factor(raw_value, factor_name)

    def _check_profit_decline(self, code, current_date) -> bool:
        """基本面排雷: 利润下滑且降幅在扩大(恶化), 而非收窄(拐点)

        国茂股份案例: 利润同比 -33%→-27%→-11%, 降幅收窄+营收转正=拐点, 不应过滤
        真正该过滤的是: -10%→-20%→-35%, 降幅扩大=基本面加速恶化
        """
        try:
            if not hasattr(self, 'fundamental_data') or not self.fundamental_data:
                return False
            fd = self.fundamental_data
            if code not in fd.stock_data:
                return False

            df = fd.stock_data[code]
            if '数据可用日期' not in df.columns or len(df) == 0:
                return False

            # 缓存已排序的基本面DataFrame（每股票只排序一次，避免逐bar重复sort）
            if code not in self._fd_sorted_cache:
                df_sorted = df.sort_values('数据可用日期').reset_index(drop=True)
                df_sorted['数据可用日期_str'] = df_sorted['数据可用日期'].astype(str)
                self._fd_sorted_cache[code] = df_sorted
            df_sorted = self._fd_sorted_cache[code]

            # 找到当前日期之前的最新报告，然后取最近3份报告
            current_date_str = str(current_date)[:10].replace('-', '')
            mask = df_sorted['数据可用日期_str'] <= current_date_str
            if not mask.any():
                return False

            # 取最近的3份报告
            recent = df_sorted[mask].tail(3)
            if len(recent) < 3:
                return False

            pg_values = []
            rg_values = []
            for _, row in recent.iterrows():
                pg = row.get('净利润-同比增长')
                rg = row.get('营业总收入-同比增长')
                if pg is not None:
                    try:
                        pg_values.append(float(str(pg).replace('%', '')))
                    except (ValueError, TypeError):
                        pass
                if rg is not None:
                    try:
                        rg_values.append(float(str(rg).replace('%', '')))
                    except (ValueError, TypeError):
                        pass

            if len(pg_values) >= 3:
                # 三个季度利润同比均为负 → 持续下滑
                if all(v < 0 for v in pg_values):
                    # 检查趋势: 降幅扩大(恶化) or 收窄(拐点)
                    # pg_values[0]=最早, pg_values[-1]=最新
                    trend = pg_values[-1] - pg_values[0]
                    if trend < -5:
                        # 降幅扩大超过5个百分点 → 加速恶化 → 过滤
                        return True
                    # 降幅收窄或持平 → 可能是拐点 → 再看看营收
                    if trend > 5 and len(rg_values) >= 3 and all(v < 0 for v in rg_values):
                        # 利润改善但营收仍在下滑 → 利润改善不可持续 → 过滤
                        return True
                    # 利润+营收都在改善 → 拐点确认, 不过滤
                    return False
            return False
        except Exception:
            logger.debug(f"基本面过滤异常 code={code}", exc_info=True)
            return False

    def _get_raw_fundamental_value(self, code, current_date, factor_name: str) -> Optional[float]:
        """获取基本面因子原始值（使用预加载缓存）"""
        # 优先使用预加载缓存
        if hasattr(self, '_fund_cache') and self._has_fund_data_cache:
            cache_key = str(current_date)[:10]
            entry = self._fund_cache.get(cache_key)
            if entry:
                row = entry.get('row')
                if row is not None:
                    try:
                        return self._extract_fund_value_from_row(row, factor_name)
                    except Exception as e:
                        import traceback; print(f"[ERR] " + __file__ + ":" + str(e)); traceback.print_exc()

        # Fallback
        if not hasattr(self, 'fundamental_data') or not self.fundamental_data:
            return None
        try:
            if factor_name == 'fund_score':
                return self.fundamental_data.get_fundamental_score(code, current_date)
            elif factor_name == 'fund_profit_growth':
                return self.fundamental_data.get_profit_growth(code, current_date)
            elif factor_name == 'fund_roe':
                return self.fundamental_data.get_roe(code, current_date)
            elif factor_name == 'fund_revenue_growth':
                return self.fundamental_data.get_revenue_growth(code, current_date)
            elif factor_name == 'fund_eps':
                return self.fundamental_data.get_eps(code, current_date)
            elif factor_name == 'fund_cf_to_profit':
                return self.fundamental_data.get_cf_to_profit(code, current_date)
            elif factor_name == 'fund_debt_ratio':
                return self.fundamental_data.get_debt_ratio(code, current_date)
            elif factor_name == 'fund_gross_margin':
                return self.fundamental_data.get_gross_margin(code, current_date)
            elif factor_name == 'fund_pg_improve':
                return self.fundamental_data.get_profit_growth_improve(code, current_date)
            elif factor_name == 'fund_rg_improve':
                return self.fundamental_data.get_revenue_growth_improve(code, current_date)
            elif factor_name in ('fund_pe', 'fund_pb'):
                return None  # PE/PB 通过 _select_factor_dynamic 特殊处理
        except Exception:
            logger.debug(f"基本面因子值获取失败 code={code} factor={factor_name}", exc_info=True)
            pass
        return None

    @staticmethod
    def _extract_fund_value_from_row(row, factor_name: str) -> Optional[float]:
        """从缓存row提取基本面原始值"""
        col_map = {
            'fund_score': None,  # 单独计算
            'fund_profit_growth': '净利润-同比增长',
            'fund_roe': '净资产收益率',
            'fund_revenue_growth': '营业总收入-同比增长',
            'fund_eps': '每股收益',
            'fund_cf_to_profit': None,
            'fund_debt_ratio': 'zcfz_资产负债率',
            'fund_gross_margin': '销售毛利率',
            'fund_pg_improve': None,
            'fund_rg_improve': None,
        }
        col = col_map.get(factor_name)
        if col is None:
            return None
        val = row.get(col)
        if val is None:
            return None
        try:
            if isinstance(val, str):
                return float(val.strip('%')) / 100
            return float(val)
        except (ValueError, TypeError):
            return None

    def _calculate_default_factor(self, ind: dict, idx: int, regime: int, industry_category: str,
                                    code=None, current_date=None, trend_score=0.0, volatility=0.15) -> tuple:
        """多因子评分: SL定价 + 买点alpha + FactorLibrary因子(统一入口)

        FactorLibrary.get_scoring_factors() 是因子选择的唯一来源:
        - 有实盘IC数据 → 用实盘IC选出的因子
        - 有YAML行业配置 → 用行业标定因子
        - 都没有 → 用通用因子兜底
        """
        sl = int(self._safe_get(ind, 'signal_level', idx, 0))
        trend_type = int(self._safe_get(ind, 'trend_type', idx, 0))
        bp = int(self._safe_get(ind, 'buy_point', idx, 0))

        # === Layer 1: 因子库贡献 (IC驱动, 作为评分的唯一连续成分) ===
        # SL/趋势/结构: 仅做门控(准入), 不参与评分(无预测力)
        try:
            lib = self.dynamic_factor_selector.factor_library if (
                hasattr(self, 'dynamic_factor_selector') and
                self.dynamic_factor_selector is not None
            ) else None
        except Exception:
            lib = None

        if lib is not None:
            industry = (self._get_specific_industry(code, current_date)
                        if (code and current_date) else '')
            scoring_factors = lib.get_scoring_factors(
                industry or '', as_of_date=current_date,
                fallback_config=INDUSTRY_FACTOR_CONFIG)
        else:
            scoring_factors = [
                ('trend_lowvol', 0.30), ('relative_strength', 0.25),
                ('low_downside', 0.25), ('momentum_reversal', 0.20),
            ]

        score = 0.0
        for factor_name, weight in scoring_factors[:5]:
            fv = self._safe_get(ind, factor_name, idx, 0.0)
            if regime == 1:
                pass  # 牛市: 因子原值
            elif regime == -1 and factor_name in ('momentum_reversal',):
                fv = -fv  # 熊市反转
            score += fv * weight

        # === Layer 5: 因子名 ===
        if regime == 1:
            factor_name = 'MOM'
        else:
            factor_name = 'REV'

        factor_value = float(np.clip(score, -1.0, 1.0))  # 纯因子评分, 宽范围保留区分度
        risk_info = {'is_high_vol': False}
        return factor_name, factor_value, risk_info

    def _record_factor_selection(self, factor_name: str, is_industry: bool, is_dynamic: bool):
        """回测诊断：记录因子选择"""
        try:
            if self._diag is None:
                from analysis.backtest_diagnostics import get_diagnostics
                self._diag = get_diagnostics()
            self._diag.record_factor_selection(factor_name, is_industry, is_dynamic)
        except Exception as e:
            import traceback; print(f"[ERR] " + __file__ + ":" + str(e)); traceback.print_exc()

    def _calculate_industry_factor_score(self, ind: dict, idx: int, industry: str,
                                           code=None, current_date=None, regime=0,
                                           dynamic_ic_weights=None,
                                           trend_score=0.0, volatility=0.15) -> tuple:
        """计算行业特定因子得分 — 市场状态连续插值版

        支持按市场状态选择不同的因子组合，使用IC权重加权
        支持tech_fund_combo等复合因子
        dynamic_ic_weights: walk-forward IC权重 (reweight模式), {factor_name: ic_weight}
        """
        config = INDUSTRY_FACTOR_CONFIG.get(industry)
        if not config:
            return None

        # === 因子择时: 市场状态连续插值替代离散切换 ===
        # 参数来源: _collect_bar_scalars 传入的 trend_score(趋势强度) 和 volatility(波动率)
        # trend_score ∈ [-1,1]: 正值=趋势强(偏牛因子), 负值=趋势弱(偏熊因子)
        # volatility ∈ [0.05,0.40]: 高波=市场恐慌(偏防御), 低波=稳定(偏趋势)
        ts = trend_score if trend_score is not None else 0.0
        vol = volatility if volatility is not None else 0.15
        trend_sig = 1.0 / (1.0 + np.exp(-ts * 5.0))   # sigmoid: 0~1
        vol_factor = np.clip(vol / 0.20, 0.5, 2.0)

        # 连续三状态权重
        bull_w = trend_sig * (2.0 - vol_factor * 0.5)
        bear_w = (1.0 - trend_sig) * vol_factor * 0.7
        neutral_w = 1.0 - bull_w - bear_w
        total = bull_w + neutral_w + bear_w + 1e-10
        bull_w /= total; neutral_w /= total; bear_w /= total

        # 获取三套因子
        bf = config.get('bull_factors', config.get('factors', []))
        bw_cfg = config.get('bull_weights', None)
        nf = config.get('factors', [])
        nw_cfg = config.get('weights', None)
        # 熊市用中性因子(OOS表现差), 但通过连续权重自动降权
        brf = config.get('bear_factors', config.get('factors', []))
        brw_cfg = config.get('bear_weights', None)

        if not nf:
            return None

        # 合并三套因子: 连续加权
        all_factor_names = list(dict.fromkeys(bf + nf + brf))  # 去重保序
        factors = all_factor_names
        weights = []
        for fn in all_factor_names:
            w = 0.0
            if fn in bf and bw_cfg:
                bi = bf.index(fn)
                if bi < len(bw_cfg): w += bull_w * abs(bw_cfg[bi])
            if fn in nf and nw_cfg:
                ni = nf.index(fn)
                if ni < len(nw_cfg): w += neutral_w * abs(nw_cfg[ni])
            if fn in brf and brw_cfg:
                bri = brf.index(fn)
                if bri < len(brw_cfg): w += bear_w * abs(brw_cfg[bri])
            if w > 0.001:
                weights.append(w)
            else:
                weights.append(0.0)  # 占位，后续跳过

        # 获取基本面压缩评分（用于tech_fund_combo等复合因子）
        compressed_fund_score = 0.0
        if code and current_date and hasattr(self, 'fundamental_data') and self.fundamental_data:
            raw_fund_score = self._get_raw_fundamental_value(code, current_date, 'fund_score')
            if raw_fund_score is not None and isinstance(raw_fund_score, (int, float)):
                compressed_fund_score = compress_fundamental_factor(raw_fund_score, 'fund_score')

        direction = config.get('direction', {}) if 'direction' in config else {}

        factor_scores = []
        valid_factors = []
        valid_weights = []

        for i, factor_name in enumerate(factors):
            factor_val = None

            if factor_name == 'tech_fund_combo':
                # tech_fund_combo 需要基本面数据，通过 compute_composite_factors 计算
                combo = compute_composite_factors(ind, idx, fund_score=compressed_fund_score)
                factor_val = combo.get('tech_fund_combo')
            elif factor_name in ('fund_pe', 'fund_pb'):
                close_p = self._safe_get(ind, 'close', idx, None)
                if close_p and close_p > 0 and hasattr(self, 'fundamental_data') and self.fundamental_data:
                    if factor_name == 'fund_pe':
                        eps_val = self.fundamental_data.get_eps(code, current_date)
                        if eps_val and eps_val > 0:
                            factor_val = compress_fundamental_factor(close_p / eps_val, 'fund_pe')
                    else:  # fund_pb
                        bps_val = self.fundamental_data.get_bps(code, current_date)
                        if bps_val and bps_val > 0:
                            factor_val = compress_fundamental_factor(close_p / bps_val, 'fund_pb')
            elif factor_name.startswith('fund_'):
                # 基本面因子：使用统一压缩函数
                factor_val = self._get_fundamental_factor_value(code, current_date, factor_name)
            else:
                # 技术因子：从 ind 字典获取
                factor_val = self._safe_get(ind, factor_name, idx, None)

            # 因子标定归一化：跟踪量级 → 归一化 → IC权重主导贡献
            if factor_val is not None and not np.isnan(factor_val):
                factor_dir = direction.get(factor_name, 1)
                val = factor_val * factor_dir
                # 使用原始压缩因子值（标定IC权重已基于压缩值计算，归一化会破坏权重语义）
                factor_scores.append(val)
                valid_factors.append(factor_name)
                # 获取IC权重
                if weights and i < len(weights):
                    valid_weights.append(weights[i])

        if not factor_scores:
            return None

        # Top-N因子精选: 只取IC权重最强的3个,避免8-12个弱因子平均后噪声淹没信号
        if len(factor_scores) > 3 and valid_weights and len(valid_weights) == len(factor_scores):
            # 按 |score * weight| 排序取top3
            ranked = sorted(zip(factor_scores, valid_factors, valid_weights),
                          key=lambda x: abs(x[0]) * abs(x[2]), reverse=True)
            factor_scores = [s for s, _, _ in ranked[:3]]
            valid_factors = [f for _, f, _ in ranked[:3]]
            valid_weights = [w for _, _, w in ranked[:3]]

        # 使用IC权重加权平均
        if valid_weights and len(valid_weights) == len(factor_scores):
            # reweight模式: 混合静态权重和walk-forward IC权重
            if dynamic_ic_weights and self.factor_mode == 'reweight':
                blend = self.dynamic_factor_selector.reweight_blend
                blended_weights = []
                for i, (w_s, fn) in enumerate(zip(valid_weights, valid_factors)):
                    w_d = dynamic_ic_weights.get(fn, None)
                    if w_d is not None and w_d > 0:
                        blended_weights.append(blend * abs(w_s) + (1 - blend) * w_d)
                    else:
                        blended_weights.append(abs(w_s))
                total_w = sum(blended_weights)
                if total_w > 0:
                    factor_value = sum(s * w for s, w in zip(factor_scores, blended_weights)) / total_w
                else:
                    factor_value = np.mean(factor_scores)
            else:
                # 标准模式: 使用标定产出的权重
                total_w = sum(abs(w) for w in valid_weights)
                if total_w > 0:
                    factor_value = sum(s * abs(w) for s, w in zip(factor_scores, valid_weights)) / total_w
                else:
                    factor_value = np.mean(factor_scores)
        else:
            # 无权重时等权平均
            factor_value = np.mean(factor_scores)

        # 因子分校准: IND静态因子composite偏负(力竭永负+均值回归追高扣分),加bias居中
        factor_value = float(factor_value) + self.composite_bias

        # 因子名称: IND_{行业}_{主导状态}
        if bull_w > neutral_w and bull_w > bear_w:
            tag = '_B'
        elif bear_w > neutral_w and bear_w > bull_w:
            tag = '_E'
        else:
            tag = ''
        return f'IND_{industry[:4]}{tag}', factor_value, {'is_high_vol': volatility > 0.25 if volatility else False, 'industry_factor': True, 'n_factors': len(factor_scores)}

    def _get_style_score(self, ind: dict, idx: int, market_info: dict) -> float:
        """获取风格因子分数"""
        style_regime = market_info.get('style_regime', 'balanced')
        style_confidence = market_info.get('style_confidence', 0.0)

        if style_confidence < 0.3 or style_regime.startswith('balanced'):
            return 0.0

        if 'small_cap' in style_regime:
            price_pos = self._safe_get(ind, 'price_position_20', idx, 0.5)
            return -price_pos * 0.5 + 0.25
        elif 'large_cap' in style_regime:
            price_pos = self._safe_get(ind, 'price_position_20', idx, 0.5)
            return price_pos * 0.5 - 0.25
        elif '_growth' in style_regime:
            mom_10 = self._safe_get(ind, 'mom_10', idx, 0)
            # 使用 np.tanh 压缩替代 clip，保留相对大小
            return np.tanh(mom_10 * 2) * 0.3
        elif '_value' in style_regime:
            vol_10 = self._safe_get(ind, 'volatility_10', idx, 0.02)
            # 使用 np.tanh 压缩替代 clip，保留相对大小
            return np.tanh((0.02 - vol_10) * 5) * 0.3
        return 0.0

    def _preload_stock_fundamentals(self, code, dates):
        """预加载股票基本面数据缓存，避免每根K线查询DataFrame

        核心优化：基本面数据按季度发布，对2000根K线只需计算~16次而非2000次。
        通过指针推进找到每根K线适用的最新财报，预计算score和industry。
        """
        self._fund_cache = {}  # date_str -> {'score': float, 'industry': str, ...}
        self._has_fund_data_cache = False

        if not hasattr(self, 'fundamental_data') or not self.fundamental_data or not code:
            return

        fd = self.fundamental_data
        if code not in fd.stock_data:
            return

        df = fd.stock_data[code]
        if '数据可用日期' not in df.columns or len(df) == 0:
            return

        df = df.copy()
        df['数据可用日期_str'] = df['数据可用日期'].astype(str)
        df = df.sort_values('数据可用日期_str').reset_index(drop=True)

        # 对每个K线日期，用指针推进找到适用财报
        sorted_dates = sorted(dates)
        report_idx = -1
        n_reports = len(df)

        for d in sorted_dates:
            d_ts = pd.Timestamp(d)
            d_str = d_ts.strftime('%Y%m%d')

            # 推进指针：下一份财报已可用
            while report_idx + 1 < n_reports and df.iloc[report_idx + 1]['数据可用日期_str'] <= d_str:
                report_idx += 1

            if report_idx >= 0:
                row = df.iloc[report_idx]
                cache_key = str(d)[:10]
                if cache_key not in self._fund_cache:
                    score = self._compute_fund_score_from_row(row)
                    industry = row.get('所处行业', None)
                    # 存原始row数据供 _get_fundamental_factor_value 使用
                    self._fund_cache[cache_key] = {
                        'score': score,
                        'industry': industry,
                        'row': row,
                    }

        self._has_fund_data_cache = True

    @staticmethod
    def _compute_fund_score_from_row(row) -> float:
        """从基本面数据行计算评分 — 委托给 factor_calculator.compute_fundamental_score"""
        from .factor_calculator import compute_fundamental_score

        def _parse_pct(val):
            if val is None:
                return None
            try:
                if isinstance(val, str):
                    return float(val.strip('%')) / 100
                return float(val)
            except (ValueError, TypeError):
                return None

        return compute_fundamental_score(
            roe=_parse_pct(row.get('净资产收益率')),
            profit_growth=_parse_pct(row.get('净利润-同比增长')),
            revenue_growth=_parse_pct(row.get('营业总收入-同比增长')),
            eps=_parse_pct(row.get('每股收益')),
        )

    def _get_industry_category(self, code, current_date) -> str:
        """获取股票所属行业类型"""
        # 使用预加载缓存
        if hasattr(self, '_fund_cache') and self._has_fund_data_cache:
            cache_key = str(current_date)[:10]
            entry = self._fund_cache.get(cache_key)
            if entry:
                industry = entry.get('industry')
                if industry:
                    return industry
            return 'default'

        if not hasattr(self, 'fundamental_data') or not self.fundamental_data or not code:
            return 'default'
        try:
            industry = self.fundamental_data.get_industry(code, current_date)
            if not industry:
                return 'default'
            return industry
        except Exception:
            return 'default'

    def _get_specific_industry(self, code, current_date) -> str:
        """获取具体行业名（优先industry_codes反向映射保证DYN缓存命中，回退概念板块）"""
        has_ic = hasattr(self, 'industry_codes') and self.industry_codes

        # P0: 优先从 industry_codes 反向映射查找（保证与DYN缓存的key一致）
        if has_ic:
            if not hasattr(self, '_industry_code_reverse_map'):
                self._industry_code_reverse_map = {
                    c: ind_name for ind_name, codes in self.industry_codes.items() for c in codes
                }
            result = self._industry_code_reverse_map.get(code)
            if result:
                return result

        # 回退：概念板块映射（与离线标定对齐）
        if code in STOCK_CONCEPT_MAP:
            for concept in STOCK_CONCEPT_MAP[code]:
                if concept in INDUSTRY_FACTOR_CONFIG:
                    return concept

        # 使用预加载缓存
        if hasattr(self, '_fund_cache') and self._has_fund_data_cache:
            cache_key = str(current_date)[:10]
            entry = self._fund_cache.get(cache_key)
            if entry:
                raw_industry = entry.get('industry')
                if raw_industry:
                    cleaned = raw_industry.replace('Ⅱ', '').replace('Ⅲ', '').replace('Ⅳ', '').strip()
                    # 返回细行业名：如果在配置中直接返回，否则返回原始名
                    if cleaned in INDUSTRY_FACTOR_CONFIG:
                        return cleaned
                    # 尝试关键词匹配作为回退
                    for config_key, keywords in INDUSTRY_KEYWORDS.items():
                        if any(kw in cleaned for kw in keywords):
                            if config_key in INDUSTRY_FACTOR_CONFIG:
                                return config_key
                    return cleaned

        # Fallback: 直接查询 fundamental_data
        has_fd = hasattr(self, 'fundamental_data') and self.fundamental_data
        if has_fd:
            try:
                raw_industry = self.fundamental_data.get_industry(code, current_date)
                if not raw_industry:
                    return None
                cleaned = raw_industry.replace('Ⅱ', '').replace('Ⅲ', '').replace('Ⅳ', '').strip()
                # 返回细行业名：如果在配置中直接返回
                if cleaned in INDUSTRY_FACTOR_CONFIG:
                    return cleaned
                # 关键词回退
                for config_key, keywords in INDUSTRY_KEYWORDS.items():
                    if any(kw in cleaned for kw in keywords):
                        if config_key in INDUSTRY_FACTOR_CONFIG:
                            return config_key
                return cleaned
            except Exception as e:
                import traceback; print(f"[ERR] " + __file__ + ":" + str(e)); traceback.print_exc()
        return None

    def _get_fundamental_score(self, code, current_date) -> float:
        """获取基本面因子评分（使用预加载缓存）"""
        if hasattr(self, '_fund_cache') and self._has_fund_data_cache:
            cache_key = str(current_date)[:10]
            entry = self._fund_cache.get(cache_key)
            if entry:
                return entry.get('score', 0.0)

        if not hasattr(self, 'fundamental_data') or not self.fundamental_data or not code:
            return 0.0

        from .factor_calculator import compute_fundamental_score
        return compute_fundamental_score(
            roe=self.fundamental_data.get_roe(code, current_date),
            profit_growth=self.fundamental_data.get_profit_growth(code, current_date),
            revenue_growth=self.fundamental_data.get_revenue_growth(code, current_date),
            eps=self.fundamental_data.get_eps(code, current_date),
        )

    # === 辅助函数 ===
    @staticmethod
    def _safe_get(ind: dict, key: str, idx: int, default: float = 0.0) -> float:
        """快速安全获取指标值（假设ind值为numpy数组，跳过昂贵类型检查）"""
        arr = ind.get(key)
        if arr is None:
            return default
        try:
            if idx >= len(arr):
                return default
            val = arr[idx]
            # NaN/Inf检查: val!=val 只在NaN时为True, 比np.isnan快
            if val != val or val == float('inf') or val == float('-inf'):
                return default
            return float(val)
        except (TypeError, IndexError):
            return default

    def _sma(self, arr, window):
        result = np.zeros_like(arr, dtype=float)
        result[:] = np.nan
        result[window-1:] = np.convolve(arr, np.ones(window)/window, mode='valid')
        return result


    def _rsi(self, close, window):
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = self._sma(gain, window)
        avg_loss = self._sma(loss, window)
        rs = avg_gain / (avg_loss + 1e-10)
        return 100 - (100 / (1 + rs))

    def _bollinger(self, close, window, num_std):
        middle = self._sma(close, window)
        n = len(close)
        if n < window:
            std = np.zeros(n)
        else:
            from numpy.lib.stride_tricks import sliding_window_view
            sw = sliding_window_view(close, window)
            std = np.zeros(n)
            std[window:] = sw.std(axis=1)[:n-window]
        return middle + num_std * std, middle, middle - num_std * std

    def _atr(self, high, low, close, window):
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = 0
        return self._sma(tr, window)

    def _rolling_max(self, arr, window):
        """滚动最大值 - 不包含当天，避免数据泄露"""
        n = len(arr)
        if n < window:
            return np.full(n, np.nan)
        from numpy.lib.stride_tricks import sliding_window_view
        sw = sliding_window_view(arr, window)
        # sliding_window_view creates (n - window + 1, window) windows
        # But we need (n - window) results for indices [window, n-1]
        result = np.full(n, np.nan)
        result[window:] = sw.max(axis=1)[:n-window]
        return result

    def _rolling_min(self, arr, window):
        """滚动最小值 - 不包含当天，避免数据泄露"""
        n = len(arr)
        if n < window:
            return np.full(n, np.nan)
        from numpy.lib.stride_tricks import sliding_window_view
        sw = sliding_window_view(arr, window)
        result = np.full(n, np.nan)
        result[window:] = sw.min(axis=1)[:n-window]
        return result

    def _shift(self, arr, periods):
        result = np.zeros_like(arr, dtype=float)
        result[periods:] = arr[:-periods]
        result[:periods] = np.nan
        return result

    def _rolling_std(self, arr, window):
        """滚动标准差 - 不包含当天，避免数据泄露"""
        n = len(arr)
        if n < window:
            return np.full(n, np.nan)
        from numpy.lib.stride_tricks import sliding_window_view
        sw = sliding_window_view(arr, window)
        result = np.full(n, np.nan)
        result[window:] = sw.std(axis=1)[:n-window]
        return result

    def _rolling_mean(self, arr, window):
        """滚动均值 - 不包含当天，避免数据泄露"""
        n = len(arr)
        if n < window:
            return np.full(n, np.nan)
        from numpy.lib.stride_tricks import sliding_window_view
        sw = sliding_window_view(arr, window)
        result = np.full(n, np.nan)
        result[window:] = sw.mean(axis=1)[:n-window]
        return result

    def _compute_max_dd(self, close, window):
        """计算滚动窗口内的最大回撤（负值，如-0.15表示-15%）

        向量化实现：对每个 i，计算 [i-window, i) 内的最大回撤
        max_dd = min over j in [i-window, i) of (price[j] - peak[j]) / peak[j]
        """
        n = len(close)
        result = np.full(n, np.nan)
        if n < window + 1:
            return result

        # 向量化：对每个起点，计算窗口内的回撤
        from numpy.lib.stride_tricks import sliding_window_view
        # 取 [window-1, n-1) 的窗口视图，每个窗口有 window 个元素
        sw = sliding_window_view(close, window)
        # sw.shape = (n - window + 1, window)
        # sw[i] = close[i : i+window]

        # 对每个窗口计算：当前价格/累计最大值 - 1
        peak = np.maximum.accumulate(sw, axis=1)
        dd = (sw - peak) / np.where(peak > 0, peak, 1)
        min_dd = dd.min(axis=1)  # shape = (n - window + 1,)

        result[window:] = min_dd[:n - window]
        return result

    def _macd(self, close, fast=12, slow=26, signal=9):
        ema_fast = _ema(close, fast)
        ema_slow = _ema(close, slow)
        macd = ema_fast - ema_slow
        macd_signal = _ema(macd, signal)
        return macd, macd_signal, macd - macd_signal
