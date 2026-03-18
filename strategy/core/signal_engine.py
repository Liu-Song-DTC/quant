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
from typing import Dict, Any, List, Optional, Tuple
from scipy import stats
from multiprocessing import Pool
from tqdm import tqdm

from .signal import Signal
from .signal_store import SignalStore
from .config_loader import load_config
from .industry_factor_config import INDUSTRY_FACTOR_CONFIG
from .factors import calc_all_factors_for_validation

import warnings
warnings.filterwarnings('ignore')


# ============================================================
# 公共函数：预计算因子数据（供评估和回测使用）
# ============================================================

def prepare_factor_data(stock_data: dict, fd,
                       detailed_industries: dict,
                       num_workers: int = 8) -> Tuple[pd.DataFrame, dict, list]:
    """预计算所有股票的因子数据（用于动态因子选择）

    这个函数在系统初始化时调用一次，之后 SignalEngine 生成信号时
    直接使用缓存的因子数据进行动态因子选择。

    实时系统扩展：
    - 每天新数据来临时，调用此函数增量更新因子数据
    - 或者直接更新因子DataFrame，SignalEngine会自动使用

    Args:
        stock_data: {code: DataFrame} 股票历史数据
        fd: FundamentalData 实例
        detailed_industries: 行业分类配置
        num_workers: 并行进程数

    Returns:
        tuple: (factor_data, industry_codes, all_dates)
            - factor_data: 所有股票在所有日期的因子值 DataFrame
            - industry_codes: {category: [codes]} 行业映射
            - all_dates: 所有交易日期列表
    """
    config_loader = load_config()
    lookback = config_loader.get('industry_factor_config.lookback_days', 120)
    forward_period = config_loader.get('dynamic_factor.forward_period', 20)

    # 构建行业映射
    industry_codes = {cat: [] for cat in detailed_industries.keys()}
    all_dates = set()
    for df in stock_data.values():
        all_dates.update(df.index.tolist())
    all_dates = sorted(all_dates)

    for code in stock_data.keys():
        try:
            sample_date = all_dates[100]  # 用较早的日期获取行业
            ind = fd.get_industry(code, sample_date)
            for cat, keywords in detailed_industries.items():
                if any(kw in str(ind) for kw in keywords):
                    industry_codes[cat].append(code)
                    break
        except:
            pass

    # 采样日期（每5天采样一次，减少计算量）
    sample_dates = all_dates[lookback:-forward_period:5]
    print(f"预计算因子数据: {len(sample_dates)} 个时间点, {len(stock_data)} 只股票")

    # 并行计算因子
    args_list = [
        (code, stock_data[code], sample_dates, lookback, forward_period)
        for code in stock_data.keys()
    ]

    all_factor_data = []
    with Pool(num_workers) as pool:
        for res in tqdm(pool.imap(_compute_stock_factors_worker, args_list, chunksize=10),
                       total=len(args_list), desc="计算因子"):
            all_factor_data.extend(res)

    factor_data = pd.DataFrame(all_factor_data) if all_factor_data else pd.DataFrame()
    print(f"因子数据: {len(factor_data)} 条")

    return factor_data, industry_codes, all_dates


def _compute_stock_factors_worker(args):
    """多进程 worker: 计算单只股票的因子数据"""
    code, df, sample_dates, lookback, forward_period = args
    stock_dates = sorted(df.index.tolist())

    results = []
    for sample_date in sample_dates:
        valid_dates = [d for d in stock_dates if d <= sample_date]
        if len(valid_dates) < lookback:
            continue

        eval_date = valid_dates[-1]
        idx = stock_dates.index(eval_date)

        if idx < lookback:
            continue

        history = df.iloc[:idx+1].iloc[-lookback:]
        if len(history) < 60:
            continue

        # 计算因子
        from .factors import calc_all_factors_for_validation
        factors = calc_all_factors_for_validation(
            history['close'].values,
            history['high'].values if 'high' in history.columns else history['close'].values,
            history['low'].values if 'low' in history.columns else history['close'].values,
            history['volume'].values if 'volume' in history.columns else np.ones(len(history)),
            fundamental_data=None,  # worker 中不传递 fd
            code=code,
            eval_date=eval_date
        )

        row = {'code': code, 'date': eval_date}
        for fn, vals in factors.items():
            if hasattr(vals, '__len__') and len(vals) > 0:
                val = vals[-1]
            else:
                val = vals
            if val is not None and not np.isnan(val):
                row[fn] = float(val)

        # 计算未来收益
        if idx + forward_period < len(df):
            future_price = df.iloc[idx + forward_period]['close']
            current_price = df.iloc[idx]['close']
            if current_price > 0:
                row['future_ret'] = (future_price - current_price) / current_price
                results.append(row)

    return results


class DynamicFactorSelector:
    """动态因子选择器 - 基于Walk-Forward验证的动态因子选择

    在每个验证时点，使用训练窗口内的历史数据计算各因子的IC/IR，
    动态选择IR最高的Top-N因子，避免静态配置的过拟合问题。
    """

    def __init__(self, config: dict = None):
        self.config = config or {}
        self._load_config()

        # 缓存: {date: {industry: [factors]}}
        self._factor_cache = {}

        # 因子数据: DataFrame with columns [code, date, factor1, factor2, ..., future_ret]
        self.factor_df = None

        # 行业映射: {category: [codes]}
        self.industry_codes = {}

    def _load_config(self):
        """加载配置"""
        config_loader = load_config()
        dynamic_config = config_loader.get('dynamic_factor', {})

        self.enabled = dynamic_config.get('enabled', False)
        self.train_window = dynamic_config.get('train_window', 250)
        self.forward_period = dynamic_config.get('forward_period', 20)
        self.top_n_factors = dynamic_config.get('top_n_factors', 3)
        self.min_train_samples = dynamic_config.get('min_train_samples', 50)
        self.min_ic_dates = dynamic_config.get('min_ic_dates', 5)

    def set_factor_data(self, factor_df: pd.DataFrame):
        """设置因子数据

        Args:
            factor_df: DataFrame with columns [code, date, factor1, factor2, ..., future_ret]
        """
        self.factor_df = factor_df
        self._factor_cache.clear()

    def set_industry_mapping(self, industry_codes: Dict[str, List[str]]):
        """设置行业映射

        Args:
            industry_codes: {category: [stock_codes]}
        """
        self.industry_codes = industry_codes

    def _calc_ic(self, factor_values: np.ndarray, returns: np.ndarray) -> float:
        """计算IC (Spearman秩相关系数)"""
        valid_mask = ~(np.isnan(factor_values) | np.isnan(returns))
        if valid_mask.sum() < 5:
            return np.nan
        ic, _ = stats.spearmanr(factor_values[valid_mask], returns[valid_mask])
        return ic

    def _select_factors_for_industry(self, train_df: pd.DataFrame, industry: str,
                                     factor_names: List[str]) -> List[str]:
        """为一个行业选择最优因子

        Args:
            train_df: 训练数据
            industry: 行业类别
            factor_names: 可选因子列表

        Returns:
            选中的因子列表
        """
        codes = self.industry_codes.get(industry, [])
        if not codes:
            return []

        # 筛选该行业的股票
        ind_df = train_df[train_df['code'].isin(codes)]
        if len(ind_df) < self.min_train_samples:
            return []

        # 计算每个因子的IC和IR
        factor_metrics = []
        for fn in factor_names:
            if fn not in ind_df.columns:
                continue

            ic_list = []
            for date, group in ind_df.groupby('date'):
                if len(group) >= 3 and 'future_ret' in group.columns:
                    ic = self._calc_ic(group[fn].values, group['future_ret'].values)
                    if not np.isnan(ic):
                        ic_list.append(ic)

            if len(ic_list) >= self.min_ic_dates:
                ic_mean = np.mean(ic_list)
                ic_std = np.std(ic_list) + 1e-10
                factor_metrics.append({
                    'factor': fn,
                    'ic_mean': ic_mean,
                    'ir': ic_mean / ic_std,
                    'ic_list': ic_list
                })

        if not factor_metrics:
            return []

        # 按IR排序选Top-N
        factor_metrics.sort(key=lambda x: x['ir'], reverse=True)
        return [f['factor'] for f in factor_metrics[:self.top_n_factors]]

    def select_factors_for_date(self, val_date: str, all_dates: List[str]) -> Dict[str, List[str]]:
        """为指定日期选择各行业的最优因子

        Args:
            val_date: 验证日期
            all_dates: 所有可用日期列表

        Returns:
            {industry: [factors]} 各行业选中的因子
        """
        # 检查缓存
        if val_date in self._factor_cache:
            return self._factor_cache[val_date]

        if self.factor_df is None or len(self.factor_df) == 0:
            return {}

        # 确定训练窗口
        try:
            val_idx = all_dates.index(val_date)
        except ValueError:
            return {}

        train_start_idx = max(0, val_idx - self.train_window)
        train_start_date = all_dates[train_start_idx]

        # 训练数据（严格只用val_date之前的数据）
        train_df = self.factor_df[
            (self.factor_df['date'] >= train_start_date) &
            (self.factor_df['date'] < val_date)
        ]

        if len(train_df) < self.min_train_samples:
            return {}

        # 因子列表
        exclude_cols = ['code', 'date', 'future_ret']
        factor_names = [c for c in self.factor_df.columns if c not in exclude_cols]

        # 为每个行业选择因子
        result = {}
        for industry in self.industry_codes.keys():
            factors = self._select_factors_for_industry(train_df, industry, factor_names)
            if factors:
                result[industry] = factors

        # 缓存结果
        self._factor_cache[val_date] = result
        return result

    def clear_cache(self):
        """清空缓存"""
        self._factor_cache.clear()


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

    def _load_config(self):
        """从配置文件加载参数"""
        config_loader = load_config()

        # 技术面因子权重
        tech_weights = config_loader.get_technical_weights()
        self.vol_weight = tech_weights.get('volatility_10', 0.20)
        self.rsi_weight = tech_weights.get('rsi_average', 0.30)
        self.bb_weight = tech_weights.get('bb_width', 0.20)
        self.mom_weight = tech_weights.get('momentum', 0.30)

        # 信号阈值
        thresholds = config_loader.get_signal_thresholds()
        self.buy_threshold = thresholds.get('buy', 0.15)
        self.adjusted_buy_threshold = thresholds.get('adjusted_buy', 0.05)
        self.sell_threshold = thresholds.get('sell', -0.05)

        # 极端状态权重
        self.extreme_weight = config_loader.get('extreme_adjustments.regime_weight', 0.85)

        # 基本面因子配置
        fundamental_config = config_loader.get('fundamental_weights', {})
        self.fundamental_enabled = fundamental_config.get('enabled', True)
        self.fundamental_weight = fundamental_config.get('weight', 0.30)
        self.technical_weight = fundamental_config.get('technical_weight', 0.70)

        # 市场状态动态权重
        self.regime_weights = config_loader.get('regime_weights', {})

        # 风格因子配置
        style_config = config_loader.get('style_weights', {})
        self.style_enabled = style_config.get('enabled', True)
        self.style_weight = style_config.get('weight', 0.25)

        # 行业因子配置
        industry_config = config_loader.get_industry_factor_config()
        self.industry_factor_enabled = industry_config.get('enabled', True)

        # 行业分类映射（原始行业名 -> 配置行业键）
        self.detailed_industries = config_loader.get('detailed_industries', {})

        # 技术指标参数
        self.indicator_params = config_loader.get_indicator_params()

        # 动态因子选择器
        self.dynamic_factor_selector = DynamicFactorSelector()
        self.dynamic_factor_enabled = self.dynamic_factor_selector.enabled

        # 动态因子数据（用于信号生成时选择因子）
        self._dynamic_industry_factors = {}  # {industry: [factors]}

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

    def set_fundamental_data(self, fundamental_data):
        """设置基本面数据"""
        self.fundamental_data = fundamental_data

    def set_market_regime(self, regime_df: pd.DataFrame):
        """设置市场状态数据"""
        if 'datetime' in regime_df.columns:
            regime_df = regime_df.copy()
            regime_df['datetime'] = pd.to_datetime(regime_df['datetime'])
            self.market_regime_data = regime_df.set_index('datetime')

    def generate(self, code: str, market_data: pd.DataFrame, signal_store: SignalStore):
        """生成信号"""
        dates = market_data["datetime"].values
        close = market_data['close'].values

        if len(market_data) < 60:
            return

        indicators = self._calculate_indicators(market_data)

        last_sig = None
        for i in range(len(close)):
            sig = self._generate_signal(indicators, i, last_sig, dates[i], code)
            last_sig = sig
            date = pd.to_datetime(dates[i]).date()
            signal_store.set(code, date, sig)

    def generate_at_indices(self, code: str, market_data: pd.DataFrame,
                          indices: list, signal_store: SignalStore):
        """在指定索引位置生成信号"""
        dates = market_data["datetime"].values

        if len(market_data) < 60:
            return

        # 使用因子库计算所有因子
        eval_date = market_data.index[-1]
        indicators = calc_all_factors_for_validation(
            market_data['close'].values,
            market_data['high'].values if 'high' in market_data.columns else market_data['close'].values,
            market_data['low'].values if 'low' in market_data.columns else market_data['close'].values,
            market_data['volume'].values if 'volume' in market_data.columns else np.ones(len(market_data)),
            fundamental_data=getattr(self, 'fundamental_data', None),
            code=code,
            eval_date=eval_date
        )

        # 添加必要的基础数据
        indicators['close'] = market_data['close'].values
        indicators['high'] = market_data['high'].values if 'high' in market_data.columns else market_data['close'].values
        indicators['low'] = market_data['low'].values if 'low' in market_data.columns else market_data['close'].values
        indicators['volume'] = market_data['volume'].values if 'volume' in market_data.columns else np.ones(len(market_data))

        last_sig = None
        # 按时间顺序排序索引
        sorted_indices = sorted(indices)

        for i in sorted_indices:
            if i < 60:
                continue
            sig = self._generate_signal(indicators, i, last_sig, dates[i], code)
            last_sig = sig
            date = pd.to_datetime(dates[i]).date()
            signal_store.set(code, date, sig)

    def _calculate_indicators(self, data: pd.DataFrame) -> dict:
        """计算技术指标"""
        params = self.indicator_params
        close = data['close'].values
        high = data['high'].values
        low = data['low'].values
        volume = data['volume'].values

        result = {
            'close': close,
            'high': high,
            'low': low,
            'volume': volume
        }

        # EMA
        for span in params.get('ema_periods', [5, 10, 20, 60]):
            result[f'ema{span}'] = self._ema(close, span)

        # MA
        for span in params.get('ma_periods', [5, 10, 20, 30, 60]):
            result[f'ma{span}'] = self._sma(close, span)

        # RSI
        for period in params.get('rsi_periods', [6, 8, 10, 14]):
            result[f'rsi_{period}'] = self._rsi(close, period)
        result['rsi'] = result['rsi_14']

        # 布林带
        bb_window = params.get('bb_window', 20)
        bb_std = params.get('bb_std', 2)
        result['bb_upper'], result['bb_middle'], result['bb_lower'] = self._bollinger(close, bb_window, bb_std)

        # 布林带宽度
        bb_std_arr = np.zeros_like(close)
        bb_std_arr[bb_window:] = np.array([np.std(close[max(0, i-bb_window):i]) for i in range(bb_window, len(close))])
        result['bb_width_20'] = 4 * bb_std_arr / (result['bb_middle'] + 1e-10)

        # 布林带位置
        result['bb_pos_30'] = (close - result['bb_lower']) / (result['bb_upper'] - result['bb_lower'] + 1e-10)

        # 成交量
        vol_ma_period = params.get('volume_ma_period', 20)
        result['volume_ma20'] = self._sma(volume, vol_ma_period)
        result['volume_ratio'] = volume / (result['volume_ma20'] + 1e-10)

        # ATR
        for period in params.get('atr_periods', [10, 14, 20]):
            result[f'atr_{period}'] = self._atr(high, low, close, period)
        result['atr'] = result['atr_14']
        result['atr_ratio'] = result['atr_14'] / close

        # 动量
        for period in params.get('momentum_periods', [3, 5, 10, 20, 30]):
            result[f'mom_{period}'] = close / self._shift(close, period) - 1

        # 价格位置
        result['high_20'] = self._rolling_max(high, 20)
        result['low_20'] = self._rolling_min(low, 20)
        result['price_position_20'] = (close - result['low_20']) / (result['high_20'] - result['low_20'] + 1e-10)

        # 波动率
        returns = np.diff(close, prepend=close[0]) / close
        result['ret'] = returns
        for period in params.get('volatility_periods', [5, 10, 20]):
            result[f'volatility_{period}'] = self._rolling_std(returns, period)

        # 均线关系
        result['ema5_above_20'] = result['ema5'] > result['ema20']
        result['ema20_above_60'] = result['ema20'] > result['ema60']
        result['full_golden'] = (result['ema5'] > result['ema20']) & (result['ema20'] > result['ema60'])
        result['full_death'] = (result['ema5'] < result['ema20']) & (result['ema20'] < result['ema60'])

        # 趋势强度
        result['trend_strength'] = (result['ema20'] - result['ema60']) / (result['ema60'] + 1e-10)

        # 斜率
        result['ema20_slope'] = result['ema20'] / self._shift(result['ema20'], 10) - 1

        # MACD
        result['macd'], result['macd_signal'], result['macd_hist'] = self._macd(close)

        # === 行业因子配置中需要的指标 ===
        # 价格相对均线
        result['price_ma_30'] = close / (result['ma30'] + 1e-10) - 1
        result['price_ma_60'] = close / (result['ma60'] + 1e-10) - 1

        # 均线交叉
        ma10, ma20, ma60 = result['ma10'], result['ma20'], result['ma60']
        result['ma_cross_10_60'] = ma10 / (ma60 + 1e-10) - 1
        result['ma_cross_10_20'] = ma10 / (ma20 + 1e-10) - 1
        result['ma_golden_20_60'] = (ma20 > ma60).astype(float)
        result['ma_all'] = ((result['ma5'] > ma20) & (ma20 > ma60)).astype(float)

        # 趋势因子
        result['trend_30'] = (close - result['ma30']) / (result['ma30'] + 1e-10)

        # 价格位置
        result['price_pos_20'] = result['price_position_20']

        # 收益率
        result['ret_30'] = close / self._shift(close, 30) - 1

        # 波动率
        result['vol_20'] = self._rolling_std(returns, 20)
        result['vol_30'] = self._rolling_std(returns, 30)

        # ATR比率
        result['atr_ratio_20'] = result['atr_20'] / close

        # === 复合因子（基于行业验证结果） ===
        # 动量×低波动
        if 'mom_10' in result and 'volatility_10' in result:
            result['mom_x_lowvol_10_10'] = result['mom_10'] * (-result['volatility_10'])
            result['mom_x_lowvol_10_20'] = result['mom_10'] * (-result['volatility_20'])
        if 'mom_20' in result and 'volatility_10' in result:
            result['mom_x_lowvol_20_10'] = result['mom_20'] * (-result['volatility_10'])
        if 'mom_20' in result and 'volatility_20' in result:
            result['mom_x_lowvol_20_20'] = result['mom_20'] * (-result['volatility_20'])

        # RSI + 波动率组合
        if 'rsi_14' in result and 'volatility_20' in result:
            result['rsi_vol_combo'] = (50 - result['rsi_14']) / 100 - result['volatility_20'] * 0.5

        # 布林带 + RSI组合
        if 'bb_pos_30' in result and 'rsi_14' in result:
            result['bb_rsi_combo'] = (50 - result['rsi_14']) / 100 - result['bb_pos_30'] * 0.3

        # 收益波动率比
        if 'mom_10' in result and 'volatility_10' in result:
            result['ret_vol_ratio_10'] = result['mom_10'] / (result['volatility_10'] + 1e-10)
        if 'mom_20' in result and 'volatility_20' in result:
            result['ret_vol_ratio_20'] = result['mom_20'] / (result['volatility_20'] + 1e-10)

        # 动量反转/加速度
        result['momentum_reversal'] = -result.get('mom_20', 0)
        if 'mom_10' in result and 'mom_20' in result:
            result['momentum_acceleration'] = result['mom_10'] - result['mom_20']

        # 收益风险比
        result['return_risk_ratio'] = result.get('ret_vol_ratio_10', 0)

        return result

    def _get_market_info(self, date) -> Dict[str, Any]:
        """获取指定日期的市场状态信息"""
        if self.market_regime_data is None:
            return {
                'regime': 0, 'confidence': 0.0, 'momentum_score': 0.0,
                'trend_score': 0.0, 'volatility': 0.15, 'is_extreme': False,
                'style_regime': 'balanced', 'style_score': 0.0,
                'size_score': 0.0, 'style_confidence': 0.0,
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
            }
        return {
            'regime': 0, 'confidence': 0.0, 'momentum_score': 0.0,
            'trend_score': 0.0, 'volatility': 0.15, 'is_extreme': False,
            'style_regime': 'balanced', 'style_score': 0.0,
            'size_score': 0.0, 'style_confidence': 0.0,
        }

    def _generate_signal(self, ind: dict, idx: int, last_sig, current_date=None, code=None) -> Signal:
        """生成信号"""
        if idx < 60:
            return Signal(
                buy=False, sell=False, score=0.0, factor_value=0.0,
                factor_name='V41', risk_vol=0.03, risk_regime=0,
                risk_confidence=0.0, risk_extreme=False, adjusted_score=0.0
            )

        # 市场状态
        market_info = self._get_market_info(current_date)
        risk_regime = market_info['regime']
        risk_extreme = market_info['is_extreme']
        style_regime = market_info.get('style_regime', 'balanced')
        style_score = market_info.get('style_score', 0.0)
        style_confidence = market_info.get('style_confidence', 0.0)

        # 获取行业类型
        industry_category = self._get_industry_category(code, current_date)

        # 因子选择和计算
        factor_name, factor_value, risk_info, is_industry = self._select_factor(
            ind, idx, risk_regime, industry_category, code=code, current_date=current_date
        )

        # 基本面因子
        fundamental_score = 0.0
        has_fundamental = False
        if self.fundamental_enabled and code:
            fundamental_score = self._get_fundamental_score(code, current_date)
            has_fundamental = fundamental_score > 0

        # 风格因子
        style_factor_score = self._get_style_score(ind, idx, market_info)

        # === 信号系统 v2 (2026-03-18) ===
        # 核心思想: 因子值标准化后作为排序分数

        # 1. 基础分数 = 因子值
        # 注意: 不同因子值域不同，需要在组合层用排名而非绝对值
        base_score = factor_value

        # 2. 基本面增强（仅对非行业因子生效）
        if not is_industry and fundamental_score > 0:
            base_score = base_score + fundamental_score * 0.1

        # 3. 最终分数（用于排序，不做复杂变换）
        score = base_score

        # 4. 波动率风险指标
        risk_vol = self._safe_get(ind, 'volatility_10', idx, 0.02)

        # 5. 极端市场调整
        regime_weight = 0.9 if risk_extreme else 1.0
        adjusted_score = score * regime_weight

        # 6. 交易信号
        # 买入: 分数在正向区间（用分位数判断更合理，但这里简化处理）
        # 不同因子值域不同，这里用相对宽松的条件，让组合层用排名筛选
        # 关键: 买入信号应该是"候选池"，最终选股靠排名
        buy = score > 0  # 放宽条件，让组合层决定
        sell = factor_value < -0.1  # 因子明显看空时卖出

        # 添加标签
        factor_tags = []
        if has_fundamental:
            factor_tags.append('F')
        if style_confidence > 0.3:
            factor_tags.append(style_regime[:2].upper())
        factor_name = factor_name + ('_' + ''.join(factor_tags) if factor_tags else '_T')

        # 获取具体行业用于组合层减配
        specific_industry = self._get_specific_industry(code, current_date) if code else ''

        return Signal(
            buy=buy, sell=sell, score=score, factor_value=factor_value,
            factor_name=factor_name, industry=specific_industry or '',
            risk_vol=risk_vol, risk_regime=risk_regime,
            risk_confidence=market_info.get('confidence', 0.0),
            risk_extreme=risk_extreme or risk_info.get('is_high_vol', False),
            adjusted_score=adjusted_score
        )

    def _select_factor(self, ind: dict, idx: int, regime: int, industry_category: str = 'default',
                       code=None, current_date=None) -> tuple:
        """根据行业选择因子

        Returns:
            (factor_name, factor_value, risk_info, is_industry_factor)
        """
        # 优先使用动态因子选择（如果启用）
        if self.dynamic_factor_enabled and code and current_date:
            result = self._select_factor_dynamic(ind, idx, regime, code, current_date)
            if result:
                return result

        # 静态行业因子配置
        if self.industry_factor_enabled and code and current_date:
            specific_industry = self._get_specific_industry(code, current_date)
            if specific_industry and specific_industry in INDUSTRY_FACTOR_CONFIG:
                result = self._calculate_industry_factor_score(ind, idx, specific_industry,
                                                               code=code, current_date=current_date)
                if result:
                    factor_name, factor_value, risk_info = result
                    if regime == -1:  # 熊市
                        factor_value = factor_value * 0.7
                        factor_name = factor_name + '_bear'
                    factor_name = factor_name + f'_{specific_industry[:2]}'
                    return factor_name, factor_value, risk_info, True

        # 默认因子组合
        factor_name, factor_value, risk_info = self._calculate_default_factor(ind, idx, regime, industry_category)
        return factor_name, factor_value, risk_info, False

    def _select_factor_dynamic(self, ind: dict, idx: int, regime: int,
                                code=None, current_date=None) -> Optional[tuple]:
        """动态因子选择

        使用DynamicFactorSelector在每个时点动态选择最优因子

        Returns:
            (factor_name, factor_value, risk_info, is_industry_factor) or None
        """
        if not code or not current_date:
            return None

        # 获取当前日期的字符串形式
        if hasattr(current_date, 'date'):
            current_date_str = str(current_date.date())
        else:
            current_date_str = str(current_date)

        # 获取股票所属行业
        specific_industry = self._get_specific_industry(code, current_date)
        if not specific_industry:
            return None

        # 获取动态选择的因子
        # 检查缓存中是否有当前时点对应的因子
        # 这里简化处理：每次调用都尝试更新行业因子
        try:
            all_dates = sorted(self.dynamic_factor_selector.factor_df['date'].unique().tolist()) if self.dynamic_factor_selector.factor_df is not None else []
            industry_factors = self.dynamic_factor_selector.select_factors_for_date(current_date_str, all_dates)
        except:
            industry_factors = {}

        if not industry_factors or specific_industry not in industry_factors:
            return None

        selected_factors = industry_factors[specific_industry]
        if not selected_factors:
            return None

        # 计算动态因子得分
        factor_scores = []
        valid_factors = []

        for factor_name in selected_factors:
            # 基本面因子
            if factor_name.startswith('fund_'):
                factor_val = self._get_fundamental_factor_value(code, current_date, factor_name)
            else:
                # 技术因子
                factor_val = self._safe_get(ind, factor_name, idx, None)

            if factor_val is not None and not np.isnan(factor_val):
                factor_scores.append(factor_val)
                valid_factors.append(factor_name)

        if not factor_scores:
            return None

        # 等权平均
        factor_value = np.mean(factor_scores)

        # 熊市调整
        if regime == -1:
            factor_value = factor_value * 0.7

        factor_name = f'DYN_{specific_industry[:4]}_{len(valid_factors)}F'
        risk_info = {'is_high_vol': False, 'dynamic_factor': True, 'n_factors': len(valid_factors)}

        return factor_name, factor_value, risk_info, True

    def _get_fundamental_factor_value(self, code, current_date, factor_name: str) -> Optional[float]:
        """获取基本面因子值"""
        if not hasattr(self, 'fundamental_data') or not self.fundamental_data:
            return None

        try:
            if factor_name == 'fund_score':
                raw = self.fundamental_data.get_fundamental_score(code, current_date)
                return (raw - 50) / 100 if raw is not None else None
            elif factor_name == 'fund_profit_growth':
                raw = self.fundamental_data.get_profit_growth(code, current_date)
                return np.clip(raw, -1, 1) if raw is not None else None
            elif factor_name == 'fund_roe':
                raw = self.fundamental_data.get_roe(code, current_date)
                return (raw - 10) / 20 if raw is not None else None
            elif factor_name == 'fund_revenue_growth':
                raw = self.fundamental_data.get_revenue_growth(code, current_date)
                return np.clip(raw, -1, 1) if raw is not None else None
            elif factor_name == 'fund_eps':
                raw = self.fundamental_data.get_eps(code, current_date)
                return np.clip(raw / 2, -1, 1) if raw is not None else None
            elif factor_name == 'fund_cf_to_profit':
                raw = self.fundamental_data.get_cf_to_profit(code, current_date)
                return np.clip((raw - 1), -1, 1) if raw is not None else None
            elif factor_name == 'fund_debt_ratio':
                raw = self.fundamental_data.get_debt_ratio(code, current_date)
                return (50 - raw) / 50 if raw is not None else None
            elif factor_name == 'fund_gross_margin':
                raw = self.fundamental_data.get_gross_margin(code, current_date)
                return (raw - 30) / 30 if raw is not None else None
        except:
            pass
        return None

    def _calculate_default_factor(self, ind: dict, idx: int, regime: int, industry_category: str) -> tuple:
        """计算默认因子组合

        使用稳定的基本面+动量因子组合，值域标准化到[-1, 1]左右
        """
        # 获取原始指标
        vol_10 = self._safe_get(ind, 'volatility_10', idx, 0.02)
        mom_10 = self._safe_get(ind, 'mom_10', idx, 0)
        mom_20 = self._safe_get(ind, 'mom_20', idx, 0)

        # 动量×低波动因子（最稳定的技术面因子）
        # 值域约在 [-0.1, 0.1] 之间
        mom_lowvol = mom_20 * (1 - vol_10 * 10)  # 低波动加成
        mom_lowvol = np.clip(mom_lowvol, -0.5, 0.5)  # 截断极端值

        # 组合因子值
        factor_value = mom_lowvol

        # 熊市折扣
        if regime == -1:
            factor_value = factor_value * 0.7

        factor_name = 'DEFAULT_MomLowVol'
        if regime == -1:
            factor_name = factor_name + '_bear'

        risk_info = {'is_high_vol': vol_10 > 0.04}
        return factor_name, factor_value, risk_info

    def _calculate_industry_factor_score(self, ind: dict, idx: int, industry: str,
                                           code=None, current_date=None) -> tuple:
        """计算行业特定因子得分

        直接使用原始因子值（与因子验证一致），然后等权平均合成
        """
        config = INDUSTRY_FACTOR_CONFIG.get(industry)
        if not config:
            return None

        factors = config.get('factors', [])
        direction = config.get('direction', {}) if 'direction' in config else {}

        factor_scores = []
        valid_factors = []

        for i, factor_name in enumerate(factors):
            factor_val = None

            # 基本面因子：从 fundamental_data 获取，并标准化到合理范围
            if factor_name.startswith('fund_'):
                if hasattr(self, 'fundamental_data') and self.fundamental_data and code and current_date:
                    try:
                        if factor_name == 'fund_score':
                            raw = self.fundamental_data.get_fundamental_score(code, current_date)
                            # fund_score 原始范围 0-100，标准化到 -0.5 ~ 0.5
                            factor_val = (raw - 50) / 100 if raw is not None else None
                        elif factor_name == 'fund_profit_growth':
                            raw = self.fundamental_data.get_profit_growth(code, current_date)
                            # 利润增长率，截断到 -1 ~ 1
                            factor_val = np.clip(raw, -1, 1) if raw is not None else None
                        elif factor_name == 'fund_roe':
                            raw = self.fundamental_data.get_roe(code, current_date)
                            # ROE 原始是百分比数值如 8.28，标准化
                            factor_val = (raw - 10) / 20 if raw is not None else None  # 10%为中位
                        elif factor_name == 'fund_revenue_growth':
                            raw = self.fundamental_data.get_revenue_growth(code, current_date)
                            # 营收增长率，截断到 -1 ~ 1
                            factor_val = np.clip(raw, -1, 1) if raw is not None else None
                        elif factor_name == 'fund_eps':
                            raw = self.fundamental_data.get_eps(code, current_date)
                            # EPS 标准化
                            factor_val = np.clip(raw / 2, -1, 1) if raw is not None else None
                        elif factor_name == 'fund_cf_to_profit':
                            raw = self.fundamental_data.get_cf_to_profit(code, current_date)
                            # 现金流/利润比，1为正常，截断
                            factor_val = np.clip((raw - 1), -1, 1) if raw is not None else None
                        elif factor_name == 'fund_debt_ratio':
                            raw = self.fundamental_data.get_debt_ratio(code, current_date)
                            # 负债率，反向（低负债好）
                            factor_val = (50 - raw) / 50 if raw is not None else None
                        elif factor_name == 'fund_gross_margin':
                            raw = self.fundamental_data.get_gross_margin(code, current_date)
                            # 毛利率，标准化
                            factor_val = (raw - 30) / 30 if raw is not None else None  # 30%为中位
                    except:
                        factor_val = None
            else:
                # 技术因子：从 ind 字典获取
                factor_val = self._safe_get(ind, factor_name, idx, None)

            # 直接使用原始因子值
            if factor_val is not None and not np.isnan(factor_val):
                factor_dir = direction.get(factor_name, 1)
                factor_scores.append(factor_val * factor_dir)
                valid_factors.append(factor_name)

        if not factor_scores:
            return None

        # 根据method选择合并方式 (2026-03-18优化: 默认使用equal)
        method = config.get('method', 'equal')
        weights = config.get('weights', None)

        if method == 'weighted' and weights and len(weights) >= len(factor_scores):
            # 使用指定权重
            w = weights[:len(factor_scores)]
            total_w = sum(w)
            factor_value = sum(s * w_i for s, w_i in zip(factor_scores, w)) / total_w
        elif method == 'equal':
            # 等权平均（推荐，更稳健）
            factor_value = np.mean(factor_scores)
        else:
            # Top1（保留兼容性，但不推荐）
            factor_value = factor_scores[0]

        return f'IND_{industry[:4]}', factor_value, {'is_high_vol': False, 'industry_factor': True, 'n_factors': len(factor_scores)}

    def _get_style_score(self, ind: dict, idx: int, market_info: dict) -> float:
        """获取风格因子分数"""
        style_regime = market_info.get('style_regime', 'balanced')
        style_confidence = market_info.get('style_confidence', 0.0)

        if style_confidence < 0.3 or style_regime == 'balanced':
            return 0.0

        if style_regime == 'small_cap':
            price_pos = self._safe_get(ind, 'price_position_20', idx, 0.5)
            return -price_pos * 0.5 + 0.25
        elif style_regime == 'large_cap':
            price_pos = self._safe_get(ind, 'price_position_20', idx, 0.5)
            return price_pos * 0.5 - 0.25
        elif style_regime == 'growth':
            mom_10 = self._safe_get(ind, 'mom_10', idx, 0)
            return np.clip(mom_10 * 2, -0.3, 0.3)
        elif style_regime == 'value':
            vol_10 = self._safe_get(ind, 'volatility_10', idx, 0.02)
            return np.clip((0.02 - vol_10) * 5, -0.3, 0.3)
        return 0.0

    def _get_industry_category(self, code, current_date) -> str:
        """获取股票所属行业类型"""
        if not hasattr(self, 'fundamental_data') or not self.fundamental_data or not code:
            return 'default'
        try:
            industry = self.fundamental_data.get_industry(code, current_date)
            if not industry:
                return 'default'
            # 简化：返回行业本身
            return industry
        except:
            return 'default'

    def _get_specific_industry(self, code, current_date) -> str:
        """获取具体行业名（使用detailed_industries映射）"""
        if not hasattr(self, 'fundamental_data') or not self.fundamental_data or not code:
            return None
        try:
            # 获取原始行业名（去除可能的特殊字符）
            raw_industry = self.fundamental_data.get_industry(code, current_date)
            if not raw_industry:
                return None

            # 清理行业名（去除Ⅱ、Ⅲ等特殊字符）
            cleaned_industry = raw_industry.replace('Ⅱ', '').replace('Ⅲ', '').replace('Ⅳ', '').strip()

            # 使用映射将原始行业名转换为配置行业键
            for config_key, keywords in self.detailed_industries.items():
                # 精确匹配或包含匹配
                if raw_industry in keywords or cleaned_industry in keywords:
                    # 检查该行业键是否在 INDUSTRY_FACTOR_CONFIG 中
                    if config_key in INDUSTRY_FACTOR_CONFIG:
                        return config_key
                # 额外检查：关键词是否包含在原始行业中
                for kw in keywords:
                    if kw in raw_industry or kw in cleaned_industry:
                        if config_key in INDUSTRY_FACTOR_CONFIG:
                            return config_key
        except:
            pass
        return None

    def _get_fundamental_score(self, code, current_date) -> float:
        """获取基本面因子评分"""
        if not hasattr(self, 'fundamental_data') or not self.fundamental_data or not code:
            return 0.0

        score = 0.0
        roe = self.fundamental_data.get_roe(code, current_date)
        if roe is not None:
            if roe > 0.15:
                score += 0.35
            elif roe > 0.10:
                score += 0.25
            elif roe > 0.05:
                score += 0.15

        profit_growth = self.fundamental_data.get_profit_growth(code, current_date)
        if profit_growth is not None:
            if profit_growth > 0.50:
                score += 0.30
            elif profit_growth > 0.20:
                score += 0.20
            elif profit_growth > 0:
                score += 0.10

        revenue_growth = self.fundamental_data.get_revenue_growth(code, current_date)
        if revenue_growth is not None:
            if revenue_growth > 0.30:
                score += 0.20
            elif revenue_growth > 0.15:
                score += 0.12
            elif revenue_growth > 0:
                score += 0.05

        eps = self.fundamental_data.get_eps(code, current_date)
        if eps is not None and eps > 0:
            if eps > 1.0:
                score += 0.20
            elif eps > 0.5:
                score += 0.12

        return min(1.0, score)

    # === 辅助函数 ===
    def _safe_get(self, ind: dict, key: str, idx: int, default: float = 0.0) -> float:
        arr = ind.get(key)
        if arr is None:
            return default
        if isinstance(arr, (int, float)):
            return default
        if hasattr(arr, '__len__'):
            if len(arr) <= idx:
                return default
            val = arr[idx]
            if isinstance(val, (int, float)) and not np.isnan(val):
                return val
        return default

    def _sma(self, arr, window):
        result = np.zeros_like(arr, dtype=float)
        result[:] = np.nan
        result[window-1:] = np.convolve(arr, np.ones(window)/window, mode='valid')
        return result

    def _ema(self, arr, span):
        result = np.zeros_like(arr, dtype=float)
        result[0] = arr[0]
        alpha = 2 / (span + 1)
        for i in range(1, len(arr)):
            result[i] = alpha * arr[i] + (1 - alpha) * result[i-1]
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
        std = np.array([np.std(close[i-window:i]) if i >= window else 0 for i in range(len(close))])
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
        result = np.zeros_like(arr, dtype=float)
        result[:] = np.nan
        for i in range(window, len(arr)):
            # 使用 arr[i-window:i]，不包含当天 arr[i]
            result[i] = np.max(arr[i-window:i])
        return result

    def _rolling_min(self, arr, window):
        """滚动最小值 - 不包含当天，避免数据泄露"""
        result = np.zeros_like(arr, dtype=float)
        result[:] = np.nan
        for i in range(window, len(arr)):
            # 使用 arr[i-window:i]，不包含当天 arr[i]
            result[i] = np.min(arr[i-window:i])
        return result

    def _shift(self, arr, periods):
        result = np.zeros_like(arr, dtype=float)
        result[periods:] = arr[:-periods]
        result[:periods] = np.nan
        return result

    def _rolling_std(self, arr, window):
        result = np.zeros_like(arr, dtype=float)
        result[:] = np.nan
        for i in range(window, len(arr)):
            result[i] = np.std(arr[i-window:i])
        return result

    def _macd(self, close, fast=12, slow=26, signal=9):
        ema_fast = self._ema(close, fast)
        ema_slow = self._ema(close, slow)
        macd = ema_fast - ema_slow
        macd_signal = self._ema(macd, signal)
        return macd, macd_signal, macd - macd_signal
