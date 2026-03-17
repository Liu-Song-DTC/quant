# core/signal_engine.py
"""
信号生成引擎

基于行业验证结果的因子配置，支持:
- 行业自适应因子选择
- 市场状态动态权重
- 风格因子调整
"""

import numpy as np
import pandas as pd
from typing import Dict, Any

from .signal import Signal
from .signal_store import SignalStore
from .config_loader import load_config
from .industry_factor_config import INDUSTRY_FACTOR_CONFIG
from .factors import calc_all_factors_for_validation

import warnings
warnings.filterwarnings('ignore')


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

        # 技术面分数 - 保留正负信号，不要强制变成0
        # 但如果是行业因子，factor_value 已经是合成后的分数，保持其正负
        tech_score = factor_value

        # 组合分数 - 行业因子已经包含了基本面信息，不需要再叠加
        if is_industry:
            # 行业因子直接使用，保留其原始信号
            combined_score = tech_score
        else:
            # 非行业因子，使用原来的逻辑
            if fundamental_score > 0:
                bonus = 1.0 + fundamental_score * self.fundamental_weight
                combined_score = max(0, tech_score) * bonus
            else:
                combined_score = max(0, tech_score)

        # 加上风格因子调整（只在正信号时加成）
        if style_confidence > 0.3 and abs(style_factor_score) > 0.05 and combined_score > 0:
            combined_score = combined_score + style_factor_score * self.style_weight * 0.5

        # 保留负信号，但最终 score 只取正（用于排序）
        raw_score = combined_score
        score = max(0, combined_score)

        # 使用 volatility_10 作为风险指标（因子库有）
        risk_vol = self._safe_get(ind, 'volatility_10', idx, 0.02) * 2

        # 极端波动调整
        regime_weight = 0.85 if risk_extreme else 1.0
        adjusted_score = score * regime_weight

        # 交易信号 - 基于原始 factor_value
        buy = factor_value > 0.15 and adjusted_score > 0.05
        sell = factor_value <= -0.05

        # 添加标签
        factor_tags = []
        if has_fundamental:
            factor_tags.append('F')
        if style_confidence > 0.3:
            factor_tags.append(style_regime[:2].upper())
        factor_name = factor_name + ('_' + ''.join(factor_tags) if factor_tags else '_T')

        return Signal(
            buy=buy, sell=sell, score=score, factor_value=factor_value,
            factor_name=factor_name, risk_vol=risk_vol, risk_regime=risk_regime,
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
        # 优先使用行业特定因子
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

    def _calculate_default_factor(self, ind: dict, idx: int, regime: int, industry_category: str) -> tuple:
        """计算默认因子组合"""
        vol_10 = self._safe_get(ind, 'volatility_10', idx, 0.02)
        rsi_8 = self._safe_get(ind, 'rsi_8', idx, 50)
        rsi_6 = self._safe_get(ind, 'rsi_6', idx, 50)
        rsi_10 = self._safe_get(ind, 'rsi_10', idx, 50)
        bb_width = self._safe_get(ind, 'bb_width_20', idx, 0.05)
        mom_10 = self._safe_get(ind, 'mom_10', idx, 0)

        # 因子值
        vol_factor = vol_10 * 10
        rsi_avg = (rsi_6 + rsi_8 + rsi_10) / 3
        rsi_avg_val = (rsi_avg - 50) / 50
        bb_val = bb_width
        mom_val = mom_10 * 2

        # 动态权重
        vol_w, rsi_w, bb_w, mom_w = self.vol_weight, self.rsi_weight, self.bb_weight, self.mom_weight
        if self.regime_weights:
            regime_key = {1: 'bull', -1: 'bear', 0: 'neutral'}.get(regime, 'neutral')
            cfg = self.regime_weights.get(regime_key, {})
            vol_w = cfg.get('volatility_10', vol_w)
            rsi_w = cfg.get('rsi_average', rsi_w)
            bb_w = cfg.get('bb_width', bb_w)
            mom_w = cfg.get('momentum', mom_w)

        factor_value = vol_factor * vol_w + rsi_avg_val * rsi_w + bb_val * bb_w + mom_val * mom_w

        if regime == -1:
            factor_value = factor_value * 0.7

        factor_name = f'V{int(vol_w*100)}_RSIavg{int(rsi_w*100)}_BB{int(bb_w*100)}_Mom{int(mom_w*100)}'
        if regime == -1:
            factor_name = factor_name + '_bear'

        risk_info = {'is_high_vol': self._safe_get(ind, 'volatility_20', idx, 0.02) > 0.05}
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
        direction = config.get('direction', {})

        factor_scores = []
        valid_factors = []

        for i, factor_name in enumerate(factors):
            factor_val = None

            # 基本面因子：从 fundamental_data 获取
            if factor_name.startswith('fund_'):
                if hasattr(self, 'fundamental_data') and self.fundamental_data and code and current_date:
                    try:
                        if factor_name == 'fund_score':
                            factor_val = self.fundamental_data.get_fundamental_score(code, current_date)
                        elif factor_name == 'fund_profit_growth':
                            factor_val = self.fundamental_data.get_profit_growth(code, current_date)
                        elif factor_name == 'fund_roe':
                            factor_val = self.fundamental_data.get_roe(code, current_date)
                        elif factor_name == 'fund_revenue_growth':
                            factor_val = self.fundamental_data.get_revenue_growth(code, current_date)
                        elif factor_name == 'fund_eps':
                            factor_val = self.fundamental_data.get_eps(code, current_date)
                        elif factor_name == 'fund_cf_to_profit':
                            factor_val = self.fundamental_data.get_cf_to_profit(code, current_date)
                        elif factor_name == 'fund_debt_ratio':
                            factor_val = self.fundamental_data.get_debt_ratio(code, current_date)
                        elif factor_name == 'fund_gross_margin':
                            factor_val = self.fundamental_data.get_gross_margin(code, current_date)
                    except:
                        factor_val = None
            else:
                # 技术因子：从 ind 字典获取
                factor_val = self._safe_get(ind, factor_name, idx, None)

            # 直接使用原始因子值，不做归一化（与因子验证一致）
            if factor_val is not None and not np.isnan(factor_val):
                factor_dir = direction.get(factor_name, 1)
                factor_scores.append(factor_val * factor_dir)
                valid_factors.append(factor_name)

        if not factor_scores:
            return None

        # 直接使用 Top1 因子（与因子验证一致）
        # 取第一个因子（Top1）
        factor_value = factor_scores[0]

        return f'IND_{industry[:4]}', factor_value, {'is_high_vol': False, 'industry_factor': True}

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
        result = np.zeros_like(arr, dtype=float)
        result[:] = np.nan
        for i in range(window-1, len(arr)):
            result[i] = np.max(arr[i-window+1:i+1])
        return result

    def _rolling_min(self, arr, window):
        result = np.zeros_like(arr, dtype=float)
        result[:] = np.nan
        for i in range(window-1, len(arr)):
            result[i] = np.min(arr[i-window+1:i+1])
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
