# core/market_regime_detector.py
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class MarketRegimeInfo:
    """
    市场状态信息 - 包含多种信号供策略使用
    """
    regime: int           # 离散状态: -1(熊), 0(震荡), 1(牛)
    confidence: float     # 置信度: 0-1
    momentum_score: float  # 动量分数: -1到1
    trend_score: float    # 趋势分数: -1到1 (EMA排列)
    volatility: float     # 波动率
    is_extreme: bool      # 是否处于极端状态
    # 风格因子 (新增)
    style_regime: str     # 风格状态: 'large_cap', 'small_cap', 'value', 'growth', 'balanced'
    style_score: float    # 风格分数: -1(大盘/价值) 到 1(小盘/成长)
    size_score: float     # 大小盘分数: -1(大盘) 到 1(小盘)
    style_confidence: float  # 风格置信度: 0-1
    # 熊市风险信号 (新增)
    bear_risk: bool       # 熊市风险（用于风险管理）

    def to_dict(self) -> dict:
        return {
            'regime': self.regime,
            'confidence': self.confidence,
            'momentum_score': self.momentum_score,
            'trend_score': self.trend_score,
            'volatility': self.volatility,
            'is_extreme': self.is_extreme,
            'style_regime': self.style_regime,
            'style_score': self.style_score,
            'size_score': self.size_score,
            'style_confidence': self.style_confidence,
            'bear_risk': self.bear_risk,
        }


class MarketRegimeDetector:
    """
    市场状态检测器 - 输出多种信号
    """

    def __init__(self, config=None):
        self.config = config or {}
        self.index_data = None
        self._init_params()

    def _init_params(self):
        """初始化参数"""
        # 熊市阈值（改为更敏感：20日动量 < 0）
        self.mom5_bear = -0.02    # 5日动量 < -2%
        self.mom_bear = 0.00       # 20日动量 < 0（改为0以更敏感）
        self.mom_bear_sustained = -0.03  # 持续熊市：20日动量 < -3%

        # 牛市阈值
        self.mom_bull = 0.03      # 20日动量 > 3%
        self.mom60_bull = 0.02    # 60日动量 > 2%

        # 极端波动阈值
        self.vol_extreme_high = 0.30
        self.vol_extreme_low = 0.15

        # 均线参数
        self.ema_short = 20
        self.ema_medium = 60
        self.ema_long = 120

        # 风格检测参数 (新增)
        # 大小盘轮动: 比较沪深300与中证1000的相对强弱
        self.size_lookback = 20  # 短期动量比较
        self.size_lookback_long = 60  # 长期动量比较
        self.size_threshold = 0.03  # 3%阈值判断大小盘

        # === 新增：熊市风险检测参数 ===
        # 用于风险管理的熊市检测（不同于短期超跌检测）
        self.bear_risk_ma_period = 120  # 均线周期
        self.bear_risk_drawdown = 0.15  # 回撤阈值 15%
        self.bear_risk_momentum = -0.10  # 120日动量阈值 -10%

    def generate(self, index_df: pd.DataFrame):
        """
        生成市场状态序列
        """
        self.index_data = index_df.copy()
        self._calculate_indicators()

        # 生成完整信号
        regime_info_list = []
        for i in range(len(self.index_data)):
            info = self._detect_detailed(i)
            regime_info_list.append(info)

        # 展开为列
        self.index_data['regime'] = [r.regime for r in regime_info_list]
        self.index_data['confidence'] = [r.confidence for r in regime_info_list]
        self.index_data['momentum_score'] = [r.momentum_score for r in regime_info_list]
        self.index_data['trend_score'] = [r.trend_score for r in regime_info_list]
        self.index_data['volatility'] = [r.volatility for r in regime_info_list]
        self.index_data['is_extreme'] = [r.is_extreme for r in regime_info_list]
        # 风格因子 (新增)
        self.index_data['style_regime'] = [r.style_regime for r in regime_info_list]
        self.index_data['style_score'] = [r.style_score for r in regime_info_list]
        self.index_data['size_score'] = [r.size_score for r in regime_info_list]
        self.index_data['style_confidence'] = [r.style_confidence for r in regime_info_list]
        # 熊市风险信号
        self.index_data['bear_risk'] = [r.bear_risk for r in regime_info_list]

        return self.index_data

    def _calculate_indicators(self):
        """计算技术指标"""
        close = self.index_data['close']
        high = self.index_data['high']
        low = self.index_data['low']

        # EMA
        self.ema20 = close.ewm(span=self.ema_short).mean()
        self.ema60 = close.ewm(span=self.ema_medium).mean()
        self.ema120 = close.ewm(span=self.ema_long).mean()

        # 动量
        self.momentum = close / close.shift(20) - 1
        self.momentum_60 = close / close.shift(60) - 1
        self.momentum_10 = close / close.shift(10) - 1
        self.momentum_5 = close / close.shift(5) - 1
        self.momentum_120 = close / close.shift(120) - 1  # 新增120日动量

        # 波动率
        returns = close.pct_change()
        self.volatility = returns.rolling(20).std() * np.sqrt(252)

        # 风格指标 (新增)
        # 大小盘相对强弱: 使用短期动量差异
        self.size_momentum = self.momentum_10  # 个股相对于指数的动量
        self.size_momentum_long = self.momentum  # 长期动量

        # === 新增：熊市风险指标 ===
        # 计算回撤
        self.rolling_max = close.rolling(window=120, min_periods=1).max()
        self.drawdown = (close - self.rolling_max) / self.rolling_max

    def _detect_detailed(self, i: int) -> MarketRegimeInfo:
        """
        详细检测 - 输出多种信号
        """
        if i < 120:
            return MarketRegimeInfo(
                regime=0, confidence=0.0, momentum_score=0.0,
                trend_score=0.0, volatility=0.0, is_extreme=False,
                style_regime='balanced', style_score=0.0, size_score=0.0, style_confidence=0.0,
                bear_risk=False
            )

        # 获取各指标
        mom_5 = self.momentum_5.iloc[i]
        mom = self.momentum.iloc[i]
        mom_60 = self.momentum_60.iloc[i]
        vol = self.volatility.iloc[i]

        # EMA趋势分数
        ema20 = self.ema20.iloc[i]
        ema60 = self.ema60.iloc[i]
        ema120 = self.ema120.iloc[i]
        ema20_above_60 = ema20 > ema60
        ema60_above_120 = ema60 > ema120

        # === 动量分数 (-1 到 1) ===
        momentum_score = np.clip(mom / 0.15, -1.0, 1.0)

        # === 趋势分数 (-1 到 1) ===
        if ema20_above_60 and ema60_above_120:
            trend_score = 1.0
        elif not ema20_above_60 and not ema60_above_120:
            trend_score = -1.0
        elif ema20_above_60:
            trend_score = 0.5
        else:
            trend_score = -0.5

        # === 离散状态判断 ===
        regime = 0
        confidence = 0.0

        # 均线空头排列检测（熊市辅助判断）
        ema_bearish = not ema20_above_60 and not ema60_above_120

        # 熊市判断：快速下跌 或 持续下跌+均线空头
        # 修改：要求60日动量也为负，才确认熊市（避免假熊市）
        if (mom_5 < self.mom5_bear or mom < self.mom_bear) and mom_60 < 0:
            regime = -1
            confidence = 1.0
        elif mom < self.mom_bear_sustained and ema_bearish and mom_60 < 0:
            # 持续下跌 + 均线空头排列 = 熊市
            regime = -1
            confidence = 0.8

        # 牛市判断：动量强劲 且 均线多头
        elif mom > self.mom_bull and mom_60 > self.mom60_bull:
            regime = 1
            confidence = 1.0
        elif mom > self.mom_bull * 0.5 and ema20_above_60 and ema60_above_120:
            # 动量较好 + 均线多头排列 = 牛市
            regime = 1
            confidence = 0.7

        # === 极端状态判断 ===
        is_extreme = vol > self.vol_extreme_high or vol < self.vol_extreme_low

        # === 熊市风险检测（用于风险管理）===
        # 条件：深度回撤 + 长期动量为负
        # 不同于短期下跌检测，这是真正的系统性风险
        drawdown = self._safe_get(self.drawdown, i, 0)
        mom_120 = self._safe_get(self.momentum_120, i, 0)
        bear_risk = (drawdown < -self.bear_risk_drawdown and
                     mom_120 < self.bear_risk_momentum and
                     ema_bearish)

        # === 风格状态检测 (新增) ===
        style_regime, style_score, size_score, style_confidence = self._detect_style_regime(i)

        return MarketRegimeInfo(
            regime=regime,
            confidence=confidence,
            momentum_score=momentum_score,
            trend_score=trend_score,
            volatility=vol,
            is_extreme=is_extreme,
            style_regime=style_regime,
            style_score=style_score,
            size_score=size_score,
            style_confidence=style_confidence,
            bear_risk=bear_risk
        )

    def _detect_style_regime(self, i: int):
        """
        检测风格状态 - 大小盘/价值成长

        返回: (style_regime, style_score, size_score, confidence)
        - style_regime: 'large_cap', 'small_cap', 'value', 'growth', 'balanced'
        - style_score: -1到1 (大盘/价值 -> 小盘/成长)
        - size_score: -1到1 (大盘 -> 小盘)
        - confidence: 0-1
        """
        if i < 60:
            return 'balanced', 0.0, 0.0, 0.0

        # 获取动量指标
        size_mom_short = self._safe_get(self.size_momentum, i, 0)
        size_mom_long = self._safe_get(self.size_momentum_long, i, 0)

        # 大小盘判断: 使用动量差异
        # 如果个股动量 > 阈值，认为市场偏向小盘
        size_score = np.clip(size_mom_short / self.size_threshold, -1.0, 1.0)

        # 价值/成长判断: 使用长期动量趋势
        # 长期动量为正倾向于成长风格 (上涨趋势中成长更强)
        # 长期动量为负倾向于价值风格 (下跌趋势中价值更稳)
        if size_mom_long > self.size_threshold:
            style_score = np.clip(size_mom_long / 0.1, 0.0, 1.0)  # 偏向成长
        elif size_mom_long < -self.size_threshold:
            style_score = np.clip(size_mom_long / 0.1, -1.0, 0.0)  # 偏向价值
        else:
            style_score = 0.0

        # 计算置信度
        confidence = min(1.0, abs(size_score) * 2)

        # 确定风格状态
        if abs(size_score) < 0.3:
            style_regime = 'balanced'
        elif size_score > 0.3:
            style_regime = 'small_cap'
        else:
            style_regime = 'large_cap'

        # 如果在牛市中，增加成长偏好
        if i < len(self.momentum) and self.momentum.iloc[i] > self.mom_bull:
            if style_score > 0:
                style_regime = 'growth'
            else:
                style_regime = 'value'

        return style_regime, style_score, size_score, confidence

    def _safe_get(self, series, idx, default=0.0):
        """安全获取序列值"""
        try:
            if series is None:
                return default
            val = series.iloc[idx]
            if pd.isna(val):
                return default
            return val
        except:
            return default

    # 兼容旧接口
    def _detect_single(self, i: int) -> int:
        """简化版判断 - 保持原有逻辑"""
        return self._detect_detailed(i).regime

    def get_info(self, date) -> Optional[MarketRegimeInfo]:
        """获取指定日期的市场状态信息"""
        if self.index_data is None:
            return None

        row = self.index_data[self.index_data['datetime'].dt.date == date]
        if row.empty:
            return None

        idx = row.index[0]
        return MarketRegimeInfo(
            regime=int(self.index_data.loc[idx, 'regime']),
            confidence=float(self.index_data.loc[idx, 'confidence']),
            momentum_score=float(self.index_data.loc[idx, 'momentum_score']),
            trend_score=float(self.index_data.loc[idx, 'trend_score']),
            volatility=float(self.index_data.loc[idx, 'volatility']),
            is_extreme=bool(self.index_data.loc[idx, 'is_extreme']),
            style_regime=str(self.index_data.loc[idx, 'style_regime']),
            style_score=float(self.index_data.loc[idx, 'style_score']),
            size_score=float(self.index_data.loc[idx, 'size_score']),
            style_confidence=float(self.index_data.loc[idx, 'style_confidence']),
            bear_risk=bool(self.index_data.loc[idx, 'bear_risk']),
        )

    def get_regime(self, date) -> int:
        """获取指定日期的市场状态"""
        if self.index_data is None:
            return 0

        row = self.index_data[self.index_data['datetime'].dt.date == date]
        if row.empty:
            return 0
        return int(row['regime'].values[0])

    def get_regime_stats(self) -> dict:
        """获取各年份市场状态统计"""
        if self.index_data is None:
            return {}

        self.index_data['year'] = self.index_data['datetime'].dt.year
        regime_by_year = self.index_data.groupby('year')['regime'].value_counts().unstack(fill_value=0)

        stats = {}
        for year in regime_by_year.index:
            total = regime_by_year.loc[year].sum()
            bull = regime_by_year.loc[year].get(1, 0)
            bear = regime_by_year.loc[year].get(-1, 0)
            stats[year] = {
                'bull_days': bull,
                'bear_days': bear,
                'sideways_days': total - bull - bear,
                'bull_ratio': bull / total if total > 0 else 0,
            }
        return stats


# 便捷函数
def create_market_regime(index_df: pd.DataFrame, config=None) -> pd.DataFrame:
    """创建市场状态序列"""
    detector = MarketRegimeDetector(config)
    return detector.generate(index_df)
