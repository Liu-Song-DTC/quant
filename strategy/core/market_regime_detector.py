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

    def to_dict(self) -> dict:
        return {
            'regime': self.regime,
            'confidence': self.confidence,
            'momentum_score': self.momentum_score,
            'trend_score': self.trend_score,
            'volatility': self.volatility,
            'is_extreme': self.is_extreme,
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
        # 熊市阈值
        self.mom5_bear = -0.10
        self.mom_bear = -0.12

        # 牛市阈值
        self.mom_bull = 0.08
        self.mom60_bull = 0.05

        # 极端波动阈值
        self.vol_extreme_high = 0.30
        self.vol_extreme_low = 0.15

        # 均线参数
        self.ema_short = 20
        self.ema_medium = 60
        self.ema_long = 120

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

        # 波动率
        returns = close.pct_change()
        self.volatility = returns.rolling(20).std() * np.sqrt(252)

    def _detect_detailed(self, i: int) -> MarketRegimeInfo:
        """
        详细检测 - 输出多种信号
        """
        if i < 120:
            return MarketRegimeInfo(
                regime=0, confidence=0.0, momentum_score=0.0,
                trend_score=0.0, volatility=0.0, is_extreme=False
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

        # 熊市
        if mom_5 < self.mom5_bear or mom < self.mom_bear:
            regime = -1
            confidence = 1.0
        # 牛市
        elif mom > self.mom_bull and mom_60 > self.mom60_bull:
            regime = 1
            confidence = 1.0

        # === 极端状态判断 ===
        is_extreme = vol > self.vol_extreme_high or vol < self.vol_extreme_low

        return MarketRegimeInfo(
            regime=regime,
            confidence=confidence,
            momentum_score=momentum_score,
            trend_score=trend_score,
            volatility=vol,
            is_extreme=is_extreme
        )

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
