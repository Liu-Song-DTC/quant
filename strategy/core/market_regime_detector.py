# core/market_regime_detector.py
import pandas as pd
import numpy as np


class MarketRegimeDetector:
    """
    市场状态检测器 - 独立模块，更容易调试和优化
    """

    def __init__(self, config=None):
        self.config = config or {}
        self.index_data = None
        self._init_params()

    def _init_params(self):
        """初始化参数"""
        # 快速下跌检测
        self.mom5_bear = -0.10  # 5日跌超10%

        # 下跌判断
        self.momentum_bear1 = -0.12  # 20日跌超12%
        self.momentum_bear2 = -0.05  # 20日跌超5% + 趋势向下

        # 上涨判断 - 用更严格的条件来确保判断正确
        self.momentum_bull1 = 0.10  # 20日涨超10%
        self.momentum_bull2 = 0.12  # 60日涨超12% + 均线多头

        # 趋势判断
        self.ema_short = 20
        self.ema_medium = 60
        self.ema_long = 120

    def generate(self, index_df: pd.DataFrame):
        """
        生成市场状态序列

        Args:
            index_df: 指数数据，包含 datetime, close, high, low 等
        """
        self.index_data = index_df.copy()
        self._calculate_indicators()

        regime = []
        for i in range(len(self.index_data)):
            r = self._detect_single(i)
            regime.append(r)

        self.index_data['regime'] = regime
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

        # 趋势
        self.trend_up = self.ema20 > self.ema60
        self.long_up = self.ema60 > self.ema120

        # 动量
        self.momentum = close / close.shift(20) - 1
        self.momentum_60 = close / close.shift(60) - 1
        self.momentum_10 = close / close.shift(10) - 1
        self.momentum_5 = close / close.shift(5) - 1

        # 动量变化趋势（前瞻性指标）
        self.momentum_change = self.momentum - self.momentum.shift(5)

        # 波动率
        returns = close.pct_change()
        self.volatility = returns.rolling(20).std() * np.sqrt(252)

        # 成交量的变化
        if 'volume' in self.index_data.columns:
            volume = self.index_data['volume']
            self.volume_ma = volume.rolling(20).mean()
            self.volume_ratio = volume / (self.volume_ma + 1e-10)
            self.volume_change = self.volume_ratio / self.volume_ratio.shift(5) - 1

    def _detect_single(self, i: int) -> int:
        """
        基于风险调整的市场状态判断
        不追求准确预测，而是识别极端风险
        """
        if i < 120:
            return 0

        # 极端风险情况 - 快速下跌（可能股灾）
        if self.momentum_5.iloc[i] < -0.10:
            return -1

        # 极端乐观情况 - 快速上涨
        if self.momentum_5.iloc[i] > 0.10:
            return 1

        # 持续上涨趋势（需要动量支持）
        if (self.momentum.iloc[i] > 0.08 and
            self.trend_up.iloc[i] and
            self.long_up.iloc[i]):
            return 1

        # 持续下跌趋势
        if (self.momentum.iloc[i] < -0.08 and
            not self.trend_up.iloc[i] and
            not self.long_up.iloc[i]):
            return -1

        # 默认震荡市
        return 0

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
