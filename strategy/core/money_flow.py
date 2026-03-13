# core/money_flow.py
import pandas as pd
import numpy as np

class MoneyFlow:
    """资金流向分析"""

    def __init__(self):
        pass

    def calculate_money_flow(self, df):
        """计算资金流向指标

        基于OBV (On-Balance Volume) 思想
        近似计算：收盘价上涨日放量 = 资金流入
        """
        if len(df) < 10:
            return 0.0

        close = df['close'].values
        volume = df['volume'].values

        # 计算价格变化
        price_change = np.diff(close, prepend=close[0])

        # 成交量加权：上涨时正向，下跌时负向
        flow = np.where(price_change > 0, volume, -volume)

        # 短期资金流向 (5日)
        flow_5 = np.sum(flow[-5:])

        # 中期资金流向 (20日)
        flow_20 = np.sum(flow[-20:])

        # 归一化
        avg_volume = np.mean(volume[-20:]) if len(volume) >= 20 else np.mean(volume)
        if avg_volume > 0:
            flow_score = flow_20 / (avg_volume * 20)
        else:
            flow_score = 0.0

        return flow_score

    def calculate_volume_price_trend(self, df, window=10):
        """量价趋势分析

        上涨时放量 = 资金流入
        下跌时缩量 = 资金观望
        """
        if len(df) < window:
            return 0.0

        close = df['close'].values
        volume = df['volume'].values

        # 计算价格变化率
        returns = np.diff(close, prepend=close[0]) / (close[:-1] + 1e-10)

        # 成交量变化
        vol_change = np.diff(volume, prepend=volume[0]) / (volume[:-1] + 1e-10)

        # 量价配合度：上涨且放量为正向
        score = 0.0
        for i in range(-window, 0):
            if returns[i] > 0 and vol_change[i] > 0:
                score += 0.1
            elif returns[i] < 0 and vol_change[i] < 0:
                score += 0.05
            elif returns[i] > 0 and vol_change[i] < 0:
                score -= 0.05

        return score

    def is资金流入(self, df, threshold=0.1):
        """判断是否资金流入"""
        flow = self.calculate_money_flow(df)
        return flow > threshold
