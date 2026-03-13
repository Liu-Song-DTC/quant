# core/signal_fusion.py
"""
信号融合模块 - 将技术面、基本面、趋势等信号融合为最终信号
"""
import numpy as np
from typing import Optional

from .fundamental import FundamentalData


class SignalFusion:
    """
    信号融合器
    负责将不同来源的信号按照规则融合
    """

    def __init__(self, fundamental_data: FundamentalData = None, config: dict = None):
        self.fundamental_data = fundamental_data
        self.config = config or {}
        self._init_weights()

    def _init_weights(self):
        """初始化各类信号的权重"""
        # 技术面权重
        self.tech_weight = 0.50  # 技术面占比50%

        # 基本面权重 - 提高基本面占比
        self.fund_weight = 0.35  # 基本面占比35%

        # 趋势权重
        self.trend_weight = 0.15  # 趋势占比15%

    def fuse(
        self,
        code: str,
        date,
        tech_score: float,
        risk_vol: float,
        market_regime: int = 0,
    ) -> tuple:
        """
        融合信号

        Args:
            code: 股票代码
            date: 当前日期
            tech_score: 技术面得分
            risk_vol: 风险波动率
            market_regime: 市场状态

        Returns:
            (final_score, buy, sell, adjusted_risk_vol)
        """
        # 获取基本面得分
        fund_score = self._get_fundamental_score(code, date)

        # 根据市场状态调整权重
        weights = self._adjust_weights_by_regime(market_regime)

        # 计算融合得分
        final_score = (
            weights['tech'] * tech_score +
            weights['fund'] * fund_score +
            weights['trend'] * 0  # 暂时没有独立的趋势得分
        )

        # 根据市场状态调整最低阈值
        min_score = self._get_min_score(market_regime)

        # 决策
        buy = final_score >= min_score and tech_score > 0.1
        sell = final_score < -0.10

        # 调整风险波动率
        adjusted_risk = self._adjust_risk(risk_vol, market_regime, fund_score)

        return final_score, buy, sell, adjusted_risk

    def _get_fundamental_score(self, code: str, date) -> float:
        """获取基本面得分"""
        if self.fundamental_data is None:
            return 0.0

        score = 0.0

        # ROE得分 (最高0.4)
        roe = self.fundamental_data.get_roe(code, date)
        if roe is not None:
            if roe > 0.15:
                score += 0.4
            elif roe > 0.10:
                score += 0.25
            elif roe > 0.05:
                score += 0.10

        # 净利润增长得分 (最高0.3)
        profit_growth = self.fundamental_data.get_profit_growth(code, date)
        if profit_growth is not None:
            if profit_growth > 0.5:
                score += 0.3
            elif profit_growth > 0.2:
                score += 0.2
            elif profit_growth > 0:
                score += 0.1

        # 营业收入增长得分 (最高0.2)
        revenue_growth = self.fundamental_data.get_revenue_growth(code, date)
        if revenue_growth is not None:
            if revenue_growth > 0.3:
                score += 0.2
            elif revenue_growth > 0.1:
                score += 0.1

        # 每股收益得分 (最高0.1)
        eps = self.fundamental_data.get_eps(code, date)
        if eps is not None and eps > 0:
            score += min(eps * 0.5, 0.1)

        return min(score, 1.0)  # 最高1.0

    def _adjust_weights_by_regime(self, market_regime: int) -> dict:
        """根据市场状态调整权重"""
        if market_regime == 1:  # 牛市
            return {
                'tech': 0.55,  # 技术面更重要
                'fund': 0.30,  # 基本面可以放宽
                'trend': 0.15,
            }
        elif market_regime == -1:  # 熊市
            return {
                'tech': 0.30,  # 技术面降低
                'fund': 0.50,  # 基本面更重要
                'trend': 0.20,
            }
        else:  # 震荡市
            return {
                'tech': 0.45,
                'fund': 0.40,
                'trend': 0.15,
            }

    def _get_min_score(self, market_regime: int) -> float:
        """根据市场状态获取最低得分阈值"""
        if market_regime == 1:  # 牛市可以降低阈值
            return 0.20
        elif market_regime == -1:  # 熊市提高阈值
            return 0.35
        else:  # 震荡市
            return 0.25

    def _adjust_risk(
        self,
        risk_vol: float,
        market_regime: int,
        fund_score: float,
    ) -> float:
        """调整风险波动率"""
        # 熊市降低风险
        if market_regime == -1:
            risk_vol *= 0.7

        # 基本面好的降低风险
        if fund_score > 0.5:
            risk_vol *= 0.85

        return max(risk_vol, 0.02)  # 最低2%
