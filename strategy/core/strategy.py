import pandas as pd
from typing import Optional

from .signal_engine import SignalEngine
from .signal_store import SignalStore
from .portfolio import PortfolioConstructor
from .market_regime_detector import MarketRegimeDetector
from .config_loader import load_config


class Strategy:
    """带股灾检测的市场状态判断 + 情绪分析集成"""

    def __init__(self, init_cash, fundamental_data=None, sentiment_orchestrator=None):
        self.signal_engine = SignalEngine()
        self.fundamental_data = fundamental_data
        if fundamental_data:
            self.signal_engine.set_fundamental_data(fundamental_data)

        self.portfolio = PortfolioConstructor()
        self.market_regime = []
        self.signal_store = SignalStore()
        self.init_cash = init_cash

        self.index_data = None
        self.sentiment_orchestrator = sentiment_orchestrator

        # 使用独立的市场状态检测器
        self.regime_detector = MarketRegimeDetector()

    def set_factor_data(self, factor_df, industry_codes):
        """设置因子数据（用于动态因子选择）"""
        self.signal_engine.set_factor_data(factor_df)
        self.signal_engine.set_industry_mapping(industry_codes)
        print(f"动态因子数据已设置: {len(factor_df)} 条记录")

    def generate_market_regime(self, index_df):
        # 使用独立的市场状态检测器
        self.index_data = self.regime_detector.generate(index_df)

    def generate_signal(self, code, market_data):
        self.signal_engine.generate(code, market_data, self.signal_store)

    def set_sentiment_multipliers(self, date, market_regime: int = 0):
        """从情绪编排器获取行业情绪权重并注入组合构建器"""
        if self.sentiment_orchestrator is None:
            return
        try:
            multipliers = self.sentiment_orchestrator.get_sentiment_weights(
                market_regime=market_regime
            )
            if multipliers:
                self.portfolio.set_sentiment_multipliers(multipliers)
        except Exception as e:
            pass  # 情绪模块异常不中断主流程

    def generate_positions(
        self,
        date,
        universe,
        current_positions,
        cash,
        prices,
        cost,
        rebalance,
    ):
        market_regime = 0
        momentum_score = 0.0
        bear_risk = False
        trend_score = 0.0
        if self.index_data is not None:
            row = self.index_data[self.index_data["datetime"].dt.date == date]
            if not row.empty:
                market_regime = int(row["regime"].values[0])
                momentum_score = float(row["momentum_score"].values[0])
                bear_risk = bool(row["bear_risk"].values[0]) if "bear_risk" in row.columns else False
                trend_score = float(row["trend_score"].values[0]) if "trend_score" in row.columns else 0.0

        # 再平衡日更新情绪权重
        if rebalance:
            self.set_sentiment_multipliers(date, market_regime)

        return self.portfolio.build(
            date=date,
            universe=universe,
            current_positions=current_positions,
            signal_store=self.signal_store,
            cash=cash,
            prices=prices,
            market_regime=market_regime,
            momentum_score=momentum_score,
            bear_risk=bear_risk,
            trend_score=trend_score,
            cost=cost,
            rebalance=rebalance
        )
