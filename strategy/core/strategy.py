import pandas as pd
import numpy as np
from typing import Optional

from .signal_engine import SignalEngine
from .signal_store import SignalStore
from .portfolio import PortfolioConstructor
from .market_regime_detector import MarketRegimeDetector
from .sector_rotation import SectorRotation
from .config_loader import load_config


class Strategy:
    """带股灾检测的市场状态判断 + 情绪分析集成"""

    def __init__(self, init_cash, fundamental_data=None, sentiment_orchestrator=None):
        self.signal_engine = SignalEngine()
        self.fundamental_data = fundamental_data
        if fundamental_data:
            self.signal_engine.set_fundamental_data(fundamental_data)

        self.portfolio = PortfolioConstructor()
        self.sector_rotation = SectorRotation()
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

    def generate_market_regime(self, index_df, small_cap_df=None, growth_df=None):
        # Fix#6: 支持多指数风格检测
        self.index_data = self.regime_detector.generate(
            index_df, small_cap_df=small_cap_df, growth_df=growth_df
        )

    def generate_signal(self, code, market_data, latest_only=False):
        self.signal_engine.generate(code, market_data, self.signal_store, latest_only=latest_only)

    def set_sentiment_multipliers(self, date, market_regime: int = 0):
        """从情绪编排器获取行业情绪权重并注入组合构建器"""
        if self.sentiment_orchestrator is None:
            return
        try:
            multipliers = self.sentiment_orchestrator.get_sentiment_weights(
                market_regime=market_regime,
                current_date=date,
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
        bear_risk_fast = False
        trend_score = 0.0
        if self.index_data is not None:
            row = self.index_data[self.index_data["datetime"].dt.date == date]
            if not row.empty:
                market_regime = int(row["regime"].values[0])
                momentum_score = float(row["momentum_score"].values[0])
                bear_risk = bool(row["bear_risk"].values[0]) if "bear_risk" in row.columns else False
                bear_risk_fast = bool(row["bear_risk_fast"].values[0]) if "bear_risk_fast" in row.columns else False
                trend_score = float(row["trend_score"].values[0]) if "trend_score" in row.columns else 0.0

        # 再平衡日更新情绪权重
        if rebalance:
            self.set_sentiment_multipliers(date, market_regime)

        # 注入板块轮动分析器（含信号密度领先指标）
        self.portfolio._sector_rotation = self.sector_rotation

        # 计算当日买入信号密度（领先指标，无滞后）
        if rebalance and hasattr(self, 'industry_stock_counts'):
            buy_signals = []
            for code in universe:
                sig = self.signal_store.get(code, date)
                if sig and sig.buy:
                    ind = getattr(sig, 'industry', '')
                    buy_signals.append((code, ind))
            self.sector_rotation.compute_signal_density(
                buy_signals, self.industry_stock_counts
            )

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
            bear_risk_fast=bear_risk_fast,
            trend_score=trend_score,
            cost=cost,
            rebalance=rebalance
        )
