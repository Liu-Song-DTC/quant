import pandas as pd

from .signal_engine import SignalEngine
from .signal_store import SignalStore
from .portfolio import PortfolioConstructor
from .market_regime_detector import MarketRegimeDetector
from .config_loader import load_config


class Strategy:
    """带股灾检测的市场状态判断"""

    def __init__(self, init_cash, fundamental_data=None):
        self.signal_engine = SignalEngine()
        self.fundamental_data = fundamental_data
        if fundamental_data:
            self.signal_engine.set_fundamental_data(fundamental_data)

        self.portfolio = PortfolioConstructor()
        self.market_regime = []
        self.signal_store = SignalStore()
        self.init_cash = init_cash

        self.index_data = None

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
