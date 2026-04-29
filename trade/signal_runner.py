"""实盘信号生成 — 从 bt_execution 提取的轻量版

关键区别 vs bt_execution:
- 只加载最近2年数据（不需要全历史）
- 只为最新日期生成目标持仓
- 不需要Backtrader框架
- 不需要多进程（股票池~500只，单进程足够）
"""
import os
import sys
import pandas as pd
from datetime import datetime, timedelta

from .config import ROOT


def _ensure_strategy_path():
    p = str(ROOT / "strategy")
    if p not in sys.path:
        sys.path.insert(0, p)


class SignalRunner:
    """实盘信号生成器"""

    MIN_PRICE = 2.0
    MIN_VOLUME = 100

    def __init__(self, bt_data_dir: str, fund_data_dir: str, max_position: int = 10):
        _ensure_strategy_path()
        from core.config_loader import load_config

        self.bt_data_dir = bt_data_dir
        self.fund_data_dir = fund_data_dir
        self.config = load_config()
        self.max_position = max_position

        # 加载数据
        self.stock_data_dict = {}
        self.prices = {}
        self._load_data()

    def prepare(self, max_position: int = None, exposure: float = 1.0, peak_equity: float = 0.0):
        """加载策略、生成信号（在知道max_position后调用）"""
        if max_position is not None:
            self.max_position = max_position

        from core.strategy import Strategy
        from core.fundamental import FundamentalData

        # 加载基本面数据
        stock_codes = [n for n in self.stock_data_dict if n != "sh000001"]
        fundamental_data = None
        if self.fund_data_dir and os.path.exists(self.fund_data_dir):
            fundamental_data = FundamentalData(self.fund_data_dir + "/", stock_codes=stock_codes)

        # 初始化策略
        self.strategy = Strategy(
            init_cash=self.config.get('backtest.cash', 100000),
            max_position=self.max_position,
            fundamental_data=fundamental_data,
        )

        # 恢复持久化的组合状态(exposure平滑值, peak_equity)
        if hasattr(self.strategy, 'portfolio'):
            self.strategy.portfolio.current_exposure = exposure
            if peak_equity > 0:
                self.strategy.portfolio.peak_equity = peak_equity

        # 生成市场状态
        if "sh000001" in self.stock_data_dict:
            self.strategy.generate_market_regime(self.stock_data_dict["sh000001"])
            self.strategy.signal_engine.set_market_regime(self.strategy.index_data)
            print(f"市场状态已生成，共 {len(self.strategy.index_data)} 条记录")

        # 准备动态因子数据
        self._setup_dynamic_factors(stock_codes, fundamental_data)

        # 生成信号
        self._generate_all_signals(stock_codes)

    def _load_data(self):
        """加载股票数据"""
        print(f"加载数据: {self.bt_data_dir}")
        if not os.path.exists(self.bt_data_dir):
            print(f"数据目录不存在: {self.bt_data_dir}")
            return

        min_date = (datetime.today() - timedelta(days=730)).strftime("%Y-%m-%d")

        for item in os.listdir(self.bt_data_dir):
            if item.endswith('_qfq.csv'):
                name = item[:-8]
            elif item.endswith('_hfq.csv'):
                name = item[:-8]
            else:
                continue

            filepath = os.path.join(self.bt_data_dir, item)
            try:
                data = pd.read_csv(filepath, parse_dates=['datetime'])
                data = data[data['datetime'] >= min_date]
                if len(data) > 20:
                    self.stock_data_dict[name] = data
                    last_row = data.iloc[-1]
                    if last_row['close'] > 0 and last_row.get('volume', 0) > 0:
                        self.prices[name] = float(last_row['close'])
            except Exception as e:
                print(f"加载 {name} 失败: {e}")

        print(f"已加载 {len(self.stock_data_dict)} 只股票数据，{len(self.prices)} 只有最新价格")

    def _setup_dynamic_factors(self, stock_codes, fundamental_data):
        """设置动态因子（如果启用）"""
        factor_mode = self.config.config.get('factor_mode', 'fixed')
        if factor_mode == 'fixed':
            print("因子模式: fixed，跳过动态因子")
            return

        print(f"准备因子数据 (factor_mode={factor_mode})...")
        from core.factor_preparer import prepare_factor_data
        from core.industry_mapping import INDUSTRY_KEYWORDS

        num_workers = self.config.get('backtest.num_workers', 4)
        factor_df, industry_codes, all_dates = prepare_factor_data(
            self.stock_data_dict,
            fundamental_data,
            INDUSTRY_KEYWORDS,
            num_workers,
        )
        self.strategy.set_factor_data(factor_df, industry_codes)
        print(f"因子模式: {factor_mode}, {len(industry_codes)} 个行业")

        if hasattr(self.strategy.signal_engine, 'dynamic_factor_selector') and \
           self.strategy.signal_engine.dynamic_factor_selector.enabled:
            print("预计算因子选择...")
            self.strategy.signal_engine.dynamic_factor_selector.precompute_all_factor_selections(
                progress_callback=lambda curr, total: print(f"\r因子选择进度: {curr}/{total}", end="", flush=True),
                num_workers=num_workers,
            )
            cache = self.strategy.signal_engine.dynamic_factor_selector._factor_cache
            print(f"\n因子选择预计算完成，共 {len(cache)} 个日期")

    def _generate_all_signals(self, stock_codes):
        """为所有股票生成信号"""
        from core.signal_store import SignalStore

        print(f"生成信号 ({len(stock_codes)} 只股票)...")
        for code in stock_codes:
            if code not in self.stock_data_dict:
                continue
            data = self.stock_data_dict[code]
            self.strategy.generate_signal(code, data)

        total = len(self.strategy.signal_store._store)
        print(f"信号生成完成: {total} 条")

    def get_prices(self) -> dict:
        """获取当前价格 {code: price}"""
        return dict(self.prices)

    def run(self, current_positions: dict, cash: float, cost: dict = None) -> dict:
        """为最新日期生成目标持仓

        Args:
            current_positions: {code: market_value}
            cash: 可用现金
            cost: {code: [shares, avg_cost]}

        Returns:
            {
                "target_positions": {code: target_value},
                "prices": {code: price},
                "market_regime": {"regime": int, "momentum_score": float, "bear_risk": bool, "trend_score": float},
                "selections": [{code, score, weight, industry}],
                "date": str,
            }
        """
        # 获取最新日期
        if "sh000001" in self.stock_data_dict:
            latest_date = self.stock_data_dict["sh000001"]["datetime"].max().date()
        elif self.stock_data_dict:
            first_key = next(iter(self.stock_data_dict))
            latest_date = self.stock_data_dict[first_key]["datetime"].max().date()
        else:
            print("没有可用数据")
            return None

        # 可交易股票池
        tradable_universe = []
        for code, price in self.prices.items():
            if code == "sh000001":
                continue
            if price < self.MIN_PRICE:
                continue
            if code in self.stock_data_dict:
                last_vol = self.stock_data_dict[code].iloc[-1].get('volume', 0)
                if last_vol < self.MIN_VOLUME:
                    continue
            tradable_universe.append(code)

        print(f"\n生成目标持仓: 日期={latest_date}, 股票池={len(tradable_universe)}")

        # 获取市场状态
        market_regime = 0
        momentum_score = 0.0
        bear_risk = False
        trend_score = 0.0
        if self.strategy.index_data is not None:
            row = self.strategy.index_data[self.strategy.index_data["datetime"].dt.date == latest_date]
            if not row.empty:
                market_regime = int(row["regime"].values[0])
                momentum_score = float(row["momentum_score"].values[0])
                bear_risk = bool(row["bear_risk"].values[0]) if "bear_risk" in row.columns else False
                trend_score = float(row["trend_score"].values[0]) if "trend_score" in row.columns else 0.0

        # 实盘中：每次运行都视为再平衡日（由用户决定是否执行）
        rebalance = True

        # 生成目标持仓
        target = self.strategy.generate_positions(
            date=latest_date,
            universe=tradable_universe,
            current_positions=current_positions,
            cash=cash,
            prices=self.prices,
            cost=cost or {},
            rebalance=rebalance,
        )

        # 获取选股明细
        selections = []
        if hasattr(self.strategy.portfolio, 'last_selection'):
            for s in self.strategy.portfolio.last_selection:
                selections.append({
                    "code": s["code"],
                    "score": s["score"],
                    "weight": s["weight"],
                    "industry": s.get("industry", ""),
                })

        regime_names = {1: "牛市", 0: "震荡", -1: "熊市"}
        print(f"市场状态: {regime_names.get(market_regime, '未知')}, "
              f"momentum={momentum_score:.2f}, bear_risk={bear_risk}")
        print(f"目标持仓: {len(target)} 只股票")
        for code, value in sorted(target.items(), key=lambda x: -x[1]):
            print(f"  {code}: ¥{value:,.0f}")

        return {
            "target_positions": target,
            "prices": dict(self.prices),
            "market_regime": {
                "regime": market_regime,
                "momentum_score": momentum_score,
                "bear_risk": bear_risk,
                "trend_score": trend_score,
            },
            "selections": selections,
            "date": str(latest_date),
        }
