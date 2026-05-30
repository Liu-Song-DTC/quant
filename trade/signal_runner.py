"""实盘信号生成 — 从 bt_execution 提取的轻量版

关键区别 vs bt_execution:
- 只加载最近2年数据（不需要全历史）
- 只为最新日期生成目标持仓
- 不需要Backtrader框架
- 信号生成使用单进程（factor_preparer 已使用 fork Pool, 避免嵌套 fork）
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


_ensure_strategy_path()
from core.monitor import monitor, get_logger


class SignalRunner:
    """实盘信号生成器"""

    MIN_PRICE = 2.0
    MIN_VOLUME = 100

    def __init__(self, bt_data_dir: str, fund_data_dir: str):
        _ensure_strategy_path()
        from core.config_loader import load_config

        self.bt_data_dir = bt_data_dir
        self.fund_data_dir = fund_data_dir
        self.config = load_config()

        # 加载数据
        self.stock_data_dict = {}
        self.prices = {}
        self._load_data()

    def prepare(self, exposure: float = 1.0, peak_equity: float = 0.0):
        """加载策略、生成信号"""
        from core.strategy import Strategy
        from core.fundamental import FundamentalData

        # 加载基本面数据
        self._stock_codes = [n for n in self.stock_data_dict if n != "sh000001"]
        self._fundamental_data = None
        if self.fund_data_dir and os.path.exists(self.fund_data_dir):
            self._fundamental_data = FundamentalData(self.fund_data_dir + "/", stock_codes=self._stock_codes)

        # 初始化策略（max_position由PortfolioConstructor自动计算）
        # Fix P2: 尝试创建情绪编排器 (与回测一致)
        sentiment_orch = None
        try:
            sentiment_cfg = self.config.config.get('industry_sentiment', {})
            if sentiment_cfg.get('enabled'):
                from strategy.sentiment.orchestrator import SentimentOrchestrator
                sentiment_orch = SentimentOrchestrator(self.config, backtest_mode=False)
                print("情绪编排器已启用")
        except Exception as e:
            print(f"情绪编排器跳过: {e}")

        self.strategy = Strategy(
            init_cash=self.config.get('backtest.cash', 100000),
            fundamental_data=self._fundamental_data,
            sentiment_orchestrator=sentiment_orch,
        )

        # 恢复持久化的 EMA 平滑状态
        if hasattr(self.strategy, 'portfolio'):
            self.strategy.portfolio.current_exposure = exposure
            if peak_equity > 0:
                self.strategy.portfolio.peak_equity = peak_equity

        monitor.memory("after data load")

        # 生成市场状态 (Fix#6: 加载辅助指数用于风格检测)
        if "sh000001" in self.stock_data_dict:
            small_cap_df = self.stock_data_dict.get("000852")  # 中证1000
            growth_df = self.stock_data_dict.get("399006")     # 创业板指
            self.strategy.generate_market_regime(
                self.stock_data_dict["sh000001"],
                small_cap_df=small_cap_df,
                growth_df=growth_df,
            )
            self.strategy.signal_engine.set_market_regime(self.strategy.index_data)
            _logger = get_logger("signal_runner")
            _logger.info(f"市场状态已生成，共 {len(self.strategy.index_data)} 条记录"
                        f" (辅助指数: 中证1000={'✓' if small_cap_df is not None else '✗'}, "
                        f"创业板指={'✓' if growth_df is not None else '✗'})")

        # Fix P2: 加载ML模型 (如果存在)
        ml_config = self.config.config.get('ml', {})
        if ml_config.get('enabled'):
            try:
                from core.ml_predictor import MLFactorPredictor
                model_dir = ml_config.get('model_dir', 'strategy/models')
                model_path = os.path.join(str(ROOT), model_dir, 'xgb_strategy_model.json')
                if os.path.exists(model_path):
                    ml_predictor = MLFactorPredictor(self.config.config)
                    ml_predictor.load_model(model_path)
                    self.strategy.signal_engine.set_ml_predictor(ml_predictor)
                    print(f"ML模型已加载: {model_path}")
                else:
                    print(f"ML模型未找到: {model_path}, 跳过ML混合")
            except Exception as e:
                print(f"ML加载失败: {e}, 跳过ML混合")

        # 准备动态因子数据
        self._setup_dynamic_factors(self._stock_codes, self._fundamental_data)

        # 生成信号
        t0 = __import__('time').time()
        self._generate_all_signals(self._stock_codes)
        gen_elapsed = __import__('time').time() - t0
        monitor.timings['signal_generation'].append(gen_elapsed)
        monitor.count('stocks_processed', len(self._stock_codes))
        monitor.memory("after signal gen")

        # 板块轮动分析 (在信号生成之后，组合构建之前)
        self._sector_rotation = None
        try:
            from core.sector_rotation import SectorRotation
            industry_codes = getattr(self.strategy.signal_engine, 'industry_codes', None)
            if industry_codes:
                sr = SectorRotation()
                # 1. 行业动量（当日行情，滞后验证）
                sr.compute(self.stock_data_dict, industry_codes)
                # 2. 买入信号密度（领先指标，预判轮动）
                buy_signals = []
                industry_counts = {ind: len(codes) for ind, codes in industry_codes.items()}
                store = getattr(self.strategy, 'signal_store', None)
                if store:
                    for (code, _), sig in store._store.items():
                        if sig and sig.buy:
                            ind = getattr(sig, 'industry', '')
                            if ind:
                                buy_signals.append((code, ind))
                sr.compute_signal_density(buy_signals, industry_counts)
                if sr.is_ready():
                    self._sector_rotation = sr
                    if hasattr(self.strategy, 'portfolio'):
                        self.strategy.portfolio.set_sector_rotation(sr)
                    _logger = get_logger("signal_runner")
                    sig_top = sorted(sr._signal_density.items(), key=lambda x:-x[1])[:3] if hasattr(sr, '_signal_density') and sr._signal_density else []
                    _logger.info(f"板块轮动: 动量领涨={sr.get_strong_sectors(3)}, "
                               f"信号密度领涨={[s[0] for s in sig_top]}")
        except Exception as e:
            self._sector_rotation = None

    def set_sentiment_multipliers(self, multipliers: dict):
        """注入行业情绪乘数到组合构建器"""
        if hasattr(self.strategy, 'portfolio') and multipliers:
            self.strategy.portfolio.set_sentiment_multipliers(multipliers)

    def _load_data(self):
        """加载股票数据"""
        print(f"加载数据: {self.bt_data_dir}")
        if not os.path.exists(self.bt_data_dir):
            print(f"数据目录不存在: {self.bt_data_dir}")
            return

        min_date = (datetime.today() - timedelta(days=730)).strftime("%Y-%m-%d")

        load_errors = 0
        for item in os.listdir(self.bt_data_dir):
            # 跳过 macOS 隐藏文件
            if item.startswith('._'):
                continue
            if item.endswith('_qfq.csv'):
                name = item[:-8]
            elif item.endswith('_hfq.csv'):
                name = item[:-8]
            else:
                continue

            # 剔除科创板 (688xxx)
            if name.startswith('688'):
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
            except Exception:
                load_errors += 1

        if load_errors > 0:
            print(f"数据加载: {load_errors} 个文件失败")

        print(f"已加载 {len(self.stock_data_dict)} 只股票数据，{len(self.prices)} 只有最新价格")

        # 应用股票池过滤（与回测对齐）
        from core.stock_pool import get_stock_pool
        pool = get_stock_pool(data_dir=self.bt_data_dir)
        pool.discard('sh000001')
        before = len(self.stock_data_dict)
        self.stock_data_dict = {k: v for k, v in self.stock_data_dict.items() if k in pool}
        self.prices = {k: v for k, v in self.prices.items() if k in pool}
        print(f"股票池过滤: {before} -> {len(self.stock_data_dict)} 只 (质量筛选)")

    def _setup_dynamic_factors(self, stock_codes, fundamental_data):
        """设置动态因子（如果启用）"""
        factor_mode = self.config.config.get('factor_mode', 'both')  # Fix: 与bt_execution.py统一
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

            # 释放 factor_df（预计算后不再需要，避免 OOM）
            self.strategy.signal_engine.dynamic_factor_selector.factor_df = None
            import gc
            gc.collect()
            print("已释放 factor_df 内存")

    def _generate_all_signals(self, stock_codes):
        """为所有股票生成信号 — 单进程 latest_only 模式

        仅计算最新一根K线的复杂标量（_select_factor/_get_chan_boost等），
        其余bar仅做向量化运算。portfolio.build() 只读取最新日期信号，
        历史信号对实盘无用。

        性能: latest_only 跳过了 ~99.8% 的逐bar方法调用（1/500 vs 500/500），
        预期从原 30-60s 降至 2-5s。
        """
        from tqdm import tqdm

        valid_codes = [c for c in stock_codes if c in self.stock_data_dict]
        if len(valid_codes) == 0:
            print("无有效股票，跳过信号生成")
            return

        print(f"信号生成: {len(valid_codes)} 只 (单进程, latest_only)")

        for code in tqdm(valid_codes, desc="生成信号", unit="只"):
            data = self.stock_data_dict[code]
            self.strategy.generate_signal(code, data, latest_only=True)

        total = len(self.strategy.signal_store._store)
        print(f"信号生成完成: {total} 条")

    def _should_rebalance(self, latest_date, market_regime: int = 0) -> bool:
        """Fix P2: 与回测一致的周期性调仓检查

        读取 portfolio_state.json 中的 last_rebalance_date,
        如果距上次调仓 >= 调仓周期, 返回 True。
        调仓周期: 牛市30天, 震荡20天, 熊市15天 (与回测动态周期一致)
        """
        import json
        state_file = ROOT / "trade" / "portfolio_state.json"
        # 从 factor_config 读取动态调仓周期（与回测 bt_execution 一致）
        from core.config_loader import load_config
        dyn_cfg = load_config().get('dynamic_rebalance', {})
        rebalance_days = {
            1: dyn_cfg.get('bull_period', 30),
            0: dyn_cfg.get('neutral_period', 20),
            -1: dyn_cfg.get('bear_period', 15),
        }.get(market_regime, 20)

        try:
            if state_file.exists():
                with open(state_file) as f:
                    state = json.load(f)
                last_str = state.get("last_rebalance_date", "")
                if last_str:
                    from datetime import date as dt_date
                    last_date = dt_date.fromisoformat(last_str)
                    days_since = (latest_date - last_date).days
                    if days_since < rebalance_days:
                        return False
        except Exception:
            pass
        return True

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

        # 可交易股票池 (Fix P3: 加入涨跌停检查)
        tradable_universe = []
        for code, price in self.prices.items():
            if code == "sh000001":
                continue
            if price < self.MIN_PRICE:
                continue
            if code in self.stock_data_dict:
                data = self.stock_data_dict[code]
                last_row = data.iloc[-1]
                last_vol = last_row.get('volume', 0)
                if last_vol < self.MIN_VOLUME:
                    continue
                # Fix P3: 涨跌停检查 — 涨停不能买, 跌停不能卖
                if 'pct_chg' in data.columns:
                    pct_chg = float(last_row.get('pct_chg', 0) or 0)
                    if pct_chg >= 9.5:   # 涨停板 → 不可买入
                        continue
                    if pct_chg <= -9.5:  # 跌停板 → 不可买入(但持仓可卖出)
                        if code not in current_positions:
                            continue
            tradable_universe.append(code)

        print(f"\n生成目标持仓: 日期={latest_date}, 股票池={len(tradable_universe)}")

        # 获取市场状态
        market_regime = 0
        momentum_score = 0.0
        bear_risk = False
        bear_risk_fast = False
        trend_score = 0.0
        if self.strategy.index_data is not None:
            row = self.strategy.index_data[self.strategy.index_data["datetime"].dt.date == latest_date]
            if not row.empty:
                market_regime = int(row["regime"].values[0])
                momentum_score = float(row["momentum_score"].values[0])
                bear_risk = bool(row["bear_risk"].values[0]) if "bear_risk" in row.columns else False
                bear_risk_fast = bool(row["bear_risk_fast"].values[0]) if "bear_risk_fast" in row.columns else False
                trend_score = float(row["trend_score"].values[0]) if "trend_score" in row.columns else 0.0

        # Fix P2: 与回测一致的周期性调仓 (默认20天, 市场状态动态调整)
        rebalance = self._should_rebalance(latest_date, market_regime)

        # 生成目标持仓
        t0 = __import__('time').time()
        target = self.strategy.generate_positions(
            date=latest_date,
            universe=tradable_universe,
            current_positions=current_positions,
            cash=cash,
            prices=self.prices,
            cost=cost or {},
            rebalance=rebalance,
        )
        portfolio_elapsed = __import__('time').time() - t0
        monitor.timings['portfolio_build'].append(portfolio_elapsed)
        monitor.count('candidates_in_universe', len(tradable_universe))
        if target:
            monitor.count('positions_selected', len(target))

        # 获取选股明细（含缠论+因子详情）
        selections = []
        if hasattr(self.strategy.portfolio, 'last_selection'):
            for s in self.strategy.portfolio.last_selection:
                selections.append({
                    "code": s["code"],
                    "score": s["score"],
                    "weight": s["weight"],
                    "industry": s.get("industry", ""),
                    # 选股理由字段
                    "factor_name": s.get("factor_name", ""),
                    "factor_value": s.get("factor_value", 0),
                    "chan_buy_point": s.get("chan_buy_point", 0),
                    "chan_sell_point": s.get("chan_sell_point", 0),
                    "chan_buy_strength": s.get("chan_buy_strength", 0.0),
                    "trend_type": s.get("trend_type", 0),
                    "signal_level": s.get("signal_level", 0),
                    "chan_bonus": s.get("chan_bonus", 0.0),
                    "effective_score": s.get("effective_score", s["score"]),
                    "rank_pct": s.get("rank_pct", 0),
                    "confidence": s.get("confidence", 1.0),
                })

        regime_names = {1: "牛市", 0: "震荡", -1: "熊市"}
        print(f"市场状态: {regime_names.get(market_regime, '未知')}, "
              f"momentum={momentum_score:.2f}, bear_risk={bear_risk}")
        print(f"目标持仓: {len(target)} 只股票 (调仓={'是' if rebalance else '否'})")

        # 分类输出：新买入 / 继续持有 / 卖出
        buys, holds, sells = [], [], []
        for code, value in sorted(target.items(), key=lambda x: -x[1]):
            was_held = code in current_positions
            if value > 0 and not was_held:
                buys.append((code, value))
            elif value > 0 and was_held:
                holds.append((code, value))
            elif was_held:
                sells.append((code, current_positions.get(code, 0)))
            else:
                sells.append((code, 0))

        if buys:
            print(f"\n  [买入] {len(buys)} 只:")
            for code, v in buys:
                print(f"    {code}: ¥{v:,.0f}")
        if holds:
            print(f"\n  [持有] {len(holds)} 只:")
            for code, v in holds:
                print(f"    {code}: ¥{v:,.0f}")
        if sells:
            print(f"\n  [卖出] {len(sells)} 只:")
            for code, old_val in sells:
                print(f"    {code}: 原市值 ¥{old_val:,.0f} → 清仓")

        # 持久化监控数据（跨交易日不丢失）
        monitor.save_report()

        return {
            "target_positions": target,
            "prices": dict(self.prices),
            "market_regime": {
                "regime": market_regime,
                "momentum_score": momentum_score,
                "bear_risk": bear_risk,
                "bear_risk_fast": bear_risk_fast,
                "trend_score": trend_score,
            },
            "selections": selections,
            "date": str(latest_date),
            "monitor": monitor.report(),
        }
