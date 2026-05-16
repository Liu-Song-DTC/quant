import backtrader as bt
import pandas as pd
import datetime
import os
from tqdm import tqdm
import math
from collections import defaultdict
import multiprocessing
from functools import partial

from core.strategy import Strategy
from core.fundamental import FundamentalData
from core.signal_engine import SignalEngine
from core.factor_preparer import prepare_factor_data
from core.signal_store import SignalStore
from core.config_loader import load_config
from core.industry_mapping import INDUSTRY_KEYWORDS
from core.market_regime_detector import MarketRegimeDetector
from core.stock_pool import get_stock_pool, get_exclusion_set

# 情绪分析集成
try:
    from sentiment.orchestrator import SentimentOrchestrator
    SENTIMENT_AVAILABLE = True
except ImportError:
    SENTIMENT_AVAILABLE = False

# 加载配置
config = load_config()

# 回测参数 - 从配置文件读取
CASH = config.get('backtest.cash', 100000.0)
COMMISSION = config.get('backtest.commission', 0.0015)
PERC = config.get('backtest.slippage', 0.0015)
REBALANCE_DAYS = config.get('backtest.rebalance_days', 20)

# === 小说优化：动态调仓周期 ===
DYNAMIC_REBALANCE_CONFIG = config.get('dynamic_rebalance', {})
DYNAMIC_REBALANCE_ENABLED = DYNAMIC_REBALANCE_CONFIG.get('enabled', True)
REBALANCE_BULL = DYNAMIC_REBALANCE_CONFIG.get('bull_period', 30)
REBALANCE_NEUTRAL = DYNAMIC_REBALANCE_CONFIG.get('neutral_period', 20)
REBALANCE_BEAR = DYNAMIC_REBALANCE_CONFIG.get('bear_period', 15)
NUM_WORKERS = config.get('backtest.num_workers', 8)

# 数据路径 - 从配置文件读取
DATA_PATH = config.get('paths.data', '../data/stock_data/backtrader_data/')
FUNDAMENTAL_PATH = config.get('paths.fundamental', '../data/stock_data/fundamental_data/')


# 全局变量用于 worker 进程
_worker_engine = None
_worker_use_dynamic = False


def _init_worker(fundamental_path, stock_codes, use_dynamic, industry_codes, factor_cache, all_dates, regime_df,
                 ml_model_path=None, ml_preds=None):
    """Worker 进程初始化函数

    注意: 不传递 factor_df (巨大DataFrame) 到每个worker，预计算的 factor_cache 已包含所有因子选择结果。
    ml_preds 只传递当前股票相关的预测值（由调用方过滤），大幅减少内存占用。
    """
    global _worker_engine, _worker_use_dynamic
    _worker_use_dynamic = use_dynamic

    # 每个 worker 创建自己的 engine 和 fundamental_data
    _worker_engine = SignalEngine()

    if fundamental_path and os.path.exists(fundamental_path):
        fd = FundamentalData(fundamental_path, stock_codes)
        _worker_engine.set_fundamental_data(fd)

    # 设置市场状态数据（关键：用于牛市优化）
    if regime_df is not None:
        _worker_engine.set_market_regime(regime_df)

    # 使用预计算的因子选择缓存（不传递原始factor_df，节省 ~400MB/worker）
    if use_dynamic and factor_cache is not None and all_dates is not None:
        _worker_engine.set_industry_mapping(industry_codes)
        _worker_engine.dynamic_factor_selector.set_factor_cache(factor_cache, all_dates)

    # 设置ML预测（每个worker加载模型副本）
    if ml_model_path is not None and os.path.exists(ml_model_path):
        try:
            from core.ml_predictor import MLFactorPredictor
            worker_ml = MLFactorPredictor()
            worker_ml.load_model(ml_model_path)
            _worker_engine.set_ml_predictor(worker_ml)
        except Exception:
            pass  # ML加载失败不阻塞回测
    if ml_preds is not None:
        _worker_engine.set_ml_predictions(ml_preds)


def _generate_stock_signal_worker(args):
    """Worker 函数：为一个股票生成信号 — 直接从文件读取，避免 pickle 传大数据"""
    global _worker_engine, _worker_use_dynamic
    code, filepath = args

    try:
        engine = _worker_engine

        if engine is None:
            engine = SignalEngine()
            if FUNDAMENTAL_PATH and os.path.exists(FUNDAMENTAL_PATH):
                fd = FundamentalData(FUNDAMENTAL_PATH, [code])
                engine.set_fundamental_data(fd)

        store = SignalStore()
        data = pd.read_csv(filepath, parse_dates=['datetime'])
        engine.generate(code, data, store)
        return (code, store._store)
    except Exception as e:
        # 返回异常信息，避免worker静默失败
        print(f"[Worker Error] {code}: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return (code, {})


def add_data_and_signal(cerebro, strategy, fundamental_data=None):
    all_items = os.listdir(DATA_PATH)
    stock_codes = []  # 获取回测池中的股票列表

    # 只读取一次CSV数据，同时记录文件路径
    stock_data_dict = {}
    stock_file_map = {}  # code → filepath，传给worker直接从文件读
    for item in tqdm(all_items, desc="loading data"):
        if item.startswith('._'):
            continue
        filepath = DATA_PATH + item
        if item.endswith('_qfq.csv'):
            name = item[:-8]
        elif item.endswith('_hfq.csv'):
            name = item[:-8]
        else:
            continue
        data = pd.read_csv(filepath, parse_dates=['datetime'])
        stock_data_dict[name] = data
        stock_file_map[name] = filepath

    # === 股票池过滤 ===
    stock_pool_enabled = config.get('stock_pool.enabled', True)
    if stock_pool_enabled:
        stock_pool = get_stock_pool()
        pool_codes = stock_pool | {'sh000001'}
        before_count = len(stock_data_dict)
        stock_data_dict = {k: v for k, v in stock_data_dict.items() if k in pool_codes}
        stock_file_map = {k: v for k, v in stock_file_map.items() if k in pool_codes}
        after_count = len(stock_data_dict)
        print(f"股票池过滤: {before_count} -> {after_count} 只 (全部通过质量筛选)")
    else:
        print(f"股票池过滤: 已关闭，使用全市场 {len(stock_data_dict)} 只股票")

    # === 剔除科创板 ===
    star_codes = {k for k in stock_data_dict if k.startswith('688')}
    before_excl = len(stock_data_dict)
    stock_data_dict = {k: v for k, v in stock_data_dict.items() if k not in star_codes}
    stock_file_map = {k: v for k, v in stock_file_map.items() if k not in star_codes}
    print(f"科创板过滤: {before_excl} -> {len(stock_data_dict)} 只")

    # 从任意一个DataFrame获取日期（所有股票数据共享日历）
    dates = set()
    for data in stock_data_dict.values():
        dates.update(data['datetime'])
    calendar_index = pd.DatetimeIndex(sorted(dates))
    del dates

    # 处理指数数据 - 生成市场状态
    regime_df = None
    if "sh000001" in stock_data_dict:
        strategy.generate_market_regime(stock_data_dict["sh000001"])
        # 获取regime_df用于传递给worker进程
        regime_df = strategy.index_data
        print(f"市场状态数据已生成，共 {len(regime_df)} 条记录")

    # 准备动态因子数据
    factor_mode = config.config.get('factor_mode', 'both')
    factor_df = None
    industry_codes = {}
    # 只有当 factor_mode 不是 'fixed' 时才需要IC计算
    # reweight模式也需要IC计算（用于动态调整权重）
    if factor_mode != 'fixed':
        print(f"准备因子数据 (factor_mode={factor_mode})...")
        # 获取股票代码列表（排除指数）
        stock_codes = [name for name in stock_data_dict.keys() if name != "sh000001"]
        factor_df, industry_codes, all_dates = prepare_factor_data(
            stock_data_dict,
            fundamental_data,
            INDUSTRY_KEYWORDS,
            NUM_WORKERS
        )
        strategy.set_factor_data(factor_df, industry_codes)
        print(f"因子模式: {factor_mode}, {len(industry_codes)} 个行业")
    else:
        print(f"跳过IC计算: factor_mode={factor_mode} (fixed模式)")

    # === ML预测层训练 ===
    ml_config = config.config.get('ml', {})
    _ml_model_path = None
    _ml_preds = {}
    if ml_config.get('enabled', False) and factor_df is not None:
        try:
            from core.ml_predictor import MLFactorPredictor
            print("训练XGBoost预测模型...")
            ml_predictor = MLFactorPredictor(config.config)
            val_ic = ml_predictor.train(factor_df)
            if val_ic is not None and val_ic > 0:
                # 保存模型到项目目录（避免临时目录被清理）
                strategy_dir = os.path.dirname(os.path.abspath(__file__))
                model_dir = os.path.join(strategy_dir, 'models')
                os.makedirs(model_dir, exist_ok=True)
                _ml_model_path = os.path.join(model_dir, 'xgb_strategy_model.json')
                ml_predictor.save_model(_ml_model_path)
                print(f"ML模型已保存: {_ml_model_path}")

                # 生成所有日期/股票的ML预测（主进程）
                print("生成ML预测...")
                all_dates_sorted = sorted(factor_df['date'].unique())
                for date in tqdm(all_dates_sorted, desc="ML预测"):
                    date_df = factor_df[factor_df['date'] == date]
                    if len(date_df) == 0:
                        continue
                    preds = ml_predictor.predict(date_df.copy())
                    for i, (_, row) in enumerate(date_df.iterrows()):
                        _ml_preds[(row['code'], date)] = preds[i]
                print(f"ML预测完成: {len(_ml_preds)} 条预测")
            else:
                print(f"[ML] 验证IC不足({val_ic}), 跳过ML预测")
        except ImportError:
            print("[ML] xgboost未安装，跳过ML预测")
        except Exception as e:
            print(f"[ML] 训练失败: {e}")
            import traceback
            traceback.print_exc()

    # 生成信号（多进程并行）
    # use_dynamic 表示是否使用动态因子选择器
    use_dynamic = factor_mode != 'fixed'
    stock_codes = [name for name in stock_data_dict.keys() if name != "sh000001"]

    # 创建带动态因子的 SignalEngine（如果启用）
    main_engine = None
    if use_dynamic:
        main_engine = SignalEngine()
        main_engine.set_factor_data(factor_df)
        main_engine.set_industry_mapping(industry_codes)
        main_engine.set_fundamental_data(fundamental_data)
        # 设置ML预测值和模型（用于主引擎）
        if _ml_model_path is not None and _ml_preds:
            from core.ml_predictor import MLFactorPredictor
            ml_predictor = MLFactorPredictor(config.config)
            ml_predictor.load_model(_ml_model_path)
            main_engine.set_ml_predictor(ml_predictor)
            main_engine.set_ml_predictions(_ml_preds)
        print(f"主引擎已设置动态因子数据")

        # 预计算所有日期的因子选择（避免多进程中重复计算）
        print("预计算因子选择...")
        main_engine.dynamic_factor_selector.precompute_all_factor_selections(
            progress_callback=lambda curr, total: print(f"\r因子选择进度: {curr}/{total}", end="", flush=True),
            num_workers=NUM_WORKERS
        )
        print(f"\n因子选择预计算完成，共 {len(main_engine.dynamic_factor_selector._factor_cache)} 个日期")

        # 提取预计算的缓存传递给workers
        precomputed_cache = main_engine.dynamic_factor_selector._factor_cache
        precomputed_all_dates = main_engine.dynamic_factor_selector._all_dates_cache

        # === 释放 factor_df 以避免 fork 复制到 worker 进程 ===
        # factor_df 约 2-3GB，fork 后每个 worker 复制一份 → 4 workers = 8-12GB
        # 预计算完成后 factor_cache 已包含所有需要的因子选择结果，不再需要 factor_df
        strategy.signal_engine.dynamic_factor_selector.factor_df = None
        main_engine.dynamic_factor_selector.factor_df = None
        del factor_df
        factor_df = None
        import gc
        gc.collect()
        print("已释放 factor_df 内存（避免 fork 复制到 worker 进程）")
    else:
        precomputed_cache = None
        precomputed_all_dates = None

    # 准备参数：传文件路径给worker（而非dict），避免内存翻倍
    stock_items = [
        (name, stock_file_map[name])
        for name in stock_data_dict if name != "sh000001"
    ]

    # 增量写入信号CSV（而非内存中累积 ~1.3GB 的 all_signals 列表）
    strategy_dir = os.path.dirname(os.path.abspath(__file__))
    signals_output_path = os.path.join(strategy_dir, 'rolling_validation_results', 'backtest_signals.csv')
    os.makedirs(os.path.dirname(signals_output_path), exist_ok=True)
    signal_csv = open(signals_output_path, 'w', encoding='utf-8')
    signal_csv.write('code,date,buy,sell,score,factor_value,factor_name,industry,factor_quality,'
                     'chan_divergence_type,chan_divergence_strength,chan_structure_score,'
                     'chan_buy_point,chan_sell_point,signal_level,trend_type,'
                     'chan_pivot_zg,chan_pivot_zd\n')
    signal_count = [0]  # 用list实现闭包写入计数

    # 多进程并行生成信号
    print(f"多进程生成信号 ({NUM_WORKERS} workers)...")

    # 动态因子统计
    dynamic_factor_stats = {'hit': 0, 'miss': 0, 'factor_names': {}}

    ctx = multiprocessing.get_context('fork')
    with ctx.Pool(
        processes=NUM_WORKERS,
        initializer=_init_worker,
        initargs=(FUNDAMENTAL_PATH, stock_codes, use_dynamic, industry_codes, precomputed_cache, precomputed_all_dates, regime_df,
                  _ml_model_path, _ml_preds)
    ) as pool:
        for result in tqdm(
            pool.imap_unordered(_generate_stock_signal_worker, stock_items, chunksize=10),
            total=len(stock_items),
            desc="generating signals"
        ):
            code, store_data = result
            # 写入 signal_store
            for (c, date), signal in store_data.items():
                if hasattr(date, 'date'):
                    date = date.date()
                strategy.signal_store.set(c, date, signal)
            # 增量写入CSV（避免内存中累积上千万条信号）
            for (c, date), sig in store_data.items():
                if hasattr(date, 'date'):
                    date = date.date()
                signal_csv.write(
                    f'{c},{date},{sig.buy},{sig.sell},{sig.score},{sig.factor_value},'
                    f'{sig.factor_name},{sig.industry},'
                    f'{getattr(sig, "factor_quality", 0.0)},'
                    f'{getattr(sig, "chan_divergence_type", "")},'
                    f'{getattr(sig, "chan_divergence_strength", 0.0)},'
                    f'{getattr(sig, "chan_structure_score", 0.0)},'
                    f'{getattr(sig, "chan_buy_point", 0)},'
                    f'{getattr(sig, "chan_sell_point", 0)},'
                    f'{getattr(sig, "signal_level", 0)},'
                    f'{getattr(sig, "trend_type", 0)},'
                    f'{getattr(sig, "chan_pivot_zg", float("nan"))},'
                    f'{getattr(sig, "chan_pivot_zd", float("nan"))}\n'
                )
                signal_count[0] += 1
                # 动态因子统计
                if sig.factor_name and sig.factor_name.startswith('DYN_'):
                    dynamic_factor_stats['hit'] += 1
                    fn = sig.factor_name.split('_')[1] if '_' in sig.factor_name else sig.factor_name
                    dynamic_factor_stats['factor_names'][fn] = dynamic_factor_stats['factor_names'].get(fn, 0) + 1
                else:
                    dynamic_factor_stats['miss'] += 1

    signal_csv.close()
    print(f"信号数据已保存: {signal_count[0]} 条 -> {signals_output_path}")

    # 释放不再需要的大对象
    del stock_items
    import gc
    gc.collect()

    # 打印动态因子统计
    total = dynamic_factor_stats['hit'] + dynamic_factor_stats['miss']
    if total > 0:
        hit_rate = dynamic_factor_stats['hit'] / total * 100
        print(f"\n=== 动态因子统计 ===")
        print(f"动态因子命中: {dynamic_factor_stats['hit']:,} / {total:,} ({hit_rate:.1f}%)")
        print(f"非动态因子: {dynamic_factor_stats['miss']:,}")
        if dynamic_factor_stats['factor_names']:
            print("行业因子分布:")
            for fn, cnt in sorted(dynamic_factor_stats['factor_names'].items(), key=lambda x: -x[1])[:10]:
                print(f"  {fn}: {cnt:,}")

    price_cols = ['open', 'high', 'low', 'close']
    for name, data in tqdm(stock_data_dict.items(), desc="preparing datafeeds"):
        if name == "sh000001":
            continue
        data = data.set_index('datetime')
        data = data.reindex(calendar_index)
        data[price_cols] = data[price_cols].ffill()
        if 'volume' in data.columns:
            data['volume'] = data['volume'].fillna(0)
        datafeed = bt.feeds.PandasData(dataname=data)
        cerebro.adddata(datafeed, name=name)

    # 更新策略的基本面数据加载范围
    if hasattr(strategy, 'portfolio') and hasattr(strategy.portfolio, 'fundamental_data'):
        from core.fundamental import FundamentalData
        fundamental_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'data', 'stock_data', 'fundamental_data'
        )
        strategy.portfolio.fundamental_data = FundamentalData(fundamental_path + '/', stock_codes=stock_codes)
        print(f"基本面数据已限制为 {len(stock_codes)} 只股票")

    # 释放 stock_data_dict 和 stock_file_map 的内存（cerebro 已持有数据副本）
    stock_data_dict.clear()
    stock_file_map.clear()
    del stock_data_dict, stock_file_map
    import gc
    gc.collect()
    print("已释放数据加载缓存")

class BacktraderExecution(bt.Strategy):
    params = dict(
        real_strategy=None,
    )

    # 数据质量过滤参数
    MIN_PRICE = 2.0  # 最低价格限制（前复权后），避免低价股异常
    MIN_VOLUME = 100  # 最低成交量，过滤停牌股

    def __init__(self):
        self.universe = [d._name for d in self.datas]
        self.count = 0
        self.current_rebalance_period = REBALANCE_DAYS  # 动态周期，默认20天
        self.orders_list = defaultdict(list)
        self.last_date = None
        self.cost = defaultdict(list)
        self.portfolio_selections = []  # 记录每期选股结果
        self._prev_total_value = None  # 用于计算每日收益率

    def _is_tradable(self, d):
        """检查股票是否可交易（非停牌、价格正常）"""
        price = d.close[0]
        volume = d.volume[0] if hasattr(d, 'volume') else 1

        # 1. 价格必须为正
        if price is None or math.isnan(price) or price <= 0:
            return False

        # 2. 价格不能太低（前复权后）- 避免低价股异常放大收益
        if price < self.MIN_PRICE:
            return False

        # 3. 成交量必须大于0（非停牌）
        if volume is None or math.isnan(volume) or volume < self.MIN_VOLUME:
            return False

        return True

    def next(self):
        if self.last_date is not None and self.last_date in self.orders_list:
            for order in self.orders_list[self.last_date]:
                self.cancel(order)
            del self.orders_list[self.last_date]
        self.count += 1
        date = self.datas[0].datetime.date(0)
        self.last_date = date

        prices = {}
        tradable_universe = []  # 可交易的股票池
        for d in self.datas:
            if not self._is_tradable(d):
                continue
            price = d.close[0]
            prices[d._name] = price
            tradable_universe.append(d._name)

        current_positions = {
            d._name: self.getposition(d).size * prices[d._name]
            for d in self.datas
            if d._name in prices and self.getposition(d).size != 0
        }

        rebalance = False
        # === 动态调仓周期（小说：牛市捂股，熊市灵活）===
        if DYNAMIC_REBALANCE_ENABLED and self.p.real_strategy.index_data is not None:
            date_ts = pd.to_datetime(date)
            idx_row = self.p.real_strategy.index_data[
                self.p.real_strategy.index_data["datetime"].dt.date == date
            ]
            if not idx_row.empty:
                regime = int(idx_row["regime"].values[0])
                if regime == 1:
                    self.current_rebalance_period = REBALANCE_BULL
                elif regime == -1:
                    self.current_rebalance_period = REBALANCE_BEAR
                else:
                    self.current_rebalance_period = REBALANCE_NEUTRAL

        if self.count >= self.current_rebalance_period:
            self.count = 1
            rebalance = True
        target = self.p.real_strategy.generate_positions(
            date=date,
            universe=tradable_universe,  # 只传递可交易的股票
            current_positions=current_positions,
            cash=self.broker.getcash(),
            prices=prices,
            cost=self.cost,
            rebalance=rebalance,
        )

        # 记录选股结果
        if rebalance:
            selection = self.p.real_strategy.portfolio.last_selection
            for s in selection:
                self.portfolio_selections.append({
                    'date': date,
                    'code': s['code'],
                    'score': s['score'],
                    'weight': s['weight'],
                    'industry': s.get('industry', ''),
                })

        active_codes = set(target.keys()) | set(current_positions.keys())
        for d in self.datas:
            code = d._name
            if code not in active_codes:
                continue

            # 再次检查可交易性（买入时必须可交易，卖出可以放宽）
            if not self._is_tradable(d) and code not in current_positions:
                continue

            price = d.close[0]
            if price is None or math.isnan(price) or price <= 0:
                continue

            pos = self.getposition(d)
            current_value = pos.size * price
            target_value = target.get(code, 0.0)

            diff_value = target_value - current_value

            # 忽略极小调整
            if abs(diff_value) < price * 100:
                continue

            raw = diff_value / price / 100
            # A 股最小 100 股
            size = max(int(raw), 1) * 100 if raw > 0 else min(int(raw), -1) * 100

            if size > 0:
                max_affordable = int(self.broker.getcash() / price / 100) * 100
                size = min(size, max_affordable)
                if size > 0:
                    order = self.buy(data=d, size=size)
                    self.orders_list[date].append(order)
            elif size < 0:
                order = self.sell(data=d, size=size)
                self.orders_list[date].append(order)

        # 计算并记录每日收益率（供组合层波动率控制使用）
        total_value = self.broker.getvalue()
        if self._prev_total_value is not None and self._prev_total_value > 0:
            daily_return = (total_value - self._prev_total_value) / self._prev_total_value
            self.p.real_strategy.portfolio.update_returns(daily_return)
        self._prev_total_value = total_value

    def notify_order(self, order):
        # 未被处理的订单
        if order.status in [order.Submitted, order.Accepted]:
            return
        # 已经处理的订单
        if order.status in [order.Completed, order.Canceled, order.Margin]:
            if order.status == order.Completed:
                if order.isbuy():
                    cost = self.cost[order.data._name]
                    if len(cost) == 0:
                        cost = [0, 0.0]
                    cost[1] = (cost[0] * cost[1] + order.executed.size * order.executed.price) / (cost[0] + order.executed.size)
                    cost[0] = cost[0] + order.executed.size
                    self.cost[order.data._name] = cost
                else:
                    cost = self.cost[order.data._name]
                    cost[0] = cost[0] + order.executed.size
                    if cost[0] == 0:
                        del self.cost[order.data._name]
                    else:
                        self.cost[order.data._name] = cost
            date = datetime.date.fromordinal(int(order.executed.dt))
            if order.isbuy():
                print(f'BUY EXECUTED, date {date}, ref: {order.ref}，Price: {order.executed.price}, '
                      f'Cost: {order.executed.value}, Comm {order.executed.comm}, Size: {order.executed.size}, Stock: {order.data._name}')
            else: # Sell
                print(f'SELL EXECUTED, date {date}, ref: {order.ref}, Price: {order.executed.price}, '
                      f'Cost: {order.executed.value}, Comm {order.executed.comm}, Size: {order.executed.size}, Stock: {order.data._name}')

if __name__ == "__main__":
    # 加载基本面数据 (支持 _qfq.csv 和 _hfq.csv)
    stock_pool_enabled = config.get('stock_pool.enabled', True)
    stock_codes = []
    for f in os.listdir(DATA_PATH):
        if f.startswith('._'):
            continue
        if f.endswith('_qfq.csv') and f != 'sh000001_qfq.csv':
            stock_codes.append(f.replace('_qfq.csv', ''))
        elif f.endswith('_hfq.csv') and f != 'sh000001_hfq.csv':
            stock_codes.append(f.replace('_hfq.csv', ''))

    # 股票池过滤
    if stock_pool_enabled:
        stock_pool = get_stock_pool()
        stock_codes = [c for c in stock_codes if c in stock_pool]
        print(f"基本面数据加载(股票池): {len(stock_codes)} 只")
    else:
        print(f"基本面数据加载(全市场): {len(stock_codes)} 只")

    # 剔除科创板
    star_codes = {c for c in stock_codes if c.startswith('688')}
    stock_codes = [c for c in stock_codes if c not in star_codes]
    print(f"基本面数据(科创板过滤后): {len(stock_codes)} 只")

    fundamental_data = FundamentalData(FUNDAMENTAL_PATH, stock_codes)

    # 初始化情绪分析编排器
    sentiment_orch = None
    sentiment_enabled = config.get('industry_sentiment.enabled', False)
    if sentiment_enabled and SENTIMENT_AVAILABLE:
        print("[Sentiment] 初始化情绪分析模块...")
        try:
            sentiment_orch = SentimentOrchestrator(config, backtest_mode=True)
        except Exception as e:
            print(f"[Sentiment] 初始化异常 (将跳过情绪调整): {e}")
            sentiment_orch = None

    cerebro = bt.Cerebro()
    strategy = Strategy(
        init_cash=CASH,
        fundamental_data=fundamental_data,
        sentiment_orchestrator=sentiment_orch,
    )

    add_data_and_signal(cerebro, strategy, fundamental_data)
    cerebro.broker.setcash(CASH)
    cerebro.broker.setcommission(commission=COMMISSION)
    cerebro.broker.set_slippage_perc(perc=PERC)

    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='pnl')  # 返回收益率时序数据
    cerebro.addanalyzer(bt.analyzers.AnnualReturn, _name='_AnnualReturn')  # 年化收益率
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='_SharpeRatio')  # 夏普比率
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='_DrawDown')  # 回撤

    cerebro.addstrategy(
        BacktraderExecution,
        real_strategy=strategy,
    )
    # 启动回测
    print("启动回测引擎...")
    try:
        result = cerebro.run()
    except MemoryError:
        print("\n[ERROR] 内存不足！尝试减少股票数量或增大WSL2内存限制")
        import sys
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] 回测异常退出: {e}")
        import traceback
        traceback.print_exc()
        import sys
        sys.exit(1)

    # 从返回的 result 中提取回测结果
    strat = result[0]
    # 返回日度收益率序列
    daily_return = pd.Series(strat.analyzers.pnl.get_analysis())
    # 打印评价指标
    print("--------------- AnnualReturn -----------------")
    print(strat.analyzers._AnnualReturn.get_analysis())
    print("--------------- SharpeRatio -----------------")
    print(strat.analyzers._SharpeRatio.get_analysis())
    print("--------------- DrawDown -----------------")
    print(strat.analyzers._DrawDown.get_analysis())

    # 保存选股结果供验证使用
    if strat.portfolio_selections:
        selections_df = pd.DataFrame(strat.portfolio_selections)
        strategy_dir = os.path.dirname(os.path.abspath(__file__))
        selections_path = os.path.join(strategy_dir, 'rolling_validation_results', 'portfolio_selections.csv')
        os.makedirs(os.path.dirname(selections_path), exist_ok=True)
        selections_df.to_csv(selections_path, index=False)
        print(f"\n选股结果已保存: {len(selections_df)} 条 -> {selections_path}")

    # 打印因子选择统计
    strategy.signal_engine.print_factor_stats()
