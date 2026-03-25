import backtrader as bt
import pandas as pd
import datetime
import os
from tqdm import tqdm
import math
from collections import defaultdict
from multiprocessing import Pool, Manager
from functools import partial

from core.strategy import Strategy
from core.fundamental import FundamentalData
from core.signal_engine import SignalEngine
from core.factor_preparer import prepare_factor_data
from core.signal_store import SignalStore
from core.config_loader import load_config
from core.industry_mapping import INDUSTRY_KEYWORDS

# 加载配置
config = load_config()

# 回测参数 - 从配置文件读取
CASH = config.get('backtest.cash', 100000.0)
COMMISSION = config.get('backtest.commission', 0.0015)
PERC = config.get('backtest.slippage', 0.0015)
MAX_POSITION = config.get('backtest.max_position', 10)
REBALANCE_DAYS = config.get('backtest.rebalance_days', 20)
NUM_WORKERS = config.get('backtest.num_workers', 8)

# 数据路径 - 从配置文件读取
DATA_PATH = config.get('paths.data', '../data/stock_data/backtrader_data/')
FUNDAMENTAL_PATH = config.get('paths.fundamental', '../data/stock_data/fundamental_data/')


# 全局变量用于 worker 进程
_worker_engine = None
_worker_use_dynamic = False


def _init_worker(fundamental_path, stock_codes, use_dynamic, factor_df, industry_codes, factor_cache, all_dates):
    """Worker 进程初始化函数"""
    global _worker_engine, _worker_use_dynamic
    _worker_use_dynamic = use_dynamic

    # 每个 worker 创建自己的 engine 和 fundamental_data
    _worker_engine = SignalEngine()

    import sys
    print(f"[Worker PID {os.getpid()}] fundamental_path: {fundamental_path}, exists: {os.path.exists(fundamental_path) if fundamental_path else False}", flush=True)

    if fundamental_path and os.path.exists(fundamental_path):
        fd = FundamentalData(fundamental_path, stock_codes)
        _worker_engine.set_fundamental_data(fd)
        print(f"[Worker PID {os.getpid()}] fundamental_data set, stock_data count: {len(fd.stock_data)}", flush=True)
    else:
        print(f"[Worker PID {os.getpid()}] fundamental_path not found or invalid", flush=True)

    # 设置动态因子数据（只设置一次，避免重复清除缓存）
    if use_dynamic and factor_df is not None:
        _worker_engine.set_factor_data(factor_df)
        _worker_engine.set_industry_mapping(industry_codes)

        # 设置预计算的因子选择缓存（避免worker重复计算）
        if factor_cache is not None and all_dates is not None:
            _worker_engine.dynamic_factor_selector.set_factor_cache(factor_cache, all_dates)
            print(f"[Worker PID {os.getpid()}] factor cache set from precomputed, {len(factor_cache)} dates", flush=True)
        else:
            print(f"[Worker PID {os.getpid()}] no precomputed cache, will compute on demand", flush=True)


def _generate_stock_signal_worker(args):
    """Worker 函数：为一个股票生成信号"""
    global _worker_engine, _worker_use_dynamic
    code, data_dict = args

    engine = _worker_engine

    if engine is None:
        # Fallback: 创建新 engine
        engine = SignalEngine()
        if fundamental_path and os.path.exists(fundamental_path):
            fd = FundamentalData(fundamental_path, [code])
            engine.set_fundamental_data(fd)

    store = SignalStore()
    data = pd.DataFrame(data_dict)
    engine.generate(code, data, store)
    return (code, store._store)


fundamental_path = FUNDAMENTAL_PATH  # 全局变量供 worker 使用


def add_data_and_signal(cerebro, strategy, fundamental_data=None):
    all_items = os.listdir(DATA_PATH)
    stock_codes = []  # 获取回测池中的股票列表

    # 只读取一次CSV数据
    stock_data_dict = {}
    for item in tqdm(all_items, desc="loading data"):
        # 根据后缀提取股票代码
        if item.endswith('_qfq.csv'):
            name = item[:-8]  # 去掉 '_qfq.csv'
        elif item.endswith('_hfq.csv'):
            name = item[:-8]  # 去掉 '_hfq.csv'
        else:
            continue
        data = pd.read_csv(DATA_PATH + item, parse_dates=['datetime'])
        stock_data_dict[name] = data

    # 从任意一个DataFrame获取日期（所有股票数据共享日历）
    dates = set()
    for data in stock_data_dict.values():
        dates.update(data['datetime'])
    calendar_index = pd.DatetimeIndex(sorted(dates))

    # 处理指数数据
    if "sh000001" in stock_data_dict:
        strategy.generate_market_regime(stock_data_dict["sh000001"])

    # 准备动态因子数据
    dynamic_config = config.config.get('dynamic_factor', {})
    if dynamic_config.get('enabled', False):
        print("准备动态因子数据...")
        # 获取股票代码列表（排除指数）
        stock_codes = [name for name in stock_data_dict.keys() if name != "sh000001"]
        factor_df, industry_codes, all_dates = prepare_factor_data(
            stock_data_dict,
            fundamental_data,
            INDUSTRY_KEYWORDS,
            NUM_WORKERS
        )
        strategy.set_factor_data(factor_df, industry_codes)
        print(f"动态因子模式已启用: {len(industry_codes)} 个行业")

    # 生成信号（多进程并行）
    use_dynamic = dynamic_config.get('enabled', False)
    stock_codes = [name for name in stock_data_dict.keys() if name != "sh000001"]

    # 创建带动态因子的 SignalEngine（如果启用）
    main_engine = None
    if use_dynamic:
        main_engine = SignalEngine()
        main_engine.set_factor_data(factor_df)
        main_engine.set_industry_mapping(industry_codes)
        main_engine.set_fundamental_data(fundamental_data)
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
    else:
        precomputed_cache = None
        precomputed_all_dates = None

    # 准备参数：每只股票的数据
    # 传递 (code, data_dict, factor_df, industry_codes) 给每个 worker
    stock_items = [
        (name, data.to_dict())
        for name, data in stock_data_dict.items() if name != "sh000001"
    ]

    # 保存信号数据用于验证
    all_signals = []

    # 多进程并行生成信号
    print(f"多进程生成信号 ({NUM_WORKERS} workers)...")

    # 动态因子统计
    dynamic_factor_stats = {'hit': 0, 'miss': 0, 'factor_names': {}}

    # 使用 Pool + initializer 模式
    # 注意：由于 FundamentalData 可能较大，每个 worker 创建自己的实例
    with Pool(
        processes=NUM_WORKERS,
        initializer=_init_worker,
        initargs=(FUNDAMENTAL_PATH, stock_codes, use_dynamic, factor_df, industry_codes, precomputed_cache, precomputed_all_dates)
    ) as pool:
        # imap_unordered 比 map 更快，且结果顺序不影响
        results = []
        for result in tqdm(
            pool.imap_unordered(_generate_stock_signal_worker, stock_items, chunksize=10),
            total=len(stock_items),
            desc="generating signals"
        ):
            results.append(result)
            # 收集信号数据用于保存（实时收集，避免最后遍历大列表）
            code, store_data = result
            for (c, date), sig in store_data.items():
                if hasattr(date, 'date'):
                    date = date.date()
                all_signals.append({
                    'code': c,
                    'date': date,
                    'buy': sig.buy,
                    'sell': sig.sell,
                    'score': sig.score,
                    'factor_value': sig.factor_value,
                    'factor_name': sig.factor_name,
                    'industry': sig.industry,
                })
                # 统计动态因子命中情况
                if sig.factor_name and sig.factor_name.startswith('DYN_'):
                    dynamic_factor_stats['hit'] += 1
                    fn = sig.factor_name.split('_')[1] if '_' in sig.factor_name else sig.factor_name
                    dynamic_factor_stats['factor_names'][fn] = dynamic_factor_stats['factor_names'].get(fn, 0) + 1
                else:
                    dynamic_factor_stats['miss'] += 1

                # 调试：每10000条打印一次进度
                if (dynamic_factor_stats['hit'] + dynamic_factor_stats['miss']) % 100000 == 0:
                    print(f"  [DEBUG] Processed {dynamic_factor_stats['hit'] + dynamic_factor_stats['miss']:,} signals, hit={dynamic_factor_stats['hit']:,}", flush=True)

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

    # 保存信号数据到文件（供验证脚本使用）
    strategy_dir = os.path.dirname(os.path.abspath(__file__))
    signals_output_path = os.path.join(strategy_dir, 'rolling_validation_results', 'backtest_signals.csv')
    os.makedirs(os.path.dirname(signals_output_path), exist_ok=True)
    signals_df = pd.DataFrame(all_signals)
    signals_df.to_csv(signals_output_path, index=False)
    print(f"信号数据已保存: {len(signals_df)} 条 -> {signals_output_path}")

    # 将结果写入 signal_store
    # _store 的键是 (code, datetime.date) -> signal
    for code, store_data in results:
        for (c, date), signal in store_data.items():
            # 确保 date 是 datetime.date 类型
            if hasattr(date, 'date'):
                date = date.date()
            strategy.signal_store.set(c, date, signal)

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
        self.orders_list = defaultdict(list)
        self.last_date = None
        self.cost = defaultdict(list)
        self.portfolio_selections = []  # 记录每期选股结果

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
        if self.count >= REBALANCE_DAYS:
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
            size = math.floor(raw) * 100 if raw > 0 else math.ceil(raw) * 100

            if size > 0:
                max_affordable = int(self.broker.getcash() / price / 100) * 100
                size = min(size, max_affordable)
                if size > 0:
                    order = self.buy(data=d, size=size)
                    self.orders_list[date].append(order)
            elif size < 0:
                order = self.sell(data=d, size=size)
                self.orders_list[date].append(order)

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
    stock_codes = []
    for f in os.listdir(DATA_PATH):
        if f.endswith('_qfq.csv') and f != 'sh000001_qfq.csv':
            stock_codes.append(f.replace('_qfq.csv', ''))
        elif f.endswith('_hfq.csv') and f != 'sh000001_hfq.csv':
            stock_codes.append(f.replace('_hfq.csv', ''))
    fundamental_data = FundamentalData(FUNDAMENTAL_PATH, stock_codes)

    cerebro = bt.Cerebro()
    strategy = Strategy(
        init_cash=CASH,
        max_position=MAX_POSITION,
        fundamental_data=fundamental_data
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
    result = cerebro.run()
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
