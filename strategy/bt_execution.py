import backtrader as bt
import pandas as pd
import datetime
import os
from tqdm import tqdm
import math
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

from core.strategy import Strategy
from core.fundamental import FundamentalData
from core.signal_engine import SignalEngine
from core.signal_store import SignalStore
from core.config_loader import load_config

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


def generate_signal_worker(args):
    """子进程工作函数：生成单只股票的信号"""
    code, data_dict = args
    # 在子进程中重新创建 SignalEngine（因为不能pickle）
    engine = SignalEngine()
    store = SignalStore()

    # 重建 DataFrame
    data = pd.DataFrame(data_dict)

    # 生成信号
    engine.generate(code, data, store)

    # 返回结果：code -> {(date,): signal} 需要转换为可哈希的键
    return code, store._store

def add_data_and_signal(cerebro, strategy):
    all_items = os.listdir(DATA_PATH)
    stock_codes = []  # 获取回测池中的股票列表

    # 只读取一次CSV数据
    stock_data_dict = {}
    for item in tqdm(all_items, desc="loading data"):
        name = item[:-8]
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

    # 并行生成信号（排除指数）
    stock_items = [(name, data.to_dict()) for name, data in stock_data_dict.items()
                  if name != "sh000001"]
    stock_codes = [name for name, _ in stock_items]

    print(f"使用 {NUM_WORKERS} 个进程并行生成信号...")
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        results = list(tqdm(
            executor.map(generate_signal_worker, stock_items),
            total=len(stock_items),
            desc="generating signals"
        ))

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

    def __init__(self):
        self.universe = [d._name for d in self.datas]
        self.count = 0
        self.orders_list = defaultdict(list)
        self.last_date = None
        self.cost = defaultdict(list)

    def next(self):
        if self.last_date is not None and self.last_date in self.orders_list:
            for order in self.orders_list[self.last_date]:
                self.cancel(order)
            del self.orders_list[self.last_date]
        self.count += 1
        date = self.datas[0].datetime.date(0)
        self.last_date = date

        prices = {}
        for d in self.datas:
            price = d.close[0]
            if price is None or math.isnan(price) or price <= 0:
                continue
            prices[d._name] = price

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
            universe=self.universe,
            current_positions=current_positions,
            cash=self.broker.getcash(),
            prices=prices,
            cost=self.cost,
            rebalance=rebalance,
        )

        active_codes = set(target.keys()) | set(current_positions.keys())
        for d in self.datas:
            code = d._name
            if code not in active_codes:
                continue
            price = d.close[0]

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
    # 加载基本面数据
    stock_codes = [f.replace('_qfq.csv', '') for f in os.listdir(DATA_PATH)
                   if f.endswith('_qfq.csv') and f != 'sh000001_qfq.csv']
    fundamental_data = FundamentalData(FUNDAMENTAL_PATH, stock_codes)

    cerebro = bt.Cerebro()
    strategy = Strategy(
        init_cash=CASH,
        max_position=MAX_POSITION,
        fundamental_data=fundamental_data
    )

    add_data_and_signal(cerebro, strategy)
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
