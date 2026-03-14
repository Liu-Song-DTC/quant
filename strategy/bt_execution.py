import backtrader as bt
import pandas as pd
import datetime
import os
from tqdm import tqdm
import math
from collections import defaultdict

from core.strategy import Strategy
from core.fundamental import FundamentalData
from utils.utils import (
    plot_signal_diagnosis,
)

CASH = 100000.0
COMMISSION = 0.0015
PERC = 0.0015
MAX_POSITION = 10
REBALANCE_DAYS = 20

DATA_PATH = "../data/stock_data/backtrader_data/"
FUNDAMENTAL_PATH = "../data/stock_data/fundamental_data/"

def add_data_and_signal(cerebro, strategy):
    all_items = os.listdir(DATA_PATH)
    stock_codes = []  # 获取回测池中的股票列表

    dates = set()
    for item in (all_items):
        data = pd.read_csv(DATA_PATH+item, parse_dates=['datetime'])
        dates.update(data['datetime'])
    calendar_index = pd.DatetimeIndex(sorted(dates))

    price_cols = ['open', 'high', 'low', 'close']
    for item in tqdm(all_items, desc="loading data"):
        name = item[:-8]
        data = pd.read_csv(DATA_PATH+item, parse_dates=['datetime'])
        if name == "sh000001":
            strategy.generate_market_regime(data)
            continue
        stock_codes.append(name)  # 记录股票代码
        strategy.generate_signal(name, data)
        #  plot_signal_diagnosis(
        #      name,
        #      data,
        #      strategy.signal_engine,
        #      strategy.signal_store,
        #  )
        #  exit(0)
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
