import backtrader as bt
import pandas as pd
import datetime
import os
from tqdm import tqdm
import math
from collections import defaultdict

from core.strategy import Strategy

CASH = 100000.0
COMMISSION = 0.0003
PERC = 0.005
MAX_POSITION = 3
REBALANCE_DAYS = 1

DATA_PATH = "../data/stock_data/backtrader_data/"
def add_data_and_signal(cerebro, strategy):
    all_items = os.listdir(DATA_PATH)
    dates = set()
    for item in (all_items):
        data = pd.read_csv(DATA_PATH+item, parse_dates=['datetime'])
        dates.update(data['datetime'])
    calendar_index = pd.DatetimeIndex(sorted(dates))

    price_cols = ['open', 'high', 'low', 'close']
    for item in tqdm(all_items, desc="loading data"):
        name = item[:-8]
        data = pd.read_csv(DATA_PATH+item, parse_dates=['datetime'])
        strategy.generate_signal(name, data)
        data = data.set_index('datetime')
        data = data.reindex(calendar_index)
        data[price_cols] = data[price_cols].ffill()
        if 'volume' in data.columns:
            data['volume'] = data['volume'].fillna(0)
        datafeed = bt.feeds.PandasData(dataname=data)
        cerebro.adddata(datafeed, name=name)

class BacktraderExecution(bt.Strategy):
    params = dict(
        real_strategy=None,
    )

    def __init__(self):
        self.universe = [d._name for d in self.datas]
        self.count = 0
        self.orders_list = defaultdict(list)
        self.last_date = None

    def next(self):
        if self.last_date is not None and self.last_date in self.orders_list:
            for order in self.orders_list[self.last_date]:
                self.cancel(order)
            del self.orders_list[self.last_date]
        self.count += 1
        if self.count < REBALANCE_DAYS:
            return
        else:
            self.count = 1
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

        target = self.p.real_strategy.generate_positions(
            date=date,
            universe=self.universe,
            current_positions=current_positions,
            cash=self.broker.getcash(),
            prices=prices,
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
            date = datetime.date.fromordinal(int(order.executed.dt))
            if order.isbuy():
                print(f'BUY EXECUTED, date {date}, ref: {order.ref}，Price: {order.executed.price}, '
                      f'Cost: {order.executed.value}, Comm {order.executed.comm}, Size: {order.executed.size}, Stock: {order.data._name}')
            else: # Sell
                print(f'SELL EXECUTED, date {date}, ref: {order.ref}, Price: {order.executed.price}, '
                      f'Cost: {order.executed.value}, Comm {order.executed.comm}, Size: {order.executed.size}, Stock: {order.data._name}')

if __name__ == "__main__":
    cerebro = bt.Cerebro()
    strategy = Strategy(
        init_cash=CASH,
        max_position=MAX_POSITION
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
    #  cerebro.plot()
