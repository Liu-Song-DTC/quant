import backtrader as bt
import pandas as pd
import datetime
import os
from tqdm import tqdm

CASH = 10000.0
COMMISSION = 0.0003
PERC = 0.005
MAX_POSITION = 10
POSITION_SIZE = 1.0 / MAX_POSITION
REBALANCE_DAYS = 1

DATA_PATH = "../data/stock_data/backtrader_data/"
def add_data(cerebro):
    all_items = os.listdir(DATA_PATH)
    for item in tqdm(all_items, desc="loading data"):
        name = item[:-8]
        data = pd.read_csv(DATA_PATH+item, parse_dates=['datetime'])
        datafeed = bt.feeds.PandasData(dataname=data, datetime='datetime')
        cerebro.adddata(datafeed, name=name)


class BacktraderExecution(bt.Strategy):
    params = dict(
        signal_store=None,
        portfolio=None,
    )

    def __init__(self):
        self.universe = [d._name for d in self.datas]
        self.position_value = {}

    def next(self):
        date = self.datas[0].datetime.date(0)

        prices = {
            d._name: d.close[0]
            for d in self.datas
        }

        cash = self.broker.getcash()

        target = self.p.portfolio.build(
            date=date,
            universe=self.universe,
            current_positions=self.position_value,
            signal_store=self.p.signal_store,
            cash=cash,
            prices=prices,
        )

        current = set(self.position_value)
        target_set = set(target)

        # 卖出
        for code in current - target_set:
            data = self.getdatabyname(code)
            self.close(data)
            del self.position_value[code]

        # 买入 / 调仓
        for code, value in target.items():
            data = self.getdatabyname(code)
            price = data.close[0]
            size = int(value / price)

            if size > 0:
                self.order_target_size(data, size)
                self.position_value[code] = value

if __name__ == "__main__":
    cerebro = bt.Cerebro()
    add_data(cerebro)
    cerebro.broker.setcash(CASH)
    cerebro.broker.setcommission(commission=COMMISSION)
    cerebro.broker.set_slippage_perc(perc=PERC)

    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='pnl')  # 返回收益率时序数据
    cerebro.addanalyzer(bt.analyzers.AnnualReturn, _name='_AnnualReturn')  # 年化收益率
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='_SharpeRatio')  # 夏普比率
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='_DrawDown')  # 回撤

    cerebro.addstrategy(BacktraderExecution)
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
