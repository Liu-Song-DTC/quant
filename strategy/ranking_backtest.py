#!/usr/bin/env python
"""
排名策略回测执行器

核心思想：
- 因子值高 -> 排名高 -> 入选
- 直接利用IC的排序能力，而非阈值判断

简化流程：
1. 加载股票数据
2. 每个调仓日计算所有股票因子值
3. 截面标准化后排名
4. 选择Top-N股票
5. 等权或风险平价分配仓位
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from core.config_loader import load_config
from core.ranking_engine import RankingEngine, RankingSignal
from core.industry_mapping import get_industry_category


class RankingBacktest:
    """
    排名策略回测

    特点：
    - 基于因子排名选股
    - 截面标准化消除极端值
    - 简单透明的策略逻辑
    """

    def __init__(self, config=None):
        self.config = config or load_config()
        self.engine = RankingEngine(self.config)

        # 回测参数
        backtest_config = self.config.get('backtest', {})
        self.cash = backtest_config.get('cash', 100000.0)
        self.commission = backtest_config.get('commission', 0.0015)
        self.slippage = backtest_config.get('slippage', 0.0015)
        self.rebalance_days = backtest_config.get('rebalance_days', 10)
        self.max_position = backtest_config.get('max_position', 10)

        # 组合参数
        portfolio_config = self.config.get('portfolio', {})
        self.target_volatility = portfolio_config.get('target_volatility', 0.20)

        # 数据
        self.stock_data = {}
        self.dates = []

        # 结果
        self.trades = []
        self.daily_values = []
        self.positions = {}

    def load_data(self):
        """加载股票数据"""
        data_path = os.path.join(os.path.dirname(SCRIPT_DIR), 'data/stock_data/backtrader_data')

        print("加载股票数据...")
        for f in os.listdir(data_path):
            if not f.endswith('_qfq.csv'):
                continue
            code = f.replace('_qfq.csv', '')
            df = pd.read_csv(os.path.join(data_path, f))

            # 统一日期列
            if 'datetime' in df.columns:
                df['date'] = pd.to_datetime(df['datetime'])
            elif 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])

            # 确保必要列存在
            if 'close' not in df.columns:
                continue

            self.stock_data[code] = df

        print(f"  加载 {len(self.stock_data)} 只股票")

        # 获取所有日期
        all_dates = set()
        for df in self.stock_data.values():
            all_dates.update(df['date'].tolist())
        self.dates = sorted(list(all_dates))

        print(f"  日期范围: {self.dates[0]} ~ {self.dates[-1]}")

    def get_price(self, code: str, date) -> float:
        """获取指定日期的收盘价"""
        df = self.stock_data.get(code)
        if df is None:
            return None

        row = df[df['date'] == date]
        if len(row) == 0:
            return None

        return row.iloc[0]['close']

    def run(self):
        """运行回测"""
        print("\n" + "=" * 60)
        print("排名策略回测")
        print("=" * 60)

        self.load_data()

        # 初始化
        cash = self.cash
        positions = {}  # {code: shares}
        cost_basis = {}  # {code: avg_cost}

        last_rebalance = None
        equity_curve = []

        # 遍历日期
        for i, date in enumerate(self.dates):
            # 计算当前市值
            market_value = 0
            for code, shares in positions.items():
                price = self.get_price(code, date)
                if price:
                    market_value += shares * price

            total_equity = cash + market_value
            equity_curve.append({
                'date': date,
                'equity': total_equity,
                'cash': cash,
                'market_value': market_value,
                'n_positions': len(positions),
            })

            # 判断是否调仓日
            should_rebalance = False
            if last_rebalance is None:
                should_rebalance = True
            elif (date - last_rebalance).days >= self.rebalance_days:
                should_rebalance = True

            if not should_rebalance:
                continue

            last_rebalance = date

            # === 调仓逻辑 ===

            # 1. 计算排名
            signals = self.engine.compute_ranking(date, self.stock_data, positions)

            if not signals:
                continue

            # 2. 选择Top-N
            selected = self.engine.select_top_n(signals, n=self.max_position, industry_cap=2)

            if not selected:
                continue

            # 3. 计算目标仓位
            target_codes = {code for code, _, _ in selected}
            target_weights = {code: weight for code, weight, _ in selected}

            # 4. 卖出不在目标中的持仓
            for code in list(positions.keys()):
                if code not in target_codes:
                    price = self.get_price(code, date)
                    if price:
                        shares = positions[code]
                        sell_value = shares * price * (1 - self.commission - self.slippage)
                        cash += sell_value

                        self.trades.append({
                            'date': date,
                            'code': code,
                            'action': 'SELL',
                            'shares': shares,
                            'price': price,
                            'value': sell_value,
                        })

                        del positions[code]
                        if code in cost_basis:
                            del cost_basis[code]

            # 5. 买入新选中的股票
            total_equity = cash + sum(
                positions.get(c, 0) * (self.get_price(c, date) or 0)
                for c in positions
            )

            for code, weight, sig in selected:
                if code in positions:
                    continue  # 已持有

                price = self.get_price(code, date)
                if not price or price <= 0:
                    continue

                # 计算目标金额
                target_value = total_equity * weight

                # 考虑交易成本
                shares = int(target_value / price / 100) * 100  # A股100股一手

                if shares <= 0:
                    continue

                cost = shares * price * (1 + self.commission + self.slippage)

                if cost > cash:
                    shares = int(cash / price / 100) * 100
                    cost = shares * price * (1 + self.commission + self.slippage)

                if shares > 0:
                    cash -= cost
                    positions[code] = shares
                    cost_basis[code] = price

                    self.trades.append({
                        'date': date,
                        'code': code,
                        'action': 'BUY',
                        'shares': shares,
                        'price': price,
                        'value': cost,
                        'factor_value': sig.factor_value,
                        'rank': sig.rank,
                        'industry': sig.industry,
                    })

        # 保存结果
        self.equity_curve = equity_curve
        self._print_summary()

        return equity_curve

    def _print_summary(self):
        """打印回测摘要"""
        if not self.equity_curve:
            return

        df = pd.DataFrame(self.equity_curve)

        # 计算收益
        df['returns'] = df['equity'].pct_change()

        # 计算指标
        total_return = (df['equity'].iloc[-1] / df['equity'].iloc[0] - 1) * 100
        annual_return = total_return * 252 / len(df)
        volatility = df['returns'].std() * np.sqrt(252) * 100
        sharpe = (annual_return / volatility) if volatility > 0 else 0

        # 最大回撤
        df['cummax'] = df['equity'].cummax()
        df['drawdown'] = (df['equity'] - df['cummax']) / df['cummax']
        max_drawdown = df['drawdown'].min() * 100

        # 胜率
        winning_days = (df['returns'] > 0).sum()
        total_days = len(df[df['returns'].notna()])
        win_rate = winning_days / total_days * 100 if total_days > 0 else 0

        print("\n" + "=" * 60)
        print("回测结果")
        print("=" * 60)
        print(f"总收益率:     {total_return:>8.2f}%")
        print(f"年化收益率:   {annual_return:>8.2f}%")
        print(f"年化波动率:   {volatility:>8.2f}%")
        print(f"夏普比率:     {sharpe:>8.2f}")
        print(f"最大回撤:     {max_drawdown:>8.2f}%")
        print(f"日胜率:       {win_rate:>8.2f}%")
        print(f"交易次数:     {len(self.trades):>8}")
        print(f"最终净值:     {df['equity'].iloc[-1]:>8.2f}")

        # 保存详细结果
        output_dir = os.path.join(SCRIPT_DIR, 'rolling_validation_results')
        os.makedirs(output_dir, exist_ok=True)

        df.to_csv(os.path.join(output_dir, 'ranking_equity.csv'), index=False)

        if self.trades:
            trades_df = pd.DataFrame(self.trades)
            trades_df.to_csv(os.path.join(output_dir, 'ranking_trades.csv'), index=False)

        print(f"\n详细结果已保存到: {output_dir}")


def main():
    backtest = RankingBacktest()
    backtest.run()


if __name__ == '__main__':
    main()
