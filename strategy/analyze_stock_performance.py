#!/usr/bin/env python
"""
股票表现分析脚本
分析策略在每只股票上的表现，找出效果好和效果不好的股票的特点
"""
import re
import os
import sys
from collections import defaultdict
from datetime import datetime
import pandas as pd
import numpy as np

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def run_backtest_and_capture():
    """运行回测并捕获所有交易记录"""
    print("=" * 60)
    print("步骤1: 运行回测获取完整交易记录")
    print("=" * 60)

    # 运行回测，捕获所有输出
    import subprocess
    result = subprocess.run(
        [sys.executable, 'bt_execution.py'],
        capture_output=True,
        text=True,
        cwd=os.path.dirname(os.path.abspath(__file__))
    )

    if result.returncode != 0:
        print(f"回测运行失败: {result.stderr}")
        return None

    print("回测完成！")
    return result.stdout + result.stderr


def parse_trades(log_content):
    """解析日志中的所有交易记录"""
    print("\n" + "=" * 60)
    print("步骤2: 解析交易记录")
    print("=" * 60)

    trades = []
    # 匹配 BUY/SELL EXECUTED 行
    pattern = r'(BUY|SELL) EXECUTED, date (\d{4}-\d{2}-\d{2}).*?Price: ([0-9.]+).*?Size: ([-\d]+).*?Stock: (\w+)'

    for match in re.finditer(pattern, log_content):
        trade_type = match.group(1)
        date = match.group(2)
        price = float(match.group(3))
        size = int(match.group(4))
        stock = match.group(5)

        trades.append({
            'type': trade_type,
            'date': date,
            'price': price,
            'size': size,
            'stock': stock
        })

    print(f"共解析到 {len(trades)} 笔交易")
    return trades


def calculate_stock_performance(trades):
    """计算每只股票的交易表现"""
    print("\n" + "=" * 60)
    print("步骤3: 计算每只股票的表现")
    print("=" * 60)

    # 按股票分组，记录买卖
    stock_trades = defaultdict(list)
    for trade in trades:
        stock_trades[trade['stock']].append(trade)

    # 计算每只股票的收益
    stock_performance = {}

    for stock, stock_trade_list in stock_trades.items():
        # 按日期排序
        stock_trade_list.sort(key=lambda x: x['date'])

        total_buy_value = 0
        total_buy_size = 0
        total_sell_value = 0
        total_sell_size = 0
        buy_dates = []
        sell_dates = []

        for trade in stock_trade_list:
            if trade['type'] == 'BUY':
                total_buy_value += trade['price'] * trade['size']
                total_buy_size += trade['size']
                buy_dates.append(trade['date'])
            else:
                total_sell_value += trade['price'] * abs(trade['size'])
                total_sell_size += abs(trade['size'])
                sell_dates.append(trade['date'])

        # 计算平均成本和卖出价格
        avg_buy_price = total_buy_value / total_buy_size if total_buy_size > 0 else 0
        avg_sell_price = total_sell_value / total_sell_size if total_sell_size > 0 else 0

        # 持有股数
        hold_size = total_buy_size - total_sell_size

        # 收益计算: (卖出收入 - 买入成本) / 买入成本
        if total_buy_value > 0 and total_sell_size > 0:
            profit_ratio = (total_sell_value - total_buy_value) / total_buy_value
        else:
            profit_ratio = 0

        # 交易次数
        buy_count = len([t for t in stock_trade_list if t['type'] == 'BUY'])
        sell_count = len([t for t in stock_trade_list if t['type'] == 'SELL'])

        # 持仓天数（粗略估算）
        if buy_dates and sell_dates:
            first_buy = datetime.strptime(buy_dates[0], '%Y-%m-%d')
            last_sell = datetime.strptime(sell_dates[-1], '%Y-%m-%d')
            hold_days = (last_sell - first_buy).days
        else:
            hold_days = 0

        stock_performance[stock] = {
            'stock': stock,
            'total_buy_value': total_buy_value,
            'total_sell_value': total_sell_value,
            'avg_buy_price': avg_buy_price,
            'avg_sell_price': avg_sell_price,
            'profit_ratio': profit_ratio,
            'profit_value': total_sell_value - total_buy_value,
            'buy_count': buy_count,
            'sell_count': sell_count,
            'hold_days': hold_days,
            'first_buy': buy_dates[0] if buy_dates else None,
            'last_sell': sell_dates[-1] if sell_dates else None,
        }

    print(f"共分析了 {len(stock_performance)} 只股票")
    return stock_performance


def get_fundamental_data(stocks):
    """获取股票的基本面数据"""
    print("\n" + "=" * 60)
    print("步骤4: 获取基本面数据")
    print("=" * 60)

    from core.fundamental import FundamentalData

    # 使用正确的数据路径
    strategy_dir = os.path.dirname(os.path.abspath(__file__))
    fundamental_path = os.path.join(strategy_dir, '..', 'data', 'stock_data', 'fundamental_data')

    if not os.path.exists(fundamental_path):
        print(f"基本面数据目录不存在: {fundamental_path}")
        return {}

    fd = FundamentalData(fundamental_path + '/')

    stock_fundamentals = {}
    for stock in stocks:
        # 使用最近的时间点获取基本面数据
        test_date = '2024-06-01'

        industry = fd.get_industry(stock, test_date)
        roe = fd.get_roe(stock, test_date)
        profit_growth = fd.get_profit_growth(stock, test_date)
        revenue_growth = fd.get_revenue_growth(stock, test_date)
        eps = fd.get_eps(stock, test_date)

        # 简化行业分类
        def simplify_industry(ind):
            if not ind:
                return '其他'
            ind = str(ind)
            if '白酒' in ind or '食品' in ind or '饮料' in ind:
                return '消费'
            elif '银行' in ind or '证券' in ind or '保险' in ind:
                return '金融'
            elif '光伏' in ind or '电池' in ind or '风电' in ind or '电力' in ind:
                return '新能源'
            elif '半导体' in ind or '软件' in ind or '计算机' in ind or '电子' in ind:
                return '科技'
            elif '医药' in ind or '医疗' in ind:
                return '医药'
            elif '制造' in ind or '机械' in ind or '汽车' in ind:
                return '制造'
            else:
                return '周期'

        stock_fundamentals[stock] = {
            'industry': industry,
            'industry_cat': simplify_industry(industry),
            'roe': roe,
            'profit_growth': profit_growth,
            'revenue_growth': revenue_growth,
            'eps': eps,
        }

    print(f"获取了 {len(stock_fundamentals)} 只股票的基本面数据")
    return stock_fundamentals


def analyze_by_industry(stock_performance, stock_fundamentals):
    """按行业分析股票表现"""
    print("\n" + "=" * 60)
    print("步骤5: 按行业分析表现")
    print("=" * 60)

    industry_stats = defaultdict(lambda: {
        'stocks': [],
        'total_profit': 0,
        'winning_trades': 0,
        'losing_trades': 0,
    })

    for stock, perf in stock_performance.items():
        fund = stock_fundamentals.get(stock, {})
        industry = fund.get('industry_cat', '其他')

        industry_stats[industry]['stocks'].append(stock)
        industry_stats[industry]['total_profit'] += perf['profit_value']

        if perf['profit_ratio'] > 0:
            industry_stats[industry]['winning_trades'] += 1
        else:
            industry_stats[industry]['losing_trades'] += 1

    # 打印行业统计
    print("\n行业表现统计:")
    print("-" * 80)
    print(f"{'行业':<10}{'股票数':<10}{'总收益':<15}{'盈利次数':<10}{'亏损次数':<10}{'胜率':<10}")
    print("-" * 80)

    for industry, stats in sorted(industry_stats.items(), key=lambda x: x[1]['total_profit'], reverse=True):
        total = stats['winning_trades'] + stats['losing_trades']
        win_rate = stats['winning_trades'] / total * 100 if total > 0 else 0
        print(f"{industry:<10}{len(stats['stocks']):<10}{stats['total_profit']:>12.2f}{'':3}{stats['winning_trades']:<10}{stats['losing_trades']:<10}{win_rate:>8.1f}%")

    return industry_stats


def analyze_by_roe(stock_performance, stock_fundamentals):
    """按ROE分析股票表现"""
    print("\n" + "=" * 60)
    print("步骤6: 按ROE分析表现")
    print("=" * 60)

    # 分组
    roe_groups = {
        '高ROE(>10%)': [],
        '中ROE(5-10%)': [],
        '低ROE(<5%)': [],
        '无数据': []
    }

    for stock, perf in stock_performance.items():
        fund = stock_fundamentals.get(stock, {})
        roe = fund.get('roe')

        if roe is None:
            roe_groups['无数据'].append((stock, perf))
        elif roe > 0.10:
            roe_groups['高ROE(>10%)'].append((stock, perf))
        elif roe > 0.05:
            roe_groups['中ROE(5-10%)'].append((stock, perf))
        else:
            roe_groups['低ROE(<5%)'].append((stock, perf))

    print("\nROE分组表现统计:")
    print("-" * 60)
    print(f"{'ROE分组':<15}{'股票数':<10}{'平均收益':<15}{'盈利占比':<10}")
    print("-" * 60)

    for group, stocks in roe_groups.items():
        if not stocks:
            continue
        profits = [s[1]['profit_ratio'] * 100 for s in stocks]
        avg_profit = np.mean(profits)
        win_rate = len([p for p in profits if p > 0]) / len(profits) * 100
        print(f"{group:<15}{len(stocks):<10}{avg_profit:>12.2f}%{win_rate:>12.1f}%")

    return roe_groups


def find_best_worst_stocks(stock_performance, stock_fundamentals, top_n=20):
    """找出表现最好和最差的股票"""
    print("\n" + "=" * 60)
    print(f"步骤7: 找出表现最好和最差的各{top_n}只股票")
    print("=" * 60)

    # 按收益排序
    sorted_stocks = sorted(
        stock_performance.items(),
        key=lambda x: x[1]['profit_ratio'],
        reverse=True
    )

    # 最好
    print("\n【表现最好的20只股票】")
    print("-" * 100)
    print(f"{'股票代码':<10}{'收益率':<12}{'行业':<10}{'ROE':<10}{'净利润增长':<15}{'买入次数':<10}{'卖出次数':<10}")
    print("-" * 100)

    for stock, perf in sorted_stocks[:top_n]:
        fund = stock_fundamentals.get(stock, {})
        roe = fund.get('roe', 0)
        profit_growth = fund.get('profit_growth', 0)
        roe_str = f"{roe*100:.1f}%" if roe else "N/A"
        pg_str = f"{profit_growth*100:.1f}%" if profit_growth else "N/A"
        print(f"{stock:<10}{perf['profit_ratio']*100:>9.2f}%{'':2}{fund.get('industry_cat', 'N/A'):<10}{roe_str:<10}{pg_str:<15}{perf['buy_count']:<10}{perf['sell_count']:<10}")

    # 最差
    print("\n【表现最差的20只股票】")
    print("-" * 100)
    print(f"{'股票代码':<10}{'收益率':<12}{'行业':<10}{'ROE':<10}{'净利润增长':<15}{'买入次数':<10}{'卖出次数':<10}")
    print("-" * 100)

    for stock, perf in sorted_stocks[-top_n:]:
        fund = stock_fundamentals.get(stock, {})
        roe = fund.get('roe', 0)
        profit_growth = fund.get('profit_growth', 0)
        roe_str = f"{roe*100:.1f}%" if roe else "N/A"
        pg_str = f"{profit_growth*100:.1f}%" if profit_growth else "N/A"
        print(f"{stock:<10}{perf['profit_ratio']*100:>9.2f}%{'':2}{fund.get('industry_cat', 'N/A'):<10}{roe_str:<10}{pg_str:<15}{perf['buy_count']:<10}{perf['sell_count']:<10}")

    return sorted_stocks[:top_n], sorted_stocks[-top_n:]


def analyze_trading_patterns(stock_performance, stock_fundamentals):
    """分析交易模式"""
    print("\n" + "=" * 60)
    print("步骤8: 分析交易模式")
    print("=" * 60)

    # 按交易次数分组
    trade_groups = {
        '1次交易': [],
        '2-5次交易': [],
        '6-10次交易': [],
        '10次以上': []
    }

    for stock, perf in stock_performance.items():
        total_trades = perf['buy_count'] + perf['sell_count']
        if total_trades == 1:
            trade_groups['1次交易'].append((stock, perf))
        elif total_trades <= 5:
            trade_groups['2-5次交易'].append((stock, perf))
        elif total_trades <= 10:
            trade_groups['6-10次交易'].append((stock, perf))
        else:
            trade_groups['10次以上'].append((stock, perf))

    print("\n交易次数与收益关系:")
    print("-" * 60)
    print(f"{'交易次数分组':<20}{'股票数':<10}{'平均收益率':<15}{'盈利占比':<10}")
    print("-" * 60)

    for group, stocks in trade_groups.items():
        if not stocks:
            continue
        profits = [s[1]['profit_ratio'] * 100 for s in stocks]
        avg_profit = np.mean(profits)
        win_rate = len([p for p in profits if p > 0]) / len(profits) * 100
        print(f"{group:<20}{len(stocks):<10}{avg_profit:>12.2f}%{win_rate:>12.1f}%")


def generate_report(stock_performance, stock_fundamentals, output_file='stock_analysis_report.txt'):
    """生成详细报告"""
    print("\n" + "=" * 60)
    print(f"步骤9: 生成报告到 {output_file}")
    print("=" * 60)

    # 准备数据
    sorted_stocks = sorted(
        stock_performance.items(),
        key=lambda x: x[1]['profit_ratio'],
        reverse=True
    )

    # 计算统计数据
    all_profits = [p['profit_ratio'] * 100 for p in stock_performance.values()]
    winning_stocks = len([p for p in all_profits if p > 0])
    losing_stocks = len([p for p in all_profits if p < 0])

    # 行业统计
    industry_stats = defaultdict(lambda: {'stocks': [], 'total_profit': 0, 'profits': []})
    for stock, perf in stock_performance.items():
        fund = stock_fundamentals.get(stock, {})
        industry = fund.get('industry_cat', '其他')
        industry_stats[industry]['stocks'].append(stock)
        industry_stats[industry]['total_profit'] += perf['profit_value']
        industry_stats[industry]['profits'].append(perf['profit_ratio'] * 100)

    # ROE分组
    roe_groups = {'高ROE(>10%)': [], '中ROE(5-10%)': [], '低ROE(<5%)': [], '无数据': []}
    for stock, perf in stock_performance.items():
        fund = stock_fundamentals.get(stock, {})
        roe = fund.get('roe')
        if roe is None:
            roe_groups['无数据'].append(perf['profit_ratio'] * 100)
        elif roe > 0.10:
            roe_groups['高ROE(>10%)'].append(perf['profit_ratio'] * 100)
        elif roe > 0.05:
            roe_groups['中ROE(5-10%)'].append(perf['profit_ratio'] * 100)
        else:
            roe_groups['低ROE(<5%)'].append(perf['profit_ratio'] * 100)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("策略股票表现分析报告\n")
        f.write("=" * 80 + "\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # 总体统计
        f.write("一、总体统计\n")
        f.write("-" * 80 + "\n")
        f.write(f"分析股票数量: {len(stock_performance)}\n")
        f.write(f"盈利股票数: {winning_stocks} ({winning_stocks/len(stock_performance)*100:.1f}%)\n")
        f.write(f"亏损股票数: {losing_stocks} ({losing_stocks/len(stock_performance)*100:.1f}%)\n")
        f.write(f"平均收益率: {np.mean(all_profits):.2f}%\n")
        f.write(f"收益率中位数: {np.median(all_profits):.2f}%\n")
        f.write(f"收益率标准差: {np.std(all_profits):.2f}%\n")
        f.write(f"最大盈利: {max(all_profits):.2f}%\n")
        f.write(f"最大亏损: {min(all_profits):.2f}%\n\n")

        # 行业统计
        f.write("二、行业表现统计\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'行业':<10}{'股票数':<10}{'总收益':<15}{'平均收益':<15}{'盈利占比':<10}\n")
        f.write("-" * 80 + "\n")
        for industry, stats in sorted(industry_stats.items(), key=lambda x: x[1]['total_profit'], reverse=True):
            total = len(stats['profits'])
            winning = len([p for p in stats['profits'] if p > 0])
            win_rate = winning / total * 100 if total > 0 else 0
            avg = np.mean(stats['profits']) if total > 0 else 0
            f.write(f"{industry:<10}{total:<10}{stats['total_profit']:>12.2f}{avg:>15.2f}%{win_rate:>12.1f}%\n")

        # ROE分组
        f.write("\n三、ROE分组表现\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'ROE分组':<20}{'股票数':<10}{'平均收益率':<15}{'盈利占比':<10}\n")
        f.write("-" * 80 + "\n")
        for group, profits in roe_groups.items():
            if not profits:
                continue
            total = len(profits)
            winning = len([p for p in profits if p > 0])
            win_rate = winning / total * 100 if total > 0 else 0
            avg = np.mean(profits)
            f.write(f"{group:<20}{total:<10}{avg:>12.2f}%{win_rate:>12.1f}%\n")

        # 最好和最差
        f.write("\n四、表现最好的20只股票\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'股票代码':<10}{'收益率':<12}{'行业':<10}{'ROE':<10}{'净利润增长':<15}\n")
        f.write("-" * 80 + "\n")
        for stock, perf in sorted_stocks[:20]:
            fund = stock_fundamentals.get(stock, {})
            roe = fund.get('roe', 0)
            profit_growth = fund.get('profit_growth', 0)
            roe_str = f"{roe*100:.1f}%" if roe else "N/A"
            pg_str = f"{profit_growth*100:.1f}%" if profit_growth else "N/A"
            f.write(f"{stock:<10}{perf['profit_ratio']*100:>9.2f}%{'':2}{fund.get('industry_cat', 'N/A'):<10}{roe_str:<10}{pg_str:<15}\n")

        f.write("\n五、表现最差的20只股票\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'股票代码':<10}{'收益率':<12}{'行业':<10}{'ROE':<10}{'净利润增长':<15}\n")
        f.write("-" * 80 + "\n")
        for stock, perf in sorted_stocks[-20:]:
            fund = stock_fundamentals.get(stock, {})
            roe = fund.get('roe', 0)
            profit_growth = fund.get('profit_growth', 0)
            roe_str = f"{roe*100:.1f}%" if roe else "N/A"
            pg_str = f"{profit_growth*100:.1f}%" if profit_growth else "N/A"
            f.write(f"{stock:<10}{perf['profit_ratio']*100:>9.2f}%{'':2}{fund.get('industry_cat', 'N/A'):<10}{roe_str:<10}{pg_str:<15}\n")

        # 结论
        f.write("\n六、结论与建议\n")
        f.write("-" * 80 + "\n")

        # 找出表现好的行业
        good_industries = [(ind, stats) for ind, stats in industry_stats.items()
                          if np.mean(stats['profits']) > 0]
        bad_industries = [(ind, stats) for ind, stats in industry_stats.items()
                         if np.mean(stats['profits']) < 0]

        f.write("1. 行业表现分析:\n")
        if good_industries:
            f.write(f"   - 表现最好的行业: {', '.join([i[0] for i in sorted(good_industries, key=lambda x: np.mean(x[1]['profits']), reverse=True)[:3]])}\n")
        if bad_industries:
            f.write(f"   - 表现最差的行业: {', '.join([i[0] for i in sorted(bad_industries, key=lambda x: np.mean(x[1]['profits']))[:3]])}\n")

        # ROE分析
        f.write("\n2. ROE表现分析:\n")
        for group, profits in roe_groups.items():
            if profits:
                avg = np.mean(profits)
                f.write(f"   - {group}: 平均收益 {avg:.2f}%\n")

        f.write("\n3. 改进建议:\n")
        f.write("   - 针对表现差的行业，可以降低配置权重\n")
        f.write("   - 针对ROE高的股票，可以增加持仓权重\n")
        f.write("   - 考虑对不同行业使用不同的因子权重\n")

    print(f"报告已生成: {output_file}")
    return output_file


def main():
    """主函数"""
    # 步骤1: 运行回测
    log_content = run_backtest_and_capture()
    if not log_content:
        print("回测失败，程序退出")
        return

    # 保存完整日志
    with open('bt_result.log', 'w', encoding='utf-8') as f:
        f.write(log_content)

    # 步骤2: 解析交易
    trades = parse_trades(log_content)
    if not trades:
        print("未解析到交易记录，程序退出")
        return

    # 步骤3: 计算股票表现
    stock_performance = calculate_stock_performance(trades)

    # 步骤4: 获取基本面数据
    stock_fundamentals = get_fundamental_data(list(stock_performance.keys()))

    # 步骤5-8: 各种分析
    analyze_by_industry(stock_performance, stock_fundamentals)
    analyze_by_roe(stock_performance, stock_fundamentals)
    find_best_worst_stocks(stock_performance, stock_fundamentals)
    analyze_trading_patterns(stock_performance, stock_fundamentals)

    # 步骤9: 生成报告
    generate_report(stock_performance, stock_fundamentals)

    print("\n" + "=" * 60)
    print("分析完成！")
    print("=" * 60)


if __name__ == '__main__':
    main()
