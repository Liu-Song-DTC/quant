# test_quarterly_filter.py
"""
测试季度股票池筛选
"""
import sys
import os
sys.path.insert(0, '.')

from core.fundamental import FundamentalData
from core.stock_pool_filter import StockPoolFilter


def test_quarterly_filter():
    """测试季度筛选"""
    FUND_PATH = '../data/stock_data/fundamental_data/'
    DATA_PATH = '../data/stock_data/backtrader_data/'

    # 获取所有有基本面数据的股票
    fund_files = [f.replace('.csv', '') for f in os.listdir(FUND_PATH) if f.endswith('.csv')]
    print(f"基本面数据股票数量: {len(fund_files)}")

    # 获取有交易数据的股票
    trade_files = [f.replace('_qfq.csv', '') for f in os.listdir(DATA_PATH)
                   if f.endswith('_qfq.csv') and f != 'sh000001_qfq.csv']

    # 取交集
    stock_codes = [c for c in trade_files if c in fund_files][:50]
    print(f"有交易+基本面数据: {len(stock_codes)}")

    # 加载基本面数据
    fd = FundamentalData(FUND_PATH, stock_codes)

    # 创建季度筛选器
    filter_obj = StockPoolFilter(fd, min_roe=10, min_profit_growth=0)

    # 测试几个日期
    test_dates = ['2024-03-31', '2024-06-30', '2024-09-30', '2024-12-31', '2025-03-31']

    print("\n季度筛选结果:")
    for date_str in test_dates:
        qualified = filter_obj.filter_quarterly(stock_codes, date_str)
        print(f"  {date_str}: {len(qualified)} 只股票符合条件")

        # 如果有符合的，显示前5只的评分
        if qualified:
            scores = []
            for code in list(qualified)[:10]:
                score = filter_obj.get_stock_score(code, date_str)
                scores.append((code, score))
            scores.sort(key=lambda x: x[1], reverse=True)
            print(f"    Top 5: {[(s[0], f'{s[1]:.1f}') for s in scores[:5]]}")


if __name__ == "__main__":
    test_quarterly_filter()
