# core/stock_pool_filter.py
"""
季度股票池筛选器
每季度根据基本面因子筛选优质股票
"""
import pandas as pd
from .fundamental import FundamentalData


class StockPoolFilter:
    """
    季度股票池筛选 - 根据基本面因子预筛选股票
    """

    def __init__(self, fundamental_data: FundamentalData, min_roe=10, min_profit_growth=0):
        """
        Args:
            fundamental_data: 基本面数据
            min_roe: 最低ROE要求(%)
            min_profit_growth: 最低净利润增长要求(%)
        """
        self.fundamental_data = fundamental_data
        self.min_roe = min_roe
        self.min_profit_growth = min_profit_growth
        self.current_quarter_stocks = set()

    def get_quarter(self, date):
        """获取季度标识"""
        if isinstance(date, str):
            date = pd.to_datetime(date)
        return (date.year, (date.month - 1) // 3 + 1)

    def filter_quarterly(self, stock_codes: list, date):
        """
        季度筛选股票池

        Args:
            stock_codes: 候选股票列表
            date: 当前日期

        Returns:
            set: 筛选后的股票代码集合
        """
        quarter = self.get_quarter(date)

        # 检查是否需要更新
        if quarter == getattr(self, '_last_quarter', None):
            return self.current_quarter_stocks

        # 筛选符合条件的股票
        qualified = set()
        for code in stock_codes:
            if self._is_qualified(code, date):
                qualified.add(code)

        self.current_quarter_stocks = qualified
        self._last_quarter = quarter

        return qualified

    def _is_qualified(self, code, date):
        """检查股票是否满足基本面条件"""
        # ROE
        roe = self.fundamental_data.get_roe(code, date)
        if roe is None or roe < self.min_roe:
            return False

        # 净利润增长
        profit_growth = self.fundamental_data.get_profit_growth(code, date)
        if profit_growth is None or profit_growth < self.min_profit_growth:
            return False

        return True

    def get_stock_score(self, code, date):
        """
        获取股票基本面评分（用于排序）

        Returns:
            float: 基本面评分 (0-100)
        """
        score = 0.0

        # ROE评分 (最高40分)
        roe = self.fundamental_data.get_roe(code, date)
        if roe is not None:
            score += min(roe, 40)

        # 净利润增长评分 (最高30分)
        profit_growth = self.fundamental_data.get_profit_growth(code, date)
        if profit_growth is not None:
            score += min(profit_growth / 2, 30)

        # 营业收入增长评分 (最高20分)
        revenue_growth = self.fundamental_data.get_revenue_growth(code, date)
        if revenue_growth is not None:
            score += min(revenue_growth / 3, 20)

        # 每股收益评分 (最高10分)
        eps = self.fundamental_data.get_eps(code, date)
        if eps is not None and eps > 0:
            score += min(eps * 5, 10)

        return score


def create_quarterly_filter(fundamental_data, min_roe=10, min_profit_growth=0):
    """创建季度筛选器"""
    return StockPoolFilter(fundamental_data, min_roe, min_profit_growth)
