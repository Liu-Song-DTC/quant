# core/fundamental.py
import pandas as pd
import os
from datetime import datetime


class FundamentalData:
    """基本面数据加载器 - 支持历史数据，防止信息泄露"""

    def __init__(self, data_path, stock_codes=None):
        """初始化基本面数据加载器

        Args:
            data_path: 基本面数据目录
            stock_codes: 可选，只加载指定股票的基本面数据（提高效率）
        """
        self.data_path = data_path
        self.stock_data = {}  # {code: DataFrame}
        self._load_all_stocks(stock_codes)

    def _load_all_stocks(self, stock_codes=None):
        """加载股票的基本面数据

        Args:
            stock_codes: 可选，只加载指定股票
        """
        if not os.path.exists(self.data_path):
            print(f"基本面数据目录不存在: {self.data_path}")
            return

        if stock_codes:
            # 只加载指定的股票
            files = [f"{code}.csv" for code in stock_codes]
            files = [f for f in files if os.path.exists(os.path.join(self.data_path, f))]
            print(f"加载基本面数据: {len(files)} 只股票（指定范围）")
        else:
            # 加载所有股票文件
            files = [f for f in os.listdir(self.data_path) if f.endswith('.csv')]
            print(f"加载基本面数据: {len(files)} 只股票")

        for f in files:
            code = f.replace('.csv', '')
            try:
                df = pd.read_csv(os.path.join(self.data_path, f))
                # 确保数据类型正确
                if '报告期' in df.columns:
                    df['报告期'] = df['报告期'].astype(str)
                if '数据可用日期' in df.columns:
                    df['数据可用日期'] = df['数据可用日期'].astype(str)
                self.stock_data[code] = df
            except Exception as e:
                continue

        print(f"成功加载 {len(self.stock_data)} 只股票的基本面数据")

    def _get_available_data(self, code, current_date):
        """获取当前日期可用的基本面数据（防止信息泄露）

        Args:
            code: 股票代码
            current_date: 当前交易日期 (datetime或str)

        Returns:
            DataFrame: 可用的基本面数据（按报告期排序）
        """
        if code not in self.stock_data:
            return pd.DataFrame()

        df = self.stock_data[code]

        # 将current_date转换为字符串格式
        if isinstance(current_date, datetime):
            current_date = current_date.strftime('%Y%m%d')
        elif hasattr(current_date, 'strftime'):
            current_date = current_date.strftime('%Y%m%d')
        else:
            current_date = str(current_date).replace('-', '')

        # 只返回数据可用日期 <= 当前日期的数据
        if '数据可用日期' in df.columns:
            df = df[df['数据可用日期'] <= current_date]

        # 按报告期排序，取最新的
        if '报告期' in df.columns:
            df = df.sort_values('报告期', ascending=False)

        return df

    def _get_latest(self, code, current_date):
        """获取最新可用的基本面数据"""
        df = self._get_available_data(code, current_date)
        if df.empty:
            return None
        return df.iloc[0].to_dict()

    def get_eps(self, code, current_date):
        """获取每股收益"""
        latest = self._get_latest(code, current_date)
        if latest is None:
            return None
        return latest.get('每股收益', None)

    def get_revenue(self, code, current_date):
        """获取营业收入"""
        latest = self._get_latest(code, current_date)
        if latest is None:
            return None
        return latest.get('营业总收入-营业总收入', None)

    def get_revenue_growth(self, code, current_date):
        """获取营业收入增长率"""
        latest = self._get_latest(code, current_date)
        if latest is None:
            return None
        growth = latest.get('营业总收入-同比增长', None)
        if growth is None:
            return None
        try:
            if isinstance(growth, str):
                return float(growth.strip('%')) / 100
            return float(growth)
        except:
            return None

    def get_profit(self, code, current_date):
        """获取净利润"""
        latest = self._get_latest(code, current_date)
        if latest is None:
            return None
        return latest.get('净利润-净利润', None)

    def get_profit_growth(self, code, current_date):
        """获取净利润增长率"""
        latest = self._get_latest(code, current_date)
        if latest is None:
            return None
        growth = latest.get('净利润-同比增长', None)
        if growth is None:
            return None
        try:
            if isinstance(growth, str):
                return float(growth.strip('%')) / 100
            return float(growth)
        except:
            return None

    def get_roe(self, code, current_date):
        """获取净资产收益率"""
        latest = self._get_latest(code, current_date)
        if latest is None:
            return None
        roe = latest.get('净资产收益率', None)
        if roe is None:
            return None
        try:
            if isinstance(roe, str):
                return float(roe.strip('%')) / 100
            return float(roe)
        except:
            return None

    def get_bps(self, code, current_date):
        """获取每股净资产"""
        latest = self._get_latest(code, current_date)
        if latest is None:
            return None
        return latest.get('每股净资产', None)

    def get_industry(self, code, current_date):
        """获取行业"""
        latest = self._get_latest(code, current_date)
        if latest is None:
            return None
        return latest.get('所处行业', None)

    def is_st(self, code, current_date):
        """是否ST股票"""
        latest = self._get_latest(code, current_date)
        if latest is None:
            return False
        name = latest.get('股票简称', '')
        return '*ST' in str(name) or 'ST' in str(name)

    def get_fundamental_score(self, code, current_date):
        """基本面综合评分"""
        score = 0.0

        # ROE评分
        roe = self.get_roe(code, current_date)
        if roe:
            score += min(roe * 100, 30)  # 最多30分

        # 净利润增长评分
        growth = self.get_profit_growth(code, current_date)
        if growth:
            if growth > 0.5:
                score += 25
            elif growth > 0.2:
                score += 15
            elif growth > 0:
                score += 5

        # 每股收益
        eps = self.get_eps(code, current_date)
        if eps and eps > 0:
            score += min(eps * 10, 20)

        # 营业收入增长
        revenue_growth = self.get_revenue_growth(code, current_date)
        if revenue_growth:
            if revenue_growth > 0.3:
                score += 15
            elif revenue_growth > 0.1:
                score += 10

        return score

    def is优质股(self, code, current_date):
        """是否是优质股：ROE > 10% 且 净利润增长 > 0"""
        roe = self.get_roe(code, current_date)
        profit_growth = self.get_profit_growth(code, current_date)

        if roe is None or profit_growth is None:
            return False

        return roe > 0.10 and profit_growth > 0

    # ========== 新增基本面指标 ==========

    def get_debt_ratio(self, code, current_date):
        """获取资产负债率"""
        latest = self._get_latest(code, current_date)
        if latest is None:
            return None
        debt = latest.get('zcfz_资产负债率', None)
        if debt is None:
            return None
        try:
            if isinstance(debt, str):
                return float(debt.strip('%')) / 100
            return float(debt)
        except:
            return None

    def get_gross_margin(self, code, current_date):
        """获取销售毛利率"""
        latest = self._get_latest(code, current_date)
        if latest is None:
            return None
        margin = latest.get('销售毛利率', None)
        if margin is None:
            return None
        try:
            if isinstance(margin, str):
                return float(margin.strip('%')) / 100
            return float(margin)
        except:
            return None

    def get_operating_cash_flow(self, code, current_date):
        """获取经营性现金流"""
        latest = self._get_latest(code, current_date)
        if latest is None:
            return None
        return latest.get('xjll_经营性现金流-现金流量净额', None)

    def get_cf_to_profit(self, code, current_date):
        """获取经营性现金流/净利润比率"""
        latest = self._get_latest(code, current_date)
        if latest is None:
            return None
        cf = latest.get('xjll_经营性现金流-现金流量净额', None)
        profit = latest.get('lrb_净利润', None)
        if cf is not None and profit is not None and profit > 0:
            return cf / profit
        return None

    def get_total_assets(self, code, current_date):
        """获取总资产"""
        latest = self._get_latest(code, current_date)
        if latest is None:
            return None
        return latest.get('zcfz_资产-总资产', None)

    def get_total_liability(self, code, current_date):
        """获取总负债"""
        latest = self._get_latest(code, current_date)
        if latest is None:
            return None
        return latest.get('zcfz_负债-总负债', None)

    def get_quarterly_report_date(self, code, current_date):
        """获取最新财报的报告期"""
        df = self._get_available_data(code, current_date)
        if df.empty:
            return None
        return df.iloc[0].get('报告期', None)
