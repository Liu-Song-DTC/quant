# core/fundamental.py
import pandas as pd
import logging
import os
from datetime import datetime

logger = logging.getLogger(__name__)


class FundamentalData:
    """基本面数据加载器 - 支持历史数据，防止信息泄露"""

    def __init__(self, data_path, stock_codes=None):
        """初始化基本面数据加载器（惰性加载，按需从磁盘读取单只股票CSV）

        Args:
            data_path: 基本面数据目录
            stock_codes: 可选，只索引指定股票的基本面数据
        """
        self.data_path = data_path
        self.stock_data = {}  # {code: DataFrame} 惰性缓存，首次访问时加载
        self._file_map = {}   # {code: file_path}
        self._scan_stock_files(stock_codes)

    def _scan_stock_files(self, stock_codes=None):
        """扫描可用股票文件路径（不加载数据）"""
        if not os.path.exists(self.data_path):
            print(f"基本面数据目录不存在: {self.data_path}")
            return

        if stock_codes:
            for code in stock_codes:
                fpath = os.path.join(self.data_path, f"{code}.csv")
                if os.path.exists(fpath):
                    self._file_map[code] = fpath
            print(f"基本面数据索引: {len(self._file_map)} 只股票（指定范围，惰性加载）")
        else:
            for f in os.listdir(self.data_path):
                if f.endswith('.csv'):
                    code = f.replace('.csv', '')
                    self._file_map[code] = os.path.join(self.data_path, f)
            print(f"基本面数据索引: {len(self._file_map)} 只股票（惰性加载）")

    def _load_stock(self, code):
        """按需加载单只股票的基本面CSV到缓存"""
        if code in self.stock_data:
            return
        fpath = self._file_map.get(code)
        if fpath is None:
            self.stock_data[code] = pd.DataFrame()  # 标记为已尝试，避免重复文件检查
            return
        try:
            df = pd.read_csv(fpath)
            if '报告期' in df.columns:
                df['报告期'] = df['报告期'].astype(str)
            if '数据可用日期' in df.columns:
                df['数据可用日期'] = df['数据可用日期'].astype(str)
            self.stock_data[code] = df
        except Exception:
            logger.warning(f"基本面数据读取失败 code={code} path={fpath}", exc_info=True)
            self.stock_data[code] = pd.DataFrame()

    def clear_stock_cache(self, code=None):
        """清除基本面数据缓存以释放内存。

        Args:
            code: 指定股票代码，None 则清除全部缓存。
        """
        if code is not None:
            self.stock_data.pop(code, None)
        else:
            self.stock_data.clear()

    def _get_available_data(self, code, current_date):
        """获取当前日期可用的基本面数据（防止信息泄露）

        Args:
            code: 股票代码
            current_date: 当前交易日期 (datetime或str)

        Returns:
            DataFrame: 可用的基本面数据（按报告期排序）
        """
        if code not in self.stock_data:
            self._load_stock(code)

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

    def _get_nth_latest(self, code, current_date, n=0):
        """获取第n+1新的基本面数据（n=0最新, n=1上一季度...）"""
        df = self._get_available_data(code, current_date)
        if len(df) <= n:
            return None
        return df.iloc[n].to_dict()

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
        except Exception:
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
        except Exception:
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
        except Exception:
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
        except Exception:
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
        except Exception:
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

    def get_profit_growth_improve(self, code, current_date):
        """盈利增长改善度 = 当前profit_growth - 上一季度profit_growth

        正值表示盈利加速，负值表示盈利减速
        """
        latest = self._get_latest(code, current_date)
        prev = self._get_nth_latest(code, current_date, n=1)
        if latest is None or prev is None:
            return None

        def _parse_growth(val):
            if val is None:
                return None
            try:
                if isinstance(val, str):
                    return float(val.strip('%')) / 100
                return float(val)
            except Exception:
                return None

        cur_pg = _parse_growth(latest.get('净利润-同比增长'))
        prev_pg = _parse_growth(prev.get('净利润-同比增长'))
        if cur_pg is not None and prev_pg is not None:
            return cur_pg - prev_pg
        return None

    def get_revenue_growth_improve(self, code, current_date):
        """营收增长改善度 = 当前revenue_growth - 上一季度revenue_growth

        正值表示营收加速，负值表示营收减速
        """
        latest = self._get_latest(code, current_date)
        prev = self._get_nth_latest(code, current_date, n=1)
        if latest is None or prev is None:
            return None

        def _parse_growth(val):
            if val is None:
                return None
            try:
                if isinstance(val, str):
                    return float(val.strip('%')) / 100
                return float(val)
            except Exception:
                return None

        cur_rg = _parse_growth(latest.get('营业总收入-同比增长'))
        prev_rg = _parse_growth(prev.get('营业总收入-同比增长'))
        if cur_rg is not None and prev_rg is not None:
            return cur_rg - prev_rg
        return None

    # ========== 估值因子 (Fix#1) ==========

    def get_pe(self, code, current_date, price=None):
        """获取市盈率(PE) — 需要当前价格

        返回 PE = price / EPS。值越低越便宜。
        若price未提供，尝试从latest数据中获取。
        """
        latest = self._get_latest(code, current_date)
        if latest is None:
            return None
        eps = latest.get('每股收益')
        if eps is None or eps <= 0:
            return None
        try:
            eps = float(eps)
        except (ValueError, TypeError):
            return None
        if price is None or price <= 0:
            return None
        return price / eps

    def get_pb(self, code, current_date, price=None):
        """获取市净率(PB) — 需要当前价格

        返回 PB = price / BPS。值越低越便宜。
        """
        latest = self._get_latest(code, current_date)
        if latest is None:
            return None
        bps = latest.get('每股净资产')
        if bps is None or bps <= 0:
            return None
        try:
            bps = float(bps)
        except (ValueError, TypeError):
            return None
        if price is None or price <= 0:
            return None
        return price / bps
