"""实盘交易配置"""
import math
import os
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).parent.parent.resolve()


class TradeConfig:
    def __init__(self):
        config_path = Path(__file__).parent / "config.yaml"
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                self._cfg = yaml.safe_load(f) or {}
        else:
            self._cfg = {}
        self._trading_dates = None  # 缓存交易日历

    def _load_trading_calendar(self) -> list:
        """从 sh000001 数据加载真实交易日历"""
        if self._trading_dates is not None:
            return self._trading_dates

        index_file = self.bt_data_dir / "sh000001_qfq.csv"
        if not index_file.exists():
            # fallback: 使用近似日历
            return None

        try:
            df = pd.read_csv(str(index_file), parse_dates=['datetime'])
            self._trading_dates = sorted(df['datetime'].dt.date.unique().tolist())
            return self._trading_dates
        except Exception:
            return None

    @property
    def start_date(self) -> str:
        """实盘开始日期, 用于自动计算调仓日"""
        return self._cfg.get("start_date", datetime.today().strftime("%Y-%m-%d"))

    def max_position(self, total_asset: float, prices: dict) -> int:
        """根据资金和股票价格推算最大持仓数

        与回测对齐: 默认上限=10(from factor_config.yaml backtest.max_position)
        同时考虑价格可行性: 过滤价格过高的股票
        """
        config_max = self._cfg.get("max_position", 10)
        # 每只股票至少需要 price*100 元买1手
        n = int(total_asset / 5000)
        upper = max(3, min(n, config_max))

        # 价格感知: 如果高价股多, 适当降低上限
        if prices:
            sorted_prices = sorted(prices.values(), reverse=True)
            affordable = 0
            remaining = total_asset * (upper / config_max)  # 按比例分配
            for p in sorted_prices[:upper]:
                if p > 0 and remaining >= p * 100:
                    remaining -= p * 100
                    affordable += 1
                else:
                    break
            return max(3, min(affordable, upper))
        return upper

    def _count_trading_days(self, from_date, to_date) -> int:
        """计算两个日期之间的真实交易日数"""
        dates = self._load_trading_calendar()
        if dates is None:
            # fallback: 近似
            return int((to_date - from_date).days * 252 / 365)
        return sum(1 for d in dates if from_date <= d <= to_date)

    def _trading_day_offset(self, start, n_days: int):
        """从 start 开始偏移 n 个交易日, 返回日期"""
        dates = self._load_trading_calendar()
        if dates is None:
            # fallback
            return start + timedelta(days=int(n_days * 365 / 252))
        # 找到 start 之后的第 n_days 个交易日
        count = 0
        for d in dates:
            if d >= start:
                if count == n_days:
                    return d
                count += 1
        return dates[-1] if dates else start

    def is_rebalance_day(self, today: datetime) -> bool:
        """自动判断今天是否调仓日(每20个交易日, 使用真实交易日历)"""
        start = datetime.strptime(self.start_date, "%Y-%m-%d").date()
        today_date = today.date() if hasattr(today, 'date') else today
        trading_days = self._count_trading_days(start, today_date)
        return trading_days >= 0 and trading_days % 20 == 0

    def rebalance_info(self, today: datetime) -> dict:
        """返回调仓日信息: 是否调仓日、上次/下次调仓日"""
        start = datetime.strptime(self.start_date, "%Y-%m-%d").date()
        today_date = today.date() if hasattr(today, 'date') else today
        trading_days = self._count_trading_days(start, today_date)
        is_rebal = trading_days >= 0 and trading_days % 20 == 0

        # 上次调仓日: 最近的第 20 的倍数个交易日
        last_n = (trading_days // 20) * 20
        next_n = last_n + 20

        last_rebal = self._trading_day_offset(start, last_n)
        next_rebal = self._trading_day_offset(start, next_n)

        return {
            "is_rebalance_day": is_rebal,
            "last_rebalance": last_rebal.strftime("%Y-%m-%d") if last_rebal else "-",
            "next_rebalance": next_rebal.strftime("%Y-%m-%d") if next_rebal else "-",
        }

    @property
    def stock_data_dir(self) -> Path:
        return ROOT / "data" / "stock_data"

    @property
    def bt_data_dir(self) -> Path:
        return self.stock_data_dir / "backtrader_data"

    @property
    def fund_data_dir(self) -> Path:
        return self.stock_data_dir / "fundamental_data"

    @property
    def state_file(self) -> Path:
        return Path(__file__).parent / "portfolio_state.json"

    @property
    def report_dir(self) -> Path:
        return Path(__file__).parent / "reports"

    @property
    def rec_file(self) -> Path:
        return Path(__file__).parent / "last_recommendations.json"

    @property
    def notification_enabled(self) -> bool:
        return self._cfg.get("notification", {}).get("enabled", False)

    @property
    def notification_sckey(self) -> str:
        return self._cfg.get("notification", {}).get("sckey", "")

    @property
    def proxy_host(self) -> str:
        return self._cfg.get("proxy", {}).get("host", "")

    @property
    def proxy_auth_token(self) -> str:
        return self._cfg.get("proxy", {}).get("auth_token", "")

    @property
    def proxy_retry(self) -> int:
        return self._cfg.get("proxy", {}).get("retry", 30)
