import os
import sys
from datetime import datetime, timedelta
from pathlib import Path


class DataUpdater:
    """数据更新 — 封装现有 StockDataManager 做增量更新"""

    def __init__(self, stock_data_dir: str):
        # 将 data/ 目录加入 sys.path 以便导入
        data_dir = str(Path(stock_data_dir).parent)
        if data_dir not in sys.path:
            sys.path.insert(0, data_dir)

        from data_manager import StockDataManager
        self.manager = StockDataManager(data_dir=stock_data_dir)

    def _get_current_universe(self) -> list:
        """获取当前股票池（backtrader_data目录中已有的股票）"""
        bt_dir = self.manager.backtrader_data_dir
        symbols = []
        if bt_dir.exists():
            for f in bt_dir.iterdir():
                if f.name.endswith("_qfq.csv"):
                    symbols.append(f.name.replace("_qfq.csv", ""))
        if not symbols:
            # Fallback: 从股票列表获取
            stock_list = self.manager.get_stock_list()
            if not stock_list.empty:
                symbols = stock_list["symbol"].tolist()
                symbols.insert(0, "sh000001")
        return symbols

    def update_daily(self, update_fundamental: bool = False) -> dict:
        """每日增量更新

        Returns:
            dict with update stats
        """
        stats = {"price_updated": 0, "fundamental_updated": False, "errors": []}

        try:
            # 1. 更新股票列表
            print("更新股票列表...")
            self.manager.get_stock_list(force_update=True)

            # 2. 增量更新行情
            symbols = self._get_current_universe()
            print(f"增量更新 {len(symbols)} 只股票行情...")
            results = self.manager.batch_download(symbols=symbols, force=False)
            stats["price_updated"] = sum(1 for _, s in (results or []) if s == "success")

            # 3. 重新生成backtrader格式数据
            end_date = datetime.today().strftime("%Y-%m-%d")
            start_date = (datetime.today() - timedelta(days=730)).strftime("%Y-%m-%d")
            print(f"生成backtrader格式数据 ({start_date} ~ {end_date})...")
            self.manager.create_backtrader_data(
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                adj_type="qfq",
            )

            # 4. 基本面数据（可选，季度更新即可）
            if update_fundamental:
                print("更新基本面数据...")
                self.manager.incremental_update_fundamental()
                stats["fundamental_updated"] = True

        except Exception as e:
            stats["errors"].append(str(e))
            print(f"数据更新错误: {e}")

        return stats
