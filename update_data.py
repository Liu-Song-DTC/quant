"""
独立数据更新脚本 — 从 main.py run 的数据更新部分抽取

用法:
  python update_data.py              # 增量更新（行情+基本面）
  python update_data.py --full       # 全量下载
  python update_data.py --days 365   # 只更新最近1年
"""
import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT / "data"))
sys.path.insert(0, str(ROOT / "trade"))

from data_manager import StockDataManager
from trade.config import TradeConfig


def get_universe(manager) -> list:
    symbols = []
    bt_dir = manager.backtrader_data_dir
    if bt_dir.exists():
        for f in bt_dir.iterdir():
            if f.name.endswith("_qfq.csv") and not f.name.startswith("._"):
                symbols.append(f.name.replace("_qfq.csv", ""))
    if not symbols:
        stock_list = manager.get_stock_list()
        if not stock_list.empty:
            symbols = stock_list["symbol"].tolist()
    return symbols


def main():
    parser = argparse.ArgumentParser(description="更新股票数据")
    parser.add_argument("--full", action="store_true", help="全量下载（忽略增量）")
    parser.add_argument("--days", type=int, default=730, help="回看天数（默认730）")
    args = parser.parse_args()

    cfg = TradeConfig()
    today = datetime.today()

    print("=" * 50)
    print(f"  数据更新 - {today.strftime('%Y-%m-%d')}")
    print(f"  模式: {'全量' if args.full else '增量'}")
    print("=" * 50)

    # 1. 初始化
    manager = StockDataManager(data_dir=str(cfg.stock_data_dir))
    manager.get_stock_list(force_update=args.full)

    # 2. 获取股票池
    symbols = get_universe(manager)
    print(f"股票池: {len(symbols)} 只")

    # 3. 下载行情
    print(f"下载行情 (force={args.full})...")
    manager.batch_download(symbols=symbols, force=args.full)

    # 4. 生成 backtrader 数据
    end_date = today.strftime("%Y-%m-%d")
    start_date = (today - timedelta(days=args.days)).strftime("%Y-%m-%d")
    print(f"生成前复权数据 ({start_date} ~ {end_date})...")
    manager.create_backtrader_data(symbols, start_date, end_date, adj_type="qfq")

    # 5. 基本面数据
    print("更新基本面数据...")
    manager.incremental_update_fundamental()

    print("\n数据更新完成")


if __name__ == "__main__":
    main()
