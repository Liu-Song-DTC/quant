"""
量化交易系统 — 主入口

用法:
  python main.py run                    # 每日运行: 数据更新 + 信号 + 建议
  python main.py run --skip-update      # 跳过数据更新
  python main.py status                 # 查看当前持仓
  python main.py confirm -i             # 交互式逐笔确认交易
  python main.py history                # 查看历史交易
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent.resolve()


def main():
    parser = argparse.ArgumentParser(description="量化交易系统")
    sub = parser.add_subparsers(dest="command")

    p_run = sub.add_parser("run", help="每日运行: 数据更新+信号+建议")
    p_run.add_argument("--skip-update", action="store_true")
    p_run.add_argument("--update-fundamental", action="store_true")

    sub.add_parser("status", help="查看当前持仓")

    p_cf = sub.add_parser("confirm", help="确认交易")
    p_cf.add_argument("-i", "--interactive", action="store_true", help="交互式逐笔确认")
    p_cf.add_argument("--buy", nargs="*", help="CODE:SHARES[:PRICE]")
    p_cf.add_argument("--sell", nargs="*", help="CODE:SHARES[:PRICE]")
    p_cf.add_argument("--from-rec", action="store_true")

    sub.add_parser("history", help="查看历史交易")

    args = parser.parse_args()

    # 延迟导入，避免启动慢
    if args.command == "run":
        from trade.runner import run_daily
        run_daily(skip_update=args.skip_update, update_fundamental=args.update_fundamental)
    elif args.command == "status":
        from trade.runner import show_status
        show_status()
    elif args.command == "confirm":
        from trade.confirm import confirm_trades
        confirm_trades(args)
    elif args.command == "history":
        from trade.confirm import show_history
        show_history()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
