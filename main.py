"""
量化交易系统 — 主入口

用法:
  python main.py run                    # 每日运行: 数据更新 + 信号 + 建议
  python main.py run --skip-update      # 跳过数据更新，只生成信号
  python main.py run --force            # 非调仓日也强制运行
  python main.py status                 # 查看当前持仓
"""
import argparse
from pathlib import Path

ROOT = Path(__file__).parent.resolve()


def main():
    parser = argparse.ArgumentParser(description="量化交易系统")
    sub = parser.add_subparsers(dest="command")

    p_run = sub.add_parser("run", help="每日运行: 数据更新+信号+建议")
    p_run.add_argument("--skip-update", action="store_true")
    p_run.add_argument("--force", action="store_true", help="非调仓日也强制运行")

    sub.add_parser("status", help="查看当前持仓")

    args = parser.parse_args()

    if args.command == "run":
        from trade.runner import run_daily
        run_daily(skip_update=args.skip_update, force=args.force)
    elif args.command == "status":
        from trade.runner import show_status
        show_status()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
