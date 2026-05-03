"""
量化交易系统 — 主入口

用法:
  python main.py run                    # 每日运行: 数据更新 + 信号 + 建议
  python main.py run --skip-update      # 跳过数据更新，只生成信号
  python main.py run --force            # 非调仓日也强制运行
  python main.py status                 # 查看当前持仓和止损
  python main.py sentiment              # 运行行业情绪分析
  python main.py sentiment --date 2026-05-01  # 指定日期
  python main.py sentiment --no-notify        # 跳过微信通知
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

    p_sentiment = sub.add_parser("sentiment", help="运行行业情绪分析")
    p_sentiment.add_argument("--date", type=str, default=None, help="指定日期 (YYYY-MM-DD)")
    p_sentiment.add_argument("--no-notify", action="store_true", help="跳过微信通知")

    args = parser.parse_args()

    if args.command == "run":
        from trade.runner import run_daily
        run_daily(skip_update=args.skip_update, force=args.force)
    elif args.command == "status":
        from trade.runner import show_status
        show_status()
    elif args.command == "sentiment":
        import sys
        sys.path.insert(0, str(ROOT / "strategy"))
        from datetime import date as date_type
        from sentiment.orchestrator import SentimentOrchestrator
        from core.config_loader import load_config

        config = load_config()
        orchestrator = SentimentOrchestrator(config)

        target_date = date_type.fromisoformat(args.date) if args.date else date_type.today()
        scores = orchestrator.run_daily(target_date=target_date, notify=not args.no_notify)

        if scores and not args.no_notify:
            from trade.config import TradeConfig
            cfg = TradeConfig()
            if cfg.notification_enabled and cfg.notification_sckey:
                from trade.notifier import Notifier
                Notifier(cfg.notification_sckey).send_industry_sentiment(
                    target_date.isoformat(), scores
                )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
