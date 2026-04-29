"""实盘交易服务器 — CLI入口"""
import argparse
import json
import os
import sys
from datetime import date, datetime
from pathlib import Path

import yaml


def load_trade_config() -> dict:
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(base: str, config_path: str) -> str:
    """将配置中的相对路径转为绝对路径"""
    p = Path(config_path)
    if p.is_absolute():
        return str(p)
    return str((Path(base) / p).resolve())


def cmd_run(args):
    """每日运行：数据更新 → 信号生成 → 交易建议"""
    config = load_trade_config()
    base_dir = Path(__file__).parent

    # 解析路径
    stock_data_dir = resolve_path(base_dir, config["data"]["stock_data_dir"])
    bt_data_dir = resolve_path(base_dir, config["data"]["backtrader_data_dir"])
    fund_data_dir = resolve_path(base_dir, config["data"]["fundamental_data_dir"])
    state_file = resolve_path(base_dir, config.get("state_file", "portfolio_state.json"))
    report_dir = resolve_path(base_dir, config.get("report_dir", "reports/"))

    # Step 1: 更新数据
    if not args.skip_update:
        from data_updater import DataUpdater
        print("=" * 50)
        print("Step 1: 更新数据")
        print("=" * 50)
        updater = DataUpdater(stock_data_dir)
        stats = updater.update_daily(update_fundamental=args.update_fundamental)
        print(f"数据更新完成: {stats}")
    else:
        print("跳过数据更新 (--skip-update)")

    # Step 2: 生成信号和建议
    print("\n" + "=" * 50)
    print("Step 2: 生成交易建议")
    print("=" * 50)

    # 添加项目路径
    strategy_dir = str(base_dir.parent / "strategy")
    if strategy_dir not in sys.path:
        sys.path.insert(0, strategy_dir)

    from signal_runner import SignalRunner
    from portfolio_state import PortfolioState
    from recommender import Recommender
    from reporter import Reporter

    # 加载组合状态
    ps = PortfolioState.load(state_file)
    if ps.cash == 0 and not ps.data.get("positions"):
        ps.init_cash(config["portfolio"]["init_cash"])
        print(f"初始化组合: ¥{config['portfolio']['init_cash']:,.0f}")

    # 生成信号
    runner = SignalRunner(
        bt_data_dir=bt_data_dir,
        fund_data_dir=fund_data_dir,
        max_position=config["portfolio"]["max_position"],
    )
    result = runner.run(
        current_positions=ps.get_current_positions(runner.get_prices()),
        cash=ps.cash,
        cost=ps.get_cost_basis(),
    )

    if result is None:
        print("信号生成失败，请检查数据")
        return

    # 生成建议
    rec = Recommender()
    recommendations = rec.generate(
        target_positions=result["target_positions"],
        current_positions=ps.get_current_positions(result["prices"]),
        prices=result["prices"],
        cash=ps.cash,
        cost=ps.get_cost_basis(),
        market_regime=result.get("market_regime", {}),
        selections=result.get("selections", []),
    )

    # 输出报告
    reporter = Reporter(report_dir=report_dir)
    reporter.print_report(recommendations, ps.summary(result["prices"]))
    report_path = reporter.save_report(recommendations, ps.summary(result["prices"]))

    # 微信推送
    if config["notification"]["enabled"] and config["notification"]["wechat_sckey"]:
        from notifier import WeChatNotifier
        notifier = WeChatNotifier(config["notification"]["wechat_sckey"])
        notifier.send_recommendations(recommendations, report_path)
        print("微信推送已发送")

    # 保存建议供后续confirm使用
    rec_file = resolve_path(base_dir, "last_recommendations.json")
    with open(rec_file, "w", encoding="utf-8") as f:
        json.dump(recommendations, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n建议已保存: {rec_file}")


def cmd_status(args):
    """查看当前持仓"""
    config = load_trade_config()
    base_dir = Path(__file__).parent
    state_file = resolve_path(base_dir, config.get("state_file", "portfolio_state.json"))

    from portfolio_state import PortfolioState
    ps = PortfolioState.load(state_file)

    # 尝试获取当前价格
    prices = {}
    if ps.data.get("positions"):
        bt_data_dir = resolve_path(base_dir, config["data"]["backtrader_data_dir"])
        strategy_dir = str(base_dir.parent / "strategy")
        if strategy_dir not in sys.path:
            sys.path.insert(0, strategy_dir)
        from signal_runner import SignalRunner
        runner = SignalRunner(bt_data_dir=bt_data_dir, fund_data_dir="", max_position=10)
        prices = runner.get_prices()

    summary = ps.summary(prices)
    print(f"\n{'=' * 50}")
    print(f"  组合状态  更新于: {summary['last_update'] or '未初始化'}")
    print(f"{'=' * 50}")
    print(f"  现金:     ¥{summary['cash']:>12,.2f}")
    print(f"  持仓市值: ¥{summary['market_value']:>12,.2f}")
    print(f"  总资产:   ¥{summary['total_value']:>12,.2f}")
    print(f"  持仓数:   {len(summary['positions'])}")

    if summary["positions"]:
        print(f"\n  {'代码':<8} {'数量':>6} {'成本价':>10} {'现价':>10} {'市值':>12} {'盈亏%':>8}")
        print(f"  {'-' * 60}")
        for p in summary["positions"]:
            pnl_str = f"{p['pnl_pct']:+.2%}"
            print(f"  {p['code']:<8} {p['shares']:>6} {p['cost_price']:>10.2f} "
                  f"{p['current_price']:>10.2f} {p['market_value']:>12,.2f} {pnl_str:>8}")

    print(f"\n{'=' * 50}")


def cmd_confirm(args):
    """确认执行交易"""
    config = load_trade_config()
    base_dir = Path(__file__).parent
    state_file = resolve_path(base_dir, config.get("state_file", "portfolio_state.json"))

    from portfolio_state import PortfolioState
    ps = PortfolioState.load(state_file)

    trades = []

    # 解析买入
    if args.buy:
        for item in args.buy:
            code, shares = item.split(":")
            trades.append({"action": "buy", "code": code, "shares": int(shares), "price": 0})

    # 解析卖出
    if args.sell:
        for item in args.sell:
            code, shares = item.split(":")
            trades.append({"action": "sell", "code": code, "shares": int(shares), "price": 0})

    # 如果从last_recommendations确认
    if args.from_rec:
        rec_file = resolve_path(base_dir, "last_recommendations.json")
        if os.path.exists(rec_file):
            with open(rec_file, "r", encoding="utf-8") as f:
                rec = json.load(f)
            for trade in rec.get("trades", []):
                trades.append(trade)

    if not trades:
        print("没有交易需要确认。使用 --buy CODE:SHARES 或 --sell CODE:SHARES 或 --from-rec")
        return

    # 需要价格 — 尝试获取
    bt_data_dir = resolve_path(base_dir, config["data"]["backtrader_data_dir"])
    strategy_dir = str(base_dir.parent / "strategy")
    if strategy_dir not in sys.path:
        sys.path.insert(0, strategy_dir)
    from signal_runner import SignalRunner
    runner = SignalRunner(bt_data_dir=bt_data_dir, fund_data_dir="", max_position=10)
    prices = runner.get_prices()

    # 填充价格
    for trade in trades:
        if trade["price"] == 0:
            trade["price"] = prices.get(trade["code"], 0)
        if trade["price"] == 0:
            price_input = input(f"  请输入 {trade['code']} 成交价: ")
            trade["price"] = float(price_input)

    # 确认
    print(f"\n确认交易:")
    total_buy = 0
    total_sell = 0
    for trade in trades:
        amount = trade["shares"] * trade["price"]
        if trade["action"] == "buy":
            total_buy += amount
            print(f"  买入 {trade['code']} × {trade['shares']}股 @ ¥{trade['price']:.2f} = ¥{amount:,.2f}")
        else:
            total_sell += amount
            print(f"  卖出 {trade['code']} × {trade['shares']}股 @ ¥{trade['price']:.2f} = ¥{amount:,.2f}")
    print(f"  合计: 买入 ¥{total_buy:,.2f}, 卖出 ¥{total_sell:,.2f}")

    confirm = input("\n确认执行? (y/N): ")
    if confirm.lower() == "y":
        ps.update_after_trade(trades)
        print("交易已确认，组合状态已更新")
    else:
        print("已取消")


def cmd_history(args):
    """查看历史交易"""
    config = load_trade_config()
    base_dir = Path(__file__).parent
    state_file = resolve_path(base_dir, config.get("state_file", "portfolio_state.json"))

    from portfolio_state import PortfolioState
    ps = PortfolioState.load(state_file)

    history = ps.data.get("trade_history", [])
    if not history:
        print("暂无交易记录")
        return

    print(f"\n{'=' * 70}")
    print(f"  交易历史  共 {len(history)} 笔")
    print(f"{'=' * 70}")
    print(f"  {'日期':<12} {'操作':<6} {'代码':<8} {'数量':>6} {'价格':>10}")
    print(f"  {'-' * 50}")
    for t in history[-30:]:  # 最近30笔
        action = "买入" if t["action"] == "buy" else "卖出"
        print(f"  {t['date']:<12} {action:<6} {t['code']:<8} {t['shares']:>6} {t['price']:>10.2f}")
    print(f"{'=' * 70}")


def main():
    parser = argparse.ArgumentParser(description="实盘交易系统")
    subparsers = parser.add_subparsers(dest="command")

    # run
    run_parser = subparsers.add_parser("run", help="每日运行")
    run_parser.add_argument("--skip-update", action="store_true", help="跳过数据更新")
    run_parser.add_argument("--update-fundamental", action="store_true", help="同时更新基本面数据")

    # status
    subparsers.add_parser("status", help="查看当前持仓")

    # confirm
    confirm_parser = subparsers.add_parser("confirm", help="确认执行交易")
    confirm_parser.add_argument("--buy", nargs="*", help="买入 CODE:SHARES")
    confirm_parser.add_argument("--sell", nargs="*", help="卖出 CODE:SHARES")
    confirm_parser.add_argument("--from-rec", action="store_true", help="从上次建议确认")

    # history
    subparsers.add_parser("history", help="查看历史交易")

    args = parser.parse_args()

    if args.command == "run":
        cmd_run(args)
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "confirm":
        cmd_confirm(args)
    elif args.command == "history":
        cmd_history(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
