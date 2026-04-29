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
    if recommendations.get("buys") or recommendations.get("sells"):
        print(f"交易完成后请运行: python server.py confirm -i")


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
    """确认执行交易 — 交互式逐笔确认"""
    config = load_trade_config()
    base_dir = Path(__file__).parent
    state_file = resolve_path(base_dir, config.get("state_file", "portfolio_state.json"))

    from portfolio_state import PortfolioState
    ps = PortfolioState.load(state_file)

    # 收集待确认交易
    pending = []

    if args.interactive:
        # 交互模式：从上次建议逐笔确认
        rec_file = resolve_path(base_dir, "last_recommendations.json")
        if not os.path.exists(rec_file):
            print("没有上次建议记录，请先运行 python server.py run")
            return
        with open(rec_file, "r", encoding="utf-8") as f:
            rec = json.load(f)
        pending = rec.get("trades", [])
        if not pending:
            print("上次建议中没有交易")
            return

    elif args.from_rec:
        # 快速模式：直接从建议确认（使用推荐价）
        rec_file = resolve_path(base_dir, "last_recommendations.json")
        if not os.path.exists(rec_file):
            print("没有上次建议记录")
            return
        with open(rec_file, "r", encoding="utf-8") as f:
            rec = json.load(f)
        pending = rec.get("trades", [])
        if not pending:
            print("上次建议中没有交易")
            return

    elif args.buy or args.sell:
        # 手动模式
        if args.buy:
            for item in args.buy:
                parts = item.split(":")
                code = parts[0]
                shares = int(parts[1])
                price = float(parts[2]) if len(parts) > 2 else 0
                pending.append({"action": "buy", "code": code, "shares": shares, "price": price})
        if args.sell:
            for item in args.sell:
                parts = item.split(":")
                code = parts[0]
                shares = int(parts[1])
                price = float(parts[2]) if len(parts) > 2 else 0
                pending.append({"action": "sell", "code": code, "shares": shares, "price": price})

    else:
        print("用法:")
        print("  python server.py confirm -i          # 交互式逐笔确认（推荐）")
        print("  python server.py confirm --from-rec   # 快速确认上次建议")
        print("  python server.py confirm --buy 600519:100:75.5")
        print("  python server.py confirm --sell 601318:300:48.2")
        return

    if not pending:
        print("没有交易需要确认")
        return

    # 尝试获取参考价格
    ref_prices = {}
    bt_data_dir = resolve_path(base_dir, config["data"]["backtrader_data_dir"])
    strategy_dir = str(base_dir.parent / "strategy")
    if strategy_dir not in sys.path:
        sys.path.insert(0, strategy_dir)
    try:
        from signal_runner import SignalRunner
        runner = SignalRunner(bt_data_dir=bt_data_dir, fund_data_dir="", max_position=10)
        ref_prices = runner.get_prices()
    except Exception:
        pass

    # 交互式逐笔确认
    confirmed = []
    action_names = {"buy": "买入", "sell": "卖出"}

    print(f"\n{'═' * 60}")
    print(f"  交易确认  共 {len(pending)} 笔待确认")
    print(f"  每笔可修改: 数量(shares)、成交价(price)、跳过(skip)")
    print(f"{'═' * 60}")

    for i, trade in enumerate(pending):
        action = action_names.get(trade["action"], trade["action"])
        code = trade["code"]
        ref_price = ref_prices.get(code, trade.get("price", 0))
        rec_shares = trade["shares"]
        rec_price = trade.get("price", ref_price)

        print(f"\n  [{i+1}/{len(pending)}] {action} {code}")
        print(f"    建议: ×{rec_shares}股 @ ¥{rec_price:.2f} (参考价: ¥{ref_price:.2f})")

        # 交互输入
        while True:
            resp = input(f"    确认? (y=确认/s=跳过/修改数量/修改价格): ").strip().lower()

            if resp in ("y", ""):
                # 使用建议的数量和价格
                actual_shares = rec_shares
                actual_price = rec_price
                # 如果没有价格，要求输入
                if actual_price == 0:
                    try:
                        actual_price = float(input(f"    请输入 {code} 成交价: "))
                    except (ValueError, EOFError):
                        print("    跳过此笔")
                        break
                confirmed.append({
                    "action": trade["action"],
                    "code": code,
                    "shares": actual_shares,
                    "price": actual_price,
                })
                amount = actual_shares * actual_price
                print(f"    ✓ {action} {code} ×{actual_shares} @ ¥{actual_price:.2f} = ¥{amount:,.0f}")
                break

            elif resp == "s":
                print(f"    ✗ 跳过 {code}")
                break

            elif resp.isdigit() or (resp.startswith("-") and resp[1:].isdigit()):
                # 修改数量
                actual_shares = abs(int(resp))
                actual_shares = (actual_shares // 100) * 100  # 整手
                if actual_shares <= 0:
                    print(f"    ✗ 跳过 {code}")
                    break
                try:
                    price_input = input(f"    成交价 (回车用参考价¥{ref_price:.2f}): ").strip()
                    actual_price = float(price_input) if price_input else ref_price
                except (ValueError, EOFError):
                    actual_price = ref_price
                if actual_price <= 0:
                    print("    价格无效，跳过")
                    break
                confirmed.append({
                    "action": trade["action"],
                    "code": code,
                    "shares": actual_shares,
                    "price": actual_price,
                })
                amount = actual_shares * actual_price
                print(f"    ✓ {action} {code} ×{actual_shares} @ ¥{actual_price:.2f} = ¥{amount:,.0f}")
                break

            else:
                # 尝试作为价格解析
                try:
                    actual_price = float(resp)
                    if actual_price > 0:
                        confirmed.append({
                            "action": trade["action"],
                            "code": code,
                            "shares": rec_shares,
                            "price": actual_price,
                        })
                        amount = rec_shares * actual_price
                        print(f"    ✓ {action} {code} ×{rec_shares} @ ¥{actual_price:.2f} = ¥{amount:,.0f}")
                        break
                except ValueError:
                    pass
                print("    无效输入。y=确认, s=跳过, 数字=数量或价格")

    # 汇总确认
    if not confirmed:
        print(f"\n没有确认的交易")
        return

    print(f"\n{'─' * 60}")
    print(f"  确认的交易:")
    total_buy = 0
    total_sell = 0
    for trade in confirmed:
        amount = trade["shares"] * trade["price"]
        action = action_names.get(trade["action"], trade["action"])
        if trade["action"] == "buy":
            total_buy += amount
        else:
            total_sell += amount
        print(f"    {action} {trade['code']} ×{trade['shares']} @ ¥{trade['price']:.2f} = ¥{amount:,.0f}")

    net = total_buy - total_sell
    print(f"\n  买入合计: ¥{total_buy:,.0f}")
    print(f"  卖出合计: ¥{total_sell:,.0f}")
    print(f"  净流出:   ¥{net:,.0f}")
    print(f"  当前现金: ¥{ps.cash:,.0f}")
    if ps.cash - net < 0:
        print(f"  ⚠ 操作后现金不足: ¥{ps.cash - net:,.0f}")

    confirm = input(f"\n  确认记录? (y/N): ").strip().lower()
    if confirm == "y":
        ps.update_after_trade(confirmed)
        print("  交易已记录，组合状态已更新")
    else:
        print("  已取消")


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
    confirm_parser.add_argument("-i", "--interactive", action="store_true", help="交互式逐笔确认（推荐）")
    confirm_parser.add_argument("--buy", nargs="*", help="买入 CODE:SHARES[:PRICE]")
    confirm_parser.add_argument("--sell", nargs="*", help="卖出 CODE:SHARES[:PRICE]")
    confirm_parser.add_argument("--from-rec", action="store_true", help="快速确认上次建议")

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
