"""每日运行流程: 数据更新 → 信号生成 → 交易建议"""
import json
import sys
from pathlib import Path

from .config import TradeConfig
from .portfolio_state import PortfolioState
from .signal_runner import SignalRunner
from .recommender import Recommender
from .reporter import Reporter

ROOT = Path(__file__).parent.parent.resolve()


def _add_path(name: str):
    p = str(ROOT / name)
    if p not in sys.path:
        sys.path.insert(0, p)


def _get_universe(manager) -> list:
    symbols = []
    bt_dir = manager.backtrader_data_dir
    if bt_dir.exists():
        for f in bt_dir.iterdir():
            if f.name.endswith("_qfq.csv"):
                symbols.append(f.name.replace("_qfq.csv", ""))
    if not symbols:
        stock_list = manager.get_stock_list()
        if not stock_list.empty:
            symbols = stock_list["symbol"].tolist()
            symbols.insert(0, "sh000001")
    return symbols


def run_daily(skip_update: bool = False, update_fundamental: bool = False):
    cfg = TradeConfig()

    # Step 1: 数据更新
    if not skip_update:
        print("=" * 50)
        print("Step 1: 更新数据")
        print("=" * 50)
        _add_path("data")
        from data_manager import StockDataManager
        from datetime import datetime, timedelta

        manager = StockDataManager(data_dir=str(cfg.stock_data_dir))
        manager.get_stock_list(force_update=True)

        symbols = _get_universe(manager)
        print(f"增量更新 {len(symbols)} 只股票行情...")
        manager.batch_download(symbols=symbols, force=False)

        end_date = datetime.today().strftime("%Y-%m-%d")
        start_date = (datetime.today() - timedelta(days=730)).strftime("%Y-%m-%d")
        manager.create_backtrader_data(symbols, start_date, end_date, adj_type="qfq")

        if update_fundamental:
            print("更新基本面数据...")
            manager.incremental_update_fundamental()
        print("数据更新完成")
    else:
        print("跳过数据更新 (--skip-update)")

    # Step 2: 信号 + 建议
    print("\n" + "=" * 50)
    print("Step 2: 生成交易建议")
    print("=" * 50)

    _add_path("strategy")
    ps = PortfolioState.load(str(cfg.state_file))
    if ps.cash == 0 and not ps.data.get("positions"):
        ps.init_cash(cfg.init_cash)
        print(f"初始化组合: ¥{cfg.init_cash:,.0f}")

    runner = SignalRunner(
        bt_data_dir=str(cfg.bt_data_dir),
        fund_data_dir=str(cfg.fund_data_dir),
        max_position=cfg.max_position,
    )
    result = runner.run(
        current_positions=ps.get_current_positions(runner.get_prices()),
        cash=ps.cash,
        cost=ps.get_cost_basis(),
    )

    if result is None:
        print("信号生成失败，请检查数据")
        return

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

    reporter = Reporter(report_dir=str(cfg.report_dir))
    reporter.print_report(recommendations, ps.summary(result["prices"]))
    reporter.save_report(recommendations, ps.summary(result["prices"]))

    # 保存建议
    cfg.rec_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cfg.rec_file, "w", encoding="utf-8") as f:
        json.dump(recommendations, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n建议已保存: {cfg.rec_file}")
    if recommendations.get("buys") or recommendations.get("sells"):
        print(f"交易完成后请运行: python main.py confirm -i")

    # 微信推送
    if cfg.notification_enabled and cfg.notification_sckey:
        from .notifier import Notifier
        Notifier(cfg.notification_sckey).send_recommendations(recommendations)


def show_status():
    _add_path("strategy")
    ps = PortfolioState.load(str(TradeConfig().state_file))

    prices = {}
    if ps.data.get("positions"):
        cfg = TradeConfig()
        runner = SignalRunner(bt_data_dir=str(cfg.bt_data_dir), fund_data_dir="", max_position=10)
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
            print(f"  {p['code']:<8} {p['shares']:>6} {p['cost_price']:>10.2f} "
                  f"{p['current_price']:>10.2f} {p['market_value']:>12,.2f} {p['pnl_pct']:+.2%}")
    print(f"\n{'=' * 50}")
