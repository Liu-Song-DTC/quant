"""每日运行流程: 数据更新 → 信号生成 → 交易建议"""
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

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


def _get_latest_prices(bt_data_dir: str, codes: list = None) -> dict:
    """轻量获取最新价格，不加载策略引擎"""
    prices = {}
    if not os.path.exists(bt_data_dir):
        return prices
    for item in os.listdir(bt_data_dir):
        if not item.endswith('_qfq.csv'):
            continue
        name = item[:-8]
        if codes and name not in codes:
            continue
        try:
            df = pd.read_csv(os.path.join(bt_data_dir, item))
            if len(df) > 0 and 'close' in df.columns:
                prices[name] = float(df['close'].iloc[-1])
        except Exception:
            pass
    return prices


def run_daily(skip_update: bool = False, force: bool = False):
    cfg = TradeConfig()

    # 判断是否调仓日
    today = datetime.today()
    rebal = cfg.rebalance_info(today)
    print(f"日期: {today.strftime('%Y-%m-%d')}  调仓日: {'是' if rebal['is_rebalance_day'] else '否'}  "
          f"上次: {rebal['last_rebalance']}  下次: {rebal['next_rebalance']}")

    ps = PortfolioState.load(str(cfg.state_file))

    if not rebal['is_rebalance_day'] and not force:
        # 非调仓日: 轻量止损检查
        prices = _get_latest_prices(str(cfg.bt_data_dir), list(ps.positions.keys()))
        if not prices:
            print("\n今日非调仓日，无需生成建议")
            print("如需强制运行: python main.py run --force")
            return

        triggered = ps.check_stop_loss(prices)
        if triggered:
            print(f"\n⚠ 止损警告: {len(triggered)} 只股票触及止损线")
            for t in triggered:
                print(f"  {t['code']}: 成本¥{t['cost_price']:.2f} → ¥{t['current_price']:.2f} ({t['pnl_pct']:.1%})")
            if cfg.notification_enabled and cfg.notification_sckey:
                from .notifier import Notifier
                lines = ["**⚠ 止损警告**", ""]
                for t in triggered:
                    lines.append(f"- {t['code']}: 成本¥{t['cost_price']:.2f} → ¥{t['current_price']:.2f} ({t['pnl_pct']:.1%})")
                Notifier(cfg.notification_sckey).send("⚠ 止损警告", "\n".join(lines))
        else:
            print("\n今日非调仓日，持仓正常，无需操作")
        return

    # Step 1: 数据更新
    if not skip_update:
        print("\n" + "=" * 50)
        print("Step 1: 更新数据")
        print("=" * 50)
        _add_path("data")
        from data_manager import StockDataManager

        manager = StockDataManager(data_dir=str(cfg.stock_data_dir))
        manager.get_stock_list(force_update=False)

        symbols = _get_universe(manager)
        print(f"增量更新 {len(symbols)} 只股票行情...")
        manager.batch_download(symbols=symbols, force=False)

        end_date = today.strftime("%Y-%m-%d")
        start_date = (today - timedelta(days=730)).strftime("%Y-%m-%d")
        manager.create_backtrader_data(symbols, start_date, end_date, adj_type="qfq")

        manager.incremental_update_fundamental()
        print("数据更新完成")
    else:
        print("跳过数据更新 (--skip-update)")

    # Step 2: 信号 + 建议
    print("\n" + "=" * 50)
    print("Step 2: 生成交易建议")
    print("=" * 50)

    # Step 2a: 先加载数据取价格
    _add_path("strategy")
    runner = SignalRunner(
        bt_data_dir=str(cfg.bt_data_dir),
        fund_data_dir=str(cfg.fund_data_dir),
    )
    prices = runner.get_prices()

    # 总资产 = 现金 + 持仓市值
    positions_value = ps.get_current_positions(prices)
    total_asset = ps.cash + sum(positions_value.values())
    print(f"现金: ¥{ps.cash:,.0f}  持仓: ¥{sum(positions_value.values()):,.0f}  "
          f"总资产: ¥{total_asset:,.0f}")

    # Step 2b: 生成信号 (max_position由PortfolioConstructor自动计算)
    runner.prepare(exposure=ps.exposure, peak_equity=ps.peak_equity)

    result = runner.run(
        current_positions=ps.get_current_positions(prices),
        cash=ps.cash,
        cost=ps.get_cost_basis(),
    )

    # 持久化组合状态(exposure/peak_equity)
    if hasattr(runner.strategy, 'portfolio'):
        ps.exposure = runner.strategy.portfolio.current_exposure
        ps.peak_equity = runner.strategy.portfolio.peak_equity or 0.0
        ps.save()

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
    recommendations["date"] = today.strftime("%Y-%m-%d")
    recommendations["rebalance_info"] = rebal
    cfg.rec_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cfg.rec_file, "w", encoding="utf-8") as f:
        json.dump(recommendations, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n建议已保存: {cfg.rec_file}")
    if recommendations.get("buys") or recommendations.get("sells"):
        print(f"执行后请手动更新: trade/portfolio_state.json")

    # 微信推送
    if cfg.notification_enabled and cfg.notification_sckey:
        from .notifier import Notifier
        Notifier(cfg.notification_sckey).send_recommendations(recommendations)


def show_status():
    """查看组合状态 — 轻量版，不加载策略引擎"""
    cfg = TradeConfig()
    ps = PortfolioState.load(str(cfg.state_file))

    # 轻量获取现价
    prices = _get_latest_prices(str(cfg.bt_data_dir), list(ps.positions.keys())) if ps.positions else {}

    rebal = cfg.rebalance_info(datetime.today())
    summary = ps.summary(prices)
    print(f"\n{'=' * 50}")
    print(f"  组合状态")
    print(f"{'=' * 50}")
    print(f"  今日:     {datetime.today().strftime('%Y-%m-%d')}")
    print(f"  调仓日:   {'是' if rebal['is_rebalance_day'] else '否'}  "
          f"上次: {rebal['last_rebalance']}  下次: {rebal['next_rebalance']}")
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
