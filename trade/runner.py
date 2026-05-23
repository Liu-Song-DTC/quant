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


def _last_recommendation_empty(cfg) -> bool:
    """检查上次建议是否为空（无买入且无卖出）"""
    rec_file = cfg.rec_file
    if not rec_file.exists():
        return True  # 从未运行过，视为空
    try:
        with open(rec_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        buys = data.get("buys", [])
        sells = data.get("sells", [])
        return len(buys) == 0 and len(sells) == 0
    except (json.JSONDecodeError, IOError):
        return True


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


def _check_holdings_daily(cfg, ps):
    """每日持仓深度诊断：价格止损 + 缠论结构 + 趋势/MA/背离"""
    codes = list(ps.positions.keys())
    if not codes:
        print("\n当前空仓，无需检查")
        return

    prices = _get_latest_prices(str(cfg.bt_data_dir), codes)
    if not prices:
        print("\n无法获取最新价格，跳过持仓检查")
        return

    print(f"\n{'=' * 50}")
    print(f"  每日持仓诊断 ({datetime.today().strftime('%Y-%m-%d')})")
    print(f"{'=' * 50}")

    # ---- 读取最新缠论分析数据 ----
    chan_data = {}
    chan_csv = ROOT / 'strategy' / 'daily_review' / f'缠论全量分析_{datetime.today().strftime("%Y%m%d")}.csv'
    if chan_csv.exists():
        try:
            df = pd.read_csv(chan_csv, dtype={'股票代码': str})
            for _, row in df.iterrows():
                code = str(row.get('股票代码', ''))
                if code in codes:
                    chan_data[code] = {
                        'trend_type': row.get('trend_type', 0),
                        'buy_point': row.get('buy_point', 0),
                        'sell_point': row.get('sell_point', 0),
                        'confirmed_buy': row.get('confirmed_buy', False),
                        'confirmed_sell': row.get('confirmed_sell', False),
                        'composite_score': row.get('composite_score', 0),
                        'top_divergence': row.get('top_divergence', False),
                        'bottom_divergence': row.get('bottom_divergence', False),
                        'stop_loss_price': row.get('stop_loss_price', 0),
                        'pivot_breakdown': row.get('pivot_breakdown', False),
                        'buy_strength': row.get('buy_strength', 0),
                        'sell_strength': row.get('sell_strength', 0),
                        'close': row.get('收盘', 0),
                        'change_pct': row.get('涨跌幅', 0),
                    }
        except Exception as e:
            print(f"  (读取缠论数据失败: {e})")

    # ---- 逐只诊断 ----
    warnings = []
    for code, pos_info in ps.positions.items():
        shares = pos_info.get('shares', 0) if isinstance(pos_info, dict) else pos_info
        cost = pos_info.get('cost_price', 0) if isinstance(pos_info, dict) else 0
        price = prices.get(code, 0)
        if not price:
            continue

        pnl_pct = (price / cost - 1) * 100 if cost > 0 else 0
        cd = chan_data.get(code, {})

        issues = []
        severity = 'normal'  # normal / warning / danger

        # 1. 价格硬止损 (-12%)
        if pnl_pct <= -12:
            issues.append(f'触及硬止损线 ({pnl_pct:+.1f}%)')
            severity = 'danger'

        # 2. 缠论卖点出现
        if cd.get('confirmed_sell'):
            sp = cd.get('sell_point', 0)
            issues.append(f'缠论卖点已确认 (S{sp})')
            severity = 'danger'
        elif cd.get('sell_point', 0) >= 2:
            issues.append(f'潜在卖点信号 (S{cd["sell_point"]})，需密切关注')
            severity = max(severity, 'warning')

        # 3. 顶背驰
        if cd.get('top_divergence'):
            issues.append('出现顶背驰 — 上涨力竭信号')
            severity = max(severity, 'warning')

        # 4. 趋势反转
        trend = cd.get('trend_type', 0)
        if trend == -2:
            issues.append('趋势已转为下跌')
            severity = max(severity, 'danger')
        elif trend == 0:
            issues.append('趋势不明朗（盘整/无方向）')
            severity = max(severity, 'warning')

        # 5. 中枢破位
        if cd.get('pivot_breakdown'):
            issues.append('跌破中枢下沿 — 结构破坏')
            severity = max(severity, 'danger')

        # 6. 综合评分为负
        score = cd.get('composite_score', 0)
        if score < -0.2:
            issues.append(f'综合评分偏低 ({score:+.2f})')
            severity = max(severity, 'warning')

        # 7. 买点失效
        if cd.get('buy_point', 0) > 0 and not cd.get('confirmed_buy'):
            if cd.get('sell_point', 0) > 0 or trend == -2:
                issues.append('原有买点已被破坏（卖点出现/趋势反转）')
                severity = max(severity, 'danger')

        # 输出诊断
        sev_icon = {'normal': '✓', 'warning': '⚠', 'danger': '🔴'}[severity]
        print(f"\n  {sev_icon} {code} | 成本¥{cost:.2f} → ¥{price:.2f} ({pnl_pct:+.1f}%)")

        if cd:
            trend_label = {2: '上涨', 1: '盘整', 0: '-', -2: '下跌'}.get(trend, '?')
            bp = cd.get('buy_point', 0)
            sp = cd.get('sell_point', 0)
            print(f"    走势:{trend_label} | 买点:B{bp if bp>0 else '-'} | "
                  f"卖点:S{sp if sp>0 else '-'} | 评分:{score:+.2f} | "
                  f"止损价:{cd.get('stop_loss_price', 0):.2f}")

        if issues:
            for issue in issues:
                print(f"    → {issue}")
            warnings.append({'code': code, 'severity': severity, 'issues': issues,
                           'cost': cost, 'price': price, 'pnl_pct': pnl_pct})
        else:
            print(f"    结构健康，继续持有")

    # ---- 汇总 ----
    if warnings:
        danger = [w for w in warnings if w['severity'] == 'danger']
        warn = [w for w in warnings if w['severity'] == 'warning']
        print(f"\n{'=' * 50}")
        print(f"  诊断汇总: 🔴 {len(danger)} 只危险, ⚠ {len(warn)} 只警告")
        if danger:
            print(f"  建议立即处理: {', '.join(w['code'] for w in danger)}")
        if warn:
            print(f"  需要密切关注: {', '.join(w['code'] for w in warn)}")
        print(f"{'=' * 50}")

        # 通知
        if cfg.notification_enabled and cfg.notification_sckey:
            from .notifier import Notifier
            lines = [f"**持仓诊断 ({datetime.today().strftime('%Y-%m-%d')})**", ""]
            for w in warnings:
                icon = '🔴' if w['severity'] == 'danger' else '⚠'
                lines.append(f"{icon} {w['code']}: {', '.join(w['issues'])}")
            Notifier(cfg.notification_sckey).send("持仓诊断警告", "\n".join(lines))
    else:
        print(f"\n  全部持仓结构健康 ✓")

    # ---- 调仓替换建议 ----
    danger_codes = {w['code'] for w in warnings if w['severity'] == 'danger'}
    warn_codes = {w['code'] for w in warnings if w['severity'] == 'warning'}
    replace_candidates = danger_codes | warn_codes

    if replace_candidates:
        today_str = datetime.today().strftime('%Y-%m-%d')
        sel_csv = ROOT / 'output' / '选股历史.csv'
        selections = []
        if sel_csv.exists():
            try:
                df = pd.read_csv(sel_csv, dtype={'股票代码': str})
                today_sel = df[df['选股日期'] == today_str]
                selections = today_sel.to_dict('records')
            except Exception:
                pass

        if selections:
            hold_codes = {p['code'] for p in warnings} | set(ps.positions.keys())
            # 候补: 不在当前持仓中的选股, 按评分降序
            available = [s for s in selections
                        if str(s.get('股票代码', '')) not in hold_codes]
            available.sort(key=lambda x: float(x.get('综合评分', 0)), reverse=True)

            print(f"\n{'=' * 50}")
            print(f"  调仓替换建议")
            print(f"{'=' * 50}")

            for code in sorted(danger_codes):
                if not available:
                    break
                replacement = available.pop(0)
                r_code = replacement.get('股票代码', '')
                r_name = replacement.get('股票名称', '')
                r_score = float(replacement.get('综合评分', 0))
                r_price = float(replacement.get('当前价格', 0))
                print(f"\n  🔴 {code} → {r_code} {r_name}")
                print(f"     卖出原因: 结构恶化(趋势走坏/卖点/破位)")
                print(f"     替换标的: 评分{r_score:.3f}, 现价{r_price:.2f}, "
                      f"信号:{replacement.get('信号强度', '')}")

            for code in sorted(warn_codes):
                if not available:
                    break
                replacement = available.pop(0)
                r_code = replacement.get('股票代码', '')
                r_name = replacement.get('股票名称', '')
                r_score = float(replacement.get('综合评分', 0))
                print(f"\n  ⚠ {code} → {r_code} {r_name} (预备)")
                print(f"     关注: 暂不卖出，若明日继续恶化则替换")
                print(f"     备选标的: 评分{r_score:.3f}")

            print(f"\n  替换后持仓建议: 卖出{len(danger_codes)}只, 买入{len(danger_codes)}只")
            if not danger_codes:
                print(f"  当前无危险持仓，警告标的继续观察")
        else:
            today_str = datetime.today().strftime('%Y-%m-%d')
            print(f"\n  (当日选股数据未生成, 无法提供替换建议)")
            print(f"  请先运行: python export_selection.py")


def run_daily(skip_update: bool = False, force: bool = False):
    cfg = TradeConfig()

    # 判断是否调仓日
    today = datetime.today()
    rebal = cfg.rebalance_info(today)
    print(f"日期: {today.strftime('%Y-%m-%d')}  调仓日: {'是' if rebal['is_rebalance_day'] else '否'}  "
          f"上次: {rebal['last_rebalance']}  下次: {rebal['next_rebalance']}")

    ps = PortfolioState.load(str(cfg.state_file))

    if not rebal['is_rebalance_day'] and not force:
        # 非调仓日: 检查上次建议是否为空（整手取整导致0股等）
        last_empty = _last_recommendation_empty(cfg)
        if last_empty:
            print("\n上次建议为空（买入/卖出均为0），自动重新生成...")
        else:
            _check_holdings_daily(cfg, ps)
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

    # Step 1b: 行业情绪分析（在数据更新之后，信号生成之前）
    sentiment_multipliers = {}
    if not skip_update and cfg.industry_sentiment_enabled:
        print("\n" + "=" * 50)
        print("Step 1b: LLM 行业情绪分析")
        print("=" * 50)
        try:
            _add_path("strategy")
            from sentiment.orchestrator import SentimentOrchestrator
            from core.config_loader import load_config

            config = load_config()
            orchestrator = SentimentOrchestrator(config)
            scores = orchestrator.run_daily(notify=cfg.notification_enabled)

            if scores:
                sentiment_multipliers = orchestrator.get_sentiment_weights()
                print(f"情绪乘数已计算: {len(sentiment_multipliers)} 个行业")

            # 发送微信情绪摘要
            if scores and cfg.notification_enabled and cfg.notification_sckey:
                from .notifier import Notifier
                Notifier(cfg.notification_sckey).send_industry_sentiment(
                    today.strftime("%Y-%m-%d"), scores
                )
        except ImportError as e:
            print(f"情绪分析模块导入失败 (非关键): {e}")
        except Exception as e:
            print(f"情绪分析失败 (非关键): {e}")
            import traceback
            traceback.print_exc()

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
    internal = ps.load_internal()
    runner.prepare(exposure=internal["exposure"], peak_equity=internal["peak_equity"])

    # 注入行业情绪乘数
    if sentiment_multipliers:
        runner.set_sentiment_multipliers(sentiment_multipliers)

    result = runner.run(
        current_positions=ps.get_current_positions(prices),
        cash=ps.cash,
        cost=ps.get_cost_basis(),
    )

    # 持久化内部状态(exposure/peak_equity → .internal.json, 用户不可见)
    if hasattr(runner.strategy, 'portfolio'):
        ps.save_internal(
            runner.strategy.portfolio.current_exposure,
            runner.strategy.portfolio.peak_equity or 0.0,
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
