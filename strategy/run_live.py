#!/usr/bin/env python3
"""
实盘选股一键执行:
    1. 更新行情数据 (data_manager.py)
    2. 生成信号 (bt_execution.py)
    3. 选股 → trade_orders.json (generate_trade_orders.py)

用法:
    .venv/bin/python strategy/run_live.py [--date 2026-07-16] [--cash 300000]

不带 --date 时默认使用今天。
"""

import os, sys, subprocess, argparse, shutil
from datetime import date as date_type, datetime
import yaml

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
_VENV_PYTHON = os.path.join(_PROJECT_DIR, '.venv', 'bin', 'python')

CONFIG_PATH = os.path.join(_SCRIPT_DIR, 'config', 'factor_config.yaml')
CONFIG_BAK = CONFIG_PATH + '.live_bak'


def update_market_data(target_date: str):
    """Step 1: 增量更新行情数据."""
    print("=" * 60)
    print("Step 1/3: 更新行情数据")
    print("=" * 60)
    result = subprocess.run(
        [_VENV_PYTHON, os.path.join(_PROJECT_DIR, 'data', 'data_manager.py')],
        cwd=_PROJECT_DIR,
        capture_output=False,
    )
    if result.returncode != 0:
        print("[WARN] 数据更新失败, 继续使用本地缓存...")


def patch_config_todate(target_date: str):
    """临时修改 factor_config.yaml 的 todate."""
    if os.path.exists(CONFIG_BAK):
        return  # 已经 patch 过了

    shutil.copy2(CONFIG_PATH, CONFIG_BAK)

    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        content = f.read()

    import re
    content = re.sub(r"todate:\s*['\"].*?['\"]", f"todate: '{target_date}'", content)

    with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"  配置 todate 已更新为: {target_date}")


def restore_config():
    """恢复原始配置."""
    if os.path.exists(CONFIG_BAK):
        shutil.move(CONFIG_BAK, CONFIG_PATH)
        print("  配置已恢复")


def generate_signals():
    """Step 2: 运行 bt_execution.py 生成信号."""
    print("=" * 60)
    print("Step 2/3: 生成信号 (bt_execution.py)")
    print("=" * 60)
    result = subprocess.run(
        [_VENV_PYTHON, os.path.join(_SCRIPT_DIR, 'bt_execution.py')],
        cwd=_SCRIPT_DIR,
        capture_output=False,
    )
    if result.returncode != 0:
        print("[ERROR] 信号生成失败")
        return False
    return True


def generate_orders(target_date: str, cash: float, dry_run: bool = False):
    """Step 3: 选股 → trade_orders.json."""
    print("=" * 60)
    print("Step 3/3: 选股 → trade_orders.json")
    print("=" * 60)

    # 直接导入 generate_trade_orders 的逻辑
    sys.path.insert(0, _SCRIPT_DIR)
    from generate_trade_orders import (
        build_stock_file_map, load_prices_for_date, build_orders,
        get_index_regime, load_current_positions, save_current_positions,
    )
    from core.config_loader import load_config
    from core.signal_store import SignalStore
    from core.portfolio import PortfolioConstructor

    config = load_config()
    SIGNALS_CSV = os.path.join(_SCRIPT_DIR, 'rolling_validation_results', 'backtest_signals.csv')
    ORDERS_FILE = os.path.join(_PROJECT_DIR, 'trade_orders.json')
    POSITIONS_FILE = os.path.join(_PROJECT_DIR, 'current_positions.json')

    import pandas as pd

    # ── 加载信号 ──────────────────────────────────────────
    stock_file_map = build_stock_file_map()
    print(f"股票文件: {len(stock_file_map)} 只")

    if not os.path.exists(SIGNALS_CSV):
        print("错误: 信号文件不存在")
        return

    signal_store = SignalStore()
    signal_store.finalize(SIGNALS_CSV)

    test_code = next((c for c in stock_file_map if c != 'sh000001'), None)
    if test_code and signal_store.get(test_code, target_date) is None:
        # 回退到信号中最新日期
        sig_df = pd.read_csv(SIGNALS_CSV, usecols=['date'])
        latest_sig_date = pd.Timestamp(sig_df['date'].max()).date()
        del sig_df
        print(f"  目标日期 {target_date} 无信号, 回退到 {latest_sig_date}")
        target_date = latest_sig_date

    # ── 获取价格 ──────────────────────────────────────────
    prices = load_prices_for_date(stock_file_map, target_date)

    # ── 构建 universe ─────────────────────────────────────
    stock_pool_enabled = config.get('stock_pool.enabled', True)
    allowed_codes = None
    if stock_pool_enabled:
        from core.stock_pool import get_stock_pool
        allowed_codes = get_stock_pool()

    universe = []
    for code in stock_file_map:
        if code == 'sh000001':
            continue
        if code not in prices or prices[code] <= 0:
            continue
        if prices[code] < 2.0:
            continue
        if allowed_codes is not None and code not in allowed_codes:
            continue
        universe.append(code)
    print(f"可交易股票: {len(universe)} 只")

    # ── 市场状态 ──────────────────────────────────────────
    market_regime, momentum_score, trend_score, bear_risk, bear_risk_fast = get_index_regime(target_date)
    regime_names = {1: '牛市', 0: '中性', -1: '熊市'}
    print(f"市场状态: {regime_names.get(market_regime, '未知')}")

    # ── 组合构建 ──────────────────────────────────────────
    portfolio = PortfolioConstructor()
    prev_positions = load_current_positions()
    current_positions = {code: info['amount'] for code, info in prev_positions.items()}

    cost = {}
    for code, info in prev_positions.items():
        cost[code] = [info['entry_price'], info['entry_price']]

    total_equity = cash + sum(current_positions.values())
    cash_available = max(cash - sum(info['amount'] for info in prev_positions.values()), 0)

    adjusted = portfolio.build(
        date=target_date,
        universe=universe,
        current_positions=current_positions,
        signal_store=signal_store,
        cash=cash_available,
        prices=prices,
        market_regime=market_regime,
        momentum_score=momentum_score,
        trend_score=trend_score,
        bear_risk=bear_risk,
        bear_risk_fast=bear_risk_fast,
        cost=cost,
    )

    # ── 生成订单 ──────────────────────────────────────────
    orders, new_positions, order_details = build_orders(
        str(target_date), adjusted, prices, prev_positions, total_equity,
        signal_store, stock_file_map,
    )

    output = {'date': str(target_date), 'orders': orders}

    print(f"\n{'='*60}")
    print(f"订单: {len(orders)} 条")
    for o in orders:
        if o['action'] == 'close':
            print(f"  {o['action']:8s} {o['stock_code']}")
        else:
            print(f"  {o['action']:8s} {o['stock_code']} "
                  f"金额={o['amount']:>8,} "
                  f"止损={o.get('stop_loss_price', 'N/A')} "
                  f"止盈={o.get('take_profit_price', 'N/A')}")

    # ── 支撑/压力明细 ─────────────────────────────────────
    if order_details:
        print(f"\n止盈止损计算明细:")
        print(f"  {'代码':>12s} {'现价':>8s} {'止损(支撑)':>10s} {'来源':>12s} {'止盈(压力)':>10s} {'来源':>12s}")
        print(f"  {'-'*70}")
        for code, action, d in order_details:
            sl_price = next((o['stop_loss_price'] for o in orders if o.get('stock_code') == code and 'stop_loss_price' in o), 0)
            tp_price = next((o['take_profit_price'] for o in orders if o.get('stock_code') == code and 'take_profit_price' in o), 0)
            print(f"  {code:>12s} {d['price']:>8.2f} {sl_price:>10.2f} {d.get('support_reason',''):>12s} {tp_price:>10.2f} {d.get('resistance_reason',''):>12s}")

    if dry_run:
        import json
        print("\n[Dry-run] 不写入文件")
        print(json.dumps(output, ensure_ascii=False, indent=2))
        return

    # ── 写入文件 ──────────────────────────────────────────
    import json
    with open(ORDERS_FILE, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n订单已写入: {ORDERS_FILE}")

    from generate_trade_orders import save_current_positions as _save_pos
    _save_pos(new_positions)

    # ── 选股详情 ──────────────────────────────────────────
    if hasattr(portfolio, 'last_selection') and portfolio.last_selection:
        print(f"\n选股详情:")
        for sel in portfolio.last_selection:
            print(f"  {sel['code']:>8s}  score={sel['score']:.4f}  "
                  f"weight={sel.get('weight', 0):.3f}  "
                  f"industry={sel.get('industry', '')}")


def main():
    parser = argparse.ArgumentParser(description='实盘选股一键执行')
    parser.add_argument('--date', type=str, default=None,
                        help='目标日期 (默认: 今天)')
    parser.add_argument('--cash', type=float, default=300000.0,
                        help='总资金 (默认: 300000)')
    parser.add_argument('--skip-data', action='store_true',
                        help='跳过数据更新')
    parser.add_argument('--skip-signals', action='store_true',
                        help='跳过信号生成 (用已有信号)')
    parser.add_argument('--dry-run', action='store_true',
                        help='仅预览, 不写入文件')
    args = parser.parse_args()

    # ── 确定日期 ──────────────────────────────────────────
    if args.date:
        target_date = args.date
    else:
        target_date = date_type.today().strftime('%Y-%m-%d')
    print(f"实盘选股: {target_date} | 资金: {args.cash:,.0f}")

    # ── Step 1: 更新数据 ──────────────────────────────────
    if not args.skip_data:
        update_market_data(target_date)
    else:
        print("跳过数据更新")

    # ── Step 2: 生成信号 ──────────────────────────────────
    if not args.skip_signals:
        try:
            patch_config_todate(target_date)
            success = generate_signals()
            if not success:
                print("[ERROR] 信号生成失败, 退出")
                sys.exit(1)
        finally:
            restore_config()
    else:
        print("跳过信号生成 (使用已有信号)")

    # ── Step 3: 选股 + 输出 JSON ──────────────────────────
    generate_orders(target_date, args.cash, dry_run=args.dry_run)

    print(f"\n完成!")


if __name__ == '__main__':
    main()
