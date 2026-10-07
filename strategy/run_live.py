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


def update_kline_data():
    """更新K线行情数据 (data_manager.py: 下载raw + 转换backtrader)."""
    print("=" * 60)
    print("Step 1a: 更新K线行情数据")
    print("=" * 60)
    result = subprocess.run(
        [_VENV_PYTHON, os.path.join(_PROJECT_DIR, 'data', 'data_manager.py')],
        cwd=_PROJECT_DIR,
        capture_output=False,
    )
    if result.returncode != 0:
        print("[WARN] K线数据更新失败, 继续使用本地缓存...")


def update_macro_data():
    """更新宏观数据 + 另类数据 (akshare, 积分消耗极低)."""
    print("=" * 60)
    print("Step 1b: 更新宏观/另类数据")
    print("=" * 60)
    try:
        macro_script = os.path.join(_SCRIPT_DIR, 'core', 'macro_data.py')
        result = subprocess.run(
            [_VENV_PYTHON, macro_script, 'download'],
            cwd=_SCRIPT_DIR, capture_output=True, text=True, timeout=120
        )
        if result.returncode != 0:
            print(f"  [WARN] 宏观数据更新失败: {result.stderr[-200:]}")
        else:
            print(f"  宏观数据更新完成")
    except Exception as e:
        print(f"  [WARN] 宏观数据更新异常: {e}")


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

    # 实盘必须重新生成: 删除旧信号文件, 强制 bt_execution 重新计算
    _signals_csv = os.path.join(_SCRIPT_DIR, 'rolling_validation_results', 'backtest_signals.csv')
    if os.path.exists(_signals_csv):
        os.remove(_signals_csv)
        print(f"  已删除旧信号文件, 强制重新生成")

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
        build_monitor_list, print_monitor_list, build_st_filter,
        compute_cb_sleeve, cb_load_state, cb_load_snapshot, cb_save_state,
        compute_repo_sweep, sweep_guidance,
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
        latest_sig_date = pd.Timestamp(sig_df['date'].max()).strftime('%Y-%m-%d')
        del sig_df
        print(f"  目标日期 {target_date} 无信号, 回退到 {latest_sig_date}")
        target_date = latest_sig_date

    # ── 获取价格 ──────────────────────────────────────────
    prices = load_prices_for_date(stock_file_map, target_date)

    # ── 构建 universe ─────────────────────────────────────
    stock_pool_enabled = config.get('stock_pool.enabled', True)
    allowed_codes = None
    if stock_pool_enabled:
        from core.stock_pool import get_stock_pool, _quarter_boundary_prev, _month_boundary_prev, get_pool_membership_map
        pool_calendar = config.get('stock_pool.pool_calendar', 'off')
        if pool_calendar in ('quarterly', 'monthly', 'daily'):
            # 0f日历池: quarterly=target_date最近季度末 / monthly=最近月末 /
            # daily=target_date当日 的as-of成员 ∩ 最新流动性池(实盘安全滤镜)。
            # 与回测信号membership闸同构。
            if pool_calendar == 'daily':
                _b = pd.Timestamp(target_date)
            elif pool_calendar == 'monthly':
                _b = _month_boundary_prev(pd.Timestamp(target_date))
            else:
                _b = _quarter_boundary_prev(pd.Timestamp(target_date))
            # 0f-v3松弛参数: 日历membership与安全滤镜用同一relax规则 (回测同构)
            _rf = config.get('stock_pool.pool_relax_floor', 1.0)
            _rm = config.get('stock_pool.pool_relax_momentum', None)
            _m = get_pool_membership_map([_b], relax_floor=_rf, relax_momentum=_rm)
            _cal = _m[_b]
            _today = get_stock_pool(relax_floor=_rf, relax_momentum=_rm)
            _dropped = _cal - _today
            allowed_codes = _cal & _today
            if _dropped:
                print(f"  [安全滤镜] {len(_dropped)} 只日历成员近期失流动性, "
                      f"排除买单: {sorted(_dropped)[:15]}")
            print(f"股票池(日历{_b.date()}+安全滤镜): {len(_cal)} ∩ {len(_today)} "
                  f"= {len(allowed_codes)} 只")
        else:
            allowed_codes = get_stock_pool()
            print(f"股票池: {len(allowed_codes)} 只 (含指数)")

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

    # ST过滤 + 基本面新鲜度检查 (2026-08-31修复): 出单路径独立校验, 不依赖回测信号CSV
    st_codes, stale_codes = build_st_filter(universe, target_date)
    if st_codes:
        print(f"  [ST排除] {len(st_codes)} 只: {sorted(st_codes)}")
        universe = [c for c in universe if c not in st_codes]
    if stale_codes:
        print(f"  [WARN] {len(stale_codes)} 只基本面数据陈旧>120天, ST判定可能失效")

    # 实盘账户约束(2026-09-14用户确认): 无科创板交易权限, 候选universe剔除688/689.
    _no_star_codes = [c for c in universe if c.startswith(('688', '689'))]
    if _no_star_codes:
        universe = [c for c in universe if c not in _no_star_codes]
        print(f"  [实盘约束] 剔除科创板候选 {len(_no_star_codes)} 只 (无交易权限)")

    print(f"可交易股票: {len(universe)} 只")

    # ── 市场状态 ──────────────────────────────────────────
    market_regime, momentum_score, trend_score, bear_risk, bear_risk_fast = get_index_regime(target_date)
    regime_names = {1: '牛市', 0: '中性', -1: '熊市'}
    print(f"市场状态: {regime_names.get(market_regime, '未知')}")

    # ── 组合构建 ──────────────────────────────────────────
    portfolio = PortfolioConstructor(rebalance_interval=1)  # 实盘每日选股
    prev_positions = load_current_positions()

    # 实盘账户约束(2026-09-14用户确认): 无科创板交易权限, 688/689前缀不可买卖.
    # 持仓文件里此类代码=历史幻影单(从未成交), 剔除并释放资金重新部署.
    _no_star_pos = [c for c in prev_positions if c.startswith(('688', '689'))]
    if _no_star_pos:
        print(f"  [实盘约束] 剔除无权限持仓 {len(_no_star_pos)} 只 (科创板, 假定从未成交): "
              f"{sorted(_no_star_pos)}")
        for c in _no_star_pos:
            prev_positions.pop(c)

    # 账户资金=总可部署资金; 实盘持仓市值含浮盈可合理略超名义资金,
    # 只有显著超出(>1.5倍)才判脏数据重置 — 修复(2026-08-30 review P0):
    # 旧阈值1.0倍导致8/28实盘误触发(305,420>300,000), 清空真实持仓重选
    account_capital = cash
    _pos_sum = sum(info['amount'] for info in prev_positions.values())
    if _pos_sum > account_capital * 1.5:
        print(f"  [WARN] 持仓市值{_pos_sum:,.0f} > 1.5×账户资金{account_capital:,.0f}, "
              f"判定为脏数据, 重置为无持仓按满仓资金{account_capital:,.0f}重新选股")
        prev_positions = {}
        _pos_sum = 0.0
    current_positions = {code: info['amount'] for code, info in prev_positions.items()}

    cost = {}
    for code, info in prev_positions.items():
        cost[code] = [info['entry_price'], info['entry_price']]

    # 排仓尺度=账户资金(而非 现金+持仓 双重放大); 可用现金=账户-已占用
    total_equity = account_capital
    cash_available = max(account_capital - _pos_sum, 0)

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

    # 持仓中的ST强制清仓 (2026-08-31修复): 从目标中移除 → build_orders生成close
    held_st = [c for c in current_positions if c in st_codes]
    if held_st:
        print(f"  [ST强制卖出] 持仓含ST: {held_st}")
        for c in held_st:
            adjusted.pop(c, None)
    held_stale = [c for c in current_positions if c in stale_codes]
    if held_stale:
        print(f"  [WARN] 持仓基本面数据陈旧, ST状态无法验证: {held_stale}")

    # ── 生成订单 ──────────────────────────────────────────
    orders, new_positions, order_details = build_orders(
        str(target_date), adjusted, prices, prev_positions, total_equity,
        signal_store, stock_file_map,
    )

    # ── 0h 现金闸 (2026-09-17, 与generate_trade_orders同款): 买单总额 ≤
    #    可用现金 + 同日卖出回款 — 只限实盘出单路径, 回测不动 ──
    _sell_proceeds = 0.0
    _buy_total = 0.0
    for o in orders:
        _code = o['stock_code'].split('.')[0].zfill(6)
        _prev_amt = prev_positions.get(_code, {}).get('amount', 0)
        if o['action'] == 'close':
            _sell_proceeds += _prev_amt
        elif o['action'] == 'open':
            _buy_total += o['amount']
        elif o['action'] == 'adjust':
            _delta = o['amount'] - _prev_amt
            if _delta > 0:
                _buy_total += _delta
            else:
                _sell_proceeds += -_delta
    _avail = max(account_capital - _pos_sum, 0.0) + _sell_proceeds
    if _buy_total > _avail + 1:
        _scale = (_avail / _buy_total) if _buy_total > 0 else 0.0
        print(f"\n[现金闸] 买单总额{_buy_total:,.0f} > 可用资金{_avail:,.0f} "
              f"(现金{max(account_capital - _pos_sum, 0):,.0f} + 卖出回款{_sell_proceeds:,.0f})")
        _drop = []
        for o in orders:
            if o['action'] not in ('open', 'adjust'):
                continue
            _code = o['stock_code'].split('.')[0].zfill(6)
            _prev_amt = prev_positions.get(_code, {}).get('amount', 0)
            if o['action'] == 'adjust' and o['amount'] <= _prev_amt:
                continue  # 净卖出单不受现金闸约束
            _price = prices.get(_code, 0)
            if _price <= 0:
                continue
            _orig = o['amount']
            if o['action'] == 'open':
                _new_amt = o['amount'] * _scale
                _prev_amt = 0
            else:
                _new_amt = _prev_amt + (o['amount'] - _prev_amt) * _scale
            _shares = int(_new_amt / _price / 100) * 100
            if _shares < 100 or _new_amt <= _prev_amt:
                print(f"  [现金闸] {o['stock_code']} 缩减后不足1手/无净买入, 取消该单")
                _drop.append(o)
                if o['action'] == 'open':
                    new_positions.pop(_code, None)
                else:
                    new_positions[_code] = prev_positions[_code]
                continue
            _amt = int(_shares * _price)
            o['amount'] = _amt
            new_positions[_code]['amount'] = _amt
            new_positions[_code]['shares'] = _shares
            print(f"  [现金闸] {o['stock_code']} {_amt:>8,} ({_shares}股×{_price:.2f}, 原{_orig:,})")
        for o in _drop:
            orders.remove(o)

    # ── 逆回购/转债 sleeve建议块 (2026-10-07接线, 与generate_trade_orders同口径:
    #    转债先出资(闲置现金alloc_ratio), sweep扫余量留缓冲; 纯估算, QMT侧以实际余额为准) ──
    _buy_final = 0.0
    for o in orders:
        _code = o['stock_code'].split('.')[0].zfill(6)
        _prev_amt = prev_positions.get(_code, {}).get('amount', 0)
        if o['action'] == 'open':
            _buy_final += o['amount']
        elif o['action'] == 'adjust':
            _delta = o['amount'] - _prev_amt
            if _delta > 0:
                _buy_final += _delta
    _post_order_cash = max(account_capital - _pos_sum, 0.0) + _sell_proceeds - _buy_final

    _cb_cfg = config.get('live_monitoring.cb_sleeve', {}) or {}
    _cb_result = None
    if (_cb_cfg or {}).get('enabled', False):
        try:
            _cb_cache_dir = (os.environ.get('QUANT_CB_CACHE')
                             or os.path.join(_PROJECT_DIR, 'data', 'alternative_data'))
            _cb_result = compute_cb_sleeve(
                _post_order_cash, _cb_cfg, cb_load_state(),
                cb_load_snapshot(_cb_cache_dir, pd.Timestamp(target_date)),
                pd.Timestamp(target_date))
        except Exception as e:
            print(f"  [cb_sleeve] 生成失败(不影响股票订单): {e}")
    _cb_spend = (_cb_result['spend_est']
                 if _cb_result and _cb_result.get('rebalanced') else 0.0)
    _sweep_cfg = config.get('live_monitoring.repo_sweep', {}) or {}
    _sweep = compute_repo_sweep(max(_post_order_cash - _cb_spend, 0.0), _sweep_cfg)

    output = {'date': str(target_date), 'orders': orders}
    if _cb_result and _cb_result.get('rebalanced'):
        output['cb_sleeve'] = {
            'month': _cb_result['month'],
            'picks': _cb_result['picks'],
            'alloc_total': _cb_result['alloc_total'],
            'spend_est': _cb_result['spend_est'],
            'reason': _cb_result['reason'],
            'orders': _cb_result['orders'],
            'notes': _cb_result['notes'],
        }
        print(f"\n[转债sleeve] {_cb_result['reason']}")
        for o in _cb_result['orders']:
            _act = '买入' if o['action'] == 'open' else '卖出'
            print(f"  {_act:4s} {o['code']} {o['qty_zhang']:>4}张 "
                  f"≈{o['est_amount']:>8,.0f}元 (估价{o['est_price']:.2f}/张)")
    if _sweep['enabled']:
        output['repo_sweep'] = {
            'code': _sweep['code'],
            'tenor': _sweep['tenor'],
            'amount': _sweep['amount'],
            'reserve': _sweep['reserve'],
            'estimated_annual_yield': _sweep['estimated_annual_yield'],
            'reason': _sweep['reason'],
            'notes': sweep_guidance(_sweep),
        }
        print(f"\n[逆回购sweep] {_sweep['reason']}")
        if _sweep['amount'] >= 1000:
            print(f"  建议单: 卖出 {_sweep['code']} {_sweep['amount']:,}元 "
                  f"(年化估算≈{_sweep['estimated_annual_yield']:,.0f}元, 股票单成交后最后下)")

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

    # ── 持仓监控清单: 全持仓最新支撑/压力 + 过滤警示 ──────────
    try:
        monitor = build_monitor_list(str(target_date), new_positions, prices,
                                     signal_store, stock_file_map)
        output['monitor'] = monitor
        print_monitor_list(monitor)
    except Exception as e:
        print(f"  [monitor] 生成失败(不影响订单): {e}")

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

    # 转债sleeve状态落盘 (月度节奏跟踪, 与generate_trade_orders同口径)
    if _cb_result and _cb_result.get('rebalanced'):
        cb_save_state(_cb_result['new_state'])
        print(f"[转债sleeve] 状态已更新: {_cb_result['new_state']['month']} "
              f"{len(_cb_result['new_state']['positions'])}只")

    # ── 选股详情 ──────────────────────────────────────────
    if hasattr(portfolio, 'last_selection') and portfolio.last_selection:
        print(f"\n选股详情:")
        for sel in portfolio.last_selection:
            print(f"  {sel['code']:>8s}  score={sel['score']:.4f}  "
                  f"weight={sel.get('weight', 0):.3f}  "
                  f"industry={sel.get('industry', '')}")

    # ── K线画图 ──────────────────────────────────────────
    try:
        from live_chart import generate_position_charts
        print(f"\n生成K线图...")
        generate_position_charts(ORDERS_FILE, POSITIONS_FILE, target_date)
    except Exception as e:
        print(f"  [WARN] K线图生成失败: {e}")


def main():
    parser = argparse.ArgumentParser(description='实盘选股一键执行')
    parser.add_argument('--date', type=str, default=None,
                        help='目标日期 (默认: 今天)')
    parser.add_argument('--cash', type=float, default=500000.0,
                        help='总资金 (默认: 500000; 2026-09-09对齐真实账户~50万, 入金/出金后同步)')
    parser.add_argument('--skip-data', action='store_true',
                        help='跳过K线数据更新(用xtquant已下载的数据), 宏观/另类照常更新')
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
        update_kline_data()
    else:
        print("跳过K线数据更新 (使用 xtquant 已下载的数据)")
    update_macro_data()

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
