#!/usr/bin/env python3
"""
生成 trade_orders.json 给 miniqmt 做盘中高抛低吸。

用法:
    python strategy/generate_trade_orders.py [--date 2026-07-16] [--cash 250000]

前置条件:
    需要 backtest_signals.csv 覆盖目标日期。若信号过期:
      cd strategy && python bt_execution.py  # 重新生成信号

流程:
    1. 加载现有信号 CSV (backtest_signals.csv)
    2. 若目标日期不在 CSV 中 → 报错退出
    3. 运行组合构建 → 得到目标持仓
    4. 对比当前持仓 → 生成 open/adjust/close
    5. 计算止盈止损价 (stop=现价*0.94, tp=现价*1.15)
    6. 输出 trade_orders.json + 更新 current_positions.json
"""

import os, sys, json, argparse
import pandas as pd
import numpy as np
from datetime import date as date_type
from typing import Dict, List, Tuple

# 确保 strategy 目录在 path 中
_STRATEGY_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_STRATEGY_DIR)
if _STRATEGY_DIR not in sys.path:
    sys.path.insert(0, _STRATEGY_DIR)

from core.config_loader import load_config
from core.signal_store import SignalStore
from core.portfolio import PortfolioConstructor
from core.market_regime_detector import MarketRegimeDetector

config = load_config()

# ── 路径 ────────────────────────────────────────────────────────
DATA_PATH = config.get('paths.data', os.path.join(_PROJECT_DIR, 'data/stock_data/backtrader_data/'))
SIGNALS_CSV = os.path.join(_STRATEGY_DIR, 'rolling_validation_results', 'backtest_signals.csv')
POSITIONS_FILE = os.path.join(_PROJECT_DIR, 'current_positions.json')
ORDERS_FILE = os.path.join(_PROJECT_DIR, 'trade_orders.json')
INDEX_DATA_PATH = os.path.join(DATA_PATH, 'sh000001_qfq.csv')

# ── 默认参数 ─────────────────────────────────────────────────────
DEFAULT_CASH = 300000.0


def build_stock_file_map() -> Dict[str, str]:
    """扫描 DATA_PATH, 构建 {code: filepath} 映射."""
    file_map = {}
    for f in os.listdir(DATA_PATH):
        if f.startswith('._'):
            continue
        path = os.path.join(DATA_PATH, f)
        if f.endswith('_qfq.csv'):
            code = f[:-8]
        elif f.endswith('_hfq.csv'):
            code = f[:-8]
        else:
            continue
        file_map[code] = path
    return file_map


def load_prices_for_date(stock_file_map: Dict[str, str], target_date) -> Dict[str, float]:
    """读取所有股票在 target_date 的收盘价."""
    prices = {}
    target_dt = pd.Timestamp(target_date)
    for code, filepath in stock_file_map.items():
        if code == 'sh000001':
            continue
        try:
            df = pd.read_csv(filepath, parse_dates=['datetime'], usecols=['datetime', 'close'])
            row = df[df['datetime'] == target_dt]
            if not row.empty:
                prices[code] = float(row['close'].iloc[0])
            del df
        except Exception:
            continue
    print(f"价格数据: {len(prices)} 只股票有 {target_date} 的数据")
    return prices


def get_index_regime(target_date) -> Tuple[int, float, float, bool, bool]:
    """获取目标日期的市场状态."""
    if not os.path.exists(INDEX_DATA_PATH):
        return 0, 0.0, 0.0, False, False

    df = pd.read_csv(INDEX_DATA_PATH, parse_dates=['datetime'])
    detector = MarketRegimeDetector()
    regime_df = detector.generate(df)
    del df

    target_dt = pd.Timestamp(target_date)
    row = regime_df[regime_df['datetime'].dt.date == target_dt]
    if row.empty:
        return 0, 0.0, 0.0, False, False

    regime = int(row['regime'].values[0]) if 'regime' in row.columns else 0
    momentum = float(row['momentum_score'].values[0]) if 'momentum_score' in row.columns else 0.0
    trend = float(row['trend_score'].values[0]) if 'trend_score' in row.columns else 0.0
    bear_risk = bool(row['bear_risk'].values[0]) if 'bear_risk' in row.columns else False
    bear_risk_fast = bool(row['bear_risk_fast'].values[0]) if 'bear_risk_fast' in row.columns else False
    return regime, momentum, trend, bear_risk, bear_risk_fast


def compute_support_resistance(
    code: str,
    target_date,
    price: float,
    signal_store,
    stock_file_map: Dict[str, str],
) -> Tuple[float, float, dict]:
    """基于缠论结构计算支撑位(止损)和压力位(止盈).

    止损 (支撑):
      1. 最近60日最低点 × 0.995
      2. 中枢下沿 ZD × 0.995
      3. 兜底: 当前价 × 0.93
      → 取三者最大值 (最紧的支撑)

    止盈 (压力):
      1. 最近60日最高点 × 1.005
      2. 中枢上沿 ZG × 1.005
      3. 上限: 当前价 × 1.25
      → 取三者最小值 (最早的压力)

    Returns:
        (support_price, resistance_price, detail_dict)
    """
    detail = {'code': code, 'price': price}

    # ── 1. 从信号中取缠论中枢数据 ──────────────────────────
    sig = signal_store.get(code, target_date) if signal_store else None
    pivot_zg = float(getattr(sig, 'chan_pivot_zg', np.nan)) if sig else np.nan
    pivot_zd = float(getattr(sig, 'chan_pivot_zd', np.nan)) if sig else np.nan
    detail['pivot_zg'] = pivot_zg
    detail['pivot_zd'] = pivot_zd

    # ── 2. 从市场数据取60日高低点 ──────────────────────────
    recent_high = price
    recent_low = price
    filepath = stock_file_map.get(code)
    if filepath:
        try:
            df = pd.read_csv(filepath, parse_dates=['datetime'],
                            usecols=['datetime', 'high', 'low'])
            target_dt = pd.Timestamp(target_date)
            df = df[df['datetime'] <= target_dt]
            if len(df) >= 20:
                lookback = min(60, len(df))
                recent_high = float(df['high'].iloc[-lookback:].max())
                recent_low = float(df['low'].iloc[-lookback:].min())
            del df
        except Exception:
            pass
    detail['high_60d'] = recent_high
    detail['low_60d'] = recent_low

    # ── 3. 计算支撑位 (止损) ────────────────────────────────
    supports = [('93%兜底', price * 0.93)]
    if recent_low > 0:
        supports.append(('60日低点', recent_low * 0.995))
    if not np.isnan(pivot_zd) and pivot_zd > 0:
        supports.append(('中枢下沿ZD', pivot_zd * 0.995))

    best_support = max(supports, key=lambda x: x[1])
    support_price = round(best_support[1], 2)
    detail['support_reason'] = best_support[0]
    detail['support_candidates'] = [(r, round(v, 2)) for r, v in supports]

    # ── 4. 计算压力位 (止盈) ────────────────────────────────
    resistances = [('125%上限', price * 1.25)]
    if recent_high > 0:
        resistances.append(('60日高点', recent_high * 1.005))
    if not np.isnan(pivot_zg) and pivot_zg > 0:
        resistances.append(('中枢上沿ZG', pivot_zg * 1.005))

    best_resistance = min(resistances, key=lambda x: x[1])
    resistance_price = round(best_resistance[1], 2)
    detail['resistance_reason'] = best_resistance[0]
    detail['resistance_candidates'] = [(r, round(v, 2)) for r, v in resistances]

    return support_price, resistance_price, detail


def load_current_positions() -> Dict[str, dict]:
    """加载当前持仓."""
    if os.path.exists(POSITIONS_FILE):
        with open(POSITIONS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def save_current_positions(positions: Dict[str, dict]):
    """保存当前持仓."""
    with open(POSITIONS_FILE, 'w', encoding='utf-8') as f:
        json.dump(positions, f, ensure_ascii=False, indent=2)
    print(f"持仓已保存: {POSITIONS_FILE}")


def build_orders(
    target_date: str,
    adjusted_positions: Dict[str, float],
    prices: Dict[str, float],
    prev_positions: Dict[str, dict],
    total_equity: float,
    signal_store,
    stock_file_map: Dict[str, str],
) -> Tuple[List[dict], Dict[str, dict]]:
    """
    对比当前持仓与目标持仓, 生成订单列表.
    止盈止损使用缠论结构的支撑/压力位.

    Returns:
        (orders, new_positions)
    """
    orders = []
    new_positions = {}
    order_details = []  # 收集计算明细用于打印

    target_codes = set(adjusted_positions.keys())
    prev_codes = set(prev_positions.keys())

    # ── close: 之前持有但目标中已无 ──
    for code in sorted(prev_codes - target_codes):
        orders.append({'stock_code': _normalize_code(code), 'action': 'close'})

    # ── open / adjust / keep ──
    for code in sorted(target_codes):
        target_amount = adjusted_positions[code]
        if target_amount <= 0:
            if code in prev_codes:
                orders.append({'stock_code': _normalize_code(code), 'action': 'close'})
            continue

        price = prices.get(code, 0)
        if price <= 0:
            continue

        prev = prev_positions.get(code)

        if prev is None:
            # 新建仓
            sl, tp, detail = compute_support_resistance(
                code, target_date, price, signal_store, stock_file_map
            )
            orders.append({
                'stock_code': _normalize_code(code),
                'action': 'open',
                'amount': int(target_amount),
                'stop_loss_price': sl,
                'take_profit_price': tp,
            })
            new_positions[code] = {
                'entry_price': price,
                'amount': int(target_amount),
                'entry_date': target_date,
            }
            order_details.append((_normalize_code(code), 'open', detail))
        else:
            entry_price = prev.get('entry_price', price)
            prev_amount = prev.get('amount', 0)

            if abs(target_amount - prev_amount) > total_equity * 0.01:
                sl, tp, detail = compute_support_resistance(
                    code, target_date, price, signal_store, stock_file_map
                )
                orders.append({
                    'stock_code': _normalize_code(code),
                    'action': 'adjust',
                    'amount': int(target_amount),
                    'stop_loss_price': sl,
                    'take_profit_price': tp,
                })
                new_positions[code] = {
                    'entry_price': entry_price,
                    'amount': int(target_amount),
                    'entry_date': prev.get('entry_date', target_date),
                }
                order_details.append((_normalize_code(code), 'adjust', detail))
            else:
                new_positions[code] = prev

    return orders, new_positions, order_details


def _normalize_code(code: str) -> str:
    """归一化股票代码: 6位数字 → 交易所后缀格式."""
    code = str(code).zfill(6)
    if code.startswith(('60', '68')):
        return f"{code}.SH"
    else:
        return f"{code}.SZ"


def main():
    parser = argparse.ArgumentParser(description='生成 trade_orders.json 给 miniqmt')
    parser.add_argument('--date', type=str, default=None,
                        help='目标日期 (YYYY-MM-DD), 默认最新交易日')
    parser.add_argument('--cash', type=float, default=DEFAULT_CASH,
                        help=f'总资金 (默认: {DEFAULT_CASH})')
    parser.add_argument('--dry-run', action='store_true',
                        help='仅打印结果, 不写入文件')
    args = parser.parse_args()

    # ── 确定目标日期 ──────────────────────────────────────────
    if args.date:
        target_date = pd.Timestamp(args.date).date()
        date_specified = True
    else:
        target_date = date_type.today()
        date_specified = False
    print(f"目标日期: {target_date} | 总资金: {args.cash:,.0f}")

    # ── 1. 加载信号 ──────────────────────────────────────────
    stock_file_map = build_stock_file_map()
    print(f"股票文件: {len(stock_file_map)} 只")

    if not os.path.exists(SIGNALS_CSV):
        print(f"\n错误: 信号文件不存在 ({SIGNALS_CSV})")
        print("请先运行: cd strategy && python bt_execution.py")
        sys.exit(1)

    signal_store = SignalStore()
    signal_store.finalize(SIGNALS_CSV)

    # 检查目标日期是否有信号
    test_code = next((c for c in stock_file_map if c != 'sh000001'), None)
    if test_code and signal_store.get(test_code, target_date) is None:
        if date_specified:
            print(f"\n错误: 目标日期 {target_date} 无信号数据")
            print(f"请更新信号CSV: cd strategy && python bt_execution.py")
            sys.exit(1)
        else:
            # 自动回退到信号CSV中的最新日期
            sig_df = pd.read_csv(SIGNALS_CSV, usecols=['date'])
            latest_sig_date = pd.Timestamp(sig_df['date'].max()).date()
            del sig_df
            print(f"  (今日 {target_date} 无信号, 回退到最新交易日期 {latest_sig_date})")
            target_date = latest_sig_date

    # ── 2. 获取价格 ───────────────────────────────────────────
    prices = load_prices_for_date(stock_file_map, target_date)

    # ── 3. 构建候选 universe ─────────────────────────────────
    # 基础过滤: 有价格, 非指数, 非低价股
    # ST 过滤: 使用 stock_pool (与回测统一), 不在池内则跳过
    stock_pool_enabled = config.get('stock_pool.enabled', True)
    allowed_codes = None
    if stock_pool_enabled:
        from core.stock_pool import get_stock_pool
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

    print(f"可交易股票: {len(universe)} 只 (过滤后)")

    # ── 4. 获取市场状态 ──────────────────────────────────────
    market_regime, momentum_score, trend_score, bear_risk, bear_risk_fast = get_index_regime(
        target_date
    )
    regime_names = {1: '牛市', 0: '中性', -1: '熊市'}
    print(f"市场状态: {regime_names.get(market_regime, '未知')} (regime={market_regime})")

    # ── 5. 运行组合构建 ──────────────────────────────────────
    portfolio = PortfolioConstructor()

    # 读取当前持仓
    prev_positions = load_current_positions()
    current_positions = {
        code: info['amount'] for code, info in prev_positions.items()
    }

    # 构建成本信息
    cost = {}
    for code, info in prev_positions.items():
        cost[code] = [info['entry_price'], info['entry_price']]  # [avg_cost, current_cost]

    total_equity = args.cash + sum(current_positions.values())
    cash = args.cash - sum(
        info['amount'] for code, info in prev_positions.items() if code not in current_positions
    )
    cash = max(cash, 0)

    adjusted = portfolio.build(
        date=target_date,
        universe=universe,
        current_positions=current_positions,
        signal_store=signal_store,
        cash=cash,
        prices=prices,
        market_regime=market_regime,
        momentum_score=momentum_score,
        trend_score=trend_score,
        bear_risk=bear_risk,
        bear_risk_fast=bear_risk_fast,
        cost=cost,
    )

    # ── 6. 生成订单 ──────────────────────────────────────────
    orders, new_positions, order_details = build_orders(
        str(target_date), adjusted, prices, prev_positions, total_equity,
        signal_store, stock_file_map,
    )

    output = {
        'date': str(target_date),
        'orders': orders,
    }

    print(f"\n{'='*60}")
    print(f"订单生成: {len(orders)} 条")
    for o in orders:
        if o['action'] == 'close':
            print(f"  {o['action']:8s} {o['stock_code']}")
        else:
            print(f"  {o['action']:8s} {o['stock_code']} "
                  f"金额={o['amount']:>8,} "
                  f"止损={o.get('stop_loss_price', 'N/A'):>8} "
                  f"止盈={o.get('take_profit_price', 'N/A'):>8}")

    # ── 打印支撑/压力计算明细 ─────────────────────────────────
    if order_details:
        print(f"\n止盈止损计算明细:")
        print(f"  {'代码':>12s} {'现价':>8s} {'止损(支撑)':>10s} {'来源':>12s} {'止盈(压力)':>10s} {'来源':>12s}")
        print(f"  {'-'*70}")
        for code, action, d in order_details:
            sl = d.get('support_reason', '')
            tp = d.get('resistance_reason', '')
            # 从 orders 中找到对应的 sl/tp
            sl_price = next((o['stop_loss_price'] for o in orders if o.get('stock_code') == code and 'stop_loss_price' in o), 0)
            tp_price = next((o['take_profit_price'] for o in orders if o.get('stock_code') == code and 'take_profit_price' in o), 0)
            print(f"  {code:>12s} {d['price']:>8.2f} {sl_price:>10.2f} {sl:>12s} {tp_price:>10.2f} {tp:>12s}")

    if args.dry_run:
        print("\n[Dry-run] 不写入文件")
        print(json.dumps(output, ensure_ascii=False, indent=2))
        return

    # ── 7. 写入文件 ──────────────────────────────────────────
    with open(ORDERS_FILE, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n订单已写入: {ORDERS_FILE}")

    # 更新持仓状态
    save_current_positions(new_positions)

    # ── 8. 打印选股详情 ──────────────────────────────────────
    if hasattr(portfolio, 'last_selection') and portfolio.last_selection:
        print(f"\n选股详情:")
        for sel in portfolio.last_selection:
            print(f"  {sel['code']:>8s}  score={sel['score']:.4f}  "
                  f"weight={sel.get('weight', 0):.3f}  "
                  f"rank_pct={sel.get('rank_pct', 0):.3f}  "
                  f"industry={sel.get('industry', '')}  "
                  f"factor={sel.get('factor_name', '')}")


if __name__ == '__main__':
    main()
