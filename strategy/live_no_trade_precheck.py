#!/usr/bin/env python3
"""熊市 + 空仓免算前置闸 (2026-10-22).

用户问题: "如果是熊市，为什么还要进行这么久的实盘计算，开始不就能知道了，
不用进行信号生成"。

判定条件 (与 portfolio.build 的 trend_bear 双确认早退同源):
  ① current_positions.json 存在且为 **空 dict** (真正无仓可卖);
  ② 目标日是交易日 (指数数据最后一行 == 目标日);
  ③ get_index_regime(target) 给出 trend_score < 0 且 bear_risk == True;
  ④ v_recovery_enabled == False (V反可能把 bear_risk 降级为 FAST, 不跳);
  ⑤ 已有 trade_orders.json 若已是今日且 orders 非空 → 不覆盖, 不跳.
满足时: 信号生成/回测**确定**不会产生任何订单 (desired_value={}; 无持仓 ⇒ 无卖单),
直接写空 trade_orders.json, 并把今日 0 收益追加到 equity_curve (历史记录不中断).

退出码: 0=已跳过 (免算) / 1=不能跳过, 继续完整管线 / 2+=错误.
绝不使用 --skip-freshness: 本脚本只应放在 refresh_all.sh 的新鲜度门禁**之后**.
"""
import os
import sys
import json
import argparse
from datetime import date as date_type

_STRATEGY_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_STRATEGY_DIR)
if _STRATEGY_DIR not in sys.path:
    sys.path.insert(0, _STRATEGY_DIR)

import pandas as pd

from core.config_loader import load_config
from generate_trade_orders import (get_index_regime, INDEX_DATA_PATH, ORDERS_FILE,
                                   PORTFOLIO_STATE_FILE, DEFAULT_CASH)
POSITIONS_FILE = os.path.join(_PROJECT_DIR, 'current_positions.json')

EQUITY_FILE = os.path.join(_STRATEGY_DIR, 'rolling_validation_results', 'equity_curve.csv')
LOG_DIR = os.path.join(_PROJECT_DIR, '.arms', 'logs')


def _index_last_date():
    if not os.path.exists(INDEX_DATA_PATH):
        return None
    d = pd.read_csv(INDEX_DATA_PATH, usecols=['datetime'])
    return str(pd.to_datetime(d['datetime']).max().date())


def _existing_orders(target):
    if not os.path.exists(ORDERS_FILE):
        return None
    try:
        with open(ORDERS_FILE, 'r', encoding='utf-8') as f:
            o = json.load(f)
        return o if str(o.get('date')) == target else None
    except Exception:
        return None


def _append_flat_nav(target, prev_trading_day):
    """把 target 的 0 收益行追加到 equity_curve; 仅当上一行=前一交易日."""
    if not os.path.exists(EQUITY_FILE):
        return False, 'equity_curve 不存在'
    with open(EQUITY_FILE, 'r', encoding='utf-8') as f:
        lines = f.read().splitlines()
    if not lines:
        return False, 'equity_curve 为空'
    last = lines[-1].split(',')
    if last[0] == target:
        return True, '今日净值行已存在'
    if last[0] != prev_trading_day:
        return False, f'最后净值日 {last[0]} != 前一交易日 {prev_trading_day}, 拒绝追加'
    nav_str = last[1]
    # 二进制追加, 保证历史字节逐位不变 (只在末尾补一行)
    with open(EQUITY_FILE, 'rb') as f:
        raw = f.read()
    with open(EQUITY_FILE, 'ab') as f:
        if raw and not raw.endswith(b'\n'):
            f.write(b'\n')
        f.write(f'{target},{nav_str},0.0\n'.encode('utf-8'))
    return True, '已追加 0 收益行'


def _append_flat_state(target):
    """跳过日: 把平盘净值(0收益)记入实盘台账 + 组合状态.

    与回测 bt_execution:2413 每日 update_returns(0) 同构. 空仓日 NAV 不变,
    但波动率降仓窗口需要连续的日收益序列 — 缺一天会让恢复交易后的窗口
    混入更早的非零收益, 与实盘自身历史不同构.
    幂等: 以 live_account.last_date / 状态文件 as_of 为提交标记, 同日重跑不重复.
    """
    from core.live_account import load_account, last_nav, record_nav, save_account
    acct = load_account(DEFAULT_CASH, {})
    if str(acct.get('last_date') or '')[:10] != target:
        _nav = last_nav(acct)
        _cash = float(acct.get('cash', DEFAULT_CASH))
        if _nav is None:
            _nav = _cash
        record_nav(acct, target, _nav, _cash, 0.0)
        save_account(acct)
        print(f'[no-trade] 台账: 平盘记账 NAV {_nav:,.0f} ({target})')
    else:
        print(f'[no-trade] 台账: {target} 已记账, 跳过')

    if not os.path.exists(PORTFOLIO_STATE_FILE):
        print('[no-trade] 组合状态: 尚无状态文件 (首次完整管线后才会生成)')
        return
    try:
        with open(PORTFOLIO_STATE_FILE, 'r', encoding='utf-8') as f:
            _asof = str(json.load(f).get('as_of') or '')
        if _asof[:10] == target:
            print(f'[no-trade] 组合状态: as_of={_asof}, 已推进, 跳过')
            return
        from core.portfolio import PortfolioConstructor
        p = PortfolioConstructor()
        p.restore_tracking_state(PORTFOLIO_STATE_FILE)
        p.update_returns(0.0)
        p.save_tracking_state(PORTFOLIO_STATE_FILE, as_of=target)
        print(f'[no-trade] 组合状态: 追加 0 收益 (as_of {_asof or "?"} → {target}, '
              f'窗口 {len(p._daily_returns)}/{p.vol_lookback})')
    except Exception as e:
        print(f'[no-trade] WARN 组合状态平盘推进失败: {e}')


def main():
    ap = argparse.ArgumentParser(description='熊市空仓免算前置闸')
    ap.add_argument('--date', default=date_type.today().isoformat())
    ap.add_argument('--check-only', action='store_true', help='只判定不写文件')
    args = ap.parse_args()
    target = str(args.date)[:10]

    cfg = load_config()
    if bool(cfg.get('portfolio.v_recovery_enabled', False)):
        print('[no-trade] v_recovery_enabled=True ⇒ 不跳 (V反可能解除空仓)')
        return 1

    if not os.path.exists(POSITIONS_FILE):
        print(f'[no-trade] 持仓文件不存在: {POSITIONS_FILE} ⇒ 不跳')
        return 1
    with open(POSITIONS_FILE, 'r', encoding='utf-8') as f:
        positions = json.load(f)
    if positions:
        print(f'[no-trade] 当前持仓 {len(positions)} 只 ⇒ 不跳 (熊市双确认需生成卖单)')
        return 1

    idx_last = _index_last_date()
    if idx_last != target:
        print(f'[no-trade] 指数最后日期 {idx_last} != 目标日 {target} ⇒ 不跳 (非交易日/数据未更新)')
        return 1

    regime, momentum, trend, bear_risk, bear_risk_fast = get_index_regime(target)
    print(f'[no-trade] 目标日 {target}: regime={regime} momentum={momentum:.6f} '
          f'trend={trend:.4f} bear_risk={bear_risk} bear_risk_fast={bear_risk_fast}')
    if not (trend < 0 and bear_risk):
        print('[no-trade] 非 (trend<0 且 bear_risk) ⇒ 不跳, 继续完整管线')
        return 1

    existing = _existing_orders(target)
    if existing and existing.get('orders'):
        print(f'[no-trade] 今日已有 {len(existing["orders"])} 条订单 ⇒ 不覆盖, 不跳')
        return 1

    prev_day = None
    if os.path.exists(EQUITY_FILE):
        eq = pd.read_csv(EQUITY_FILE, usecols=['date'])
        eq_dates = pd.to_datetime(eq['date'])
        prev = eq_dates[eq_dates < pd.Timestamp(target)]
        prev_day = str(prev.max().date()) if len(prev) else None

    if args.check_only:
        print(f'[no-trade] CHECK-ONLY: 条件满足, 将跳过 (prev_nav_date={prev_day})')
        return 0

    orders = {'date': target, 'price_date': target, 'orders': [], 'monitor': []}
    tmp = ORDERS_FILE + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(orders, f, ensure_ascii=False, indent=2)
    os.replace(tmp, ORDERS_FILE)
    print(f'[no-trade] 已写空订单: {ORDERS_FILE}')

    ok, msg = _append_flat_nav(target, prev_day)
    print(f'[no-trade] equity_curve: {msg}')
    if not ok:
        print('[no-trade] WARN: 净值未追加; 订单已写空, 下一完整管线会补齐')

    # 实盘台账/组合状态连续性 (2026-10-22): 跳过日 = 0 单+平盘, 但状态必须推进.
    try:
        _append_flat_state(target)
    except Exception as e:
        print(f'[no-trade] WARN 台账/状态平盘推进失败: {e}')

    os.makedirs(LOG_DIR, exist_ok=True)
    ledger = {
        'date': target, 'skipped': True, 'regime': regime,
        'momentum': momentum, 'trend_score': trend,
        'bear_risk': bear_risk, 'bear_risk_fast': bear_risk_fast,
        'positions_empty': True, 'orders': [],
        'equity_appended': ok, 'equity_msg': msg,
    }
    lp = os.path.join(LOG_DIR, f'no_trade_skip_{target}.json')
    with open(lp, 'w', encoding='utf-8') as f:
        json.dump(ledger, f, ensure_ascii=False, indent=2)
    print(f'[no-trade] 记账: {lp}')
    print('[no-trade] 已跳过信号生成/回测 (0 单确定)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
