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

熊市观察名单 (2026-10-24, 用户裁定"供自行判断"): 跳过日订单为空但**不是黑箱** —
附最后信号日的 buy=True 高分候选 (名单数=系统设定 max_positions, 与推送层同源),
显式标注信号日期与"策略不进场、非订单"。供用户自行判断是否手动参与;
`--force-full` 可绕过本闸跑完整管线。

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
from core.live_state import live_run_lock, commit_json_files
POSITIONS_FILE = os.path.join(_PROJECT_DIR, 'current_positions.json')

EQUITY_FILE = os.path.join(_STRATEGY_DIR, 'rolling_validation_results', 'equity_curve.csv')
SIGNALS_FILE = os.path.join(_STRATEGY_DIR, 'rolling_validation_results', 'backtest_signals.csv')
LOG_DIR = os.path.join(_PROJECT_DIR, '.arms', 'logs')


def _watchlist():
    """熊市观察名单: 最后信号日 buy=True 的 score top-N (跳过日不重算, 标注日期).

    与 notify_wechat._top_candidates 同源 (2026-10-24 用户裁定: 名单长度=系统设定
    portfolio.params.max_positions, 非硬编码; 行内含 factor_name 供推送层渲染权重)。
    3.2GB code-major 单遍 awk 扫描 ~1min (NTFS IO 为主); buy 列是 True/False 字符串.
    """
    from notify_wechat import _top_candidates
    return _top_candidates()


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
    from core.live_account import load_account, record_nav, last_nav, ACCOUNT_FILE
    acct = load_account(DEFAULT_CASH, {})
    from core.portfolio import PortfolioConstructor
    p = PortfolioConstructor()
    p.restore_tracking_state(PORTFOLIO_STATE_FILE)
    if str(acct.get('last_date') or '')[:10] != target:
        _cash = float(acct.get('cash', DEFAULT_CASH))
        _previous_nav = last_nav(acct)
        record_nav(acct, target, _cash, _cash, 0.0)
        p.update_returns(_cash / _previous_nav - 1 if _previous_nav and _previous_nav > 0 else 0.0)
    targets = p.build(date=pd.Timestamp(target).date(), universe=[], current_positions={},
                      signal_store=None, cash=float(acct['cash']), prices={},
                      bear_risk=True, trend_score=-1.0)
    if any(targets.values()):
        raise RuntimeError('免算路径出现非空目标，禁止跳过完整决策')
    return acct, p.tracking_state(as_of=target), ACCOUNT_FILE


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

    # 连熊计数与完整 build 同源；第61日必须恢复完整计算，不能永远跳过。
    from core.portfolio import PortfolioConstructor
    from core.live_account import load_account
    from core.execution_ledger import unresolved_orders
    _p = PortfolioConstructor()
    _p.restore_tracking_state(PORTFOLIO_STATE_FILE)
    if _p._sector_rotation is not None:
        print('[no-trade] 板块历史状态依赖当日信号，需完整计算以免跨日状态漂移')
        return 1
    if _p.v_recovery_enabled or not _p.advance_market_state(pd.Timestamp(target).date(), True):
        print('[no-trade] 熊市降级/V反需完整决策，不跳过')
        return 1
    _acct = load_account(DEFAULT_CASH, {})
    if unresolved_orders(_acct):
        print('[no-trade] 有未完成订单，必须先处理券商回报')
        return 1
    if _acct.get('last_date') and str(_acct['last_date'])[:10] > target:
        raise RuntimeError('禁止向过去推进实盘状态')

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

    orders = {'date': target, 'price_date': target, 'orders': [], 'monitor': [],
              'run_id': target.replace('-', ''), 'accounting_mode': 'actual_fills'}

    # 熊市观察名单 (2026-10-24): 空单≠黑箱, 附最后信号日高分候选供用户自行判断
    sig_date, watchlist = _watchlist()
    if watchlist:
        orders['watchlist'] = watchlist
        orders['watchlist_signal_date'] = sig_date
        orders['watchlist_note'] = (f'熊市双确认+空仓: 策略不进场(回测同款), 空单确定。'
                                    f'以下为最后信号日 {sig_date} 的 buy=True 高分候选, '
                                    f'仅供自行判断, 非订单。')
        print(f'[no-trade] 观察名单: 信号日 {sig_date}, top{len(watchlist)}:')
        for w in watchlist:
            print(f'  {w["code"]} score={w["score"]} ind={w["industry"]}')
    # 实盘空仓不能证明历史回测也空仓；禁止人工向回测净值追加平盘行。
    ok, msg = False, '回测产物保持独立，跳过日仅提交实际账户与策略状态'
    acct, state, account_file = _append_flat_state(target)
    commit_json_files({account_file: acct, PORTFOLIO_STATE_FILE: state, ORDERS_FILE: orders},
                      os.path.join(_PROJECT_DIR, '.live_state_journal.json'))
    print(f'[no-trade] 空订单和状态已一起提交: {ORDERS_FILE}')

    os.makedirs(LOG_DIR, exist_ok=True)
    ledger = {
        'date': target, 'skipped': True, 'regime': regime,
        'momentum': momentum, 'trend_score': trend,
        'bear_risk': bear_risk, 'bear_risk_fast': bear_risk_fast,
        'positions_empty': True, 'orders': [],
        'watchlist_signal_date': sig_date, 'watchlist': watchlist,
        'equity_appended': ok, 'equity_msg': msg,
    }
    lp = os.path.join(LOG_DIR, f'no_trade_skip_{target}.json')
    with open(lp, 'w', encoding='utf-8') as f:
        json.dump(ledger, f, ensure_ascii=False, indent=2)
    print(f'[no-trade] 记账: {lp}')
    print('[no-trade] 已跳过信号生成/回测 (0 单确定)')
    return 0


if __name__ == '__main__':
    with live_run_lock(_PROJECT_DIR):
        sys.exit(main())
