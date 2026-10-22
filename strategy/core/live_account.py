#!/usr/bin/env python3
"""实盘账户台账 (2026-10-22): 让实盘资本像回测一样复利.

问题 (读码 generate_trade_orders.py, 2026-10-22):
  旧口径 total_equity = 静态 --cash(50万), cash = 静态资金 − Σ持仓['amount'],
  而 amount 是**下单时**的市值, 之后从不更新. 后果:
    ① 持仓浮盈浮亏不进入净值/目标仓位分母 ⇒ 账户涨了实盘系统性低配、亏了系统性超配;
    ② 平仓回款只按 amount(下单时值)回到现金, 已实现盈亏永久丢失;
    ③ 1.5×静态资金守卫会在账户增长 ~50% 后误报.
  回测 bt_execution 用 nav = cash + Σ shares×close 复利 —— 这是"实盘=回测"
  (用户铁律) 的资本维度, 旧实盘路径缺失.

本模块口径 (与 bt_execution 主循环同构):
  nav(t)   = cash + Σ shares(t) × close(t)        # 收盘后估值
  return(t)= nav(t)/nav(t−1) − 1                  # 喂 PortfolioConstructor.update_returns
  买入成交: fill = px×(1+slip), cash −= shares×fill×(1+comm)
  卖出成交: fill = px×(1−slip), cash += shares×fill×(1−comm−stamp)
  entry_price = fill (成本), 与回测 buy_px/sell_px 同口径.
  成本参数取 yaml backtest.* (comm 0.0005 / stamp 0.0005 / slip 0.001).

限制 (必须如实告知):
  无券商成交回报 ⇒ 以出单价+滑点**估算**成交. 若实际成交价/数量不同, 以真实
  账户为准, 手工修正 live_account.json 的 cash / initial_capital 与
  current_positions.json. 台账只保证"同一条决策在同样的成交假设下可复现".
"""
import json
import os
from datetime import datetime

import pandas as pd

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ACCOUNT_FILE = os.path.join(PROJECT_DIR, 'live_account.json')

# 与 yaml backtest.* 同源 (延迟读取, 避免 import 循环)
def _cost_params():
    try:
        from core.config_loader import load_config
        c = load_config()
        return (float(c.get('backtest.commission', 0.0005)),
                float(c.get('backtest.stamp_tax', 0.0005)),
                float(c.get('backtest.slippage', 0.001)))
    except Exception:
        return 0.0005, 0.0005, 0.001


def load_account(initial_capital: float, positions: dict = None) -> dict:
    """读取台账; 首次运行(无文件)按初始资金初始化.

    迁移态: 无台账但已有持仓 ⇒ 现金按成本法 = max(initial − Σshares×entry_price, 0),
    与旧实盘口径一致, 并打印警告 (已实现盈亏从此日起算).
    """
    positions = positions or {}
    if os.path.exists(ACCOUNT_FILE):
        try:
            with open(ACCOUNT_FILE, 'r', encoding='utf-8') as f:
                acct = json.load(f)
            acct.setdefault('nav_history', [])
            acct.setdefault('last_date', None)
            if float(acct.get('initial_capital', 0)) <= 0:
                acct['initial_capital'] = float(initial_capital)
            if not isinstance(acct.get('cash'), (int, float)):
                acct['cash'] = float(initial_capital)
            return acct
        except Exception as e:
            print(f'  [WARN] live_account.json 读取失败({e}), 按首次运行初始化')

    acct = {
        'initial_capital': float(initial_capital),
        'cash': float(initial_capital),
        'nav_history': [],
        'last_date': None,
        'created': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }
    if positions:
        _cost = 0.0
        for _c, _p in positions.items():
            _sh = float(_p.get('shares', 0) or 0)
            _ep = float(_p.get('entry_price', 0) or 0)
            _cost += _sh * _ep if _sh > 0 and _ep > 0 else float(_p.get('amount', 0) or 0)
        acct['cash'] = max(float(initial_capital) - _cost, 0.0)
        print(f'  [台账] 首次运行但已有持仓: 现金按成本法初始化 '
              f'{acct["cash"]:,.0f} (初始{initial_capital:,.0f} − 成本{_cost:,.0f}); '
              f'已实现盈亏从今日起算')
    else:
        print(f'  [台账] 首次运行: 初始资金 {initial_capital:,.0f}, 空仓')
    return acct


def save_account(acct: dict):
    """原子写台账 (先写临时文件再替换, 防止中断留下半截 JSON)."""
    acct['updated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    tmp = ACCOUNT_FILE + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(acct, f, ensure_ascii=False, indent=2)
    os.replace(tmp, ACCOUNT_FILE)
    print(f'  [台账] 已保存: {ACCOUNT_FILE} (现金 {acct["cash"]:,.0f})')


def mark_to_market(positions: dict, prices: dict) -> tuple:
    """按收盘价估值. 返回 (pos_value, n_missing_price).

    无行情的持仓用 entry_price 兜底 (停牌), 与 generate_trade_orders 停牌补价同向.
    shares 缺失时退回 amount/entry_price (旧格式持仓文件).
    """
    pos_value = 0.0
    missing = 0
    for code, p in positions.items():
        sh = float(p.get('shares', 0) or 0)
        ep = float(p.get('entry_price', 0) or 0)
        if sh <= 0 and ep > 0:
            sh = float(p.get('amount', 0) or 0) / ep
        px = float(prices.get(code, 0) or 0)
        if px <= 0:
            px = ep
            missing += 1
        pos_value += sh * px
    return pos_value, missing


def last_nav(acct: dict):
    h = acct.get('nav_history') or []
    return float(h[-1]['nav']) if h else None


def record_nav(acct: dict, date_str: str, nav: float, cash: float, pos_value: float):
    """追加/覆盖当日净值行 (同一天重跑幂等)."""
    d = str(date_str)[:10]
    h = [r for r in acct.get('nav_history', []) if str(r.get('date'))[:10] != d]
    h.append({'date': d, 'nav': round(float(nav), 6),
              'cash': round(float(cash), 2), 'pos_value': round(float(pos_value), 2)})
    h.sort(key=lambda r: str(r['date']))
    acct['nav_history'] = h
    acct['last_date'] = d


def _order_fill(code: str, px: float) -> tuple:
    """返回 (买入成交价, 卖出成交价)."""
    _comm, _stamp, _slip = _cost_params()
    return px * (1.0 + _slip), px * (1.0 - _slip)


def apply_orders_to_cash(acct: dict, orders: list, prev_positions: dict,
                         prices: dict, dry_run: bool = False) -> dict:
    """按订单估算成交, 更新台账现金. 返回摘要 (供打印/核对), 不修改持仓.

    - close: shares 取订单/持仓; 卖出现金 += shares×fill×(1−comm−stamp)
    - open : 买入现金 −= shares×fill×(1+comm)
    - adjust: 差分 shares (新−旧) 同 open/close 方向
    无价格的订单跳过并计入 skipped (不猜价格).
    """
    _comm, _stamp, _slip = _cost_params()
    summary = {'buy_amt': 0.0, 'sell_amt': 0.0, 'fees': 0.0, 'skipped': []}
    cash = float(acct.get('cash', 0.0))
    for o in orders:
        code = str(o.get('stock_code', '')).split('.')[0].zfill(6)
        act = o.get('action')
        prev = prev_positions.get(code, {}) or {}
        px = float(o.get('price', 0) or prices.get(code, 0) or 0)
        if px <= 0:
            summary['skipped'].append(code)
            continue
        buy_px, sell_px = _order_fill(code, px)
        if act == 'close':
            sh = float(o.get('shares', 0) or prev.get('shares', 0) or 0)
            if sh <= 0:
                summary['skipped'].append(code)
                continue
            gross = sh * sell_px
            fee = gross * (_comm + _stamp)
            cash += gross - fee
            summary['sell_amt'] += gross
            summary['fees'] += fee
        elif act == 'open':
            sh = float(o.get('shares', 0) or 0)
            if sh <= 0:
                summary['skipped'].append(code)
                continue
            gross = sh * buy_px
            fee = gross * _comm
            cash -= gross + fee
            summary['buy_amt'] += gross
            summary['fees'] += fee
        elif act == 'adjust':
            new_sh = float(o.get('shares', 0) or 0)
            old_sh = float(o.get('prev_shares', prev.get('shares', 0)) or 0)
            delta = new_sh - old_sh
            if abs(delta) < 0.5:
                continue
            if delta > 0:
                gross = delta * buy_px
                fee = gross * _comm
                cash -= gross + fee
                summary['buy_amt'] += gross
                summary['fees'] += fee
            else:
                gross = (-delta) * sell_px
                fee = gross * (_comm + _stamp)
                cash += gross - fee
                summary['sell_amt'] += gross
                summary['fees'] += fee
    if not dry_run:
        acct['cash'] = cash
    summary['cash_after'] = cash
    return summary


def est_fill_price(code: str, px: float, side: str) -> float:
    """给持仓文件的 entry_price 用: 买入成本含滑点 (与台账同口径)."""
    b, s = _order_fill(code, px)
    return b if side == 'buy' else s


# ─────────────────────────── self-test ───────────────────────────
if __name__ == '__main__':
    _a = {'initial_capital': 500000.0, 'cash': 500000.0, 'nav_history': [], 'last_date': None}
    _orders = [
        {'stock_code': '600000.SH', 'action': 'open', 'shares': 10000, 'price': 10.0},
        {'stock_code': '000001.SZ', 'action': 'open', 'shares': 5000, 'price': 20.0},
    ]
    _s = apply_orders_to_cash(_a, _orders, {}, {}, dry_run=False)
    print('buy:', _s)
    # 10000*10*1.001×(1+0.0005)=100150.05 ; 两笔共 200300.10
    assert abs(_a['cash'] - (500000 - 100150.05 * 2)) < 0.5, _a['cash']

    _pos = {'600000': {'shares': 10000, 'entry_price': 10.01},
            '000001': {'shares': 5000, 'entry_price': 20.02}}
    _nav, _miss = mark_to_market(_pos, {'600000': 11.0, '000001': 19.0})
    assert _miss == 0
    assert abs(_nav - (110000 + 95000)) < 1e-6
    record_nav(_a, '2026-10-22', _a['cash'] + _nav, _a['cash'], _nav)
    record_nav(_a, '2026-10-22', _a['cash'] + _nav, _a['cash'], _nav)  # 幂等
    assert len(_a['nav_history']) == 1

    _orders2 = [{'stock_code': '600000.SH', 'action': 'close', 'shares': 10000, 'price': 11.0},
                {'stock_code': '000001.SZ', 'action': 'adjust', 'shares': 4000,
                 'prev_shares': 5000, 'price': 19.0}]
    _s2 = apply_orders_to_cash(_a, _orders2, _pos, {}, dry_run=False)
    print('sell/adjust:', _s2)
    assert _s2['skipped'] == []
    # close 11000*0.999*(1-0.001)=10978.9 ; adjust sell 1000*18.981*(0.9985)
    assert _s2['sell_amt'] > 0 and _s2['buy_amt'] == 0
    print('self-test OK')
