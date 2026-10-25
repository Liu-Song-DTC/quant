#!/usr/bin/env python3
"""实盘账户的真实现金、持仓估值与日净值。

生产出单只登记计划，由 execution_ledger 消费券商增量回报后更新实际账户。
本模块保留的估算成交/开盘 true-up 函数仅用于旧版离线研究，禁止修改 v2 账户。
缺失或损坏的既有台账不能静默重置为初始资金。
"""
import json
import os
import math
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


def load_account(initial_capital: float, positions: dict = None, *, account_file=None) -> dict:
    """读取实际台账；已有持仓而缺失现金账本时拒绝估算资金。"""
    account_file = ACCOUNT_FILE if account_file is None else os.fspath(account_file)
    positions = positions or {}
    if os.path.exists(account_file):
        try:
            with open(account_file, 'r', encoding='utf-8') as f:
                acct = json.load(f)
            acct.setdefault('nav_history', [])
            acct.setdefault('last_date', None)
            acct.setdefault('pending', None)   # 待次日开盘对账的成交
            acct.setdefault('trueups', [])     # 开盘对账审计流水
            if float(acct.get('initial_capital', 0)) <= 0:
                acct['initial_capital'] = float(initial_capital)
            if not isinstance(acct.get('cash'), (int, float)) or not math.isfinite(acct['cash']):
                raise ValueError('invalid account cash')
            if (acct.get('pending') or {}).get('fills'):
                raise ValueError('legacy estimated fills require broker reconciliation before migration')
            acct.setdefault('ledger_version', 2)
            acct.setdefault('pending_orders', {})
            acct.setdefault('processed_reports', {})
            return acct
        except Exception as e:
            raise RuntimeError(f'账户台账损坏或需要迁移，禁止重置资金: {e}') from e

    if positions:
        raise RuntimeError('已有实际持仓但缺少账户台账；请按券商核对现金，禁止按持仓成本估算')
    if not math.isfinite(float(initial_capital)) or float(initial_capital) <= 0:
        raise ValueError('initial capital must be finite and positive')
    acct = {
        'initial_capital': float(initial_capital),
        'cash': float(initial_capital),
        'nav_history': [],
        'last_date': None,
        'pending': None,
        'ledger_version': 2,
        'pending_orders': {},
        'processed_reports': {},
        'trueups': [],
        'created': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }
    print(f'  [台账] 首次运行: 初始资金 {initial_capital:,.0f}, 空仓')
    return acct


def save_account(acct: dict):
    """原子写台账 (先写临时文件再替换, 防止中断留下半截 JSON)."""
    acct['updated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    from .live_state import atomic_json
    atomic_json(ACCOUNT_FILE, acct)
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
            px = float(p.get('last_price') or (float(p.get('amount', 0)) / sh if sh > 0 else 0) or ep)
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
    if acct.get('ledger_version', 0) >= 2 and not dry_run:
        raise RuntimeError('真实账户禁止按订单估算成交；请导入券商回报')
    _comm, _stamp, _slip = _cost_params()
    summary = {'buy_amt': 0.0, 'sell_amt': 0.0, 'fees': 0.0, 'skipped': [],
               'fills': []}  # fills: 供次日开盘 true-up 对账
    cash = float(acct.get('cash', 0.0))

    def _rec_fill(code, side, sh, fill_px, act):
        summary['fills'].append({'code': code, 'side': side, 'shares': float(sh),
                                 'est_px': float(fill_px), 'action': act})

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
            _rec_fill(code, 'sell', sh, sell_px, act)
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
            _rec_fill(code, 'buy', sh, buy_px, act)
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
                _rec_fill(code, 'buy', delta, buy_px, act)
            else:
                gross = (-delta) * sell_px
                fee = gross * (_comm + _stamp)
                cash += gross - fee
                summary['sell_amt'] += gross
                summary['fees'] += fee
                _rec_fill(code, 'sell', -delta, sell_px, act)
    if not dry_run:
        acct['cash'] = cash
    summary['cash_after'] = cash
    return summary


def settle_pending(acct: dict, open_prices: dict, exec_date: str,
                   dry_run: bool = False) -> dict:
    """次日开盘 true-up (2026-10-22): 把决策日收盘估算成交校正为执行日开盘成交.

    回测 bt_execution 与实盘真实成交都在 **T+1 开盘**, 但夜间出单时 T+1 开盘
    未知 ⇒ apply_orders_to_cash 先用决策日收盘估算记账. 次日用执行日开盘价
    重算成交价并只调整**差额**:
      buy : Δcash = 股数 × (est_px − act_px) × (1+佣金)
      sell: Δcash = 股数 × (act_px − est_px) × (1−佣金−印花)
    plan 存于 acct['pending'] = {decision_date, fills:[...]}; 幂等: decision_date
    < exec_date 才结算, 结算后清空. 无开盘价的票保留在 pending 次日重试.

    这是估算对账, 不是券商成交回报: 若实际成交价/数量不同, 以真实账户为准手工修
    live_account.json.
    """
    if acct.get('ledger_version', 0) >= 2 and not dry_run:
        raise RuntimeError('真实账户禁止按开盘价推定成交；请导入券商回报')
    pending = acct.get('pending') or {}
    fills = list(pending.get('fills') or [])
    if not fills:
        return {'delta_cash': 0.0, 'n': 0, 'skipped': [], 'pending_left': 0,
                'not_due': False}
    _pdec = str(pending.get('decision_date') or '')[:10]
    if _pdec and str(exec_date)[:10] <= _pdec:
        # 自我保护: 执行日必须晚于决策日, 否则不结算
        return {'delta_cash': 0.0, 'n': 0, 'skipped': [], 'pending_left': len(fills),
                'not_due': True}
    _comm, _stamp, _slip = _cost_params()
    delta_cash = 0.0
    done, left = [], []
    for f in fills:
        px = float(open_prices.get(f['code'], 0) or 0)
        if px <= 0:
            left.append(f)
            continue
        sh = float(f['shares'])
        est = float(f['est_px'])
        if f['side'] == 'buy':
            act = px * (1.0 + _slip)
            delta_cash += sh * (est - act) * (1.0 + _comm)
        else:
            act = px * (1.0 - _slip)
            delta_cash += sh * (act - est) * (1.0 - _comm - _stamp)
        done.append(f)
    if not dry_run:
        acct['cash'] = float(acct.get('cash', 0.0)) + delta_cash
        if left:
            acct['pending'] = {'decision_date': pending.get('decision_date'),
                               'exec_date': exec_date, 'fills': left}
        else:
            acct['pending'] = None
        if done:
            acct.setdefault('trueups', []).append({
                'settled_on': str(exec_date)[:10],
                'decision_date': str(pending.get('decision_date'))[:10],
                'delta_cash': round(delta_cash, 6), 'n': len(done)})
    return {'delta_cash': delta_cash, 'n': len(done),
            'skipped': [f['code'] for f in left], 'pending_left': len(left),
            'not_due': False}


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

    # ── 次日开盘 true-up ──
    # 决策日收盘估算买 10000@10.0 → est 10.01; 执行日开盘 9.80 → act 9.8098.
    _a3 = {'cash': 400000.0, 'trueups': [],
           'pending': {'decision_date': '2026-10-22', 'fills': [
               {'code': '600000', 'side': 'buy', 'shares': 10000,
                'est_px': 10.01, 'action': 'open'}]}}
    _tu = settle_pending(_a3, {'600000': 9.80}, '2026-10-23')
    _delta_exp = 10000 * (10.01 - 9.80 * 1.001) * 1.0005
    assert abs(_tu['delta_cash'] - _delta_exp) < 1e-6, _tu
    assert abs(_a3['cash'] - (400000 + _delta_exp)) < 1e-6
    assert _a3['pending'] is None and len(_a3['trueups']) == 1
    # 幂等: 再结算无变化
    _tu2 = settle_pending(_a3, {'600000': 9.80}, '2026-10-23')
    assert _tu2['n'] == 0 and abs(_a3['cash'] - (400000 + _delta_exp)) < 1e-6
    # 无开盘价 ⇒ 保留 pending 供下次重试, 现金不动
    _a4 = {'cash': 100000.0, 'trueups': [], 'pending': {'decision_date': '2026-10-22',
          'fills': [{'code': '000001', 'side': 'sell', 'shares': 100,
                     'est_px': 19.98, 'action': 'close'}]}}
    _tu3 = settle_pending(_a4, {}, '2026-10-23')
    assert _tu3['n'] == 0 and _tu3['pending_left'] == 1 and _a4['cash'] == 100000.0
    assert _a4['pending'] and _a4['pending']['fills']
    # 执行日 <= 决策日 ⇒ 不结算 (防同日/历史误调)
    _a5 = {'cash': 1.0, 'trueups': [], 'pending': {'decision_date': '2026-10-23',
          'fills': [{'code': '600000', 'side': 'buy', 'shares': 1,
                     'est_px': 10.0, 'action': 'open'}]}}
    assert settle_pending(_a5, {'600000': 8.0}, '2026-10-23')['not_due'] is True
    assert _a5['cash'] == 1.0 and _a5['pending']
    print('self-test OK')
