"""Actual fills, independent of target positions and estimated opening prices.

Broker ``volume`` is the incremental fill quantity, never cumulative. Terminal
zero-quantity reports close rejected/cancelled orders without changing balances.
"""
from copy import deepcopy
import datetime
import hashlib
import json
import math

TERMINAL = {'FILLED_ALL', 'REJECTED', 'UNFILLED_CANCEL', 'SKIPPED_NOT_ARMED'}
STATUSES = TERMINAL | {'PARTIAL'}
# 人工核对券商账户后确认"未成交"的终结态 (不假定成交, 但需要显式操作员确认)。
ACKNOWLEDGED_TERMINAL = 'EXPIRED_UNFILLED'
# 券商按金额换算整手下单 (qmt_bridge/PROTOCOL.md §2): 实际成交股数正常情况下
# 就不等于决策日收盘算出的计划股数 (开盘价≠收盘价 + 整手取整)。因此买入的
# 上限用"金额"而不是"股数"约束; 容差覆盖桥的 3% 偏离护栏 + 整手取整。
VALUE_TOLERANCE = 1.05
# 股数兜底: 券商按金额换算整手, 成交股数可以比计划多 (开盘价低于参考价) 也可能少。
# 桥的偏离护栏是 3%, 这里用 20% 兜底 —— 只拦"数量级错乱", 不拦正常的价格差。
SHARE_TOLERANCE = 1.20


def queue_orders(account, orders, positions, decision_date):
    """Register intentions only; neither cash nor actual positions change."""
    run_id = str(decision_date)[:10].replace('-', '')
    pending = account.setdefault('pending_orders', {})
    for seq, order in enumerate(orders, 1):
        code = str(order['stock_code']).split('.')[0].zfill(6)
        old = int(positions.get(code, {}).get('shares', 0))
        target = 0 if order['action'] == 'close' else int(order.get('shares', 0))
        delta = target - old
        if not delta:
            continue
        key = f'{run_id}:{seq}'
        # 计划口径: 股数是决策日收盘算出的目标, 金额是本单增量的名义金额。
        # 券商按金额换算整手执行 ⇒ 成交股数可以与计划股数不同, 只有金额口径可作上限。
        planned_price = float(order.get('price') or 0)
        planned_amount = order.get('order_amount')
        planned_amount = (float(planned_amount) if planned_amount is not None
                          else (abs(delta) * planned_price if planned_price > 0 else None))
        record = {'run_id': run_id, 'seq': seq, 'code': code,
                  'side': 'BUY' if delta > 0 else 'SELL',
                  'quantity': abs(delta), 'filled': 0, 'filled_value': 0.0,
                  'status': 'PENDING', 'decision_date': str(decision_date)[:10],
                  'planned_price': planned_price or None,
                  'planned_amount': planned_amount}
        if key in pending and any(pending[key][k] != record[k]
                                  for k in ('code', 'side', 'quantity', 'decision_date')):
            raise ValueError(f'conflicting order identity: {key}')
        pending.setdefault(key, record)
        order.update(run_id=run_id, seq=seq, order_id=key,
                     order_side=record['side'], order_quantity=record['quantity'],
                     order_planned_amount=planned_amount)
    return run_id


def unresolved_orders(account, before=None):
    return [o for o in account.get('pending_orders', {}).values()
            if o['status'] not in TERMINAL and o['status'] != ACKNOWLEDGED_TERMINAL
            and (before is None or o['decision_date'] < str(before)[:10])]


def stale_orders(account, today, max_age_sessions=3, calendar=None):
    """未终结且已超过 max_age_sessions 个交易日的订单。

    不自动假定成交 (无回报既不代表成交也不代表撤单); 这些单需要操作员拿真实
    券商账户核对后显式 acknowledge, 或补齐 deviation_log / positions_snapshot 证据。
    """
    from .trading_calendar import sessions_elapsed
    today = str(today)[:10]
    out = []
    for o in account.get('pending_orders', {}).values():
        if o['status'] in TERMINAL or o['status'] == ACKNOWLEDGED_TERMINAL:
            continue
        try:
            age = sessions_elapsed(o['decision_date'], today, calendar)
        except (ValueError, OSError):
            age = None
        if age is not None and age > max_age_sessions:
            out.append({**o, 'age_sessions': age})
    return out


def acknowledge_orders(account, keys, reason, operator='operator'):
    """操作员核对真实券商账户后, 显式关闭未终结订单 (记为未成交)。

    这是显式的人工裁定, 不是程序假定: 未成交的余量写进偏差账本, 账户不变。
    """
    if not str(reason or '').strip():
        raise ValueError('acknowledgement requires a reason')
    pending = account.setdefault('pending_orders', {})
    stamp = datetime.datetime.now().isoformat(timespec='seconds')
    acked = []
    for key in keys:
        record = pending.get(key)
        if record is None:
            raise ValueError(f'unknown pending order: {key}')
        if record['status'] in TERMINAL or record['status'] == ACKNOWLEDGED_TERMINAL:
            continue
        record['status'] = ACKNOWLEDGED_TERMINAL
        record['acknowledged_reason'] = str(reason)
        record['acknowledged_by'] = operator
        record['acknowledged_at'] = stamp
        record['unfilled_shares'] = int(record['quantity']) - int(record.get('filled', 0))
        account.setdefault('execution_deviations', []).append({
            'order': key, 'kind': 'ACKNOWLEDGED_UNFILLED', 'at': stamp, 'operator': operator,
            'reason': str(reason), 'unfilled_shares': record['unfilled_shares']})
        acked.append(key)
    return acked


def apply_execution_reports(account, positions, reports, commission=0.0005, stamp_tax=0.0005):
    """Validate an entire batch before applying it; fill IDs are idempotent.

    未知 run_id/seq 或未知状态的回报被跳过并登记 (fills_today.csv 是 append-only,
    不按日切分, 昨天的键混进来不应中断今天的整批入账)。属于本单的矛盾回报
    (代码/方向不符、金额越界、现金不足) 仍拒绝整批 —— 那是真不一致, 必须人看。
    """
    acct, book = deepcopy(account), deepcopy(positions)
    pending = acct.setdefault('pending_orders', {})
    processed = acct.setdefault('processed_reports', {})
    applied = 0
    skipped = []
    for report in reports:
        run_id = str(report['run_id'])
        key = f"{run_id}:{int(report['seq'])}"
        status = str(report['order_status']).upper()
        if key not in pending:
            skipped.append({'key': key, 'status': status, 'reason': 'unknown order (非本台账订单)'})
            continue
        if status not in STATUSES:
            skipped.append({'key': key, 'status': status, 'reason': 'unknown status'})
            continue
        _raw_code = str(report.get('stock_code') or '').strip()
        code = _raw_code.split('.')[0].zfill(6) if _raw_code else ''
        side = str(report.get('order_type') or '').strip().upper()
        qty_num = float(report.get('volume', 0) or 0)
        price = float(report.get('price', 0) or 0)
        if not math.isfinite(qty_num) or qty_num < 0 or not qty_num.is_integer():
            raise ValueError('fill volume must be a finite nonnegative integer')
        qty = int(qty_num)
        if not math.isfinite(price) or (qty and price <= 0):
            raise ValueError('invalid execution price or fees')
        order = pending[key]
        # 偏差账本允许只带 run_id/seq/status 的最小行 ⇒ 缺失字段不参与一致性校验。
        if code and code != order['code']:
            raise ValueError(f'execution does not match order: {key}')
        if side and side != order['side']:
            raise ValueError(f'execution does not match order: {key}')
        code = order['code']
        side = order['side']
        fee_text = report.get('fees')
        fees = (float(fee_text) if fee_text not in (None, '')
                else qty * price * (commission + (stamp_tax if side == 'SELL' else 0)))
        if not math.isfinite(fees) or fees < 0:
            raise ValueError('invalid execution price or fees')
        canonical = {'order': key, 'code': code, 'side': side, 'quantity': qty,
                     'price': price, 'fees': float(fee_text) if fee_text not in (None, '') else None,
                     'status': status, 'ts': str(report.get('ts', ''))}
        digest = hashlib.sha256(json.dumps(canonical, sort_keys=True).encode()).hexdigest()
        fill_id = str(report.get('fill_id') or digest)
        identity = f'{run_id}:{fill_id}'
        if identity in processed:
            if processed[identity] != digest:
                raise ValueError(f'fill ID reused with different content: {identity}')
            continue
        if order['status'] in TERMINAL:
            raise ValueError(f'execution received after terminal order: {key}')
        if order['status'] == ACKNOWLEDGED_TERMINAL:
            # 操作员曾判定"未成交", 但券商回报才是账户真相 ⇒ 按成交入账并留痕。
            skipped.append({'key': key, 'status': status,
                            'reason': '已人工判定未成交的订单收到券商成交, 按券商回报入账'})
            order['status'] = 'PENDING'
        if status in {'REJECTED', 'SKIPPED_NOT_ARMED'} and qty:
            raise ValueError('rejected/skipped report cannot contain a fill')
        old = book.get(code, {})
        held = int(old.get('shares', 0))
        if qty and side == 'BUY':
            spend = qty * price + fees
            if spend > acct['cash'] + 1e-6:
                raise ValueError(f'insufficient cash for actual fill: {key}')
            # 买入上限: 股数兜底 + 金额上限。金额口径才是真约束 (券商按金额下单)。
            if order['filled'] + qty > order['quantity'] * SHARE_TOLERANCE + 1:
                raise ValueError(f'overfill: {key}')
            planned_amount = order.get('planned_amount')
            if planned_amount:
                if order.get('filled_value', 0.0) + qty * price > planned_amount * VALUE_TOLERANCE + 1.0:
                    raise ValueError(f'overfill: {key}')
            new_qty = held + qty
            book[code] = {**old, 'shares': new_qty,
                          'entry_price': (held * float(old.get('entry_price', 0)) + qty * price) / new_qty,
                          'amount': new_qty * price,
                          'entry_date': old.get('entry_date') or str(report.get('ts', ''))[:10],
                          'last_price': price}
            acct['cash'] -= spend
        elif qty and side == 'SELL':
            if qty > held:
                raise ValueError(f'cannot sell unowned shares: {key}')
            # 卖出必须不超过本单计划股数 (多卖 = 与指令矛盾)。
            if order['filled'] + qty > order['quantity']:
                raise ValueError(f'overfill: {key}')
            acct['cash'] += qty * price - fees
            if qty == held:
                book.pop(code, None)
            else:
                book[code] = {**old, 'shares': held - qty, 'amount': (held - qty) * price,
                              'last_price': price}
        order['filled'] += qty
        order['filled_value'] = float(order.get('filled_value', 0.0)) + qty * price
        order['status'] = status
        if status == 'FILLED_ALL':
            residual = int(order['quantity']) - int(order['filled'])
            order['unfilled_shares'] = residual
            if residual:
                # 券商按金额换算整手, 成交股数与计划不同是正常结果, 不是异常。
                acct.setdefault('execution_deviations', []).append({
                    'order': key, 'kind': 'FILLED_ALL_RESIDUAL', 'planned': int(order['quantity']),
                    'filled': int(order['filled']), 'residual_shares': residual,
                    'direction': 'short' if residual > 0 else 'over',
                    'planned_amount': order.get('planned_amount'),
                    'filled_value': round(float(order.get('filled_value', 0.0)), 2)})
        processed[identity] = digest
        acct.setdefault('execution_history', []).append({**canonical, 'fill_id': fill_id,
                          'booked_fees': fees, 'fees_estimated': fee_text in (None, '')})
        applied += 1
    acct['ledger_version'] = 2
    if skipped:
        acct['skipped_reports'] = skipped
    account.clear(); account.update(acct)
    positions.clear(); positions.update(book)
    return applied
