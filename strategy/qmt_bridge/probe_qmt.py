# -*- coding: utf-8 -*-
"""QMT 第0步能力探测 — 在 QMT 客户端内置策略环境运行 (VERIFY.md 清单执行体).

写入 D:\\quant\\strategy\\qmt_bridge\\probe_result.txt, 每项 PASS/FAIL + 明细。
不主动下单 (第4项需把 RUN_ORDER=True 才触发 1 手模拟单, 默认 False)。
"""
import os
import time
import traceback

BRIDGE = r'D:\quant\strategy\qmt_bridge'
OUT = os.path.join(BRIDGE, 'probe_result.txt')
RUN_ORDER = False          # 第4/5项: 置 True 且连模拟账户后运行
ORDER_CODE = '600000.SH'   # 1手主板最小标的 (模拟盘)
ORDER_QTY = 100

lines = []


def log(tag, ok, detail=''):
    lines.append(f'[{tag}] {"PASS" if ok else "FAIL"} {detail}')
    print(lines[-1])


# ---- 1. xtquant import ----
try:
    from xtquant import xttrader          # noqa: F401
    from xtquant import xtconstant        # noqa: F401
    try:
        from xtquant.xttrader import XtQuantTrader
        from xtquant.xttype import StockAccount
    except Exception:
        XtQuantTrader = StockAccount = None
    log('1-import-xtquant', True, 'xttrader/xtconstant ok')
except Exception as e:
    log('1-import-xtquant', False, repr(e))
    _write()
    raise SystemExit(1)


def _write():
    with open(OUT, 'w', encoding='utf-8') as f:
        f.write('\n'.join([time.strftime('%F %T')] + lines) + '\n')


# ---- 2. 连接 (需要 miniQMT 会话路径: QMT设置-交易里查询) ----
session_path = os.environ.get('QMT_SESSION', r'D:\国金QMT交易端\userdata_mini')
acct_id = os.environ.get('QMT_ACCT', '')
trader = None
if XtQuantTrader is not None and acct_id:
    try:
        connected = []

        class _CB:
            def on_connected(self):
                connected.append(True)

        trader = XtQuantTrader(session_path, int(time.time()) % 100000)
        trader.register_callback(_CB())
        trader.start()
        r = trader.connect()
        log('2-connect', r == 0, f'connect rc={r}')
    except Exception as e:
        log('2-connect', False, repr(e))
else:
    log('2-connect', False, 'XtQuantTrader 不可用或 QMT_ACCT 未设 (模拟账户号)')

# ---- 3. 文件交换 ----
try:
    hs = os.path.join(BRIDGE, 'handshake.txt')
    token = time.strftime('%Y%m%d%H%M%S')
    with open(hs, 'w', encoding='utf-8') as f:
        f.write(token)
    with open(hs, encoding='utf-8') as f:
        ok = f.read().strip() == token
    log('3-file-io', ok, f'token={token}')
except Exception as e:
    log('3-file-io', False, repr(e))

# ---- 4/5. 1手模拟单 + 回报落盘 (默认关) ----
fills = []
if RUN_ORDER and trader is not None and acct_id:
    try:
        acct = StockAccount(acct_id)

        class _TradeCB:
            def on_stock_order(self, o):
                lines.append(f'  [order-rpt] {o.stock_code} status={o.order_status}')

            def on_stock_trade(self, t):
                fills.append((t.stock_code, t.traded_price, t.traded_volume,
                              t.traded_id, time.strftime('%H:%M:%S')))

        trader.register_callback(_TradeCB())
        # 极限限价 (跌停价) 确保只挂不成交, 或用对手价; 模拟盘用市价即可
        from xtquant import xtconstant
        r = trader.order_stock(acct, ORDER_CODE, xtconstant.STOCK_BUY,
                               ORDER_QTY, xtconstant.FIX_PRICE, 0.0, 'probe', 'probe')
        log('4-order-1lot', r >= 0, f'order rc={r} (注意: 挂单可能不成交, 查QMT界面撤单)')
        time.sleep(8)
        with open(os.path.join(BRIDGE, 'fills_test.csv'), 'a', encoding='utf-8') as f:
            for x in fills:
                f.write(','.join(map(str, x)) + '\n')
        log('5-fill-callback', len(fills) > 0, f'{len(fills)} fills → fills_test.csv'
            if fills else '未成交也属正常 (挂单测试), 看委托回报即可')
    except Exception as e:
        log('4/5-order', False, repr(e) + '\n' + traceback.format_exc())
else:
    log('4/5-order', False, 'RUN_ORDER=False 或无连接 (跳过)')

_write()
print('\n完成 →', OUT)
