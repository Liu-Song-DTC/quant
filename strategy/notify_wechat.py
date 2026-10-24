#!/usr/bin/env python3
"""微信推送: 每日选股出单结果 + 守卫报警 (2026-10-24, P0).

通道: Server酱(sctapi.ftqq.com) / PushPlus(pushplus.plus), 二选一。
密钥: strategy/config/secrets.env (gitignored) 或环境变量
    QUANT_NOTIFY_KEY=<sendkey或token>
    QUANT_NOTIFY_CHANNEL=serverchan|pushplus   (默认 serverchan)

用法:
    # 出单后推送 (run_daily_live.sh 第5步, 非致命)
    python notify_wechat.py --file trade_orders.json

    # 守卫报警 (navlaw/histcells/新鲜度违反时, 同一通道)
    python notify_wechat.py --alert "navlaw违反" "今日净值复现失败 rc=2, 已拦截出单"

失败纪律: 未配置密钥=打印跳过(不拦链); 已配置但发送失败=重试1次后 exit 2
    (调用方 set +e 包裹, 报警必须落在日志里, 不许静默吞掉)。
"""
import argparse
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

BASE = Path(__file__).resolve().parent
SECRETS = BASE / 'config' / 'secrets.env'
ORDERS_FILE = BASE / 'trade_orders.json'
EQUITY_FILE = BASE / 'rolling_validation_results' / 'equity_curve.csv'


def load_secrets():
    """读 secrets.env (KEY=VALUE 行), 环境变量优先。"""
    conf = {}
    if SECRETS.exists():
        for ln in SECRETS.read_text(encoding='utf-8').splitlines():
            ln = ln.strip()
            if ln and not ln.startswith('#') and '=' in ln:
                k, _, v = ln.partition('=')
                conf[k.strip()] = v.strip().strip('"\'')
    return {
        'key': os.environ.get('QUANT_NOTIFY_KEY') or conf.get('QUANT_NOTIFY_KEY', ''),
        'channel': (os.environ.get('QUANT_NOTIFY_CHANNEL')
                    or conf.get('QUANT_NOTIFY_CHANNEL', 'serverchan')).lower(),
    }


def render_orders(path: Path) -> tuple:
    """trade_orders.json → (title, markdown)。含实盘状态区 (arch_1022 台账) + 守卫灯。"""
    data = json.loads(path.read_text(encoding='utf-8'))
    orders = data.get('orders', [])
    date = data.get('date', data.get('trade_date', ''))
    nav_line = ''
    try:
        acct = json.loads((BASE / 'live_account.json').read_text(encoding='utf-8'))
        h = sorted(acct.get('nav_history') or [], key=lambda r: str(r['date']))
        if h:
            last = h[-1]
            nav = float(last['nav'])
            pv = float(last.get('pos_value', 0))
            expo = pv / nav * 100 if nav > 0 else 0.0
            nav_line = (f"\n> 台账 NAV: **{nav:,.0f}** ({last['date']}) | "
                        f"现金 {float(last.get('cash', 0)):,.0f} | 敞口 {expo:.0f}%\n")
    except Exception:
        pass

    buys = [o for o in orders if o.get('action') == 'open']
    adj = [o for o in orders if o.get('action') == 'adjust']
    closes = [o for o in orders if o.get('action') == 'close']

    title = f"选股出单 {date}: 买{len(buys)} 调{len(adj)} 卖{len(closes)}"
    md = [f"# {title}", nav_line]
    if not orders:
        md.append("\n**今日无调仓单** (空仓或无新信号)\n")
    if buys:
        md.append("\n## 买入\n")
        md.append("| 代码 | 金额 | 评分 | 止损 | 止盈 |")
        md.append("|---|---|---|---|---|")
        for o in buys:
            md.append("| {code} | {amt:,} | {sc} | {sl} | {tp} |".format(
                code=o.get('stock_code', '?'), amt=o.get('amount', 0),
                sc=o.get('score', '-'),
                sl=o.get('stop_loss_price', '-'), tp=o.get('take_profit_price', '-')))
    if adj:
        md.append("\n## 调整\n")
        for o in adj:
            md.append(f"- {o.get('stock_code', '?')} → {o.get('amount', 0):,}")
    if closes:
        md.append("\n## 卖出\n")
        for o in closes:
            md.append(f"- {o.get('stock_code', '?')} 清仓")
    mon = data.get('monitor') or []
    if mon:
        warn = [m for m in mon if m.get('warn') or m.get('warning')]
        if warn:
            md.append(f"\n## ⚠️ 持仓警示 {len(warn)}条\n")
            for m in warn[:5]:
                md.append(f"- {m.get('stock_code', '?')} {m.get('warning', m.get('warn', ''))}")
    wl = data.get('watchlist') or []
    if wl:
        sig_d = data.get('watchlist_signal_date', '?')
        md.append(f"\n## 🐻 熊市观察名单 {len(wl)}只 (非订单)\n")
        md.append(f"> {data.get('watchlist_note', '策略熊市不进场, 仅供自行判断')}\n")
        md.append("\n| 代码 | 评分 | 调整分 | 行业 |")
        md.append("|---|---|---|---|")
        for w in wl:
            adj = w.get('adjusted_score')
            md.append(f"| {w.get('code', '?')} | {w.get('score', '-')} "
                      f"| {round(adj, 3) if isinstance(adj, (int, float)) else '-'} "
                      f"| {w.get('industry', '')} |")
        md.append(f"\n> ⚠️ 信号日期 **{sig_d}** — 跳过日不重算信号, 请自行判断陈旧度\n")
    lights = guard_lights()
    if lights:
        md.append(f"\n---\n{lights}")
    return title, '\n'.join(md)


def guard_lights() -> str:
    """守卫灯: 当日链各环节状态 (纯已落地数据重组, 缺=灰—)。
    路径: 快照/报警日志在项目根 logs/, navlaw/出单日志在 strategy/logs/。"""
    today = time.strftime('%Y%m%d')
    root_logs = Path('/mnt/d/quant/logs')
    strat_logs = BASE / 'logs'  # strategy/logs
    out = []
    for name, p, flag in [
        ('快照', root_logs / f'snapshot_{today}.log', None),
        ('navlaw', strat_logs / f'navlaw_{today}.log', 'violation'),
        ('出单', strat_logs / f'orders_{today}.log', None),
    ]:
        if not p.exists():
            out.append(f'{name}—')
        elif flag == 'violation':
            txt = p.read_text(encoding='utf-8', errors='ignore')
            bad = ('违反' in txt or '拦截' in txt) and 'REBASELINE' not in txt
            out.append(f'{"🔴" if bad else "🟢"}{name}')
        else:
            out.append(f'🟢{name}')
    alert_log = root_logs / f'alert_{today}.log'
    if alert_log.exists() and alert_log.stat().st_size > 0:
        out.append('🔴报警')
    return '守卫灯: ' + ' '.join(out)


def send(channel: str, key: str, title: str, md: str) -> None:
    """POST 一条 markdown 消息, 非2xx或业务错误码 → RuntimeError。"""
    if channel in ('serverchan', 'serverchan3'):
        # key 自动分代: sctp开头=Server酱³ (https://{uid}.push.ft07.com), 其余=Turbo (sctapi.ftqq.com)
        # 教训(10/24): sctp key 打 ftqq 老端点也返回 code:0 但不投递, 必须用 push.ft07.com
        if key.startswith('sctp'):
            uid = key[4:].split('t', 1)[0]          # sctp{uid}t{token} → uid 全数字
            url = f'https://{uid}.push.ft07.com/send/{key}.send'
        else:
            url = f'https://sctapi.ftqq.com/{key}.send'
        body = urllib.parse.urlencode({'title': title[:32], 'desp': md}).encode()
        req = urllib.request.Request(
            url, data=body,
            headers={'User-Agent': 'quant-notify/1.0'})  # 裸 urllib UA 被 ft07 WAF 403
        resp = urllib.request.urlopen(req, timeout=15)
        out = json.loads(resp.read())
        if out.get('code') != 0:
            raise RuntimeError(f'serverchan: {out}')
    elif channel == 'pushplus':
        url = 'https://www.pushplus.plus/send'
        payload = json.dumps({'token': key, 'title': title[:100],
                              'content': md, 'template': 'markdown'}).encode()
        req = urllib.request.Request(url, data=payload,
                                     headers={'Content-Type': 'application/json'})
        out = json.loads(urllib.request.urlopen(req, timeout=15).read())
        if out.get('code') != 200:
            raise RuntimeError(f'pushplus: {out}')
    else:
        raise ValueError(f'未知通道: {channel} (支持 serverchan|pushplus)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--file', default=str(ORDERS_FILE))
    ap.add_argument('--alert', nargs=2, metavar=('TITLE', 'MSG'), default=None,
                    help='守卫报警模式: 直接推送给定标题/内容')
    args = ap.parse_args()

    conf = load_secrets()
    if not conf['key']:
        print('[notify] 未配置 QUANT_NOTIFY_KEY (strategy/config/secrets.env), 跳过推送')
        return 0

    if args.alert:
        title, md = args.alert[0], args.alert[1]
    else:
        p = Path(args.file)
        if not p.exists():
            print(f'[notify] ⛔ 订单文件不存在: {p}')
            return 2
        title, md = render_orders(p)

    last_err = None
    for attempt in (1, 2):
        try:
            send(conf['channel'], conf['key'], title, md)
            print(f'[notify] ✓ 已推送 ({conf["channel"]}): {title}')
            return 0
        except Exception as e:  # noqa: BLE001 — 报警通道失败必须落地为日志
            last_err = e
            print(f'[notify] 第{attempt}次推送失败: {e}')
    print(f'[notify] ⛔ 推送失败, 请人工检查当日出单: {last_err}')
    return 2


if __name__ == '__main__':
    sys.exit(main())
