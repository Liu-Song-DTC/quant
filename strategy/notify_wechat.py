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
SIGNALS_FILE = BASE / 'rolling_validation_results' / 'backtest_signals.csv'


def _config_top_n() -> int:
    """名单长度 = 系统设定的持仓槽位数 (portfolio.params.max_positions), 不硬编码。

    2026-10-24 用户裁定: 名单数与系统配置一致 (当前 mp3=3), 不要 8/10 这类拍脑袋数;
    名单还须给出评分排名与合成因子权重。mp 未来若调, 名单自动跟随。
    """
    try:
        if str(BASE) not in sys.path:
            sys.path.insert(0, str(BASE))
        from core.config_loader import load_config
        return max(1, int(load_config().get('portfolio.params.max_positions', 3) or 3))
    except Exception as e:  # noqa: BLE001 — 回退不拦推送
        print(f'[notify] WARN 名单长度读配置失败, 回退3: {e}')
        return 3


_WEIGHTS_CACHE = {}


def _factor_weights(factor_name, signal_date=''):
    """组合因子名 → (口径, [(因子名, 权重)])。唯一命中才返回, 否则 None。

    生产评分链 (signal_engine._multi_factor_score): 固定行业块 P0 路径 —
    INDUSTRY_FACTOR_CONFIG[specific_industry] (熊市先查季度表 config/quarterly_factors/
    {qid}.yaml 的 bear_factors) 按块内顺序加权, factor_name = 实际用到的因子按序
    '+' 拼接 (末成分可能带 FQG 质量后缀如 _FBA)。IC 库在 bt 链未启用、ic_cache 为空
    ⇒ P0 是唯一来源。注意 CSV 的 industry 列≠评分用的 specific_industry,
    所以按成分列表(含顺序)跨 全局表+当季表 的 factors/bull_/bear_ 全部反查;
    命中 0 个或 >1 个都不显示 (不猜)。
    """
    parts = [p for p in str(factor_name or '').split('+') if p]
    if not parts:
        return None
    if 'sources' not in _WEIGHTS_CACHE:
        import yaml
        srcs = []
        try:
            cfg = yaml.safe_load(
                (BASE / 'config' / 'factor_config.yaml').read_text(encoding='utf-8'))
            srcs.append(cfg.get('industry_factors') or {})
        except Exception as e:  # noqa: BLE001
            print(f'[notify] WARN 全局权重表读取失败: {e}')
        try:
            qdir = BASE / 'config' / 'quarterly_factors'
            qid = f"{str(signal_date)[:4]}Q{(int(str(signal_date)[5:7]) - 1) // 3 + 1:02d}" \
                if len(str(signal_date)) >= 7 else ''
            for qf in sorted(qdir.glob(f'{qid}.yaml')) if qid else []:
                qc = yaml.safe_load(qf.read_text(encoding='utf-8')) or {}
                srcs.append(qc.get('industry_factors') or {})
        except Exception as e:  # noqa: BLE001
            print(f'[notify] WARN 季度权重表读取失败: {e}')
        _WEIGHTS_CACHE['sources'] = srcs
    variants = [parts]
    if len(parts) > 1:
        variants.append(parts[:-1] + [parts[-1].rsplit('_', 1)[0]])  # 去 FQG 后缀重试
    hits = []
    for table in _WEIGHTS_CACHE['sources']:
        for blk in table.values():
            if not isinstance(blk, dict):
                continue
            for kind, kf, kw in (('基准', 'factors', 'weights'),
                                 ('牛市', 'bull_factors', 'bull_weights'),
                                 ('熊市', 'bear_factors', 'bear_weights')):
                fs = [str(f) for f in (blk.get(kf) or [])]
                if not fs or len(fs) != len(blk.get(kw) or []):
                    continue
                if any(fs == v for v in variants):
                    hits.append((kind, list(zip(fs, [float(w) for w in blk[kw]]))))
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        print(f'[notify] WARN 权重成分命中 {len(hits)} 个块, 跳过权重显示 (不猜)')
    return None


def _top_candidates(top_n=None):
    """每日选股名单: 最后信号日 buy=True 的 score top-N (与熊市观察名单同源).

    列位: code=1,date=2,buy=3(True/False 字符串),score=5,factor_name=8,industry=9,
    adjusted_score=41(末列)。3.2GB code-major 单遍 awk ~1min (NTFS IO 为主)。
    熊市跳过日由 precheck 预先嵌入 watchlist (避免双扫); 正常日在推送时现提。
    """
    if top_n is None:
        top_n = _config_top_n()
    if not SIGNALS_FILE.exists():
        return '', []
    try:
        import subprocess
        tail = subprocess.run(['tail', '-1', str(SIGNALS_FILE)],
                              capture_output=True, text=True, timeout=60).stdout.strip()
        last_date = tail.split(',')[1] if ',' in tail else ''
        if not last_date:
            return '', []
        r = subprocess.run(
            ['awk', '-F,', '-v', f'd={last_date}',
             'NR>1 && $2==d && $3=="True" && NF>=41 {print $1","$5","$41","$9","$8}',
             str(SIGNALS_FILE)],
            capture_output=True, text=True, timeout=300)
        rows = [ln.split(',') for ln in (r.stdout or '').strip().splitlines() if ln]
        rows.sort(key=lambda x: float(x[1]) if len(x) > 1 and x[1] not in ('', 'nan')
                  else -9e9, reverse=True)
        sel = []
        for x in rows[:top_n]:
            sel.append({'code': x[0],
                        'score': float(x[1]) if x[1] not in ('', 'nan') else None,
                        'adjusted_score': float(x[2]) if len(x) > 2 and x[2] not in ('', 'nan')
                        else None,
                        'industry': x[3] if len(x) > 3 else '',
                        'factor_name': x[4] if len(x) > 4 else ''})
        return last_date, sel
    except Exception as e:  # noqa: BLE001 — 名单失败不拦推送
        print(f'[notify] WARN 选股名单提取失败: {e}')
        return '', []


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

    # 每日选股名单 (2026-10-24 用户裁定: 不管什么状态, 操作建议之外必须附名单;
    # 名单长度=系统设定 max_positions, 带评分排名+合成因子权重):
    # 熊市跳过日用 precheck 预嵌的 watchlist; 正常日推送时从最新信号现提。
    wl = data.get('watchlist') or []
    if wl:
        sel_date, sel = data.get('watchlist_signal_date', '?'), wl
        bear_mode = True
    else:
        sel_date, sel = _top_candidates()
        bear_mode = False

    title = (f"选股出单 {date}: 买{len(buys)} 调{len(adj)} 卖{len(closes)}"
             + (f" | 名单{len(sel)}" if sel else ""))
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
    if sel:
        if bear_mode:
            md.append(f"\n## 🐻 熊市观察名单 {len(sel)}只 (非订单)\n")
            md.append(f"> {data.get('watchlist_note', '策略熊市不进场, 仅供自行判断')}\n")
        else:
            md.append(f"\n## 📋 每日选股名单 {len(sel)}只 (非订单)\n")
            md.append("> buy=True 按评分排序, 名单数=系统持仓设定; "
                      "是否实际买入以订单区为准 (组合层闸/仓位/敞口会筛掉部分候选)\n")
        fmt = lambda v: round(v, 3) if isinstance(v, (int, float)) else '-'  # noqa: E731
        for i, w in enumerate(sel, 1):          # 排名=1..N (2026-10-24 用户裁定)
            fw = _factor_weights(w.get('factor_name'), str(sel_date))
            line = f"{i}. **{w.get('code', '?')}** 评分 {fmt(w.get('score'))}"
            adj_v = fmt(w.get('adjusted_score'))
            if adj_v != '-' and adj_v != fmt(w.get('score')):
                line += f" (调整 {adj_v})"
            line += f" | {w.get('industry', '')}"
            if fw is None and w.get('factor_name'):
                line += f" | {w['factor_name']}"   # 权重表没配上, 至少给出成分构成
            md.append(line)
            if fw:
                kind, pairs = fw
                md.append(f"   {kind}权重: "
                          + ' · '.join(f"{k} {v * 100:.0f}%" for k, v in pairs))
        md.append(f"\n> ⚠️ 信号日期 **{sel_date}** — 请自行判断陈旧度\n")
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
