#!/usr/bin/env python
"""实盘净值台账 — **只追加 + 防篡改哈希链** (2026-10-16)

动机(用户 2026-10-16 "我要把当前系统做成产品 / 给别人订阅"):
  三层分离 — ① 研究回测(±30% 数据态漂移, **永不对外**)
             ② 风险画像 + 选股α(对数据态免疫, 可对外)
             ③ **实盘净值(append-only 冻结, 主打)**
  ① 每次数据刷新全史重写(重述), 所以它不能当对外口径; ③ 必须**写一次就冻结** —
  否则订阅者看到的曲线第二天就变了, 产品信任立刻归零。

设计要点:
  · **只追加**: 同一日期重复记账被拒(除非走 amend), 原始行永不修改/删除。
  · **哈希链**: row_hash = md5(seq|date|kind|nav|note|prev_hash)。改任何历史行
    ⇒ 其后所有 row_hash 失配 ⇒ `verify` 立刻报出**第一处**断链。订阅者可自行验。
  · **重述单独披露**: 更正走 `amend`, 原值保留, 新旧并列 ⇒ "当时报的" 与
    "现在认的" 同时可见(对应 anchor.json 的重述披露机制)。
  · 与 `.arms/datastate.py` 的**回测** ledger.csv 并列: 那个记回测(会重述),
    这个记实盘(不重述)。**对外只发这个。**

用法:
  live_nav.py append <date> <nav> [note...]      # 记一天 (YYYY-MM-DD, 拒绝重复)
  live_nav.py amend  <date> <nav> <reason...>    # 更正(追加一行, 原值保留)
  live_nav.py mark   <date>                      # 交叉核对: 由本地持仓+收盘价算市值
  live_nav.py verify                             # 校验哈希链
  live_nav.py series                             # 有效序列(每日取最新一行)
  live_nav.py stats                              # 累计/年化/最大回撤/Sharpe
"""
import sys, os, json, csv, hashlib, datetime, math

ROOT = '/mnt/d/quant'
# 台账路径可用 env 覆盖 — 只为烟测/多账户预留, 生产默认即下值
STORE = os.environ.get('QUANT_LIVE_NAV') or '/mnt/d/quant/.arms/live_nav.csv'
POSITIONS = '/mnt/d/quant/current_positions.json'
BT_DATA = '/mnt/d/quant/data/stock_data/backtrader_data'
HEADER = ['seq', 'date', 'kind', 'nav', 'note', 'prev_hash', 'row_hash']


def _row_hash(seq, date, kind, nav, note, prev_hash):
    return hashlib.md5(f'{seq}|{date}|{kind}|{nav}|{note}|{prev_hash}'.encode()).hexdigest()[:16]


def _load():
    if not os.path.exists(STORE):
        return []
    with open(STORE, newline='') as f:
        rd = csv.DictReader(f)
        return [dict(r) for r in rd]


def _append_row(date, kind, nav, note):
    rows = _load()
    for r in rows:                      # 链校验: 追加前先确认既有历史未被篡改
        exp = _row_hash(r['seq'], r['date'], r['kind'], r['nav'], r['note'], r['prev_hash'])
        if exp != r['row_hash']:
            print(f"❌ 既有台账已损坏 (seq={r['seq']} {r['date']} row_hash 失配) — 拒绝追加")
            print("   先跑 live_nav.py verify 定位; 台账损坏时**不得**继续写入。")
            return False
    seq = (int(rows[-1]['seq']) + 1) if rows else 1
    prev = rows[-1]['row_hash'] if rows else 'GENESIS'
    rh = _row_hash(seq, date, kind, nav, note, prev)
    new = not rows
    with open(STORE, 'a', newline='') as f:
        w = csv.writer(f)
        if new:
            w.writerow(HEADER)
        w.writerow([seq, date, kind, nav, note, prev, rh])
    return True


def _valid_date(s):
    try:
        datetime.date.fromisoformat(s)
        return True
    except ValueError:
        return False


def cmd_append(date, nav, note):
    if not _valid_date(date):
        print(f'❌ 日期格式应为 YYYY-MM-DD: {date}')
        return 1
    try:
        nav = float(nav)
    except ValueError:
        print(f'❌ 净值应为数字: {nav}')
        return 1
    if nav <= 0:
        print(f'❌ 净值必须为正: {nav}')
        return 1
    rows = _load()
    same = [r for r in rows if r['date'] == date]
    if same:
        cur = same[-1]['nav']
        if float(cur) == nav:
            print(f'ℹ️  {date} 已记为 {cur} (同值) — 幂等跳过')
            return 0
        print(f'❌ {date} 已有记录 nav={cur} (kind={same[-1]["kind"]})')
        print(f'   台账**只追加**: 更正请用 amend (原值保留, 重述单独披露)')
        return 1
    if not _append_row(date, 'nav', nav, note):
        return 1
    print(f'✅ 追加 {date} nav={nav}' + (f'  ({note})' if note else ''))
    return 0


def cmd_amend(date, nav, reason):
    if not _valid_date(date):
        print(f'❌ 日期格式应为 YYYY-MM-DD: {date}')
        return 1
    try:
        nav = float(nav)
    except ValueError:
        print(f'❌ 净值应为数字: {nav}')
        return 1
    rows = _load()
    same = [r for r in rows if r['date'] == date]
    if not same:
        print(f'❌ {date} 尚无原始记录 — 用 append 而不是 amend')
        return 1
    old = same[-1]['nav']
    if not reason.strip():
        print('❌ amend 必须给出原因 (重述要单独披露)')
        return 1
    if not _append_row(date, 'amend', nav, reason):
        return 1
    print(f'✅ 重述 {date}: {old} → {nav}  ({reason})')
    return 0


def cmd_verify():
    rows = _load()
    if not rows:
        print('(台账为空)')
        return 0
    prev = 'GENESIS'
    for i, r in enumerate(rows, 1):
        if int(r['seq']) != i:
            print(f'❌ 第 {i} 行 seq={r["seq"]} 不连续')
            return 1
        if r['prev_hash'] != prev:
            print(f'❌ 第 {i} 行 ({r["date"]}) prev_hash 断链')
            return 1
        exp = _row_hash(r['seq'], r['date'], r['kind'], r['nav'], r['note'], r['prev_hash'])
        if exp != r['row_hash']:
            print(f'❌ 第 {i} 行 ({r["date"]}) row_hash 失配 — 该行被改过')
            return 1
        prev = r['row_hash']
    print(f'✅ 哈希链完整: {len(rows)} 行, 无篡改痕迹')
    print(f'   链头 {rows[0]["row_hash"]} → 链尾 {rows[-1]["row_hash"]}')
    return 0


def _effective():
    """有效序列: 每日取最后一行(nav 或 amend)"""
    out = {}
    for r in _load():
        out[r['date']] = r
    return [out[d] for d in sorted(out)]


def cmd_series():
    eff = _effective()
    if not eff:
        print('(台账为空)')
        return 0
    print(f'{"日期":<12}{"净值":>14}{"日收益":>10}{"累计":>10}  备注')
    base = float(eff[0]['nav'])
    prev = None
    for r in eff:
        v = float(r['nav'])
        d = (v / prev - 1) * 100 if prev else 0.0
        print(f'{r["date"]:<12}{v:>14,.0f}{d:>9.2f}%{(v/base-1)*100:>9.2f}%  '
              f'{"[重述] " if r["kind"] == "amend" else ""}{r["note"]}')
        prev = v
    return 0


def cmd_stats():
    eff = _effective()
    if len(eff) < 2:
        print(f'(样本不足: {len(eff)} 天)')
        return 0
    vals = [float(r['nav']) for r in eff]
    rets = [vals[i] / vals[i - 1] - 1 for i in range(1, len(vals))]
    n = len(rets)
    cum = vals[-1] / vals[0] - 1
    years = n / 252.0
    cagr = (1 + cum) ** (1 / years) - 1 if years > 0 else 0.0
    mu = sum(rets) / n
    sd = math.sqrt(sum((x - mu) ** 2 for x in rets) / n) if n > 1 else 0.0
    sharpe = (mu / sd * math.sqrt(252)) if sd > 0 else 0.0
    peak, mdd = vals[0], 0.0
    for v in vals:
        peak = max(peak, v)
        mdd = min(mdd, v / peak - 1)
    print(f'  区间      {eff[0]["date"]} → {eff[-1]["date"]}  ({n} 个净值点)')
    print(f'  累计收益  {cum*100:+.2f}%')
    print(f'  年化      {cagr*100:+.2f}%')
    print(f'  Sharpe    {sharpe:.4f}')
    print(f'  最大回撤  {mdd*100:.2f}%')
    return 0


def cmd_mark(date):
    """交叉核对: 由 current_positions.json + 本地收盘价算持仓市值(不含现金)。
    现金余额系统不跟踪(手工账户) ⇒ 这里只给**市值**, 供与券商总资产对账。"""
    if not os.path.exists(POSITIONS):
        print(f'❌ 无持仓文件 {POSITIONS}')
        return 1
    pos = json.load(open(POSITIONS))
    if not pos:
        print('  (空仓 — 持仓市值 0; 净值应≈现金)')
        return 0
    tot_val = tot_cost = 0.0
    print(f'{"代码":<9}{"股数":>7}{"成本价":>9}{"收盘":>9}{"市值":>11}{"盈亏":>10}')
    for code, p in sorted(pos.items()):
        fp = os.path.join(BT_DATA, f'{code}_qfq.csv')
        px = None
        if os.path.exists(fp):
            try:
                import pandas as pd
                df = pd.read_csv(fp, usecols=['datetime', 'close'])
                df = df[df['datetime'].astype(str) <= date]
                if len(df):
                    px = float(df['close'].iloc[-1])
            except Exception:
                px = None
        shares = float(p.get('shares', 0))
        cost = float(p.get('entry_price', 0))
        val = shares * px if px is not None else float(p.get('amount', 0))
        tot_val += val
        tot_cost += shares * cost
        print(f'{code:<9}{shares:>7.0f}{cost:>9.2f}'
              f'{(f"{px:.2f}" if px is not None else "NA"):>9}'
              f'{val:>11,.0f}{val - shares*cost:>+10,.0f}')
    print(f'  {"合计":<9}{"":>7}{"":>9}{"":>9}{tot_val:>11,.0f}{tot_val-tot_cost:>+10,.0f}')
    print(f'  持仓成本 {tot_cost:,.0f} | 持仓市值 {tot_val:,.0f} (截止 {date})')
    print(f'  ⚠ 不含现金 — 净值 = 市值 + 现金余额(需人工从券商App读取)')
    return 0


if __name__ == '__main__':
    a = sys.argv[1:]
    if not a:
        print(__doc__)
        sys.exit(0)
    c = a[0]
    if c == 'append' and len(a) >= 3:
        sys.exit(cmd_append(a[1], a[2], ' '.join(a[3:])))
    elif c == 'amend' and len(a) >= 3:
        sys.exit(cmd_amend(a[1], a[2], ' '.join(a[3:])))
    elif c == 'verify':
        sys.exit(cmd_verify())
    elif c == 'series':
        sys.exit(cmd_series())
    elif c == 'stats':
        sys.exit(cmd_stats())
    elif c == 'mark' and len(a) >= 2:
        sys.exit(cmd_mark(a[1]))
    else:
        print(__doc__)
        sys.exit(2)
