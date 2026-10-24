#!/usr/bin/env python3
"""风险画像报告 (产品化, 2026-10-24) — 对外可交付层的唯一数字来源.

铁律 (用户10/16三层裁定): 只报**实盘净值** (live_account.json 台账);
不报回测累计收益, 不报"优化前后"绝对值, 重述单独披露。
实盘未启动或台账不足 10 天 → 明确拒绝出数字 (不出回测替代品)。

用法: python risk_profile.py [--out strategy/reports]
输出: reports/risk_profile_YYYYMMDD.md
"""
import argparse
import json
import sys
import time
from pathlib import Path

BASE = Path('/mnt/d/quant')
STRAT = BASE / 'strategy'
ACCOUNT = STRAT / 'live_account.json'
ANCHOR = STRAT / 'rolling_validation_results' / 'anchor.json'
LOGS = STRAT / 'logs'
MIN_DAYS = 10


def load_ledger():
    if not ACCOUNT.exists():
        return None
    acct = json.loads(ACCOUNT.read_text(encoding='utf-8'))
    h = acct.get('nav_history') or []
    return acct, sorted(h, key=lambda r: str(r['date']))


def year_rows(h):
    out = {}
    prev_nav = None
    for r in h:
        y = str(r['date'])[:4]
        nav = float(r['nav'])
        base = prev_nav if prev_nav is not None else nav
        out.setdefault(y, {'ret': 1.0, 'days': 0, 'peak': nav, 'mdd': 0.0})
        if base > 0:
            out[y]['ret'] *= nav / base
        out[y]['days'] += 1
        out[y]['peak'] = max(out[y]['peak'], nav)
        out[y]['mdd'] = min(out[y]['mdd'], nav / out[y]['peak'] - 1)
        prev_nav = nav
    return out


def monthly_rows(h):
    out = {}
    prev = None
    for r in h:
        m = str(r['date'])[:7]
        nav = float(r['nav'])
        if prev is not None and prev > 0:
            out[m] = out.get(m, 1.0) * nav / prev
        prev = nav
    return out


def overall(h):
    navs = [float(r['nav']) for r in h]
    peak, mdd, peak_d, trough_d = navs[0], 0.0, h[0]['date'], h[0]['date']
    cur_peak_d = h[0]['date']
    rets = []
    for i in range(1, len(navs)):
        rets.append(navs[i] / navs[i - 1] - 1)
        if navs[i] > peak:
            peak, cur_peak_d = navs[i], h[i]['date']
        dd = navs[i] / peak - 1
        if dd < mdd:
            mdd, peak_d, trough_d = dd, cur_peak_d, h[i]['date']
    days = len(rets)
    total = navs[-1] / navs[0] - 1 if navs[0] > 0 else 0.0
    import math
    vol = (sum((x - sum(rets) / days) ** 2 for x in rets) / max(days - 1, 1)) ** .5 if days else 0.0
    yrs = max(days / 244.0, 1e-9)
    ann = (1 + total) ** (1 / yrs) - 1 if total > -1 else -1.0
    sharpe = (sum(rets) / days) / vol * (244 ** .5) if vol > 0 else 0.0
    worst = min(rets) if rets else 0.0
    return {'total': total, 'ann': ann, 'sharpe': sharpe, 'mdd': mdd,
            'mdd_peak': peak_d, 'mdd_trough': trough_d, 'days': days,
            'vol': vol, 'worst_day': worst, 'daily_win': sum(x > 0 for x in rets) / max(days, 1)}


def restatement_section():
    """重述单独披露: 锚链 + navlaw 放行记录。"""
    lines = ['', '## 重述披露', '']
    try:
        a = json.loads(ANCHOR.read_text(encoding='utf-8'))
        chain = []
        while a:
            chain.append(a)
            a = a.get('prev_anchor')
        lines.append('| 锚 | 日期 | NAV |')
        lines.append('|---|---|---|')
        for c in chain:
            lines.append(f"| {c.get('anchor_name', '?')[:48]} | {c.get('date', '?')} "
                         f"| {float(c.get('nav', 0)):,.0f} |")
    except Exception as e:  # noqa: BLE001
        lines.append(f'(锚链读取失败: {e})')
    reb = sorted(LOGS.glob('navlaw_*.log')) if LOGS.exists() else []
    hits = []
    for f in reb:
        try:
            for ln in f.read_text(encoding='utf-8', errors='ignore').splitlines():
                if ('REBASELINE' in ln or 'OVERRIDE' in ln) and '<原因>' not in ln:
                    hits.append(f'{f.name}: {ln.strip()[:120]}')
        except Exception:
            pass
    lines.append('')
    if hits:
        lines.append('navlaw 显式放行记录 (REBASELINE/OVERRIDE):')
        lines += [f'- {x}' for x in hits[-20:]]
    else:
        lines.append('navlaw 显式放行记录: 无')
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(STRAT / 'reports'))
    args = ap.parse_args()

    got = load_ledger()
    if got is None:
        print('[risk_profile] live_account.json 不存在: 实盘未启动 (2026-10-26 起)。'
              '风险画像只报实盘净值, 不出回测替代数字 — 拒绝生成。')
        return 0
    acct, h = got
    if len(h) < MIN_DAYS:
        print(f'[risk_profile] 台账仅 {len(h)} 天 (<{MIN_DAYS}): 样本不足, 拒绝生成 '
              f'(避免短窗数字被单日主导)。')
        return 0

    o = overall(h)
    L = [f'# 风险画像 (实盘台账) — 截至 {h[-1]["date"]}', '',
         f'- 台账天数: **{o["days"]}** (起始 {h[0]["date"]}, 初始资金见台账)',
         f'- 最新 NAV: **{float(h[-1]["nav"]):,.0f}** (现金 {float(h[-1].get("cash", 0)):,.0f} '
         f'/ 持仓 {float(h[-1].get("pos_value", 0)):,.0f})',
         f'- 年化: {o["ann"] * 100:.2f}% | Sharpe: {o["sharpe"]:.3f} | '
         f'波动: {o["vol"] * 100:.2f}%/日',
         f'- MDD: **{o["mdd"] * 100:.2f}%** (谷 {o["mdd_trough"]}, 峰 {o["mdd_peak"]})',
         f'- 最差单日: {o["worst_day"] * 100:.2f}% | 日胜率: {o["daily_win"] * 100:.1f}%',
         '', '> 数字口径: 实盘台账净值 (live_account.json), 非回测曲线。', '']

    L += ['## 分年', '', '| 年 | 收益 | 天数 | 年内MDD |', '|---|---|---|---|']
    for y, r in year_rows(h).items():
        L.append(f'| {y} | {(r["ret"] - 1) * 100:+.2f}% | {r["days"]} | {r["mdd"] * 100:.2f}% |')

    mon = monthly_rows(h)
    L += ['', '## 分月', '', '| 月 | 收益 |', '|---|---|']
    for m, v in mon.items():
        L.append(f'| {m} | {(v - 1) * 100:+.2f}% |')

    L += ['', '## 敞口路径 (持仓市值/NAV, 月末)', '', '| 月 | 末敞口 | 峰 | 谷 |', '|---|---|---|---|']
    exp = {}
    for r in h:
        nav = float(r['nav'])
        if nav > 0:
            e = float(r.get('pos_value', 0)) / nav
            exp.setdefault(str(r['date'])[:7], []).append(e)
    for m in sorted(exp):
        es = exp[m]
        L.append(f'| {m} | {es[-1] * 100:.1f}% | {max(es) * 100:.1f}% | {min(es) * 100:.1f}% |')

    L += restatement_section()
    L += ['', '## 已知限制', '', '见 `docs/disclosure.md` (PIT/概念层/ST/重述流程), 对外材料必须附此节。']

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f'risk_profile_{time.strftime("%Y%m%d")}.md'
    out.write_text('\n'.join(L) + '\n', encoding='utf-8')
    print(f'[risk_profile] ✓ 已生成: {out} ({len(h)} 个台账日)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
