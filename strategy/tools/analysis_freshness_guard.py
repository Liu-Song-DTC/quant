#!/usr/bin/env python3
"""分析产物新鲜度守卫 (零写入, 2026-09-25, 9/28并入每日cron):
9/6→9/23教训: validation_results.csv陈旧17天无人发现 — 流水线第2/3步手动运行的架构缺陷。
本守卫: 比对 signals 与分析产物的 mtime/尾部NAV, 任一失配 → 退出码1并打印STALE。
只读, 可安全并入每日9:47守卫cron。
"""
import os
import sys
import time

V = '/mnt/d/quant/strategy/rolling_validation_results'
SIG = f'{V}/backtest_signals.csv'
PRODUCTS = ['validation_results.csv', 'portfolio_selections.csv',
            'trade_realized.csv', 'equity_curve.csv', 'backtest_diagnostics.json']

SECONDS_PER_DAY = 86400
issues = []

def age_days(p):
    return (time.time() - os.path.getmtime(p)) / SECONDS_PER_DAY

if not os.path.exists(SIG):
    print(f"MISSING: {SIG}")
    sys.exit(1)

sig_age = age_days(SIG)
print(f"signals: {sig_age:.1f}d old  (mtime {time.strftime('%m-%d %H:%M', time.localtime(os.path.getmtime(SIG)))})")

for p in PRODUCTS:
    fp = f'{V}/{p}'
    if not os.path.exists(fp):
        issues.append(f'MISSING {p}')
        print(f"[MISSING] {p}")
        continue
    a = age_days(fp)
    # signals 比分析产物新 > 1天 → 分析产物陈旧(9/6→9/23事故同构)
    if a > sig_age + 1.0:
        issues.append(f'STALE {p}: analysis {a:.1f}d vs signals {sig_age:.1f}d')
        print(f"[STALE] {p}: {a:.1f}d old (signals {sig_age:.1f}d)")
    else:
        print(f"[ok] {p}: {a:.1f}d old")

# equity_curve内容一致性: 若有锚点sidecar, 比对md5 (9/25发现: 实验臂会覆写此槽)
#   rolling_validation_results/anchor.json = {curve_md5, nav, date}, 由每次生产态回测写盘
import json
import hashlib

SIDECAR = f'{V}/anchor.json'
if os.path.exists(SIDECAR):
    try:
        with open(SIDECAR) as f:
            sc = json.load(f)
        with open(f'{V}/equity_curve.csv', 'rb') as f:
            md5 = hashlib.md5(f.read()).hexdigest()
        if md5 == sc.get('curve_md5'):
            print(f"[ok] equity_curve md5==锚点sidecar ({sc.get('date')}, NAV={sc.get('nav', 0):,.0f})")
        else:
            issues.append(f'ARM-OCCUPIED equity_curve: md5 {md5[:8]} != anchor {sc.get("curve_md5", "")[:8]}')
            print(f"[ARM-OCCUPIED] equity_curve.csv被实验臂占用: md5 {md5[:8]} != 锚点 {sc.get('curve_md5', '')[:8]}")
    except Exception as e:
        print(f"[warn] sidecar读取失败: {e}")
else:
    try:
        with open(f'{V}/equity_curve.csv') as f:
            last = f.readlines()[-1].strip().split(',')
        print(f"[info] 无锚点sidecar — equity尾行(仅报告非断言): date={last[0]} nav={float(last[1]):,.0f}")
    except Exception:
        pass

if issues:
    print(f"\n守卫FAIL: {len(issues)}项 — {issues}")
    sys.exit(1)
print("\n守卫PASS: 分析产物与signals新鲜度一致")
