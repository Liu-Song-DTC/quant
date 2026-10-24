#!/usr/bin/env python3
"""生产目录白名单审计 (产品化, 2026-10-24) — #364 臂产物污染事故的制度化.

判据: 生产目录里允许出现 = git 跟踪的文件 + 生产运行时产物白名单 (gitignore 的)。
git ls-files --others --exclude-standard 列出"未跟踪且未被忽略"的文件 —
臂产物若写进生产目录, 就会以这个形态出现。报告不拦链, 但每次出单链都跑一遍。

审计范围: strategy/ 根 + strategy/rolling_validation_results/ + data/ 根。
用法: python tools/prod_isolation.py [--move]  (--move 只打印 mv 建议, 不执行)
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

BASE = Path('/mnt/d/quant')

# 生产运行时产物 (gitignore 或尚未产生, 出现=正常)
RUNTIME_OK = {
    'live_account.json', 'trade_orders.json', 'current_positions.json',
    'live_portfolio_state.json', 'live_rebuild_state.json', 'secrets.env',
    'backtest_report.json.livestate',
}

SCOPES = [
    BASE / 'strategy',
    BASE / 'strategy' / 'rolling_validation_results',
    BASE / 'data',
]


def untracked(scope: Path, depth_limited: bool) -> list:
    """未跟踪且未被 gitignore 的文件 (相对 BASE)。"""
    args = ['git', '-C', str(BASE), 'ls-files', '--others', '--exclude-standard',
            '--', str(scope.relative_to(BASE))]
    try:
        out = subprocess.run(args, capture_output=True, text=True, timeout=120).stdout
    except Exception as e:  # noqa: BLE001
        return [f'(git 失败: {e})']
    rels = [ln.strip() for ln in out.splitlines() if ln.strip()]
    if depth_limited:  # strategy/ 根只审一层, 子目录(.arms 等研究区)不审
        rels = [r for r in rels
                if '/' not in r[len('strategy/'):]]
    return rels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--move', action='store_true', help='附 mv 建议 (移入 .arms/)')
    args = ap.parse_args()

    L = [f'# 生产目录白名单审计 — {time.strftime("%F %T")}', '',
         '> 判据: 生产目录出现"未跟踪且未被忽略"的文件 = 研究产物误留 (#364 形态)。', '']
    n_bad = 0
    for scope, depth_limited in [(SCOPES[0], True), (SCOPES[1], False), (SCOPES[2], True)]:
        files = untracked(scope, depth_limited)
        bad = []
        for r in files:
            name = Path(r).name
            if name in RUNTIME_OK:
                continue
            bad.append(r)
        L.append(f'## {scope.relative_to(BASE)}/')
        if not bad:
            L.append('- ✓ 干净')
        else:
            n_bad += len(bad)
            for r in bad:
                sug = f'  → mv /mnt/d/quant/{r} /mnt/d/quant/.arms/ARMLEFTOVER/' \
                    if args.move else ''
                L.append(f'- ⚠ {r}{sug}')
        L.append('')

    L += ['## 白名单 (生产运行时产物, 出现=正常)', '']
    L += [f'- {x}' for x in sorted(RUNTIME_OK)]

    out = BASE / 'logs' / f'prod_isolation_{time.strftime("%Y%m%d")}.md'
    out.write_text('\n'.join(L) + '\n', encoding='utf-8')
    print(f'[isolation] 审计完成: {n_bad} 个未跟踪文件在生产目录 → {out}')
    if n_bad:
        print('[isolation] 建议: 研究产物移入 .arms/ (armout/ 或 ARMLEFTOVER/), 生产目录只留链产物')
    return 0


if __name__ == '__main__':
    sys.exit(main())
