#!/usr/bin/env python3
"""数据态指纹台账 (产品化, 2026-10-24, #319 收尾) — 把分散的数据态证据汇成一份.

来源(全部已落地, 本脚本只做汇总不新增判据):
  1. 锚链: rolling_validation_results/anchor.json prev_anchor 链 (重锚史)
  2. navlaw 放行记录: strategy/logs/navlaw_*.log 中的 REBASELINE/OVERRIDE
  3. 每日快照: data/snapshots/{daily,weekly}/ 目录清单 (供应商重述取证能力)
  4. 数据态指纹: .arms/datastate.py ls (全数据态 md5 指纹存档)

用法: python tools/data_state_ledger.py [--out docs/data_state_ledger.md]
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

BASE = Path('/mnt/d/quant')
ANCHOR = BASE / 'strategy' / 'rolling_validation_results' / 'anchor.json'
NAVLOGS = BASE / 'strategy' / 'logs'
SNAP = BASE / 'data' / 'snapshots'
DS = BASE / '.arms' / 'datastate.py'


def anchor_chain():
    try:
        a = json.loads(ANCHOR.read_text(encoding='utf-8'))
    except Exception as e:  # noqa: BLE001
        return [f'(anchor.json 读取失败: {e})']
    L = ['## 1. 锚链 (重锚史, 新→旧)', '', '| 锚 | 曲线末日 | NAV | 备注 |', '|---|---|---|---|']
    while a:
        name = str(a.get('anchor_name', '?')).replace('|', '/')
        L.append(f"| {name[:60]} | {a.get('date', '?')} | {float(a.get('nav', 0)):,.0f} | |")
        a = a.get('prev_anchor')
    return L


def navlaw_events():
    L = ['## 2. navlaw 放行记录 (重述/重锚显式登记)', '']
    hits = []
    for f in sorted(NAVLOGS.glob('navlaw_*.log')) if NAVLOGS.exists() else []:
        try:
            for ln in f.read_text(encoding='utf-8', errors='ignore').splitlines():
                if ('REBASELINE' in ln or 'OVERRIDE' in ln) and '<原因>' not in ln:
                    hits.append(f'- {f.name}: {ln.strip()[:140]}')
        except Exception:
            pass
    L += hits if hits else ['- 无 (自 2026-10-23 真闸启用以来零放行)']
    return L


def snapshot_inventory():
    L = ['## 3. 每日快照 (供应商重述取证能力)', '']
    for tier in ('daily', 'weekly'):
        d = SNAP / tier
        if not d.exists():
            L.append(f'- {tier}: (无)')
            continue
        days = sorted(x.name for x in d.iterdir() if x.is_dir() and x.name.isdigit())
        L.append(f'- {tier}: {len(days)} 份 ({days[0] if days else "-"} → '
                 f'{days[-1] if days else "-"})')
    return L


def datastate_inventory():
    L = ['## 4. 数据态指纹 (datastate snap 存档)', '']
    if not DS.exists():
        L.append('- (datastate.py 不存在)')
        return L
    try:
        r = subprocess.run([sys.executable, str(DS), 'ls'],
                           capture_output=True, text=True, timeout=120)
        lines = [ln for ln in (r.stdout or '').strip().splitlines() if ln.strip()]
        L.append(f'- 共 {len(lines)} 份指纹存档 (最近 10):')
        L += [f'  - {ln}' for ln in lines[-10:]]
    except Exception as e:  # noqa: BLE001
        L.append(f'- (datastate ls 失败: {e})')
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(BASE / 'docs' / 'data_state_ledger.md'))
    args = ap.parse_args()
    L = [f'# 数据态指纹台账 — 生成于 {time.strftime("%F %T")}', '',
         '> 数据态证据汇总: 锚链 + navlaw 放行 + 快照 + 指纹。改动数据层代码',
         '> 或发生供应商重述时, 必须在此留痕 (配 docs/disclosure.md 重述披露流程)。', '']
    L += anchor_chain() + [''] + navlaw_events() + [''] + snapshot_inventory() + [''] \
        + datastate_inventory()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text('\n'.join(L) + '\n', encoding='utf-8')
    print(f'[ledger] ✓ 已生成: {out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
