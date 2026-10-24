#!/usr/bin/env python3
"""数据每日快照 (P0, 2026-10-24) — 供应商重述三板斧的"回到过去"能力.

分级策略 (按体积×重述风险):
  每日: data/*.pkl 核心表 (~6.4M) → data/snapshots/daily/YYYYMMDD/, 留 30 份
  每周日: + data/alternative_data 全量 (~795M) → snapshots/weekly/YYYYMMDD/, 留 4 份
  backtrader_data (~829M): 不快照 — navlaw 体系 (每日存档+.bt_prev) 已覆盖历史重写检测
  指纹: 若 .arms/datastate.py 存在, 追加 datastate snap live_YYYYMMDD (全数据态秒级指纹)

用法: python snapshot_data.py [--weekly] [--retention-daily 30] [--retention-weekly 4]
接入: run_daily_live.sh 步骤 1.2/4 (refresh 之后, 信号之前; 失败不拦链但日志报警)。
判据: 供应商重述日, 用本目录按日期回到任意一天的"当日最终态"做 OHLCV 逐位对比。
"""
import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

BASE = Path('/mnt/d/quant')
DATA = BASE / 'data'
SNAP = DATA / 'snapshots'


def copy_top(src: Path, dst_dir: Path) -> int:
    """只复制 src 顶层 *.pkl, 不递归 — data/ 下含 snapshots/ 与子目录,
    递归会 (a) 把 alternative_data 795M 拖进每日层 (b) 明天复制今天的快照(自增长)。"""
    dst_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in sorted(src.iterdir()):
        if f.is_file() and f.suffix == '.pkl':
            shutil.copy2(f, dst_dir / f.name)
            n += 1
    return n


def copy_into(src: Path, dst_dir: Path) -> int:
    """递归复制 src 下全部 *.pkl (weekly 层 alternative_data 用)。"""
    dst_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in sorted(src.rglob('*')):
        if not f.is_file() or f.suffix != '.pkl':
            continue
        rel = f.relative_to(src)
        target = dst_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, target)
        n += 1
    return n


def prune(dirpath: Path, keep: int) -> int:
    """按目录名 YYYYMMDD 降序保留 keep 份。"""
    if not dirpath.exists():
        return 0
    days = sorted((d for d in dirpath.iterdir() if d.is_dir() and d.name.isdigit()),
                  reverse=True)
    for old in days[keep:]:
        shutil.rmtree(old, ignore_errors=True)
    return max(0, len(days) - keep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weekly', action='store_true', help='追加 weekly 层 (alternative_data)')
    ap.add_argument('--retention-daily', type=int, default=30)
    ap.add_argument('--retention-weekly', type=int, default=4)
    args = ap.parse_args()

    today = time.strftime('%Y%m%d')
    t0 = time.time()
    n = copy_top(DATA, SNAP / 'daily' / today)
    pruned = prune(SNAP / 'daily', args.retention_daily)
    print(f'[snapshot] daily: data/*.pkl {n} 个文件 → snapshots/daily/{today} '
          f'(prune {pruned})')

    nw = 0
    if args.weekly or time.strftime('%a') == 'Sun':
        nw = copy_into(DATA / 'alternative_data', SNAP / 'weekly' / today / 'alternative_data')
        pw = prune(SNAP / 'weekly', args.retention_weekly)
        print(f'[snapshot] weekly: alternative_data {nw} 个文件 → snapshots/weekly/{today} '
              f'(prune {pw})')

    ds = BASE / '.arms' / 'datastate.py'
    if ds.exists():
        try:
            r = subprocess.run([sys.executable, str(ds), 'snap', f'live_{today}'],
                               capture_output=True, text=True, timeout=300)
            print(f'[snapshot] datastate: rc={r.returncode} '
                  f'{r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ""}')
        except Exception as e:  # noqa: BLE001 — 指纹失败不拦快照
            print(f'[snapshot] ⚠ datastate snap 失败: {e}')

    print(f'[snapshot] ✓ 完成 {time.time() - t0:.1f}s')
    return 0


if __name__ == '__main__':
    sys.exit(main())
