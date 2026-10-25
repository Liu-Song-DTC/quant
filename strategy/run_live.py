#!/usr/bin/env python3
"""Compatibility entry point for the single supported live decision chain."""
import argparse
from datetime import date
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def generate_orders(target_date, cash=500000.0, dry_run=False):
    command = [sys.executable, str(ROOT / 'strategy/generate_trade_orders.py'),
               '--date', str(target_date), '--cash', str(cash)]
    if dry_run:
        command.append('--dry-run')
    return subprocess.run(command, cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--date', default=date.today().isoformat())
    parser.add_argument('--cash', type=float, default=500000.0)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--skip-data', action='store_true', help='兼容旧参数；正式运行仍执行刷新与门禁')
    parser.add_argument('--skip-signals', action='store_true', help='仅可与 --dry-run 一起预览已有信号')
    args = parser.parse_args()
    if args.dry_run:
        return generate_orders(args.date, args.cash, True)
    if args.skip_signals:
        parser.error('正式出单必须使用完整链；已有信号预览请加 --dry-run')
    if args.date != date.today().isoformat() or args.cash != 500000.0:
        parser.error('完整链按当天实际账户运行；历史/自定义初始资金预览请加 --dry-run')
    return subprocess.run(['bash', str(ROOT / 'strategy/run_daily_live.sh')], cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
