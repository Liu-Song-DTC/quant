#!/usr/bin/env python3
"""上市名册对账 CLI (2026-10-28, P0-a): 换名前/新鲜度门禁的缺股检查。

用法:
  python data/check_listing_reconciliation.py --roster <listed_codes.csv> \
      --raw <raw_data目录> --qfq <backtrader_data目录> [--allow <codes.txt>]

退出码:
  0 = 对账通过 (或名册不可信但已显式放行 --allow-roster-anomaly)
  2 = 对账失败 (有在册代码缺原始文件/缺QFQ, 或名册本身异常)

名册文件由 data/akshare_kline_downloader.py 落盘:
  stock_metadata/listed_codes_<MMDD>.csv  (第一列 code)
上一份名册用于"上游名单异常"判定(比上次少10%以上 ⇒ 判为上游异常, 不据此报缺股,
但也不允许据此换名 ⇒ 需人工处置)。
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from listing_reconciliation import reconcile_listing  # noqa: E402


def load_roster(path):
    """读名册文件 ⇒ (codes, 上一次名册的只数或None)。"""
    path = Path(path)
    with path.open(encoding='utf-8') as fh:
        codes = [row[0].strip() for row in csv.reader(fh) if row and row[0].strip()]
    reference = None
    # MMDD 名称跨年不代表先后，按文件实际生成时间选上一次名册。
    siblings = sorted((p for p in path.parent.glob('listed_codes_*.csv')
                       if p != path), key=lambda p: p.stat().st_mtime, reverse=True)
    for other in siblings:
        try:
            with other.open(encoding='utf-8') as fh:
                reference = sum(1 for row in csv.reader(fh) if row and row[0].strip())
        except OSError:
            reference = None
        break
    return codes, reference


def load_allowlist(path):
    """允许清单: 一行一码, 空格后为备注 (与 kline_stale_allowlist.txt 同格式)。"""
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        return []
    out = []
    for ln in p.read_text(encoding='utf-8').splitlines():
        ln = ln.strip()
        if not ln or ln.startswith('#'):
            continue
        out.append(ln.split()[0])
    return out


def raw_codes(path):
    """raw_data 目录: 每个子目录 = 一只股票的原始价。"""
    p = Path(path)
    if not p.exists() or not p.is_dir():
        return []
    return [d.name for d in p.iterdir() if d.is_dir()]


def qfq_codes(path):
    """backtrader_data 目录: 每个 *_qfq.csv 文件 = 一只股票的复权价。"""
    p = Path(path)
    if not p.exists():
        return []
    if p.is_dir():
        return [f.name[:-8] for f in p.iterdir()
                if f.is_file() and f.name.endswith('_qfq.csv')]
    return [p.name[:-8]] if p.name.endswith('_qfq.csv') else []


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--roster', required=True)
    ap.add_argument('--raw', required=True)
    ap.add_argument('--qfq', required=True)
    ap.add_argument('--allow', default='', help='允许缺席的代码(每行一个)')
    ap.add_argument('--allow-roster-anomaly', action='store_true',
                    help='名册异常时仍放行(需在日志留痕)')
    ap.add_argument('--report', default='', help='把JSON报告写到该路径')
    args = ap.parse_args()

    if not Path(args.roster).exists():
        print(f'[listing] 缺少名册文件 {args.roster} ⇒ 无法对账(拒绝放行)')
        return 2
    codes, reference = load_roster(args.roster)
    allow = load_allowlist(args.allow)
    report = reconcile_listing(
        codes,
        raw_codes(args.raw),
        qfq_codes(args.qfq),
        allowlist=allow,
        reference_count=reference)
    print(f'[listing] {report.summary()}')
    if report.missing_raw:
        print(f'[listing] 缺原始文件(前10): {report.missing_raw[:10]}')
    if report.missing_qfq:
        print(f'[listing] 缺QFQ(前10): {report.missing_qfq[:10]}')
    if report.extra_raw:
        print(f'[listing] 名册外原始文件(退市/隔离, 仅报告, 前5): {report.extra_raw[:5]}')
    if report.extra_qfq:
        print(f'[listing] 名册外QFQ(仅报告, 前5): {report.extra_qfq[:5]}')
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2),
                                     encoding='utf-8')
    if report.roster_anomaly:
        if args.allow_roster_anomaly:
            print(f'[listing] 名册异常但已显式放行: {report.roster_note}')
            return 0
        print(f'[listing] 上游名册异常({report.roster_note}) ⇒ 拒绝放行')
        return 2
    if not report.ok:
        print('[listing] 存在在册代码缺数据 ⇒ 拒绝放行(股票池会静默缩小)')
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
