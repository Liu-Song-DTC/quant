"""现状取证 (2026-10-28, P0-a): 当日应有上市名册 vs 本地实际文件。

只读: 取上游在册名册(经中继代理), 与生产 raw_data / backtrader_data 对账,
回答"现在到底缺几只股票"(计划 §5 P0-a: 原证据未证明实际漏股数量)。
用法: python data/probe_listing_gap_1028.py [--floor 3000]
"""
import argparse
import csv
import os
import sys
from pathlib import Path

PROD = Path('/mnt/d/quant')
sys.path.insert(0, str(Path(__file__).resolve().parent))
from listing_reconciliation import reconcile_listing  # noqa: E402

RAW = PROD / 'data/stock_data/raw_data'
BT = PROD / 'data/stock_data/backtrader_data'
META = PROD / 'data/stock_data/stock_metadata'


def fetch_listed():
    sys.path.insert(0, str(PROD / 'data'))
    import akshare_kline_downloader as dl
    dl.install_patch()
    return dl.get_listed_codes()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--floor', type=int, default=3000)
    ap.add_argument('--save', default=str(META / 'listed_codes_probe.csv'))
    args = ap.parse_args()

    listed = fetch_listed()
    raw = {d.name for d in RAW.iterdir() if d.is_dir()} if RAW.exists() else set()
    qfq = {f[:-8] for f in os.listdir(BT) if f.endswith('_qfq.csv')}
    # 上一次名册(若存在)作参考值
    ref = None
    prev = sorted(META.glob('listed_codes_*.csv'), reverse=True)
    if prev:
        with prev[0].open(encoding='utf-8') as fh:
            ref = sum(1 for r in csv.reader(fh) if r and r[0].strip())
    rep = reconcile_listing(listed, raw, qfq, floor=args.floor, reference_count=ref)
    print(rep.summary())
    if rep.roster_anomaly:
        print(f'  名册异常: {rep.roster_note}')
    if rep.missing_raw:
        print(f'  缺原始文件({len(rep.missing_raw)}): {rep.missing_raw[:30]}')
    if rep.missing_qfq:
        print(f'  缺QFQ({len(rep.missing_qfq)}): {rep.missing_qfq[:30]}')
    print(f'  名册外原始文件({len(rep.extra_raw)}): {rep.extra_raw[:20]}')
    print(f'  名册外QFQ({len(rep.extra_qfq)}): {rep.extra_qfq[:20]}')
    try:
        with open(args.save, 'w', encoding='utf-8') as fh:
            for c in sorted(listed):
                fh.write(f'{c}\n')
        print(f'  名册已存: {args.save}')
    except OSError as e:
        print(f'  名册保存失败(只读环境?): {e}')


if __name__ == '__main__':
    main()
