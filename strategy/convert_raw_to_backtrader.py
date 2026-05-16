#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
将 raw_data 转换为 backtrader_data 格式，10年期（2014-2026）
"""
import os
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import multiprocessing
import argparse

# 路径配置 (相对于 strategy/ 目录)
SCRIPT_DIR = Path(__file__).parent
RAW_DATA_DIR = SCRIPT_DIR.parent / 'data' / 'stock_data' / 'raw_data'
OUTPUT_DIR = SCRIPT_DIR.parent / 'data' / 'stock_data' / 'backtrader_data'

START_YEAR = 2014
END_YEAR = 2026


def convert_stock(args):
    """Convert single stock raw data to backtrader format"""
    stock_dir, start_year, end_year = args
    stock_name = stock_dir.name

    qfq_file = stock_dir / 'qfq.csv'
    if not qfq_file.exists():
        return (stock_name, False, 'no qfq.csv')

    try:
        df = pd.read_csv(qfq_file)

        if stock_name == 'sh000001':
            # Index format: date,open,close,high,low,volume,amount
            df = df.rename(columns={'date': 'datetime'})
            cols_avail = set(df.columns)
        else:
            # Regular stock: 日期,股票代码,开盘,收盘,最高,最低,成交量,成交额,振幅,涨跌幅,涨跌额,换手率
            df = df.rename(columns={
                '日期': 'datetime',
                '开盘': 'open',
                '最高': 'high',
                '最低': 'low',
                '收盘': 'close',
                '成交量': 'volume',
                '成交额': 'amount',
                '振幅': 'amplitude',
                '涨跌幅': 'change_percent',
                '涨跌额': 'change_amount',
                '换手率': 'turnover_rate',
            })
            cols_avail = set(df.columns)

        # Add openinterest
        df['openinterest'] = 0.0

        # Parse dates and filter
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df[(df['datetime'].dt.year >= start_year) &
                (df['datetime'].dt.year <= end_year)]

        if len(df) == 0:
            return (stock_name, False, 'no data in range')

        # Ensure required columns exist
        required_cols = ['datetime', 'open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in df.columns:
                # Copy from close as fallback
                if col in ['open', 'high', 'low']:
                    df[col] = df.get('close', 0)
                elif col == 'volume':
                    df[col] = 0

        # Standard column order
        columns_order = ['datetime', 'open', 'high', 'low', 'close',
                        'volume', 'openinterest', 'amount', 'amplitude',
                        'change_percent', 'change_amount', 'turnover_rate']

        # Keep only columns that exist
        columns_order = [c for c in columns_order if c in df.columns]
        df = df[columns_order].sort_values('datetime')

        return (stock_name, True, df)

    except Exception as e:
        return (stock_name, False, str(e))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start-year', type=int, default=START_YEAR)
    parser.add_argument('--end-year', type=int, default=END_YEAR)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()

    start_year = args.start_year
    end_year = args.end_year

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Get all stock directories
    stock_dirs = sorted([d for d in RAW_DATA_DIR.iterdir() if d.is_dir()])
    print(f"Found {len(stock_dirs)} stock directories")
    print(f"Date range: {start_year}-{end_year}")
    print(f"Output: {OUTPUT_DIR}")

    # Prepare args
    work_args = [(d, start_year, end_year) for d in stock_dirs]

    success = 0
    skipped = 0
    failed = 0

    ctx = multiprocessing.get_context('fork')
    with ctx.Pool(args.workers) as pool:
        results = list(tqdm(
            pool.imap(convert_stock, work_args, chunksize=50),
            total=len(work_args),
            desc="Converting"
        ))

    for name, ok, data in results:
        if ok:
            output_file = OUTPUT_DIR / f'{name}_qfq.csv'
            data.to_csv(output_file, index=False)
            success += 1
        elif 'no data in range' in str(data):
            skipped += 1
        else:
            failed += 1
            if failed <= 5:
                print(f"  Failed: {name} - {data}")

    print(f"\nDone: {success} converted, {skipped} no data, {failed} errors")
    print(f"Output: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
