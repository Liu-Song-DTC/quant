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

        if stock_name in ('sh000001', '399006', 'sh000852'):
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


def _refuse_if_production_bt(out_dir) -> None:
    """遗留链路守卫 (2026-10-18) —— 与 data/data_manager.py 同款, 但这里**此前完全没有**。

    ⚠ 本脚本直通东财**加性**前复权(raw_data/*/qfq.csv), 而生产 `backtrader_data/`
    是 `data/rebuild_table_qfq.py` 的**表驱动乘性**产物。两者写同一批文件, 跑一次
    就把乘性整体覆盖成加性(且 raw 的 qfq.csv 一直在被 akshare 下载器更新 ⇒ 不是陈旧
    而是"另一种错"), 非除权日不保收益、事件日整条重锚 ⇒ 违反铁律。
    见 memory: adjustment-method-affine-20261016 / artifact-path-dependence-20261017。

    判据 = "目录是表驱动生产库", 用**两个标记任一存在**:
      · qfq_anchor.json      —— 冻结合约(业务文件, 迁移中可能被正当删除)
      · .table_driven_prod   —— **专用哨兵**, 只由 refresh_all.sh 换名后 touch
    单用前者有静默失效风险(3638f9d 当天就删过一次锚文件 ⇒ 守卫退化成 no-op)。

    可达路径: `strategy/run_live.py:48`(README 记载的实盘命令, dashboard_server.py
    可触发)与 `strategy/run_live_daily.sh:18` —— 点一下就把数据层毁掉。
    逃生门: QUANT_ALLOW_LEGACY_BT=1 (仅限一次性对比/临时目录)。
    """
    d = os.path.dirname(os.path.abspath(str(out_dir)))
    _marks = [os.path.join(d, 'qfq_anchor.json'),
              os.path.join(d, '.table_driven_prod')]
    _hit = [m for m in _marks if os.path.exists(m)]
    if _hit and os.environ.get('QUANT_ALLOW_LEGACY_BT') != '1':
        raise SystemExit(
            f"[遗留链路守卫] {d} 是表驱动乘性复权生产库 (存在 {[os.path.basename(m) for m in _hit]}),\n"
            f"  拒绝用旧**加性**链路 convert_raw_to_backtrader.py 覆盖 backtrader_data/。\n"
            f"  生产每日链路: bash strategy/run_daily_live.sh (内含 data/refresh_all.sh)\n"
            f"  只重算 bt 产物: python data/rebuild_table_qfq.py --events "
            f"data/alternative_data/events_table.pkl --anchor frozen --out <临时目录> (再原子换名)\n"
            f"  确要跑遗留链路(仅限一次性对比/临时目录): QUANT_ALLOW_LEGACY_BT=1")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start-year', type=int, default=START_YEAR)
    parser.add_argument('--end-year', type=int, default=END_YEAR)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()

    start_year = args.start_year
    end_year = args.end_year

    _refuse_if_production_bt(OUTPUT_DIR)   # ← 必须在任何写盘之前
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

    import platform; ctx = multiprocessing.get_context('fork' if platform.system() != 'Windows' else 'spawn')
    with ctx.Pool(args.workers) as pool:
        results = list(tqdm(
            pool.imap(convert_stock, work_args, chunksize=50),
            total=len(work_args),
            desc="Converting"
        ))

    for name, ok, data in results:
        if ok:
            output_file = OUTPUT_DIR / f'{name}_qfq.csv'
            # 2026-10-16: 原子写(tmp + os.replace) — 数据冻结快照层前置。
            # 原地 to_csv 有两个问题: ①硬链接快照与该文件共享 inode, 原地写会
            # **静默污染历史快照**(快照不再是当时的数据态); ②中途被杀留下截断
            # 的 CSV, 回测读到残缺数据(旧代码无任何校验)。
            # os.replace 在同一文件系统内是原子的 ⇒ 快照的旧 inode 不受影响。
            _tmp = output_file.with_name(output_file.name + '.tmp')
            data.to_csv(_tmp, index=False)
            os.replace(_tmp, output_file)
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
