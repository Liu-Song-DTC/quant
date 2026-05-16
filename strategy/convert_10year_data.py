"""
批量数据转换脚本: raw_data/qfq.csv → backtrader_data/_qfq.csv
- 多进程并行转换
- 按股票池过滤（从factor_preparer获取）
- 日期范围: 2016-01-01 到 2026-05-08
- 输出: backtrader标准格式
"""
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import shutil

# 路径配置
RAW_BASE = r"D:\quant\data\stock_data\raw_data"
BT_BASE = r"D:\quant\data\stock_data\backtrader_data"
OUT_DIR = BT_BASE  # 直接写入同一个目录

# 日期范围
DATE_START = "2016-01-01"
DATE_END = "2026-05-08"

# 转换参数
MAX_WORKERS = 6
BATCH_SIZE = 200  # 每批处理的股票数
SKIP_EXISTING = True  # 跳过已存在的文件（只转换缺失的）

# backtrader格式列顺序
BT_COLS = ['datetime', 'open', 'high', 'low', 'close', 'volume',
           'openinterest', 'amount', 'amplitude', 'change_percent',
           'change_amount', 'turnover_rate']


def get_existing_codes():
    """获取backtrader_data中已有完整数据的股票代码"""
    existing = set()
    if os.path.exists(OUT_DIR):
        for item in os.listdir(OUT_DIR):
            if item.endswith('_qfq.csv') and not item.startswith('._'):
                code = item[:-8]
                fpath = os.path.join(OUT_DIR, item)
                try:
                    df = pd.read_csv(fpath, nrows=5)
                    if len(df) >= 5:
                        existing.add(code)
                except Exception:
                    pass
    return existing


def convert_stock(args):
    """转换单只股票的前复权数据"""
    code, raw_path, out_path = args

    try:
        # 读取原始数据
        df = pd.read_csv(raw_path, dtype={'股票代码': str}, encoding='utf-8')

        # 重命名列
        col_map = {
            '日期': 'datetime',
            '股票代码': '_code',
            '开盘': 'open',
            '收盘': 'close',
            '最高': 'high',
            '最低': 'low',
            '成交量': 'volume',
            '成交额': 'amount',
            '振幅': 'amplitude',
            '涨跌幅': 'change_percent',
            '涨跌额': 'change_amount',
            '换手率': 'turnover_rate'
        }
        df = df.rename(columns=col_map)

        # 标准化股票代码（补齐6位）
        if '_code' in df.columns:
            df['_code'] = df['_code'].astype(str).str.zfill(6)

        # 日期筛选
        df['datetime'] = pd.to_datetime(df['datetime'])
        mask = (df['datetime'] >= DATE_START) & (df['datetime'] <= DATE_END)
        df = df[mask].copy()

        if len(df) < 50:  # 至少50个交易日
            return code, None, "数据不足"

        # 格式化日期
        df['datetime'] = df['datetime'].dt.strftime('%Y-%m-%d')

        # 添加openinterest（backtrader必需，设为0）
        df['openinterest'] = 0.0

        # 选择并排序列
        missing_cols = [c for c in BT_COLS if c not in df.columns]
        for c in missing_cols:
            df[c] = 0.0

        df = df[BT_COLS].copy()

        # 排序
        df = df.sort_values('datetime').reset_index(drop=True)

        # 去重（同日期保留最后一条）
        df = df.drop_duplicates(subset=['datetime'], keep='last')

        # 写入CSV（UTF-8无BOM，Windows Excel兼容）
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        df.to_csv(out_path, index=False, encoding='utf-8')

        return code, len(df), None

    except Exception as e:
        return code, None, str(e)


def main():
    print(f"=" * 60)
    print(f"批量数据转换: raw_data → backtrader_data")
    print(f"日期范围: {DATE_START} → {DATE_END}")
    print(f"输出目录: {OUT_DIR}")
    print(f"=" * 60)

    # 获取已有数据的股票（避免重复转换）
    existing = get_existing_codes()
    print(f"\n已有数据股票: {len(existing)} 只")

    # 扫描raw_data
    raw_dirs = [d for d in os.listdir(RAW_BASE)
                if os.path.isdir(os.path.join(RAW_BASE, d)) and not d.startswith('._')]

    print(f"raw_data 股票目录: {len(raw_dirs)} 只")

    # 准备转换任务
    tasks = []
    for code in raw_dirs:
        if SKIP_EXISTING and code in existing:
            continue

        raw_path = os.path.join(RAW_BASE, code, 'qfq.csv')
        out_path = os.path.join(OUT_DIR, f"{code}_qfq.csv")

        if os.path.exists(raw_path):
            tasks.append((code, raw_path, out_path))

    print(f"待转换: {len(tasks)} 只股票")
    if not tasks:
        print("没有需要转换的股票，跳过。")
        return

    # 分批处理
    total = len(tasks)
    done = 0
    errors = 0
    results_summary = {}

    for batch_start in range(0, total, BATCH_SIZE):
        batch = tasks[batch_start:batch_start + BATCH_SIZE]
        batch_num = batch_start // BATCH_SIZE + 1
        batch_total = (total + BATCH_SIZE - 1) // BATCH_SIZE

        print(f"\n--- 批次 {batch_num}/{batch_total} ({len(batch)} 只) ---")

        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(convert_stock, t): t[0] for t in batch}

            for future in tqdm(as_completed(futures), total=len(futures), desc="转换中"):
                code, n_rows, err = future.result()
                done += 1
                if err:
                    errors += 1
                    if "数据不足" not in str(err):
                        print(f"\n  错误 {code}: {err}")
                else:
                    results_summary[code] = n_rows

        print(f"  批次完成: {done}/{total}, 错误: {errors}")

    # 统计
    print(f"\n{'=' * 60}")
    print(f"转换完成!")
    print(f"成功: {len(results_summary)} 只")
    print(f"失败/跳过: {errors} 只")

    # 统计输出文件
    out_files = [f for f in os.listdir(OUT_DIR)
                 if f.endswith('_qfq.csv') and not f.startswith('._')]
    total_rows = 0
    for f in out_files[:100]:
        try:
            df = pd.read_csv(os.path.join(OUT_DIR, f), nrows=2)
            total_rows += 1
        except Exception:
            pass

    print(f"backtrader_data 文件总数: {len(out_files)}")
    print(f"前100个文件验证通过: {total_rows}/100")

    # 抽样检查
    sample_code = '000001'
    sample_path = os.path.join(OUT_DIR, f"{sample_code}_qfq.csv")
    if os.path.exists(sample_path):
        df = pd.read_csv(sample_path)
        print(f"\n样本检查 [{sample_code}]: {len(df)} 行")
        print(f"  日期范围: {df['datetime'].iloc[0]} → {df['datetime'].iloc[-1]}")
        print(f"  列: {list(df.columns)}")


if __name__ == '__main__':
    main()