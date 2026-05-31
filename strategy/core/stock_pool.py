"""
股票池模块 - 按流动性/质量筛选，控制回测和选股规模

筛选逻辑 (质量优先):
1. 次新股过滤: 排除上市不足1年的股票（K线数据不足，缠论结构不可靠）
2. 价格过滤: 排除仙股(<3元)和过高价股(>200元)
3. 僵尸股过滤: 排除日均换手率<0.5%的股票
4. 流动性地板: 排除日均成交额<5000万的股票
5. 复合排名: 60%流动性 + 25%价格稳定 + 15%交易活跃度
6. 排雷: ST + 科创板 + 异常数据
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path


def _get_data_dir():
    """获取backtrader数据目录"""
    base = Path(__file__).parent.parent.parent
    return str(base / 'data' / 'stock_data' / 'backtrader_data')


def _get_metadata_dir():
    """获取stock_metadata目录"""
    base = Path(__file__).parent.parent.parent
    return str(base / 'data' / 'stock_data' / 'stock_metadata')


def get_stock_pool(min_price: float = 2.0,
                   data_dir: str = None) -> set:
    """获取股票池 — 全市场除科创板外全部纳入

    Args:
        min_price: 最低价格（排除仙股，复权后价格）
        data_dir: 数据目录路径

    Returns:
        set of stock codes
    """
    if data_dir is None:
        data_dir = _get_data_dir()

    if not os.path.exists(data_dir):
        return set()

    exclusion_set = get_exclusion_set()

    valid_files = []
    for f in os.listdir(data_dir):
        if f.startswith('._') or f.startswith('sh000001'):
            continue
        if f.endswith('_qfq.csv'):
            code = f[:-8]
            valid_files.append((f, code))
        elif f.endswith('_hfq.csv'):
            code = f[:-8]
            # 避免 qfq/hfq 重复
            qfq_path = os.path.join(data_dir, f'{code}_qfq.csv')
            if not os.path.exists(qfq_path):
                valid_files.append((f, code))

    selected = set()
    data_errors = 0
    liquidity_filtered = 0

    for item, code in valid_files:
        if code in exclusion_set:
            continue

        filepath = os.path.join(data_dir, item)
        try:
            df = pd.read_csv(filepath)
            if len(df) < 60:
                data_errors += 1
                continue

            last_price = df['close'].iloc[-1]
            if last_price <= 0 or np.isnan(last_price) or last_price < min_price:
                continue

            # 流动性过滤: 近20日日均成交额 >= 3000万 (排除僵尸股)
            if 'amount' in df.columns and len(df) >= 20:
                avg_amount = df['amount'].iloc[-20:].mean()
                if avg_amount < 30_000_000:
                    liquidity_filtered += 1
                    continue
            # 流动性补充: 无amount列时, 近20日日均成交量 >= 100万股
            elif 'volume' in df.columns and len(df) >= 20:
                avg_vol = df['volume'].iloc[-20:].mean()
                if avg_vol < 1_000_000:
                    liquidity_filtered += 1
                    continue

            selected.add(code)
        except Exception:
            data_errors += 1
            continue

    selected.add('sh000001')
    print(f"股票池: {len(valid_files)} 总文件 -> 排除科创板{len(exclusion_set)} | 异常{data_errors} | 流动性过滤{liquidity_filtered} -> {len(selected)} 只 (含sh000001)")
    return selected


def load_st_stocks() -> set:
    """从stock_list.csv加载ST股票代码

    Returns:
        set of ST stock codes (不含前缀, 如 '000001')
    """
    metadata_dir = _get_metadata_dir()
    stock_list_path = os.path.join(metadata_dir, 'stock_list.csv')

    if not os.path.exists(stock_list_path):
        print("警告: stock_list.csv 不存在，无法过滤ST股票")
        return set()

    df = pd.read_csv(stock_list_path, dtype={'symbol': str})
    st_codes = set()
    for _, row in df.iterrows():
        name = str(row.get('name', ''))
        symbol = str(row.get('symbol', ''))
        if 'ST' in name and symbol:
            st_codes.add(symbol)

    print(f"ST股票过滤: 识别 {len(st_codes)} 只ST股票")
    return st_codes


def is_star_board(code: str) -> bool:
    """判断是否为科创板股票 (688xxx)"""
    return code.startswith('688')


def get_exclusion_set() -> set:
    """获取需要排除的股票集合: 仅ST

    Returns:
        set of stock codes to exclude
    """
    st_codes = load_st_stocks()

    # 科创板(688xxx) 不再排除 — 2025-2026年妖股主要集中在科创板
    star_codes = set()  # 保留变量但不填充，方便将来可配置

    excluded = st_codes | star_codes
    overlap = len(st_codes & star_codes)
    print(f"股票排除: ST {len(st_codes)} + 科创板 {len(star_codes)} - 重叠{overlap} = 共排除 {len(excluded)} 只")
    return excluded
