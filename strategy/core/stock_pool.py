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


def _load_market_cap_whitelist():
    """加载市值白名单（>=500亿总市值），用于加速回测"""
    import os as _os
    base = Path(__file__).parent.parent
    whitelist_path = str(base / 'config' / 'large_cap_whitelist.txt')
    if not _os.path.exists(whitelist_path):
        return None  # 无白名单文件 → 不过滤
    with open(whitelist_path, 'r') as f:
        codes = {line.strip() for line in f if line.strip()}
    return codes if codes else None


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

            # 流动性过滤: 近20日日均成交额(过滤500亿市值以下, 500亿×0.5%换手≈2.5亿)
            # 主板: 2亿 | 创业板(300): 1亿 | 科创板(688): 5000万
            if code.startswith('688'):
                min_amount = 50_000_000
            elif code.startswith('300'):
                min_amount = 100_000_000
            else:
                min_amount = 200_000_000
            min_vol = 300_000 if code.startswith('688') else 1_000_000
            if 'amount' in df.columns and len(df) >= 20:
                avg_amount = df['amount'].iloc[-20:].mean()
                if avg_amount < min_amount:
                    liquidity_filtered += 1
                    continue
            # 流动性补充: 无amount列时, 近20日日均成交量
            elif 'volume' in df.columns and len(df) >= 20:
                avg_vol = df['volume'].iloc[-20:].mean()
                if avg_vol < min_vol:
                    liquidity_filtered += 1
                    continue

            selected.add(code)
        except Exception:
            data_errors += 1
            continue

    selected.add('sh000001')
    print(f"股票池: {len(valid_files)} 总文件 -> 科创板{len(exclusion_set)} | 异常{data_errors} | 流动性{liquidity_filtered} -> {len(selected)} 只 (含sh000001)")
    return selected


def load_st_stocks() -> set:
    """从stock_list_full.csv加载ST股票代码（含*ST和ST前缀的股票）

    Returns:
        set of ST stock codes (不含前缀, 如 '000001')
    """
    metadata_dir = _get_metadata_dir()
    # stock_list_full.csv 包含ST标记, stock_list.csv 不包含
    stock_list_path = os.path.join(metadata_dir, 'stock_list_full.csv')

    if not os.path.exists(stock_list_path):
        # 回退到 stock_list.csv
        stock_list_path = os.path.join(metadata_dir, 'stock_list.csv')
        if not os.path.exists(stock_list_path):
            print("警告: 未找到 stock_list，无法过滤ST股票")
            return set()

    df = pd.read_csv(stock_list_path, dtype={'symbol': str})
    st_codes = set()
    for _, row in df.iterrows():
        name = str(row.get('name', ''))
        symbol = str(row.get('symbol', ''))
        # ST命名: '*ST香雪', 'ST逸飞' 等
        if ('ST' in name or '*ST' in name) and symbol:
            st_codes.add(symbol)

    # 同时过滤退市股票（名称含"退市"）
    for _, row in df.iterrows():
        name = str(row.get('name', ''))
        symbol = str(row.get('symbol', ''))
        if '退市' in name and symbol:
            st_codes.add(symbol)

    print(f"ST股票过滤: 识别 {len(st_codes)} 只ST/退市股票")
    return st_codes


def is_star_board(code: str) -> bool:
    """判断是否为科创板股票 (688xxx)"""
    return code.startswith('688')


def get_exclusion_set() -> set:
    """获取需要排除的股票集合.

    ST不再静态排除 — ST状态随时间变化(每年有新增/摘帽)，静态快照会错误排除非ST期的股票。
    改为在回测中由 fundamental_data.is_st() 逐日判断（bt_execution 向量化路径已实现）。

    Returns:
        set of stock codes to exclude
    """
    # 科创板(688xxx) 不再排除 — 2025-2026年妖股主要集中在科创板
    star_codes = set()

    excluded = star_codes
    print(f"股票排除: 科创板 {len(star_codes)} = 共排除 {len(excluded)} 只 (ST改为逐日判断)")
    return excluded
