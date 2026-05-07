"""
股票池模块 - 按流动性/市值筛选，控制回测和选股规模

默认筛选 top 500 流动性最好的股票，将 5351 -> 500 只
"""
import os
import pandas as pd
import numpy as np
from pathlib import Path


def _get_data_dir():
    """获取backtrader数据目录"""
    base = Path(__file__).parent.parent.parent
    return str(base / 'data' / 'stock_data' / 'backtrader_data')


def get_stock_pool(min_price: float = 3.0, top_n: int = 500,
                   data_dir: str = None) -> set:
    """获取流动性筛选后的股票池

    筛选逻辑:
    1. 价格 > min_price（过滤仙股）
    2. 最近20日日均成交额 top N

    Args:
        min_price: 最低价格过滤
        top_n: 选取前N只股票
        data_dir: 数据目录路径

    Returns:
        set of stock codes
    """
    if data_dir is None:
        data_dir = _get_data_dir()

    if not os.path.exists(data_dir):
        return set()

    # 快速扫描：只读取每只股票的最后30行
    candidates = []
    for item in os.listdir(data_dir):
        if not (item.endswith('_qfq.csv') or item.endswith('_hfq.csv')):
            continue
        if item.startswith('sh000001'):
            continue

        code = item[:-8] if item.endswith('_qfq.csv') else item[:-8]
        filepath = os.path.join(data_dir, item)

        try:
            # 只读最后50行（够判断流动性和价格）
            df = pd.read_csv(filepath, nrows=50)
            if len(df) < 20:
                continue

            # 用最后20个交易日
            recent = df.tail(20)
            last_price = recent['close'].iloc[-1]
            if last_price < min_price or last_price > 500:
                continue

            # 日均成交额 = mean(close * volume)
            if 'volume' in recent.columns and 'close' in recent.columns:
                avg_amount = (recent['close'] * recent['volume']).mean()
            else:
                continue

            if avg_amount > 0:
                candidates.append((code, avg_amount))
        except Exception:
            continue

    # 按日均成交额降序，取 top N
    candidates.sort(key=lambda x: -x[1])
    selected = {code for code, _ in candidates[:top_n]}

    # 始终包含指数
    selected.add('sh000001')

    print(f"股票池: {len(candidates)} 只有效股票 -> 选取 top {top_n} (按日均成交额)")
    return selected
