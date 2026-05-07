"""
因子数据/IC缓存管理器 - 避免重复计算

缓存策略:
- factor_df: parquet格式, 按股票数量和日期范围生成缓存键
- IC selection: pickle格式, 按config hash生成缓存键
"""
import os
import pickle
import hashlib
import json
from pathlib import Path
import pandas as pd


def _get_cache_dir() -> str:
    """获取缓存目录"""
    base = Path(__file__).parent.parent
    cache_dir = base / 'cache'
    cache_dir.mkdir(exist_ok=True)
    return str(cache_dir)


def _config_hash(config_snippet: dict) -> str:
    """生成配置哈希"""
    raw = json.dumps(config_snippet, sort_keys=True, default=str)
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def get_factor_cache_path(stock_count: int, date_count: int) -> str:
    """获取factor_df缓存路径"""
    cache_dir = _get_cache_dir()
    return os.path.join(cache_dir, f'factor_df_{stock_count}s_{date_count}d.parquet')


def save_factor_cache(factor_df: pd.DataFrame, stock_count: int, date_count: int):
    """保存factor_df缓存"""
    path = get_factor_cache_path(stock_count, date_count)
    factor_df.to_parquet(path, index=False)
    print(f"因子缓存已保存: {path} ({len(factor_df)} 行)")


def load_factor_cache(stock_count: int, date_count: int) -> pd.DataFrame:
    """加载factor_df缓存"""
    path = get_factor_cache_path(stock_count, date_count)
    if os.path.exists(path):
        df = pd.read_parquet(path)
        print(f"因子缓存已加载: {path} ({len(df)} 行)")
        return df
    return None


def get_ic_cache_path(stock_count: int, date_count: int, config_hash: str) -> str:
    """获取IC缓存路径"""
    cache_dir = _get_cache_dir()
    return os.path.join(cache_dir, f'ic_cache_{stock_count}s_{date_count}d_{config_hash}.pkl')


def save_ic_cache(cache: dict, all_dates: list, stock_count: int,
                  date_count: int, dynamic_config: dict):
    """保存IC预计算缓存"""
    config_hash = _config_hash(dynamic_config)
    path = get_ic_cache_path(stock_count, date_count, config_hash)
    data = {'cache': cache, 'all_dates': all_dates}
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    print(f"IC缓存已保存: {path} ({len(cache)} 个日期)")


def load_ic_cache(stock_count: int, date_count: int,
                  dynamic_config: dict) -> dict:
    """加载IC预计算缓存. Returns None if not found or config changed."""
    config_hash = _config_hash(dynamic_config)
    path = get_ic_cache_path(stock_count, date_count, config_hash)
    if os.path.exists(path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        print(f"IC缓存已加载: {path} ({len(data['cache'])} 个日期)")
        return data
    return None
