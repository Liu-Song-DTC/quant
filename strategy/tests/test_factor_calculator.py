"""冒烟测试: factor_calculator 核心函数"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from core.factor_calculator import (
    calculate_indicators,
    compute_composite_factors,
    compress_fundamental_factor,
)


def _make_arrays(n: int = 200):
    """生成模拟价格数组"""
    rng = np.random.RandomState(42)
    close = 10.0 + np.cumsum(rng.randn(n) * 0.1)
    high = close + np.abs(rng.randn(n) * 0.03)
    low = close - np.abs(rng.randn(n) * 0.03)
    open_ = close + rng.randn(n) * 0.01
    volume = rng.randint(100000, 1000000, n).astype(float)
    return close, high, low, open_, volume


def test_calculate_indicators_returns_dict():
    """验证 calculate_indicators 返回包含核心指标的dict"""
    close, high, low, open_, volume = _make_arrays(200)
    result = calculate_indicators(close, high, low, volume, open_arr=open_)

    assert isinstance(result, dict), f"期望dict, 得到{type(result)}"
    required = ['close', 'ma5', 'ma20', 'ma60', 'rsi_14', 'volatility_20']
    for key in required:
        arr = result.get(key)
        assert arr is not None, f"缺少指标: {key}"
        assert len(arr) == 200, f"{key}长度应为200, 实际{len(arr)}"


def test_calculate_indicators_short_data():
    """数据不足时不应crash"""
    close, high, low, open_, volume = _make_arrays(10)
    result = calculate_indicators(close, high, low, volume, open_arr=open_)
    assert isinstance(result, dict)
    assert 'close' in result


def test_calculate_indicators_handles_nan():
    """close价格含NaN时不应crash"""
    close, high, low, open_, volume = _make_arrays(60)
    close[30:35] = np.nan
    result = calculate_indicators(close, high, low, volume, open_arr=open_)
    assert isinstance(result, dict)


def test_compute_composite_factors():
    """验证 compute_composite_factors 单bar计算返回dict且包含核心因子"""
    close, high, low, open_, volume = _make_arrays(200)
    indicators = calculate_indicators(close, high, low, volume, open_arr=open_)

    # compute_composite_factors 对单个bar计算
    result = compute_composite_factors(indicators, idx=150, fund_score=0.5)
    assert isinstance(result, dict)

    expected = ['volatility', 'momentum_reversal', 'trend_vol',
                'smart_money_flow', 'relative_strength']
    for col in expected:
        assert col in result, f"缺少因子: {col}"
    # 所有值应是float
    for v in result.values():
        assert isinstance(v, (float, int, np.floating)), f"非float值: {type(v)}"


def test_compress_fundamental_factor():
    """fundamental压缩: tanh映射"""
    result = compress_fundamental_factor(0.5, 'fund_roe')
    assert isinstance(result, float)
    assert -3.0 <= result <= 3.0

    result2 = compress_fundamental_factor(0.95, 'fund_score')
    assert isinstance(result2, float)


def test_calculate_indicators_volume_metrics():
    """成交量指标存在"""
    close, high, low, open_, volume = _make_arrays(200)
    result = calculate_indicators(close, high, low, volume, open_arr=open_)

    vol_keys = ['volume_ma20', 'volume_ratio']
    for key in vol_keys:
        assert key in result, f"缺少成交量指标: {key}"
