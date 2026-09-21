"""单测: 逆回购闲置现金sweep估算 (core/repo_sweep.py)

覆盖: 开关/门槛/取整/缓冲/tenor钳制/异常输入。纯算术无IO, 秒级。
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.repo_sweep import (
    compute_repo_sweep, sweep_guidance,
    MIN_AMOUNT, INCREMENT, MIN_RESERVE,
)


def test_disabled_by_default():
    s = compute_repo_sweep(250000, {})
    assert s['enabled'] is False
    assert s['reason'] == 'repo_sweep关闭'


def test_basic_sweep():
    # 50万账户典型场景: 持仓43%≈21.5万, 订单后剩余现金28.5万
    s = compute_repo_sweep(285000, {'enabled': True})
    assert s['enabled'] is True
    assert s['code'] == '204001.SH'
    assert s['tenor'] == '1d'
    reserve = max(285000 * 0.10, MIN_RESERVE)  # 28,500
    expected = int((285000 - reserve) // INCREMENT) * INCREMENT
    assert s['amount'] == expected  # 256,000
    assert s['reserve'] == int(reserve)
    assert s['estimated_annual_yield'] == s['amount'] * 0.015


def test_rounding_floor_to_1000():
    # 12,345 → 缓冲1,234.5 → 可扫11,110.5 → 取整11,000
    s = compute_repo_sweep(12345, {'enabled': True})
    assert s['amount'] == 11000
    assert s['amount'] % INCREMENT == 0


def test_below_threshold_no_sweep():
    # 可扫金额不足1000元 → amount=0, 但仍报告enabled(机制开启, 无可扫)
    s = compute_repo_sweep(1000, {'enabled': True})
    assert s['enabled'] is True
    assert s['amount'] == 0
    assert '不足一笔' in s['reason']


def test_min_reserve_never_touched():
    # 现金恰好=MIN_RESERVE: 缓冲钳制到1000, 可扫=0
    s = compute_repo_sweep(MIN_RESERVE, {'enabled': True})
    assert s['amount'] == 0
    # 现金=MIN_RESERVE+999: 可扫999 < 门槛 → 0
    s2 = compute_repo_sweep(MIN_RESERVE + INCREMENT - 1, {'enabled': True})
    assert s2['amount'] == 0


def test_zero_or_negative_cash():
    assert compute_repo_sweep(0, {'enabled': True})['enabled'] is False
    assert compute_repo_sweep(-500, {'enabled': True})['enabled'] is False
    assert compute_repo_sweep(None, {'enabled': True})['enabled'] is False


def test_unsupported_tenor_rejected():
    # 7d/14d会锁死10d调仓周期资金, v1拒绝
    s = compute_repo_sweep(285000, {'enabled': True, 'tenor': '7d'})
    assert s['enabled'] is False
    assert 'tenor' in s['reason']
    s2 = compute_repo_sweep(285000, {'enabled': True, 'tenor': '14d'})
    assert s2['enabled'] is False


def test_reserve_ratio_clamped():
    # 非法ratio钳制: 2.0→1.0(全留缓冲→无可扫), -0.5→0.0
    s = compute_repo_sweep(285000, {'enabled': True, 'reserve_ratio': 2.0})
    assert s['reserve'] == 285000
    assert s['amount'] == 0
    s2 = compute_repo_sweep(285000, {'enabled': True, 'reserve_ratio': -0.5})
    assert s2['reserve'] == MIN_RESERVE
    assert s2['amount'] == int((285000 - MIN_RESERVE) // INCREMENT) * INCREMENT


def test_guidance_matches_amount():
    s = compute_repo_sweep(285000, {'enabled': True})
    notes = sweep_guidance(s)
    assert len(notes) == 6
    assert any(str(s['amount']) in n for n in notes)
    assert any('204001.SH' in n for n in notes)
    assert any('卖出' in n for n in notes)
    assert any(str(s['reserve']) in n for n in notes)
    # 关闭态无指引
    assert sweep_guidance(compute_repo_sweep(285000, {})) == []


def test_exact_boundary():
    # 现金10,000: 缓冲1,000 → 可扫9,000 恰好整数
    s = compute_repo_sweep(10000, {'enabled': True})
    assert s['amount'] == 9000


if __name__ == '__main__':
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_') and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
