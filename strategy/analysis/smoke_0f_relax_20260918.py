# smoke_0f_relax_20260918.py — 0f-v3 relax臂 (B/C) 烟测
# 1. _relax_momentum_ok 单元断言
# 2. 真实membership三图关系: strict ⊆ B(0.5/0.10) ⊆ C(0.5/None)
# 3. arm B/C运行缓存键预构建 (冷跑启动秒级命中, 省~4min/臂)
import sys, os, hashlib
sys.path.insert(0, '/mnt/d/quant/strategy')
import pandas as pd

from core.stock_pool import _relax_momentum_ok, get_pool_membership_map, _monthly_boundaries

# ── 1. 单元 ──
s = pd.Series([100.0] * 60)
assert _relax_momentum_ok(s, 60, None) is True
assert _relax_momentum_ok(s, 60, 0.10) is False       # 0收益 < 门槛
s2 = s.copy(); s2.iloc[-1] = 110.0
assert _relax_momentum_ok(s2, 60, 0.10) is True       # +10% ≥ 门槛 (含边界)
assert _relax_momentum_ok(s2, 60, 0.11) is False
s3 = s.copy(); s3.iloc[39] = float('nan')             # cut-21位置NaN
assert _relax_momentum_ok(s3, 60, 0.10) is False
s4 = s.copy(); s4.iloc[-1] = float('nan')
assert _relax_momentum_ok(s4, 60, 0.10) is False
print("单元: _relax_momentum_ok 6断言 ✓")

# ── 2/3. 真实membership + arm B/C缓存键预构建 ──
import bt_execution as bt
from core.factor_preparer import _data_fingerprint

earliest = pd.Timestamp(bt.FROMDATE) - pd.Timedelta(days=730)
boundaries = _monthly_boundaries(earliest, pd.Timestamp(bt.TODATE))
_all_map = {}
for _item in os.listdir(bt.DATA_PATH):
    if _item.endswith('_qfq.csv'):
        _all_map[_item[:-8]] = bt.DATA_PATH + _item
    elif _item.endswith('_hfq.csv'):
        _all_map[_item[:-8]] = bt.DATA_PATH + _item
_fp = _data_fingerprint(_all_map, None, [])
_sp = os.path.join(os.path.dirname(os.path.abspath(bt.__file__)), 'core', 'stock_pool.py')
_st = os.stat(_sp)


def _key(rf, rm):
    """复刻bt_execution._load_pool_membership的缓存键公式."""
    return hashlib.md5(
        f"{_fp}|{_st.st_mtime_ns}|{_st.st_size}|monthly|"
        f"{[str(_b.date()) for _b in boundaries]}|True|{rf}|{rm}".encode('utf-8')).hexdigest()[:12]


# 严格图 = arm A已认证缓存 (monthly 1.0/None; relax=1.0与arm A准入逐位同构,
# 且本烟测构建的B/C缓存键用当前stock_pool.py stat — arm A键因代码编辑已变,
# 故直接读arm A parquet文件, 缺失时才重算)
_kA = '94635f773433'
strict_parquet = f'/mnt/d/quant/strategy/cache/pool_membership_{_kA}.parquet'
if os.path.exists(strict_parquet):
    strict_df = pd.read_parquet(strict_parquet, columns=['boundary', 'code'])
    strict = {}
    for b, codes in strict_df.groupby('boundary')['code']:
        strict[pd.Timestamp(b)] = set(codes)
    print(f"strict图 (arm A缓存 {_kA}): {len(strict_df)} 条 ✓")
else:
    mS = get_pool_membership_map(boundaries, cache_key=_key(1.0, None),
                                 relax_floor=1.0, relax_momentum=None)
    strict = {b: set(v) for b, v in mS.items()}
    print(f"strict图 (重算, 当前代码态): {sum(len(v) for v in strict.values())} 条 ✓")

mB = get_pool_membership_map(boundaries, cache_key=_key(0.5, 0.10),
                             relax_floor=0.5, relax_momentum=0.10)
mC = get_pool_membership_map(boundaries, cache_key=_key(0.5, None),
                             relax_floor=0.5, relax_momentum=None)
assert set(mB.keys()) == set(mC.keys()) == set(strict.keys())
for b in boundaries:
    assert strict[b] <= mB[b] <= mC[b], f"包含关系破坏 @{b.date()}"
nB_only = sum(len(mB[b] - strict[b]) for b in boundaries)
nC_only = sum(len(mC[b] - strict[b]) for b in boundaries)
nB_vs_C = sum(len(mC[b] - mB[b]) for b in boundaries)
print(f"关系: strict ⊆ B ⊆ C ✓ (B较严格新增 {nB_only} 边界-码, "
      f"C新增 {nC_only}, 动量门拦截 {nB_vs_C})")
assert nB_only > 0 and nC_only > 0, "松弛臂无实际准入效果!"
print(f"arm B缓存键 {_key(0.5, 0.10)} / arm C缓存键 {_key(0.5, None)} 已预构建")
print("SMOKE PASS")
