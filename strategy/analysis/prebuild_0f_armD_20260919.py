# prebuild_0f_armD_20260919.py — 臂D (monthly + floor 0.25/None) membership缓存预构建
# 复用烟测键公式; 断言 C(0.5) ⊆ D(0.25) 单调性; 冷跑启动秒级命中省~4min
import sys, os, hashlib
sys.path.insert(0, '/mnt/d/quant/strategy')
import pandas as pd

from core.stock_pool import get_pool_membership_map, _monthly_boundaries
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
    return hashlib.md5(
        f"{_fp}|{_st.st_mtime_ns}|{_st.st_size}|monthly|"
        f"{[str(_b.date()) for _b in boundaries]}|True|{rf}|{rm}".encode('utf-8')).hexdigest()[:12]


# 臂C认证缓存 (dc01283cfce4, 冷跑0918c已命中)
mC = {}
_cC = '/mnt/d/quant/strategy/cache/pool_membership_dc01283cfce4.parquet'
assert os.path.exists(_cC), "臂C缓存缺失!"
_dfC = pd.read_parquet(_cC, columns=['boundary', 'code'])
for b, codes in _dfC.groupby('boundary')['code']:
    mC[pd.Timestamp(b)] = set(codes)
print(f"臂C图 (缓存dc01283cfce4): {len(_dfC)} 条 ✓")

mD = get_pool_membership_map(boundaries, cache_key=_key(0.25, None),
                             relax_floor=0.25, relax_momentum=None)
assert set(mD.keys()) == set(mC.keys())
nD = sum(len(v) for v in mD.values())
for b in boundaries:
    assert mC[b] <= mD[b], f"包含关系破坏 @{b.date()}"
nD_only = sum(len(mD[b] - mC[b]) for b in boundaries)
print(f"臂D图: {nD} 条, 较臂C新增 {nD_only} 边界-码 (C⊆D ✓)")
print(f"arm D缓存键 {_key(0.25, None)} 已预构建")
print("PREBUILD PASS")
