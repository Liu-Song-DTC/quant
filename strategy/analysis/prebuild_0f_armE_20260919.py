# prebuild_0f_armE_20260919.py — 臂E (monthly + floor 0.10/None) membership缓存预构建
# 复用烟测键公式; 断言 D(0.25) ⊆ E(0.10) 单调性; 冷跑启动秒级命中省~4min
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


# 臂D认证缓存 (d2825a292629, 冷跑0919a已命中)
mD = {}
_cD = '/mnt/d/quant/strategy/cache/pool_membership_d2825a292629.parquet'
assert os.path.exists(_cD), "臂D缓存缺失!"
_dfD = pd.read_parquet(_cD, columns=['boundary', 'code'])
for b, codes in _dfD.groupby('boundary')['code']:
    mD[pd.Timestamp(b)] = set(codes)
print(f"臂D图 (缓存d2825a292629): {len(_dfD)} 条 ✓")

mE = get_pool_membership_map(boundaries, cache_key=_key(0.1, None),
                             relax_floor=0.1, relax_momentum=None)
assert set(mE.keys()) == set(mD.keys())
nE = sum(len(v) for v in mE.values())
for b in boundaries:
    assert mD[b] <= mE[b], f"包含关系破坏 @{b.date()}"
nE_only = sum(len(mE[b] - mD[b]) for b in boundaries)
print(f"臂E图: {nE} 条, 较臂D新增 {nE_only} 边界-码 (D⊆E ✓)")
print(f"arm E缓存键 {_key(0.1, None)} 已预构建")
print("PREBUILD PASS")
