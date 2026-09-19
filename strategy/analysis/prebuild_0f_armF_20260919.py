# prebuild_0f_armF_20260919.py — 臂F (monthly + floor 0.0/None) membership缓存预构建
# floor 0.0 = 纯准入端点: 无流动性地板, 仅cut≥60+close≥2.0+数据存在.
# 复用烟测键公式; 断言 E(0.10) ⊆ F(0.0) 单调性; 冷跑启动秒级命中省~4min
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


# 臂E认证缓存 (50c8d10602e8, 冷跑0919b已命中)
mE = {}
_cE = '/mnt/d/quant/strategy/cache/pool_membership_50c8d10602e8.parquet'
assert os.path.exists(_cE), "臂E缓存缺失!"
_dfE = pd.read_parquet(_cE, columns=['boundary', 'code'])
for b, codes in _dfE.groupby('boundary')['code']:
    mE[pd.Timestamp(b)] = set(codes)
print(f"臂E图 (缓存50c8d10602e8): {len(_dfE)} 条 ✓")

mF = get_pool_membership_map(boundaries, cache_key=_key(0.0, None),
                             relax_floor=0.0, relax_momentum=None)
assert set(mF.keys()) == set(mE.keys())
nF = sum(len(v) for v in mF.values())
for b in boundaries:
    assert mE[b] <= mF[b], f"包含关系破坏 @{b.date()}"
nF_only = sum(len(mF[b] - mE[b]) for b in boundaries)
print(f"臂F图: {nF} 条, 较臂E新增 {nF_only} 边界-码 (E⊆F ✓)")
print(f"arm F缓存键 {_key(0.0, None)} 已预构建")
print("PREBUILD PASS")
