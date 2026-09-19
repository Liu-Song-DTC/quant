# prebuild_0f_armG_20260919.py — 臂G (monthly + floor 0.05/None) membership缓存预构建
# flat-top中心探针: E(0.10)与F(0.0)rank-sum并列7, 中心=0.05 (主板8M/创业4M/科创2M).
# 断言 E(0.10) ⊆ G(0.05) ⊆ F(0.0) 双向包含; 冷跑启动秒级命中省~4min
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


def _load(key, tag):
    p = f'/mnt/d/quant/strategy/cache/pool_membership_{key}.parquet'
    assert os.path.exists(p), f"{tag}缓存缺失!"
    d = pd.read_parquet(p, columns=['boundary', 'code'])
    m = {}
    for b, codes in d.groupby('boundary')['code']:
        m[pd.Timestamp(b)] = set(codes)
    print(f"{tag}图 (缓存{key}): {len(d)} 条 ✓")
    return m


mE = _load('50c8d10602e8', '臂E')
mF = _load('bbb0b2719700', '臂F')

mG = get_pool_membership_map(boundaries, cache_key=_key(0.05, None),
                             relax_floor=0.05, relax_momentum=None)
assert set(mG.keys()) == set(mE.keys()) == set(mF.keys())
nG = sum(len(v) for v in mG.values())
for b in boundaries:
    assert mE[b] <= mG[b] <= mF[b], f"包含关系破坏 @{b.date()}"
nGE = sum(len(mG[b] - mE[b]) for b in boundaries)
nFG = sum(len(mF[b] - mG[b]) for b in boundaries)
print(f"臂G图: {nG} 条 (E⊆G⊆F ✓, 较E新增 {nGE}, 较F少 {nFG})")
print(f"arm G缓存键 {_key(0.05, None)} 已预构建")
print("PREBUILD PASS")
