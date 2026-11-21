import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.execution_quote_snapshot import ExecutionQuoteSnapshot, FIELDS


@pytest.fixture
def fixture(tmp_path):
    data=tmp_path/'prices';data.mkdir()
    raw=tmp_path/'raw';(raw/'000001').mkdir(parents=True)
    calendar=pd.date_range('2025-01-02',periods=3)
    bars=pd.DataFrame({'datetime':calendar,'close':[1.,2.,3.]})
    for code in ('sh000001','000001','000002'):bars.to_csv(data/f'{code}_qfq.csv',index=False)
    (raw/'000001/none.csv').write_text('date,close\n2025-01-02,1\n')
    source=tmp_path/'producer.py';source.write_text('VERSION=1\n')
    def make():
        return ExecutionQuoteSnapshot(tmp_path/'cache',data,raw,['000001','000002'],calendar,
                                      calendar[0],calendar[-1],[source])
    arrays={name:np.asfortranarray(np.array([[np.nan,1.],[2.,-0.],[3.,4.]],dtype=np.float32)) for name in FIELDS}
    return make,arrays,data,raw,source


def test_roundtrip_preserves_bytes_nan_signed_zero_and_layout(fixture):
    make,arrays,*_=fixture
    cache=make();assert cache.load() is None
    cache.save(arrays,['000001','000002'],[])
    actual,codes,excluded=make().load()
    assert codes==['000001','000002'] and excluded==[]
    for name in FIELDS:
        assert actual[name].tobytes(order='A')==arrays[name].tobytes(order='A')
        assert actual[name].flags.f_contiguous and actual[name].dtype==np.float32


@pytest.mark.parametrize('role',['qfq','raw','source','missing_raw_created'])
def test_content_change_invalidates_even_with_same_size_and_mtime(fixture,role):
    make,arrays,data,raw,source=fixture
    first=make();first.save(arrays,['000001','000002'],[])
    path={'qfq':data/'000001_qfq.csv','raw':raw/'000001/none.csv','source':source,
          'missing_raw_created':raw/'000002/none.csv'}[role]
    if path.exists():
        stat=path.stat();old=path.read_bytes();assert b'1' in old
        path.write_bytes(old.replace(b'1',b'2',1));os.utime(path,ns=(stat.st_atime_ns,stat.st_mtime_ns))
        assert path.stat().st_size==stat.st_size
    else:
        path.parent.mkdir();path.write_text('date,close\n2025-01-02,1\n')
    second=make();assert second.key!=first.key and second.load() is None
    with pytest.raises(ValueError,match='changed'):first.assert_unchanged()


def test_changed_data_during_build_cannot_be_saved(fixture):
    make,arrays,data,*_=fixture
    cache=make();(data/'000001_qfq.csv').write_text('changed')
    with pytest.raises(ValueError,match='changed'):cache.save(arrays,['000001','000002'],[])
    assert not cache.path.exists()


def test_index_calendar_and_stock_listing_are_invalidation_inputs(fixture):
    make,arrays,data,*_=fixture
    cache=make();cache.save(arrays,['000001','000002'],[])
    (data/'000003_qfq.csv').write_text('new stock')
    with pytest.raises(ValueError,match='universe'):make()
    (data/'000003_qfq.csv').unlink()
    index=data/'sh000001_qfq.csv';index.write_text(index.read_text().replace('2025-01-02','2025-01-01'))
    with pytest.raises(ValueError,match='calendar'):make()


def test_exclusions_preserve_original_order(fixture):
    make,arrays,*_=fixture
    reduced={name:np.ascontiguousarray(a[:,:1]) for name,a in arrays.items()}
    cache=make();cache.save(reduced,['000001'],['000002'])
    assert cache.load()[1:]==(['000001'],['000002'])
    with pytest.raises(ValueError,match='order'):cache.save(arrays,['000002','000001'],[])


@pytest.mark.parametrize('damage',['bytes','dtype','manifest','extra_member'])
def test_corruption_is_rejected_not_silently_used(fixture,damage):
    make,arrays,*_=fixture
    cache=make();cache.save(arrays,['000001','000002'],[])
    with np.load(cache.path,allow_pickle=False) as f:stored={name:f[name] for name in f.files}
    if damage=='bytes':stored['open_px'][1,0]=123
    elif damage=='dtype':stored['open_px']=stored['open_px'].astype(np.float64)
    elif damage=='extra_member':stored['unknown']=np.ones(1)
    else:
        record=json.loads(stored['manifest'].tobytes());record['identity']['version']=2
        stored['manifest']=np.frombuffer(json.dumps(record).encode(),dtype=np.uint8)
    np.savez_compressed(cache.path,**stored)
    with pytest.raises(ValueError):make().load()
