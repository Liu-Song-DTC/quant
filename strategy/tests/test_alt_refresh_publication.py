"""Ordinary reads cannot mutate published inputs; explicit writes keep versions."""
import json
import os
import time
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from strategy.core import alternative_data as module
from strategy.core.alt_data_versions import record_snapshot, resolve_asof, verify_current


@pytest.mark.parametrize('name,method', [('reduction_records.pkl','load_reduction'),
                                        ('reduction_plans.pkl','load_reduction_plans')])
def test_versioned_stale_read_does_not_fetch_or_publish(tmp_path, name, method):
    path=tmp_path/name
    pd.DataFrame({'code':['000001'], 'eitime':[pd.Timestamp('2020-01-01')],
                  'ann_date':['2020-01-01']}).to_pickle(path)
    record_snapshot(path,'2026-11-15');before=path.read_bytes()
    os.utime(path,(time.time()-72*3600,)*2)
    provider=module.AlternativeDataProvider(tmp_path)
    with (patch.object(module,'_AUTOREFRESH_OFF',False),
          patch.object(provider,'_fetch_reduction_em',side_effect=AssertionError('network')),
          patch.object(provider,'_fetch_reduction_plans_recent',side_effect=AssertionError('network'))):
        assert len(getattr(provider,method)())==1
    assert path.read_bytes()==before
    assert verify_current(path)=='2026-11-15'


def test_explicit_refresh_records_new_version_and_keeps_old_queries(tmp_path):
    path=tmp_path/'reduction_records.pkl'
    old=pd.DataFrame({'code':['000001'],'eitime':[pd.Timestamp('2026-01-01')],
        'holder':['old'],'start_date':[pd.Timestamp('2026-01-01')],'end_date':[pd.Timestamp('2026-01-02')]})
    old.to_pickle(path);record_snapshot(path,'2026-11-15')
    archive=resolve_asof(path,'2026-11-15');old_bytes=archive.read_bytes()
    new=old.copy();new['code']='000002';os.utime(path,(time.time()-72*3600,)*2)
    with (patch.object(module,'_AUTOREFRESH_OFF',False),
          patch.object(module.AlternativeDataProvider,'_fetch_reduction_em',return_value=new)):
        result=module.AlternativeDataProvider(tmp_path,strict_refresh=True).load_reduction()
    assert len(result)==2 and verify_current(path)==date.today().isoformat()
    assert resolve_asof(path,'2026-11-15').read_bytes()==old_bytes
    state=json.loads((tmp_path/'alt_versions/reduction_records.json').read_text())
    before=len(state['versions']);assert record_snapshot(path)=='same'
    assert len(json.loads((tmp_path/'alt_versions/reduction_records.json').read_text())['versions'])==before


def test_publication_failure_keeps_current_bytes(tmp_path):
    path=tmp_path/'reduction_records.pkl';pd.DataFrame({'code':['000001']}).to_pickle(path)
    record_snapshot(path);before=path.read_bytes()
    with patch('strategy.core.alt_data_versions.record_snapshot',side_effect=OSError('archive unavailable')):
        with pytest.raises(OSError,match='archive unavailable'):
            module._atomic_to_pickle(pd.DataFrame({'code':['000002']}),path)
    assert path.read_bytes()==before and not path.with_name(path.name+'.tmp').exists()
    verify_current(path)


def test_untracked_mutation_is_not_silently_accepted_by_a_writer(tmp_path):
    path=tmp_path/'reduction_records.pkl';pd.DataFrame({'code':['000001']}).to_pickle(path)
    record_snapshot(path);pd.DataFrame({'code':['000002']}).to_pickle(path);before=path.read_bytes()
    with pytest.raises(ValueError,match='differs'):
        module._atomic_to_pickle(pd.DataFrame({'code':['000003']}),path)
    assert path.read_bytes()==before
