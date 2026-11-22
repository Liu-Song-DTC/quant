import pandas as pd
import pytest

from core.live_price_basis import LivePriceBasis


def basis(tmp_path, raw_dates, qfq_dates, raw_close=None, qfq_close=None):
    raw=tmp_path/'none.csv';qfq=tmp_path/'qfq.csv'
    pd.DataFrame({'日期':raw_dates,'收盘':raw_close or [10.]*len(raw_dates)}).to_csv(raw,index=False)
    pd.DataFrame({'datetime':qfq_dates,'close':qfq_close or [20.]*len(qfq_dates)}).to_csv(qfq,index=False)
    return LivePriceBasis({'600001':raw},{'600001':qfq})


def test_only_visible_bars_and_verified_suspension_scale(tmp_path):
    b=basis(tmp_path,['2026-11-19','2026-11-23'],['2026-11-19','2026-11-23'],[10.,9.],[20.,30.])
    assert b.raw_per_qfq('600001','2026-11-20')==.5
    assert b.raw_per_qfq('600001','2026-11-23')==.3


@pytest.mark.parametrize('raw_dates,qfq_dates',[
    (['2026-11-20'],['2026-11-19']),
    (['2026-11-20','2026-11-20'],['2026-11-20']),
    (['2026-11-23'],['2026-11-23']),
])
def test_unverified_scale_does_not_fallback_to_one(tmp_path,raw_dates,qfq_dates):
    b=basis(tmp_path,raw_dates,qfq_dates)
    with pytest.raises(ValueError):b.raw_per_qfq('600001','2026-11-20')
