"""CSV representation must not alter scores, filters or compressed factors."""
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.fundamental_units import parse_percent_points
from core.fundamental import FundamentalData
from core.factor_calculator import compress_fundamental_factor, compute_fundamental_score
from core.factor_preparer import _preload_fundamental_cache
from core.signal_engine import SignalEngine
from core.stock_pool_filter import StockPoolFilter


@pytest.mark.parametrize('value', [12.5, '12.5', '12.5%'])
def test_percent_representation(value):
    assert parse_percent_points(value) == .125


@pytest.mark.parametrize('value', [None, '', '--'])
def test_missing(value):
    assert parse_percent_points(value) is None


@pytest.mark.parametrize('value', [np.nan, np.inf, '-inf'])
def test_nonfinite_keeps_existing_semantics(value):
    expected = float(value) / 100.
    actual = parse_percent_points(value)
    assert np.isnan(actual) if np.isnan(expected) else actual == expected


@pytest.mark.parametrize('value', [1.2, '1.2'])
def test_consumer_parity_and_eps(value):
    row = {'数据可用日期': '20250115', '净资产收益率': 12.,
           '净利润-同比增长': 25., '营业总收入-同比增长': 18.,
           '销售毛利率': 30., '每股收益': value, 'zcfz_资产负债率': 50.}
    fd = FundamentalData.__new__(FundamentalData)
    fd._get_latest = lambda *args: row
    fd.stock_data = {'000001': pd.DataFrame([row])}
    expected = compute_fundamental_score(.12, .25, .18, 1.2)
    if not isinstance(value, str):
        assert fd.get_fundamental_score('000001', '20250116') == 49.0
    assert SignalEngine._compute_fund_score_from_row(row) == expected
    cache, _ = _preload_fundamental_cache(fd, '000001', [pd.Timestamp('2025-01-16')])
    assert cache[pd.Timestamp('2025-01-16')]['fund_score'] == expected
    assert fd.get_eps('000001', '20250116') == value
    assert SignalEngine._extract_fund_value_from_row(row, 'fund_eps') == 1.2
    for name, getter in [('fund_roe', fd.get_roe), ('fund_profit_growth', fd.get_profit_growth),
                         ('fund_revenue_growth', fd.get_revenue_growth),
                         ('fund_gross_margin', fd.get_gross_margin), ('fund_debt_ratio', fd.get_debt_ratio)]:
        assert SignalEngine._extract_fund_value_from_row(row, name) == getter('000001', '20250116')


@pytest.mark.parametrize('name,ratio', [('fund_roe', .10), ('fund_gross_margin', .30),
                                       ('fund_debt_ratio', .50)])
def test_compression_economic_midpoint(name, ratio):
    assert compress_fundamental_factor(ratio, name) == pytest.approx(0.)


def test_quarterly_filter_keeps_percentage_point_configuration():
    fd = SimpleNamespace(get_roe=lambda *a: .12, get_profit_growth=lambda *a: .25,
                         get_revenue_growth=lambda *a: .18, get_eps=lambda *a: 1.2)
    filt = StockPoolFilter(fd, min_roe=10, min_profit_growth=20)
    assert filt._is_qualified('000001', '20250116')
    assert filt.get_stock_score('000001', '20250116') == 36.5
    fd.get_roe = lambda *a: .09
    assert not filt._is_qualified('000001', '20250116')


def test_growth_acceleration_is_ratio_difference():
    fd = FundamentalData.__new__(FundamentalData)
    fd._get_latest = lambda *a: {'净利润-同比增长': '25%', '营业总收入-同比增长': 18.}
    fd._get_nth_latest = lambda *a, **kw: {'净利润-同比增长': 10., '营业总收入-同比增长': '8%'}
    assert fd.get_profit_growth_improve('', '') == pytest.approx(.15)
    assert fd.get_revenue_growth_improve('', '') == pytest.approx(.10)


@pytest.mark.parametrize('name', ['fund_profit_growth', 'fund_revenue_growth'])
@pytest.mark.parametrize('points', [-25., -.5, 0., .5, 25.])
def test_growth_compression_preserves_percentage_point_curve(name, points):
    # Historical calibration consumes tanh(percentage points), not tanh(ratios).
    assert compress_fundamental_factor(parse_percent_points(points), name) == pytest.approx(
        np.tanh(np.clip(points, -100, 100)))


@pytest.mark.parametrize('name', ['fund_pg_improve', 'fund_rg_improve'])
@pytest.mark.parametrize('current,previous', [(25., 10.), (10.5, 10.), (9.5, 10.)])
def test_growth_change_compression_preserves_percentage_point_curve(name, current, previous):
    delta = parse_percent_points(current) - parse_percent_points(previous)
    assert compress_fundamental_factor(delta, name) == pytest.approx(
        np.tanh(np.clip(current - previous, -5, 5) * 2))


@pytest.mark.parametrize('points', [.03, -.03, 0., 300., -100.])
def test_percent_units_do_not_depend_on_magnitude(points):
    assert parse_percent_points(points) == parse_percent_points(str(points))
    assert parse_percent_points(points) == points / 100.




@pytest.mark.parametrize('profit,growth,flawed', [(1., -.31, True), (1., -.30, False),
                                                 (1., -.29, False), (-1., -.01, True),
                                                 (-1., .01, False), (1., None, False)])
def test_portfolio_profit_drop_threshold_uses_ratio(profit, growth, flawed):
    from core.portfolio import _has_fundamental_flaw
    fd = SimpleNamespace(get_profit=lambda *a: profit, get_profit_growth=lambda *a: growth)
    assert _has_fundamental_flaw(fd, '000001', '20250116') is flawed
