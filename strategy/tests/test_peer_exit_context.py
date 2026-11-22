"""Canonical numeric boundaries, report availability and immutable refresh."""
import ast
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'strategy'))
from core.peer_exit_context import (SCHEMA, PEER_FIELDS, canonical_batch,
    peer_without_self, peer_deterioration, read_report_stock,
    report_industries_asof, require_exact_rows, sha256)
from core.profit_guard import DailyExitContext


def archived_kernels():
    namespace = dict(pd=pd, np=np, SCHEMA=SCHEMA)
    for filename, names in [
        ('research_contextual_profit_guard.py', {'peer_without_self', 'peer_deterioration'}),
        ('build_full_market_peer_exit_context.py', {'canonical_batch'}),
    ]:
        path = ROOT/'logs/optimization_20261120'/filename
        tree = ast.parse(path.read_text())
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


def test_kernels_have_the_archived_operation_order_and_exact_outputs():
    original = archived_kernels()
    current = ast.parse((ROOT/'strategy/core/peer_exit_context.py').read_text())
    for filename, names in [
        ('research_contextual_profit_guard.py', {'peer_without_self', 'peer_deterioration'}),
        ('build_full_market_peer_exit_context.py', {'canonical_batch'}),
    ]:
        archived = ast.parse((ROOT/'logs/optimization_20261120'/filename).read_text())
        for name in names:
            a = next(n for n in current.body if isinstance(n, ast.FunctionDef) and n.name == name)
            b = next(n for n in archived.body if isinstance(n, ast.FunctionDef) and n.name == name)
            assert ast.dump(a, include_attributes=False) == ast.dump(b, include_attributes=False)
    dates = pd.bdate_range('2024-01-01', periods=400)
    rng = np.random.default_rng(17)
    daily = pd.DataFrame(dict(code='000001', date=dates, sector='A',
        ret_1d=rng.normal(0, .03, len(dates)), observed=8.,
        total_ret=rng.normal(0, .06, len(dates)), advancing=rng.integers(0, 9, len(dates))))
    daily.loc[60:65, 'observed'] = 5
    daily.loc[120:125, 'sector'] = None
    daily.loc[180:, 'sector'] = 'B'
    daily = daily.drop(index=[200, 201])
    benchmark = pd.Series(rng.normal(0, .01, len(dates)), index=dates)
    pd.testing.assert_frame_equal(canonical_batch(daily, dates, benchmark),
        original['canonical_batch'](daily, dates, benchmark), check_exact=True)


def test_peer_count_breadth_and_strict_zero_boundaries():
    means, breadth = peer_without_self(np.array([.06, .06, .06]),
        np.array([3, 3, 3]), np.array([5, 6, 7]), np.array([.01, .01, .01]))
    assert np.isnan(means[0]) and np.isnan(breadth[0])
    assert means[1] == (.06-.01)/5 and breadth[1] == 2/5
    dates = pd.bdate_range('2024-01-01', periods=10)
    peer = pd.DataFrame(dict(sector='A', peer_count=5., peer_ret=0., peer_breadth=.5), index=dates)
    panel = peer_deterioration(peer, dates, pd.Series(0., index=dates))
    assert pd.isna(panel.peer_ebb.iloc[3])
    assert panel.peer_ebb.iloc[4:].eq(False).all()  # strict < 0 and < .5
    peer.peer_ret = -.01; peer.peer_breadth = .4
    assert peer_deterioration(peer, dates, pd.Series(0., index=dates)).peer_ebb.iloc[4:].eq(True).all()


def test_suspensions_industry_changes_and_future_rows_do_not_fabricate_evidence():
    dates = pd.bdate_range('2024-01-01', periods=25)
    daily = pd.DataFrame(dict(code='000001', date=dates, sector='A', ret_1d=-.01,
        observed=6., total_ret=-.12, advancing=0.))
    daily.loc[12:, 'sector'] = 'B'
    daily = daily.drop(index=7)
    benchmark = pd.Series(0., index=dates)
    result = canonical_batch(daily, dates, benchmark).set_index('date')
    assert dates[7] not in result.index
    assert not result.loc[dates[8]:dates[11], 'peer_known5'].any()
    assert not result.loc[dates[12]:dates[15], 'peer_known5'].any()
    assert result.at[dates[16], 'peer_known5']
    prefix = canonical_batch(daily[daily.date.le(dates[16])], dates[:17], benchmark.iloc[:17])
    pd.testing.assert_frame_equal(prefix, result.reset_index().loc[lambda x: x.date.le(dates[16]), prefix.columns].reset_index(drop=True), check_exact=True)


def test_report_revisions_use_latest_available_report_period_not_latest_event(tmp_path):
    pd.DataFrame({'报告期': ['20231231', '20230930', '20240331', '20240630'],
        '数据可用日期': ['20240301', '20240304', '20240307', '20240401'],
        '所处行业': ['A', 'OLDER_REVISION', 'B', 'FUTURE']}).to_csv(tmp_path/'000001.csv', index=False)
    dates = pd.to_datetime(['2024-02-29', '2024-03-01', '2024-03-04', '2024-03-07'])
    actual, evidence = report_industries_asof('000001', dates, tmp_path, dates[-1])
    assert actual.tolist() == [None, 'A', 'A', 'B']
    assert evidence['events'] == 3
    missing, _ = report_industries_asof('000002', dates, tmp_path, dates[-1])
    assert missing.tolist() == [None]*4


def tool_module():
    spec = importlib.util.spec_from_file_location('refresh_accepted465_peers', ROOT/'strategy/tools/refresh_accepted465_peers.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture_archive(tmp_path):
    data = tmp_path/'prices'; data.mkdir()
    funds = tmp_path/'fundamentals'; funds.mkdir()
    industry = tmp_path/'industry'; (industry/'stock_context').mkdir(parents=True)
    base = tmp_path/'sealed'; base.mkdir()
    dates = pd.bdate_range('2024-01-01', periods=125)
    codes = [f'{i:06d}' for i in range(1, 10)]
    for i, code in enumerate(codes):
        close = 20*np.cumprod(1+np.sin(np.arange(len(dates))+i)*.01)
        prices = pd.DataFrame(dict(datetime=dates, close=close, high=close+.2, low=close-.2,
            volume=1000., amount=close*1000*100))
        if code == '000009':  # no eligible quote history; still scanned, never counted as evidence
            prices = prices.iloc[:50]
        prices.to_csv(data/f'{code}_qfq.csv', index=False)
        if code != '000008':  # unknown report industry remains in the quote universe
            pd.DataFrame({'报告期': ['20231231'], '数据可用日期': ['20240101'],
                '所处行业': ['A']}).to_csv(funds/f'{code}.csv', index=False)
    pd.DataFrame(dict(datetime=dates, close=3000+np.arange(len(dates)))).to_csv(data/'sh000001_qfq.csv', index=False)
    old_end, end = dates[110], dates[115]
    source, stamps = [], []
    for code in codes:
        frame, _ = read_report_stock((code, str(data/f'{code}_qfq.csv'), str(dates[0].date()), str(old_end.date()), str(funds)))
        source.append(frame)
        stamps.append(dict(price=dict(path=str(data/f'{code}_qfq.csv'))))
    source = pd.concat(source, ignore_index=True)
    part = industry/'stock_context/part_000.parquet'; source.to_parquet(part, index=False)
    (industry/'input_hashes.json').write_text(json.dumps(stamps))
    cal = dates[70:111]
    benchmark = pd.Series((3000+np.arange(len(dates))), index=dates).pct_change(fill_method=None)
    daily = source[source.date.ge(cal[0])].copy()
    groups = daily[daily.sector.notna()].groupby(['date', 'sector']).ret_1d.agg(['sum', 'size'])
    advancing = daily[daily.sector.notna()].assign(up=lambda x: x.ret_1d.gt(0)).groupby(['date', 'sector']).up.sum()
    daily = daily.merge(groups.rename(columns={'sum': 'total_ret', 'size': 'observed'}), on=['date', 'sector'], how='left')
    daily = daily.merge(advancing.rename('advancing'), on=['date', 'sector'], how='left').sort_values(['code', 'date'])
    context = archived_kernels()['canonical_batch'](daily, cal, benchmark.reindex(cal))
    hashes = {}
    for month, frame in context.groupby(context.date.dt.strftime('%Y%m')):
        relative = f'months/month={month}/context.parquet'; target = base/relative; target.parent.mkdir(parents=True)
        frame.to_parquet(target, index=False); hashes[relative] = sha256(target)
    manifest = dict(start=str(cal[0].date()), end=str(old_end.date()), all_market_universe=True,
        coverage=dict(rows=len(context), codes=context.code.nunique(), days=len(cal)),
        validation=dict(known_mask_mismatches=0, ebb_mismatches=0),
        output_sha256=hashes, source_sha256={str(part): sha256(part)})
    (base/'manifest.json').write_text(json.dumps(manifest))
    return data, funds, industry, base, dates, old_end, end


def test_refresh_preserves_archive_and_is_consumable_by_daily_exit_context(tmp_path):
    data, funds, industry, base, dates, old_end, end = fixture_archive(tmp_path)
    protected = {p: sha256(p) for p in base.rglob('*') if p.is_file()}
    output = tmp_path/'extended'
    manifest = tool_module().refresh(base, industry, data, funds, output, str(end.date()), workers=1, batch_size=3)
    assert manifest['validation']['exact']
    assert manifest['stock_file_universe'] == 9
    assert manifest['coverage']['codes'] == 8
    assert manifest['appended_peer_rows'] == 8*5
    assert manifest['ranking_predictions_changed'] is False
    assert all(sha256(p) == digest for p, digest in protected.items())
    original_manifest = json.loads((base/'manifest.json').read_text())
    for relative, digest in original_manifest['output_sha256'].items():
        if 'month=202406' not in relative:
            assert sha256(output/relative) == digest
    old = DailyExitContext(data, base, dates)
    fresh = DailyExitContext(data, output, dates)
    assert fresh.observe('000001', old_end) == old.observe('000001', old_end)
    assert old.observe('000001', end)['peer_known'] is False
    assert fresh.observe('000001', end)['peer_known'] is True
    assert fresh.observe('000008', end)['peer_known'] is False
    with pytest.raises(ValueError, match='new directory'):
        tool_module().refresh(base, industry, data, funds, output, str(end.date()))


def test_source_drift_blocks_publication_and_exact_check_detects_one_ulp(tmp_path):
    data, funds, industry, base, dates, old_end, end = fixture_archive(tmp_path)
    p = data/'000001_qfq.csv'; frame = pd.read_csv(p)
    frame.loc[frame.datetime.eq(str(old_end.date())), 'close'] *= 1.001
    frame.to_csv(p, index=False)
    output = tmp_path/'drifted'
    with pytest.raises(ValueError, match='exact value mismatches'):
        tool_module().refresh(base, industry, data, funds, output, str(end.date()), workers=1)
    assert not (output/'manifest.json').exists()
    expected = pd.DataFrame(dict(code=['000001'], date=[old_end], value=[.1]))
    actual = expected.copy(); actual['value'] = np.nextafter(.1, 1.)
    with pytest.raises(ValueError, match='exact value mismatches'):
        require_exact_rows(actual, expected, ['value'], 'one ULP')


def test_live_refresh_does_not_import_research_or_read_signals():
    for path in [ROOT/'strategy/core/peer_exit_context.py', ROOT/'strategy/tools/refresh_accepted465_peers.py']:
        tree = ast.parse(path.read_text())
        imports = [n.module or '' for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
        assert not any('research_' in name or 'signal' in name or 'portfolio' in name for name in imports)
