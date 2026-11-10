"""Read-only attribution of two sealed percentage-contract runs.

Example (bounded signal scan; omit --max-chunks for complete coverage)::

    .venv/bin/python -B strategy/analysis/attribute_percent_repair.py \
        --baseline .review_runs/arm_fundscore_repro_20261106/corrected \
        --candidate .review_runs/arm_fund_contract_20261107/corrected \
        --max-chunks 2

Only the two output artifacts are written. No production modules are imported.
CSV contains the union of actual selection (date, code) keys, including missing
signals; JSON contains coverage, first differences and annual summaries. Signal
dates are selection dates, not next-day fill dates. Delta always means B - A.
Annual NAV gap increments telescope to the final gap (relative to initial gap).
They are accounting contributions, not causal estimates of a particular field.
"""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / '.review_runs/arm_fund_contract_20261107/corrected/percent_signal_attribution'
METRICS = ('score', 'factor_value', 'ml_score')
EXTRA_GATES = ('profit_declining', 'ma_trend_up')


def clean(value):
    """Strict JSON: represent missing/non-finite values by null."""
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def dates(series):
    return pd.to_datetime(series, errors='raise').dt.strftime('%Y-%m-%d')


def read_table(path):
    frame = pd.read_csv(path, dtype={'code': str})
    frame['date'] = dates(frame['date'])
    if 'code' in frame:
        frame['code'] = frame['code'].str.strip().str.zfill(6)
        if frame['code'].isna().any():
            raise ValueError(f'Missing stock code: {path}')
    return frame


def different(a, b, tolerance):
    # Missing on both sides is equal; missing on only one side is a change.
    if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
        return ~pd.Series(np.isclose(a, b, atol=tolerance, rtol=0, equal_nan=True), index=a.index)
    return ~(a.eq(b) | (a.isna() & b.isna()))


def event_diff(a, b):
    """Compare daily multisets, preserving duplicate fills and ignoring row order."""
    common = sorted(set(a.columns) & set(b.columns) - {'date'})
    def grouped(frame):
        result = {}
        for date, group in frame.groupby('date', sort=True):
            result[date] = Counter(
                tuple(None if pd.isna(v) else v for v in row)
                for row in group[common].itertuples(index=False, name=None))
        return result
    ga, gb = grouped(a), grouped(b)
    differing = []
    annual = {}
    for date in sorted(set(ga) | set(gb)):
        ca, cb = ga.get(date, Counter()), gb.get(date, Counter())
        only_a, only_b = ca - cb, cb - ca
        year = annual.setdefault(date[:4], {'baseline_rows': 0, 'candidate_rows': 0,
                                          'baseline_only_rows': 0, 'candidate_only_rows': 0})
        for key, val in [('baseline_rows', sum(ca.values())), ('candidate_rows', sum(cb.values())),
                         ('baseline_only_rows', sum(only_a.values())), ('candidate_only_rows', sum(only_b.values()))]:
            year[key] += val
        if only_a or only_b:
            differing.append({'date': date,
                              'baseline_only': [{'row': dict(zip(common, row)), 'count': n} for row, n in only_a.items()],
                              'candidate_only': [{'row': dict(zip(common, row)), 'count': n} for row, n in only_b.items()]})
    return {'first_difference': differing[0] if differing else None,
            'different_dates': len(differing), 'annual': annual,
            'baseline_only_columns': sorted(set(a) - set(b)),
            'candidate_only_columns': sorted(set(b) - set(a))}


def nav_diff(a, b, tolerance):
    for frame in (a, b):
        if frame['date'].duplicated().any():
            raise ValueError('Duplicate NAV dates')
        if frame.empty or not (pd.to_numeric(frame['nav'], errors='raise') > 0).all():
            raise ValueError('NAV must be nonempty and positive')
    nav = a[['date', 'nav']].merge(b[['date', 'nav']], on='date', how='outer',
                                  suffixes=('_baseline', '_candidate'), indicator=True).sort_values('date')
    changed = different(nav.nav_baseline, nav.nav_candidate, tolerance)
    first = nav.loc[changed].head(1).to_dict('records')
    # Do not silently bridge absent dates in annual return/contribution calculations.
    if not nav['_merge'].eq('both').all():
        return {'first_difference': first[0] if first else None,
                'unmatched_dates': nav.loc[nav['_merge'].ne('both')].to_dict('records'),
                'annual': [], 'annual_unavailable_reason': 'NAV calendars differ'}
    gap = nav.nav_candidate - nav.nav_baseline
    previous_a, previous_b = float(nav.iloc[0].nav_baseline), float(nav.iloc[0].nav_candidate)
    initial_gap = previous_b - previous_a
    annual = []
    for year, group in nav.groupby(nav.date.str[:4], sort=True):
        end_a, end_b = float(group.iloc[-1].nav_baseline), float(group.iloc[-1].nav_candidate)
        annual.append({'year': year, 'baseline_return': end_a / previous_a - 1,
                       'candidate_return': end_b / previous_b - 1,
                       'return_delta': end_b / previous_b - end_a / previous_a,
                       'year_end_nav_gap': end_b - end_a,
                       'nav_gap_contribution': (end_b - previous_b) - (end_a - previous_a)})
        previous_a, previous_b = end_a, end_b
    return {'first_difference': first[0] if first else None, 'different_dates': int(changed.sum()),
            'initial_nav_gap': initial_gap, 'final_nav_gap': float(gap.iloc[-1]),
            'contribution_sum': sum(row['nav_gap_contribution'] for row in annual),
            'annual': annual}


def extract_signals(path, keys, chunksize, max_chunks):
    columns = list(pd.read_csv(path, nrows=0).columns)
    required = {'date', 'code', 'buy', *METRICS}
    if required - set(columns):
        raise ValueError(f'{path}: missing columns {sorted(required - set(columns))}')
    gates = [c for c in columns if 'gate' in c.lower() or c in EXTRA_GATES]
    usecols = ['date', 'code', 'buy', *METRICS, *gates]
    target = pd.MultiIndex.from_frame(keys[['date', 'code']])
    hits, scanned, chunks, complete = [], 0, 0, False
    with pd.read_csv(path, usecols=usecols, dtype={'code': str}, chunksize=chunksize) as reader:
        while max_chunks is None or chunks < max_chunks:
            try:
                chunk = next(reader)
            except StopIteration:
                complete = True
                break
            scanned += len(chunk)
            chunks += 1
            chunk['date'] = dates(chunk['date'])
            chunk['code'] = chunk['code'].str.strip().str.zfill(6)
            hit = chunk.loc[pd.MultiIndex.from_frame(chunk[['date', 'code']]).isin(target)]
            if not hit.empty:
                hits.append(hit)
    result = pd.concat(hits, ignore_index=True) if hits else pd.DataFrame(columns=usecols)
    if result.duplicated(['date', 'code']).any():
        raise ValueError(f'Duplicate selected signal keys: {path}')
    return result, {'path': str(path), 'rows_scanned': scanned, 'chunks_scanned': chunks,
                    'scan_complete': complete, 'matched_keys': len(result),
                    'missing_columns': [], 'gate_columns': gates}


def bool_values(series):
    mapped = series.astype('string').str.lower().map({'true': True, 'false': False, '1': True,
                                                    '0': False, '1.0': True, '0.0': False})
    if (series.notna() & mapped.isna()).any():
        raise ValueError('Unrecognized buy boolean')
    return mapped.astype('boolean')


def summarize_signals(panel, fields):
    paired = panel.signal_present_baseline & panel.signal_present_candidate
    result = {'keys': len(panel), 'paired_keys': int(paired.sum()),
              'baseline_missing': int((~panel.signal_present_baseline).sum()),
              'candidate_missing': int((~panel.signal_present_candidate).sum()), 'fields': {}}
    for field in fields:
        delta = panel.loc[paired, f'{field}_delta'].dropna()
        result['fields'][field] = {
            'changed': int(panel.loc[paired, f'{field}_changed'].sum()),
            'numeric_pairs': len(delta), 'mean_delta': delta.mean(),
            'mean_abs_delta': delta.abs().mean(), 'max_abs_delta': delta.abs().max()}
    a, b = panel.loc[paired, 'buy_baseline'], panel.loc[paired, 'buy_candidate']
    result['buy'] = {'comparable': int((a.notna() & b.notna()).sum()),
                     'false_to_true': int(((a == False) & (b == True)).sum()),
                     'true_to_false': int(((a == True) & (b == False)).sum())}
    return result


def attribute(args):
    baseline, candidate = args.baseline.resolve(), args.candidate.resolve()
    frames = {name: [read_table(root / name) for root in (baseline, candidate)]
              for name in ('equity_curve.csv', 'portfolio_selections.csv', 'execution_fills.csv')}
    selections = frames['portfolio_selections.csv']
    keys = pd.concat([s[['date', 'code']] for s in selections]).drop_duplicates()
    if args.start_date:
        keys = keys[keys.date.ge(args.start_date)]
    if args.end_date:
        keys = keys[keys.date.le(args.end_date)]
    panel = keys.sort_values(['date', 'code']).reset_index(drop=True)
    for side, selection in zip(('baseline', 'candidate'), selections):
        if selection.duplicated(['date', 'code']).any():
            raise ValueError('Duplicate selection keys')
        selected = pd.MultiIndex.from_frame(selection[['date', 'code']])
        panel[f'selected_{side}'] = pd.MultiIndex.from_frame(panel[['date', 'code']]).isin(selected)
    paths = [args.baseline_signals or baseline.parent / 'strategy/rolling_validation_results/backtest_signals.csv',
             args.candidate_signals or candidate.parent / 'strategy/rolling_validation_results/backtest_signals.csv']
    output_paths = [args.output_prefix.with_suffix(ext).resolve() for ext in ('.json', '.csv')]
    inputs = [root / name for root in (baseline, candidate) for name in frames] + paths
    if any(out == Path(inp).resolve() for out in output_paths for inp in inputs):
        raise ValueError('Output must not overwrite input')
    coverage, signals = {}, []
    for side, path in zip(('baseline', 'candidate'), paths):
        signal, info = extract_signals(path, keys, args.chunksize, args.max_chunks)
        coverage[side] = info
        signals.append(signal)
    fields = list(METRICS) + sorted(set(coverage['baseline']['gate_columns']) |
                                  set(coverage['candidate']['gate_columns']))
    for side, signal in zip(('baseline', 'candidate'), signals):
        signal = signal.reindex(columns=['date', 'code', 'buy', *fields])
        signal[f'signal_present_{side}'] = True
        signal = signal.rename(columns={c: f'{c}_{side}' for c in ['buy', *fields]})
        panel = panel.merge(signal, on=['date', 'code'], how='left', validate='one_to_one')
        panel[f'signal_present_{side}'] = panel[f'signal_present_{side}'].eq(True)
        panel[f'buy_{side}'] = bool_values(panel[f'buy_{side}'])
    paired = panel.signal_present_baseline & panel.signal_present_candidate
    for field in fields:
        a, b = panel[f'{field}_baseline'], panel[f'{field}_candidate']
        # Fully matched boolean gates retain numpy bool dtype after the merge.
        # Convert BEFORE subtraction: False -> True is +1, True -> False is -1.
        numeric_a = pd.to_numeric(a, errors='coerce').astype('float64')
        numeric_b = pd.to_numeric(b, errors='coerce').astype('float64')
        panel[f'{field}_delta'] = (numeric_b - numeric_a).where(paired)
        panel[f'{field}_changed'] = different(a, b, args.atol) & paired
    panel['buy_flip'] = (panel.buy_baseline.ne(panel.buy_candidate) & paired).astype('boolean')
    changed = panel[[f'{f}_changed' for f in fields]].any(axis=1) | panel.buy_flip.fillna(False)
    first = panel.loc[changed].head(1).to_dict('records')
    report = {'baseline': str(baseline), 'candidate': str(candidate), 'delta_direction': 'candidate - baseline',
              'atol': args.atol, 'signal_scope': {'start_date': args.start_date, 'end_date': args.end_date,
              'keys': 'union of actual selections; same selection date, no fill-date shift'},
              'interpretation': 'Descriptive association, not causal field attribution. NAV contributions telescope; returns do not add.',
              'nav': nav_diff(*frames['equity_curve.csv'], args.atol),
              'selections': event_diff(*selections), 'fills': event_diff(*frames['execution_fills.csv']),
              'signal_coverage': coverage,
              'signals': {'first_observed_difference': first[0] if first else None,
                          'first_difference_is_global': all(c['scan_complete'] for c in coverage.values())
                          and not (args.start_date or args.end_date),
                          'summary': summarize_signals(panel, fields),
                          'annual': {year: summarize_signals(group, fields)
                                     for year, group in panel.groupby(panel.date.str[:4])}}}
    for side in ('baseline', 'candidate'):
        coverage[side]['unmatched_keys'] = int((~panel[f'signal_present_{side}']).sum())
        coverage[side]['unmatched_interpretation'] = ('absent from complete file' if coverage[side]['scan_complete']
                                                     else 'not observed in bounded scan; absence unproven')
    for out in output_paths:
        out.parent.mkdir(parents=True, exist_ok=True)
    output_paths[0].write_text(json.dumps(clean(report), ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    panel.to_csv(output_paths[1], index=False)
    return report


def self_test():
    """Small end-to-end regression fixture; writes only to a temporary directory."""
    import tempfile

    with tempfile.TemporaryDirectory(prefix='percent-attribution-', dir='/tmp') as temp:
        root = Path(temp)
        for side in ('baseline', 'candidate'):
            directory = root / side
            directory.mkdir()
            pd.DataFrame({'date': ['2021-01-04', '2021-12-31', '2022-12-30'],
                          'nav': [100, 110, 121] if side == 'baseline' else [100, 100, 90]}
                         ).to_csv(directory / 'equity_curve.csv', index=False)
            pd.DataFrame({'date': ['2021-01-04'] * 3,
                          'code': ['000001', '000002', '000003'],
                          'weight': [0.3] * 3, 'score': [0.5] * 3}
                         ).to_csv(directory / 'portfolio_selections.csv', index=False)
            pd.DataFrame({'date': ['2021-01-05'], 'code': ['000001'],
                          'side': ['BUY'], 'shares': [100]}
                         ).to_csv(directory / 'execution_fills.csv', index=False)
            pd.DataFrame({'date': ['2021-01-04'] * 3,
                          'code': ['000001', '000002', '000003'],
                          'buy': [True, False, True] if side == 'baseline' else [False, True, True],
                          'score': [0.5, 0.4, 0.3] if side == 'baseline' else [0.4, 0.4, 0.3],
                          'factor_value': [0.2] * 3, 'ml_score': [0.1] * 3,
                          'gate_quality': [0.5] * 3,
                          'profit_declining': [False, True, True] if side == 'baseline' else [True, False, True],
                          'ma_trend_up': [True, False, True]}
                         ).to_csv(directory / 'signals.csv', index=False)
        args = argparse.Namespace(
            baseline=root / 'baseline', candidate=root / 'candidate',
            baseline_signals=root / 'baseline/signals.csv',
            candidate_signals=root / 'candidate/signals.csv',
            output_prefix=root / 'result', chunksize=2, max_chunks=None,
            start_date=None, end_date=None, atol=1e-10)
        # All selected keys exist in both arms: no missing merge rows to coerce
        # bool to object/float and inadvertently hide the numpy bool subtraction.
        for chunksize in (1, 2, 10):
            args.chunksize = chunksize
            report = attribute(args)
            summary = report['signals']['summary']
            assert summary['paired_keys'] == summary['keys'] == 3
            assert summary['baseline_missing'] == summary['candidate_missing'] == 0
            assert all(c['scan_complete'] for c in report['signal_coverage'].values())
            assert report['signals']['first_difference_is_global']
            assert summary['buy']['true_to_false'] == summary['buy']['false_to_true'] == 1
            gate = summary['fields']['profit_declining']
            assert gate['changed'] == 2 and gate['numeric_pairs'] == 3
            assert gate['mean_delta'] == 0 and gate['max_abs_delta'] == 1
            assert summary['fields']['ma_trend_up']['changed'] == 0
            panel = pd.read_csv(root / 'result.csv', dtype={'code': str})
            assert panel.code.tolist() == ['000001', '000002', '000003']
            assert panel.profit_declining_delta.tolist() == [1.0, -1.0, 0.0]
            assert report['nav']['contribution_sum'] == report['nav']['final_nav_gap'] == -31
        args.chunksize, args.max_chunks = 1, 1
        report = attribute(args)
        assert report['signals']['summary']['paired_keys'] == 1
        assert report['signals']['summary']['baseline_missing'] == 2
        assert not report['signals']['first_difference_is_global']
    print('PASS: full matching boolean gates (+1/-1/0), buy flips, chunk sizes 1/2/10, '
          'NAV annual bridge, leading zeros, bounded coverage')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--self-test', action='store_true', help='Run temporary lightweight regression fixture')
    parser.add_argument('--baseline', type=Path, help='Baseline corrected directory')
    parser.add_argument('--candidate', type=Path, help='Candidate corrected directory')
    parser.add_argument('--baseline-signals', type=Path)
    parser.add_argument('--candidate-signals', type=Path)
    parser.add_argument('--output-prefix', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--chunksize', type=int, default=100_000)
    parser.add_argument('--max-chunks', type=int, help='Bound scan per file; missing keys remain unverified')
    parser.add_argument('--start-date', help='YYYY-MM-DD; limits signal keys only')
    parser.add_argument('--end-date', help='YYYY-MM-DD; limits signal keys only')
    parser.add_argument('--atol', type=float, default=1e-10, help='Absolute numeric tolerance; rtol=0')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.baseline is None or args.candidate is None:
        parser.error('--baseline and --candidate are required unless --self-test is used')
    if args.chunksize <= 0 or (args.max_chunks is not None and args.max_chunks <= 0) or args.atol < 0:
        parser.error('chunksize/max-chunks must be positive; atol must be nonnegative')
    for name in ('start_date', 'end_date'):
        value = getattr(args, name)
        if value:
            setattr(args, name, pd.Timestamp(value).strftime('%Y-%m-%d'))
    if args.start_date and args.end_date and args.start_date > args.end_date:
        parser.error('start-date must not exceed end-date')
    report = attribute(args)
    print(json.dumps(clean({'output_prefix': str(args.output_prefix), 'nav_first_difference': report['nav']['first_difference'],
                            'signal_coverage': report['signal_coverage'],
                            'signals': report['signals']['summary']}), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
