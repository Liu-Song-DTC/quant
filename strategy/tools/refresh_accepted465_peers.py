"""Append canonical all-market peer evidence to a NEW accepted465 context.

No signals, ranking forecasts, accounts or old archives are changed. The old
stock-context prefix is retained to preserve pandas rolling arithmetic exactly.
Only new quote/report observations and new peer dates are materialized.
"""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import json
from pathlib import Path
import resource
import shutil
import sys
import time

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'strategy'))
from core.peer_exit_context import (SCHEMA, PEER_FIELDS, canonical_batch, is_report_stock,
    read_report_stock, require_exact_rows, sha256)

DEFAULT_BASE = Path('/home/liusong/quant_mainline_runs/full_market_peer_exit_context_20261120_202003')
DEFAULT_INDUSTRY = Path('/home/liusong/quant_mainline_runs/report_industry_context_latest_20261116')


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def link_or_copy(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copyfile(source, target)


def refresh(base, industry_archive, data_path, fundamental_path, output, end, workers=2, batch_size=100):
    start_clock = time.monotonic()
    base, industry_archive, data_path, fundamental_path, output = map(Path,
        (base, industry_archive, data_path, fundamental_path, output))
    if workers not in (1, 2) or batch_size < 1:
        raise ValueError('Use one or two workers and a positive batch_size')
    if output.exists():
        raise ValueError('Output must be a new directory; existing archives are never overwritten')
    if any(output.resolve().is_relative_to(p.resolve()) for p in (base, industry_archive, data_path, fundamental_path)):
        raise ValueError('Output must be outside every protected input directory')
    base_manifest = json.loads((base/'manifest.json').read_text())
    if base_manifest.get('all_market_universe') is not True:
        raise ValueError('Base context must cover the independent all-market universe')
    validation = base_manifest['validation']
    if validation['known_mask_mismatches'] or validation['ebb_mismatches']:
        raise ValueError('Base context failed canonical validation')
    old_end, end = pd.Timestamp(base_manifest['end']), pd.Timestamp(end)
    if end <= old_end:
        raise ValueError('Requested end must follow the sealed context end')
    index_path = data_path/'sh000001_qfq.csv'
    index_hash = sha256(index_path)
    index = pd.read_csv(index_path, usecols=['datetime', 'close'], parse_dates=['datetime']).set_index('datetime')
    if not index.index.is_unique or not index.index.is_monotonic_increasing:
        raise ValueError('Index calendar must be unique and ordered')
    if end not in index.index or old_end not in index.index:
        raise ValueError('Requested end and sealed end must both be real index sessions')
    calendar = pd.DatetimeIndex(index.index[(index.index >= base_manifest['start']) & (index.index <= end)])
    benchmark = index.close.pct_change(fill_method=None).reindex(calendar)
    overlap_dates = calendar[calendar <= old_end][-5:]
    if len(overlap_dates) != 5:
        raise ValueError('At least five sealed overlap sessions are required')
    stock_files = sorted(p for p in data_path.glob('*_qfq.csv') if is_report_stock(p.name[:-8]))
    if not stock_files:
        raise ValueError('No all-market QFQ stock files')
    # Protect every archived input, not just the newly computed date range.
    protected = {str(p): sha256(p) for p in [base/'manifest.json', Path(__file__),
        ROOT/'strategy/core/peer_exit_context.py', ROOT/'strategy/core/fundamental.py',
        ROOT/'strategy/core/fundamental_units.py', ROOT/'strategy/core/stock_pool.py']}
    for relative, digest in base_manifest['output_sha256'].items():
        p = base/relative
        if sha256(p) != digest:
            raise ValueError(f'Sealed peer month hash mismatch: {p}')
        protected[str(p)] = digest
    archived_sources = {p: h for p, h in base_manifest['source_sha256'].items()
        if '/stock_context/' in p and p.endswith('.parquet')}
    source_files = sorted((industry_archive/'stock_context').glob('*.parquet'))
    if set(map(str, source_files)) != set(archived_sources):
        raise ValueError('Industry archive must be the exact sealed stock-context source')
    for p in source_files:
        if sha256(p) != archived_sources[str(p)]:
            raise ValueError(f'Sealed industry source changed: {p}')
        protected[str(p)] = archived_sources[str(p)]
    industry_stamps = json.loads((industry_archive/'input_hashes.json').read_text())
    archived_codes = {Path(s['price']['path']).name[:-8] for s in industry_stamps}
    if not archived_codes.issubset({p.name[:-8] for p in stock_files}):
        raise ValueError('An archived all-market stock price file is missing')
    output.mkdir(parents=True)
    def progress(stage, **fields):
        print(json.dumps(dict(stage=stage, output=str(output), **fields), ensure_ascii=False), flush=True)
    save(output/'protocol.json', dict(base=str(base), industry_archive=str(industry_archive),
        start=base_manifest['start'], sealed_end=str(old_end.date()), end=str(end.date()),
        workers=workers, batch_size=batch_size, duckdb_memory_limit='256MB',
        all_market_universe=True, signals_read=False, ranking_predictions_changed=False,
        recipe='Original peer_without_self / peer_deterioration, complete original rolling prefix; no SQL rolling.',
        industry='Canonical FundamentalData.get_industry at every supplied report availability event, including revisions.',
        limitations=['Supplied report availability/revision and real-QFQ provenance remain inherited.',
            'Missing own sessions and unknown industry remain unknown; no synthetic sessions or forward fills.']))
    progress('read_all_market_quote_and_report_tail', stocks=len(stock_files))
    tasks = [(p.name[:-8], str(p), str(overlap_dates[0].date()), str(end.date()), str(fundamental_path)) for p in stock_files]
    frames, receipts = [], []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for i, (frame, receipt) in enumerate(executor.map(read_report_stock, tasks, chunksize=1), 1):
            frames.append(frame); receipts.append(receipt)
            if i % 500 == 0:
                progress('read_all_market_quote_and_report_tail', done=i, stocks=len(stock_files))
    tail = pd.concat(frames, ignore_index=True)
    tail = tail.sort_values(['code', 'date']).reset_index(drop=True)
    if tail.duplicated(['code', 'date']).any():
        raise ValueError('Duplicate current source observations')
    if not tail.date.isin(calendar).all():
        raise ValueError('Stock source has dates outside the real index calendar')
    new = tail[tail.date.gt(old_end)].copy()
    if new.empty or new.date.max() != end:
        raise ValueError('No current all-market source observations on requested end')
    if set(new.date) != set(calendar[calendar > old_end]):
        raise ValueError('An appended index session has no all-market quote observations')
    source_dir = output/'stock_context'; source_dir.mkdir()
    new_path = source_dir/'appended.parquet'
    new.to_parquet(new_path, index=False, compression='zstd')
    save(output/'input_hashes.json', receipts)
    source = str(industry_archive/'stock_context/*.parquet')
    expected_day_parts = []
    for relative in base_manifest['output_sha256']:
        month = relative.split('month=')[1].split('/')[0]
        if month == old_end.strftime('%Y%m'):
            expected_day_parts.append(pd.read_parquet(base/relative))
    expected_day = pd.concat(expected_day_parts, ignore_index=True)
    expected_day = expected_day[expected_day.date.eq(old_end)]
    if expected_day.empty:
        raise ValueError('Sealed end has no peer evidence to validate')
    # Daily SQL sums are exactly the archived builder's primitive aggregation.
    # Old and new groups are computed separately: new rows never perturb old sums.
    with duckdb.connect(config={'threads': 1, 'memory_limit': '256MB'}) as con:
        overlap = con.execute('SELECT code,date,sector,ret_1d,active FROM read_parquet(?) WHERE date BETWEEN ? AND ?',
            [source, overlap_dates[0].to_pydatetime(), old_end.to_pydatetime()]).fetchdf()
        source_parity = require_exact_rows(tail[tail.date.le(old_end)], overlap,
            ['sector', 'ret_1d', 'active'], 'Current QFQ/report overlap')
        con.execute('CREATE TEMP TABLE peer_groups AS SELECT date,sector,count(*) AS observed,sum(ret_1d) AS total_ret, '
            'sum(CASE WHEN ret_1d>0 THEN 1 ELSE 0 END) AS advancing FROM read_parquet(?) '
            'WHERE date BETWEEN ? AND ? AND active=1 AND isfinite(ret_1d) AND sector IS NOT NULL GROUP BY date,sector',
            [source, calendar[0].to_pydatetime(), old_end.to_pydatetime()])
        con.execute('INSERT INTO peer_groups SELECT date,sector,count(*) AS observed,sum(ret_1d) AS total_ret, '
            'sum(CASE WHEN ret_1d>0 THEN 1 ELSE 0 END) AS advancing FROM read_parquet(?) '
            'WHERE active=1 AND isfinite(ret_1d) AND sector IS NOT NULL GROUP BY date,sector', [str(new_path)])
        codes = sorted(archived_codes | set(new.code))
        forecasts, sentinels, observed_codes = [], [], set()
        for offset in range(0, len(codes), batch_size):
            batch = codes[offset:offset+batch_size]
            query = ('SELECT s.code,s.date,s.sector,s.ret_1d,g.observed,g.total_ret,g.advancing '
                'FROM read_parquet(?) s LEFT JOIN peer_groups g ON s.date=g.date AND s.sector=g.sector '
                'WHERE s.date BETWEEN ? AND ? AND s.code BETWEEN ? AND ? AND s.active=1 AND isfinite(s.ret_1d) '
                'ORDER BY s.code,s.date')
            old_daily = con.execute(query, [source, calendar[0].to_pydatetime(), old_end.to_pydatetime(), batch[0], batch[-1]]).fetchdf()
            new_daily = con.execute(query, [str(new_path), (old_end+pd.Timedelta(days=1)).to_pydatetime(), end.to_pydatetime(), batch[0], batch[-1]]).fetchdf()
            daily = pd.concat([old_daily, new_daily], ignore_index=True).sort_values(['code', 'date']).reset_index(drop=True)
            if daily.empty:
                continue
            observed_codes.update(daily.code.unique())
            calculated = canonical_batch(daily, calendar, benchmark)
            sentinels.append(calculated[calculated.date.eq(old_end)])
            forecasts.append(calculated[calculated.date.gt(old_end)])
            progress('canonical_full_prefix', done=min(offset+batch_size, len(codes)), stocks=len(codes))
        peer_parity = require_exact_rows(pd.concat(sentinels, ignore_index=True), expected_day,
            ['sector', 'peer_count', *PEER_FIELDS], 'Sealed end canonical peers')
    appended = pd.concat(forecasts, ignore_index=True)
    if appended.empty or appended.duplicated(['code', 'date']).any():
        raise ValueError('Empty/duplicate appended peer evidence')
    appended_coverage = require_exact_rows(appended, new, [], 'All-market appended source coverage')
    # Complete closed months stay byte-identical. An open month is written to a
    # new inode with its existing rows unchanged, plus strictly later new rows.
    new_months = set(appended.date.dt.strftime('%Y%m'))
    hashes = {}
    historical_rows = int(base_manifest['coverage']['rows'])
    for relative, digest in base_manifest['output_sha256'].items():
        month = relative.split('month=')[1].split('/')[0]
        if month not in new_months:
            link_or_copy(base/relative, output/relative)
            hashes[relative] = digest
    for month, part in appended.groupby(appended.date.dt.strftime('%Y%m'), sort=True):
        relative = f'months/month={month}/context.parquet'
        if relative in base_manifest['output_sha256']:
            original = pd.read_parquet(base/relative)
            combined = pd.concat([original, part], ignore_index=True)
            require_exact_rows(combined[combined.date.le(old_end)], original,
                ['sector', 'peer_count', *PEER_FIELDS], 'Preserved sealed month')
        else:
            combined = part
        target = output/relative; target.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pandas(combined, schema=SCHEMA, preserve_index=False), target, compression='zstd')
        hashes[relative] = sha256(target)
    # Publication is the final manifest. Failure leaves an unconsumable new
    # directory; it never changes the original archive or an active pointer.
    for path, digest in protected.items():
        if sha256(path) != digest:
            raise ValueError(f'Protected archive changed during refresh: {path}')
    for receipt in receipts:
        stat = Path(receipt['price_path']).stat()
        if (stat.st_size, stat.st_mtime_ns) != (receipt['size'], receipt['mtime_ns']):
            raise ValueError(f'Price source changed during refresh: {receipt["code"]}')
        p = Path(receipt['financial_path'])
        if (sha256(p) if p.exists() else None) != receipt['financial_sha256']:
            raise ValueError(f'Financial source changed during refresh: {receipt["code"]}')
    if sha256(index_path) != index_hash:
        raise ValueError('Index source changed during refresh')
    if sorted(p.name for p in data_path.glob('*_qfq.csv') if is_report_stock(p.name[:-8])) != [p.name for p in stock_files]:
        raise ValueError('All-market file universe changed during refresh')
    for relative, digest in hashes.items():
        if sha256(output/relative) != digest:
            raise ValueError(f'Output month changed before publication: {relative}')
    per_day = new.groupby('date').code.nunique()
    manifest = dict(start=base_manifest['start'], end=str(end.date()), all_market_universe=True,
        coverage=dict(rows=historical_rows+len(appended), codes=len(observed_codes), days=len(calendar)),
        validation=dict(old_rows=peer_parity['rows'], known_mask_mismatches=0, ebb_mismatches=0,
            return5_max_error=0.0, relative5_max_error=0.0, breadth5_max_error=0.0,
            sealed_end=str(old_end.date()), exact=True, source_overlap=source_parity,
            canonical_peer_overlap=peer_parity, appended_source_coverage=appended_coverage,
            base_validation=validation),
        output_sha256=hashes,
        source_sha256={**protected, str(index_path): index_hash,
            str(Path(__file__)): sha256(__file__),
            str(ROOT/'strategy/core/peer_exit_context.py'): sha256(ROOT/'strategy/core/peer_exit_context.py'),
            str(ROOT/'strategy/core/fundamental.py'): sha256(ROOT/'strategy/core/fundamental.py'),
            str(ROOT/'strategy/core/stock_pool.py'): sha256(ROOT/'strategy/core/stock_pool.py'),
            str(new_path): sha256(new_path), str(output/'input_hashes.json'): sha256(output/'input_hashes.json')},
        base_context=str(base), stock_file_universe=len(stock_files),
        resources=dict(workers=workers, batch_size=batch_size, duckdb_memory_limit='256MB',
            parent_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            max_child_peak_rss_kib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss),
        appended_source_dates={str(day.date()): int(count) for day, count in per_day.items()},
        appended_peer_rows=len(appended), production_changed=False, full_account_backtest=False,
        ranking_predictions_changed=False, completed_at=datetime.now().astimezone().isoformat(),
        elapsed_seconds=time.monotonic()-start_clock)
    save(output/'manifest.json', manifest)
    progress('completed', manifest=str(output/'manifest.json'), appended_rows=len(appended), exact_overlap=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-context', type=Path, default=DEFAULT_BASE)
    parser.add_argument('--industry-archive', type=Path, default=DEFAULT_INDUSTRY)
    parser.add_argument('--data-path', type=Path, default=ROOT/'data/stock_data/backtrader_data')
    parser.add_argument('--fundamental-path', type=Path, default=ROOT/'data/stock_data/fundamental_data')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--end', required=True, help='Completed real index session, e.g. 2026-11-20')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=100)
    args = parser.parse_args()
    refresh(args.base_context, args.industry_archive, args.data_path, args.fundamental_path,
        args.output, args.end, args.workers, args.batch_size)


if __name__ == '__main__':
    main()
