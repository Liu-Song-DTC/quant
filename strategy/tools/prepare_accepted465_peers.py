"""Prepare accepted465's shared B2 peer context using the original refresher.

Reuse needs only the standard library. New contexts are generated from the
refresher's sealed defaults, never from a previous appended output. Pending
markers keep interrupted or rejected outputs out of subsequent reuse scans.
"""

import argparse
from contextlib import contextmanager
import csv
from dataclasses import dataclass
from datetime import date, datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from uuid import uuid4


ROOT = Path(__file__).resolve().parents[2]
KERNEL = 'strategy/core/peer_exit_context.py'
SOURCE_FILES = (KERNEL, 'strategy/core/fundamental.py',
                'strategy/core/fundamental_units.py', 'strategy/core/stock_pool.py')
PROFILE = 'strategy/config/deployment/accepted465.json'
MONTH = re.compile(r'months/month=(\d{6})/context\.parquet')
DIGEST = re.compile(r'[0-9a-f]{64}')


@dataclass(frozen=True)
class _Context:
    path: Path
    start: date
    end: date
    hashes: dict


def _log(message):
    print(f'[accepted465 peers] {message}', file=sys.stderr, flush=True)


def _resolve(root, value):
    path = Path(value)
    return (path if path.is_absolute() else root / path).resolve()


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _day(value):
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(value)


def _calendar(root):
    path = root / 'data/stock_data/backtrader_data/sh000001_qfq.csv'
    before = path.stat()
    with path.open(newline='', encoding='utf-8-sig') as stream:
        days = [_day(row['datetime'][:10]) for row in csv.DictReader(stream)]
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError('Index changed while reading the real session calendar')
    if not days or days != sorted(set(days)):
        raise ValueError('Real index calendar must be nonempty, unique and ordered')
    return days


def _pending(path):
    return path.parent / ('.pending-' + path.name)


def _validate(path, root, calendar):
    manifest = json.loads((path / 'manifest.json').read_text())
    if not isinstance(manifest, dict) or manifest.get('all_market_universe') is not True:
        raise ValueError('Context must have all_market_universe=true')
    validation = manifest.get('validation', {})
    for field in ('known_mask_mismatches', 'ebb_mismatches'):
        if (not isinstance(validation, dict)
                or type(validation.get(field)) not in (int, float)
                or validation[field] != 0):
            raise ValueError(f'Context validation requires {field}=0')
    start, end = _day(manifest['start']), _day(manifest['end'])
    if start > end or start not in calendar or end not in calendar:
        raise ValueError('Context bounds must be ordered real index sessions')
    sources = manifest.get('source_sha256', {})
    if not isinstance(sources, dict):
        raise ValueError('Missing canonical peer calculation source SHA map')
    for relative in SOURCE_FILES:
        recorded = [digest for name, digest in sources.items()
                    if isinstance(name, str) and (name.replace('\\', '/') == relative
                    or name.replace('\\', '/').endswith('/' + relative))]
        current_digest = _sha256(root / relative)
        if not recorded or any(digest != current_digest for digest in recorded):
            raise ValueError(f'Canonical peer source SHA missing or mismatched: {relative}')
    hashes = manifest.get('output_sha256')
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError('Missing month SHA map')
    expected = {f'months/month={day:%Y%m}/context.parquet'
                for day in calendar if start <= day <= end}
    present = {p.relative_to(path).as_posix()
               for p in (path / 'months').glob('month=*/context.parquet')}
    if set(hashes) != expected or present != expected:
        raise ValueError('Missing or unexpected month SHA/file')
    for relative, digest in hashes.items():
        if (MONTH.fullmatch(relative) is None or not isinstance(digest, str)
                or DIGEST.fullmatch(digest) is None):
            raise ValueError(f'Invalid month SHA entry: {relative}')
        if _sha256(path / relative) != digest:
            raise ValueError(f'Month SHA mismatch: {path / relative}')
    return _Context(path, start, end, hashes)


def _contexts(current, output_root, root, calendar):
    paths = [current]
    if output_root.is_dir():
        paths.extend(sorted(p for p in output_root.iterdir() if p.is_dir()))
    contexts, seen = [], set()
    for path in paths:
        path = path.resolve()
        if path in seen or _pending(path).exists():
            continue
        seen.add(path)
        try:
            contexts.append(_validate(path, root, calendar))
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            _log(f'skip {path}: {exc}')
    return sorted(contexts, key=lambda context: context.end, reverse=True)


def _check_prefix(old, new):
    """Check each old month, loading only a changed open month into memory."""
    if new.start > old.start or new.end < old.end:
        raise ValueError('Historical prefix bounds disappeared')
    for relative, digest in old.hashes.items():
        if relative not in new.hashes:
            raise ValueError(f'Historical prefix month disappeared: {relative}')
        if new.hashes[relative] == digest:
            continue
        if MONTH.fullmatch(relative).group(1) != old.end.strftime('%Y%m'):
            raise ValueError(f'Historical prefix closed month changed: {relative}')

        import pandas as pd
        import pyarrow as pa
        import pyarrow.compute as pc
        import pyarrow.parquet as pq

        def read_prefix(path):
            # ParquetFile avoids adding Hive partition columns from the path.
            with pq.ParquetFile(path) as parquet:
                table = parquet.read()
            # Later IPO nulls must not change the old prefix's pandas dtypes.
            cutoff = pa.scalar(pd.Timestamp(old.end), type=table.schema.field('date').type)
            frame = table.filter(pc.less_equal(table['date'], cutoff)).to_pandas()
            if frame.duplicated(['code', 'date']).any():
                raise ValueError(f'Historical prefix duplicate code/date: {path}')
            return frame.set_index(['code', 'date']).sort_index()

        before = read_prefix(old.path / relative)
        after = read_prefix(new.path / relative)
        try:
            pd.testing.assert_frame_equal(before, after, check_exact=True, rtol=0, atol=0)
        except AssertionError as exc:
            raise ValueError(f'Historical prefix changed: {relative}: {exc}') from exc


def _deduplicate(old, new):
    linked = 0
    for relative, digest in old.hashes.items():
        if new.hashes.get(relative) != digest:
            continue
        source, target = old.path / relative, new.path / relative
        temporary = target.with_name(target.name + '.link-' + uuid4().hex)
        try:
            if os.path.samefile(source, target):
                continue
            os.link(source, temporary)
            os.replace(temporary, target)
            linked += 1
        except OSError:
            # Copies are already validated; inability to hard-link is harmless.
            pass
        finally:
            temporary.unlink(missing_ok=True)
    _log(f'hard-linked {linked} identical months in {new.path}')


@contextmanager
def _refresh_lock(output_root):
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / '.prepare.lock').open('a') as lock:
        _log(f'waiting for refresh lock: {lock.name}')
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _subprocess_environment():
    # Same research-override exclusions as the shared execution environment.
    # Keep data selection on explicit CLI paths, not inherited QUANT_* policy.
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith(('QUANT_', 'POOL_'))
                   and key not in ('FACTOR_CONFIG_PATH', 'PYTHONPATH', 'PYTHONHOME',
                                   'PYTHONPYCACHEPREFIX')}
    environment.update(QUANT_ALT_NO_AUTOREFRESH='1', QUANT_ALT_VERSIONS_REQUIRED='1',
                       PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1',
                       OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    return environment


def ensure_peer_context(root=ROOT, end=None, current_path=None, output_root=None) -> Path:
    """Return an absolute validated context, creating a new immutable run if needed.

    No profile, active pointer, old context, signal, model or live JSON is written.
    A failed new run and its pending marker remain for diagnosis, never for reuse.
    """
    root = Path(root).resolve()
    if current_path is None:
        profile = json.loads((root / PROFILE).read_text())
        current_path = profile['peer_context']
    current = _resolve(root, current_path)
    output_root = _resolve(root, output_root or 'data/research/accepted465_peer_context_runs')

    def inspect():
        calendar = _calendar(root)
        required = _day(end) if end is not None else calendar[-1]
        if required not in calendar:
            raise ValueError(f'Requested end is not a real index session: {required}')
        contexts = _contexts(current, output_root, root, calendar)
        reusable = next((context for context in contexts
                         if context.start <= required <= context.end), None)
        return calendar, required, contexts, reusable

    calendar, required, contexts, reusable = inspect()
    if reusable is not None:
        _log(f'reuse {reusable.path} (end={reusable.end}, required={required})')
        return reusable.path
    if output_root.is_relative_to(current):
        raise ValueError('Output root must be outside the old context')
    with _refresh_lock(output_root):
        # Another caller may have published while this process waited.
        calendar, required, contexts, reusable = inspect()
        if reusable is not None:
            _log(f'reuse {reusable.path} after lock')
            return reusable.path
        name = f'accepted465_{required:%Y%m%d}_{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex}'
        output = output_root / name
        marker = _pending(output)
        marker.write_text(f'Unpublished: awaiting refresh and historical prefix checks for {required}\n')
        command = [sys.executable, '-B', str(root / 'strategy/tools/refresh_accepted465_peers.py'),
                   '--output', str(output), '--end', required.isoformat(), '--workers', '2',
                   '--data-path', str(root / 'data/stock_data/backtrader_data'),
                   '--fundamental-path', str(root / 'data/stock_data/fundamental_data')]
        _log(f'refresh to new directory: {output}')
        subprocess.run(command, cwd=root, check=True, stdout=sys.stderr, stderr=sys.stderr,
                       env=_subprocess_environment())
        new = _validate(output, root, calendar)
        if new.end != required:
            raise ValueError(f'Refresher did not cover requested end: {required}')
        if contexts:
            old = contexts[0]
            _check_prefix(old, new)
            _deduplicate(old, new)
        # Verify final files after link replacement before exposing the run.
        _validate(output, root, calendar)
        marker.unlink()
        _log(f'prepared {output} through {new.end}')
        return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--end', help='Real index session; defaults to the latest local index date')
    parser.add_argument('--current', type=Path)
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--print-path', action='store_true')
    args = parser.parse_args(argv)
    try:
        path = ensure_peer_context(root=ROOT, end=args.end, current_path=args.current,
                                   output_root=args.output_root)
    except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f'[accepted465 peers] preparation failed: {exc}\n')
    if args.print_path:
        print(path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
