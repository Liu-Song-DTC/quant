"""Only fixture contexts are generated; the real refresher is always mocked."""

from datetime import date
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import prepare_accepted465_peers as prepare


SESSIONS = ['2026-10-29', '2026-10-30', '2026-11-13', '2026-11-19', '2026-11-20', '2026-11-23']
SCHEMA = pa.schema([
    ('code', pa.string()), ('date', pa.timestamp('ns')), ('sector', pa.string()),
    ('peer_count', pa.float64()), ('peer_return5', pa.float64()),
    ('peer_relative5', pa.float64()), ('peer_breadth5', pa.float64()),
    ('peer_known5', pa.bool_()), ('peer_ebb', pa.bool_()),
])


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def snapshot(path):
    return {p.relative_to(path).as_posix(): p.read_bytes()
            for p in path.rglob('*') if p.is_file()}


def write_context(root, path, end, mutate=None):
    path.mkdir(parents=True)
    frame = pd.DataFrame([
        {'code': code, 'date': pd.Timestamp(day), 'sector': 'industry',
         'peer_count': 6., 'peer_return5': .02 if code == '000001' else np.nan,
         'peer_relative5': -.01, 'peer_breadth5': .4,
         'peer_known5': code == '000001', 'peer_ebb': False if code == '000001' else None}
        for day in SESSIONS if day <= end for code in ['000001', '600001']])
    if mutate is not None:
        frame = mutate(frame)
    hashes = {}
    for month, part in frame.groupby(frame.date.dt.strftime('%Y%m')):
        relative = f'months/month={month}/context.parquet'
        target = path / relative
        target.parent.mkdir(parents=True)
        pq.write_table(pa.Table.from_pandas(part, schema=SCHEMA, preserve_index=False),
                       target, compression='zstd')
        hashes[relative] = sha(target)
    manifest = {'start': SESSIONS[0], 'end': end, 'all_market_universe': True,
                'validation': {'known_mask_mismatches': 0, 'ebb_mismatches': 0},
                'source_sha256': {str(root / relative): sha(root / relative)
                                  for relative in prepare.SOURCE_FILES},
                'output_sha256': hashes}
    (path / 'manifest.json').write_text(json.dumps(manifest))
    return path


@pytest.fixture
def tree(tmp_path, monkeypatch):
    root = tmp_path / 'quant'
    for relative in ['strategy/core', 'strategy/tools', 'strategy/config/deployment',
                     'data/stock_data/backtrader_data', 'data/stock_data/fundamental_data']:
        (root / relative).mkdir(parents=True)
    for relative in prepare.SOURCE_FILES:
        (root / relative).write_text(f'# fixture canonical dependency: {relative}\n')
    (root / 'strategy/tools/refresh_accepted465_peers.py').write_text('# never executed\n')
    index = root / 'data/stock_data/backtrader_data/sh000001_qfq.csv'
    index.write_text('datetime,close\n' + ''.join(f'{day},1.0\n' for day in SESSIONS))
    current = write_context(root, root / 'data/research/accepted465_peer_context', '2026-11-20')
    profile = root / prepare.PROFILE
    profile.write_text(json.dumps({'peer_context': current.relative_to(root).as_posix()}))
    live = root / 'trade_orders.json'
    live.write_text('{"protected": "live account"}\n')
    protected = {profile: profile.read_bytes(), live: live.read_bytes()}
    previous = snapshot(current)

    def forbidden_refresh(*args, **kwargs):
        pytest.fail('Real refresher must never execute in these tests')

    monkeypatch.setattr(prepare.subprocess, 'run', forbidden_refresh)
    monkeypatch.setattr(prepare, 'ROOT', root)
    yield root, current, root / 'data/research/accepted465_peer_context_runs'
    assert snapshot(current) == previous, 'old context must stay byte-identical'
    assert all(path.read_bytes() == contents for path, contents in protected.items())


def refresh_mock(monkeypatch, root, mutate=None, failure=None):
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        output = Path(command[command.index('--output') + 1])
        assert not output.exists()
        assert prepare._pending(output).exists()
        if failure is not None:
            output.mkdir()
            (output / 'partial').write_text('unfinished')
            raise failure
        end = command[command.index('--end') + 1]
        write_context(root, output, end, mutate=mutate)
        print('fixture refresher log', file=kwargs['stdout'])
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(prepare.subprocess, 'run', run)
    return calls


def test_reuse_profile_context_no_imports_no_writes_no_refresh(tree, monkeypatch, capsys):
    root, current, runs = tree
    # The normal reuse path must not need pandas, pyarrow or the math module.
    import builtins
    real_import = builtins.__import__

    def stdlib_only(name, *args, **kwargs):
        assert name.split('.')[0] not in {'pandas', 'pyarrow', 'core'}
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', stdlib_only)
    path = prepare.ensure_peer_context(root, end='2026-11-20')
    assert path == current.resolve()
    assert not runs.exists()
    captured = capsys.readouterr()
    assert captured.out == ''
    assert 'reuse' in captured.err


def test_explicit_relative_current_and_output_root(tree):
    root, current, _ = tree
    assert prepare.ensure_peer_context(root, end=date(2026, 11, 19),
        current_path=current.relative_to(root), output_root='other_runs') == current
    assert not (root / 'other_runs').exists()


def test_reuse_leaves_existing_output_root_and_lock_untouched(tree, monkeypatch):
    root, current, runs = tree
    runs.mkdir()
    lock = runs / '.prepare.lock'
    lock.write_bytes(b'existing lock file\n')
    before = (snapshot(runs), lock.stat().st_mtime_ns, runs.stat().st_mtime_ns)

    def forbidden_lock(*args, **kwargs):
        pytest.fail('Reuse must not open or acquire the refresh lock')

    monkeypatch.setattr(prepare, '_refresh_lock', forbidden_lock)
    assert prepare.ensure_peer_context(root, end='2026-11-20') == current
    assert (snapshot(runs), lock.stat().st_mtime_ns, runs.stat().st_mtime_ns) == before


def test_reuse_newest_valid_previous_run(tree):
    root, _, runs = tree
    latest = write_context(root, runs / 'previous23', '2026-11-23')
    write_context(root, runs / 'older19', '2026-11-19')
    assert prepare.ensure_peer_context(root) == latest
    assert not (runs / '.prepare.lock').exists()


def test_stale_context_invokes_default_refresher_new_directory_and_links(tree, monkeypatch, capsys):
    root, current, runs = tree
    calls = refresh_mock(monkeypatch, root)
    output = prepare.ensure_peer_context(root)
    assert output.is_absolute() and output.parent == runs and output != current
    assert len(calls) == 1
    command, kwargs = calls[0]
    assert command[:3] == [sys.executable, '-B', str(root / 'strategy/tools/refresh_accepted465_peers.py')]
    assert '--base-context' not in command and '--industry-archive' not in command
    for flag, expected in [('--end', '2026-11-23'), ('--workers', '2'), ('--output', str(output)),
                           ('--data-path', str(root / 'data/stock_data/backtrader_data')),
                           ('--fundamental-path', str(root / 'data/stock_data/fundamental_data'))]:
        assert command[command.index(flag) + 1] == expected
    assert kwargs['cwd'] == root and kwargs['check'] is True
    assert kwargs['stdout'] is sys.stderr and kwargs['stderr'] is sys.stderr
    assert kwargs['env']['OPENBLAS_NUM_THREADS'] == '1'
    assert not prepare._pending(output).exists()
    relative = 'months/month=202610/context.parquet'
    assert os.path.samefile(current / relative, output / relative)
    assert (current / relative).read_bytes() == (output / relative).read_bytes()
    assert not os.path.samefile(current / 'months/month=202611/context.parquet',
                                output / 'months/month=202611/context.parquet')
    captured = capsys.readouterr()
    assert captured.out == '' and 'fixture refresher log' in captured.err


def test_refresher_cleans_inherited_research_environment(tree, monkeypatch):
    root, _, _ = tree
    overrides = {
        'QUANT_RESEARCH_FRESH_BUY_PRIORITY': 'research', 'QUANT_INDUSTRY_PIT': '1',
        'QUANT_B2_PROFIT_CONTEXT': '/wrong/context', 'QUANT_DATA_PATH': '/wrong/data',
        'QUANT_ALT_NO_AUTOREFRESH': '0', 'QUANT_ALT_VERSIONS_REQUIRED': '0',
        'POOL_SOURCE': 'research', 'POOL_TODATE': '2099-01-01',
        'FACTOR_CONFIG_PATH': '/wrong/factors.yaml', 'PYTHONPATH': '/wrong/imports',
        'PYTHONHOME': '/wrong/python', 'PYTHONPYCACHEPREFIX': '/wrong/pycache',
        'OMP_NUM_THREADS': '17', 'OPENBLAS_NUM_THREADS': '17', 'MKL_NUM_THREADS': '17',
        'PYTHONDONTWRITEBYTECODE': '0',
    }
    for key, value in overrides.items():
        monkeypatch.setenv(key, value)
    calls = refresh_mock(monkeypatch, root)
    prepare.ensure_peer_context(root)
    environment = calls[0][1]['env']
    assert {key: value for key, value in environment.items() if key.startswith('QUANT_')} == {
        'QUANT_ALT_NO_AUTOREFRESH': '1', 'QUANT_ALT_VERSIONS_REQUIRED': '1'}
    assert not any(key.startswith('POOL_') for key in environment)
    assert not {'FACTOR_CONFIG_PATH', 'PYTHONPATH', 'PYTHONHOME', 'PYTHONPYCACHEPREFIX'} & environment.keys()
    assert all(environment[key] == '1' for key in
               ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'PYTHONDONTWRITEBYTECODE'])
    assert environment['PATH'] == os.environ['PATH']
    assert all(os.environ[key] == value for key, value in overrides.items())


def test_subprocess_failure_preserves_old_and_pending_is_never_reused(tree, monkeypatch):
    root, _, runs = tree
    calls = refresh_mock(monkeypatch, root, failure=subprocess.CalledProcessError(2, ['fixture']))
    with pytest.raises(subprocess.CalledProcessError):
        prepare.ensure_peer_context(root)
    failed = Path(calls[0][0][calls[0][0].index('--output') + 1])
    assert prepare._pending(failed).exists() and (failed / 'partial').exists()
    # Even a complete-looking manifest from an interrupted run cannot be reused.
    (failed / 'partial').unlink()
    failed.rmdir()
    write_context(root, failed, '2026-11-23')
    succeeding = refresh_mock(monkeypatch, root)
    output = prepare.ensure_peer_context(root)
    assert len(succeeding) == 1 and output != failed
    assert prepare._pending(failed).exists()


@pytest.mark.parametrize('drift', ['ulp', 'missing_row', 'null_mask', 'boolean', 'duplicate', 'closed_month'])
def test_prefix_drift_rejects_new_context(tree, monkeypatch, drift):
    root, _, _ = tree

    def mutate(frame):
        frame = frame.copy()
        day = '2026-10-30' if drift == 'closed_month' else '2026-11-20'
        mask = (frame.date == day) & (frame.code == '000001')
        if drift in ('ulp', 'closed_month'):
            frame.loc[mask, 'peer_return5'] = np.nextafter(.02, 1.)
        elif drift == 'missing_row':
            frame = frame.loc[~mask]
        elif drift == 'null_mask':
            frame.loc[mask, 'peer_return5'] = np.nan
        elif drift == 'boolean':
            frame.loc[mask, 'peer_ebb'] = True
        else:
            frame = pd.concat([frame, frame.loc[mask]], ignore_index=True)
        return frame

    calls = refresh_mock(monkeypatch, root, mutate)
    with pytest.raises(ValueError, match='Historical prefix'):
        prepare.ensure_peer_context(root)
    output = Path(calls[0][0][calls[0][0].index('--output') + 1])
    assert prepare._pending(output).exists()
    # A rejected yet well-hashed manifest is still excluded on the next run.
    calls = refresh_mock(monkeypatch, root)
    assert prepare.ensure_peer_context(root) != output
    assert len(calls) == 1


@pytest.mark.parametrize('change_old_value', [False, True])
def test_prefix_without_nulls_ignores_nulls_in_appended_ipo(tmp_path, change_old_value):
    relative = 'months/month=202611/context.parquet'
    old_row = {'code': '000001', 'date': pd.Timestamp('2026-11-20'), 'sector': 'industry',
               'peer_count': 6., 'peer_return5': .02, 'peer_relative5': -.01,
               'peer_breadth5': .4, 'peer_known5': True, 'peer_ebb': False}
    new_row = dict(old_row, peer_ebb=True) if change_old_value else old_row.copy()
    ipo = dict(old_row, code='001999', date=pd.Timestamp('2026-11-23'),
               peer_return5=None, peer_relative5=None, peer_breadth5=None,
               peer_known5=False, peer_ebb=None)
    contexts = []
    for name, rows, end in [('old', [old_row], date(2026, 11, 20)),
                             ('new', [new_row, ipo], date(2026, 11, 23))]:
        path = tmp_path / name
        target = path / relative
        target.parent.mkdir(parents=True)
        pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), target)
        contexts.append(prepare._Context(path, date(2026, 11, 20), end, {relative: sha(target)}))
    # The unfiltered new table has object dtype solely because of the IPO null.
    with pq.ParquetFile(contexts[0].path / relative) as parquet:
        assert parquet.read().to_pandas().peer_ebb.dtype == np.dtype('bool')
    with pq.ParquetFile(contexts[1].path / relative) as parquet:
        assert parquet.read().to_pandas().peer_ebb.dtype == np.dtype('object')
    if change_old_value:
        with pytest.raises(ValueError, match='Historical prefix changed'):
            prepare._check_prefix(*contexts)
    else:
        prepare._check_prefix(*contexts)


def test_prefix_checks_latest_valid_run_instead_of_older_profile(tree, monkeypatch):
    root, _, runs = tree

    def mutate(frame):
        frame = frame.copy()
        frame.loc[(frame.date == '2026-11-20') & (frame.code == '000001'), 'peer_return5'] = .03
        return frame

    # A later valid immutable archive supersedes the profile's earlier context.
    older = write_context(root, runs / 'previous20', '2026-11-20', mutate)
    write_context(root, runs / 'previous19', '2026-11-19')
    # Both latest dates are equal: current wins ties, so make current ineligible
    # as a baseline without modifying its protected contents.
    calls = refresh_mock(monkeypatch, root)
    with pytest.raises(ValueError, match='Historical prefix changed'):
        prepare.ensure_peer_context(root, current_path=runs / 'previous19')
    assert calls and older.exists()


@pytest.mark.parametrize('damage', ['missing_hash', 'missing_file', 'wrong_hash', 'kernel',
                                   'known', 'ebb', 'universe', 'bad_json', 'not_session'])
def test_invalid_manifest_is_rejected_before_reuse(tree, damage):
    root, _, runs = tree
    path = write_context(root, runs / 'invalid23', '2026-11-23')
    manifest_path = path / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    relative = 'months/month=202611/context.parquet'
    if damage == 'missing_hash':
        del manifest['output_sha256'][relative]
    elif damage == 'missing_file':
        (path / relative).unlink()
    elif damage == 'wrong_hash':
        manifest['output_sha256'][relative] = '0' * 64
    elif damage == 'kernel':
        manifest['source_sha256'][str(root / prepare.KERNEL)] = '0' * 64
    elif damage in ('known', 'ebb'):
        manifest['validation']['known_mask_mismatches' if damage == 'known' else 'ebb_mismatches'] = 1
    elif damage == 'universe':
        manifest['all_market_universe'] = False
    elif damage == 'not_session':
        manifest['end'] = '2026-11-22'
    manifest_path.write_text('{bad' if damage == 'bad_json' else json.dumps(manifest))
    # End20 still reuses the intact current context, never the invalid newer run.
    assert prepare.ensure_peer_context(root, end='2026-11-20').name == 'accepted465_peer_context'
    with pytest.raises(ValueError):
        prepare._validate(path, root, prepare._calendar(root))


@pytest.mark.parametrize('relative', prepare.SOURCE_FILES)
@pytest.mark.parametrize('damage', ['missing_sha', 'wrong_sha', 'current_source_changed'])
def test_each_calculation_dependency_must_match_current_source(tree, relative, damage):
    root, current, runs = tree
    path = write_context(root, runs / 'invalid23', '2026-11-23')
    manifest_path = path / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if damage == 'missing_sha':
        del manifest['source_sha256'][str(root / relative)]
    elif damage == 'wrong_sha':
        manifest['source_sha256'][str(root / relative)] = '0' * 64
    else:
        source = root / relative
        source.write_text(source.read_text() + '# changed after context generation\n')
    manifest_path.write_text(json.dumps(manifest))
    calendar = prepare._calendar(root)
    with pytest.raises(ValueError, match=Path(relative).name):
        prepare._validate(path, root, calendar)
    if damage == 'current_source_changed':
        assert prepare._contexts(current, runs, root, calendar) == []
    else:
        assert prepare.ensure_peer_context(root, end='2026-11-20') == current


def test_identical_month_does_not_import_dataframe_libraries(tree, monkeypatch):
    root, current, runs = tree
    same = write_context(root, runs / 'same20', '2026-11-20')
    calendar = prepare._calendar(root)
    old, new = [prepare._validate(path, root, calendar) for path in [current, same]]
    import builtins
    real_import = builtins.__import__

    def stdlib_only(name, *args, **kwargs):
        assert name.split('.')[0] not in {'pandas', 'pyarrow'}
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', stdlib_only)
    prepare._check_prefix(old, new)


def test_hardlink_failure_keeps_valid_copies(tree, monkeypatch):
    root, current, _ = tree
    refresh_mock(monkeypatch, root)

    def no_link(*args):
        raise OSError('cross-device link')

    monkeypatch.setattr(prepare.os, 'link', no_link)
    output = prepare.ensure_peer_context(root)
    relative = 'months/month=202610/context.parquet'
    assert not os.path.samefile(current / relative, output / relative)
    assert (current / relative).read_bytes() == (output / relative).read_bytes()
    assert not list(output.rglob('*.link-*'))


def test_second_lock_inspection_reuses_other_callers_publication(tree, monkeypatch):
    from contextlib import contextmanager
    root, _, runs = tree
    output = runs / 'other_caller23'

    @contextmanager
    def publish_while_waiting(path):
        assert path == runs
        write_context(root, output, '2026-11-23')
        yield

    monkeypatch.setattr(prepare, '_refresh_lock', publish_while_waiting)
    assert prepare.ensure_peer_context(root) == output


def test_cli_print_path_stdout_contract(tree, capsys):
    root, current, _ = tree
    for print_path in [False, True]:
        args = ['--end', '2026-11-20', '--current', str(current), '--output-root', str(root / 'unused')]
        assert prepare.main(args + (['--print-path'] if print_path else [])) == 0
        captured = capsys.readouterr()
        assert captured.out == (str(current) + '\n' if print_path else '')
        assert 'reuse' in captured.err


def test_cli_failure_has_no_path_on_stdout(tree, capsys):
    with pytest.raises(SystemExit) as error:
        prepare.main(['--end', '2026-11-22', '--print-path'])
    assert error.value.code == 1
    captured = capsys.readouterr()
    assert captured.out == '' and 'not a real index session' in captured.err
