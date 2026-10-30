#!/usr/bin/env python3
"""Freeze source into an isolated research run; production outputs are never reused.

Examples:
  python strategy/analysis/review_validation.py --name corrected --run
  python strategy/analysis/review_validation.py --name corrected --replay --risk-limits

Full-universe generation can take hours. Replay requires the same frozen run's
signals. The manifest separates original historical research from untouched OOS.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime

ROOT = Path(__file__).resolve().parents[2]
IGNORED = {'__pycache__', 'cache', 'models', 'logs', 'reports', 'rolling_validation_results',
           'analysis_results', 'daily_review', 'state_0915_todate0915', 'tmp_factor'}


_RUN_ENV_KEYS = (
    'QUANT_DATA_PATH', 'QUANT_CONCEPT_MAP', 'QUANT_CONCEPT_MAP_HIST',
    'QUANT_EDGE_ASOF', 'QUANT_MARKET_ALT_WEIGHT', 'QUANT_ACCOUNT_LIMITS',
    'QUANT_STAR_EXCLUDE', 'QUANT_REBAL_PHASE', 'POOL_TODATE',
)


def _sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def input_stamp(env=None, root=None):
    """Content-address every backtest input and record the effective price source.

    The previous stamp called itself SHA-256 but only hashed file size/mtime. That
    missed same-size rewrites with restored timestamps and omitted the effective
    QUANT_DATA_PATH. Research runs must fail closed when data changes mid-run.
    """
    env = os.environ if env is None else env
    root = Path(root or ROOT).resolve()
    data = (root / 'data').resolve()
    data_path_value = env.get('QUANT_DATA_PATH') or str(
        data / 'stock_data/backtrader_data')
    data_path = Path(data_path_value).expanduser()
    if not data_path.is_absolute():
        data_path = (root / 'strategy' / data_path).resolve()
    else:
        data_path = data_path.resolve()

    patterns = [
        'stock_data/fundamental_data/*.csv',
        'stock_data/stock_metadata/*.csv',
        'stock_data/raw_data/*/none.csv',
        'stock_data/raw_data/*/qfq.csv',
        'stock_data/raw_data/*/hfq.csv',
        'alternative_data/**/*.pkl', 'alternative_data/**/*.csv',
        'alternative_data/**/*.json',
        'stock_data/*.pkl', 'stock_data/*.json',
        'macro/*.csv',
        'stock_concept_map.pkl', 'concept_map_history.pkl',
        'concept_hist.pkl', 'concept_daily.csv', 'concept_inception.pkl',
    ]
    paths = set()
    for pattern in patterns:
        paths.update(p for p in data.glob(pattern)
                     if not any(part.startswith('.') for part in p.relative_to(data).parts))
    if data_path.is_dir():
        paths.update(p for p in data_path.iterdir()
                     if p.is_file() and p.suffix.lower() == '.csv')
    for override in ('QUANT_CONCEPT_MAP', 'QUANT_CONCEPT_MAP_HIST'):
        value = env.get(override)
        if value and value.strip().lower() not in ('off', 'none', '0', 'false'):
            p = Path(value).expanduser()
            if not p.is_absolute():
                p = (root / 'strategy' / p).resolve()
            if not p.is_file():
                raise FileNotFoundError(p)
            paths.add(p)
    records = {}
    for path in sorted(paths, key=lambda p: str(p)):
        try:
            before = path.stat()
            digest = _sha256_file(path)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise RuntimeError(f'input changed while hashing: {path}')
            try:
                key = str(path.relative_to(data))
            except ValueError:
                key = str(path.resolve())
            records[key] = {'size': after.st_size, 'mtime_ns': after.st_mtime_ns,
                            'sha256': digest}
        except OSError as exc:
            raise RuntimeError(f'cannot fingerprint backtest input {path}: {exc}') from exc

    adjustment_files = [p.name for p in paths
                        if p.parent.resolve() == data_path.resolve()
                        and p.name.endswith(('_qfq.csv', '_hfq.csv'))]
    qfq_codes = {n[:-8] for n in adjustment_files if n.endswith('_qfq.csv')}
    hfq_codes = {n[:-8] for n in adjustment_files if n.endswith('_hfq.csv')}
    mixed_codes = sorted((qfq_codes & hfq_codes) - {'sh000001', 'sh000852', '399006'})
    if qfq_codes and hfq_codes:
        adjustment_state = 'mixed_qfq_hfq'
    elif qfq_codes:
        adjustment_state = 'qfq_only'
    elif hfq_codes:
        adjustment_state = 'hfq_only'
    else:
        adjustment_state = 'no_adjusted_csvs'
    run_env = {k: env.get(k) for k in sorted(_RUN_ENV_KEYS)}
    other_env_hashes = {
        k: hashlib.sha256(str(v).encode()).hexdigest()
        for k, v in sorted(env.items())
        if k not in _RUN_ENV_KEYS and k.startswith(
            ('QUANT_', 'POOL_', 'OMP_', 'MKL_', 'OPENBLAS_'))
    }
    identity = {
        'files': {k: {'size': v['size'], 'sha256': v['sha256']} for k, v in records.items()},
        'data_root': str(data),
        'effective_data_path': str(data_path),
        'adjustment_mode': {
            'state': adjustment_state,
            'qfq_files': len(qfq_codes), 'hfq_files': len(hfq_codes),
            'mixed_code_count': len(mixed_codes), 'mixed_codes': mixed_codes[:20],
        },
        'environment': run_env,
        'other_environment_sha256': other_env_hashes,
    }
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return {'sha256': digest, 'files': len(records),
            'data_path': str(data_path), 'adjustment_mode': identity['adjustment_mode'],
            'environment': run_env, 'environment_hashes': other_env_hashes}, records


def content_id(records):
    """Relocatable identity; timestamps and absolute snapshot location are not content."""
    return hashlib.sha256(json.dumps({k: [v['size'], v['sha256']]
                         for k, v in records.items()}, sort_keys=True).encode()).hexdigest()


def freeze_data(target, env=None):
    """Copy selected inputs, never hardlink live files; seal only after rechecking source."""
    env = dict(os.environ if env is None else env)
    # Unmapped path overrides could escape the snapshot. Refuse instead of ignoring them.
    for key in ('QUANT_DATA_PATH', 'QUANT_CONCEPT_MAP', 'QUANT_CONCEPT_MAP_HIST',
                'QUANT_LIQ_DATA_DIR', 'QUANT_QFEAT_DATA_DIR'):
        if env.get(key):
            raise ValueError(f'unset {key} before freezing canonical inputs')
    before, records = input_stamp(env)
    if before['adjustment_mode']['state'] != 'qfq_only':
        raise ValueError(f'qfq-only inputs required: {before["adjustment_mode"]}')
    data = target / 'data'
    data.mkdir()
    for key, record in records.items():
        rel = Path(key)
        if rel.is_absolute() or '..' in rel.parts:
            raise ValueError(f'input outside canonical data root: {key}')
        src, dst = ROOT / 'data' / rel, data / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        if _sha256_file(dst) != record['sha256']:
            raise RuntimeError(f'source changed during snapshot copy: {key}')
        dst.chmod(0o444)
    after, again = input_stamp(env)
    if before != after or content_id(records) != content_id(again):
        raise RuntimeError('source changed during freeze; incomplete snapshot retained, not sealed')
    seal = {'sha256': content_id(records), 'files': records,
            'source_stamp': before, 'created_at': datetime.now().isoformat()}
    (target / 'data_snapshot.json').write_text(json.dumps(seal, indent=2), encoding='utf-8')
    for directory in sorted((p for p in data.rglob('*') if p.is_dir()),
                            key=lambda p: len(p.parts), reverse=True):
        directory.chmod(0o555)
    data.chmod(0o555)
    return seal


def read_snapshot_seal(target):
    seal_path = target / 'data_snapshot.json'
    if not seal_path.is_file():
        raise RuntimeError('run has no sealed data snapshot; legacy live-data runs cannot be replayed')
    return json.loads(seal_path.read_text(encoding='utf-8'))


def verify_snapshot(target):
    seal = read_snapshot_seal(target)
    stamp, records = input_stamp(env={}, root=target)
    if content_id(records) != seal['sha256']:
        raise RuntimeError('snapshot content differs from seal')
    return seal, stamp


def prepare(name, target_mode=None, workers=None, snapshot_from=None, legacy_concept_boundary=False,
            strict_concept_boundary=False):
    if not name or Path(name).name != name:
        raise ValueError('name must be a single directory component')
    if workers is not None and not 1 <= workers <= 16:
        raise ValueError('workers must be between 1 and 16')
    if legacy_concept_boundary and strict_concept_boundary:
        raise ValueError('conflicting concept prehistory policies')
    import yaml
    cfg = yaml.safe_load((ROOT / 'strategy/config/factor_config.yaml').read_text(encoding='utf-8'))
    if cfg.get('paths'):
        raise ValueError('custom config paths need explicit snapshot mapping')
    target = ROOT / '.review_runs' / name
    if target.exists():
        raise FileExistsError(f'run already exists: {target}')
    target.mkdir(parents=True)
    hashes = {}
    for base, dirs, files in os.walk(ROOT / 'strategy'):
        dirs[:] = [d for d in dirs if d not in IGNORED and not d.startswith('.')]
        for name in files:
            p = Path(base) / name
            if p.suffix not in {'.py', '.yaml', '.yml', '.json', '.sh'}:
                continue
            rel = p.relative_to(ROOT)
            out = target / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, out)
            hashes[str(rel)] = hashlib.sha256(p.read_bytes()).hexdigest()
    if snapshot_from:
        source = Path(snapshot_from).resolve()
        seal, _ = verify_snapshot(source)
        (target / 'data').symlink_to((source / 'data').resolve(), target_is_directory=True)
        shutil.copy2(source / 'data_snapshot.json', target / 'data_snapshot.json')
    else:
        seal = freeze_data(target)
    (target / 'strategy/rolling_validation_results').mkdir(exist_ok=True)
    overrides = {}
    if target_mode is not None:
        overrides[('ml', 'target_mode')] = target_mode
    if workers is not None:
        if workers < 1 or workers > 16:
            raise ValueError('workers must be between 1 and 16')
        overrides[('backtest', 'num_workers')] = workers
    config_path = target / 'strategy/config/factor_config.yaml'
    lines = config_path.read_text(encoding='utf-8').splitlines(keepends=True)
    section = ''; applied = set()
    for i, line in enumerate(lines):
        if line and not line[0].isspace() and ':' in line:
            section = line.split(':', 1)[0]
        for (group, key), value in overrides.items():
            if section == group and line.startswith('  ' + key + ':'):
                lines[i] = f'  {key}: {value}\n'; applied.add((group, key))
    if applied != set(overrides):
        raise ValueError('requested configuration override not found')
    config_path.write_text(''.join(lines), encoding='utf-8')
    hashes['strategy/config/factor_config.yaml'] = hashlib.sha256(config_path.read_bytes()).hexdigest()
    if strict_concept_boundary:
        # Empty prehistory changes the strategy; keep it as an explicit experiment.
        replacements = {
            'strategy/core/concept_heat.py': ("        ver = self._cmap_versions[0][0]\n", "        ver = ''\n"),
            'strategy/core/signal_engine.py': ("    ver, m = CONCEPT_MAP_HIST[0]\n", "    ver, m = '', {}\n"),
        }
        for rel, (old, new) in replacements.items():
            path = target / rel
            source = path.read_text(encoding='utf-8')
            if source.count(old) != 1:
                raise ValueError(f'strict concept control patch not unique: {rel}')
            path.write_text(source.replace(old, new), encoding='utf-8')
            hashes[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {'created_at': datetime.now().isoformat(), 'source_sha256': hashes,
                'data_source': str((target / 'data').resolve()), 'data_mode': 'sealed_content_snapshot',
                'data_sha256': seal['sha256'],
                'concept_boundary': ('empty_before_first_version' if strict_concept_boundary
                                     else 'first_snapshot_proxy_before_first_version'),
                'historical_period_already_used_for_research': True,
                'production_outputs_reused': False,
                'configuration_overrides': {'.'.join(k): v for k, v in overrides.items()},
                'effective_config_sha256': hashlib.sha256(config_path.read_bytes()).hexdigest()}
    (target / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return target


def write_execution_audits(target, archive):
    """Run point-of-use price/entitlement checks against a sealed run's outputs."""
    target, archive = Path(target).resolve(), Path(archive).resolve()
    audit_tools = {}
    for script_name, report_name, extra_args in (
        ('audit_qfq_lot_sizes.py', 'raw_lot_audit.json',
         ['--raw', str(target / 'data/stock_data/raw_data'),
          '--qfq', str(target / 'data/stock_data/backtrader_data')]),
        ('audit_held_corporate_actions.py', 'held_corporate_actions.json',
         ['--events', str(target / 'data/alternative_data/events_table.pkl')]),
    ):
        frozen_script = target / 'strategy/analysis' / script_name
        tool_path = frozen_script if frozen_script.is_file() else ROOT / 'strategy/analysis' / script_name
        command = [sys.executable, '-B', str(tool_path), '--results', str(archive),
                   *extra_args, '--out', str(archive / report_name)]
        check = subprocess.run(command, capture_output=True, text=True)
        if check.returncode:
            raise RuntimeError(f'execution audit {script_name} failed (rc={check.returncode}): '
                               f'{check.stderr.strip() or check.stdout.strip()}')
        audit_tools[script_name] = {'source': str(tool_path), 'sha256': _sha256_file(tool_path),
                                    'report': report_name}
    return audit_tools


def run(target, risk_limits=False):
    target = Path(target).resolve()
    seal = read_snapshot_seal(target)
    manifest_path = target / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    for rel, digest in manifest['source_sha256'].items():
        if hashlib.sha256((target / rel).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f'frozen source changed: {rel}')
    env = os.environ.copy()
    # The frozen code must use its own data tree. Unknown external hooks are unsafe.
    for key in ('QUANT_DATA_PATH', 'QUANT_CONCEPT_MAP', 'QUANT_CONCEPT_MAP_HIST',
                'QUANT_LIQ_DATA_DIR', 'QUANT_QFEAT_DATA_DIR', 'QUANT_LOWSH_PEN', 'QUANT_MRG_TILT'):
        if env.get(key):
            raise ValueError(f'unset external research hook {key} for a sealed run')
    env['QUANT_DATA_PATH'] = str((target / 'data/stock_data/backtrader_data').resolve())
    env['QUANT_LIQ_DATA_DIR'] = env['QUANT_DATA_PATH']
    env['QUANT_QFEAT_DATA_DIR'] = env['QUANT_DATA_PATH']
    env.update(PYTHONDONTWRITEBYTECODE='1', QUANT_ALT_NO_AUTOREFRESH='1',
               QUANT_ALT_VERSIONS_REQUIRED='1')
    if risk_limits:
        env['QUANT_ACCOUNT_LIMITS'] = '1'
    else:
        env.pop('QUANT_ACCOUNT_LIMITS', None)
    before, records = input_stamp(env=env, root=target)
    # The execution-environment scan covers the same sealed file set. Compare
    # its content identity directly instead of reading every file a second time
    # through verify_snapshot(). Keep a second full scan after execution.
    if content_id(records) != seal['sha256']:
        raise RuntimeError('snapshot content differs from seal before execution')
    if before['adjustment_mode']['state'] != 'qfq_only':
        raise RuntimeError(
            'the current vectorized execution path is validated for qfq-only data; '
            'refusing an unverified/mixed adjustment source: '
            f'{before["adjustment_mode"]}')
    (target / 'input_files.json').write_text(json.dumps(records, indent=2))
    manifest['input_stamp_algorithm'] = 'sha256_file_content_v2'
    manifest['effective_data_path'] = before['data_path']
    manifest['adjustment_mode'] = before['adjustment_mode']
    manifest['run_environment'] = before['environment']
    manifest['run_environment_hashes'] = before['environment_hashes']
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    label = 'risk_limits' if risk_limits else 'corrected'
    log = target / f'{label}.log'
    if log.exists() or (target / label).exists():
        raise FileExistsError(f'results already exist; prepare a fresh run: {target}')
    with log.open('w', encoding='utf-8') as stream:
        result = subprocess.run([sys.executable, '-B', '-u', 'bt_execution.py'],
                                cwd=target / 'strategy', env=env, stdout=stream,
                                stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'research run failed (rc={result.returncode}): {log}')
    after, after_records = input_stamp(env=env, root=target)
    unchanged = before == after and content_id(after_records) == seal['sha256']
    (target / f'{label}_input_stamp.json').write_text(json.dumps(
        {'before': before, 'after': after, 'unchanged': unchanged}, indent=2))
    if not unchanged:
        raise RuntimeError(f'input data changed during research; results are not a frozen-data comparison: {log}')
    results = target / 'strategy/rolling_validation_results'
    archive = target / label
    archive.mkdir(exist_ok=True)
    for name in ['equity_curve.csv', 'portfolio_selections.csv', 'trade_realized.csv', 'regime_state.csv', 'execution_fills.csv']:
        p = results / name
        if p.exists():
            shutil.copy2(p, archive / name)
    # These audits inspect the same sealed input after the backtest has exited;
    # they do not modify the calculated NAV or any executed fill.
    manifest['execution_audits'] = write_execution_audits(target, archive)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return log


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', required=True)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--replay', action='store_true')
    parser.add_argument('--risk-limits', action='store_true')
    parser.add_argument('--target-mode', choices=['legacy_10d', 'net_short_3_5'])
    parser.add_argument('--workers', type=int)
    parser.add_argument('--snapshot-from', type=Path, help='existing sealed run sharing the exact data snapshot')
    parser.add_argument('--legacy-concept-boundary', action='store_true',
                        help='compatibility flag: use the default fixed first-snapshot prehistory proxy')
    parser.add_argument('--strict-concept-boundary', action='store_true',
                        help='research experiment: use empty concept membership before the first snapshot')
    args = parser.parse_args()
    if args.replay and (args.target_mode is not None or args.workers is not None
                        or args.legacy_concept_boundary or args.strict_concept_boundary):
        parser.error('replay must use the existing immutable run configuration')
    target = ROOT / '.review_runs' / args.name if args.replay else prepare(
        args.name, args.target_mode, args.workers, args.snapshot_from,
        args.legacy_concept_boundary, args.strict_concept_boundary)
    print(target, flush=True)
    if args.run or args.replay:
        print(run(target, args.risk_limits), flush=True)
