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


def input_stamp():
    data = ROOT / 'data'
    patterns = ['stock_data/backtrader_data/*.csv', 'stock_data/fundamental_data/*.csv',
                'stock_data/stock_metadata/*.csv', 'stock_data/raw_data/*/none.csv',
                'stock_data/raw_data/000001/qfq.csv', 'alternative_data/*.pkl',
                'macro/*.csv', 'stock_concept_map.pkl', 'concept_map_history.pkl',
                'concept_hist.pkl', 'concept_daily.csv']
    records = {}
    for pattern in patterns:
        for path in data.glob(pattern):
            stat = path.stat()
            records[str(path.relative_to(data))] = [stat.st_size, stat.st_mtime_ns]
    digest = hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()
    return {'sha256': digest, 'files': len(records)}, records


def prepare(name, target_mode=None, workers=None):
    if not name or Path(name).name != name:
        raise ValueError('name must be a single directory component')
    target = ROOT / '.review_runs' / name
    if target.exists():
        raise FileExistsError(f'run already exists: {target}')
    target.mkdir(parents=True)
    hashes = {}
    for base, dirs, files in os.walk(ROOT / 'strategy'):
        dirs[:] = [d for d in dirs if d not in IGNORED and not d.startswith('.')]
        for name in files:
            p = Path(base) / name
            if p.suffix not in {'.py', '.yaml', '.sh'}:
                continue
            rel = p.relative_to(ROOT)
            out = target / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, out)
            hashes[str(rel)] = hashlib.sha256(p.read_bytes()).hexdigest()
    (target / 'data').symlink_to(ROOT / 'data', target_is_directory=True)
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
    manifest = {'created_at': datetime.now().isoformat(), 'source_sha256': hashes,
                'data_source': str(ROOT / 'data'), 'data_mode': 'read_only_existing_cache',
                'historical_period_already_used_for_research': True,
                'production_outputs_reused': False,
                'configuration_overrides': {'.'.join(k): v for k, v in overrides.items()},
                'effective_config_sha256': hashlib.sha256(config_path.read_bytes()).hexdigest()}
    (target / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return target


def run(target, risk_limits=False):
    before, records = input_stamp()
    (target / 'input_files.json').write_text(json.dumps(records, indent=2))
    env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE='1', QUANT_ALT_NO_AUTOREFRESH='1')
    if risk_limits:
        env['QUANT_ACCOUNT_LIMITS'] = '1'
    else:
        env.pop('QUANT_ACCOUNT_LIMITS', None)
    label = 'risk_limits' if risk_limits else 'corrected'
    log = target / f'{label}.log'
    with log.open('w', encoding='utf-8') as stream:
        result = subprocess.run([sys.executable, '-B', '-u', 'bt_execution.py'],
                                cwd=target / 'strategy', env=env, stdout=stream,
                                stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'research run failed (rc={result.returncode}): {log}')
    after, _ = input_stamp()
    (target / f'{label}_input_stamp.json').write_text(json.dumps(
        {'before': before, 'after': after, 'unchanged': before == after}, indent=2))
    if before != after:
        raise RuntimeError(f'input data changed during research; results are not a frozen-data comparison: {log}')
    results = target / 'strategy/rolling_validation_results'
    archive = target / label
    archive.mkdir(exist_ok=True)
    for name in ['equity_curve.csv', 'portfolio_selections.csv', 'trade_realized.csv', 'regime_state.csv']:
        p = results / name
        if p.exists():
            shutil.copy2(p, archive / name)
    return log


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', required=True)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--replay', action='store_true')
    parser.add_argument('--risk-limits', action='store_true')
    parser.add_argument('--target-mode', choices=['legacy_10d', 'net_short_3_5'])
    parser.add_argument('--workers', type=int)
    args = parser.parse_args()
    if args.replay and (args.target_mode is not None or args.workers is not None):
        parser.error('replay must use the existing immutable run configuration')
    target = ROOT / '.review_runs' / args.name if args.replay else prepare(args.name, args.target_mode, args.workers)
    print(target, flush=True)
    if args.run or args.replay:
        print(run(target, args.risk_limits), flush=True)
