#!/usr/bin/env python3
"""Report provenance limits of versioned alternative data in a backtest window.

An event date in a retrospectively downloaded table does not establish when the
supplier first exposed that row. This report does not change research/live data.
"""
import argparse
import json
from datetime import date
from pathlib import Path

if __package__:
    from strategy.core.alt_data_versions import TABLES, _read_manifest
else:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from core.alt_data_versions import TABLES, _read_manifest


def audit(root, start, end):
    start, end = date.fromisoformat(str(start)), date.fromisoformat(str(end))
    if end < start:
        raise ValueError('end precedes start')
    root = Path(root)
    rows = []
    for filename in sorted(TABLES):
        stem = Path(filename).stem
        manifest = root / 'alt_versions' / f'{stem}.json'
        if not manifest.is_file():
            raise FileNotFoundError(f'missing observation manifest: {manifest}')
        state = _read_manifest(manifest)
        first = state['versions'][0]['observed_on']
        last = state['versions'][-1]['observed_on']
        rows.append({'table': filename, 'first_observed_on': first,
                     'last_observed_on': last, 'observation_versions': len(state['versions']),
                     'entire_window_pre_observation': first > end.isoformat(),
                     'window_starts_before_first_observation': first > start.isoformat()})
    return {'start': start.isoformat(), 'end': end.isoformat(), 'tables': rows,
            'tables_without_any_observed_version_by_end': sum(
                r['entire_window_pre_observation'] for r in rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True,
                        help='alternative_data directory containing alt_versions')
    parser.add_argument('--start', required=True)
    parser.add_argument('--end', required=True)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    result = audit(args.root, args.start, args.end)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
