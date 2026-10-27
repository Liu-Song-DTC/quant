#!/usr/bin/env python3
"""Validate the financial availability *proxy*, including invalid/missing rows.

Passing this check only establishes agreement with the configured deadline
proxy. It does not prove original publication times or unrevised financial data.
Exit: 0=proxy contract satisfied; 1=invalid dates/values; 2=unreadable inputs.
"""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'data'))
import pandas as pd
from fundamental_pit import pit_available_series


def audit(directory, max_report=5):
    directory = Path(directory)
    files = sorted(directory.glob('*.csv')) if directory.is_dir() else []
    result = {'files': len(files), 'rows': 0, 'checked_files': 0, 'bad_rows': 0,
              'invalid_period_rows': 0, 'read_errors': [], 'examples': []}
    if not files:
        result['read_errors'].append('financial directory missing or empty')
        return 2, result
    for path in files:
        try:
            frame = pd.read_csv(path, dtype=str, usecols=['报告期', '数据可用日期'])
            if frame.empty:
                raise ValueError('empty financial history')
        except Exception as exc:
            result['read_errors'].append(f'{path.name}: {type(exc).__name__}: {exc}')
            continue
        result['checked_files'] += 1
        result['rows'] += len(frame)
        expected = pit_available_series(frame['报告期'])
        invalid = expected.eq('')
        mismatch = ~invalid & frame['数据可用日期'].fillna('').ne(expected)
        bad = invalid | mismatch
        result['invalid_period_rows'] += int(invalid.sum())
        result['bad_rows'] += int(bad.sum())
        if bad.any() and len(result['examples']) < max_report:
            row = int(bad.to_numpy().nonzero()[0][0])
            result['examples'].append(
                f'{path.stem}: period={frame["报告期"].iloc[row]!r}, '
                f'available={frame["数据可用日期"].iloc[row]!r}, expected={expected.iloc[row]!r}')
    code = 2 if result['read_errors'] else (1 if result['bad_rows'] else 0)
    return code, result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quiet', action='store_true')
    parser.add_argument('--dir', type=Path,
                        default=Path(__file__).resolve().parents[2] / 'data/stock_data/fundamental_data')
    parser.add_argument('--max-report', type=int, default=5)
    args = parser.parse_args(argv)
    code, result = audit(args.dir, max_report=max(0, args.max_report))
    if code == 0:
        if not args.quiet:
            print(f'[fund-pit] deadline proxy satisfied: {result["checked_files"]} files / {result["rows"]} rows; '
                  'publication/revision PIT still requires separate verification')
        return 0
    print(f'[fund-pit] rejected: bad rows={result["bad_rows"]}, '
          f'invalid periods={result["invalid_period_rows"]}, read errors={len(result["read_errors"])}')
    for example in result['examples'] + result['read_errors'][:max(0, args.max_report)]:
        print('  ' + example)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
