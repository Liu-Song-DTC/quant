#!/usr/bin/env python3
"""Compare old date-keyed stock rows between two sealed data snapshots."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path


PROTECTED = {'raw_none', 'bt_qfq', 'fundamental'}


def category(key):
    parts = Path(key).parts
    if len(parts) == 4 and parts[:2] == ('stock_data', 'raw_data'):
        return {'none.csv': 'raw_none', 'qfq.csv': 'vendor_qfq',
                'hfq.csv': 'vendor_hfq'}.get(parts[-1], 'other')
    if len(parts) == 3 and parts[:2] == ('stock_data', 'backtrader_data'):
        return 'bt_qfq' if key.endswith('_qfq.csv') else 'other'
    if len(parts) == 3 and parts[:2] == ('stock_data', 'fundamental_data'):
        return 'fundamental'
    return 'other'


def _csv_rows(path, date_col):
    rows = {}
    with path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or date_col not in reader.fieldnames:
            raise ValueError(f'{path}: date column {date_col!r} absent')
        columns = reader.fieldnames
        for row in reader:
            key = row[date_col]
            if not key or key in rows:
                raise ValueError(f'{path}: missing/duplicate date {key!r}')
            rows[key] = row
    return columns, rows


def compare_rows(old_path, new_path, date_col, sample_limit=5):
    old_cols, before = _csv_rows(old_path, date_col)
    new_cols, after = _csv_rows(new_path, date_col)
    if old_cols != new_cols:
        return {'schema_changed': True, 'old_columns': old_cols, 'new_columns': new_cols}
    old_dates, new_dates = set(before), set(after)
    fields = Counter()
    samples = []
    changed_dates = 0
    for day in sorted(old_dates & new_dates):
        changed = [col for col in old_cols if before[day][col] != after[day][col]]
        if changed:
            changed_dates += 1
            fields.update(changed)
            if len(samples) < sample_limit:
                samples.append({'date': day, 'fields': changed[:12],
                                'old': {col: before[day][col] for col in changed[:4]},
                                'new': {col: after[day][col] for col in changed[:4]}})
    return {'schema_changed': False, 'old_rows': len(before),
            'new_rows': len(after), 'added_dates': len(new_dates - old_dates),
            'removed_dates': len(old_dates - new_dates), 'changed_dates': changed_dates,
            'changed_fields': dict(fields), 'examples': samples}


def _digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def compare(old_root, new_root, example_limit=30):
    old_root, new_root = Path(old_root), Path(new_root)
    old = json.loads((old_root / 'data_snapshot.json').read_text())
    new = json.loads((new_root / 'data_snapshot.json').read_text())
    older, newer = old['files'], new['files']
    totals = defaultdict(Counter)
    examples = []
    problems = []
    for key in sorted(older.keys() | newer.keys()):
        group = category(key)
        if key not in newer or key not in older:
            status = 'removed_file' if key not in newer else 'new_file'
            totals[group][status] += 1
            if status == 'removed_file' and group in PROTECTED:
                problems.append({'file': key, 'reason': status})
            continue
        totals[group]['common_files'] += 1
        if older[key]['sha256'] == newer[key]['sha256']:
            totals[group]['identical_files'] += 1
            continue
        totals[group]['changed_files'] += 1
        before = old_root / 'data' / key
        after = new_root / 'data' / key
        try:
            if _digest(before) != older[key]['sha256'] or _digest(after) != newer[key]['sha256']:
                raise ValueError('sealed file differs from its snapshot digest')
            date_col = {'raw_none': '日期', 'vendor_qfq': '日期',
                        'vendor_hfq': '日期', 'bt_qfq': 'datetime',
                        'fundamental': '报告期'}.get(group)
            if date_col is None:
                if len(examples) < example_limit:
                    examples.append({'file': key, 'group': group, 'reason': 'changed_non_price_file'})
                continue
            detail = compare_rows(before, after, date_col)
            if detail.get('schema_changed'):
                totals[group]['schema_changed_files'] += 1
                if group in PROTECTED:
                    problems.append({'file': key, 'reason': 'schema_changed'})
            else:
                for metric in ('added_dates', 'removed_dates', 'changed_dates'):
                    totals[group][metric] += detail[metric]
                totals[group].update({f'field:{col}': n for col, n in detail['changed_fields'].items()})
                if group in PROTECTED and (detail['removed_dates'] or detail['changed_dates']):
                    problems.append({'file': key, 'reason': 'historical_rows_changed',
                                     'removed_dates': detail['removed_dates'],
                                     'changed_dates': detail['changed_dates']})
            if (detail.get('schema_changed') or detail.get('removed_dates')
                    or detail.get('changed_dates')) and len(examples) < example_limit:
                examples.append({'file': key, 'group': group, **detail})
        except (OSError, ValueError, csv.Error) as exc:
            problems.append({'file': key, 'reason': 'unreadable_or_corrupt', 'error': str(exc)[:180]})
    return {'before': str(old_root), 'after': str(new_root),
            'before_data_sha256': old['sha256'], 'after_data_sha256': new['sha256'],
            'totals': {name: dict(counts) for name, counts in sorted(totals.items())},
            'examples': examples, 'problem_count': len(problems), 'problems': problems[:100]}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('before', type=Path)
    ap.add_argument('after', type=Path)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    report = compare(args.before, args.after)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'totals': report['totals'], 'problem_count': report['problem_count'],
                      'report': str(args.output)}, ensure_ascii=False))
    return 2 if report['problem_count'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
