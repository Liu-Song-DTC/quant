#!/usr/bin/env python3
"""Fail before swapping the production price directory if any old QFQ row changed.

Daily refresh may append trading days or new symbols. It must not silently
replace previously published price, volume, calendar, or auxiliary values.
Use an explicitly versioned/rebased research snapshot to handle genuine
historical corrections; this tool never approves them automatically.
"""
import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path


AUX_COLUMNS = {b'turnover_rate', b'amplitude', b'change_percent', b'change_amount'}


def audit(old_root, new_root, sample_limit=20, preserve_auxiliary=False):
    old_root, new_root = Path(old_root), Path(new_root)
    if not old_root.is_dir() or not new_root.is_dir():
        raise ValueError('both old and new QFQ directories must exist')
    old_files = sorted(old_root.glob('*_qfq.csv'))
    if not old_files:
        raise ValueError('no production QFQ files; cannot establish old history')
    checked_rows = appended_rows = auxiliary_rows = 0
    blocked = []
    auxiliary_examples = []
    auxiliary_columns = {}
    repairs = {}
    for previous in old_files:
        candidate = new_root / previous.name
        if not candidate.is_file():
            blocked.append({'file': previous.name, 'reason': 'missing_file'})
            continue
        with previous.open('rb') as old, candidate.open('rb') as new:
            old_header, new_header = old.readline(), new.readline()
            if old_header != new_header:
                blocked.append({'file': previous.name, 'reason': 'header_changed',
                                'old': old_header[:120].decode('utf-8', 'replace'),
                                'new': new_header[:120].decode('utf-8', 'replace')})
                continue
            names = old_header.rstrip(b'\r\n').split(b',')
            problem = None
            fix = {}
            for line_no, old_line in enumerate(old, 2):
                new_line = new.readline()
                checked_rows += 1
                if old_line != new_line:
                    old_fields, new_fields = old_line.rstrip(b'\r\n').split(b','), new_line.rstrip(b'\r\n').split(b',')
                    positions = [i for i in range(max(len(old_fields), len(new_fields)))
                                 if i >= len(old_fields) or i >= len(new_fields) or old_fields[i] != new_fields[i]]
                    differing = [names[i].decode('utf-8', 'replace') if i < len(names) else str(i)
                                 for i in positions] or ['row_bytes_or_line_ending']
                    if (positions and all(i < len(names) and names[i] in AUX_COLUMNS for i in positions)
                            and len(old_fields) == len(new_fields) == len(names)
                            and old_line.endswith(b'\n') == new_line.endswith(b'\n')
                            and old_line.endswith(b'\r\n') == new_line.endswith(b'\r\n')):
                        fix[line_no] = old_line
                        auxiliary_rows += 1
                        for i in positions:
                            column = names[i].decode('utf-8', 'replace')
                            auxiliary_columns[column] = auxiliary_columns.get(column, 0) + 1
                        if len(auxiliary_examples) < sample_limit:
                            auxiliary_examples.append({'file': previous.name, 'line': line_no,
                                                      'date': old_fields[0].decode('utf-8', 'replace'),
                                                      'columns': differing,
                                                      'old': old_line[:160].decode('utf-8', 'replace').rstrip(),
                                                      'new': new_line[:160].decode('utf-8', 'replace').rstrip()})
                        continue
                    problem = {'file': previous.name, 'reason': 'old_row_changed',
                               'line': line_no, 'columns': differing[:15],
                               'old': old_line[:160].decode('utf-8', 'replace').rstrip(),
                               'new': new_line[:160].decode('utf-8', 'replace').rstrip()}
                    break
            if problem is not None:
                blocked.append(problem)
                continue
            if fix:
                repairs[candidate] = fix
            appended_rows += sum(1 for _ in new)
    # No partial repairs: if any price/date/volume row changed, leave the
    # entire candidate directory untouched for inspection and versioning.
    if preserve_auxiliary and not blocked:
        for candidate, fix in repairs.items():
            temp_path = None
            try:
                with tempfile.NamedTemporaryFile(dir=candidate.parent, prefix='.qfq_history_',
                                                 delete=False) as output:
                    temp_path = Path(output.name)
                    with candidate.open('rb') as original:
                        for line_no, row in enumerate(original, 1):
                            output.write(fix.get(line_no, row))
                    output.flush()
                    os.fsync(output.fileno())
                shutil.copymode(candidate, temp_path)
                os.replace(temp_path, candidate)
            finally:
                if temp_path is not None:
                    temp_path.unlink(missing_ok=True)
    changed_count = len(blocked) + (0 if preserve_auxiliary and not blocked else len(repairs))
    return {'old_files_checked': len(old_files), 'old_rows_checked': checked_rows,
            'new_rows_appended': appended_rows, 'changed_files': changed_count,
            'blocked_files': len(blocked), 'auxiliary_only_files': len(repairs),
            'auxiliary_file_names': sorted(path.name for path in repairs),
            'auxiliary_only_rows': auxiliary_rows,
            'auxiliary_columns': auxiliary_columns,
            'auxiliary_rows_preserved': auxiliary_rows if preserve_auxiliary and not blocked else 0,
            'examples': blocked[:sample_limit], 'auxiliary_examples': auxiliary_examples}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old', type=Path, required=True)
    parser.add_argument('--new', type=Path, required=True)
    parser.add_argument('--preserve-auxiliary', action='store_true',
                        help='Carry forward only four known auxiliary columns when core bars remain byte-identical')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.old, args.new, preserve_auxiliary=args.preserve_auxiliary)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False))
    if report['changed_files']:
        print('Historical QFQ rows changed: refusing production price swap')
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
