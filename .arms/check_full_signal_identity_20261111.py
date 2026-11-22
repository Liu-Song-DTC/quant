"""Bounded-memory, order-independent stock-block comparison of all signal fields."""
from pathlib import Path
import datetime
import hashlib
import json
import struct
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path('/mnt/d/quant')
REFERENCE = ROOT / '.review_runs/arm_pct_main_final_compat_20261108'
FRESH = ROOT / '.review_runs/arm_tv406_clean_rebuild_20261111'


def canonical_table(path):
    before = path.stat()
    parquet = pq.ParquetFile(path)
    names = parquet.schema_arrow.names
    numeric = {f.name for f in parquet.schema_arrow
               if pa.types.is_integer(f.type) or pa.types.is_floating(f.type) or pa.types.is_boolean(f.type)}
    states = {}
    count = 0
    for batch in parquet.iter_batches(batch_size=65536, use_threads=False):
        frame = batch.to_pandas()
        count += len(frame)
        for code, rows in frame.groupby('code', sort=False):
            code = str(code)
            days = rows.date.astype(str)
            assert days.is_monotonic_increasing and days.is_unique, code
            if code not in states:
                states[code] = {'rows': 0, 'last_date': None,
                                'columns': {name: hashlib.sha256() for name in names}}
            state = states[code]
            assert state['last_date'] is None or state['last_date'] < days.iloc[0], code
            state['last_date'] = days.iloc[-1]
            state['rows'] += len(rows)
            for name in names:
                if name in numeric:
                    values = rows[name].to_numpy(dtype=np.float64, na_value=np.nan, copy=True)
                    values[np.isnan(values)] = np.nan
                    values[values == 0.] = 0.  # Numeric equality treats -0 and +0 identically.
                    data = values.astype('<f8', copy=False).tobytes()
                else:
                    data = bytearray()
                    for value in rows[name]:
                        if pd.isna(value):
                            data.extend(b'\xff')
                        else:
                            encoded = str(value).encode('utf-8')
                            data.extend(b'\x00')
                            data.extend(struct.pack('<Q', len(encoded)))
                            data.extend(encoded)
                state['columns'][name].update(data)
    after = path.stat()
    assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), str(path)
    result = {}
    for code, state in states.items():
        result[code] = {'rows': state['rows'], 'last_date': state['last_date'],
                        'columns': {name: digest.hexdigest() for name, digest in state['columns'].items()}}
    return {'rows': count, 'column_order': names, 'numeric_columns': sorted(numeric), 'codes': result}


def main():
    start = time.perf_counter()
    suffix = 'strategy/rolling_validation_results/backtest_signals.csv.store.parquet'
    original = canonical_table(REFERENCE / suffix)
    print('ORIGINAL_SIGNAL_CANONICALIZED', original['rows'], len(original['codes']), flush=True)
    fresh = canonical_table(FRESH / suffix)
    assert original['column_order'] == fresh['column_order']
    assert original['numeric_columns'] == fresh['numeric_columns']
    left, right = set(original['codes']), set(fresh['codes'])
    differences = []
    for code in sorted(left & right):
        a, b = original['codes'][code], fresh['codes'][code]
        columns = [name for name in original['column_order'] if a['columns'][name] != b['columns'][name]]
        if a['rows'] != b['rows'] or columns:
            differences.append({'code': code, 'original_rows': a['rows'], 'fresh_rows': b['rows'],
                                'different_columns': columns})
    passed = not differences and left == right and original['rows'] == fresh['rows']
    identity_payload = json.dumps(original, sort_keys=True, separators=(',', ':')).encode()
    fresh_payload = json.dumps(fresh, sort_keys=True, separators=(',', ':')).encode()
    report = {'checked_at': datetime.datetime.now().isoformat(), 'passed': passed,
        'original_rows': original['rows'], 'fresh_rows': fresh['rows'],
        'columns': len(original['column_order']), 'original_codes': len(left), 'fresh_codes': len(right),
        'missing_codes': sorted(left - right), 'extra_codes': sorted(right - left),
        'different_code_count': len(differences), 'differences': differences,
        'original_canonical_sha256': hashlib.sha256(identity_payload).hexdigest(),
        'fresh_canonical_sha256': hashlib.sha256(fresh_payload).hexdigest(),
        'elapsed_seconds': time.perf_counter() - start,
        'canonicalization': 'Per-stock/date chronological column streams; numeric float64 little-endian, canonical NaN and unsigned zero; UTF-8 length-prefixed nonmissing text. Independent of worker completion order and Parquet chunk boundaries. Not byte equality of differently ordered files.',
        'scope': 'All unfiltered generated signal rows/fields. Does not replace account parity or pre/post sealed-input validation. Processing uses 65536-row batches rather than loading two full tables.',
        'production_changed': False}
    (ROOT / 'logs/tv406_full_signal_identity_20261111.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print('FULL_SIGNAL_IDENTITY', json.dumps({k: v for k, v in report.items() if k != 'differences'}), flush=True)


if __name__ == '__main__':
    main()
