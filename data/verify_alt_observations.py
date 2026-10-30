#!/usr/bin/env python3
"""Fail closed if a production alternative table lacks a sealed observation."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'strategy'))
from core.alt_data_versions import TABLES, resolve_asof, verify_current

root = Path(__file__).resolve().parent / 'alternative_data'
for name in sorted(TABLES):
    source = root / name
    if not source.is_file():
        raise FileNotFoundError(source)
    latest = verify_current(source)
    # Check both the immutable first-snapshot proxy and the current observation.
    first = resolve_asof(source, '2000-01-01')
    current = resolve_asof(source, latest)
    print(f'[alt PIT] {name}: latest={latest}, first={first.name}, current={current.name}')
