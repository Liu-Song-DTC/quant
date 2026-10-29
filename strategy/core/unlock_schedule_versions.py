"""Daily observed snapshots for the unlock filter's time-varying vendor table.

The first snapshot is only a frozen prehistory proxy. Without issuer notices
we cannot reconstruct when an unlock or share-count revision became public.
"""
from datetime import date
import hashlib
import json
import os
from pathlib import Path

import pandas as pd


def _manifest_path(source):
    return Path(source).with_name('unlock_schedule_versions.json')


def record_version(source, observed_on=None):
    source = Path(source)
    day = date.today() if observed_on is None else date.fromisoformat(str(observed_on))
    if day > date.today():
        raise ValueError('unlock snapshot cannot be future dated')
    content = source.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    manifest = _manifest_path(source)
    state = json.loads(manifest.read_text()) if manifest.exists() else {'schema': 1, 'versions': []}
    if state.get('schema') != 1 or not isinstance(state.get('versions'), list):
        raise ValueError('invalid unlock schedule version manifest')
    versions = state['versions']
    if versions and day.isoformat() < versions[-1]['observed_on']:
        raise ValueError('backdated unlock snapshot would rewrite history')
    archive = source.parent / 'unlock_versions' / (digest + '.pkl')
    archive.parent.mkdir(parents=True, exist_ok=True)
    if archive.exists():
        if hashlib.sha256(archive.read_bytes()).hexdigest() != digest:
            raise ValueError('unlock snapshot archive content mismatch')
    else:
        temp = archive.with_suffix('.tmp')
        temp.write_bytes(content)
        os.replace(temp, archive)
    entry = {'observed_on': day.isoformat(), 'sha256': digest}
    if versions and versions[-1]['observed_on'] == entry['observed_on']:
        if versions[-1] == entry:
            return 'same'
        versions[-1] = entry
        result = 'updated_today'
    else:
        versions.append(entry)
        result = 'added'
    temp = manifest.with_suffix('.tmp')
    temp.write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.replace(temp, manifest)
    return result


def resolve_asof(source, decision_date):
    """Use last observed snapshot <= decision date; first for prehistory proxy."""
    source = Path(source)
    manifest = _manifest_path(source)
    if not manifest.exists():
        raise FileNotFoundError(f'unlock version manifest missing: {manifest}')
    state = json.loads(manifest.read_text())
    versions = state.get('versions') or []
    if state.get('schema') != 1 or not versions:
        raise ValueError('unlock snapshot manifest is empty or invalid')
    target = pd.Timestamp(str(decision_date)[:10]).date().isoformat()
    eligible = [v for v in versions if v['observed_on'] <= target]
    entry = eligible[-1] if eligible else versions[0]
    digest = entry['sha256']
    if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
        raise ValueError('invalid unlock snapshot digest')
    archive = source.parent / 'unlock_versions' / (digest + '.pkl')
    if not archive.exists() or hashlib.sha256(archive.read_bytes()).hexdigest() != digest:
        raise ValueError(f'unlock snapshot missing/corrupt: {archive}')
    return archive


def verify_current(source):
    """The latest published table and the latest version must match exactly."""
    source = Path(source)
    manifest = _manifest_path(source)
    if not manifest.exists():
        raise ValueError('unlock version manifest missing')
    state = json.loads(manifest.read_text())
    versions = state.get('versions') or []
    if state.get('schema') != 1 or not versions:
        raise ValueError('unlock version manifest empty/invalid')
    latest = versions[-1]
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if latest['sha256'] != digest:
        raise ValueError('latest unlock version differs from published table')
    resolve_asof(source, latest['observed_on'])
    return latest['observed_on']
