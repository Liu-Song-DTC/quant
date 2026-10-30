"""Immutable observation snapshots for mutable alternative-data supplier tables.

An event's publication date is not sufficient: the supplier can add that event
later or revise old figures. Historical queries use the last snapshot actually
observed by that date. Dates earlier than the first observation are explicitly
first-snapshot proxies; they are not proven point-in-time data.
"""
import hashlib
import json
import os
from datetime import date
from functools import lru_cache
from pathlib import Path

TABLES = frozenset({
    'dragon_tiger.pkl', 'dragon_tiger_history.pkl', 'margin_daily.pkl',
    'northbound_daily.pkl', 'reduction_records.pkl', 'reduction_plans.pkl',
})


def _paths(source):
    source = Path(source)
    if source.name not in TABLES:
        raise ValueError(f'not a versioned alternative input: {source.name}')
    folder = source.parent / 'alt_versions'
    return folder, folder / (source.stem + '.json')


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _read_manifest(path):
    state = json.loads(path.read_text(encoding='utf-8'))
    versions = state.get('versions')
    if state.get('schema') != 1 or not isinstance(versions, list) or not versions:
        raise ValueError(f'invalid alternative snapshot manifest: {path}')
    previous = ''
    for entry in versions:
        observed = date.fromisoformat(entry['observed_on']).isoformat()
        digest = entry['sha256']
        if observed < previous or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError(f'invalid alternative snapshot entry: {path}')
        previous = observed
    return state


def record_snapshot(source, observed_on=None, *, content_from=None):
    """Archive published bytes. content_from seeds a real earlier observation.

    The caller must provide the actual observation date, never the event date.
    Same-day refreshed states are retained as separate immutable blobs.
    """
    source = Path(source)
    folder, manifest = _paths(source)
    day = date.today() if observed_on is None else date.fromisoformat(str(observed_on))
    if day > date.today():
        raise ValueError('observation cannot be future dated')
    data = Path(content_from or source).read_bytes()
    digest = _digest(data)
    folder.mkdir(parents=True, exist_ok=True)
    archive = folder / f'{source.stem}_{digest}.pkl'
    if archive.exists():
        if _digest(archive.read_bytes()) != digest:
            raise ValueError(f'corrupt immutable alternative snapshot: {archive}')
    else:
        tmp = archive.with_name(archive.name + '.tmp')
        try:
            tmp.write_bytes(data)
            os.replace(tmp, archive)
        finally:
            tmp.unlink(missing_ok=True)
    state = _read_manifest(manifest) if manifest.exists() else {'schema': 1, 'versions': []}
    versions = state['versions']
    entry = {'observed_on': day.isoformat(), 'sha256': digest}
    if versions and entry['observed_on'] < versions[-1]['observed_on']:
        raise ValueError('backdated alternative snapshot would rewrite history')
    if versions and versions[-1] == entry:
        return 'same'
    versions.append(entry)
    tmp = manifest.with_name(manifest.name + '.tmp')
    try:
        tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        os.replace(tmp, manifest)
    finally:
        tmp.unlink(missing_ok=True)
    return 'added'


def verify_current(source):
    source = Path(source)
    _, manifest = _paths(source)
    state = _read_manifest(manifest)
    if _digest(source.read_bytes()) != state['versions'][-1]['sha256']:
        raise ValueError(f'alternative table differs from recorded latest observation: {source}')
    return state['versions'][-1]['observed_on']


@lru_cache(maxsize=128)
def _verify_blob(path, digest, size, mtime_ns):
    if _digest(Path(path).read_bytes()) != digest:
        raise ValueError(f'corrupt alternative snapshot: {path}')


def resolve_asof(source, decision_date, *, prehistory_proxy=True):
    source = Path(source)
    folder, manifest = _paths(source)
    state = _read_manifest(manifest)
    target = date.fromisoformat(str(decision_date)[:10]).isoformat()
    eligible = [entry for entry in state['versions'] if entry['observed_on'] <= target]
    if not eligible and not prehistory_proxy:
        return None
    entry = eligible[-1] if eligible else state['versions'][0]
    archive = folder / f'{source.stem}_{entry["sha256"]}.pkl'
    if not archive.exists():
        raise ValueError(f'missing alternative snapshot: {archive}')
    stat = archive.stat()
    _verify_blob(str(archive), entry['sha256'], stat.st_size, stat.st_mtime_ns)
    return archive
