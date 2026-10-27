"""Content identities for the input roles actually consumed by factor builders."""
import hashlib
import json
import os
from pathlib import Path


def file_content_sha256(path, optional=False):
    path = Path(path)
    try:
        stream = path.open('rb')
    except FileNotFoundError:
        if optional and not path.exists() and not path.is_symlink():
            return None
        raise
    with stream:
        first = os.fstat(stream.fileno())
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
        final = os.fstat(stream.fileno())
    try:
        current = path.stat()
    except FileNotFoundError as exc:
        raise RuntimeError(f'input disappeared while computing content identity: {path}') from exc
    def stamp(stat):
        return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    if stamp(first) != stamp(final) or stamp(final) != stamp(current):
        raise RuntimeError(f'input changed while computing content identity: {path}')
    return digest.hexdigest()


def factor_input_identity(stock_file_map, financial_path=None, extra_paths=()):
    """Hash logical roles and bytes, independent of checkout location/mtime.

    Price and financial files are required once indexed. Optional concept files
    have an explicit missing marker. Unreadable inputs never masquerade as
    missing files. No persistent size/mtime digest cache is used.
    """
    entries = []
    for code, path in sorted(stock_file_map.items()):
        entries.append(('price', str(code), file_content_sha256(path)))
    if financial_path is not None:
        directory = Path(financial_path)
        if not directory.is_dir():
            raise FileNotFoundError(f'financial directory missing: {directory}')
        entries.append(('financial_directory', '', 'present'))
        # FundamentalData scans only immediate CSV files, not hidden backup trees.
        for path in sorted(directory.glob('*.csv')):
            entries.append(('financial', path.name, file_content_sha256(path)))
    for index, value in enumerate(extra_paths):
        path = Path(value)
        role = f'extra:{index}'
        if path.is_dir():
            entries.append((role, '', 'directory'))
            for member in sorted(path.rglob('*')):
                if member.is_file():
                    entries.append((role, member.relative_to(path).as_posix(), file_content_sha256(member)))
        else:
            entries.append((role, '', file_content_sha256(path, optional=True)))
    payload = json.dumps(entries, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


def signal_identity_path(signals_path):
    path = Path(signals_path)
    return path.with_name(path.name + '.inputs.json')


def write_signal_identity(signals_path, inputs_sha256):
    """Bind a signal CSV's bytes to the data identity used to generate it."""
    from .live_state import atomic_json
    atomic_json(signal_identity_path(signals_path), {
        'version': 1, 'inputs_sha256': inputs_sha256,
        'signals_sha256': file_content_sha256(signals_path),
    })


def signal_identity_matches(signals_path, inputs_sha256):
    try:
        record = json.loads(signal_identity_path(signals_path).read_text())
        if record.get('version') != 1 or record.get('inputs_sha256') != inputs_sha256:
            return False
        return record.get('signals_sha256') == file_content_sha256(signals_path)
    except (OSError, ValueError, TypeError, AttributeError):
        return False
