"""Persist the exact pre-execution context alongside generated signals."""
import hashlib
import os
from pathlib import Path
import pickle
import tempfile


def context_path(signals_path):
    return Path(signals_path).with_name('execution_context.pkl')


def load_context(signals_path, verify_stat=True):
    try:
        stat = Path(signals_path).stat()
        with context_path(signals_path).open('rb') as stream:
            context = pickle.load(stream)
        if context.get('version') != 1:
            return None
        if verify_stat and context.get('signals_stat') != (stat.st_size, stat.st_mtime_ns):
            return None
        if context['index_data'] is None or not isinstance(context['industry_stock_counts'], dict):
            return None
        return context
    except (OSError, ValueError, EOFError, pickle.UnpicklingError, KeyError):
        return None


def load_sealed_context(signals_path, expected_sha256):
    digest = hashlib.sha256()
    with Path(signals_path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    if digest.hexdigest() != expected_sha256:
        raise ValueError('sealed signal content differs from replay identity')
    return load_context(signals_path, verify_stat=False)


def save_context(strategy, signals_path):
    stat = Path(signals_path).stat()
    path = context_path(signals_path)
    context = {'version': 1, 'signals_stat': (stat.st_size, stat.st_mtime_ns),
               'index_data': strategy.index_data,
               'industry_stock_counts': getattr(strategy, 'industry_stock_counts', {})}
    fd, temp = tempfile.mkstemp(dir=path.parent, prefix=path.name + '.')
    try:
        with os.fdopen(fd, 'wb') as stream:
            pickle.dump(context, stream, protocol=pickle.HIGHEST_PROTOCOL)
            stream.flush(); os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)
