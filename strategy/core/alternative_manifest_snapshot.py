"""Opt-in manifest lookup memoization for one immutable signal batch.

Default behavior retains live filesystem checks. The accelerated scope is
available only with alternative-data autorefresh disabled, and checks all
touched manifests again before any batch is reported successful.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import os
from pathlib import Path

_ACTIVE = ContextVar('quant_alternative_manifest_snapshot', default=None)


def _identity(path):
    if not path.exists():
        return None
    state=path.stat()
    return (state.st_dev,state.st_ino,state.st_size,state.st_mtime_ns,state.st_ctime_ns)


def manifest_exists(path):
    """Keep ordinary behavior outside an explicitly guarded batch."""
    state=_ACTIVE.get()
    if state is None:
        return path.exists()
    path=Path(path)
    if path not in state:
        state[path]=_identity(path)
    return state[path] is not None


@contextmanager
def immutable_manifest_reads():
    state={};token=_ACTIVE.set(state)
    try:
        yield
    finally:
        _ACTIVE.reset(token)
        for path,original in state.items():
            if _identity(path)!=original:
                raise RuntimeError(f'Alternative data manifest changed during signal generation: {path}')


def guarded_signal_generation(function):
    @wraps(function)
    def generate(engine,*args,**kwargs):
        # The provider is initialized lazily inside generate; checking its
        # current value here would silently bypass the first/all fresh engines.
        if os.environ.get('QUANT_ALT_MANIFEST_SNAPSHOT')!='1':
            return function(engine,*args,**kwargs)
        # Import lazily to avoid a cycle with AlternativeDataProvider.
        from .alternative_data import _AUTOREFRESH_OFF
        if not _AUTOREFRESH_OFF:
            raise RuntimeError('QUANT_ALT_MANIFEST_SNAPSHOT requires read-only alternative inputs')
        with immutable_manifest_reads():
            return function(engine,*args,**kwargs)
    return generate
