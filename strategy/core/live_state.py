"""Crash-recoverable JSON commits. Publish orders after the corresponding state."""
import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path


def atomic_json(path, value):
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode('utf-8')
    atomic_bytes(path, payload)


def atomic_bytes(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def recover_commit(journal):
    journal = Path(journal)
    if not journal.exists():
        return False
    entries = json.loads(journal.read_text(encoding='utf-8'))['updates']
    # Only direct siblings of the journal can be transaction members.
    for name, value in entries:
        if Path(name).name != name:
            raise ValueError('invalid state journal member')
        atomic_json(journal.parent / name, value)
    journal.unlink()
    return True


def commit_json_files(updates, journal):
    journal = Path(journal)
    recover_commit(journal)
    entries = []
    for path, value in updates.items():
        path = Path(path)
        if path.resolve().parent != journal.resolve().parent:
            raise ValueError('state files must share a directory')
        entries.append([path.name, value])
    atomic_json(journal, {'version': 1, 'updates': entries})
    recover_commit(journal)


@contextmanager
def live_run_lock(root):
    import fcntl
    with open(Path(root) / '.live_run.lock', 'a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('another live decision/reconciliation is running') from exc
        try:
            recover_commit(Path(root) / '.live_state_journal.json')
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)
