"""Run maintained code at its canonical path with isolated filesystem writes.

Example (the caller owns input seals, execution locks and financial gates)::

    layout = prepare_layout(output_path=native_run, data_path=verified_data)
    env = dict(os.environ, NUMBA_CACHE_DIR=str(layout.output / 'runtime/numba'),
               XDG_CACHE_HOME=str(layout.output / 'runtime/xdg'))
    argv = build_command(layout, [python, '-B', 'bt_execution.py'], env=env)
    subprocess.run(argv, check=True)

Only existing, explicitly permitted output directories are overlaid. Missing
source directories are never created. Config and Python source stay read-only;
there are no source copies, import hooks or runtime patches. The host filesystem
is otherwise read-only, /tmp is private, and output itself remains writable at
its canonical path (including Numba/XDG caches and caller metadata).

Use an exclusively owned output directory: validation is not a defense against
another host process racing to replace paths between validation and exec. This
is filesystem isolation for trusted jobs, not a network/security sandbox.
"""

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
from typing import Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
# No arbitrary source-directory overlays: accepting strategy/core or config
# here would undermine the single-source contract.
WRITABLE_DIRS = (
    'strategy/cache', 'strategy/models', 'strategy/logs',
    'strategy/rolling_validation_results', 'strategy/reports',
    'strategy/analysis_results', 'strategy/daily_review',
    'strategy/live_charts', 'strategy/media_out', 'strategy/tmp_factor',
    'logs', 'reports', 'output',
)


@dataclass(frozen=True)
class RuntimeLayout:
    root: Path
    data: Path
    output: Path
    writable_dirs: tuple[str, ...]


def _overlap(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def _plain_directory(path: Path, *, required: bool) -> None:
    # Reject symlinks in any component below the canonical root, including
    # dangling links. Otherwise an overlay could resolve onto code or inputs.
    if path.resolve() != path or path.is_symlink():
        raise ValueError(f'directory must not redirect through symlinks: {path}')
    if path.exists() and not path.is_dir():
        raise ValueError(f'not a directory: {path}')
    if required and not path.is_dir():
        raise ValueError(f'missing directory: {path}')


def _validate(layout: RuntimeLayout, *, prepared: bool) -> None:
    for path in (layout.root, layout.data, layout.output):
        if not path.is_absolute() or path.resolve() != path:
            raise ValueError(f'path is no longer canonical: {path}')
    for path in (layout.root, layout.data, layout.root / 'strategy', layout.root / 'data'):
        _plain_directory(path, required=True)
    if _overlap(layout.output, layout.root) or _overlap(layout.output, layout.data):
        raise ValueError('output must be disjoint from maintained source and input data')
    # These mounts are private namespace infrastructure, never whole-directory
    # source/output mounts. Children of /tmp remain useful for small fixtures.
    for path in (layout.root, layout.output):
        if path == Path('/tmp') or any(
                path == special or special in path.parents
                for special in (Path('/proc'), Path('/dev'), Path('/sys'))):
            raise ValueError(f'cannot mount source/output over namespace infrastructure: {path}')
    _plain_directory(layout.output, required=prepared)
    if len(set(layout.writable_dirs)) != len(layout.writable_dirs):
        raise ValueError('duplicate writable directory')
    for relative in layout.writable_dirs:
        if relative not in WRITABLE_DIRS:
            raise ValueError(f'not an approved output directory: {relative}')
        _plain_directory(layout.root / relative, required=True)
        _plain_directory(layout.output / relative, required=prepared)
    for relative in ('runtime', 'runtime/numba', 'runtime/xdg'):
        _plain_directory(layout.output / relative, required=prepared)


def prepare_layout(root=ROOT, data_path=None, output_path=None, *,
                   writable_dirs: Sequence[str] | None = None,
                   create: bool = True) -> RuntimeLayout:
    """Validate paths and optionally create empty directories ONLY under output.

    Paths are expanded and canonicalized before checking ancestry. Default
    overlays select existing WRITABLE_DIRS; explicit selections must exist in
    the source tree. ``prepare_layout(root, data_path, output_path)`` and keyword
    calls are supported. Output must not already exist; nothing is copied or
    cleared. ``create=False`` validates a proposed layout without writing;
    build_command requires its output directories to exist.
    """
    if data_path is None or output_path is None:
        raise TypeError('data_path and output_path are required')
    source = Path(root).expanduser().resolve(strict=True)
    data = Path(data_path).expanduser().resolve(strict=True)
    output = Path(output_path).expanduser().resolve()
    selected = tuple(writable_dirs) if writable_dirs is not None else tuple(
        name for name in WRITABLE_DIRS
        if (source / name).exists() or (source / name).is_symlink())
    layout = RuntimeLayout(source, data, output, selected)
    _validate(layout, prepared=False)
    if output.exists():
        raise FileExistsError(f'output must be a new directory: {output}')
    if create:
        output.mkdir(parents=True, exist_ok=False)
        for relative in (*selected, 'runtime/numba', 'runtime/xdg'):
            (output / relative).mkdir(parents=True, exist_ok=True)
        _validate(layout, prepared=True)
    return layout


def build_command(layout: RuntimeLayout, command: Sequence[str], *,
                  env: Mapping[str, str] | None = None, cwd=None,
                  bwrap: str = 'bwrap') -> list[str]:
    """Return argv for a direct subprocess call; never invoke a shell or job.

    By default the child inherits the environment passed to the eventual
    subprocess call, including cache paths. An explicit ``env`` instead clears
    that inherited environment and installs the supplied complete mapping.
    The default/required cwd is canonical ROOT/strategy. The launcher executable
    is resolved now, independently of the child's PATH. Source/data checks are
    repeated here so a stale layout fails before launching.
    """
    _validate(layout, prepared=True)
    if isinstance(command, (str, bytes)) or not command:
        raise ValueError('command must be a nonempty argument sequence')
    if any(not isinstance(arg, str) or '\0' in arg for arg in command) or not command[0]:
        raise ValueError('command arguments must be strings without NUL')
    workdir = layout.root / 'strategy'
    if cwd is not None and Path(cwd).expanduser().resolve(strict=True) != workdir:
        raise ValueError('cwd must be ROOT/strategy')
    environment = {} if env is None else dict(env)
    for key, value in environment.items():
        if (not isinstance(key, str) or not key or '=' in key or '\0' in key
                or not isinstance(value, str) or '\0' in value):
            raise ValueError('environment must contain valid string keys and values')
    executable = shutil.which(bwrap)
    if executable is None:
        raise FileNotFoundError(f'bubblewrap executable not found: {bwrap}')
    argv = [executable, '--die-with-parent', '--new-session',
            '--unshare-user', '--unshare-pid', '--cap-drop', 'ALL',
            '--ro-bind', '/', '/', '--tmpfs', '/tmp',
            '--proc', '/proc', '--dev', '/dev',
            # Explicit source bind also restores fixtures living under /tmp.
            '--ro-bind', str(layout.root), str(layout.root),
            '--ro-bind', str(layout.data), str(layout.root / 'data'),
            '--bind', str(layout.output), str(layout.output)]
    for relative in layout.writable_dirs:
        argv.extend(('--bind', str(layout.output / relative), str(layout.root / relative)))
    argv.extend(('--chdir', str(workdir)))
    if env is not None:
        argv.append('--clearenv')
    for key, value in sorted(environment.items()):
        argv.extend(('--setenv', key, value))
    argv.append('--')
    argv.extend(command)
    return argv
