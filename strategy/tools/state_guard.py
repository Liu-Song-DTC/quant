#!/usr/bin/env python3
"""state_guard.py — 确定性重放+数据态自检 (缺口5工程硬化最小闭环)

背景(2026-09-13): 本周数据态事故账 — 锚点跳变888,434→979,885(数据层重生成,
无指纹可查)、venv分裂(A/B回测差41万, 三指纹未覆盖解释器)、脏数据重置靠人肉
备份。本工具回答一个前置问题: 回测状态漂移时, 是代码层/环境层/数据层哪一层变了?

指纹分层:
  code    git HEAD + git status porcelain哈希 + factor_config.yaml sha256
  env     解释器路径 + python/numpy/pandas/xgboost/scipy/backtrader版本
  data    quick=全文件(size,mtime)清单 + 分层聚合; deep=逐文件sha256 (--deep)
  outputs 信号/净值/持仓/成交/诊断 五个回测产物的sha256

用法 (必须显式.venv, verify会警告非.venv解释器):
  register  .venv/bin/python strategy/tools/state_guard.py register --label 基线0913 [--deep]
  verify    .venv/bin/python strategy/tools/state_guard.py verify [--vs <id>] [--deep]
  list      .venv/bin/python strategy/tools/state_guard.py list

工作流: 冷跑/出单前 verify (exit 0=无漂移, 1=有漂移, 分层定位);
  每次成功出单后 register 新锚。registry元数据入库跟踪,
  逐文件清单走 sidecar (state_manifest_latest.json, gitignore, 每日变化)。
只读。串行。
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(os.environ.get('STATE_GUARD_ROOT', '/mnt/d/quant'))
TOOLS = ROOT / 'strategy' / 'tools'
REGISTRY = TOOLS / 'state_registry.json'
MANIFEST_SIDECAR = TOOLS / 'state_manifest_latest.json'
VENV = '/mnt/d/quant/.venv/bin/python'  # 真实venv路径, 不随ROOT覆盖

# 指纹覆盖层: (层名, 类型dir/file, 相对路径)
LAYERS = [
    ('bt', 'dir', 'data/stock_data/backtrader_data'),
    ('fund', 'dir', 'data/stock_data/fundamental_data'),
    ('alt', 'dir', 'data/alternative_data'),
    ('raw', 'dir', 'data/stock_data/raw_data'),
    ('concept', 'file', 'data/concept_hist.pkl'),
    ('concept_map', 'file', 'data/stock_concept_map.pkl'),
    ('yaml', 'file', 'strategy/config/factor_config.yaml'),
]
OUTPUTS = [
    'strategy/rolling_validation_results/backtest_signals.csv',
    'strategy/rolling_validation_results/equity_curve.csv',
    'strategy/rolling_validation_results/portfolio_selections.csv',
    'strategy/rolling_validation_results/trade_realized.csv',
    'strategy/rolling_validation_results/backtest_diagnostics.json',
]
PKGS = ['numpy', 'pandas', 'xgboost', 'scipy', 'backtrader']


def _sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _git_head():
    import subprocess
    try:
        r = subprocess.run(['git', '-C', str(ROOT), 'log', '-1', '--format=%H'],
                           capture_output=True, text=True, timeout=30)
        return r.stdout.strip() if r.returncode == 0 else 'git不可用'
    except Exception as e:
        return f'git不可用: {e}'


def _git_porcelain_hash():
    import subprocess
    try:
        r = subprocess.run(['git', '-C', str(ROOT), 'status', '--porcelain'],
                           capture_output=True, text=True, timeout=60)
        if r.returncode != 0:
            return 'git不可用'
        lines = sorted(r.stdout.splitlines())
        return hashlib.sha256('\n'.join(lines).encode()).hexdigest()
    except Exception as e:
        return f'git不可用: {e}'


def _manifest(roots):
    """roots: [(name, Path)] → {name: [ (relpath, size, mtime), ... ]} 排序稳定"""
    out = {}
    for name, p in roots:
        files = []
        if p.is_dir():
            for dirpath, dirnames, filenames in os.walk(p):
                dirnames.sort()
                for fn in sorted(filenames):
                    fp = Path(dirpath) / fn
                    try:
                        st = fp.stat()
                    except OSError:
                        continue
                    files.append((str(fp.relative_to(p)), st.st_size, st.st_mtime))
        else:
            try:
                st = p.stat()
                files = [(p.name, st.st_size, st.st_mtime)]
            except OSError:
                files = []
        out[name] = files
    return out


def _manifest_blob(man):
    parts = []
    for name in sorted(man):
        for rel, size, mtime in man[name]:
            parts.append(f'{name}|{rel}|{size}|{mtime:.6f}')
    return '\n'.join(parts)


def _env_fingerprint():
    import importlib.metadata as md
    env = {'executable': sys.executable, 'python': sys.version.split()[0]}
    for p in PKGS:
        try:
            env[p] = md.version(p)
        except Exception:
            env[p] = '缺失'
    return env


def _collect(deep):
    roots = [(name, ROOT / rel) for name, kind, rel in LAYERS]
    t0 = time.time()
    man = _manifest(roots)
    st = {'manifest': man}
    st['quick_secs'] = time.time() - t0
    n_files = sum(len(v) for v in man.values())
    total = sum(sz for v in man.values() for _, sz, _ in v)
    st['aggregates'] = {name: {'files': len(man[name]),
                               'bytes': sum(sz for _, sz, _ in man[name])}
                        for name in man}
    st['manifest_hash'] = hashlib.sha256(_manifest_blob(man).encode()).hexdigest()
    if deep:
        t1 = time.time()
        layer_path = {n: ROOT / rel for n, _k, rel in LAYERS}
        st['deep'] = {}
        for name in man:
            for rel, size, mtime in man[name]:
                fp = layer_path[name] if layer_path[name].is_file() \
                    else layer_path[name] / rel
                st['deep'][f'{name}|{rel}'] = _sha256_file(fp)
        st['deep_secs'] = time.time() - t1
    st['outputs'] = {}
    for rel in OUTPUTS:
        fp = ROOT / rel
        st['outputs'][rel] = _sha256_file(fp) if fp.exists() else '缺失'
    st['code'] = {'git_head': _git_head(), 'porcelain_hash': _git_porcelain_hash(),
                  'yaml_hash': _sha256_file(ROOT / 'strategy/config/factor_config.yaml')}
    st['env'] = _env_fingerprint()
    return st, n_files, total


def _load_registry():
    if REGISTRY.exists():
        try:
            return json.loads(REGISTRY.read_text())
        except Exception:
            pass
    return {'entries': []}


def _save_registry(reg):
    REGISTRY.write_text(json.dumps(reg, ensure_ascii=False, indent=1))


def _load_sidecar():
    if MANIFEST_SIDECAR.exists():
        try:
            return json.loads(MANIFEST_SIDECAR.read_text())
        except Exception:
            pass
    return None


def cmd_register(args):
    st, n_files, total = _collect(deep=args.deep)
    reg = _load_registry()
    eid = time.strftime('%Y%m%d-%H%M%S') + f'-{int(time.time() * 1000) % 1000:03d}'
    entry = {
        'id': eid,
        'label': args.label,
        'deep': bool(args.deep),
        'code': st['code'],
        'env': st['env'],
        'aggregates': st['aggregates'],
        'manifest_hash': st['manifest_hash'],
        'outputs': st['outputs'],
        'n_files': n_files,
        'total_bytes': total,
    }
    reg['entries'].insert(0, entry)
    reg['entries'] = reg['entries'][:50]
    _save_registry(reg)
    sidecar = {'id': eid, 'manifest_hash': st['manifest_hash'],
               'manifest': {k: [[r, s, m] for r, s, m in v]
                            for k, v in st['manifest'].items()}}
    if st.get('deep'):
        sidecar['deep'] = st['deep']
    MANIFEST_SIDECAR.write_text(json.dumps(sidecar))
    print(f'[register] 锚点 {eid} (标签: {args.label})')
    print(f'  code:  HEAD {st["code"]["git_head"][:12]}')
    print(f'  env:   {st["env"]["executable"]}')
    print(f'  data:  {n_files}文件 {total/1e9:.2f}GB '
          f'(quick {st["quick_secs"]:.1f}s'
          + (f', deep {st["deep_secs"]:.1f}s' if args.deep else '') + ')')
    print(f'  outputs: 已记录 {sum(1 for v in st["outputs"].values() if v != "缺失")}'
          f'/{len(st["outputs"])}')


def _diff_manifests(cur_man, prev_man):
    added, removed, modified, touched = [], [], [], []
    for name in sorted(set(cur_man) | set(prev_man)):
        a = {r: (s, m) for r, s, m in prev_man.get(name, [])}
        b = {r: (s, m) for r, s, m in cur_man.get(name, [])}
        for rel in sorted(set(b) - set(a)):
            added.append(f'{name}|{rel}')
        for rel in sorted(set(a) - set(b)):
            removed.append(f'{name}|{rel}')
        for rel in sorted(set(a) & set(b)):
            if a[rel][0] != b[rel][0]:
                modified.append(f'{name}|{rel} ({a[rel][0]}->{b[rel][0]}B)')
            elif a[rel][1] != b[rel][1]:
                touched.append(f'{name}|{rel}')
    return added, removed, modified, touched


def cmd_verify(args):
    reg = _load_registry()
    if not reg['entries']:
        print('[verify] 无锚点, 先 register'); sys.exit(2)
    entry = reg['entries'][0]
    if args.vs:
        for e in reg['entries']:
            if e['id'].startswith(args.vs) or e['id'] == args.vs:
                entry = e
                break
    st, n_files, total = _collect(deep=args.deep)
    if sys.executable != VENV:
        print(f'[verify] 警告: 非.venv解释器 {sys.executable} (钉.venv纪律)')
    drift = False
    print(f'[verify] vs 锚点 {entry["id"]} (标签: {entry.get("label")}, '
          f'deep={entry.get("deep", False)})')

    # code层
    code_d = []
    if st['code']['git_head'] != entry['code']['git_head']:
        code_d.append(f'HEAD {entry["code"]["git_head"][:12]} -> '
                      f'{st["code"]["git_head"][:12]}')
    if st['code']['porcelain_hash'] != entry['code']['porcelain_hash']:
        code_d.append('工作区有未提交改动')
    if st['code']['yaml_hash'] != entry['code']['yaml_hash']:
        code_d.append('factor_config.yaml 变化')
    print(f'  code:   {"漂移: " + "; ".join(code_d) if code_d else "一致"}')
    drift = drift or bool(code_d)

    # env层
    env_d = []
    if st['env']['executable'] != entry['env']['executable']:
        env_d.append(f'解释器 {entry["env"]["executable"]} -> '
                     f'{st["env"]["executable"]}')
    for p in PKGS:
        if entry['env'].get(p) != st['env'].get(p):
            env_d.append(f'{p} {entry["env"].get(p)} -> {st["env"].get(p)}')
    print(f'  env:    {"漂移: " + "; ".join(env_d) if env_d else "一致"}')
    drift = drift or bool(env_d)

    # data层: 先聚合, 再逐文件(vs最新sidecar)
    agg_d = []
    for name in sorted(set(st['aggregates']) | set(entry['aggregates'])):
        a = st['aggregates'].get(name, {'files': 0, 'bytes': 0})
        b = entry['aggregates'].get(name, {'files': 0, 'bytes': 0})
        if a != b:
            agg_d.append(f'{name}: {b} -> {a}')
    sidecar = _load_sidecar()
    file_d = []
    if sidecar and sidecar['id'] == entry['id']:
        prev_man = {k: [(r, s, m) for r, s, m in v]
                    for k, v in sidecar['manifest'].items()}
        added, removed, modified, touched = _diff_manifests(st['manifest'],
                                                            prev_man)
        file_d = added + removed + modified + touched
        if args.deep and sidecar.get('deep') and st.get('deep'):
            content_d = [k for k in set(st['deep']) | set(sidecar['deep'])
                         if st['deep'].get(k) != sidecar['deep'].get(k)]
            if content_d:
                file_d.append(f'内容级漂移{len(content_d)}文件(首3: '
                              f'{", ".join(content_d[:3])})')
    print(f'  data:   {"漂移: " + "; ".join(agg_d) if agg_d else "一致"}')
    if file_d:
        print(f'          逐文件: 共{len(file_d)}处 (前8: '
              f'{", ".join(file_d[:8])}{"..." if len(file_d) > 8 else ""})')
    drift = drift or bool(agg_d) or bool(file_d)

    # outputs层
    out_d = []
    for rel in OUTPUTS:
        if st['outputs'][rel] != entry['outputs'].get(rel):
            out_d.append(f'{rel.split("/")[-1]} '
                         f'{entry["outputs"].get(rel, "缺失")[:8]}... -> '
                         f'{st["outputs"][rel][:8]}...')
    print(f'  outputs: {"漂移: " + "; ".join(out_d) if out_d else "一致"}')
    drift = drift or bool(out_d)

    print(f'结论: {"有漂移" if drift else "无漂移"} '
          f'(code{"✗" if code_d else "✓"}/env{"✗" if env_d else "✓"}/'
          f'data{"✗" if (agg_d or file_d) else "✓"}/outputs{"✗" if out_d else "✓"})')
    sys.exit(1 if drift else 0)


def cmd_list(args):
    reg = _load_registry()
    if not reg['entries']:
        print('[list] 无锚点')
        return
    print(f'[list] {len(reg["entries"])}个锚点 (最新在前)')
    for i, e in enumerate(reg['entries']):
        mark = ' *' if i == 0 else '  '
        print(f'{mark} {e["id"]}  {e.get("label", ""):<24s} '
              f'HEAD {e["code"]["git_head"][:12]}  '
              f'{e["n_files"]}文件 {e["total_bytes"]/1e9:.2f}GB')


def main():
    ap = argparse.ArgumentParser(description='数据态自检 (缺口5)')
    sub = ap.add_subparsers(dest='cmd', required=True)
    p1 = sub.add_parser('register', help='注册当前状态为锚点')
    p1.add_argument('--label', default='', help='锚点标签')
    p1.add_argument('--deep', action='store_true', help='逐文件sha256(全数据层)')
    p2 = sub.add_parser('verify', help='与锚点比对漂移')
    p2.add_argument('--vs', default=None, help='锚点id(前缀), 默认最新')
    p2.add_argument('--deep', action='store_true', help='当前侧逐文件sha256')
    sub.add_parser('list', help='列出锚点')
    args = ap.parse_args()
    if args.cmd == 'register':
        cmd_register(args)
    elif args.cmd == 'verify':
        cmd_verify(args)
    else:
        cmd_list(args)


if __name__ == '__main__':
    main()
