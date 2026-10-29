#!/usr/bin/env python
"""历史格哨兵 — 证明"更新今天的数据, 不改变历史" (2026-10-17)

**为什么现有的两个台账不够**

`state_guard.py` 与 `.arms/datastate.py` 都只对**整文件**做哈希(mtime+size / sha256)。
但数据文件每天都在长大 —— 合法追加与违规改写会产生**同样响的警报**, 于是警报
没有判别力, 只能靠人肉看 diff。铁律要的判据不是"文件变了没", 而是
"**历史格变了没**"。

**判据**

把每层数据拆成 (实体, 日期) 格, 对**每个日期**单独算一个摘要:

    追加(合法): 老日期的摘要**逐位不变**, 只多出新日期
    改写(违规): 老日期的摘要**变了**  ← 这就是要抓的

实测(见下)证明这不是假想问题: `anchor_1015c → anchor_1015d` 只隔 5 小时、
只多一天数据(10-15), 却改写了 **55/132 = 42%** 的历史调仓日的选股。

**用法**

    histcells.py snap <label>     # 记录当前各层历史格摘要
    histcells.py diff <label>     # 与快照比: 报出哪些格被改写
    histcells.py ls               # 列出已有快照

`diff` 对 (mtime, size) 未变的文件直接复用快照摘要 ⇒ 日常跑只读当天变过的文件。
"""
import argparse
import hashlib
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

ROOT = '/mnt/d/quant'
SNAP_DIR = os.path.join(ROOT, '.arms', 'histcells')

# (层名, 相对路径或 glob, 类型, 日期列候选, 实体列候选, 期望)
#   期望 'append' = 历史格必须零改写; 'restate' = 已知会重述(仅记录, 不判违规)
# 类型:
#   csv / pkl_df       : 平面表, 按日期列切块
#   pkl_dict_df        : {key: DataFrame}, 逐表按日期列切块
#   pkl_dict_set       : {code: set[date]} → 反转成 日期→代码集合
#   pkl_dict_list      : {code: [标签]} → 每实体一个摘要(无日期维度, 只能测实体级)
#   cmap_hist          : 概念映射版本表 {versions:[{date,map}]}
LAYERS = [
    ('raw_none',  'data/stock_data/raw_data/*/none.csv',      'csv', ['日期', 'date', 'datetime'], None, 'append'),
    ('bt',        'data/stock_data/backtrader_data/*.csv',    'csv', ['datetime', '日期', 'date'],   None, 'append'),
    ('fund',      'data/stock_data/fundamental_data/*.csv',   'csv', ['报告期', '数据可用日期'],      ['股票代码'], 'append'),
    ('concept_hist', 'data/concept_hist.pkl',                 'pkl_dict_df', ['date', '日期'], None, 'append'),
    ('concept_map_hist', 'data/concept_map_history.pkl',      'cmap_hist', None, None, 'append'),
    ('corporate_actions', 'data/alternative_data/corporate_actions.pkl', 'pkl_df', ['ex_date'], ['code'], 'append'),
    ('events_table', 'data/alternative_data/events_table.pkl', 'pkl_df', ['ex_date'], ['code'], 'append'),
    ('sina_factors', 'data/alternative_data/sina_factors.pkl', 'pkl_df', ['ex_date'], ['code'], 'append'),
    ('dragon_tiger_hist', 'data/alternative_data/dragon_tiger_history.pkl', 'pkl_dict_set', None, None, 'append'),
    ('margin_daily', 'data/alternative_data/margin_daily.pkl', 'pkl_df', ['date'], None, 'append'),
    # 供应商会重述整张日程表；历史回放只读按观测日归档的版本，完整性由 check_unlock_versions 校验。
    ('unlock_schedule', 'data/alternative_data/unlock_schedule.pkl', 'pkl_df', ['unlock_date'], ['code'], 'restate'),
    ('holdernum', 'data/alternative_data/holdernum_det.pkl',   'pkl_df', ['end_date'], ['code'], 'append'),
    # ── 2026-10-18 补: 此前 4 层**完全不在受检范围**(哨兵盲区) ───────────────
    #   前 3 层是"纯追加"语义的事件流; dragon_tiger 是**滚动 1 年快照**
    #   (最近上榜日 2025-10-17→2026-10-16), 老日期会**滚出窗口** ⇒ 天生非追加,
    #   标 'restate' 只记录不判违规(它已被 0g 裁决关闭, 不进回测)。
    ('reduction_plans', 'data/alternative_data/reduction_plans.pkl', 'pkl_df', ['ann_date'], ['code'], 'append'),
    ('reduction_records', 'data/alternative_data/reduction_records.pkl', 'pkl_df', ['end_date'], ['code'], 'append'),
    ('yjyg_records', 'data/alternative_data/yjyg_records.pkl', 'pkl_df', ['notice_date'], ['code'], 'append'),
    ('dragon_tiger', 'data/alternative_data/dragon_tiger.pkl', 'pkl_df', ['最近上榜日'], ['代码'], 'restate'),
    # ⚠ 2026-10-18 修: 原来指向 `northbound_daily.parquet` —— 那是**死文件**
    #   (mtime 停在 2026-07-05), 回测实际读的是 `northbound_daily.pkl`
    #   (alternative_data.py:305)。哨兵盯错了文件 ⇒ 该层一直是假警报静默。
    ('northbound', 'data/alternative_data/northbound_daily.pkl', 'pkl_df', ['date'], None, 'append'),
    # ── 已知会重述的层: 仅记录, 不判违规 ──────────────────────────────────
    ('raw_qfq',   'data/stock_data/raw_data/*/qfq.csv',       'csv', ['日期', 'date', 'datetime'], None, 'restate'),
    ('raw_hfq',   'data/stock_data/raw_data/*/hfq.csv',       'csv', ['日期', 'date', 'datetime'], None, 'restate'),
    ('concept_map',  'data/stock_concept_map.pkl',            'pkl_dict_list', None, None, 'restate'),
]


def _sha(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()[:16]


def _dates_from_df(df: pd.DataFrame, date_cands, key_cols):
    """按日期切块算摘要。向量化: 每行哈希只算一次, 再按日期分组拼接。"""
    dc = None
    for c in date_cands or []:
        if c in df.columns:
            dc = c
            break
    if dc is None:
        return None, f'无日期列(候选 {date_cands})'
    if df.empty:
        return {}, None

    dts = df[dc].astype(str).str[:10].values
    d = df.reindex(sorted(df.columns), axis=1).reset_index(drop=True)
    colrepr = repr(list(d.columns)) + repr([str(t) for t in d.dtypes])

    by = [c for c in (key_cols or []) if c in d.columns]
    if by:
        keys = [dts] + [d[c].astype(str).values for c in by]
        order = np.lexsort(tuple(reversed(keys)))
        d = d.iloc[order]
        dts = dts[order]

    rh = pd.util.hash_pandas_object(d, index=False).values.tobytes()
    w = 8  # uint64 → 8 字节/行
    out, start, n = {}, 0, len(dts)
    for i in range(1, n + 1):
        if i == n or dts[i] != dts[start]:
            dt = dts[start]
            if dt not in ('nan', 'NaT', ''):
                out[dt] = _sha(rh[start * w:i * w] + colrepr.encode())
            start = i
    return out, None


def _scan_file(path, kind, date_cands, key_cols):
    """返回 (dates_dict 或 None, err)。dates_dict: {key: digest}"""
    try:
        if kind == 'csv':
            o = pd.read_csv(path)
        elif kind == 'pq_df':
            o = pd.read_parquet(path)
        elif kind == 'pkl_df':
            o = pickle.load(open(path, 'rb'))
        else:
            o = pickle.load(open(path, 'rb'))
    except Exception as e:
        return None, f'读失败 {str(e)[:50]}'

    if kind in ('csv', 'pq_df', 'pkl_df'):
        if not isinstance(o, pd.DataFrame):
            return None, f'非DataFrame({type(o).__name__})'
        return _dates_from_df(o, date_cands, key_cols)

    if not isinstance(o, dict):
        return None, f'非dict({type(o).__name__})'

    if kind == 'pkl_dict_df':
        acc = {}
        for k, df in o.items():
            if not isinstance(df, pd.DataFrame):
                continue
            dd, _ = _dates_from_df(df, date_cands, key_cols)
            if dd:
                for d, dg in dd.items():
                    acc[f'{k}@{d}'] = dg
        return acc, None

    if kind == 'pkl_dict_set':          # {code: set[date]} → 日期→该日代码集合
        inv = {}
        for code, ds in o.items():
            try:
                for d in ds:
                    inv.setdefault(str(d)[:10], []).append(str(code))
            except TypeError:
                return None, '值非集合'
        return {d: _sha(repr(sorted(v)).encode()) for d, v in inv.items()}, None

    if kind == 'pkl_dict_list':         # {code: [标签]} → 每实体一个摘要
        return {str(k): _sha(repr(sorted(map(str, v))).encode()) for k, v in o.items()}, None

    if kind == 'cmap_hist':
        vs = o.get('versions', [])
        out = {}
        for v in vs:
            d = str(v.get('date'))[:10]
            m = v.get('map') or {}
            out[d] = _sha(repr(sorted((str(k), tuple(sorted(map(str, val))))
                                     for k, val in m.items())).encode())
        return out, None

    return None, f'未知类型 {kind}'


def _files_for(pat):
    import glob
    if any(c in pat for c in '*?['):      # 任一 glob 元字符, 不只是 '*'
        return sorted(glob.glob(os.path.join(ROOT, pat)))
    p = os.path.join(ROOT, pat)
    return [p] if os.path.exists(p) else []


def _scan_layer(name, pat, kind, date_cands, key_cols, prev=None, verbose=True):
    files = _files_for(pat)
    res, errs, reused, rescanned = {}, [], 0, 0
    t0 = time.time()
    for p in files:
        rel = os.path.relpath(p, ROOT)
        try:
            st = os.stat(p)
        except OSError:
            continue
        key = (int(st.st_mtime), st.st_size)
        cached = (prev or {}).get(rel)
        if cached and cached.get('k') == key and cached.get('err') is None:
            res[rel] = cached
            reused += 1
            continue
        dd, err = _scan_file(p, kind, date_cands, key_cols)
        rescanned += 1
        if err:
            errs.append(f'{rel}: {err}')
        res[rel] = {'k': key, 'd': dd, 'err': err}
    if verbose:
        print(f'  {name:<18} {len(files):>6} 文件  (重读 {rescanned}, 复用 {reused})  '
              f'{time.time()-t0:.1f}s' + (f'  ⚠{len(errs)} 错' if errs else ''), flush=True)
    return res, errs


def _scan_all(prev_all=None, verbose=True):
    out, all_errs = {}, []
    for name, pat, kind, dc, kc, expect in LAYERS:
        prev = (prev_all or {}).get(name, {}).get('files')
        files, errs = _scan_layer(name, pat, kind, dc, kc, prev, verbose)
        out[name] = {'files': files, 'spec': (pat, kind, expect)}
        all_errs += [f'[{name}] {e}' for e in errs]
    return out, all_errs


def _load(label):
    p = os.path.join(SNAP_DIR, f'{label}.pkl')
    return pickle.load(open(p, 'rb')) if os.path.exists(p) else None


def cmd_snap(args):
    os.makedirs(SNAP_DIR, exist_ok=True)
    t0 = time.time()
    print(f'[snap {args.label}] 扫描 {len(LAYERS)} 层 ...', flush=True)
    snap, errs = _scan_all(None)
    n_cells = sum(len(f['d'] or {}) for L in snap.values() for f in L['files'].values())
    n_files = sum(len(L['files']) for L in snap.values())
    blob = {'label': args.label, 'ts': time.strftime('%Y-%m-%d %H:%M:%S'),
            'layers': snap, 'git': _git_head()}
    p = os.path.join(SNAP_DIR, f'{args.label}.pkl')
    pickle.dump(blob, open(p, 'wb'), protocol=4)
    print(f'  ✓ {n_files} 文件 / {n_cells:,} 历史格 → {os.path.relpath(p, ROOT)}  '
          f'({time.time()-t0:.0f}s)')
    if errs:
        print(f'  ⚠ {len(errs)} 个文件读失败:')
        for e in errs[:10]:
            print(f'    {e}')
    return 0


def cmd_diff(args):
    old = _load(args.label)
    if old is None:
        print(f'✗ 快照 {args.label} 不存在; 先跑 snap')
        return 1
    print(f'[diff {args.label}]  快照于 {old["ts"]} (git {old.get("git", "?")[:8]})')
    new, errs = _scan_all(old['layers'])
    print()

    viol_rew = viol_gone = viol_gonef = 0
    rest_rew = 0
    tot_add = tot_newf = 0
    for name, pat, kind, dc, kc, expect in LAYERS:
        if getattr(args, 'layer', None) and name != args.layer:
            continue
        of = old['layers'].get(name, {}).get('files', {})
        nf = new.get(name, {}).get('files', {})
        rew, add, gone, newf, gonef = [], 0, [], [], []
        for rel, cur in nf.items():
            o = of.get(rel)
            if o is None:
                newf.append(rel)
                continue
            od, cd = o.get('d') or {}, cur.get('d') or {}
            if o.get('err') and cur.get('err'):
                continue
            for d in cd:
                if d not in od:
                    add += 1
                elif od[d] != cd[d]:
                    rew.append((rel, d))
            for d in od:
                if d not in cd:
                    gone.append((rel, d))
        for rel in of:
            if rel not in nf:
                gonef.append(rel)

        tag = '' if expect == 'append' else ' [已知重述层]'
        clean = not (rew or gone or gonef)
        mark = '✓' if clean else ('·' if expect == 'restate' else '✗')
        print(f'  {mark} {name:<18}{tag}  追加 {add:>6} 格 | 改写 {len(rew):>5} | '
              f'消失 {len(gone):>4} | 新实体 {len(newf):>4} | 丢文件 {len(gonef):>3}')
        if not clean:
            full = getattr(args, 'full', False)
            for rel, d in (rew if full else rew[:5]):
                print(f'      {"⚠" if expect == "append" else "·"} 改写 {rel} @ {d}')
            if not full and len(rew) > 5:
                print(f'      ... 另有 {len(rew)-5} 个格被改写')
            for rel, d in (gone if full else gone[:3]):
                print(f'      ⚠ 消失 {rel} @ {d}')
            if full:
                for rel in gonef:
                    print(f'      ⚠ 丢文件 {rel}')
        tot_add += add
        tot_newf += len(newf)
        if expect == 'append':
            viol_rew += len(rew)
            viol_gone += len(gone)
            viol_gonef += len(gonef)
        else:
            rest_rew += len(rew)

    print()
    print('=' * 72)
    if viol_rew or viol_gone or viol_gonef:
        print(f'✗ 违反铁律: 历史格被改写 {viol_rew} 个 / 消失 {viol_gone} 个 / '
              f'丢文件 {viol_gonef} 个')
        print(f'  (合法追加 {tot_add} 格 / 新实体 {tot_newf} 个 不算违规; '
              f'已知重述层另有 {rest_rew} 格)')
        return 2
    print(f'✓ 铁律成立: 受检层历史格 **零改写** (合法追加 {tot_add} 格, 新实体 {tot_newf} 个)')
    if rest_rew:
        print(f'  · 已知重述层另有 {rest_rew} 格变化(仅记录)')
    return 0


def cmd_ls(args):
    if not os.path.isdir(SNAP_DIR):
        print('(无快照)')
        return 0
    for f in sorted(os.listdir(SNAP_DIR)):
        if not f.endswith('.pkl'):
            continue
        b = pickle.load(open(os.path.join(SNAP_DIR, f), 'rb'))
        n = sum(len(L['files']) for L in b['layers'].values())
        c = sum(len(x['d'] or {}) for L in b['layers'].values() for x in L['files'].values())
        print(f'  {f[:-4]:<24} {b["ts"]}  {n} 文件 / {c:,} 格')
    return 0


def _git_head():
    try:
        import subprocess
        return subprocess.run(['git', '-C', ROOT, 'rev-parse', 'HEAD'],
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        return ''


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('snap'); s.add_argument('label'); s.set_defaults(fn=cmd_snap)
    d = sub.add_parser('diff'); d.add_argument('label'); d.set_defaults(fn=cmd_diff)
    # 2026-10-19 补: 只打前 5 个格时, "204 个改写"里的 199 个看不见 ⇒ 无法判读
    # 是"历史被改写"还是"未来日程被刷新"。--full 全量打印; --layer 只查一层。
    d.add_argument('--full', action='store_true', help='打印全部改写/消失格(不截断)')
    d.add_argument('--layer', default=None, help='只检查该层 (名字同 LAYERS)')
    l = sub.add_parser('ls'); l.set_defaults(fn=cmd_ls)
    a = ap.parse_args()
    sys.exit(a.fn(a))


if __name__ == '__main__':
    main()
