#!/usr/bin/env python
"""数据态指纹台账 — 让净值漂移可秒级归因 (2026-10-16)

动机: 10-14→10-15 一次数据刷新使 anchor_1015c 2,461,946 → anchor_1015d 1,518,086 (−38.3%),
      而归因全靠人工回忆("concept_heat 908只 + 22只fq_reload"). 产品化(对外订阅)后,
      任何一次净值重述都必须能立刻回答"是哪个文件变了" —— 靠回忆不可交付。

用法:
  datastate.py snap <label>                     # 计算当前数据态指纹 → .arms/datastate/<label>.json
  datastate.py diff <labelA> <labelB>           # 对比两个指纹, 列出变化文件(按影响面排序)
  datastate.py ls                               # 列出所有快照
  datastate.py rec <label> <tag> <nav> <md5> [env...]   # 记一次回测到 ledger.csv
  datastate.py show                             # 打印 ledger.csv

指纹口径:
  - 大文件(pkl/csv ≥ 8MB): md5 全量 (concept_hist / margin_detail / blocktrades ...)
  - 小文件: md5 全量
  - 大目录(backtrader_data / fundamental_data / raw_data, 数千文件): 流式 md5 over
    (相对路径, 字节数, mtime_ns) 三元组 ⇒ 捕捉 增/删/改 且 5000+ 文件 <2s
  ⇒ 指纹只对"数据内容"敏感, 对读取顺序/绝对路径不敏感.
"""
import sys, os, json, hashlib, time

ROOT = '/mnt/d/quant'
STORE = '/mnt/d/quant/.arms/datastate'
LEDGER = '/mnt/d/quant/.arms/ledger.csv'

# 需要逐文件全量 md5 的目录(文件数少或单文件重要)
FILE_DIRS = [
    'data',
    'data/alternative_data',
    'data/macro',
    'data/sentiment_data',
]
# 需要"路径+大小+mtime"流式指纹的大目录
TREE_DIRS = [
    'data/stock_data/backtrader_data',
    'data/stock_data/fundamental_data',
    'data/stock_data/raw_data',
    'data/research',
]
# 顶层 data/ 下不递归(避免把 TREE_DIRS 重复算一遍)
FILE_DIRS_TOPONLY = {'data'}


def md5_file(p, chunk=1 << 22):
    h = hashlib.md5()
    with open(p, 'rb') as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def fingerprint():
    fp = {}
    for d in FILE_DIRS:
        full = os.path.join(ROOT, d)
        if not os.path.isdir(full):
            continue
        toponly = d in FILE_DIRS_TOPONLY
        for name in sorted(os.listdir(full)):
            p = os.path.join(full, name)
            if toponly and os.path.isdir(p):
                continue  # 大目录另走 TREE_DIRS
            if not os.path.isfile(p):
                continue
            if not name.endswith(('.pkl', '.csv', '.parquet', '.json', '.txt')):
                continue
            try:
                fp[f'{d}/{name}'] = md5_file(p)
            except Exception as e:
                fp[f'{d}/{name}'] = f'ERR:{e}'
    for d in TREE_DIRS:
        full = os.path.join(ROOT, d)
        if not os.path.isdir(full):
            continue
        h = hashlib.md5()
        n = 0
        tot = 0
        for dirpath, dirnames, filenames in os.walk(full):
            dirnames.sort()
            for fn in sorted(filenames):
                p = os.path.join(dirpath, fn)
                try:
                    st = os.stat(p)
                except OSError:
                    continue
                rel = os.path.relpath(p, full)
                h.update(f'{rel}|{st.st_size}|{st.st_mtime_ns}\n'.encode())
                n += 1
                tot += st.st_size
        fp[d] = f'TREE:{h.hexdigest()}:{n}f:{tot}B'
    return fp


def cmd_snap(label):
    os.makedirs(STORE, exist_ok=True)
    t0 = time.time()
    fp = fingerprint()
    out = {'label': label, 'ts': time.strftime('%Y-%m-%d %H:%M:%S'), 'files': fp}
    p = os.path.join(STORE, f'{label}.json')
    with open(p, 'w') as f:
        json.dump(out, f, indent=1)
    print(f'✅ 快照 {label} → {p}')
    print(f'   {len(fp)} 项, 耗时 {time.time()-t0:.1f}s')


def cmd_ls():
    if not os.path.isdir(STORE):
        print('(无快照)')
        return
    for f in sorted(os.listdir(STORE)):
        if not f.endswith('.json'):
            continue
        d = json.load(open(os.path.join(STORE, f)))
        print(f'  {d["label"]:<24} {d["ts"]}  {len(d["files"])} 项')


def cmd_diff(la, lb):
    A = json.load(open(os.path.join(STORE, f'{la}.json')))
    B = json.load(open(os.path.join(STORE, f'{lb}.json')))
    fa, fb = A['files'], B['files']
    keys = sorted(set(fa) | set(fb))
    changed = []
    for k in keys:
        va, vb = fa.get(k), fb.get(k)
        if va == vb:
            continue
        if va is None:
            kind = '新增'
        elif vb is None:
            kind = '删除'
        else:
            kind = '变更'
        changed.append((k, kind, va, vb))
    print(f'=== 数据态 diff: {la}({A["ts"]}) → {lb}({B["ts"]}) ===')
    print(f'  共 {len(keys)} 项, 变化 {len(changed)} 项\n')
    if not changed:
        print('  ✅ 逐项一致 —— 数据态未变')
        return
    for k, kind, va, vb in changed:
        # TREE 项附上文件数/字节数变化, 更可读
        extra = ''
        if va and va.startswith('TREE:') and vb and vb.startswith('TREE:'):
            try:
                na, sa = va.split(':')[2], va.split(':')[3]
                nb, sb = vb.split(':')[2], vb.split(':')[3]
                extra = f'  [{na}→{nb} 文件, {int(sa[:-1])/1e6:.1f}→{int(sb[:-1])/1e6:.1f}MB]'
            except Exception:
                pass
        print(f'  {kind:<4} {k}{extra}')


def cmd_rec(label, tag, nav, md5, env):
    hdr = 'ts,data_label,tag,nav,md5,env\n'
    new = not os.path.exists(LEDGER)
    with open(LEDGER, 'a') as f:
        if new:
            f.write(hdr)
        e = ' '.join(env).replace(',', ';')
        f.write(f'{time.strftime("%Y-%m-%d %H:%M:%S")},{label},{tag},{nav},{md5},{e}\n')
    print(f'✅ ledger ← {label} {tag} NAV={nav}')


def cmd_show():
    if not os.path.exists(LEDGER):
        print('(无 ledger)')
        return
    rows = open(LEDGER).read().strip().split('\n')
    print(rows[0])
    for r in rows[1:]:
        print(r)


if __name__ == '__main__':
    a = sys.argv[1:]
    if not a:
        print(__doc__)
    elif a[0] == 'snap' and len(a) >= 2:
        cmd_snap(a[1])
    elif a[0] == 'ls':
        cmd_ls()
    elif a[0] == 'diff' and len(a) >= 3:
        cmd_diff(a[1], a[2])
    elif a[0] == 'rec' and len(a) >= 5:
        cmd_rec(a[1], a[2], a[3], a[4], a[5:])
    elif a[0] == 'show':
        cmd_show()
    else:
        print(__doc__)
