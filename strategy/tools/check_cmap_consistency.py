#!/usr/bin/env python
"""守卫: 概念映射静态快照 ≡ 版本表最后一版。

**为什么必须守**: 概念 PIT 开启后, 决策日 τ 的成分股名单 = "截止 τ 的最近版本"。
对**今天**的实盘决策, τ ≥ 最后一版日期 ⇒ 用最后一版, 应当**逐位等于**静态快照
`stock_concept_map.pkl`。若 refresh 只更新了静态快照而漏了版本追加, 实盘会
**静默**用一个更旧的名单 —— 与回测分叉, 且没有任何报错。这正是
`live-regime-blind-bug` 的形态(查表失败返回默认=静默温床)。

反过来也守: 若最后一版比静态快照新, 说明版本表领先, 同样要报。

退出码: 0=一致  1=不一致  2=文件缺失
用法: check_cmap_consistency.py [--quiet]
"""
import os
import pickle
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SNAP = os.path.join(ROOT, 'data', 'stock_concept_map.pkl')
HIST = os.path.join(ROOT, 'data', 'concept_map_history.pkl')
QUIET = '--quiet' in sys.argv


def fail(msg, code):
    print(f'[cmap-consistency] ✗ {msg}')
    sys.exit(code)


for p in (SNAP, HIST):
    if not os.path.exists(p):
        fail(f'文件缺失: {p}', 2)

try:
    with open(SNAP, 'rb') as f:
        snap = pickle.load(f)
    with open(HIST, 'rb') as f:
        hist = pickle.load(f)
except Exception as e:
    fail(f'加载失败: {e}', 2)

vs = sorted((v for v in hist.get('versions', []) if v.get('map')),
            key=lambda v: str(v['date']))
if not vs:
    fail('版本表为空(无带 map 的版本) ⇒ 概念 PIT 会静默退回静态快照', 1)

last_d, last_m = str(vs[-1]['date']), vs[-1]['map']

# 逐股比较归属集合(顺序无关)
sc, mc = set(snap), set(last_m)
only_snap = sc - mc
only_last = mc - sc
diff = [k for k in (sc & mc) if set(snap[k]) != set(last_m[k])]

ok = not only_snap and not only_last and not diff
if ok:
    if not QUIET:
        print(f'[cmap-consistency] ✓ 静态快照 ≡ 版本表最后一版 ({last_d}), '
              f'{len(snap)} 股 / {len(vs)} 版')
    sys.exit(0)

print(f'[cmap-consistency] ✗ 静态快照 ≠ 版本表最后一版({last_d})')
print(f'    静态 {len(snap)} 股 / 最后一版 {len(last_m)} 股')
print(f'    仅静态有 {len(only_snap)} 股' + (f' 例: {sorted(only_snap)[:5]}' if only_snap else ''))
print(f'    仅最后一版有 {len(only_last)} 股' + (f' 例: {sorted(only_last)[:5]}' if only_last else ''))
print(f'    归属集合不同 {len(diff)} 股' + (f' 例: {sorted(diff)[:5]}' if diff else ''))
print('    ⇒ 今天的实盘决策将用**旧名单**而回测用新名单, 二者分叉。')
print('    ⇒ 修: 重新追加一版(把当前 stock_concept_map.pkl 存进版本表)。')
sys.exit(1)
