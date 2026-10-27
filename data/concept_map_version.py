#!/usr/bin/env python
"""概念映射**版本快照** — 每次刷新追加一版 (as-of 成分股归属) (2026-10-16)

动机(用户 2026-10-16 "概念PIT前视距也要修"):
  概念层是全系统**唯一**没有日期维度的输入 —— `stock_concept_map.pkl` 只有
  **当前**成分股名单, 拿它回测历史 = 前视; 且每次刷新全史重写 = 曲线不可复现。
  修法见 core/concept_heat.py + core/signal_engine.py 的 QUANT_CONCEPT_MAP_HIST 钩子,
  它按决策日 τ 取"截止 τ 的最近版本"名单。

⚠ **本文件是时间敏感的**: 东财只提供**当前**成分股名单, 历史名单不可回补。
  今天不记, 这一天的归属就**永久丢失** —— 事后无论花多少算力都拿不回来。
  ⇒ 必须挂到每日刷新的**成功路径**上, 而不是等要用的时候再补。

用法:
  命令行:  python concept_map_version.py record [src] [note]
  import:  from concept_map_version import record_version
           record_version(map_dict, src='refresh_concept_map', note='...')

语义:
  · 版本键 = 本地日期(YYYY-MM-DD), 粒度=日。
  · 同日重刷(md5 不同) ⇒ **覆盖当日版本**(同一天内属同一 as-of 时刻, 非重述);
    历史日期**永不修改**。
  · md5 相同 ⇒ 幂等跳过, 不产生新版本。
  · 已有台账只允许新增当日版本，拒绝插入早期/未来日期；历史补证须另建数据版本。
  · 写盘 tmp + os.replace(原子) — 见 data/refresh_concept_map_0903.py 同源修复。
"""
import os
import sys
import time
import pickle
import hashlib
from datetime import date as calendar_date

HIST = '/mnt/d/quant/data/concept_map_history.pkl'


def _md5_of_map(m):
    h = hashlib.md5()
    for code in sorted(m):
        h.update(f'{code}:{",".join(sorted(m[code]))};'.encode('utf-8'))
    return h.hexdigest()[:16]


def _same_map(a, b):
    """内容级相等 — **不用 md5 比较**: 历史里 5/31~10/16 六版的 md5 出自一次性
    ad-hoc 构建脚本(32 位, 哈希方案不明), 与本模块的 16 位方案不可比。
    实测 10/16 版与当前 pkl **内容逐只相同**但 md5 不同 ⇒ md5 判等会误判为
    "变了" ⇒ 同日重刷时白写一次。内容比较是权威判据, md5 仅作便利字段。"""
    if len(a) != len(b):
        return False
    for k, v in a.items():
        w = b.get(k)
        if w is None or sorted(v) != sorted(w):
            return False
    return True


def record_version(map_dict, src='manual', note='', hist=HIST, date=None):
    """记录当日版本；历史异值返回 exists，回填或未来日期拒绝写入。"""
    today = time.strftime('%Y-%m-%d')
    d = date or today
    if not isinstance(d, str) or calendar_date.fromisoformat(d).isoformat() != d:
        raise ValueError('concept version date must use YYYY-MM-DD')
    if d > today:
        raise ValueError(f'refusing future-dated concept snapshot: {d}')
    md5 = _md5_of_map(map_dict)
    if os.path.exists(hist):
        with open(hist, 'rb') as f:
            h = pickle.load(f)
    else:
        h = {'versions': []}
    vs = h.setdefault('versions', [])
    cur = next((v for v in vs if str(v.get('date')) == d), None)
    # An inserted earlier snapshot would change which map is the fixed first
    # version, or rewrite an already covered interval. Treat historical repairs
    # as a separate data version, never as an ordinary daily append.
    if cur is None and vs and d <= max(str(v['date']) for v in vs):
        raise ValueError(f'refusing backdated concept insertion: {d}')
    if cur is None and vs and d != today:
        raise ValueError(f'historical concept backfill requires a separate data version: {d}')
    if cur is not None and _same_map(map_dict, cur.get('map') or {}):
        return 'same'
    entry = {'date': d, 'src': src, 'note': note, 'md5': md5,
             'map': {k: list(v) for k, v in map_dict.items()}}
    if cur is not None:
        # **只允许覆盖"今天"**: 同日重刷(日内多次刷新)属同一 as-of 时刻, 非重述。
        # 历史日期**一律拒绝** —— 覆盖历史 = 事后篡改 PIT 台账, 与只追加原则冲突。
        if d != time.strftime('%Y-%m-%d'):
            return 'exists'
        vs[vs.index(cur)] = entry
        res = 'updated'
    else:
        vs.append(entry)
        res = 'added'
    vs.sort(key=lambda v: str(v['date']))
    h['built'] = time.strftime('%Y-%m-%d %H:%M:%S')
    tmp = hist + '.tmp'
    with open(tmp, 'wb') as f:
        pickle.dump(h, f)
    os.replace(tmp, hist)
    return res


if __name__ == '__main__':
    if len(sys.argv) >= 2 and sys.argv[1] == 'record':
        MAP = '/mnt/d/quant/data/stock_concept_map.pkl'
        with open(MAP, 'rb') as f:
            m = pickle.load(f)
        src = sys.argv[2] if len(sys.argv) > 2 else 'cli'
        note = ' '.join(sys.argv[3:])
        r = record_version(m, src=src, note=note)
        print(f'版本快照: {r} ({len(m)} 只) → {HIST}')
    else:
        with open(HIST, 'rb') as f:
            h = pickle.load(f)
        print(f'built={h.get("built")}  版本数={len(h.get("versions", []))}')
        for v in h['versions']:
            print(f'  {v["date"]}  {len(v.get("map") or {})} 只  md5={v.get("md5")}  '
                  f'src={v.get("src")}  {v.get("note", "")}')
