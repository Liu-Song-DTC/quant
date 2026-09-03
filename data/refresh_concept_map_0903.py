#!/usr/bin/env python3
"""2026-09-03 刷新 stock_concept_map.pkl (旧=5/31快照, 5605只)

数据源: 东方财富 push2 clist
  1) 概念板块列表: fs=m:90+t:3+f:!50 (全部概念板块, ~600个)
  2) 每板块成分: fs=b:BKxxxx, fields=f12 (股票代码), 分页拉全
  3) 反转成 {code6: [概念名...]}
  4) 覆盖率≥95%才替换 (备份 .preMapRefresh_0903)
push2 限流时由外部循环按需重跑, 本脚本每次独立执行。
"""
import os, pickle, shutil, time
import requests
import pandas as pd

MAP = '/mnt/d/quant/data/stock_concept_map.pkl'
UA = {'User-Agent': 'Mozilla/5.0'}


def get(url, params, tries=6, base=4.0):
    last = None
    for i in range(tries):
        try:
            r = requests.get(url, params=params, headers=UA, timeout=15)
            if r.status_code == 200:
                return r
            last = f'HTTP {r.status_code}'
        except Exception as e:
            last = str(e)[:70]
        time.sleep(min(base * (1.5 ** i), 30))
    raise RuntimeError(f'{url} -> {last}')


def clist(fs, fields, max_pages=80):
    """拉取 clist 全部分页, 返回 DataFrame (push2delay: push2 被限流时的可用变体, 每页硬封顶100行)"""
    out = []
    total = None
    for pn in range(1, max_pages + 1):
        r = get('https://push2delay.eastmoney.com/api/qt/clist/get', {
            'pn': str(pn), 'pz': '100', 'po': '1', 'np': '1',
            'fltt': '2', 'invt': '2', 'fid': 'f12', 'fs': fs,
            'fields': fields,
        })
        j = r.json()
        d = j.get('data') or {}
        diff = d.get('diff') or []
        out.extend(diff)
        if total is None and 'total' in d:
            total = int(d['total'])
        if not diff or (total is not None and len(out) >= total):
            break
        time.sleep(0.2)
    return pd.DataFrame(out)


def main():
    with open(MAP, 'rb') as f:
        old_map = pickle.load(f)
    print(f'旧 map: {len(old_map)} 只, 时间 {time.strftime("%H:%M")}', flush=True)

    boards = clist('m:90+t:3+f:!50', 'f12,f14')
    print(f'板块列表: {len(boards)} 个', flush=True)
    if len(boards) < 400:
        print('板块列表异常(少于400), 放弃本轮', flush=True)
        return 1
    name_of = dict(zip(boards['f12'].astype(str), boards['f14'].astype(str)))

    new_map = {}
    ok_b, fail_b = 0, 0
    for i, (bk, bname) in enumerate(name_of.items()):
        try:
            cons = clist(f'b:{bk}', 'f12')
            codes = [str(c) for c in cons['f12'] if len(str(c)) == 6]
            for c in codes:
                new_map.setdefault(c, set()).add(bname)
            ok_b += 1
        except Exception as e:
            fail_b += 1
            if fail_b <= 3:
                print(f'  板块 {bk} {bname}: FAIL {str(e)[:60]}', flush=True)
        if i % 100 == 0:
            print(f'  进度 {i+1}/{len(name_of)}: 已抓 {len(new_map)} 只股票', flush=True)
        time.sleep(0.4)
    cov = ok_b / max(len(name_of), 1)
    print(f'板块覆盖: {ok_b}/{len(name_of)} = {cov:.1%}; 新 map {len(new_map)} 只', flush=True)
    if cov < 0.95:
        print('覆盖率不足95%, 不替换旧 map', flush=True)
        return 1

    for c, names in new_map.items():
        new_map[c] = sorted(names)
    old_concepts = {n for v in old_map.values() for n in v}
    new_concepts = {n for v in new_map.values() for n in v}
    print(f'概念数: 旧{len(old_concepts)} -> 新{len(new_concepts)} '
          f'(新增{len(new_concepts-old_concepts)} 消失{len(old_concepts-new_concepts)})', flush=True)
    if not os.path.exists(MAP + '.preMapRefresh_0903'):
        shutil.copy2(MAP, MAP + '.preMapRefresh_0903')
    with open(MAP, 'wb') as f:
        pickle.dump(new_map, f)
    print(f'=== map 刷新完成: {len(new_map)} 只 / {len(new_concepts)} 概念 ===', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
