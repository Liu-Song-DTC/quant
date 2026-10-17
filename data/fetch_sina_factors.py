#!/usr/bin/env python
"""新浪**乘性复权因子**抓取 (2026-10-17)

**为什么**: 表驱动复权的事件源 `stock_fhps_detail_em`(东财分红送配表) **有洞** ——
  800 只抽样里 56 个股票日新浪有事件而东财表没有(其中 24 个幅度 >5%, 9 个 >20%,
  最大 2.5×)。漏一个 ⇒ `W(t)` 凭空造出 −17%~−60% 假跌幅。
  东财 qfq **不能当裁判**(它本身是加性失真的), 新浪因子是**纯乘性**的独立源。

**新浪语义(实测 2026-10-17)**: `ak.stock_zh_a_daily(symbol, adjust='qfq-factor')`
  返回 `(date, qfq_factor)`, **f 对日期 >= 行日期生效**, 且 `qfq(t) = raw(t)/f(t)`。
  ⇒ 除权日 d 的**乘数** `m(d) = f(上一行)/f(d)` (纯乘性 ⇒ 与锚无关 ⇒ 满足铁律)
  验证 300110 2017-05-31: 1.782664/1.187006 = 1.501815, raw 比值 12.41/8.25=1.5042
  ⇒ 当日真实收益 +0.16% ✓

⚠ **新浪因子表两个陷阱**
  ① `1900-01-01` 哨兵行 + 早于样本首日的行会经 `bisect` 塌到 `pos=0`, 造出 mult=100 伪影。
     直接剔除 `pos==0`(重建 `W(0)=raw(0)` 且循环从 t=1 起 ⇒ 第 0 日事件永不消费, 一致)。
  ② `outstanding_share` **不可当裁判**。

产出: `data/alternative_data/sina_factors.pkl`  DataFrame: code, ex_date(对齐到交易日), mult
缓存: `data/alternative_data/.sina_factors/{code}.pkl` —— **按天失效**(默认 1 天),
      否则新事件永远发现不了(旧版"存在即跳过"= 永久陈旧)。
"""
import os
import sys
import glob
import time
import pickle
import bisect
import argparse
import numpy as np
import pandas as pd
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from market_codes import is_stock_code, audit          # noqa: E402

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = f'{PROJ}/data/stock_data/raw_data'
OUT = f'{PROJ}/data/alternative_data/sina_factors.pkl'
CACHE = f'{PROJ}/data/alternative_data/.sina_factors'
MAX_AGE = 86400.0

_DAYS = {}


def prefix(code):
    return ('sh' if code[0] in '69' else 'sz' if code[0] in '03' else 'bj') + code


def work(code):
    f = os.path.join(CACHE, f'{code}.pkl')
    if os.path.exists(f) and time.time() - os.path.getmtime(f) < MAX_AGE:
        return code, 'cached'
    import akshare as ak
    time.sleep(0.05)
    days = None
    try:
        r = pd.read_csv(f'{RAW}/{code}/none.csv', encoding='utf-8-sig', usecols=['日期'])
        days = pd.to_datetime(r['日期']).values
    except Exception:
        pass
    rows = []
    ok = False
    for attempt in range(3):
        try:
            d = ak.stock_zh_a_daily(symbol=prefix(code), adjust='qfq-factor')
            if d is not None and len(d):
                d = d.copy()
                d['qfq_factor'] = d['qfq_factor'].astype(float)
                d['date'] = pd.to_datetime(d['date'])
                d = d[d['date'] > pd.Timestamp('1901-01-01')].sort_values('date')
                dv = d['date'].values
                fv = d['qfq_factor'].values
                for i in range(1, len(dv)):
                    m = fv[i - 1] / fv[i]
                    if not np.isfinite(m) or abs(m - 1.0) < 1e-9:
                        continue
                    ed = pd.Timestamp(dv[i])
                    if days is not None and len(days):
                        pos = bisect.bisect_left(days, np.datetime64(ed))
                        if pos <= 0 or pos >= len(days):
                            continue          # pos==0 ⇒ 塌陷伪影(见 docstring 陷阱①)
                        ed = pd.Timestamp(days[pos])
                    rows.append((code, ed, float(m)))
                ok = True
            break
        except Exception:
            time.sleep(2)
    if not ok:
        # 抓取失败: **不写缓存**, 让下次重试; 也不污染旧缓存
        return code, 'FAIL'
    tmp = f + '.tmp'
    with open(tmp, 'wb') as fh:
        pickle.dump(rows, fh)
    os.replace(tmp, f)
    return code, len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=OUT)
    ap.add_argument('--force', action='store_true', help='忽略缓存全量重拉')
    ap.add_argument('--workers', type=int, default=6)
    a = ap.parse_args()
    if a.force:
        global MAX_AGE
        MAX_AGE = 0.0
    os.makedirs(CACHE, exist_ok=True)

    print('排除的指数伪码(新浪按 sz000018 会抓到**同代码深市股票**的事件):')
    for c, nm in audit(RAW):
        print(f'    {c}  {nm}')
    print()

    codes = sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob(f'{RAW}/*/none.csv'))
    codes = [c for c in codes if c[0] in '036' and is_stock_code(c)]
    todo = [c for c in codes if a.force or not os.path.exists(os.path.join(CACHE, f'{c}.pkl'))
            or time.time() - os.path.getmtime(os.path.join(CACHE, f'{c}.pkl')) >= MAX_AGE]
    print(f'目标 {len(codes)} 只, 待拉 {len(todo)}', flush=True)
    t0 = time.time()
    nfail = 0
    with Pool(a.workers) as p:
        for i, (c, n) in enumerate(p.imap_unordered(work, todo, chunksize=5)):
            if n == 'FAIL':
                nfail += 1
            if i % 300 == 0:
                el = time.time() - t0
                print(f'  ...{i}/{len(todo)}  {el:.0f}s  ({i / max(el, 1):.1f}/s)', flush=True)

    rows = []
    for c in codes:
        f = os.path.join(CACHE, f'{c}.pkl')
        if os.path.exists(f):
            with open(f, 'rb') as fh:
                rows.extend(pickle.load(fh))
    # 拒绝写坏表(同 fetch_actions_table.py): 事件缺失会变成复权序列里的**假跌幅**且永久留存
    if nfail > max(10, 0.02 * max(len(todo), 1)):
        raise SystemExit(
            f'\n[拒绝落盘] 抓取失败 {nfail}/{len(todo)} 只(**未写缓存**)。\n'
            f'  事件缺失会变成复权序列里的假跌幅且**永久留存** ⇒ 保留旧 {a.out} 不动。\n'
            f'  检查网络后重跑(已成功的部分有缓存, 会自动跳过)。')
    a2 = pd.DataFrame(rows, columns=['code', 'ex_date', 'mult'])
    a2['code'] = a2['code'].astype(str).str.zfill(6)
    a2 = a2.sort_values(['code', 'ex_date']).reset_index(drop=True)
    tmp = a.out + '.tmp'
    a2.to_pickle(tmp)
    os.replace(tmp, a.out)
    print(f'\n=== 落盘 {a.out} ===')
    print(f'  事件 {len(a2)}  涉及 {a2.code.nunique()} 只')
    print(f'  乘数 >1.05 {(a2.mult > 1.05).sum()}   >1.20 {(a2.mult > 1.20).sum()}')
    print(f'  日期范围 {a2.ex_date.min().date()} ~ {a2.ex_date.max().date()}')


if __name__ == '__main__':
    main()
