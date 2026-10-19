#!/usr/bin/env python
"""拉取全市场**官方公司行动表**并冻结到本地 (2026-10-16)

背景: 复权重建的"从 2 位小数价格反推事件"路线在**送转日**系统性错
      (002706 10送10 估成 2.30 而非 2.00, 中位误差 1.232% vs 东财加性 0.092%)。
      ⇒ 改为**表驱动**: 直接读 `ak.stock_fhps_detail_em` 的
         `除权除息日 / 现金分红-现金分红比例(每10股) / 送转股份-送转总比例(每10股)`。

产出: data/alternative_data/corporate_actions.pkl
      DataFrame: code, ex_date, s(每股送转), D(每股现金分红, 税前)

⚠ 本表**不是完备事件源**(结构性不覆盖配股/重整转增, 且漏"特别股息") ⇒
  必须与 `fetch_sina_factors.py` 融合成 `events_table.pkl` 后再喂给重建。
  本脚本只负责把**东财这一路**冻结下来。

缓存 `data/alternative_data/.ca_parts/{code}.pkl` —— **按天失效**(默认 1 天)。
  旧版"存在即跳过"有两个真 bug(2026-10-17 修):
   ① 永不失效 ⇒ **新事件永远发现不了**;
   ② 抓取异常也落盘 `None` ⇒ 一次网络抖动 = 该股票**永久**标记"无事件", 静默数据丢失。
  现在: 异常**不写缓存**(下次重试), 写盘一律 tmp+os.replace(避免中途被杀留截断文件)。
"""
import os
import sys
import json
import time
import glob
import pickle
import argparse
import urllib.request
import numpy as np
from multiprocessing import Pool
import pandas as pd
import akshare as ak

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from market_codes import is_stock_code, audit          # noqa: E402

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = f'{PROJ}/data/stock_data/raw_data'
OUT = f'{PROJ}/data/alternative_data/corporate_actions.pkl'
PART = f'{PROJ}/data/alternative_data/.ca_parts'
MAX_AGE = 86400.0
RETRY = 3              # 抓取重试次数(滤瞬时抖动)

# 东财"分红送配详情"原始端点 —— 仅用于**判定**"无数据"还是"接口失败"。
_FHPS_URL = ('https://datacenter-web.eastmoney.com/api/data/v1/get'
             '?reportName=RPT_SHAREBONUS_DET&columns=ALL'
             '&filter=(SECURITY_CODE%3D%22{code}%22)'
             '&pageNumber=1&pageSize=1&sortColumns=REPORT_DATE&sortTypes=-1')
_EMPTY_CODE = 9201     # 东财自己的"返回数据为空"答复码


def _dump(f, obj):
    tmp = f + '.tmp'
    with open(tmp, 'wb') as fh:
        pickle.dump(obj, fh)
    os.replace(tmp, f)


def _api_says_empty(code, timeout=15):
    """东财**明确答复**"该码无分红送配记录" ⇒ True; 其余(含无法判定) ⇒ False。

    akshare 的 `stock_fhps_detail_em` 拿到 `{"result":null,...}` 后直接解引用 None
    抛 `TypeError: 'NoneType' object is not subscriptable`, 把**结构性无数据**与
    **接口失败**混成同一种异常。实测两者的区别在东财答复里是显式的:

        无数据: {"result":null,"success":false,"message":"返回数据为空","code":9201}
        有数据: {"result":{"pages":6,"data":[...]},"success":true,...}

    故**不用价格/停牌时长之类的外部启发式猜测**(那会把"新近退市但东财仍有页"
    误判成无数据), 直接读接口自己的答复码 —— 判据来自数据源本身, 不是推断。
    网络异常/超时/答复格式变了 ⇒ 一律 False(退回 'FAIL', 不写缓存)。
    """
    try:
        req = urllib.request.Request(_FHPS_URL.format(code=code),
                                     headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            j = json.loads(r.read().decode('utf-8', 'replace'))
        return j.get('code') == _EMPTY_CODE and not j.get('result')
    except Exception:
        return False


def fetch_one(code):
    """返回 'cached' | 'nodata' | 事件数 | 'FAIL'。真异常**不写缓存**。

    ⚠ 2026-10-19 修 **死锁**: 大量码在东财"分红送配详情"里**根本没有记录**
      (退市股页面已撤 + **从没分过红的次新股**), 东财答复
      `{"result":null,"success":false,"message":"返回数据为空","code":9201}`,
      akshare 解引用 None 抛 `TypeError` ⇒ 旧实现一律记 'FAIL' 且**异常不写缓存**
      ⇒ 这些码永远无缓存 ⇒ 落盘检查的 `n_missing > 0` **永久成立** ⇒
      `[拒绝落盘]` 变成**死锁**, 整条 `refresh_all.sh` 在第 2/9 步再也跑不完
      (实测 239 只待拉里 124 只属此类, 另 115 只同样抓不到)。

    判别式 = **东财自己的答复码**(`_api_says_empty`), 不是外部启发式:
      重试 RETRY 次仍失败 → 直连原始端点 → 仅当 `code==9201 且 result 为空`
      才记 'nodata'(缓存 `[]`), 其余一律 'FAIL'(不写缓存, 下次重试)。
      ⚠ 退市股那部分无碍: 它们的复权事件由**融合表的新浪 mult 通道**提供
        (实测 115 只中 114 只共 1,043 条**全部**来自 sina) ⇒ 本表写 `[]`
        只是如实记录"东财这一路没有它", 不造成事件缺失。
      ⚠ **自愈**: 缓存 24h 失效 ⇒ 该股若建页(新上市/补披露), 次日重拉即得真数据。
    """
    f = os.path.join(PART, f'{code}.pkl')
    if os.path.exists(f) and time.time() - os.path.getmtime(f) < MAX_AGE:
        return 'cached'
    d = None
    for attempt in range(RETRY):
        try:
            d = ak.stock_fhps_detail_em(symbol=code)
        except Exception:
            d = None
        if d is not None:
            break
        time.sleep(0.4 * (attempt + 1))
    if d is None:
        if _api_says_empty(code):
            _dump(f, [])
            return 'nodata'                # 东财明确答复"无此数据", 不是抓取失败
        return 'FAIL'                      # 真接口失败 ⇒ 不写缓存, 下次重试
    if not len(d):
        _dump(f, [])
        return 0
    d = d[d['除权除息日'].notna()].copy()
    rows = []
    for _, r in d.iterrows():
        try:
            s = float(r['送转股份-送转总比例'] or 0) / 10.0
        except Exception:
            s = 0.0
        try:
            D = float(r['现金分红-现金分红比例'] or 0) / 10.0
        except Exception:
            D = 0.0
        s = s if np.isfinite(s) else 0.0
        D = D if np.isfinite(D) else 0.0
        rows.append((code, pd.Timestamp(r['除权除息日']), s, D))
    _dump(f, rows)
    return len(rows)


def _work(c):
    r = fetch_one(c)
    time.sleep(0.05)
    return c, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=OUT)
    ap.add_argument('--force', action='store_true', help='忽略缓存全量重拉')
    ap.add_argument('--workers', type=int, default=6)
    a = ap.parse_args()
    if a.force:
        global MAX_AGE
        MAX_AGE = 0.0
    os.makedirs(PART, exist_ok=True)

    print('排除的指数伪码:')
    for c, nm in audit(RAW):
        print(f'    {c}  {nm}')
    print()

    codes = sorted(os.path.basename(os.path.dirname(p))
                   for p in glob.glob(f'{RAW}/*/none.csv'))
    codes = [c for c in codes if c[0] in '036' and is_stock_code(c)]   # 北交所排除(铁律) + 伪码排除
    todo = [c for c in codes if a.force
            or not os.path.exists(os.path.join(PART, f'{c}.pkl'))
            or time.time() - os.path.getmtime(os.path.join(PART, f'{c}.pkl')) >= MAX_AGE]
    print(f"目标 {len(codes)} 只, 待拉 {len(todo)}", flush=True)
    t0 = time.time()
    nfail = 0
    nodata = []
    with Pool(a.workers) as p:
        for i, (c, r) in enumerate(p.imap_unordered(_work, todo, chunksize=5)):
            if r == 'FAIL':
                nfail += 1
            elif r == 'nodata':
                nodata.append(c)
            if i % 500 == 0:
                el = time.time() - t0
                print(f"  ...{i}/{len(todo)}  {el:.0f}s  ({i / max(el, 1):.1f}/s)", flush=True)
    if nodata:
        # 大声列出来 —— 未知状态既不丢也不静默(与换名守卫同一原则)。
        print(f"\n[结构性无数据] {len(nodata)} 只在东财'分红送配详情'里无记录 "
              f"(答复码 {_EMPTY_CODE} '返回数据为空') ⇒ 本表如实记 [] :")
        print(f"  {' '.join(sorted(nodata))}")
        print(f"  ⚠ 其中退市股的复权事件由融合表的新浪 mult 通道提供, 不受影响;")
        print(f"    其余(未分红次新股等)本就无事件。")

    rows = []
    n_missing = 0
    for c in codes:
        f = os.path.join(PART, f'{c}.pkl')
        if not os.path.exists(f):
            n_missing += 1
            continue
        with open(f, 'rb') as fh:
            r = pickle.load(fh)
        if r:
            rows.extend(r)
    # 拒绝写坏表: 事件缺失会变成历史里的**假跌幅**, 而且永久留在复权序列里。
    # 宁可不覆盖旧表(下次重试), 也不写一张有洞的表。
    if nfail > max(10, 0.02 * max(len(todo), 1)) or n_missing > 0:
        raise SystemExit(
            f"\n[拒绝落盘] 抓取失败 {nfail}/{len(todo)} 只, 无缓存 {n_missing} 只。\n"
            f"  事件缺失会变成复权序列里的假跌幅且**永久留存** ⇒ 保留旧 {a.out} 不动。\n"
            f"  检查网络/接口后重跑(已成功的部分有缓存, 会自动跳过)。")
    a2 = pd.DataFrame(rows, columns=['code', 'ex_date', 's', 'D'])
    a2['code'] = a2['code'].astype(str).str.zfill(6)
    a2 = a2.sort_values(['code', 'ex_date']).reset_index(drop=True)
    tmp = a.out + '.tmp'
    a2.to_pickle(tmp)
    os.replace(tmp, a.out)
    print(f"\n=== 落盘 {a.out} ===")
    print(f"  事件 {len(a2)}  涉及 {a2.code.nunique()} 只")
    print(f"  有送转(s>0) {(a2.s > 0).sum()}   纯分红 {(a2.s == 0).sum()}")
    print(f"  除权日范围 {a2.ex_date.min().date()} ~ {a2.ex_date.max().date()}")
    print(f"  覆盖率 {a2.code.nunique() / len(codes):.1%} ({a2.code.nunique()}/{len(codes)})")


if __name__ == '__main__':
    main()
