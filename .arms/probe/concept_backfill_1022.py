#!/usr/bin/env python3
"""#366: 回补 concept_hist 2021-2022 的**可行性+口径校验**探针 (2026-10-22).

背景: 39 个概念全史数据都恰好起于 2023-01-03 ⇒ 原抓取 beg=20230101, 不是数据不可得.
本脚本**只读生产**, 写 .arms/ 下的候选文件:
  .arms/concept_bf_raw_1022.pkl   原始抓取 (概念 -> DataFrame[date, return])
  .arms/concept_hist_bf1022.pkl   候选完整 dict (prepend 2021-22, 现有行不动)
  .arms/concept_bf_report_1022.csv 每个概念的抓取/校验摘要

口径校验 (关键): 对同一概念重抓 2022-12-01..2023-01-20, 与现有 pkl 的 2023-01-03+
行做 merge, 要求 return 逐位相同 (容差 1e-9). 这验证:
  (a) 历史 K 线 return 语义与现有数据一致 (涨跌幅/100, 非复权口径);
  (b) 无供应商重述.
同时记录 2021 收益分布 (/首行守卫: 序列首日 return 置 0).

运行: cd /mnt/d/quant && .venv/bin/python -u .arms/probe/concept_backfill_1022.py
"""
import os
import pickle
import time

import numpy as np
import pandas as pd
import requests

HIST = '/mnt/d/quant/data/concept_hist.pkl'
RAW = '/mnt/d/quant/.arms/concept_bf_raw_1022.pkl'
OUT = '/mnt/d/quant/.arms/concept_hist_bf1022.pkl'
REP = '/mnt/d/quant/.arms/concept_bf_report_1022.csv'

UA = {'User-Agent': 'Mozilla/5.0'}
HOSTS = ['https://push2his.eastmoney.com', 'https://91.push2his.eastmoney.com',
         'https://92.push2his.eastmoney.com', 'https://34.push2his.eastmoney.com']


def get(path, params, tries=5, base=2.0):
    """快速失败: 连接被拒时 5 次退避 (~40s) 就放弃, 交给下一轮重跑。
    上一版 10 次×30s 在单概念上卡 5-10min, 39 个概念不可接受."""
    last = None
    for i in range(tries):
        u = HOSTS[i % len(HOSTS)] + path
        try:
            r = requests.get(u, params=params, headers=UA, timeout=15)
            if r.status_code == 200:
                return r
            last = f'HTTP {r.status_code}'
        except Exception as e:
            last = str(e)[:80]
        time.sleep(min(base * (1.6 ** i), 10))
    raise RuntimeError(f'{path} -> {last}')


BKMAP = '/mnt/d/quant/.arms/concept_bkmap_1022.pkl'


def board_codes(names):
    """板块名 -> BKxxxx. 10-22: clist 集群当晚极不稳, 改用 searchapi 逐个查 + 落盘缓存.
    生产 refresh 用 clist 全量; 本探针只需 39 个名字, 逐个查更稳."""
    out = {}
    if os.path.exists(BKMAP):
        with open(BKMAP, 'rb') as f:
            out = pickle.load(f)
        print(f'复用板块代码缓存 {len(out)} 个', flush=True)
    for name in names:
        if name in out:
            continue
        try:
            r = requests.get('https://searchapi.eastmoney.com/api/suggest/get',
                             params={'input': name, 'type': '14', 'count': '10'},
                             headers=UA, timeout=15)
            data = (r.json().get('QuotationCodeTable') or {}).get('Data') or []
            for it in data:
                if it.get('Name') == name and str(it.get('Classify', '')) == 'BK':
                    out[name] = str(it['Code'])
                    break
        except Exception as e:
            print(f'  [board_code] {name}: {str(e)[:50]}', flush=True)
        time.sleep(0.15)
        if len(out) % 10 == 0:
            with open(BKMAP, 'wb') as f:
                pickle.dump(out, f)
    with open(BKMAP, 'wb') as f:
        pickle.dump(out, f)
    print(f'板块代码映射 {len(out)}/{len(names)} 个', flush=True)
    return out


def fetch(code, beg, end):
    r = get('/api/qt/stock/kline/get', {
        'secid': f'90.{code}',
        'fields1': 'f1,f2,f3,f4,f5,f6',
        'fields2': 'f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61',
        'klt': '101', 'fqt': '0', 'beg': beg, 'end': end,
        'smplmt': '10000', 'lmt': '1000000'})
    lines = (r.json().get('data') or {}).get('klines') or []
    if not lines:
        return None
    cols = ['日期', '开盘', '收盘', '最高', '最低', '成交量', '成交额',
            '振幅', '涨跌幅', '涨跌额', '换手率']
    df = pd.DataFrame([l.split(',') for l in lines], columns=cols)
    df['date'] = pd.to_datetime(df['日期'])
    df['close'] = df['收盘'].astype(float)
    df['return'] = df['涨跌幅'].astype(float) / 100.0
    pct = df['close'].pct_change()
    bad = (df['return'] - pct).abs()
    # 与生产 refresh 同判据 (median>0.005 才切换), 另记录逐行最大偏差用于报告
    switched = bool(bad.median() > 0.005)
    if switched:
        df['return'] = pct
    return df[['date', 'return']].assign(_bad=bad.fillna(0.0))


def install_relay():
    """东财直连当前被远端切断(与 10-12 同病) ⇒ 走生产同款中继代理."""
    import yaml
    import akshare_proxy_patch
    cfg = yaml.safe_load(open('/mnt/d/quant/trade/config.yaml', encoding='utf-8'))['proxy']
    akshare_proxy_patch.install_patch(
        cfg['host'], auth_token=cfg['auth_token'], retry=8,
        hook_domains=['push2his.eastmoney.com', 'push2delay.eastmoney.com',
                      'searchapi.eastmoney.com'])
    print(f'[relay] 已安装中继 {cfg["host"]}', flush=True)


def main():
    if not os.environ.get('QUANT_NO_RELAY'):
        install_relay()
    hist = pd.read_pickle(HIST)
    full = [k for k, v in hist.items()
            if len(v) >= 900 and pd.Timestamp(v['date'].min()) <= pd.Timestamp('2023-01-05')]
    print(f'候选全史概念 {len(full)} 个 (生产 {len(hist)} 概念)', flush=True)

    bmap = board_codes(full)

    if os.path.exists(RAW):
        raw = pd.read_pickle(RAW)
        print(f'复用已抓取 {len(raw)} 个概念 ← {RAW}', flush=True)
    else:
        raw = {}
    rows = []
    for i, name in enumerate(full):
        if name in raw:
            continue
        code = bmap.get(name)
        if code is None:
            print(f'  [{i+1}/{len(full)}] {name}: 无 BK 代码, 跳过', flush=True)
            rows.append(dict(concept=name, code='', n21=0, d0='', d1='', ok=False,
                             note='no_board_code'))
            continue
        try:
            df = fetch(code, '20210101', '20221231')
        except Exception as e:
            print(f'  [{i+1}/{len(full)}] {name}({code}): 抓取失败 {str(e)[:60]}', flush=True)
            rows.append(dict(concept=name, code=code, n21=0, d0='', d1='', ok=False,
                             note=f'fetch_fail:{str(e)[:40]}'))
            continue
        # 口径校验: 重抓重叠段 2022-12-01..2023-01-20 (失败不丢主数据, 只降级为未校验)
        try:
            ov = fetch(code, '20221201', '20230120')
        except Exception as e:
            ov = None
            print(f'  [{i+1}/{len(full)}] {name}({code}): 重叠段校验抓取失败 {str(e)[:40]}', flush=True)
        n = 0 if df is None else len(df)
        d0 = '' if df is None or n == 0 else str(df['date'].min().date())
        d1 = '' if df is None or n == 0 else str(df['date'].max().date())
        # 重叠校验
        maxdiff = np.nan
        ok = False
        note = ''
        if ov is not None and n:
            cur = hist[name]
            mm = cur.merge(ov[['date', 'return']], on='date', suffixes=('_old', '_new'))
            if len(mm):
                maxdiff = float((mm['return_old'] - mm['return_new']).abs().max())
                ok = maxdiff < 1e-9
            else:
                note = 'no_overlap'
        else:
            note = 'no_overlap_fetch'
        first_guard = False
        if df is not None and n:
            # 首行守卫 (#367): 只有"抓取首行 = 板块上市首日"时, 涨跌幅才是
            # "自基点累计涨幅"(无前收盘), 必须置 0。若首行正好是请求窗口的
            # 第一个交易日 2021-01-04, 说明板块在 2021 之前就存在 ⇒ 那是真实
            # 日收益, **不能** 置 0 (否则白丢一天)。
            df = df.copy()
            if str(df['date'].min().date()) > '2021-01-04':
                df.loc[df.index[0], 'return'] = 0.0
                first_guard = True
            raw[name] = df[['date', 'return']].reset_index(drop=True)
            # 增量落盘: 单概念卡住/被杀不丢已完成部分 (下一轮复用)
            with open(RAW + '.tmp', 'wb') as _f:
                pickle.dump(raw, _f)
            os.replace(RAW + '.tmp', RAW)
        rows.append(dict(concept=name, code=code, n21=n, d0=d0, d1=d1,
                         ok=ok, maxdiff=maxdiff, note=note, first_guard=first_guard,
                         ret_absmax=float(df['return'].abs().max()) if n else np.nan,
                         ret_p99=float(df['return'].abs().quantile(.99)) if n else np.nan))
        print(f'  [{i+1}/{len(full)}] {name}({code}): n={n} {d0}..{d1} '
              f'overlapΔ={maxdiff:.2e} ok={ok} '
              f'|r|max={rows[-1]["ret_absmax"]:.3f}', flush=True)
        time.sleep(0.4)

    pd.DataFrame(rows).to_csv(REP, index=False)
    with open(RAW, 'wb') as f:
        pickle.dump(raw, f)

    # 构建候选完整 dict: 只 prepend date < 现有 min, 现有行逐位不动
    cand = {}
    prep = {}
    for name, cur in hist.items():
        if name in raw:
            new = raw[name]
            cur_min = pd.Timestamp(cur['date'].min())
            add = new[new['date'] < cur_min]
            if len(add):
                cand[name] = pd.concat([add[['date', 'return']],
                                        cur[['date', 'return']]], ignore_index=True)
                prep[name] = len(add)
            else:
                cand[name] = cur
        else:
            cand[name] = cur
    with open(OUT, 'wb') as f:
        pickle.dump(cand, f)

    print('\n=== 汇总 ===')
    print(f'候选写入 {OUT}')
    print(f'prepend 概念数 {len(prep)}; 新增行 {sum(prep.values())}')
    dist = pd.Series([v for v in prep.values()])
    print(f'每概念新增行: min={dist.min()} median={dist.median()} max={dist.max()}')
    rr = pd.DataFrame(rows)
    print(f'重叠校验通过 {int(rr.ok.sum())}/{len(rr)}; '
          f'失败: {rr[~rr.ok][["concept","note","maxdiff"]].to_dict("records")[:5]}')
    print(f'起于 2021-01-04 的概念 {int((rr.d0=="2021-01-04").sum())}/{len(rr)}')
    print(f'报告 {REP}')


if __name__ == '__main__':
    main()
