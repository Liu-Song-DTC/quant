#!/usr/bin/env python3
"""2026-09-11 阶段5g: 概念前视上界探针 (静态概念映射×回测窗口)

发现: stock_concept_map.pkl 是 2026-09-10 重建的静态快照; 季度配置(2021Q1等)
含 AI智能体/6G概念/2025三季报预增 等未来概念键(有效IC条目) — 季度标定与信号
生成共用该静态映射, 概念归属非PIT。

本探针: 概念指数最早行情日≈概念出现时间(上界估计 — 指数发布晚于概念创建,
故前视占比被高估; 上界小则可排除风险, 上界大需精确数据复核)。

步骤: (1) 下载全部概念指数K线最早日期 → concept_first_date.pkl 缓存
       (2) 信号CSV(基线) industry×date → 前视行占比, 按年+设计期/持有期分组
"""
import os
import pickle
import time
import numpy as np
import pandas as pd
import requests

UA = {'User-Agent': 'Mozilla/5.0'}
KLINE_HOSTS = [
    'https://push2his.eastmoney.com',
    'https://91.push2his.eastmoney.com',
    'https://92.push2his.eastmoney.com',
    'https://34.push2his.eastmoney.com',
]
CACHE = '/mnt/d/quant/strategy/rolling_validation_results/concept_first_date.pkl'
MAP = '/mnt/d/quant/data/stock_concept_map.pkl'
SIG = '/mnt/d/quant/strategy/rolling_validation_results/backtest_signals.csv'

STYLE_KW = ['融资融券', '深股通', '沪股通', '富时罗素', '标准普尔', 'MSCI',
            '创业板综', '机构重仓', 'QFII', '破增发', '破发股', '昨日高',
            '中证500', '深成500', '中盘股', '小盘股', '央国企改革',
            '西部大开发', '年报预增', '专精特新', '上证380', 'HS300',
            '微盘股', '百元股', '大盘股', '小盘成长', '小盘价值',
            '转债标的', '长江三角', '深圳特区', '破净股', '创投']


def get(url, params=None, tries=10, base=4.0, rotate=False):
    last = None
    for i in range(tries):
        u = url
        if rotate:
            path = url.split('://', 1)[1].split('/', 1)[1]
            u = KLINE_HOSTS[i % len(KLINE_HOSTS)] + '/' + path
        try:
            r = requests.get(u, params=params, headers=UA, timeout=15)
            if r.status_code == 200:
                return r
            last = f'HTTP {r.status_code}'
        except Exception as e:
            last = str(e)[:70]
        time.sleep(min(base * (1.35 ** i), 40))
    raise RuntimeError(f'{url} -> {last}')


def collect_first_dates():
    """全部概念 → 概念指数最早行情日"""
    with open(MAP, 'rb') as f:
        raw = pickle.load(f)
    concepts = set()
    for code, cs in raw.items():
        concepts.update(c for c in cs if not any(kw in c for kw in STYLE_KW))
    print(f'概念映射: {len(concepts)} 个概念(过滤宽泛标签后)', flush=True)

    name2code = {}
    total = None
    for pn in range(1, 15):
        r = get('https://push2delay.eastmoney.com/api/qt/clist/get', {
            'pn': str(pn), 'pz': '100', 'po': '1', 'np': '1',
            'fltt': '2', 'invt': '2', 'fid': 'f12',
            'fs': 'm:90+t:3+f:!50', 'fields': 'f12,f14',
        })
        d = r.json().get('data') or {}
        total = total or int(d.get('total') or 0)
        for it in d.get('diff') or []:
            name2code[str(it['f14'])] = str(it['f12'])
        if len(name2code) >= total:
            break
        time.sleep(0.2)
    print(f'clist板块代码: {len(name2code)} 个', flush=True)

    first = {}
    miss = 0
    consec_fail = 0
    for i, name in enumerate(sorted(concepts)):
        code = name2code.get(name)
        if code is None:
            miss += 1
            continue
        try:
            r = get('https://push2his.eastmoney.com/api/qt/stock/kline/get', {
                'secid': f'90.{code}',
                'fields1': 'f1,f2,f3,f4,f5,f6',
                'fields2': 'f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61',
                'klt': '101', 'fqt': '0', 'beg': '20150101', 'end': '20260911',
                'smplmt': '10000', 'lmt': '1000000',
            }, rotate=True, tries=3, base=2.0)
            lines = (r.json().get('data') or {}).get('klines') or []
            if lines:
                first[name] = lines[0].split(',')[0]
                consec_fail = 0
            else:
                consec_fail += 1
        except Exception:
            consec_fail += 1
        if consec_fail >= 8:
            print(f'  连续失败{consec_fail}次, 熔断于 {i+1}/{len(concepts)}', flush=True)
            break
        if (i + 1) % 25 == 0:
            print(f'  进度 {i+1}/{len(concepts)} 命中 {len(first)} 缺失 {miss}', flush=True)
        time.sleep(0.12)
    print(f'完成: 命中 {len(first)}/{len(concepts)} 缺失 {miss}', flush=True)
    with open(CACHE, 'wb') as f:
        pickle.dump(first, f)
    print(f'缓存写入 {CACHE}', flush=True)
    return first


def measure(sig_path=SIG):
    """信号CSV industry×date → 前视行占比 (双法: 概念指数最早日 + 硬编码2023+概念)"""
    sig = pd.read_csv(sig_path, usecols=['code', 'date', 'buy', 'industry'],
                      dtype={'code': str, 'industry': str})
    sig = sig[sig.buy == True].copy()
    sig['date'] = pd.to_datetime(sig['date'])
    sig['first'] = None

    # 法1: 概念指数最早行情日(push2his解封后可下载)
    if os.path.exists(CACHE):
        with open(CACHE, 'rb') as f:
            first = pickle.load(f)
        first_d = {k: pd.to_datetime(v) for k, v in first.items()}
        sig['first'] = sig['industry'].map(first_d)
        sig['lookahead1'] = sig['first'].notna() & (sig['date'] < sig['first'])
        print(f'\n[法1] 概念指数最早日覆盖 {sig.industry.isin(first_d).mean()*100:.1f}% 行业行', flush=True)

    # 法2: 硬编码2023+概念 (2021-2022年A股客观不存在)
    FUTURE = ['ChatGPT概念', 'CPO概念', 'AIGC概念', 'AI智能体', 'AI手机', 'AI眼镜',
              'AI制药（医疗）', 'AI医疗', '低空经济', '人形机器人', '数据要素',
              '液冷服务器', '液冷概念', '华为昇腾', '英伟达概念', 'Sora概念',
              '合成生物', '车路云', '量子科技', '卫星互联网', '商业航天',
              '可控核聚变', '深海科技', '固态电池', '钙钛矿电池', 'HBM概念',
              '玻璃基板', 'AI语料', 'AI应用', 'Kimi概念', '文生视频', '多模态AI']
    sig['fut'] = sig['industry'].isin(FUTURE)
    print(f'\n[法2] 硬编码2023+概念 前视行占比 按年:')
    for y in sorted(sig.date.dt.year.unique()):
        s = sig[sig.date.dt.year == y]
        if len(s) == 0:
            continue
        print(f'  {y}: n={len(s):6d} 前视 {s.fut.mean()*100:5.2f}% '
              f'({s.fut.sum()}行)')
    print(f'  设计期2021-2024: {sig[sig.date.dt.year<=2024].fut.mean()*100:.2f}%')
    print(f'  持有期2025+:     {sig[sig.date.dt.year>=2025].fut.mean()*100:.2f}%')
    fu = sig[sig.fut]
    if len(fu):
        print('\n[法2] 前视行涉及概念:')
        print(fu.industry.value_counts().to_string())
    if 'lookahead1' in sig:
        print('\n[法1] 前视行占比(信号日<概念指数最早日) 按年:')
        for y in sorted(sig.date.dt.year.unique()):
            s = sig[sig.date.dt.year == y]
            if len(s) == 0:
                continue
            print(f'  {y}: n={len(s):6d} 前视 {s.lookahead1.mean()*100:5.2f}%')
        print(f'  设计期2021-2024: {sig[sig.date.dt.year<=2024].lookahead1.mean()*100:.2f}%')
        print(f'  持有期2025+:     {sig[sig.date.dt.year>=2025].lookahead1.mean()*100:.2f}%')


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'download':
        collect_first_dates()
    elif len(sys.argv) > 1 and sys.argv[1] == 'measure':
        measure()
    else:
        print('用法: download(拉概念指数最早日期) | measure(信号前视占比, 需新基线信号)')
