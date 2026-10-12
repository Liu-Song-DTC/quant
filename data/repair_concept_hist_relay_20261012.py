#!/usr/bin/env python3
"""2026-10-12 概念历史修复 (中继版, 一次性维护脚本)

背景: push2his/push2delay 直连被远端切断(与概念映射同病), 0903直连版步骤颗粒无收。
损伤盘点 (gate + 逐板块探针):
  - 09-18~09-29 整段8个交易日缺失 (9/14下载器事故+实盘暂停期欠账)
  - 假周日行 ×2: 09-13 / 10-11 (clist兜底把周末冻结f3当当日bar写入)
  - 55板块缺真实10-09 bar, 60板块全部缺10-12
策略: 60板块全窗口重拉(2023-01-01起, 经akshare_proxy_patch中继), 重叠行口径校验:
  - max_diff<=0.005: 全量替换 (过去内容恒等 + 尾部修复)
  - 否则: 保留旧<2026-09-01行 + 追加新>=2026-09-01行 (告警)
校验: 与3只流动股交易历并集对齐(无非交易日行), 除死板块外末日=今天。
备份: concept_hist.pkl.preRepair_20261012
"""
import pickle
import shutil
import time

import pandas as pd
import requests
import yaml

import akshare_proxy_patch

cfg = yaml.safe_load(open('/mnt/d/quant/trade/config.yaml', encoding='utf-8'))['proxy']
akshare_proxy_patch.install_patch(cfg['host'], auth_token=cfg['auth_token'], retry=8,
    hook_domains=['push2his.eastmoney.com', 'push2delay.eastmoney.com', 'searchapi.eastmoney.com'])

HIST = '/mnt/d/quant/data/concept_hist.pkl'
UA = {'User-Agent': 'Mozilla/5.0'}
FROZEN = {'2025年报预增'}  # 东财已下架, gate白名单


def get(url, params=None, tries=6, base=3.0):
    last = None
    for i in range(tries):
        try:
            r = requests.get(url, params=params, headers=UA, timeout=20)
            if r.status_code == 200:
                return r
            last = f'HTTP {r.status_code}'
        except Exception as e:
            last = str(e)[:70]
        time.sleep(min(base * (1.5 ** i), 30))
    raise RuntimeError(f'{url} -> {last}')


BOARD_CODE = None


def board_code(name):
    global BOARD_CODE
    if BOARD_CODE is None:
        BOARD_CODE = {}
        for pn in range(1, 12):
            r = get('https://push2delay.eastmoney.com/api/qt/clist/get', {
                'pn': str(pn), 'pz': '100', 'po': '1', 'np': '1',
                'fltt': '2', 'invt': '2', 'fid': 'f12',
                'fs': 'm:90+t:3+f:!50', 'fields': 'f12,f14',
            })
            d = r.json().get('data') or {}
            for it in d.get('diff') or []:
                BOARD_CODE[str(it['f14'])] = str(it['f12'])
            if len(BOARD_CODE) >= int(d.get('total') or 0):
                break
            time.sleep(0.2)
        print(f'[board_code] clist {len(BOARD_CODE)} 板块', flush=True)
    return BOARD_CODE.get(name)


def fetch_hist(code, beg='20230101', end='20261012'):
    """与0903版fetch_hist同口径: 涨跌幅/100, 与收盘pct偏差大时退回pct_change"""
    r = get('https://push2his.eastmoney.com/api/qt/stock/kline/get', {
        'secid': f'90.{code}',
        'fields1': 'f1,f2,f3,f4,f5,f6',
        'fields2': 'f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61',
        'klt': '101', 'fqt': '0', 'beg': beg, 'end': end,
        'smplmt': '10000', 'lmt': '1000000'})
    lines = (r.json().get('data') or {}).get('klines') or []
    if not lines:
        return None
    cols = ['日期', '开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌幅', '涨跌额', '换手率']
    df = pd.DataFrame([l.split(',') for l in lines], columns=cols)
    df['date'] = pd.to_datetime(df['日期'])
    df['收盘'] = df['收盘'].astype(float)
    df['return'] = df['涨跌幅'].astype(float) / 100.0
    pct = df['收盘'].pct_change()
    if ((df['return'] - pct).abs().median() > 0.005):
        print(f'    [口径] {code} 涨跌幅与收盘pct不符, 改用 close.pct_change', flush=True)
        df['return'] = pct
    return df[['date', 'return']].drop_duplicates('date').reset_index(drop=True)


# 交易历基准: 3只流动股日期并集 (停牌互备)
cal = set()
for code in ('000001', '600519', '300750'):
    cal |= set(pd.to_datetime(pd.read_csv(
        f'/mnt/d/quant/data/stock_data/backtrader_data/{code}_qfq.csv',
        usecols=['datetime'])['datetime']))
print(f'交易历基准: {len(cal)} 天 (000001+600519+300750并集)', flush=True)

with open(HIST, 'rb') as f:
    hist = pickle.load(f)
shutil.copy2(HIST, HIST + '.preRepair_20261012')
print('备份: concept_hist.pkl.preRepair_20261012', flush=True)

today = pd.Timestamp.now().normalize()
stats = {'full': 0, 'hybrid': 0, 'kept': 0, 'fail': 0}
holes_filled_total = 0
for i, name in enumerate(list(hist.keys())):
    old = hist[name]
    old_fake = int((~old['date'].isin(cal)).sum())
    try:
        code = board_code(name)
        if code is None:
            print(f'  [{i+1}/60] {name}: 无BK码, 保留旧数据', flush=True)
            stats['kept'] += 1
            time.sleep(1.0)
            continue
        new = fetch_hist(code)
        if new is None or len(new) == 0:
            print(f'  [{i+1}/60] {name}: kline空, 保留旧数据', flush=True)
            stats['kept'] += 1
            time.sleep(1.0)
            continue
        ov = old.merge(new, on='date', suffixes=('_o', '_n'))
        diff = (ov['return_o'] - ov['return_n']).abs().max() if len(ov) else float('nan')
        if len(ov) and diff <= 0.005:
            hist[name] = new
            tag = '全量替换'
            stats['full'] += 1
        else:
            cut = pd.Timestamp('2026-09-01')
            head = old[old['date'] < cut]
            tail = new[new['date'] >= cut]
            hist[name] = (pd.concat([head, tail], ignore_index=True)
                          .drop_duplicates('date', keep='last')
                          .sort_values('date').reset_index(drop=True))
            tag = f'混合(重叠diff={diff:.4f})'
            stats['hybrid'] += 1
        fixed = hist[name]
        holes_filled = int(len(fixed) - len(old) + old_fake)  # 净增行 + 去掉的假行
        holes_filled_total += max(holes_filled, 0)
        holes_in_window = len([d for d in cal if pd.Timestamp('2023-01-01') <= d <= fixed['date'].max()]) - len(fixed)
        print(f'  [{i+1}/60] {name} ({code}): {tag}, {len(old)}->{len(fixed)}行, '
              f'末日{fixed["date"].max().date()}, 旧假行{old_fake}, 修复后缺口{holes_in_window}', flush=True)
    except Exception as e:
        print(f'  [{i+1}/60] {name}: FAIL {str(e)[:70]}', flush=True)
        stats['fail'] += 1
    time.sleep(1.2)

with open(HIST, 'wb') as f:
    pickle.dump(hist, f)

bad = []
for name, df in hist.items():
    dmax = df['date'].max()
    nontrading = int((~df['date'].isin(cal)).sum())
    if (name not in FROZEN and dmax < today) or nontrading:
        bad.append((name, str(dmax.date()), nontrading))
print(f'=== 修复完成: {stats}; 净修复行约{holes_filled_total}; 终检异常: '
      f'{bad if bad else "无"} ===', flush=True)
