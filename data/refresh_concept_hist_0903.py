#!/usr/bin/env python3
"""2026-09-03 补全 concept_hist.pkl (60个EM概念板块, 停更于2026-05-29 -> 补到最新)

push2.eastmoney.com 名字映射接口不稳, 绕开:
- searchapi.eastmoney.com 按板块名拿 BKxxxx 代码 (稳定)
- 91.push2his.eastmoney.com 直连K线 (带重试)
口径校验: 用 5/28-29 重叠行对比旧 pkl, return=涨跌幅/100 若不符则退回 close.pct_change()
只追加日期 > 旧末日的行, 历史数据不动。
"""
import os, pickle, shutil, time
import requests
import pandas as pd

HIST = '/mnt/d/quant/data/concept_hist.pkl'
UA = {'User-Agent': 'Mozilla/5.0'}


KLINE_HOSTS = [
    'https://push2his.eastmoney.com',
    'https://91.push2his.eastmoney.com',
    'https://92.push2his.eastmoney.com',
    'https://34.push2his.eastmoney.com',
]


def get(url, params=None, tries=12, base=5.0, rotate=False):
    """rotate=True: 东财K线集群限流时通时断, 4主机轮换+耐心退避。
    rotate=False: 原url直连(其他主机如searchapi/push2delay不允许被轮换)"""
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
        time.sleep(min(base * (1.4 ** i), 45))
    raise RuntimeError(f'{url} -> {last}')


BOARD_CODE = None
BOARD_PCT = None  # 板块名 -> 当日涨跌幅%(push2delay实时f3, 收盘后=最终值; push2his兜底用)


def board_code(name):
    """板块名 -> BKxxxx: 首选 push2delay clist 全量(与map同源), searchapi 备用"""
    global BOARD_CODE, BOARD_PCT
    if BOARD_CODE is None:
        BOARD_CODE = {}
        BOARD_PCT = {}
        total = None
        for pn in range(1, 10):
            r = get('https://push2delay.eastmoney.com/api/qt/clist/get', {
                'pn': str(pn), 'pz': '100', 'po': '1', 'np': '1',
                'fltt': '2', 'invt': '2', 'fid': 'f12',
                'fs': 'm:90+t:3+f:!50', 'fields': 'f12,f14,f3',
            })
            d = r.json().get('data') or {}
            total = total or int(d.get('total') or 0)
            for it in d.get('diff') or []:
                nm = str(it['f14'])
                BOARD_CODE[nm] = str(it['f12'])
                try:
                    BOARD_PCT[nm] = float(it['f3']) if it.get('f3') not in (None, '-', '') else None
                except (TypeError, ValueError):
                    BOARD_PCT[nm] = None
            if len(BOARD_CODE) >= total:
                break
            time.sleep(0.2)
        print(f'[board_code] clist加载 {len(BOARD_CODE)} 个板块名, 当日涨跌幅 {sum(1 for v in BOARD_PCT.values() if v is not None)} 个', flush=True)
    if name in BOARD_CODE:
        return BOARD_CODE[name]
    try:
        r = get('https://searchapi.eastmoney.com/api/suggest/get',
                {'input': name, 'type': '14', 'count': '10'})
        data = r.json().get('QuotationCodeTable', {}).get('Data') or []
        for it in data:
            if it.get('Name') == name and str(it.get('Classify', '')) == 'BK':
                return it['Code']
    except Exception:
        pass
    return None


def fetch_hist(code, beg, end):
    r = get('https://push2his.eastmoney.com/api/qt/stock/kline/get', {
        'secid': f'90.{code}',
        'fields1': 'f1,f2,f3,f4,f5,f6',
        'fields2': 'f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61',
        'klt': '101', 'fqt': '0', 'beg': beg, 'end': end,
        'smplmt': '10000', 'lmt': '1000000',
    }, rotate=True)
    j = r.json()
    lines = j.get('data', {}).get('klines') or []
    if not lines:
        return None
    cols = ['日期', '开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌幅', '涨跌额', '换手率']
    df = pd.DataFrame([l.split(',') for l in lines], columns=cols)
    df['date'] = pd.to_datetime(df['日期'])
    df['收盘'] = df['收盘'].astype(float)
    df['return'] = df['涨跌幅'].astype(float) / 100.0
    # 口径校验: 若与收盘 pct_change 相差过大, 用 pct_change
    pct = df['收盘'].pct_change()
    if ((df['return'] - pct).abs().median() > 0.005):
        print(f'    [口径] {code} 涨跌幅与收盘pct不符, 改用 close.pct_change', flush=True)
        df['return'] = pct
    return df[['date', 'return']]


print(f'=== concept_hist 补全 {time.strftime("%H:%M")} ===', flush=True)
with open(HIST, 'rb') as f:
    hist = pickle.load(f)
names = list(hist.keys())
print(f'旧 pkl: {len(names)} 板块', flush=True)

# 口径验证: 用第一块的重叠行对比 (失败不中断, 默认信任涨跌幅/100)
# 2026-09-04: 探测同时兼作 push2his 连通性测试 — 失败即全跑 clist 兜底(熔断)
USE_PCT = False
HIS_DOWN = False
try:
    probe = names[0]
    code = board_code(probe)
    print(f'[探测] {probe} -> {code}', flush=True)
    new0 = fetch_hist(code, '20260525', '20260605')
    old0 = hist[probe]
    ov = old0.merge(new0, on='date', suffixes=('_old', '_new'))
    if len(ov):
        diff = (ov['return_old'] - ov['return_new']).abs()
        print(f'[口径] {probe} 重叠 {len(ov)} 行, return 最大差 {diff.max():.6f}', flush=True)
        USE_PCT = diff.max() > 0.005
        if USE_PCT:
            print('[口径] 不符! 全量改用 close.pct_change', flush=True)
    else:
        print('[口径] 无重叠行, 信任 涨跌幅/100', flush=True)
except Exception as e:
    HIS_DOWN = True
    print(f'[口径] 探测失败({str(e)[:60]}) -> push2his不可用, 全量clist兜底', flush=True)

shutil.copy2(HIST, HIST + '.preFill_0903')
print(f'备份: {HIST}.preFill_0903', flush=True)

# 2026-09-04: 目标日期动态化(原硬编码9/1阈值+9/3抓取终点 → 每日跑永远补不到当天bar)
today = pd.Timestamp.now().normalize()
print(f'目标: 补到 {today.date()} (收盘后约1-2h东财发布当日板块bar)', flush=True)

def try_clist_append(name, old_df):
    """push2his不可用时: 用push2delay实时f3补当日bar (收盘后f3=当日板块涨跌幅终值).
    返回True=已补行"""
    pct = BOARD_PCT.get(name) if BOARD_PCT is not None else None
    if pct is None or (isinstance(pct, float) and pd.isna(pct)):
        return False
    hist[name] = pd.concat([old_df, pd.DataFrame({'date': [today], 'return': [pct / 100.0]})],
                           ignore_index=True)
    return True


ok, skip, fail = 0, 0, 0
clist_n = 0
his_fail_streak = 0
for i, name in enumerate(names):
    old_df = hist[name]
    last_d = old_df['date'].max()
    if last_d >= today:
        print(f'  [{i+1}/{len(names)}] {name}: 已补齐({last_d.date()}), 跳过', flush=True)
        time.sleep(0.5)
        continue
    if HIS_DOWN:
        # push2his熔断: 直接clist兜底
        if try_clist_append(name, old_df):
            print(f'  [{i+1}/{len(names)}] {name}: clist兜底 +{BOARD_PCT.get(name)}% (push2his熔断)', flush=True)
            ok += 1
            clist_n += 1
        else:
            print(f'  [{i+1}/{len(names)}] {name}: clist无当日涨跌幅, 跳过', flush=True)
            skip += 1
        time.sleep(0.3)
        continue
    try:
        code = board_code(name)
        if code is None:
            print(f'  [{i+1}/{len(names)}] {name}: 搜不到代码, 保留旧数据', flush=True)
            skip += 1
            time.sleep(1.5)
            continue
        new_df = fetch_hist(code, (last_d - pd.Timedelta(days=3)).strftime('%Y%m%d'),
                            today.strftime('%Y%m%d'))
        if new_df is None or len(new_df) == 0:
            if try_clist_append(name, old_df):
                print(f'  [{i+1}/{len(names)}] {name} ({code}): kline空, clist兜底 +{BOARD_PCT.get(name)}%', flush=True)
                ok += 1
                clist_n += 1
            else:
                print(f'  [{i+1}/{len(names)}] {name} ({code}): 无新数据', flush=True)
                skip += 1
            time.sleep(1.5)
            continue
        if USE_PCT:
            new_df['return'] = new_df['return']  # fetch_hist 已按口径调整
        tail = new_df[new_df['date'] > last_d]
        if len(tail):
            hist[name] = pd.concat([old_df, tail[['date', 'return']]], ignore_index=True)
            print(f'  [{i+1}/{len(names)}] {name} ({code}): +{len(tail)}行 -> {tail["date"].max().date()}')
            ok += 1
            his_fail_streak = 0
        else:
            if try_clist_append(name, old_df):
                print(f'  [{i+1}/{len(names)}] {name} ({code}): 无更晚bar, clist兜底 +{BOARD_PCT.get(name)}%', flush=True)
                ok += 1
                clist_n += 1
            else:
                print(f'  [{i+1}/{len(names)}] {name} ({code}): 无更晚日期 (最新 {new_df["date"].max().date()})')
                skip += 1
    except Exception as e:
        his_fail_streak += 1
        if his_fail_streak >= 2:
            HIS_DOWN = True
            print(f'  [{i+1}/{len(names)}] {name}: push2his连续失败, 熔断切clist兜底', flush=True)
        if try_clist_append(name, old_df):
            print(f'  [{i+1}/{len(names)}] {name}: push2his失败, clist兜底 +{BOARD_PCT.get(name)}%', flush=True)
            ok += 1
            clist_n += 1
        else:
            print(f'  [{i+1}/{len(names)}] {name}: FAIL {str(e)[:70]}', flush=True)
            fail += 1
    time.sleep(2.0)

with open(HIST, 'wb') as f:
    pickle.dump(hist, f)
dates_all = sorted({d.date() for df in hist.values() for d in df['date']})
print(f'=== 完成: 更新{ok} 跳过{skip} 失败{fail} (clist兜底{clist_n}); 全局日期 {dates_all[0]} -> {dates_all[-1]} ({len(dates_all)}个交易日) ===', flush=True)
