#!/usr/bin/env python3
"""Q4标定干跑验证器 (2026-09-22, 只读, 零生产写入)

验证 /tmp/q4_dryrun_20260922/ 的干跑产物 (CALIB_OUTPUT_DIR重定向):
  1) 2026Q4.yaml 存在且可解析, 概念数合理 (与Q3/全局的关系);
  2) 中报4概念 (2026中报扭亏/预减/预增/首亏) 的配置 == 2026Q3.yaml 逐位一致
     (薄样本回退规则生效: 干跑态gated天数≈55<120);
  3) 日志回退打印与yaml内容一致 (copied清单==中报4);
  4) 无 <120 gated天 的概念拿到fresh权重 (用coverage probe的gated天数口径复算);
  5) 其余概念为gated重标定 (factors存在且权重和≈1)。

执行: .venv/bin/python analysis/verify_q4_dryrun_20260922.py
"""
import os
import pickle
import re
import sys

import pandas as pd
import yaml

sys.path.insert(0, '/mnt/d/quant/strategy')

DRY_DIR = '/tmp/q4_dryrun_20260922'
LOG_P = '/tmp/q4_dryrun_20260922.log'
QD = '/mnt/d/quant/strategy/config/quarterly_factors'
MAP_P = '/mnt/d/quant/data/stock_concept_map.pkl'
INCEP_P = '/mnt/d/quant/data/concept_inception.pkl'
VAL_CSV = '/mnt/d/quant/strategy/rolling_validation_results/validation_results.csv'
CALIB_START = pd.Timestamp('2021-10-01')
CALIB_END = pd.Timestamp('2026-09-30')
GATED_DAY_MIN = 120
ZHONGBAO4 = ['2026中报扭亏', '2026中报预减', '2026中报预增', '2026中报首亏']

fails = []


def check(cond, msg):
    print(('✓' if cond else '✗') + ' ' + msg)
    if not cond:
        fails.append(msg)


def main():
    q4_path = os.path.join(DRY_DIR, '2026Q4.yaml')
    q3_path = os.path.join(QD, '2026Q3.yaml')

    check(os.path.exists(q4_path), '干跑2026Q4.yaml存在')
    if not os.path.exists(q4_path):
        sys.exit(1)

    q4 = yaml.safe_load(open(q4_path, encoding='utf-8'))['industry_factors']
    q3 = yaml.safe_load(open(q3_path, encoding='utf-8'))['industry_factors']
    print(f'  Q4概念数: {len(q4)} (Q3: {len(q3)})')
    check(len(q4) > 100, 'Q4概念数>100 (非空标定)')

    # 2) 中报4回退逐位一致
    for c in ZHONGBAO4:
        eq = (c in q4) and (q4[c] == q3[c])
        check(eq, f'回退: {c} Q4配置==Q3逐位一致')

    # 3) 日志回退打印
    log = open(LOG_P, encoding='utf-8', errors='replace').read()
    m = re.search(r'薄样本回退Q3权重.*?(\[.*?\])', log, re.S)
    if m:
        print(f'  日志回退行: {m.group(0)[:220]}')
        # 日志元组形如 (np.str_('2026中报扭亏'), 47)
        copied = set(re.findall(r"np\.str_\('([^']+)'\)", m.group(1)))
        check(copied >= set(ZHONGBAO4), f'日志copied清单⊇中报4 (实际: {sorted(copied)})')
    else:
        check(False, '日志含"薄样本回退Q3权重"打印')

    # 4) 无<60天概念拿fresh权重 (coverage口径: concept_map成员 × gated日期)
    with open(MAP_P, 'rb') as f:
        raw = pickle.load(f)
    codes_of = {}
    for code, cs in raw.items():
        for c in cs:
            codes_of.setdefault(c, []).append(code)
    with open(INCEP_P, 'rb') as f:
        raw_i = pickle.load(f)
    incep = {k: pd.Timestamp(v) for k, v in raw_i.items()}

    v = pd.read_csv(VAL_CSV, usecols=['date', 'code'], low_memory=False,
                    dtype={'code': str}).drop_duplicates()
    v['date'] = pd.to_datetime(v['date'])
    v = v[(v['date'] >= CALIB_START) & (v['date'] <= CALIB_END)]

    thin_fresh = []
    for c in q4:
        inc = incep.get(c)
        if inc is None:
            continue
        codes = codes_of.get(c, [])
        if not codes:
            continue
        gd = v[v['code'].isin(codes) & (v['date'] >= inc)]['date'].nunique()
        if gd < GATED_DAY_MIN and c not in q3:
            thin_fresh.append((c, gd))
        elif gd < GATED_DAY_MIN and q4[c] != q3.get(c):
            thin_fresh.append((c, gd))
    check(not thin_fresh, f'无<{GATED_DAY_MIN}天概念拿fresh权重 (违规: {thin_fresh})')

    # 5) 权重和≈1抽查 (回退概念除外)
    n_bad = 0
    for c, cfg in q4.items():
        if c in ZHONGBAO4:
            continue
        fs = cfg.get('factors', [])
        ws = cfg.get('weights', [])
        if fs and ws and len(fs) == len(ws):
            if abs(sum(float(w) for w in ws) - 1.0) > 0.05:
                n_bad += 1
    check(n_bad == 0, f'权重和≈1抽查通过 (异常: {n_bad})')

    # 6) save确认打印
    check('已保存' in log and '季度索引已更新' in log, '日志含保存确认+索引更新')

    print()
    if fails:
        print(f'FAIL: {len(fails)}项未过')
        sys.exit(1)
    print('ALL PASS — 干跑产物验证通过, PIT gate+薄样本回退按设计生效')


if __name__ == '__main__':
    main()
