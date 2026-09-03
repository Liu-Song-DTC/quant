#!/usr/bin/env python3
"""实盘选股前数据新鲜度门禁 (fail-closed: 任一硬指标不过 → 退出码1)

检查范围与期望:
  硬失败 (数据落后 → 拒绝出单):
    K线 (backtrader_data):        全量逐只检查 ≥ 预期交易日
                                  指数(sh000001/399006, 市场状态输入)同样硬检查;
                                  退市/停牌股用 kline_stale_allowlist.txt 豁免;
                                  北交所(43/82/83/87/88/92段)仅报告(已排除出股票池)
    融资融券 margin_daily:        最大日期 ≥ 预期交易日 - 1 (T+1晨发布)
    北向 northbound_daily:        同上
    概念历史 concept_hist:        各板块最大日期 ≥ 预期交易日 (白名单: 东财下架板块)
    基本面 fundamental_data:      最大报告期 ≥ 应披露的最新报告期
  软警告 (事件驱动型, 不阻断):
    龙虎榜/减持记录/减持计划/业绩预告/解禁: 仅报告最后日期
    概念映射 stock_concept_map:   mtime 超过14天提醒

预期交易日: 今天若为工作日且已过17:30 → 今天; 否则最近一个工作日。
            节假日误报请用 --expected YYYY-MM-DD 指定。

用法:
    python strategy/check_data_freshness.py [--expected 2026-09-02]
"""
import argparse
import os
import pickle
import random
import sys
from datetime import date as date_type, datetime, timedelta
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
BT_DIR = PROJECT / 'data' / 'stock_data' / 'backtrader_data'
FUND_DIR = PROJECT / 'data' / 'stock_data' / 'fundamental_data'
ALT_DIR = PROJECT / 'data' / 'alternative_data'
CONCEPT_HIST = PROJECT / 'data' / 'concept_hist.pkl'
CONCEPT_MAP = PROJECT / 'data' / 'stock_concept_map.pkl'
KLINE_ALLOWLIST = PROJECT / 'data' / 'stock_data' / 'kline_stale_allowlist.txt'

# 东财已下架、真实终止的板块 (2026-09-03 核实): 无后续数据, 豁免检查
CONCEPT_FROZEN = {'2025年报预增'}
# 报告期披露截止: 季度末 + 62天 (Q1 4/30, Q2 8/31, Q3 10/31, 年报 4/30)
REPORT_DEADLINE_DAYS = 62


def expected_trading_day() -> date_type:
    now = datetime.now()
    d = date_type.today()
    if now.hour * 60 + now.minute >= 17 * 60 + 30 and d.weekday() < 5:
        return d
    d -= timedelta(days=1)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def max_dt(series) -> str:
    s = pd.to_datetime(series, errors='coerce')
    return s.max().date().isoformat() if s.notna().any() else 'N/A'


def _file_last_date(fpath) -> str:
    """快速读CSV最后一行日期 (不整文件读入)"""
    with open(fpath, 'rb') as fh:
        fh.seek(0, 2)
        size = fh.tell()
        fh.seek(max(0, size - 160))
        last = fh.read().decode('utf-8', 'ignore').rstrip().splitlines()[-1]
    return last.split(',')[0].strip()[:10]


def _load_kline_allowlist() -> set:
    if not os.path.exists(KLINE_ALLOWLIST):
        return set()
    with open(KLINE_ALLOWLIST, encoding='utf-8') as f:
        return {ln.split()[0] for ln in f if ln.strip() and not ln.strip().startswith('#')}


def check_kline(exp):
    """K线全量检查 (2026-09-03 升级, 原抽样20只→全量分类):
    - 指数输入(sh000001上证/399006创业板指, 市场状态检测用) → 硬失败
    - 沪深股票 → 硬失败 (退市/停牌白名单豁免)
    - 北交所(43/82/83/87/88/92段) → 仅报告 (已整体排除出股票池, 2026-09-03用户指令)
    """
    from core.stock_pool import is_bse_code
    files = [f for f in os.listdir(BT_DIR) if f.endswith('_qfq.csv')]
    if not files:
        return False, f'K线: {BT_DIR} 无 *_qfq.csv 文件!'
    allow = _load_kline_allowlist()
    stale_hard = []
    bse_total = bse_stale = 0
    for f in files:
        code = f[:-8]
        d = _file_last_date(os.path.join(BT_DIR, f))
        if is_bse_code(code):
            bse_total += 1
            if d < exp.isoformat():
                bse_stale += 1
            continue
        if d >= exp.isoformat() or code in allow:
            continue
        stale_hard.append(f'{code}({d})')
    if stale_hard:
        s = ', '.join(stale_hard[:8]) + ('...' if len(stale_hard) > 8 else '')
        return False, f'K线: {len(files)}只, 滞后{len(stale_hard)}只(硬): {s}'
    return True, (f'K线: {len(files)}只全部新鲜'
                  f' (北交所{bse_total}只已排除, 其中滞后{bse_stale}只仅报告)')


def check_fundamental(exp):
    files = [f for f in os.listdir(FUND_DIR) if f.endswith('.csv')]
    if not files:
        return False, f'基本面: {FUND_DIR} 无 CSV!'
    # 应披露的最新报告期: 季度末 + 62天 ≤ 今天 的最近一个
    today = date_type.today()
    candidates = []
    y = today.year
    for qe in [date_type(y - 1, 12, 31), date_type(y, 3, 31), date_type(y, 6, 30),
               date_type(y, 9, 30), date_type(y, 12, 31)]:
        if today >= qe + timedelta(days=REPORT_DEADLINE_DAYS):
            candidates.append(qe)
    exp_rp = max(candidates).strftime('%Y%m%d') if candidates else 'N/A'
    sample = random.Random(7).sample(files, min(20, len(files)))
    best = '0'
    for f in sample:
        try:
            df = pd.read_csv(os.path.join(FUND_DIR, f), usecols=lambda c: '报告期' in c)
            col = df.columns[0]
            s = df[col].astype(str).str.extract(r'(\d{8})', expand=False)
            m = s.dropna().max()
            if m and m > best:
                best = m
        except Exception:
            continue
    ok = best >= exp_rp
    return ok, f'基本面: {len(files)}只, 抽样最大报告期 {best} (期望≥{exp_rp})'


def check_alt_pkl(name, col, exp, slack_days=0):
    p = ALT_DIR / name
    if not os.path.exists(p):
        return False, f'{name}: 文件不存在!'
    try:
        df = pd.read_pickle(p)
        if isinstance(df, dict):
            return True, f'{name}: dict {len(df)}键 (不校验日期)'
        m = pd.to_datetime(df[col], errors='coerce').max()
        if pd.isna(m):
            return True, f'{name}: 无有效日期 (不校验)'
        exp_d = exp - timedelta(days=slack_days)
        ok = m.date() >= exp_d
        return ok, f'{name}: 最大 {m.date()} (期望≥{exp_d})'
    except Exception as e:
        return False, f'{name}: 读取失败 {str(e)[:50]}'


def check_concept_hist(exp):
    if not os.path.exists(CONCEPT_HIST):
        return False, 'concept_hist.pkl: 文件不存在!'
    with open(CONCEPT_HIST, 'rb') as f:
        hist = pickle.load(f)
    stale = []
    for name, df in hist.items():
        if name in CONCEPT_FROZEN:
            continue
        if df['date'].max().date() < exp:
            stale.append(f'{name}({df["date"].max().date()})')
    ok = not stale
    msg = f'概念历史: {len(hist)}板块'
    if stale:
        msg += f', 落后: {stale[:5]}...' if len(stale) > 5 else f', 落后: {stale}'
    else:
        msg += f', 全部≥{exp}'
    return ok, msg


def check_concept_map():
    if not os.path.exists(CONCEPT_MAP):
        return False, 'stock_concept_map.pkl: 文件不存在!'
    age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(CONCEPT_MAP))
    ok = age.days <= 14
    return ok, f'概念映射: mtime {age.days}天前 (软警告, 阈值14天)'


def main():
    ap = argparse.ArgumentParser(description='实盘选股数据新鲜度门禁')
    ap.add_argument('--expected', type=str, default=None,
                    help='预期交易日 YYYY-MM-DD (节假日误报时指定)')
    args = ap.parse_args()

    exp = pd.Timestamp(args.expected).date() if args.expected else expected_trading_day()
    print(f'=== 数据新鲜度检查 (预期交易日 ≥ {exp}) ===')

    hard = [
        check_kline(exp),
        check_alt_pkl('margin_daily.pkl', 'date', exp, slack_days=1),
        check_alt_pkl('northbound_daily.pkl', 'date', exp, slack_days=1),
        check_concept_hist(exp),
        check_fundamental(exp),
    ]
    soft = [
        check_alt_pkl('dragon_tiger.pkl', '最近上榜日', exp),
        check_alt_pkl('reduction_plans.pkl', 'ann_date', exp),
        check_alt_pkl('reduction_records.pkl', 'start_date', exp),
        check_alt_pkl('yjyg_records.pkl', 'notice_date', exp),
        check_alt_pkl('unlock_schedule.pkl', 'unlock_date', exp),
        check_concept_map(),
    ]

    fails = 0
    for ok, msg in hard:
        mark = 'PASS' if ok else 'FAIL'
        print(f'  [{mark}] {msg}')
        fails += 0 if ok else 1
    print('  -- 软检查 (仅报告) --')
    for ok, msg in soft:
        print(f'  [{"OK " if ok else "WARN"}] {msg}')
        if not ok and '概念映射' in msg:
            print('        提示: 板块成员变化慢, 非阻塞; 超14天请跑 data/refresh_concept_map_0903.py')

    if fails:
        print(f'\n=== 门禁未通过: {fails} 项硬指标落后 ===')
        print('请先刷新数据: bash data/refresh_all.sh (另需Windows侧更新K线)')
        print('节假日误报: 用 --expected 指定上一交易日')
        sys.exit(1)
    print('\n=== 门禁通过: 所有硬指标新鲜 ===')
    sys.exit(0)


if __name__ == '__main__':
    main()
