#!/usr/bin/env python3
"""
选股漏斗记录器 — 记录每天五道关卡各阶段剩余股票数, 输出 Excel.

用法:
    .venv/bin/python strategy/funnel_record.py              # 记录今天
    .venv/bin/python strategy/funnel_record.py --backfill   # 回填信号CSV里所有历史日期
    .venv/bin/python strategy/funnel_record.py --date 2026-08-14

输出: strategy/media_out/选股漏斗记录.xlsx
Sheet1 漏斗统计: 日期 | 全市场 | 第一关股票池 | 第二~四关有信号 | 第五关最终入选
Sheet2 每日入选明细: 日期 | 代码 | 名称 | 金额 | 止损 | 止盈
"""

import os, sys, json, argparse
from datetime import datetime

import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _SCRIPT_DIR)

BT_DIR = os.path.join(_PROJECT_DIR, 'data', 'stock_data', 'backtrader_data')
SIGNALS_FP = os.path.join(_SCRIPT_DIR, 'rolling_validation_results', 'backtest_signals.csv')
POSITIONS_FP = os.path.join(_PROJECT_DIR, 'current_positions.json')
OUT_FP = os.path.join(_SCRIPT_DIR, 'media_out', '选股漏斗记录.xlsx')


def _market_size():
    """全市场股票数 (backtrader_data 中 qfq 文件数)."""
    n = 0
    for f in os.listdir(BT_DIR):
        if f.endswith('_qfq.csv') and f != 'sh000001_qfq.csv' and not f.startswith('._'):
            n += 1
    return n


def _pool_size(todate):
    """第一关后股票池数量."""
    try:
        from core.stock_pool import get_stock_pool
        pool = get_stock_pool(todate=todate)
        return len(pool)
    except Exception as e:
        print(f'  [WARN] 股票池计算失败: {e}')
        return None


def _signal_counts(date):
    """信号CSV中该日统计: (有信号记录的行数, buy=True的数量)."""
    if not os.path.exists(SIGNALS_FP):
        return None, None
    try:
        df = pd.read_csv(SIGNALS_FP, dtype={'code': str})
        df['date'] = pd.to_datetime(df['date'])
        day = df[df['date'] == pd.Timestamp(date)]
        if day.empty:
            return None, None
        return len(day), int(day['buy'].sum())
    except Exception:
        return None, None


def _final_picks(date):
    """第五关后最终入选名单."""
    picks = []
    # current_positions 中该日建仓的 (建仓日=选股次日, 所以entry_date=date+1天查找)
    if os.path.exists(POSITIONS_FP):
        try:
            positions = json.load(open(POSITIONS_FP, encoding='utf-8'))
            target_entry = pd.Timestamp(date) + pd.Timedelta(days=1)
            for code, p in positions.items():
                ed = pd.Timestamp(p.get('entry_date', ''))
                if ed == target_entry or ed == pd.Timestamp(date):
                    picks.append(code)
        except Exception:
            pass
    return picks


def record(date_str, backfill=False):
    """记录某一天的漏斗数据."""
    date = pd.Timestamp(date_str)
    rows = []

    if backfill:
        # 回填: 信号CSV中所有有信号的日期
        df = pd.read_csv(SIGNALS_FP, dtype={'code': str})
        df['date'] = pd.to_datetime(df['date'])
        dates = sorted(df['date'].dt.date.unique())
        print(f'回填 {len(dates)} 个历史日期...')
    else:
        dates = [date.date()]

    for d in dates:
        d_str = str(d)
        market = _market_size()
        pool = _pool_size(d_str)
        scored, sig = _signal_counts(d_str)
        picks = _final_picks(d_str)
        rows.append({
            '日期': d_str,
            '全市场': market,
            '第一关·股票池': pool,
            '第二~三关·完成因子打分': scored,
            '第四关·买入信号': sig,
            '第五关·最终入选': len(picks),
            '入选名单': ','.join(picks),
        })

    df_out = pd.DataFrame(rows)
    df_out = df_out.sort_values('日期')

    # 计算留存率
    df_out['留存率(全市场→入选)'] = df_out.apply(
        lambda r: f"{r['第五关·最终入选']/r['全市场']*100:.2f}%" if r['全市场'] else '-',
        axis=1)

    # 写入 Excel
    os.makedirs(os.path.dirname(OUT_FP), exist_ok=True)
    with pd.ExcelWriter(OUT_FP, engine='openpyxl') as writer:
        df_out.to_excel(writer, sheet_name='漏斗统计', index=False)
        # 列宽
        ws = writer.sheets['漏斗统计']
        widths = [14, 10, 14, 20, 16, 16, 40, 20]
        for i, w in enumerate(widths, 1):
            ws.column_dimensions[chr(64 + i)].width = w

    print(f'漏斗记录已保存: {OUT_FP} ({len(df_out)} 行)')
    print(df_out.to_string(index=False))
    return df_out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='选股漏斗记录器')
    parser.add_argument('--date', type=str, default=None, help='记录指定日期 YYYY-MM-DD')
    parser.add_argument('--backfill', action='store_true', help='回填信号CSV中所有历史日期')
    args = parser.parse_args()

    if args.backfill:
        record(None, backfill=True)
    else:
        date_str = args.date or datetime.now().strftime('%Y-%m-%d')
        record(date_str, backfill=False)
