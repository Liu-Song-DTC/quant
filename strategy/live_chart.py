#!/usr/bin/env python3
"""
实盘选股 K线画图模块 —— 在选中个股的K线图上标注成本/止损/止盈线.

用法:
    .venv/bin/python strategy/live_chart.py                          # 使用最新 trade_orders.json
    .venv/bin/python strategy/live_chart.py --date 2026-07-31       # 指定日期
    .venv/bin/python strategy/live_chart.py --code 300236           # 只画指定股票
"""

import os, sys, json, argparse, math
from datetime import date as date_type
from pathlib import Path

import pandas as pd
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
DATA_DIR = os.path.join(_PROJECT_DIR, 'data', 'stock_data', 'backtrader_data')
OUTPUT_DIR = os.path.join(_SCRIPT_DIR, 'live_charts')
BAR_COUNT = 120  # 最近K线条数


def _load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _load_ohlcv(code):
    """加载某只股票的 qfq CSV, 返回最近 BAR_COUNT 根K线的 DataFrame."""
    fp = os.path.join(DATA_DIR, f'{code}_qfq.csv')
    if not os.path.exists(fp):
        return None
    df = pd.read_csv(fp, parse_dates=['datetime'])
    df = df.set_index('datetime')
    df = df.tail(BAR_COUNT)
    if len(df) < 30:
        return None
    # mplfinance 要求列名首字母大写
    df = df.rename(columns={
        'open': 'Open', 'high': 'High', 'low': 'Low',
        'close': 'Close', 'volume': 'Volume',
    })
    return df


def _fmt(price, precision=2):
    """智能保留小数位."""
    if price < 10:
        return f'{price:.{precision}f}'
    elif price < 100:
        return f'{price:.2f}'
    else:
        return f'{price:.1f}'


def generate_position_charts(orders_path=None, positions_path=None,
                             target_date=None, single_code=None):
    """
    对每个有 open 订单的股票生成K线图, 标注成本/止损/止盈.

    Args:
        orders_path: trade_orders.json 路径
        positions_path: current_positions.json 路径
        target_date: 图表输出子目录名 (默认从 orders 中取)
        single_code: 只生成指定股票 (6位代码)
    """
    try:
        import mplfinance as mpf
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')
    except ImportError:
        print('[WARN] mplfinance 未安装, 跳过K线图生成 (pip install mplfinance)')
        return []

        matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Droid Sans Fallback']
        matplotlib.rcParams['axes.unicode_minus'] = False
    except Exception:
        pass

    # 构建自定义style, 修正charles的中文字体缺失问题
    _base_style = mpf.make_mpf_style(base_mpf_style='charles', rc={
        'font.sans-serif': ['DejaVu Sans', 'Droid Sans Fallback'],
    })

    if orders_path is None:
        orders_path = os.path.join(_PROJECT_DIR, 'trade_orders.json')
    if positions_path is None:
        positions_path = os.path.join(_PROJECT_DIR, 'current_positions.json')

    if not os.path.exists(orders_path):
        print(f'[WARN] trade_orders.json 不存在: {orders_path}')
        return []
    if not os.path.exists(positions_path):
        print(f'[WARN] current_positions.json 不存在: {positions_path}')
        return []

    orders_data = _load_json(orders_path)
    positions = _load_json(positions_path)

    if target_date is None:
        target_date = orders_data.get('date', date_type.today().strftime('%Y-%m-%d'))

    orders = orders_data.get('orders', [])

    chart_dir = os.path.join(OUTPUT_DIR, target_date)
    os.makedirs(chart_dir, exist_ok=True)
    chart_files = []

    for order in orders:
        if order.get('action') != 'open':
            continue

        stock_code_full = order['stock_code']  # e.g. "300236.SZ"
        code = stock_code_full.split('.')[0]   # e.g. "300236"

        if single_code and code != single_code:
            continue

        pos = positions.get(code)
        if pos is None:
            print(f'  [SKIP] {code} 无持仓信息')
            continue

        entry_price = pos['entry_price']
        stop_loss = order.get('stop_loss_price')
        take_profit = order.get('take_profit_price')

        df = _load_ohlcv(code)
        if df is None:
            print(f'  [SKIP] {code} K线数据不足')
            continue

        last_close = float(df['Close'].iloc[-1])

        # ── 构建 addplot ─────────────────────────────────────
        addplots = []

        # 成本线 (蓝色虚线)
        entry_series = pd.Series(entry_price, index=df.index)
        addplots.append(mpf.make_addplot(
            entry_series, color='#2B7CE9', linestyle='--', width=1.0,
            label=f'Cost:{entry_price:.2f}'))

        # 止损线 (红色虚线)
        if stop_loss is not None:
            sl_series = pd.Series(stop_loss, index=df.index)
            addplots.append(mpf.make_addplot(
                sl_series, color='#C05050', linestyle=':', width=1.0,
                label=f'Stop:{stop_loss:.2f}'))

        # 止盈线 (绿色虚线)
        if take_profit is not None:
            tp_series = pd.Series(take_profit, index=df.index)
            addplots.append(mpf.make_addplot(
                tp_series, color='#4CAF50', linestyle=':', width=1.0,
                label=f'TP:{take_profit:.2f}'))

        # 现价 (灰色点线)
        close_series = pd.Series(last_close, index=df.index)
        addplots.append(mpf.make_addplot(
            close_series, color='#888888', linestyle='-.', width=0.8,
            label=f'Now:{last_close:.2f}'))

        # ── 标题 ─────────────────────────────────────────────
        pct_from_entry = (last_close - entry_price) / entry_price * 100
        title = (f'{code}  '
                 f'成本={_fmt(entry_price)}  现价={_fmt(last_close)}  '
                 f'盈亏={pct_from_entry:+.1f}%'
                 + (f'  止损={_fmt(stop_loss)}' if stop_loss else '')
                 + (f'  止盈={_fmt(take_profit)}' if take_profit else ''))

        try:
            fig, axes = mpf.plot(
                df, type='candle', style=_base_style,
                volume=True, addplot=addplots,
                title=title,
                returnfig=True,
                figsize=(14, 7),
                panel_ratios=(3, 1),
            )

            # 在蜡烛图面板上添加文字标注
            ax0 = axes[0]
            last_idx = len(df) - 1

            # 成本线标注
            ax0.text(last_idx, entry_price, f' 成本 {entry_price:.2f}',
                     fontsize=8, color='#2B7CE9', va='bottom', fontweight='bold')

            # 止损线标注
            if stop_loss is not None:
                ax0.text(last_idx, stop_loss, f' 止损 {stop_loss:.2f}',
                         fontsize=8, color='#C05050', va='top')

            # 止盈线标注
            if take_profit is not None:
                ax0.text(last_idx, take_profit, f' 止盈 {take_profit:.2f}',
                         fontsize=8, color='#4CAF50', va='bottom')

            chart_path = os.path.join(chart_dir, f'{code}.png')
            fig.savefig(chart_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            chart_files.append({'code': code, 'path': chart_path})
            print(f'  [OK] {code} -> {chart_path}')

        except Exception as e:
            print(f'  [WARN] {code} K线图生成失败: {e}')
            continue

    if chart_files:
        print(f'\n生成K线图: {len(chart_files)} 张 -> {chart_dir}/')
    return chart_files


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='实盘选股K线画图')
    parser.add_argument('--date', type=str, default=None, help='日期')
    parser.add_argument('--code', type=str, default=None, help='只画指定股票(6位代码)')
    args = parser.parse_args()
    generate_position_charts(target_date=args.date, single_code=args.code)
