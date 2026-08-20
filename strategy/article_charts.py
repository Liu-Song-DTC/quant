#!/usr/bin/env python3
"""自媒体文章配图生成器 — 为方法论文章生成插图.

用法: .venv/bin/python strategy/article_charts.py
输出: strategy/media_out/images/
"""

import os, sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
DATA_DIR = os.path.join(_PROJECT_DIR, 'data', 'stock_data', 'backtrader_data')
OUT_DIR = os.path.join(_SCRIPT_DIR, 'media_out', 'images')

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Droid Sans Fallback']
plt.rcParams['axes.unicode_minus'] = False

# 浅色商务风: 白底深字, 与公众号文章(极简商务)配色一致
# A股惯例: 红涨绿跌
BG = '#ffffff'
FG = '#333333'
ACCENT = '#2f6fed'
ACCENT2 = '#ff9d2e'
GREEN = '#0aa878'   # 跌 / 正向二次色
RED = '#e64545'     # 涨
MUTED = '#9aa0a6'
CARD = '#f5f7fa'    # 浅卡底色, 配彩色描边


def _style_ax(ax):
    ax.set_facecolor(BG)
    for s in ax.spines.values():
        s.set_color('#e0e0e0')


def chart_funnel():
    """漏斗图: 5000 → 2000 → 几十 → 3-5只"""
    fig, ax = plt.subplots(figsize=(8, 7), facecolor=BG)
    _style_ax(ax)

    stages = [
        ('全市场 A 股', 5000, '#4da3ff'),
        ('第一关 · 股票池筛选\n剔除次新/仙股/僵尸/ST', 2000, '#3d7ec9'),
        ('第二~四关 · 因子打分\n动态验证 + 信号共振', 60, '#ffb84d'),
        ('第五关 · 风控组合\n最终入选', 4, '#4ecb8d'),
    ]

    for i, (label, n, color) in enumerate(stages):
        width = 0.9 - i * 0.18
        y = 3 - i
        ax.barh(y, width, height=0.62, color=color, alpha=0.85,
                edgecolor='none', left=(1 - width) / 2)
        ax.text(0.5, y, f'{label}\n{n} 只' if i > 0 else f'{label}\n{n}+ 只',
                ha='center', va='center', fontsize=11 + i * 1.5,
                color='white' if i > 0 else 'white', fontweight='bold')

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, 3.7)
    ax.axis('off')
    ax.set_title('我的系统怎么从 5000 只里选出最后几只', color=FG,
                 fontsize=15, fontweight='bold', pad=18)

    fig.savefig(os.path.join(OUT_DIR, '漏斗图.png'), dpi=160,
                bbox_inches='tight', facecolor=BG)
    plt.close(fig)


def chart_pipeline():
    """五道关卡流程图"""
    fig, ax = plt.subplots(figsize=(11, 3.2), facecolor=BG)
    _style_ax(ax)

    steps = [
        ('数据更新', '日线 · 基本面\n龙虎榜 · 北向', '#5a6b7f'),
        ('第一关', '股票池筛选\n先做减法', '#3d7ec9'),
        ('第二关', '因子打分\n三维画像', '#4da3ff'),
        ('第三关', '动态因子\nIC 验证', '#ffb84d'),
        ('第四关', '信号共振\n多重确认', '#ff9d4d'),
        ('第五关', '风控组合\n止损止盈', '#4ecb8d'),
    ]

    n = len(steps)
    for i, (title, desc, color) in enumerate(steps):
        x = i / (n - 1)
        box = FancyBboxPatch((x - 0.065, 0.42), 0.13, 0.5,
                             boxstyle='round,pad=0.012',
                             facecolor=color, edgecolor='none', alpha=0.9)
        ax.add_patch(box)
        ax.text(x, 0.80, title, ha='center', va='center',
                fontsize=11, fontweight='bold', color='white')
        ax.text(x, 0.55, desc, ha='center', va='center',
                fontsize=7.5, color='white', alpha=0.9)

        if i < n - 1:
            arr = FancyArrowPatch((x + 0.068, 0.67), (x + 0.128, 0.67),
                                  arrowstyle='-|>', mutation_scale=16,
                                  color=MUTED, lw=1.5)
            ax.add_patch(arr)

    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('每天盘前，程序自动跑完这条流水线', color=FG,
                 fontsize=14, fontweight='bold', pad=12)

    fig.savefig(os.path.join(OUT_DIR, '流水线.png'), dpi=160,
                bbox_inches='tight', facecolor=BG)
    plt.close(fig)


def chart_three_standards():
    """好股三标准图"""
    fig, ax = plt.subplots(figsize=(8.5, 4.6), facecolor=BG)
    _style_ax(ax)

    items = [
        ('① 走势强', '趋势向上 · 回调有支撑\n放量突破 · 缩量不慌\n—— 市场已经用钱投了票', '#4da3ff'),
        ('② 底子硬', 'ROE 够硬 · 成长够快\n估值不离谱\n—— 炒作退潮也跌不深', '#4ecb8d'),
        ('③ 聪明钱在场', '龙虎榜机构 · 北向加仓\n融资变化 · 题材热度\n—— 大资金真金白银站队', '#ffb84d'),
    ]

    for i, (title, desc, color) in enumerate(items):
        x = 0.12 + i * 0.31
        box = FancyBboxPatch((x - 0.115, 0.18), 0.23, 0.68,
                             boxstyle='round,pad=0.015',
                             facecolor=CARD, edgecolor=color, lw=2.5)
        ax.add_patch(box)
        ax.text(x, 0.68, title, ha='center', va='center',
                fontsize=13, fontweight='bold', color=color)
        ax.text(x, 0.40, desc, ha='center', va='center',
                fontsize=8.5, color=FG, linespacing=1.7)

    ax.text(0.5, 0.03, '三个条件同时满足 → 全市场通常不超过几十只',
            ha='center', fontsize=11, color=ACCENT2, fontweight='bold')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.95)
    ax.axis('off')
    ax.set_title('我系统里的"好股"，只有三个标准', color=FG,
                 fontsize=15, fontweight='bold', pad=14)

    fig.savefig(os.path.join(OUT_DIR, '三标准.png'), dpi=160,
                bbox_inches='tight', facecolor=BG)
    plt.close(fig)


def chart_walkforward():
    """walk-forward 滚动验证示意图"""
    fig, ax = plt.subplots(figsize=(9.5, 3.4), facecolor=BG)
    _style_ax(ax)

    # 时间轴
    ax.plot([0.04, 0.96], [0.55, 0.55], color='#3a4a5c', lw=3, zorder=1)
    ax.text(0.03, 0.35, '过去', color=MUTED, fontsize=10, ha='center')
    ax.text(0.97, 0.35, '现在', color=MUTED, fontsize=10, ha='center')

    # 三个滚动窗口
    windows = [
        (0.16, 0.30, '第1轮检验', '#4da3ff', '用这段历史数据\n验证因子有效性'),
        (0.45, 0.59, '第2轮检验', '#ffb84d', '窗口前移\n重新验证一遍'),
        (0.74, 0.88, '第3轮检验', '#4ecb8d', '失效的淘汰\n新有效的加入'),
    ]
    for x1, x2, label, color, desc in windows:
        ax.add_patch(plt.Rectangle((x1, 0.45), x2 - x1, 0.2,
                                   facecolor=color, alpha=0.35, edgecolor=color, lw=2))
        ax.text((x1 + x2) / 2, 0.555, label, ha='center', va='center',
                fontsize=9.5, fontweight='bold', color=color)
        ax.text((x1 + x2) / 2, 0.86, desc, ha='center', fontsize=9,
                color=FG, linespacing=1.6)

    # 箭头: 窗口前移
    for x in [0.30, 0.59]:
        arr = FancyArrowPatch((x + 0.015, 0.62), (x + 0.13, 0.62),
                              arrowstyle='-|>', mutation_scale=13,
                              color=MUTED, lw=1.2)
        ax.add_patch(arr)
    ax.text(0.375, 0.72, '时间窗口向前滚动', color=MUTED, fontsize=9)

    ax.set_xlim(0, 1)
    ax.set_ylim(0.2, 1)
    ax.axis('off')
    ax.set_title('walk-forward：永远只用"最近的证据"检验因子', color=FG,
                 fontsize=14, fontweight='bold', pad=10)

    fig.savefig(os.path.join(OUT_DIR, 'walkforward.png'), dpi=160,
                bbox_inches='tight', facecolor=BG)
    plt.close(fig)


def chart_kline_demo(code='002335', entry=32.14, stop=29.89, target=40.17):
    """真实K线图: 成本/止损/止盈线标注"""
    fp = os.path.join(DATA_DIR, f'{code}_qfq.csv')
    if not os.path.exists(fp):
        print(f'[SKIP] {code} 数据不存在')
        return
    df = pd.read_csv(fp, parse_dates=['datetime']).tail(60)
    if df.empty:
        return

    fig, (ax, axv) = plt.subplots(2, 1, figsize=(9, 5.2),
                                   gridspec_kw={'height_ratios': [3.4, 1]},
                                   facecolor=BG)
    _style_ax(ax)
    _style_ax(axv)
    ax.set_facecolor(BG)
    axv.set_facecolor(BG)

    x = np.arange(len(df))
    up = df['close'] >= df['open']
    down = ~up

    # 蜡烛
    ax.vlines(x[up], df['low'][up], df['high'][up], color=RED, lw=1.2)
    ax.vlines(x[down], df['low'][down], df['high'][down], color=GREEN, lw=1.2)
    ax.bar(x[up], (df['close'] - df['open'])[up], bottom=df['open'][up],
           color=RED, width=0.62)
    ax.bar(x[down], (df['open'] - df['close'])[down], bottom=df['close'][down],
           color=GREEN, width=0.62)

    # 成本/止损/止盈线
    ax.axhline(entry, color=ACCENT, ls='--', lw=1.6)
    ax.axhline(stop, color=RED, ls=':', lw=1.6)
    ax.axhline(target, color=GREEN, ls=':', lw=1.6)
    ax.text(len(df) - 1, entry, f' 成本 {entry:.2f}', color=ACCENT,
            fontsize=9, va='bottom', fontweight='bold')
    ax.text(len(df) - 1, stop, f' 止损 {stop:.2f}', color=RED,
            fontsize=9, va='top')
    ax.text(len(df) - 1, target, f' 止盈 {target:.2f}', color=GREEN,
            fontsize=9, va='bottom')

    # 成交量
    axv.bar(x, df['volume'], color=['#4da3ff' if c else '#ffb84d'
                                     for c in up], width=0.62, alpha=0.6)

    for a in (ax, axv):
        a.tick_params(colors=MUTED, labelsize=8)
        a.grid(False)

    # x轴日期
    idx_ticks = np.linspace(0, len(df) - 1, 6, dtype=int)
    ax.set_xticks(idx_ticks)
    ax.set_xticklabels([df['datetime'].iloc[i].strftime('%m-%d') for i in idx_ticks])
    axv.set_xticks([])
    axv.set_yticks([])

    ax.set_title('系统输出示例：每笔买入都自带止损止盈位', color=FG,
                 fontsize=13, fontweight='bold', pad=10)

    fig.savefig(os.path.join(OUT_DIR, 'K线示例.png'), dpi=160,
                bbox_inches='tight', facecolor=BG)
    plt.close(fig)


if __name__ == '__main__':
    os.makedirs(OUT_DIR, exist_ok=True)
    chart_funnel()
    print('[OK] 漏斗图.png')
    chart_pipeline()
    print('[OK] 流水线.png')
    chart_three_standards()
    print('[OK] 三标准.png')
    chart_walkforward()
    print('[OK] walkforward.png')
    chart_kline_demo()
    print('[OK] K线示例.png')
    print(f'\n全部图片输出到: {OUT_DIR}')
