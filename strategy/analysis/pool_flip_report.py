#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""池翻转报告 (2026-09-17, 2×危机产物): 对比两todate的池成员, 输出翻转码清单+阈值证据。
用于每次刷新后锚点对账的预期漂移分析 — 池滑动×承重墙 = 刷新彩票, 翻转清单=漂移预警。
用法: python analysis/pool_flip_report.py <todate_old> <todate_new>
"""
import sys
sys.path.insert(0, '/mnt/d/quant/strategy')
import pandas as pd
from core.stock_pool import get_stock_pool, _get_data_dir


def trailing20_amount(code, todate, data_dir):
    """某码在todate截断下的近20日日均成交额(万)与收盘价 — 阈值翻转证据"""
    try:
        df = pd.read_csv(f"{data_dir}/{code}_qfq.csv", parse_dates=['datetime'])
        df = df[df['datetime'] <= pd.Timestamp(todate)]
        if len(df) < 20:
            return None
        amt = df['amount'].iloc[-20:].mean()
        return float(amt), float(df['close'].iloc[-1])
    except Exception:
        return None


def main():
    old_d, new_d = sys.argv[1], sys.argv[2]
    data_dir = _get_data_dir()
    old_pool = get_stock_pool(todate=old_d, data_dir=data_dir)
    new_pool = get_stock_pool(todate=new_d, data_dir=data_dir)
    out_codes = sorted(old_pool - new_pool)  # 出池
    in_codes = sorted(new_pool - old_pool)   # 入池
    print(f"池 {old_d}: {len(old_pool)}  →  {new_d}: {len(new_pool)}   (净 {len(new_pool)-len(old_pool):+d})")
    print(f"\n出池 {len(out_codes)} 只 (旧日→新日: 成交额/收盘价):")
    for c in out_codes:
        o = trailing20_amount(c, old_d, data_dir)
        n = trailing20_amount(c, new_d, data_dir)
        if o and n:
            print(f"  {c}: amt {o[0]:.0f}→{n[0]:.0f}万  close {o[1]:.2f}→{n[1]:.2f}")
        else:
            print(f"  {c}: 数据不足 (o={o}, n={n})")
    print(f"\n入池 {len(in_codes)} 只:")
    for c in in_codes:
        o = trailing20_amount(c, old_d, data_dir)
        n = trailing20_amount(c, new_d, data_dir)
        if o and n:
            print(f"  {c}: amt {o[0]:.0f}→{n[0]:.0f}万  close {o[1]:.2f}→{n[1]:.2f}")
        else:
            print(f"  {c}: 数据不足 (o={o}, n={n})")


if __name__ == '__main__':
    main()
