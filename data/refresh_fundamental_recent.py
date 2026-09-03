#!/usr/bin/env python3
"""2026-08-31 ST漏判修复: 强制重刷最近3个报告期基本面源数据并增量并入per-stock CSV.

背景:
  fundamental_source 目录为空(源数据丢失), 5510只 fundamental_data/*.csv 停在
  2026-03-14 构建态 → ST判定(get_st_timeline/is_st)依据的"股票简称"列陈旧,
  2026年新戴帽股票(如301117佳缘科技)漏判。
  本脚本只重下 20251231/20260331/20260630 (约12次下载), 历史行不动。

合并语义:
  - 目标报告期 = 替换: 先删除CSV中同报告期的旧行, 再并入新行 (20251231年报3月快照残缺)
  - 新报告期 = 追加
  - 写文件: 临时文件 + os.replace 原子替换, 中途被杀不产生半文件
  - 列对齐: 新行 reindex 到旧列 (akshare列变动时旧值缺失补NaN, 多余新列丢弃)
  - 数据可用日期沿用系统约定 = 报告期 (与既有行一致; as-of精细化属B批工作)

用法: /mnt/d/quant/.venv/bin/python data/refresh_fundamental_recent.py
"""
import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd  # noqa: E402
from data_manager import StockDataManager  # noqa: E402 (含akshare代理patch)

QUARTERS = ['20251231', '20260331', '20260630']

# akshare 新版 yjbb 列名漂移 → 映射回系统旧命名
# (2026-09-02 修复: 新版'营业收入同比增长'等与旧列'营业总收入-同比增长'错位,
#  导致2026行同比/营收/净利润主列全NaN, 收益崩溃归因中的基本面损坏)
RENAME_YJBB = {
    '营业收入': '营业总收入-营业总收入',
    '营业收入同比增长': '营业总收入-同比增长',
    '营业收入季度环比': '营业总收入-季度环比增长',
    '净利润': '净利润-净利润',
    '净利润同比增长': '净利润-同比增长',
    '净利润季度环比': '净利润-季度环比增长',
    '公告日期': '最新公告日期',
}


def build_quarter_frame(qdir, q):
    """复刻 build_stock_fundamental_history 的单季度合并逻辑 (yjbb为基 + 三表前缀合并)."""
    yjbb_file = qdir / "yjbb.csv"
    if not yjbb_file.exists() or os.path.getsize(yjbb_file) == 0:
        print(f"  [{q}] yjbb.csv 缺失/为空, 跳过该季度")
        return None
    yjbb_df = pd.read_csv(yjbb_file, dtype={'股票代码': str})
    yjbb_df = yjbb_df.rename(columns=RENAME_YJBB)

    for name, f in [('zcfz', qdir / "zcfz.csv"), ('lrb', qdir / "lrb.csv"), ('xjll', qdir / "xjll.csv")]:
        if f.exists() and os.path.getsize(f) > 0:
            df = pd.read_csv(f, dtype={'股票代码': str})
            df = df.add_prefix(f'{name}_')
            df = df.rename(columns={f'{name}_股票代码': '股票代码'})
            yjbb_df = yjbb_df.merge(df, on='股票代码', how='left')

    yjbb_df['报告期'] = q
    yjbb_df['数据可用日期'] = q  # 系统既有约定: 数据可用日期=报告期
    return yjbb_df


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--quarters', nargs='*', default=None,
                    help='仅处理指定季度 (默认全部3个)')
    ap.add_argument('--no-download', action='store_true',
                    help='跳过下载, 仅用现有源文件重合并')
    args = ap.parse_args()
    quarters = args.quarters if args.quarters else QUARTERS

    mgr = StockDataManager()
    src = mgr.fundamental_source_dir
    out = mgr.fundamental_data_dir

    # ── 1. 确保源数据存在 (只补缺失表, 已有文件不动 — 8/31快照已核验完整) ──
    if not args.no_download:
        print(f"=== 检查 {quarters} 源数据 ===")
        for q in quarters:
            qdir = src / q
            need = [t for t in ('yjbb', 'zcfz', 'lrb', 'xjll')
                    if not (qdir / f'{t}.csv').exists()
                    or os.path.getsize(qdir / f'{t}.csv') == 0]
            if not need:
                print(f"  [{q}] 源数据完整, 跳过")
                continue
            os.makedirs(qdir, exist_ok=True)
            mgr.download_financial_data_by_date(q)  # help() 逐表跳过已存在文件, 仅补缺
            n = sum(1 for t in ('yjbb', 'zcfz', 'lrb', 'xjll')
                    if (qdir / f'{t}.csv').exists()
                    and os.path.getsize(qdir / f'{t}.csv') > 0)
            print(f"  [{q}] 补齐后 {n}/4 个文件非空 (缺失: {need})")
    else:
        print("=== --no-download: 跳过下载, 直接合并现有源文件 ===")

    # ── 2. 合并新季度frame ──
    frames = []
    for q in quarters:
        fr = build_quarter_frame(src / q, q)
        if fr is not None and len(fr):
            print(f"  [{q}] 合并 {len(fr)} 行")
            frames.append(fr)
    if not frames:
        print("无新数据可并入, 退出")
        return
    new_all = pd.concat(frames, ignore_index=True)

    # ── 3. 增量并入 per-stock CSV ──
    codes = sorted(new_all['股票代码'].dropna().unique())
    print(f"=== 并入 {len(codes)} 只股票 ===")
    updated, replaced_rows = 0, 0
    st_hits = []
    for code in codes:
        fpath = out / f"{code}.csv"
        add = new_all[new_all['股票代码'] == code]
        if fpath.exists():
            old = pd.read_csv(fpath, dtype={'股票代码': str})
            old['报告期'] = old['报告期'].astype(str)  # 旧CSV可能int, 统一为str
            n_repl = old['报告期'].isin(quarters).sum()
            replaced_rows += n_repl
            old = old[~old['报告期'].isin(quarters)]
        else:
            old = pd.DataFrame()
        add = add.copy()
        add['报告期'] = add['报告期'].astype(str)
        if not old.empty:
            add = add.reindex(columns=old.columns)
            merged = pd.concat([old, add], ignore_index=True)
        else:
            merged = add
        if '报告期' in merged.columns:
            merged = merged.sort_values('报告期', kind='stable').reset_index(drop=True)
        tmp = fpath.with_suffix('.tmp')
        merged.to_csv(tmp, index=False, encoding='utf-8')
        os.replace(tmp, fpath)
        updated += 1

        names = add['股票简称'].astype(str)
        if names.str.contains('ST').any():
            st_hits.append((code, names.iloc[-1], add['报告期'].iloc[-1] if '报告期' in add.columns else '?'))

    print(f"=== 完成: 更新 {updated} 只, 替换旧季度行 {replaced_rows} 行 ===")
    if st_hits:
        print(f"=== 新数据中发现 ST/带风险警示名称 {len(st_hits)} 只 ===")
        for code, name, rp in sorted(st_hits):
            print(f"  {code} {name} (报告期{rp})")
    else:
        print("=== 新数据中未发现ST名称 ===")


if __name__ == '__main__':
    main()
