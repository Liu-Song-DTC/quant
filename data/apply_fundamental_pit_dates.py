#!/usr/bin/env python3
"""一次性迁移: `fundamental_data/*.csv` 的 `数据可用日期` 改为法定披露截止日。

口径见 `fundamental_pit.py`。**只改这一列**, 其余列逐位不动。

为什么改数据列而不是改消费方: 消费方有 10+ 处(factor_preparer / fundamental.py /
generate_trade_orders / offline_calibration / 各探针), 漏一处就两套口径;
改列则所有消费方自动正确, 且数据自描述(`数据可用日期` 名副其实)。

铁律: 这是**数据重锚**(四指标豁免, 须登记)。改完 `_data_fingerprint`(路径+mtime+size)
必变 ⇒ 因子缓存全失效 ⇒ 下次回测全新重算, 不会拿旧缓存糊弄。

用法:
    # 干跑(默认): 只报告会改多少行, 不落盘
    /mnt/d/quant/.venv/bin/python data/apply_fundamental_pit_dates.py
    # 落盘(先自动硬链接快照)
    /mnt/d/quant/.venv/bin/python data/apply_fundamental_pit_dates.py --apply
"""
import argparse
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd  # noqa: E402

from fundamental_pit import pit_available_series  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FDIR = os.path.join(ROOT, 'data', 'stock_data', 'fundamental_data')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--apply', action='store_true', help='真正落盘(默认干跑)')
    ap.add_argument('--dir', default=FDIR)
    args = ap.parse_args()

    files = sorted(f for f in os.listdir(args.dir) if f.endswith('.csv'))
    print(f'目录 {args.dir}\n文件 {len(files)} 只  模式={"APPLY" if args.apply else "DRY-RUN"}',
          flush=True)

    if args.apply:
        snap = args.dir + '.prePIT_' + time.strftime('%Y%m%d_%H%M')
        os.makedirs(snap, exist_ok=True)
        n_link = 0
        for f in files:
            dst = os.path.join(snap, f)
            if not os.path.exists(dst):
                try:
                    os.link(os.path.join(args.dir, f), dst)   # 硬链接: 零拷贝
                except OSError:
                    shutil.copy2(os.path.join(args.dir, f), dst)
                n_link += 1
        print(f'快照 {snap}: {n_link} 文件(硬链接)', flush=True)

    n_chg = n_row = n_tot = n_bad = n_missing = 0
    examples = []
    t0 = time.time()
    for i, f in enumerate(files):
        p = os.path.join(args.dir, f)
        try:
            d = pd.read_csv(p, dtype=str)
        except Exception as e:
            n_bad += 1
            if n_bad <= 3:
                print(f'  !! {f} 读取失败: {str(e)[:70]}', flush=True)
            continue
        if '报告期' not in d.columns or '数据可用日期' not in d.columns:
            n_missing += 1
            continue
        n_tot += 1
        new = pit_available_series(d['报告期'])
        old = d['数据可用日期'].astype(str)
        # 非法报告期 → 新值为 '' ⇒ 保留旧值(不制造空洞), 但要计数
        blank = new == ''
        if blank.any():
            n_bad += int(blank.sum())
            new = new.where(~blank, old)
        chg = (new.values != old.values)
        n_row += len(d)
        if chg.any():
            n_chg += 1
            if len(examples) < 3:
                j = int(chg.nonzero()[0][0])
                examples.append(f'    {f[:-4]}: 报告期 {d["报告期"].iloc[j]} '
                                f'{old.iloc[j]} -> {new.iloc[j]}')
            if args.apply:
                d['数据可用日期'] = new
                tmp = p + '.tmp'
                d.to_csv(tmp, index=False, encoding='utf-8')
                os.replace(tmp, p)
        if (i + 1) % 1000 == 0:
            print(f'  ... {i+1}/{len(files)} (改 {n_chg} 只, {time.time()-t0:.0f}s)', flush=True)

    print(f'\n完成: 扫描 {n_tot} 只 / {n_row} 行; 需改 {n_chg} 只; '
          f'读失败 {n_bad}; 缺列 {n_missing}; {time.time()-t0:.0f}s')
    for e in examples:
        print(e)
    if not args.apply:
        print('(干跑, 未落盘) — 加 --apply 执行')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
