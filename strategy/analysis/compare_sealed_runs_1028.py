"""封存跑对照比较 (2026-10-28, 交接计划 §3 任务 2)。

对两个封存 run 的产物做逐文件 SHA256 比对 + 行级差异定位:
  - equity_curve.csv        逐日权益 (NAV 真相)
  - trade_realized.csv      已平仓收益 (扣费胜率分母)
  - execution_fills.csv     成交明细 (逐笔)
  - portfolio_selections.csv 组合选择
  - regime_state.csv        敞口报表 (目标 vs 实际 ⇒ 列集合预期不同)

用法:
  python strategy/analysis/compare_sealed_runs_1028.py --a <runA/corrected> --b <runB/corrected>
"""
import argparse
import hashlib
import sys
from pathlib import Path

import pandas as pd

FILES = ['equity_curve.csv', 'trade_realized.csv', 'execution_fills.csv',
         'portfolio_selections.csv', 'regime_state.csv']


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def row_diff(a, b):
    """返回 (行数A, 行数B, 首行不同的行号或None, 列差异)。"""
    try:
        da = pd.read_csv(a)
        db = pd.read_csv(b)
    except Exception as e:
        return ('?', '?', f'读取失败: {e}', [])
    cols_a, cols_b = list(da.columns), list(db.columns)
    only_a = [c for c in cols_a if c not in cols_b]
    only_b = [c for c in cols_b if c not in cols_a]
    first = None
    common = [c for c in cols_a if c in cols_b]
    if len(da) == len(db) and common:
        # 逐行定位第一处不同 (按行序, 不按键 —— 与信号产物一致)
        sub_a = da[common].astype(str).reset_index(drop=True)
        sub_b = db[common].astype(str).reset_index(drop=True)
        diff_mask = (sub_a != sub_b).any(axis=1)
        if diff_mask.any():
            first = int(diff_mask.idxmax())
    return (len(da), len(db), first, only_a + only_b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--a', required=True, help='对照 run 的产物目录')
    ap.add_argument('--b', required=True, help='候选 run 的产物目录')
    args = ap.parse_args()
    A, B = Path(args.a), Path(args.b)
    print(f'对照 A: {A}\n候选 B: {B}\n')
    all_same = True
    for name in FILES:
        pa, pb = A / name, B / name
        if not pa.exists() or not pb.exists():
            print(f'[缺失] {name}: A={pa.exists()} B={pb.exists()}')
            all_same = False
            continue
        ha, hb = sha256(pa), sha256(pb)
        same = ha == hb
        all_same &= same
        tag = 'SHA256 相同 ✓' if same else 'SHA256 不同 ✗'
        print(f'[{tag}] {name}')
        if not same:
            na, nb, first, colsdiff = row_diff(pa, pb)
            print(f'    行数 A={na} B={nb}; 首行差异={first}; 列差异={colsdiff}')
    print()
    print('结论: ' + ('全部产物逐字节一致 ✓' if all_same else '存在差异 (见上) ✗'))
    return 0 if all_same else 1


if __name__ == '__main__':
    sys.exit(main())
