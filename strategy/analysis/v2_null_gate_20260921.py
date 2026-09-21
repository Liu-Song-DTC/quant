#!/usr/bin/env python3
"""
V2 OOS闸统计null预计算 (2026-09-21, 冻结态9/17, 只读)

9/30正式V2的门禁: "Q3 IC符号与标定窗一致率≥70% 且 均值不崩塌(<标定窗一半)
→ 标定程序OOS成立"。这是启发式阈值 — 本脚本在冻结态(标定窗2021-07~2026-06
完全覆盖)预计算其统计null, 9/30只需比observed:

  Null A 每因子翻号概率: 该因子标定窗IC块bootstrap重采样2000次, P(重采样均值
         符号≠标定窗均值符号) — 弱t因子天然高翻号率(解释早读"4翻负全小幅基本面")。
  Null B 符号一致率null分布: H0="Q3只是标定窗分布的另一段延续"。块bootstrap
         按日期重采样(保留因子间截面相关), 抽L天伪Q3段, 与标定窗均值符号
         一致率 → null分布。L=47(9/17态Q3-so-far可算日)与L=55(9/30全窗)双口径。
         若9/30 observed一致率落在null分布低尾 → 符号衰减证据; 落在null带内 →
         无衰减证据(Q3=正常延续)。早读observed 83%可直接对表。
  Null C 均值崩塌率null分布: 同样块bootstrap, 伪Q3均值/标定窗均值 的分布 →
         5th分位=null下"自然崩塌"阈值。observed比值低于null 5th分位才算崩。

产物: v2_null_gate_20260921.csv (每因子翻号率) + 打印null分布分位表入档。
只读: 不写任何生产文件。执行时间~35-45min (Phase 1因子值重算占大头)。
执行: cd strategy && QUANT_ALT_NO_AUTOREFRESH=1 /mnt/d/quant/.venv/bin/python \
      analysis/v2_null_gate_20260921.py
"""
import os
import sys
import gc
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from v2_oos_preview_0917 import (  # noqa: E402
    load_quarter_config, selected_factors, daily_ic_series, phase1_tail,
    CALIB_WINDOW, DATA_END,
)

OUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'v2_null_gate_20260921.csv')

RNG_SEED = 20260921
N_BOOT = 2000
BLK = 10          # 块长(交易日); 敏感性另报BLK=5
LENGTHS = (47, 55)  # 9/17态Q3-so-far可算日 / 9/30全窗可算日


def block_idx(n, L, blk, rng):
    """移动块bootstrap: 从长n序列抽长L样本的索引"""
    n_blk = int(np.ceil(L / blk))
    starts = rng.integers(0, max(n - blk + 1, 1), size=n_blk)
    idx = np.concatenate([s + np.arange(blk) for s in starts])[:L]
    return idx


def main():
    print(f"V2 OOS闸统计null预计算 — 数据态 as-of {DATA_END.date()} (冻结, 只读)")
    print(f"标定窗 {CALIB_WINDOW[0].date()}~{CALIB_WINDOW[1].date()} | "
          f"bootstrap {N_BOOT}次 块长{BLK} | 伪Q3长度 {LENGTHS}")

    # ---- Phase 1: 因子值 (与早读同构, ~30min) ----
    factor_df, _concept_map = phase1_tail()
    if factor_df is None:
        return
    facts = selected_factors(load_quarter_config('2026Q3'))
    print(f"\n选中因子 {len(facts)} 个")

    # ---- 标定窗每日IC矩阵 (日期×因子) ----
    seg = factor_df[(factor_df['date'] >= CALIB_WINDOW[0]) &
                    (factor_df['date'] <= CALIB_WINDOW[1])]
    del factor_df
    gc.collect()
    M = pd.DataFrame({fn: daily_ic_series(seg, fn) for fn in facts}).sort_index()
    del seg
    gc.collect()
    n_days = len(M)
    print(f"IC矩阵: {n_days} 日 × {len(facts)} 因子 (含NaN按dropna处理)")
    calib_means = M.mean()
    calib_signs = np.sign(calib_means.values)

    rng = np.random.default_rng(RNG_SEED)

    # ---- Null A: 每因子翻号概率 ----
    rows = []
    for fn in facts:
        s = M[fn].dropna().values
        if len(s) < 30:
            rows.append({'factor': fn, 'n_days': len(s),
                         'calib_mean': M[fn].mean(), 'flip_prob': np.nan})
            continue
        flips = 0
        for _ in range(N_BOOT):
            idx = block_idx(len(s), len(s), BLK, rng)
            if np.sign(s[idx].mean()) != np.sign(s.mean()):
                flips += 1
        rows.append({'factor': fn, 'n_days': len(s),
                     'calib_mean': M[fn].mean(), 'flip_prob': flips / N_BOOT})
    dfA = pd.DataFrame(rows).sort_values('calib_mean', ascending=False)
    print(f"\n=== Null A: 每因子翻号概率 (null下P(重采样均值翻号), 高=弱t因子) ===")
    print(dfA.to_string(index=False,
                        float_format=lambda x: f'{x:+.4f}',
                        formatters={'flip_prob': lambda x: f'{x:.3f}'}))
    dfA.to_csv(OUT_CSV, index=False)
    print(f"\n已保存: {OUT_CSV}")

    # ---- Null B/C: 符号一致率 + 均值崩塌率 null分布 ----
    Mv = M.dropna(axis=0).values      # 仅全因子有IC的日期(保留截面相关)
    Md = M.dropna(axis=0).index
    n_full = len(Mv)
    signs_full = np.sign(M.dropna(axis=0).mean(axis=0).values)
    means_full = M.dropna(axis=0).mean(axis=0).values
    print(f"\n全因子共同IC日: {n_full} (dropna后)")
    for L in LENGTHS:
        cons = np.empty(N_BOOT)
        ratios = np.empty(N_BOOT)
        for b in range(N_BOOT):
            idx = block_idx(n_full, L, BLK, rng)
            pseudo = Mv[idx]
            pm = pseudo.mean(axis=0)
            cons[b] = np.mean(np.sign(pm) == signs_full)
            # 崩塌率: 伪段均值/标定均值 (标定均值近零的因子用绝对缩放避免除零爆炸)
            ratios[b] = np.nanmean(pm / np.where(np.abs(means_full) < 1e-8,
                                                 np.nan, means_full))
        q = {p: np.percentile(cons, p) for p in (2.5, 5, 25, 50, 75, 95, 97.5)}
        rq = {p: np.percentile(ratios, p) for p in (2.5, 5, 25, 50, 75, 95, 97.5)}
        print(f"\n=== Null B/C: 伪Q3段长度 L={L}日 ({N_BOOT}次块bootstrap, 块长{BLK}) ===")
        print(f"  符号一致率null分布分位: "
              f"2.5%={q[2.5]:.3f} 5%={q[5]:.3f} 25%={q[25]:.3f} "
              f"中位={q[50]:.3f} 75%={q[75]:.3f} 95%={q[95]:.3f} 97.5%={q[97.5]:.3f}")
        print(f"  崩塌率null分布分位: "
              f"2.5%={rq[2.5]:+.3f} 5%={rq[5]:+.3f} 25%={rq[25]:+.3f} "
              f"中位={rq[50]:+.3f} 75%={rq[75]:+.3f} 95%={rq[95]:+.3f} 97.5%={rq[97.5]:+.3f}")
        print(f"  9/30判定: observed一致率 < {q[2.5]:.3f}(null 2.5%分位) → 符号衰减证据;"
              f" observed崩塌率 < {rq[5]:.3f}(null 5%分位) → 均值崩塌证据")
        print(f"  早读对照(9/17态): observed一致率83%(L=47) — 见下收尾判定")

    # ---- 块长敏感性 (BLK=5, 只报L=55口径分位) ----
    L = 55
    cons5 = np.empty(N_BOOT)
    for b in range(N_BOOT):
        idx = block_idx(n_full, L, 5, rng)
        pm = Mv[idx].mean(axis=0)
        cons5[b] = np.mean(np.sign(pm) == signs_full)
    q5 = np.percentile(cons5, (2.5, 50, 97.5))
    print(f"\n=== 块长敏感性: BLK=5, L=55 → 一致率null 2.5%/50%/97.5% = "
          f"{q5[0]:.3f}/{q5[1]:.3f}/{q5[2]:.3f} (与BLK=10对照, 差异应小) ===")

    print("\n完成。9/30时: observed一致率与崩塌率对表即得p值, 无需重跑本脚本")


if __name__ == '__main__':
    main()
