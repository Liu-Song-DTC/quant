#!/usr/bin/env python3
"""10-20夜 navlaw 差异归因: 今晚基线 vs fix365臂 的 10 笔选股换血是哪列变的?

已知: 选股差 = {300247,603050,300619,301216,600183}(fix有) vs
{688479,002648,301012,300333,002142}(今晚有), 集中在 2024-07-11 与 2026-09-18;
净值 548/1401 天差>1e-12 (≤~1%), 首=2024-07-12。
候选通道: ML全量重训(概念map mtime变→缓存失效) / sina_factors 326重写 /
concept_map churn 3631格 / 尾部chunk新数据。
方法: 两份3.2GB信号CSV 单遍扫描, 抽 10 个(code,date)键 ± 邻近5行,
逐列对比 score/ml_score/factor_name/buy/chan列。
"""
import pandas as pd

P_NEW = '/mnt/d/quant/strategy/rolling_validation_results/backtest_signals.csv'
P_OLD = '/mnt/d/quant/.arms/armout/fix365_1020/backtest_signals.csv'
KEYS = {  # code -> date (换血日)
    '300247': '2024-07-11', '603050': '2026-09-18', '300619': '2026-09-18',
    '301216': '2026-09-18', '600183': '2026-09-18',
    '688479': '2024-07-11', '002648': '2026-09-18', '301012': '2026-09-18',
    '300333': '2026-09-18', '002142': '2026-09-18',
}
NB = 3          # 邻近行数(同code前后各3行)
COLS = None     # 全列读, 看哪列变


def scan(path, want):
    """want: {code: date}; 返回 dict[(code,date)] = row(Series)。单遍分块。"""
    out = {}
    codes = set(want)
    for ch in pd.read_csv(path, chunksize=2_000_000, dtype={'code': str}):
        m = ch['code'].isin(codes)
        if not m.any():
            continue
        sub = ch[m]
        for _, r in sub.iterrows():
            out[(r['code'], str(r['date'])[:10])] = r
    return out


print('扫描两份CSV (每份~3.2GB, 单遍分块)…', flush=True)
new = scan(P_NEW, KEYS)
old = scan(P_OLD, KEYS)
print(f'抽到: 今晚 {len(new)} 行 / fix {len(old)} 行\n')

# 每个code: 换血日 ± 邻近行, 逐列diff
allcols = list(next(iter(new.values())).index)
skip = {'code', 'date'}
for c, d in KEYS.items():
    print(f"══ {c} @ {d} ══")
    for off in range(-NB, NB + 1):
        dt = (pd.Timestamp(d) + pd.offsets.BDay(off)).strftime('%Y-%m-%d')
        kn, ko = (c, dt) in new, (c, dt) in old
        if not kn and not ko:
            continue
        if kn and not ko:
            print(f"  {dt}: 今晚有 fix无(新行)"); continue
        if ko and not kn:
            print(f"  {dt}: fix有 今晚无(消失)"); continue
        rn, ro = new[(c, dt)], old[(c, dt)]
        diffs = [col for col in allcols
                 if col not in skip and str(rn[col]) != str(ro[col])]
        if not diffs:
            print(f"  {dt}: 恒等")
        else:
            print(f"  {dt}: {len(diffs)}列变: " +
                  ", ".join(f"{x}({ro[x] if pd.notna(ro[x]) else 'NaN'}→"
                            f"{rn[x] if pd.notna(rn[x]) else 'NaN'})"
                            for x in diffs[:8]) +
                  ("…" if len(diffs) > 8 else ""))
    print()
