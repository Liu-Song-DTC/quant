"""填充检测探针 (2026-10-26): 验证"截面z-score==0 ⇒ 该格被中性化填成均值"的识别力。

问题: 均值填充把"池外掩码/自然缺失"的格子填成当日截面均值 ⇒ 泄漏(成员资格)进ML。
想做"缺失指示位"臂, 但 ML 层拿不到填充掩码, 只能从数值反推:
    中性化填的值 = 当日该列均值 ⇒ 之后 ML 的截面 z-score 恰好 = 0。
本探针用真实数据验证这个反推的精确度(precision/recall), 分列 + 按行(全填充行)。

数据: base 臂的 raw 因子缓存(掩码前/中性化前) + pool_membership 缓存
流程: 施加真实池掩码 → 真实 neutralize_factor_df → 真实 ML _cross_sectional_zscore
      → 与 ground truth(输入 NaN ⇒ 被填)逐格比
用法: .venv/bin/python .arms/probe/fill_detect_probe_1026.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROD = Path('/mnt/d/quant')
sys.path.insert(0, str(PROD / 'strategy'))

from core.factor_neutralizer import neutralize_factor_df  # noqa: E402

BASE = PROD / '.review_runs/arm_base_1025/strategy/cache'
RAW_PARQUET = sorted(BASE.glob('factor_df_*.parquet'))[0]
POOL_PARQUET = sorted(BASE.glob('pool_membership_*.parquet'))[0]
N_DATES = 3
COLS = 40

IDX_CODES = {'sh000001', 'sh000852', '000001', '399006'}


def load():
    df = pd.read_parquet(RAW_PARQUET)
    print(f'[load] raw 因子 {df.shape}, 日期 {df["date"].nunique()}')
    memb = pd.read_parquet(POOL_PARQUET)
    mmap = {pd.Timestamp(b): set(g["code"].astype(str))
            for b, g in memb.groupby('boundary')}
    return df, mmap


def apply_mask(df, mmap):
    """复刻 factor_preparer 的日历池掩码(0f-v3)。"""
    boundaries = sorted(mmap.keys())
    b_arr = pd.DatetimeIndex(boundaries)
    dates = pd.to_datetime(df['date'])
    b_idx = np.searchsorted(b_arr.values, dates.values, side='right') - 1
    mask_cols = [c for c in df.columns
                 if c not in ('code', 'date', 'industry', 'future_ret', 'label_end_date')]
    out = df.copy()
    for bi, b in enumerate(boundaries):
        rows = b_idx == bi
        if not rows.any():
            continue
        codes = out['code'].values[rows].astype(str)
        non_member = (~np.isin(codes, list(mmap[b]))
                      & ~np.isin(codes, list(IDX_CODES)))
        if non_member.any():
            rmask = rows.copy()
            rmask[rows] = non_member
            out.loc[rmask, mask_cols] = np.nan
    pre = b_idx < 0
    if pre.any():
        pre_idx = pre & ~out['code'].isin(IDX_CODES).values
        out.loc[pre_idx, mask_cols] = np.nan
    return out


def main():
    df, mmap = load()
    mask_cols = [c for c in df.columns
                 if c not in ('code', 'date', 'industry', 'future_ret', 'label_end_date')]
    masked = apply_mask(df, mmap)
    del df

    # 抽样日期: 取样本最多的几天(稳定复现)
    dates = sorted(pd.to_datetime(masked['date']).unique())
    step = max(1, len(dates) // N_DATES)
    sample_dates = [pd.Timestamp(dates[i]) for i in range(0, len(dates), step)][:N_DATES]
    print(f'[sample] 日期 {[str(d.date()) for d in sample_dates]}')

    sub = masked[pd.to_datetime(masked['date']).isin(sample_dates)].copy()
    del masked
    sub = sub.sort_values(['date', 'code'], kind='stable').reset_index(drop=True)
    print(f'[sample] 行数 {len(sub)}, 列数 {len(sub.columns)}')

    cols = [c for c in mask_cols if pd.api.types.is_float_dtype(sub[c])][:COLS]
    print(f'[sample] 参与中性化的列 {len(cols)}: {cols[:8]} ...')

    # ground truth: 输入非有限 ⇒ 被填(但 valid.sum()<10 时原样返回)
    raw_in = sub[cols].copy()
    gt_filled = ~np.isfinite(raw_in.to_numpy(dtype='float64'))
    print(f'[gt] 输入缺失格 {gt_filled.sum()} / {gt_filled.size} '
          f'({gt_filled.mean() * 100:.2f}%)')

    neu = neutralize_factor_df(sub, cols, industry_col='industry',
                               market_cap_col='ln_cap',
                               market_cap_transform='compressed_log',
                               date_col='date', neutralize_industry=True,
                               min_samples_per_industry=3)
    # 中性化后的值在 {col}_neu (factor_preparer 用它覆盖原列)
    neu_cols = [f'{c}_neu' for c in cols]
    neu_vals = neu[neu_cols].to_numpy(dtype='float64')
    print(f'[neu] 中性化后仍缺失格 {(~np.isfinite(neu_vals)).sum()} (填充生效则应=0)')

    # ML 的截面 z-score (真实实现同款: pandas groupby transform)
    zdf = neu[neu_cols].copy()
    zdf.columns = cols
    for c in cols:
        g = zdf.groupby(neu['date'])[c]
        mean_s = g.transform('mean')
        std_s = g.transform('std').replace(0, 1.0)
        zdf[c] = (zdf[c] - mean_s) / std_s
    z = zdf.to_numpy(dtype='float64')
    print(f'[z] z-score dtype={zdf[cols].dtypes.iloc[0]}')

    # 逐列: 被填格的 |z| 分布 与 自然格的 |z| 分布
    rows = []
    for j, c in enumerate(cols):
        zj = z[:, j]
        filled = gt_filled[:, j]
        if filled.sum() == 0:
            continue
        zf = np.abs(zj[filled])
        zn = np.abs(zj[~filled])
        zn = zn[np.isfinite(zn)]
        rows.append({
            'col': c,
            'n_filled': int(filled.sum()),
            'filled_frac': filled.mean(),
            'zf_max': np.nanmax(zf) if len(zf) else np.nan,
            'zf_p99': np.nanpercentile(zf, 99) if len(zf) else np.nan,
            'zn_min': np.nanmin(zn) if len(zn) else np.nan,
            'zn_p001': np.nanpercentile(zn, 0.1) if len(zn) else np.nan,
        })
    tab = pd.DataFrame(rows).sort_values('n_filled', ascending=False)
    pd.set_option('display.width', 200)
    print('\n=== 被填格 |z| vs 自然格 |z| (前20列) ===')
    print(tab.head(20).to_string(index=False))
    print(f'\n全部列: 被填格最大|z| = {tab["zf_max"].max():.3e}, '
          f'自然格最小|z| = {tab["zn_min"].min():.3e}')

    # 判定阈值扫描: precision/recall (按格)
    print('\n=== 阈值扫描 (按格) ===')
    for thr in (1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3):
        pred = np.isfinite(z) & (np.abs(z) < thr)
        tp = int((pred & gt_filled).sum())
        fp = int((pred & ~gt_filled).sum())
        fn = int((~pred & gt_filled).sum())
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        print(f'  thr={thr:.0e}: TP={tp:>9,} FP={fp:>9,} FN={fn:>9,} '
              f'precision={prec * 100:6.3f}% recall={rec * 100:6.3f}%')

    # 行级: "全部特征都被填" = 池外掩码行
    print('\n=== 行级检测 (全列都被填 ⇒ 池外掩码行) ===')
    allfilled_gt = gt_filled.all(axis=1)
    print(f'  ground truth 全填行: {int(allfilled_gt.sum())} / {len(sub)} '
          f'({allfilled_gt.mean() * 100:.2f}%)')
    for thr in (1e-9, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3):
        pred_all = (np.isfinite(z) & (np.abs(z) < thr)).all(axis=1)
        tp = int((pred_all & allfilled_gt).sum())
        fp = int((pred_all & ~allfilled_gt).sum())
        fn = int((~pred_all & allfilled_gt).sum())
        print(f'  thr={thr:.0e}: TP={tp:>7,} FP={fp:>7,} FN={fn:>7,} '
              f'precision={tp / max(tp + fp, 1) * 100:6.3f}% '
              f'recall={tp / max(tp + fn, 1) * 100:6.3f}%')

    # 只有部分列被填的行(自然缺失)会不会被误判成池外行
    print('\n=== 部分填充行(自然缺失)规模 ===')
    part = (~allfilled_gt) & gt_filled.any(axis=1)
    print(f'  部分填行: {int(part.sum())} ({part.mean() * 100:.2f}%), '
          f'平均被填列数 {gt_filled[part].sum(axis=1).mean() if part.any() else 0:.1f}')


if __name__ == '__main__':
    main()
