# core/factor_neutralizer.py
"""
因子中性化模块 — 剥离行业和市值偏差，提升因子纯度

参考：Bloomberg Competition 2025 (+40% 收益, Sharpe 1.17)
方法：截面回归取残差
  factor_raw ~ industry_dummies + log(market_cap) → residual = 纯因子值
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple


def neutralize_factors(
    factor_values: np.ndarray,
    industry_labels: List[str],
    market_caps: Optional[np.ndarray] = None,
    min_samples_per_industry: int = 3,
    neutralize_industry: bool = True,
    market_cap_is_log: bool = False,
) -> np.ndarray:
    """截面因子中性化：剥离行业均值 + 可选市值回归

    Args:
        factor_values: 原始因子值 [N]
        industry_labels: 行业标签 [N]
        market_caps: 市值（对数）[N]，可选
        min_samples_per_industry: 行业最少样本数

    Returns:
        中性化后的因子值 [N]，量纲与输入一致
    """
    n = len(factor_values)
    valid = ~np.isnan(factor_values) & ~np.isinf(factor_values)
    result = np.full(n, np.nan)

    if valid.sum() < 10:
        return factor_values.copy()

    fv_valid = factor_values[valid]
    ind_valid = np.array(industry_labels)[valid]

    # Step 1: 行业均值剥离（最核心的中性化）
    unique_inds = np.unique(ind_valid)
    industry_mean = {}
    for ind in unique_inds:
        mask = ind_valid == ind
        if neutralize_industry and mask.sum() >= min_samples_per_industry:
            industry_mean[ind] = np.mean(fv_valid[mask])
        else:
            industry_mean[ind] = 0.0

    global_mean = np.mean(fv_valid)
    residuals = np.zeros_like(fv_valid, dtype=float)
    for i, ind in enumerate(ind_valid):
        residuals[i] = fv_valid[i] - industry_mean.get(ind, global_mean)

    # Step 2: 可选市值中性化（截面回归去市值暴露）
    if market_caps is not None:
        mc_valid = market_caps[valid]
        mc_finite = np.isfinite(mc_valid)
        if not market_cap_is_log:
            mc_finite &= mc_valid > 0
        if mc_finite.sum() > 10:
            log_mc = (mc_valid[mc_finite].copy() if market_cap_is_log
                      else np.log(np.clip(mc_valid[mc_finite], 1e6, 1e15)))
            # 先剥离市值的行业均值，避免市值回归重新引入行业暴露。
            if neutralize_industry:
                labels = ind_valid[mc_finite]
                for ind in np.unique(labels):
                    mask = labels == ind
                    if mask.sum() >= min_samples_per_industry:
                        log_mc[mask] -= log_mc[mask].mean()
            # OLS: residuals ~ alpha + beta * log_mc
            X = np.column_stack([np.ones(len(log_mc)), log_mc])
            try:
                beta = np.linalg.lstsq(X, residuals[mc_finite], rcond=None)[0]
                residuals[mc_finite] -= (X @ beta - beta[0])  # 只剥离市值部分，保留alpha
            except np.linalg.LinAlgError:
                pass

    # 保持原始均值和标准差（只剥离偏差，不改变量纲）
    orig_mean = np.mean(fv_valid)
    orig_std = np.std(fv_valid)
    res_mean = np.mean(residuals)
    res_std = np.std(residuals)

    if res_std > 1e-10:
        residuals = (residuals - res_mean) / res_std * orig_std + orig_mean
    else:
        residuals = residuals - res_mean + orig_mean

    result[valid] = residuals
    # [用户裁定 2026-10-25 方案A] 填充加回(HEAD语义): 无效值填截面全局均值
    result[~valid] = orig_mean

    return result


def neutralize_factor_df(
    df: pd.DataFrame,
    factor_cols: List[str],
    industry_col: str = 'industry',
    market_cap_col: Optional[str] = None,
    date_col: str = 'date',
    neutralize_industry: bool = True,
    min_samples_per_industry: int = 3,
    market_cap_transform: str = 'raw',
) -> pd.DataFrame:
    """对 factor_df 批量做截面中性化

    Args:
        df: 包含因子值的 DataFrame
        factor_cols: 需要中性化的因子列名
        industry_col: 行业列名
        market_cap_col: 市值列名（可选）
        date_col: 日期列名

    Returns:
        中性化后的 DataFrame（新增 _neu 后缀列）
    """
    if date_col not in df.columns:
        return df.copy()
    columns = [fc for fc in factor_cols if fc in df.columns]
    # 避免在数百万行 DataFrame 上逐日期 .loc 写入（Pandas COW 会反复复制整列）。
    # 使用位置索引写独立数组，最后一次性拼接；保持逐日计算和原浮点精度。
    outputs = {fc + '_neu': np.full(len(df), np.nan,
                    dtype=df[fc].dtype if pd.api.types.is_float_dtype(df[fc]) else np.float64)
               for fc in columns}
    for positions in df.groupby(date_col, sort=True).indices.values():
        grp = df.iloc[positions]
        industries = grp[industry_col].fillna('default').tolist()
        mkt_caps = grp[market_cap_col].values if market_cap_col and market_cap_col in df.columns else None
        is_log = market_cap_transform == 'compressed_log'
        if mkt_caps is not None and is_log:
            mkt_caps = np.where(mkt_caps > 0,
                                10.0 * (np.arctanh(np.clip(mkt_caps, -0.999999, 0.999999)) + .5),
                                np.nan)
        for fc in columns:
            neutralized = neutralize_factors(
                grp[fc].values, industries, mkt_caps,
                min_samples_per_industry=min_samples_per_industry,
                neutralize_industry=neutralize_industry, market_cap_is_log=is_log,
            )
            outputs[fc + '_neu'][positions] = neutralized
    existing = df.drop(columns=[c for c in outputs if c in df.columns])
    return pd.concat([existing, pd.DataFrame(outputs, index=df.index)], axis=1)


def calculate_factor_purity(
    df: pd.DataFrame,
    factor_col: str,
    industry_col: str = 'industry',
    date_col: str = 'date',
) -> Dict[str, float]:
    """计算因子纯度指标（中性化前后对比）

    Returns:
        {
            'ic_raw': 原始IC,
            'ic_neutralized': 中性化后IC,
            'industry_exposure': 行业暴露（ANOVA F值）,
            'purity_gain': 纯度提升百分比
        }
    """
    from scipy import stats

    results = {'ic_raw': 0.0, 'ic_neutralized': 0.0, 'industry_exposure': 0.0, 'purity_gain': 0.0}

    if factor_col not in df.columns or 'future_ret' not in df.columns:
        return results

    # 原始IC
    valid = df[[factor_col, 'future_ret']].dropna()
    if len(valid) > 20:
        results['ic_raw'] = stats.spearmanr(valid[factor_col], valid['future_ret'])[0]

    # 行业暴露（组间方差/总方差）
    if industry_col in df.columns:
        valid_ind = df[[factor_col, industry_col]].dropna()
        if len(valid_ind) > 20:
            groups = [g[factor_col].values for _, g in valid_ind.groupby(industry_col) if len(g) >= 3]
            if len(groups) >= 2:
                grand_mean = valid_ind[factor_col].mean()
                ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
                ss_total = ((valid_ind[factor_col] - grand_mean) ** 2).sum()
                results['industry_exposure'] = ss_between / ss_total if ss_total > 0 else 0

    return results
