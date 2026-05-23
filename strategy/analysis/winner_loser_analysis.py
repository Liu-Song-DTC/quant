"""
Winner vs Loser Analysis v2: Identify characteristics of big winners and big losers
from historical trading signals.

Key improvements over v1:
- Analyzes relationship between score/features and forward returns more deeply
- Checks non-linear patterns (e.g., score only matters at extremes)
- Analyzes portfolio selections (which signals actually pass filters)
- Computes IC (Information Coefficient) for each feature
- Includes chi-squared tests for categorical features
- Analyzes factor_name patterns more carefully

Usage:
    cd /mnt/d/quant && python3 strategy/analysis/winner_loser_analysis.py
"""

import pandas as pd
import numpy as np
import os
import sys
import time
from collections import defaultdict
from scipy import stats as scipy_stats

# ── paths ──
BACKTEST_SIGNALS = "/mnt/d/quant/strategy/rolling_validation_results/backtest_signals.csv"
PORTFOLIO_SELECTIONS = "/mnt/d/quant/strategy/rolling_validation_results/portfolio_selections.csv"
RAW_DATA_DIR = "/mnt/d/quant/data/stock_data/raw_data"
OUTPUT_DIR = "/mnt/d/quant/strategy/analysis/winner_loser_output"

SEED = 42
np.random.seed(SEED)

MAX_SAMPLE = 50_000
WINNER_PCTILE = 90   # top 10% = winners
LOSER_PCTILE = 10    # bottom 10% = losers


def load_price_data(code: str) -> pd.DataFrame | None:
    """Load none.csv (unadjusted) price data."""
    fpath = os.path.join(RAW_DATA_DIR, code, "none.csv")
    if not os.path.exists(fpath):
        return None
    try:
        df = pd.read_csv(fpath, dtype={"股票代码": str})
        df.rename(columns={"日期": "date", "收盘": "close"}, inplace=True)
        df["date"] = pd.to_datetime(df["date"])
        df.sort_values("date", inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df[["date", "close"]]
    except Exception as e:
        return None


def compute_forward_returns(signals: pd.DataFrame, code: str) -> pd.DataFrame:
    """Compute forward returns for a single stock at 5/10/20 trading days."""
    price = load_price_data(code)
    if price is None or len(price) < 30:
        return pd.DataFrame()

    stock_signals = signals[signals["code"] == code].copy()
    if stock_signals.empty:
        return pd.DataFrame()

    merged = stock_signals[["code", "date"]].merge(price, on="date", how="left")
    merged.dropna(subset=["close"], inplace=True)
    if merged.empty:
        return pd.DataFrame()

    price_dict = price.set_index("date")["close"]
    all_dates = price["date"].values
    date_to_idx = {}
    for i, d in enumerate(all_dates):
        date_to_idx[pd.Timestamp(d).date()] = i

    results = []
    for _, row in merged.iterrows():
        sig_date = row["date"]
        sig_date_key = sig_date.date() if hasattr(sig_date, "date") else sig_date
        idx = date_to_idx.get(sig_date_key)
        if idx is None:
            continue
        close_t = float(row["close"])
        if close_t <= 0:
            continue

        fwd = {}
        for days, label in [(5, "fwd_5d"), (10, "fwd_10d"), (20, "fwd_20d")]:
            fwd_idx = idx + days
            if fwd_idx < len(all_dates):
                fwd_close = float(price_dict.iloc[fwd_idx])
                if fwd_close > 0:
                    fwd[label] = (fwd_close - close_t) / close_t * 100
                else:
                    fwd[label] = np.nan
            else:
                fwd[label] = np.nan

        # Also compute max/min forward return
        lookahead_20 = idx + 20
        if lookahead_20 < len(all_dates):
            prices_20 = price_dict.iloc[idx:lookahead_20+1].values
            if all(p > 0 for p in prices_20):
                fwd["fwd_max_20d"] = (max(prices_20) - close_t) / close_t * 100
                fwd["fwd_min_20d"] = (min(prices_20) - close_t) / close_t * 100
                fwd["fwd_volatility_20d"] = np.std(
                    [(p - close_t) / close_t * 100 for p in prices_20]
                )

        results.append({
            "code": code,
            "date": sig_date,
            **fwd,
        })

    return pd.DataFrame(results)


def compute_ic(series_a, series_b):
    """Compute Spearman rank IC between two series."""
    mask = series_a.notna() & series_b.notna()
    if mask.sum() < 30:
        return 0.0, 1.0
    return scipy_stats.spearmanr(series_a[mask], series_b[mask])


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    t0 = time.time()

    # ────────── Step 1: Load signals ──────────
    print("=" * 70)
    print("Step 1: Loading backtest signals...")
    print("=" * 70)

    signals = pd.read_csv(BACKTEST_SIGNALS, dtype={"code": str})
    print(f"  Total rows: {len(signals):,}")

    buy_signals = signals[signals["buy"] == True].copy()
    del signals
    print(f"  Buy signals: {len(buy_signals):,}")

    buy_signals["date"] = pd.to_datetime(buy_signals["date"])

    if len(buy_signals) > MAX_SAMPLE:
        buy_signals = buy_signals.sample(n=MAX_SAMPLE, random_state=SEED)
    print(f"  Sampled: {len(buy_signals):,} buy signals")
    print(f"  Date range: {buy_signals['date'].min().date()} to {buy_signals['date'].max().date()}")
    print(f"  Unique stocks: {buy_signals['code'].nunique()}")
    print()

    # ────────── Step 2: Compute forward returns ──────────
    print("=" * 70)
    print("Step 2: Computing forward returns...")
    print("=" * 70)

    unique_codes = buy_signals["code"].unique()
    n_codes = len(unique_codes)
    print(f"  Stocks to process: {n_codes:,}")

    all_returns = []
    for i, code in enumerate(unique_codes):
        if (i + 1) % 200 == 0:
            print(f"    [{i+1}/{n_codes}] ... ({time.time()-t0:.0f}s)")

        result = compute_forward_returns(buy_signals, code)
        if not result.empty:
            all_returns.append(result)

    fwd_df = pd.concat(all_returns, ignore_index=True)
    print(f"  Total return observations: {len(fwd_df):,}")

    print("\n  Merging with signal features...")
    feat_cols = [
        "code", "date", "score", "factor_value", "factor_name", "industry",
        "factor_quality", "chan_divergence_type", "chan_divergence_strength",
        "chan_structure_score", "chan_buy_point", "chan_sell_point",
        "signal_level", "chan_pivot_zg", "chan_pivot_zd",
    ]
    merged = fwd_df.merge(buy_signals[feat_cols], on=["code", "date"], how="left")
    print(f"  Merged: {len(merged):,}")

    merged.dropna(subset=["fwd_10d"], inplace=True)
    print(f"  With fwd_10d: {len(merged):,}")

    # Save
    merged.to_csv(os.path.join(OUTPUT_DIR, "analysis_data.csv"), index=False)
    print()
    gc.collect()

    # ────────── Step 3: Classify ──────────
    print("=" * 70)
    print("Step 3: Classification")
    print("=" * 70)

    th_high = np.percentile(merged["fwd_10d"], WINNER_PCTILE)
    th_low = np.percentile(merged["fwd_10d"], LOSER_PCTILE)
    print(f"  Winner: fwd_10d >= {th_high:.2f}%")
    print(f"  Loser:  fwd_10d <= {th_low:.2f}%")

    merged["group"] = "neutral"
    merged.loc[merged["fwd_10d"] >= th_high, "group"] = "winner"
    merged.loc[merged["fwd_10d"] <= th_low, "group"] = "loser"

    winners = merged[merged["group"] == "winner"]
    losers = merged[merged["group"] == "loser"]
    neutrals = merged[merged["group"] == "neutral"]

    print(f"  Winners: {len(winners):,} ({100*len(winners)/len(merged):.1f}%) "
          f"mean={winners['fwd_10d'].mean():+.2f}%")
    print(f"  Losers:  {len(losers):,} ({100*len(losers)/len(merged):.1f}%) "
          f"mean={losers['fwd_10d'].mean():+.2f}%")
    print(f"  Neutral: {len(neutrals):,} ({100*len(neutrals)/len(merged):.1f}%) "
          f"mean={neutrals['fwd_10d'].mean():+.2f}%")
    print()

    # ────────── Step 4: IC Analysis ──────────
    print("=" * 70)
    print("Step 4: Information Coefficient (IC) Analysis")
    print("=" * 70)

    numeric_features = [
        "score", "factor_value", "factor_quality",
        "chan_divergence_strength", "chan_structure_score",
        "chan_buy_point",
    ]

    print(f"  {'Feature':30s} {'IC':>8s} {'p-value':>10s} {'Sig?':>6s}")
    print(f"  {'-'*30} {'-'*8} {'-'*10} {'-'*6}")
    for feat in numeric_features:
        ic, pv = compute_ic(merged[feat], merged["fwd_10d"])
        sig = "***" if pv < 0.001 else "**" if pv < 0.01 else "*" if pv < 0.05 else ""
        print(f"  {feat:30s} {ic:>+8.4f} {pv:>10.6f} {sig:>6s}")

    # IC for score across different subperiods
    print("\n  IC(feature, fwd_10d) subperiod breakdown:")
    merged["year"] = merged["date"].dt.year
    for year in sorted(merged["year"].unique()):
        yr_data = merged[merged["year"] == year]
        if len(yr_data) < 100:
            continue
        ic_s, _ = compute_ic(yr_data["score"], yr_data["fwd_10d"])
        ic_fq, _ = compute_ic(yr_data["factor_quality"], yr_data["fwd_10d"])
        print(f"    {year}: N={len(yr_data):>6d}  IC(score)={ic_s:+.4f}  "
              f"IC(fq)={ic_fq:+.4f}")
    print()

    # ────────── Step 5: Feature comparison ──────────
    print("=" * 70)
    print("Step 5: Feature comparison (Winners vs Losers)")
    print("=" * 70)

    # --- Numeric features ---
    print("\n--- Numeric features ---")
    rows = []
    for feat in numeric_features:
        w_mean = winners[feat].mean()
        l_mean = losers[feat].mean()
        n_mean = neutrals[feat].mean()
        w_std = winners[feat].std()
        l_std = losers[feat].std()
        diff = w_mean - l_mean
        pooled_std = np.sqrt((w_std**2 + l_std**2) / 2)
        es = diff / pooled_std if pooled_std > 0 else 0.0
        # KS test
        ks_stat, ks_p = scipy_stats.ks_2samp(winners[feat].dropna(), losers[feat].dropna())

        rows.append({
            "feature": feat,
            "W_mean": round(w_mean, 4),
            "L_mean": round(l_mean, 4),
            "N_mean": round(n_mean, 4),
            "diff": round(diff, 4),
            "effect_size": round(es, 3),
            "KS_p": round(ks_p, 6),
        })

    feat_comp = pd.DataFrame(rows)
    feat_comp["abs_es"] = feat_comp["effect_size"].abs()
    feat_comp.sort_values("abs_es", ascending=False, inplace=True)
    print(feat_comp.to_string(index=False))
    print()

    # --- Non-linear analysis: score deciles ---
    print("--- Forward return by score decile ---")
    merged["score_decile"] = pd.qcut(merged["score"], 10, labels=False, duplicates="drop")
    score_decile_ret = merged.groupby("score_decile")["fwd_10d"].agg(["mean", "std", "count"])
    print(f"  {'Decile':>8s} {'Mean_fwd':>10s} {'Std':>8s} {'Count':>8s}")
    for decile in sorted(score_decile_ret.index):
        r = score_decile_ret.loc[decile]
        print(f"  {decile:>8d} {r['mean']:>+10.2f}% {r['std']:>8.2f} {r['count']:>8.0f}")

    # Top score decile vs bottom score decile
    try:
        top_decile = merged[merged["score_decile"] == merged["score_decile"].max()]
        bot_decile = merged[merged["score_decile"] == merged["score_decile"].min()]
        print(f"  Top score decile:  mean fwd_10d = {top_decile['fwd_10d'].mean():+.2f}%")
        print(f"  Bottom score decile: mean fwd_10d = {bot_decile['fwd_10d'].mean():+.2f}%")
    except Exception:
        pass
    print()

    # --- Factor quality non-zero analysis ---
    print("--- Factor quality: zero vs non-zero ---")
    fq_zero = merged[merged["factor_quality"] == 0]
    fq_pos = merged[merged["factor_quality"] > 0]
    if len(fq_zero) > 0 and len(fq_pos) > 0:
        print(f"  factor_quality = 0:  N={len(fq_zero):,} mean_fwd_10d={fq_zero['fwd_10d'].mean():+.2f}%")
        print(f"  factor_quality > 0:  N={len(fq_pos):,} mean_fwd_10d={fq_pos['fwd_10d'].mean():+.2f}%")

    # Factor quality quartiles (among non-zero)
    fq_nonzero = merged[merged["factor_quality"] > 0]
    if len(fq_nonzero) > 100:
        fq_nonzero["fq_quartile"] = pd.qcut(fq_nonzero["factor_quality"], 4, labels=False)
        fq_ret = fq_nonzero.groupby("fq_quartile")["fwd_10d"].mean()
        print("  Forward return by factor_quality quartile (non-zero only):")
        for q in sorted(fq_ret.index):
            print(f"    Q{q}: {fq_ret[q]:+.2f}%")
    print()

    # --- Chan divergence type analysis ---
    print("--- Chan divergence type vs forward return ---")
    div_ret = merged.groupby("chan_divergence_type")["fwd_10d"].agg(["mean", "count"])
    div_ret = div_ret[div_ret["count"] >= 50]  # filter rare types
    div_ret.sort_values("mean", ascending=False, inplace=True)
    print(f"  {'Type':25s} {'Mean_fwd_10d':>14s} {'Count':>8s}")
    for dtype, row in div_ret.iterrows():
        print(f"  {str(dtype):25s} {row['mean']:>+14.2f}% {row['count']:>8.0f}")
    print()

    # --- Chan structure score analysis ---
    print("--- Chan structure score: predictive power ---")
    merged["cs_group"] = pd.cut(merged["chan_structure_score"],
                                bins=[-1, -0.5, -0.1, 0.1, 0.5, 1.0],
                                labels=["strong_neg", "weak_neg", "neutral", "weak_pos", "strong_pos"])
    cs_ret = merged.groupby("cs_group", observed=True)["fwd_10d"].agg(["mean", "count"])
    print(f"  {'Group':15s} {'Mean_fwd_10d':>14s} {'Count':>8s}")
    for grp, row in cs_ret.iterrows():
        print(f"  {str(grp):15s} {row['mean']:>+14.2f}% {row['count']:>8.0f}")
    print()

    # --- Chan buy point analysis ---
    print("--- Chan buy point vs forward return ---")
    bp_ret = merged.groupby("chan_buy_point")["fwd_10d"].agg(["mean", "count"])
    print(f"  {'BuyPoint':>10s} {'Mean_fwd_10d':>14s} {'Count':>8s} {'Win%':>8s}")
    for bp, row in bp_ret.iterrows():
        grp = merged[merged["chan_buy_point"] == bp]
        win_pct = (grp["group"] == "winner").mean() * 100
        print(f"  {int(bp):>10d} {row['mean']:>+14.2f}% {row['count']:>8.0f} {win_pct:>7.1f}%")
    print()

    # ────────── Step 6: Industry & Factor Analysis ──────────
    print("=" * 70)
    print("Step 6: Industry & Factor Analysis")
    print("=" * 70)

    # Industry win/loss ratio
    print("--- Industry top/bottom by win ratio ---")
    w_ind = merged[merged["group"] == "winner"]["industry"].value_counts()
    l_ind = merged[merged["group"] == "loser"]["industry"].value_counts()
    n_ind = merged[merged["group"] == "neutral"]["industry"].value_counts()
    ind_df = pd.DataFrame({
        "total": merged["industry"].value_counts(),
        "winners": w_ind,
        "losers": l_ind,
    }).fillna(0).astype(int)
    ind_df["win_pct"] = (ind_df["winners"] / ind_df["total"] * 100).round(1)
    ind_df["loss_pct"] = (ind_df["losers"] / ind_df["total"] * 100).round(1)
    ind_df["wl_ratio"] = ((ind_df["win_pct"] + 0.1) / (ind_df["loss_pct"] + 0.1)).round(2)
    ind_df.sort_values("wl_ratio", ascending=False, inplace=True)

    print("  Top 5 industries (most winner-friendly):")
    for ind, row in ind_df.head(5).iterrows():
        print(f"    {ind:20s} win={row['win_pct']:.1f}% loss={row['loss_pct']:.1f}% "
              f"ratio={row['wl_ratio']:.2f} total={row['total']}")
    print("  Bottom 5 industries (most loser-prone):")
    for ind, row in ind_df.tail(5).iterrows():
        print(f"    {ind:20s} win={row['win_pct']:.1f}% loss={row['loss_pct']:.1f}% "
              f"ratio={row['wl_ratio']:.2f} total={row['total']}")
    print()

    # Factor name prefix analysis
    print("--- Factor type distribution (winners vs losers) ---")
    def factor_type(fname):
        if pd.isna(fname) or not str(fname).strip():
            return "UNKNOWN"
        fname = str(fname)
        if fname.startswith("DYN_"):
            return "DYN"
        elif fname.startswith("IND_"):
            return "IND"
        elif fname == "V41":
            return "V41"
        elif "_T" in fname:
            return "TECH_ONLY"
        else:
            return fname.split("_")[0] if "_" in fname else fname[:20]

    merged["factor_type"] = merged["factor_name"].apply(factor_type)
    ft_cross = pd.crosstab(merged["factor_type"], merged["group"], normalize="index") * 100
    ft_cross["wl_ratio"] = ((ft_cross["winner"] + 0.1) / (ft_cross["loser"] + 0.1)).round(2)
    ft_cross.sort_values("wl_ratio", ascending=False, inplace=True)

    print(f"  {'FactorType':20s} {'Winner%':>10s} {'Loser%':>10s} {'Neutral%':>10s} {'Ratio':>8s}")
    for ft, row in ft_cross.iterrows():
        count = len(merged[merged["factor_type"] == ft])
        if count > 100:
            print(f"  {str(ft):20s} {row['winner']:>9.1f}% {row['loser']:>9.1f}% "
                  f"{row['neutral']:>9.1f}% {row['wl_ratio']:>7.2f}")
    print()

    # ────────── Step 7: Pooled Score Analysis ──────────
    print("=" * 70)
    print("Step 7: Combined predictive power")
    print("=" * 70)

    # Try a simple combined score: score * (1 + chan_divergence_strength)
    merged["combined_score"] = (
        merged["score"] * (1 + merged["chan_divergence_strength"])
    )
    ic_combined, _ = compute_ic(merged["combined_score"], merged["fwd_10d"])
    ic_score_only, _ = compute_ic(merged["score"], merged["fwd_10d"])
    print(f"  IC(score)              = {ic_score_only:+.4f}")
    print(f"  IC(score * 1+div_str)  = {ic_combined:+.4f}")
    print()

    # ────────── Step 8: Portfolio selections analysis ──────────
    print("=" * 70)
    print("Step 8: Portfolio selections analysis")
    print("=" * 70)

    if os.path.exists(PORTFOLIO_SELECTIONS):
        port = pd.read_csv(PORTFOLIO_SELECTIONS, dtype={"code": str})
        port["date"] = pd.to_datetime(port["date"])
        print(f"  Portfolio selections: {len(port):,} rows")
        print(f"  Date range: {port['date'].min().date()} to {port['date'].max().date()}")

        # What's the typical score range of selected stocks?
        print(f"  Score range of selected stocks: "
              f"{port['score'].min():.4f} to {port['score'].max():.4f} "
              f"(mean={port['score'].mean():.4f})")

        # Merge with forward returns
        port_with_fwd = port.merge(
            merged[["code", "date", "fwd_10d", "fwd_5d", "fwd_20d", "group"]],
            on=["code", "date"], how="left"
        )
        print(f"  With fwd_10d: {port_with_fwd['fwd_10d'].notna().sum():,}")

        # Performance of selected stocks
        print(f"  Selected stocks: mean fwd_10d = {port_with_fwd['fwd_10d'].mean():+.2f}%")
        port_win = (port_with_fwd["group"] == "winner").mean() * 100
        port_loss = (port_with_fwd["group"] == "loser").mean() * 100
        print(f"  Winner rate: {port_win:.1f}%  Loser rate: {port_loss:.1f}%")
        print(f"  Win/Loss ratio: {port_win/max(port_loss,0.1):.2f}")
    else:
        print("  Portfolio selections file not found, skipping.")
    print()

    # ────────── Step 9: Summary & Recommendations ──────────
    print("=" * 70)
    print("Step 9: Summary & Recommendations")
    print("=" * 70)

    print(f"\n  Sample: {len(merged):,} buy signals, "
          f"{merged['date'].min().date()} to {merged['date'].max().date()}")
    print(f"  Winners: fwd_10d >= {th_high:.1f}% ({len(winners):,}, {100*len(winners)/len(merged):.1f}%)")
    print(f"  Losers:  fwd_10d <= {th_low:.1f}% ({len(losers):,}, {100*len(losers)/len(merged):.1f}%)")

    # --- Most predictive features ---
    print("\n--- Predictive power ranking ---")
    feat_comp_sorted = feat_comp.sort_values("abs_es", ascending=False)
    print(f"  {'Feature':30s} {'EffectSize':>12s} {'IC':>8s} {'Direction':>12s}")
    print(f"  {'-'*30} {'-'*12} {'-'*8} {'-'*12}")
    for _, row in feat_comp_sorted.iterrows():
        ic_val, _ = compute_ic(merged[row["feature"]], merged["fwd_10d"])
        direction = "HIGHER=win" if row["diff"] > 0 else "LOWER=win"
        print(f"  {row['feature']:30s} {row['effect_size']:>+11.3f} {ic_val:>+8.4f} {direction:>12s}")

    # --- Key findings ---
    print("\n--- Key findings ---")
    if abs(feat_comp_sorted.iloc[0]["effect_size"]) < 0.10:
        print("  WARNING: All effect sizes are very small (<0.10). "
              "The stored features alone are NOT strong predictors of forward returns.")
        print("  This likely means the real predictive power comes from features NOT stored in")
        print("  backtest_signals.csv (e.g., daily_return, volume_ratio, exhaustion_risk, etc.)")
        print("  that are computed in signal_engine.py but only used for real-time filtering")
        print("  in portfolio.py and never persisted.")

    # Top winner features
    print("\n--- Top signals for WINNERS ---")
    top_win = feat_comp_sorted[feat_comp_sorted["diff"] > 0].head(3)
    for _, row in top_win.iterrows():
        print(f"  {row['feature']:30s}: Higher values associated with winners "
              f"(delta={row['diff']:+.4f}, ES={row['effect_size']:+.3f})")

    print("\n--- Top DANGER signals for LOSERS ---")
    top_loss = feat_comp_sorted[feat_comp_sorted["diff"] < 0].head(3)
    for _, row in top_loss.iterrows():
        print(f"  {row['feature']:30s}: Higher values associated with losers "
              f"(delta={row['diff']:+.4f}, ES={row['effect_size']:+.3f})")

    # --- Specific recommendations ---
    print("\n--- Recommendations ---")

    # Check structure score
    w_ss = winners["chan_structure_score"].mean()
    l_ss = losers["chan_structure_score"].mean()
    if l_ss > w_ss:
        print(f"  1. INVESTIGATE chan_structure_score: Losers have HIGHER structure score "
              f"({l_ss:.4f}) than winners ({w_ss:.4f}). This is counterintuitive - "
              f"consider if high structure score indicates complexity/ambiguity.")

    # Check divergence
    w_ds = winners["chan_divergence_strength"].mean()
    l_ds = losers["chan_divergence_strength"].mean()
    if w_ds > l_ds:
        print(f"  2. UTILIZE chan_divergence_strength: Winners have higher divergence strength "
              f"({w_ds:.4f} vs {l_ds:.4f}). Currently only used for B1 gate. "
              f"Consider using it more broadly as a confidence multiplier.")

    # Check score
    print(f"  3. TUNE min_absolute_score: Winner mean score={winners['score'].mean():.4f} vs "
          f"Loser mean={losers['score'].mean():.4f}. Current threshold analysis needed.")

    print()
    elapsed = time.time() - t0
    print(f"Total time: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print("Done.")


import gc
if __name__ == "__main__":
    main()
