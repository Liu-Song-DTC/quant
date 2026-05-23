"""
Enhanced Winner vs Loser Analysis: Compute 30+ pre-signal features from raw price data
to identify what truly predicts big winners and big losers.

Adds features NOT stored in backtest_signals.csv:
- Pre-signal momentum, volume patterns, volatility regime
- Turnover characteristics, gap patterns, relative strength
- Price position vs moving averages, max drawdown
- Consecutive up/down days, amplitude patterns
- Up/down capture ratios, return skewness

Usage:
    cd /mnt/d/quant && python3 strategy/analysis/enhanced_winner_loser.py
"""

import pandas as pd
import numpy as np
import os
import sys
import time
import gc
import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy import stats as scipy_stats

warnings.filterwarnings("ignore")

# ── paths ──
BACKTEST_SIGNALS = "/mnt/d/quant/strategy/rolling_validation_results/backtest_signals.csv"
RAW_DATA_DIR = "/mnt/d/quant/data/stock_data/raw_data"
INDEX_DATA = "/mnt/d/quant/data/stock_data/raw_data/sh000001/none.csv"
OUTPUT_DIR = "/mnt/d/quant/strategy/analysis/winner_loser_output"

SEED = 42
np.random.seed(SEED)

MAX_SAMPLE = 50000
WINNER_PCTILE = 90
LOSER_PCTILE = 10

# ── feature computation ──

def compute_pre_signal_features(price_df: pd.DataFrame, signal_dates: list) -> pd.DataFrame:
    """Compute 30+ pre-signal features from price data for given signal dates."""
    if price_df is None or len(price_df) < 80:
        return pd.DataFrame()

    df = price_df.copy()
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)

    close = df["close"].values.astype(float)
    volume = df["volume"].values.astype(float) if "volume" in df.columns else None
    amount = df["amount"].values.astype(float) if "amount" in df.columns else None
    turnover = df["换手率"].values.astype(float) if "换手率" in df.columns else None
    amplitude = df["振幅"].values.astype(float) if "振幅" in df.columns else None
    change_pct = df["涨跌幅"].values.astype(float) if "涨跌幅" in df.columns else None

    if change_pct is None:
        change_pct = np.diff(close, prepend=close[0]) / np.where(close > 0, close, np.nan) * 100
        change_pct[0] = 0

    n = len(close)
    date_arr = df["date"].values

    results = []
    for sig_date in signal_dates:
        # find index of signal date
        matches = np.where(date_arr == pd.Timestamp(sig_date))[0]
        if len(matches) == 0:
            continue
        idx = matches[0]
        if idx < 60:  # need at least 60 days of history
            continue

        feats = {}

        # ── Price data slice up to signal date (exclusive) ──
        c_hist = close[:idx+1]
        v_hist = volume[:idx+1] if volume is not None else None
        t_hist = turnover[:idx+1] if turnover is not None else None
        a_hist = amplitude[:idx+1] if amplitude is not None else None
        r_hist = change_pct[:idx+1]

        c_now = c_hist[-1]

        # ── 1. Pre-signal momentum ──
        for lookback, label in [(5, "mom_5d"), (10, "mom_10d"), (20, "mom_20d"), (60, "mom_60d")]:
            if idx >= lookback and c_hist[-lookback-1] > 0:
                feats[label] = (c_now - c_hist[-lookback-1]) / c_hist[-lookback-1] * 100
            else:
                feats[label] = np.nan

        # ── 2. Recent return acceleration (mom_5d - mom_20d) ──
        if not np.isnan(feats.get("mom_5d", np.nan)) and not np.isnan(feats.get("mom_20d", np.nan)):
            feats["mom_accel"] = feats["mom_5d"] - feats["mom_20d"]
        else:
            feats["mom_accel"] = np.nan

        # ── 3. Volume ratio (current volume vs recent avg) ──
        if v_hist is not None and len(v_hist) >= 6:
            feats["vol_ratio_5d"] = v_hist[-1] / max(v_hist[-6:-1].mean(), 1)
            feats["vol_ratio_20d"] = v_hist[-1] / max(v_hist[-21:-1].mean(), 1)
            # volume trend: is volume increasing?
            feats["vol_trend_5d"] = v_hist[-6:-1].mean() / max(v_hist[-11:-6].mean(), 1) if len(v_hist) >= 11 else np.nan
        else:
            feats["vol_ratio_5d"] = feats["vol_ratio_20d"] = feats["vol_trend_5d"] = np.nan

        # ── 4. Turnover features ──
        if t_hist is not None and len(t_hist) >= 6:
            feats["turnover"] = t_hist[-1]
            feats["turnover_5d_avg"] = np.mean(t_hist[-6:-1])
            feats["turnover_20d_avg"] = np.mean(t_hist[-21:-1]) if len(t_hist) >= 21 else np.mean(t_hist[-6:-1])
            feats["turnover_ratio"] = t_hist[-1] / max(feats["turnover_20d_avg"], 0.01)
        else:
            feats["turnover"] = feats["turnover_5d_avg"] = feats["turnover_20d_avg"] = feats["turnover_ratio"] = np.nan

        # ── 5. Realized volatility ──
        for lb, label in [(5, "vol_5d"), (10, "vol_10d"), (20, "vol_20d")]:
            if idx >= lb:
                feats[label] = np.std(r_hist[-lb:])
            else:
                feats[label] = np.nan

        # volatility regime: current vol / long-term vol
        if idx >= 60:
            long_vol = np.std(r_hist[-60:])
            feats["vol_regime"] = feats.get("vol_20d", np.nan) / max(long_vol, 0.01) if not np.isnan(feats.get("vol_20d", np.nan)) else np.nan
        else:
            feats["vol_regime"] = np.nan

        # ── 6. Price position vs N-day high/low ──
        for lb, label in [(20, "pos_20d"), (60, "pos_60d")]:
            if idx >= lb:
                h = np.max(c_hist[-lb-1:-1])
                l = np.min(c_hist[-lb-1:-1])
                rng = h - l
                feats[f"pct_from_high_{label}"] = (c_now - h) / max(h, 0.01) * 100
                feats[f"pos_in_range_{label}"] = (c_now - l) / max(rng, 0.01) if rng > 0 else 0.5
            else:
                feats[f"pct_from_high_{label}"] = np.nan
                feats[f"pos_in_range_{label}"] = np.nan

        # ── 7. Max drawdown (recent 20d) ──
        if idx >= 20:
            window = c_hist[-21:-1]
            peak = np.maximum.accumulate(window)
            dd = (window - peak) / np.where(peak > 0, peak, 1)
            feats["max_dd_20d"] = np.min(dd) * 100
        else:
            feats["max_dd_20d"] = np.nan

        # ── 8. Consecutive up/down days ──
        cons_up = 0
        for j in range(idx-1, max(idx-20, -1), -1):
            if r_hist[j] > 0:
                cons_up += 1
            else:
                break
        cons_down = 0
        for j in range(idx-1, max(idx-20, -1), -1):
            if r_hist[j] < 0:
                cons_down += 1
            else:
                break
        feats["consecutive_up"] = cons_up
        feats["consecutive_down"] = cons_down

        # ── 9. Amplitude (daily range) features ──
        if a_hist is not None and len(a_hist) >= 6:
            feats["amplitude"] = a_hist[-1]
            feats["amplitude_5d_avg"] = np.mean(a_hist[-6:-1])
            feats["amplitude_20d_avg"] = np.mean(a_hist[-21:-1]) if len(a_hist) >= 21 else np.mean(a_hist[-6:-1])
        else:
            feats["amplitude"] = feats["amplitude_5d_avg"] = feats["amplitude_20d_avg"] = np.nan

        # ── 10. Gap up/down frequency (recent 20d) ──
        if idx >= 20:
            gaps = []
            for j in range(idx-19, idx+1):
                if j > 0 and c_hist[j-1] > 0:
                    gap = (c_hist[j] - c_hist[j-1]) / c_hist[j-1] * 100
                    gaps.append(gap)
            gaps = np.array(gaps)
            feats["gap_up_count_20d"] = np.sum(gaps > 2.0)
            feats["gap_down_count_20d"] = np.sum(gaps < -2.0)
            feats["max_gap_20d"] = np.max(gaps)
            feats["min_gap_20d"] = np.min(gaps)
        else:
            feats["gap_up_count_20d"] = feats["gap_down_count_20d"] = np.nan
            feats["max_gap_20d"] = feats["min_gap_20d"] = np.nan

        # ── 11. Return skewness ──
        if idx >= 20:
            feats["ret_skew_20d"] = scipy_stats.skew(r_hist[-20:]) if len(r_hist[-20:]) >= 10 else np.nan
            feats["ret_kurt_20d"] = scipy_stats.kurtosis(r_hist[-20:]) if len(r_hist[-20:]) >= 10 else np.nan
        else:
            feats["ret_skew_20d"] = feats["ret_kurt_20d"] = np.nan

        # ── 12. Up/down capture ratio ──
        if idx >= 20:
            up_days = r_hist[-20:][r_hist[-20:] > 0]
            down_days = r_hist[-20:][r_hist[-20:] < 0]
            if len(up_days) > 0 and len(down_days) > 0:
                feats["up_capture"] = np.mean(up_days)
                feats["down_capture"] = np.abs(np.mean(down_days))
                feats["up_down_ratio"] = feats["up_capture"] / max(feats["down_capture"], 0.01)
            else:
                feats["up_capture"] = feats["down_capture"] = feats["up_down_ratio"] = np.nan
        else:
            feats["up_capture"] = feats["down_capture"] = feats["up_down_ratio"] = np.nan

        # ── 13. Price vs MA distance ──
        for lb, label in [(5, "ma5"), (10, "ma10"), (20, "ma20"), (60, "ma60")]:
            if idx >= lb:
                ma = np.mean(c_hist[-lb-1:-1])
                feats[f"dist_{label}"] = (c_now - ma) / max(ma, 0.01) * 100
            else:
                feats[f"dist_{label}"] = np.nan

        # ── 14. MA alignment (ma5 > ma10 > ma20 > ma60) ──
        if all(not np.isnan(feats.get(f"dist_ma{lb}", np.nan)) for lb in [5, 10, 20, 60]):
            feats["ma_bull_align"] = 0
            if (np.mean(c_hist[-6:-1]) > np.mean(c_hist[-11:-1]) >
                    np.mean(c_hist[-21:-1]) > np.mean(c_hist[-61:-1])):
                feats["ma_bull_align"] = 1
        else:
            feats["ma_bull_align"] = 0

        # ── 15. Daily return at signal date ──
        feats["daily_ret"] = r_hist[-1] if len(r_hist) > 0 else np.nan

        # ── 16. Win rate (recent 20d) ──
        if idx >= 20:
            feats["win_rate_20d"] = np.mean(r_hist[-20:] > 0)
        else:
            feats["win_rate_20d"] = np.nan

        # ── 17. Average up-day return / average down-day return ──
        if idx >= 20:
            feats["avg_up_ret"] = np.mean(r_hist[-20:][r_hist[-20:] > 0]) if np.any(r_hist[-20:] > 0) else 0
            feats["avg_down_ret"] = np.mean(np.abs(r_hist[-20:][r_hist[-20:] < 0])) if np.any(r_hist[-20:] < 0) else 0
        else:
            feats["avg_up_ret"] = feats["avg_down_ret"] = np.nan

        # ── 18. Recent highest return ──
        if idx >= 20:
            feats["max_ret_20d"] = np.max(r_hist[-20:])
            feats["min_ret_20d"] = np.min(r_hist[-20:])
        else:
            feats["max_ret_20d"] = feats["min_ret_20d"] = np.nan

        results.append({"date": sig_date, **feats})

    return pd.DataFrame(results)


def load_price_data(code: str) -> pd.DataFrame | None:
    """Load none.csv (unadjusted) price data with all columns."""
    fpath = os.path.join(RAW_DATA_DIR, code, "none.csv")
    if not os.path.exists(fpath):
        return None
    try:
        df = pd.read_csv(fpath, dtype={"股票代码": str})
        df.rename(columns={
            "日期": "date", "开盘": "open", "收盘": "close",
            "最高": "high", "最低": "low", "成交量": "volume",
            "成交额": "amount", "振幅": "amplitude", "涨跌幅": "change_pct",
            "涨跌额": "change_amt", "换手率": "turnover",
        }, inplace=True)
        df["date"] = pd.to_datetime(df["date"])
        for col in ["open", "close", "high", "low", "volume", "amount", "amplitude", "change_pct", "change_amt", "turnover"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df.sort_values("date", inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df
    except Exception:
        return None


def compute_forward_returns(signals_sample: pd.DataFrame, code: str) -> pd.DataFrame:
    """Compute forward returns for a single stock."""
    price = load_price_data(code)
    if price is None or len(price) < 30:
        return pd.DataFrame()

    stock_signals = signals_sample[signals_sample["code"] == code].copy()
    if stock_signals.empty:
        return pd.DataFrame()

    price_dict = price.set_index("date")["close"]
    all_dates = price["date"].values
    date_to_idx = {pd.Timestamp(d).date(): i for i, d in enumerate(all_dates)}

    results = []
    for _, row in stock_signals.iterrows():
        sig_date = row["date"]
        sig_date_key = sig_date.date() if hasattr(sig_date, "date") else sig_date
        idx = date_to_idx.get(sig_date_key)
        if idx is None:
            continue
        close_t = float(price_dict.iloc[idx])
        if close_t <= 0:
            continue

        fwd = {"code": code, "date": sig_date}
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

        lookahead_20 = idx + 20
        if lookahead_20 < len(all_dates):
            prices_20 = price_dict.iloc[idx:lookahead_20+1].values
            if all(p > 0 for p in prices_20):
                fwd["fwd_max_20d"] = (max(prices_20) - close_t) / close_t * 100
                fwd["fwd_min_20d"] = (min(prices_20) - close_t) / close_t * 100
                fwd["fwd_volatility_20d"] = np.std(
                    [(p - close_t) / close_t * 100 for p in prices_20]
                )

        results.append(fwd)

    return pd.DataFrame(results)


def compute_ic(series_a, series_b):
    """Compute Spearman rank IC."""
    mask = series_a.notna() & series_b.notna()
    if mask.sum() < 30:
        return 0.0, 1.0
    return scipy_stats.spearmanr(series_a[mask], series_b[mask])


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    t0 = time.time()

    # ── Step 1: Load & sample signals ──
    print("=" * 70)
    print("Step 1: Loading backtest signals")
    print("=" * 70)
    signals = pd.read_csv(BACKTEST_SIGNALS, dtype={"code": str})
    print(f"  Total rows: {len(signals):,}")

    buy_signals = signals[signals["buy"] == True].copy()
    del signals; gc.collect()
    print(f"  Buy signals: {len(buy_signals):,}")

    buy_signals["date"] = pd.to_datetime(buy_signals["date"])
    if len(buy_signals) > MAX_SAMPLE:
        buy_signals = buy_signals.sample(n=MAX_SAMPLE, random_state=SEED)
    print(f"  Sampled: {len(buy_signals):,}")
    print(f"  Date range: {buy_signals['date'].min().date()} to {buy_signals['date'].max().date()}")
    print(f"  Unique stocks: {buy_signals['code'].nunique()}")

    # ── Step 2: Forward returns ──
    print("\n" + "=" * 70)
    print("Step 2: Computing forward returns")
    print("=" * 70)
    unique_codes = buy_signals["code"].unique()
    n_codes = len(unique_codes)
    print(f"  Stocks to process: {n_codes:,}")

    all_fwd = []
    for i, code in enumerate(unique_codes):
        if (i + 1) % 200 == 0:
            print(f"    [{i+1}/{n_codes}] fwd returns... ({time.time()-t0:.0f}s)")
        result = compute_forward_returns(buy_signals, code)
        if not result.empty:
            all_fwd.append(result)

    fwd_df = pd.concat(all_fwd, ignore_index=True)
    print(f"  Forward returns: {len(fwd_df):,}")

    # ── Step 3: Pre-signal features ──
    print("\n" + "=" * 70)
    print("Step 3: Computing pre-signal features from raw price data")
    print("=" * 70)

    all_features = []
    codes_processed = 0
    for i, code in enumerate(unique_codes):
        if (i + 1) % 200 == 0:
            print(f"    [{i+1}/{n_codes}] features... ({time.time()-t0:.0f}s)")

        stock_sigs = buy_signals[buy_signals["code"] == code]
        if stock_sigs.empty:
            continue

        price_df = load_price_data(code)
        if price_df is None:
            continue

        sig_dates = stock_sigs["date"].tolist()
        feats = compute_pre_signal_features(price_df, sig_dates)
        if not feats.empty:
            feats["code"] = code
            all_features.append(feats)
            codes_processed += 1

    print(f"  Processed {codes_processed} stocks with features")
    feat_df = pd.concat(all_features, ignore_index=True)
    print(f"  Feature rows: {len(feat_df):,}")

    # ── Step 4: Merge everything ──
    print("\n" + "=" * 70)
    print("Step 4: Merging all data")
    print("=" * 70)

    # Merge fwd returns + signal features + pre-signal features
    signal_feat_cols = [
        "code", "date", "score", "factor_value", "factor_name", "industry",
        "factor_quality", "chan_divergence_type", "chan_divergence_strength",
        "chan_structure_score", "chan_buy_point", "chan_sell_point",
        "signal_level", "chan_pivot_zg", "chan_pivot_zd",
    ]

    merged = fwd_df.merge(
        buy_signals[signal_feat_cols], on=["code", "date"], how="inner"
    )
    print(f"  After signal merge: {len(merged):,}")

    feat_df["date"] = pd.to_datetime(feat_df["date"])
    merged = merged.merge(feat_df, on=["code", "date"], how="inner")
    print(f"  After feature merge: {len(merged):,}")

    merged.dropna(subset=["fwd_10d"], inplace=True)
    print(f"  With fwd_10d: {len(merged):,}")

    # Save full dataset
    merged.to_csv(os.path.join(OUTPUT_DIR, "enhanced_analysis_data.csv"), index=False)
    print(f"  Saved: {len(merged):,} rows, {len(merged.columns)} columns")
    gc.collect()

    # ── Step 5: Classify winners/losers ──
    print("\n" + "=" * 70)
    print("Step 5: Classification")
    print("=" * 70)

    th_high = np.percentile(merged["fwd_10d"], WINNER_PCTILE)
    th_low = np.percentile(merged["fwd_10d"], LOSER_PCTILE)
    print(f"  Winner threshold (p{WINNER_PCTILE}): fwd_10d >= {th_high:+.2f}%")
    print(f"  Loser  threshold (p{LOSER_PCTILE}): fwd_10d <= {th_low:+.2f}%")

    merged["group"] = "neutral"
    merged.loc[merged["fwd_10d"] >= th_high, "group"] = "winner"
    merged.loc[merged["fwd_10d"] <= th_low, "group"] = "loser"

    winners = merged[merged["group"] == "winner"]
    losers = merged[merged["group"] == "loser"]
    neutrals = merged[merged["group"] == "neutral"]

    print(f"  Winners: {len(winners):,} ({100*len(winners)/len(merged):.1f}%) mean={winners['fwd_10d'].mean():+.2f}%")
    print(f"  Losers:  {len(losers):,} ({100*len(losers)/len(merged):.1f}%) mean={losers['fwd_10d'].mean():+.2f}%")
    print(f"  Neutral: {len(neutrals):,} ({100*len(neutrals)/len(merged):.1f}%) mean={neutrals['fwd_10d'].mean():+.2f}%")

    # ── Step 6: IC Analysis for ALL features ──
    print("\n" + "=" * 70)
    print("Step 6: IC Analysis (Spearman rank correlation with fwd_10d)")
    print("=" * 70)

    # All numeric features
    exclude = ["code", "date", "group", "fwd_5d", "fwd_10d", "fwd_20d",
               "fwd_max_20d", "fwd_min_20d", "fwd_volatility_20d",
               "factor_name", "industry", "chan_divergence_type",
               "chan_pivot_zg", "chan_pivot_zd"]
    numeric_cols = [c for c in merged.columns if c not in exclude
                    and merged[c].dtype in ("float64", "int64", "float32", "int32")]

    ic_results = []
    for col in numeric_cols:
        ic, pv = compute_ic(merged[col], merged["fwd_10d"])
        # also compute effect size winners vs losers
        w_mean = winners[col].mean() if col in winners.columns else np.nan
        l_mean = losers[col].mean() if col in losers.columns else np.nan
        w_std = winners[col].std() if col in winners.columns else np.nan
        l_std = losers[col].std() if col in losers.columns else np.nan
        diff = w_mean - l_mean
        pooled_std = np.sqrt((w_std**2 + l_std**2) / 2) if not np.isnan(w_std) and not np.isnan(l_std) else np.nan
        es = diff / pooled_std if pooled_std and pooled_std > 0 else 0.0

        ic_results.append({
            "feature": col,
            "IC": round(ic, 6),
            "p_value": round(pv, 6),
            "W_mean": round(w_mean, 4) if not np.isnan(w_mean) else np.nan,
            "L_mean": round(l_mean, 4) if not np.isnan(l_mean) else np.nan,
            "diff": round(diff, 4) if not np.isnan(diff) else np.nan,
            "effect_size": round(es, 4),
            "N_valid": merged[col].notna().sum(),
        })

    ic_df = pd.DataFrame(ic_results)
    ic_df["abs_IC"] = ic_df["IC"].abs()
    ic_df.sort_values("abs_IC", ascending=False, inplace=True)

    print(f"\n  {'Rank':>4s} {'Feature':<28s} {'IC':>10s} {'p-value':>10s} {'EffectSz':>10s} {'Direction':>14s} {'N':>8s}")
    print(f"  {'-'*4} {'-'*28} {'-'*10} {'-'*10} {'-'*10} {'-'*14} {'-'*8}")
    for i, (_, row) in enumerate(ic_df.head(40).iterrows()):
        sig = "***" if row["p_value"] < 0.001 else "**" if row["p_value"] < 0.01 else "*" if row["p_value"] < 0.05 else ""
        direction = "HIGHER=BETTER" if row["diff"] > 0 else "LOWER=BETTER" if row["diff"] < 0 else "NEUTRAL"
        print(f"  {i+1:>4d} {row['feature']:<28s} {row['IC']:>+10.6f}{sig:<3s} {row['p_value']:>10.6f} {row['effect_size']:>+10.4f} {direction:>14s} {row['N_valid']:>8.0f}")

    ic_df.to_csv(os.path.join(OUTPUT_DIR, "enhanced_ic_results.csv"), index=False)

    # ── Step 7: XGBoost feature importance ──
    print("\n" + "=" * 70)
    print("Step 7: XGBoost Feature Importance")
    print("=" * 70)

    try:
        import xgboost as xgb
    except ImportError:
        print("  xgboost not installed, skipping ML analysis")
        xgb = None

    if xgb is not None:
        # Prepare data: all numeric features predict winner/loser binary
        ml_features = [c for c in numeric_cols if merged[c].notna().sum() > 1000]
        ml_data = merged[ml_features + ["group"]].copy()
        ml_data = ml_data[ml_data["group"] != "neutral"].copy()
        ml_data["target"] = (ml_data["group"] == "winner").astype(int)

        # Fill remaining NaNs with median
        for col in ml_features:
            if ml_data[col].isna().any():
                ml_data[col].fillna(ml_data[col].median(), inplace=True)

        X = ml_data[ml_features].values
        y = ml_data["target"].values

        print(f"  Training data: {len(X):,} samples, {len(ml_features)} features")
        print(f"  Winner ratio: {y.mean():.3f}")

        # Train XGBoost
        dtrain = xgb.DMatrix(X, label=y, feature_names=ml_features)
        params = {
            "max_depth": 4,
            "eta": 0.05,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "seed": SEED,
        }

        model = xgb.train(params, dtrain, num_boost_round=200, verbose_eval=False)

        # Feature importance
        importance = model.get_score(importance_type="gain")
        imp_df = pd.DataFrame([
            {"feature": k, "gain": v} for k, v in importance.items()
        ]).sort_values("gain", ascending=False)

        print(f"\n  Top 30 XGBoost features (gain):")
        print(f"  {'Rank':>4s} {'Feature':<28s} {'Gain':>12s}")
        print(f"  {'-'*4} {'-'*28} {'-'*12}")
        for i, (_, row) in enumerate(imp_df.head(30).iterrows()):
            print(f"  {i+1:>4d} {row['feature']:<28s} {row['gain']:>12.2f}")

        imp_df.to_csv(os.path.join(OUTPUT_DIR, "enhanced_xgb_importance.csv"), index=False)

        # Correlation between top features and forward returns
        print(f"\n  Feature correlation matrix (top 15 features + fwd_10d):")
        top15 = imp_df.head(15)["feature"].tolist()
        corr_cols = top15 + ["fwd_10d"]
        corr = merged[corr_cols].corr()["fwd_10d"].drop("fwd_10d").sort_values(ascending=False)
        for feat, val in corr.items():
            print(f"    {feat:<28s} corr={val:+.4f}")

    # ── Step 8: Non-linear analysis ──
    print("\n" + "=" * 70)
    print("Step 8: Non-linear pattern analysis")
    print("=" * 70)

    # Top features by IC
    top_ic_features = ic_df.head(10)["feature"].tolist()

    for feat in top_ic_features[:8]:
        if merged[feat].nunique() < 5:
            continue
        print(f"\n  --- {feat} decile analysis ---")
        try:
            merged[f"{feat}_decile"] = pd.qcut(merged[feat], 10, labels=False, duplicates="drop")
            decile_ret = merged.groupby(f"{feat}_decile")["fwd_10d"].agg(["mean", "std", "count"])
            # also winner rate
            decile_win = merged.groupby(f"{feat}_decile")["group"].apply(
                lambda x: (x == "winner").mean() * 100
            )
            decile_loss = merged.groupby(f"{feat}_decile")["group"].apply(
                lambda x: (x == "loser").mean() * 100
            )

            print(f"    {'Decile':>8s} {'Mean_fwd':>10s} {'Win%':>8s} {'Loss%':>8s} {'Count':>8s}")
            for d in sorted(decile_ret.index):
                r = decile_ret.loc[d]
                wp = decile_win.get(d, 0)
                lp = decile_loss.get(d, 0)
                print(f"    {int(d):>8d} {r['mean']:>+10.2f}% {wp:>7.1f}% {lp:>7.1f}% {r['count']:>8.0f}")

            # Cleanup
            del merged[f"{feat}_decile"]
        except Exception as e:
            print(f"    Skipped: {e}")

    # ── Step 9: Conditional probability analysis ──
    print("\n" + "=" * 70)
    print("Step 9: Conditional probability (multi-feature combinations)")
    print("=" * 70)

    # Find the strongest features and analyze combinations
    top5_feats = ic_df.head(5)["feature"].tolist()
    print(f"  Top 5 features by IC: {top5_feats}")

    for i, f1 in enumerate(top5_feats[:4]):
        for f2 in top5_feats[i+1:5]:
            if merged[f1].nunique() < 5 or merged[f2].nunique() < 5:
                continue

            # High-High vs Low-Low
            m1 = merged[f1].median()
            m2 = merged[f2].median()

            hh = merged[(merged[f1] >= m1) & (merged[f2] >= m2)]
            ll = merged[(merged[f1] < m1) & (merged[f2] < m2)]
            hl = merged[(merged[f1] >= m1) & (merged[f2] < m2)]
            lh = merged[(merged[f1] < m1) & (merged[f2] >= m2)]

            if len(hh) < 100 or len(ll) < 100:
                continue

            hh_win = (hh["group"] == "winner").mean() * 100
            ll_win = (ll["group"] == "winner").mean() * 100
            hh_loss = (hh["group"] == "loser").mean() * 100
            ll_loss = (ll["group"] == "loser").mean() * 100

            spread = hh_win - ll_win
            if abs(spread) > 3:  # only show meaningful spreads
                print(f"\n  {f1} x {f2}:")
                print(f"    Both HIGH:  Win%={hh_win:.1f}%  Loss%={hh_loss:.1f}%  "
                      f"Mean_fwd={hh['fwd_10d'].mean():+.2f}%  N={len(hh):,}")
                print(f"    Both LOW:   Win%={ll_win:.1f}%  Loss%={ll_loss:.1f}%  "
                      f"Mean_fwd={ll['fwd_10d'].mean():+.2f}%  N={len(ll):,}")
                print(f"    High-Low:   Win%={hl['group'].eq('winner').mean()*100:.1f}%  "
                      f"Mean_fwd={hl['fwd_10d'].mean():+.2f}%")
                print(f"    Low-High:   Win%={lh['group'].eq('winner').mean()*100:.1f}%  "
                      f"Mean_fwd={lh['fwd_10d'].mean():+.2f}%")
                print(f"    Spread (HH-LL): Win%={spread:+.1f}%, "
                      f"fwd_ret={hh['fwd_10d'].mean()-ll['fwd_10d'].mean():+.2f}%")

    # ── Step 10: Key findings & recommendations ──
    print("\n" + "=" * 70)
    print("Step 10: KEY FINDINGS & STRATEGY RECOMMENDATIONS")
    print("=" * 70)

    # Strongest positive predictors
    pos_pred = ic_df[ic_df["IC"] > 0.01].head(10)
    neg_pred = ic_df[ic_df["IC"] < -0.01].head(10)

    print("\n  [STRONGEST POSITIVE PREDICTORS of forward returns]")
    for _, row in pos_pred.iterrows():
        print(f"    {row['feature']:<28s} IC={row['IC']:+.4f}  ES={row['effect_size']:+.3f}  "
              f"W_mean={row['W_mean']:+.2f} L_mean={row['L_mean']:+.2f}")

    print("\n  [STRONGEST NEGATIVE PREDICTORS (danger signals)]")
    for _, row in neg_pred.iterrows():
        print(f"    {row['feature']:<28s} IC={row['IC']:+.4f}  ES={row['effect_size']:+.3f}  "
              f"W_mean={row['W_mean']:+.2f} L_mean={row['L_mean']:+.2f}")

    print("\n  [RECOMMENDATIONS FOR STRATEGY]")
    print("  Based on the analysis above, the following strategy changes are recommended:")
    print()

    # Generate specific recommendations based on findings
    rec_num = 1

    # Check if momentum features are strong
    mom_features = [r for r in ic_df.itertuples() if "mom_" in str(r.feature) and abs(r.IC) > 0.005]
    if mom_features:
        best_mom = mom_features[0]
        print(f"  {rec_num}. MOMENTUM ENHANCEMENT: '{best_mom.feature}' shows IC={best_mom.IC:+.4f}")
        print(f"     -> Add pre-signal momentum as a signal quality filter in portfolio.py")
        print(f"     -> Stocks with strong positive pre-signal momentum are more likely to continue")
        rec_num += 1

    # Check if volatility features are strong
    vol_features = [r for r in ic_df.itertuples() if "vol_" in str(r.feature) and abs(r.IC) > 0.005]
    if vol_features:
        best_vol = vol_features[0]
        direction = "lower" if best_vol.IC < 0 else "higher"
        print(f"  {rec_num}. VOLATILITY FILTER: '{best_vol.feature}' shows IC={best_vol.IC:+.4f}")
        print(f"     -> {direction} pre-signal volatility predicts better forward returns")
        print(f"     -> Add volatility-based position sizing adjustment in portfolio.py")
        rec_num += 1

    # Check volume features
    vol_ratio_features = [r for r in ic_df.itertuples() if "vol_ratio" in str(r.feature) and abs(r.IC) > 0.005]
    if vol_ratio_features:
        best_vr = vol_ratio_features[0]
        direction = "higher" if best_vr.IC > 0 else "lower"
        print(f"  {rec_num}. VOLUME RATIO SIGNAL: '{best_vr.feature}' shows IC={best_vr.IC:+.4f}")
        print(f"     -> {direction} relative volume at signal entry predicts better returns")
        print(f"     -> Add volume confirmation requirement for new entries")
        rec_num += 1

    # Check gap features
    gap_features = [r for r in ic_df.itertuples() if "gap" in str(r.feature) and abs(r.IC) > 0.005]
    if gap_features:
        for gf in gap_features[:2]:
            direction = "higher" if gf.IC > 0 else "lower"
            print(f"  {rec_num}. GAP PATTERN: '{gf.feature}' shows IC={gf.IC:+.4f}")
            print(f"     -> {direction} values predict better returns")
            rec_num += 1

    # Check turnover
    turnover_features = [r for r in ic_df.itertuples() if "turnover" in str(r.feature) and abs(r.IC) > 0.005]
    if turnover_features:
        best_to = turnover_features[0]
        direction = "higher" if best_to.IC > 0 else "lower"
        print(f"  {rec_num}. TURNOVER SIGNAL: '{best_to.feature}' shows IC={best_to.IC:+.4f}")
        print(f"     -> {direction} turnover predicts better returns")
        rec_num += 1

    # Check MA features
    ma_features = [r for r in ic_df.itertuples() if "dist_ma" in str(r.feature) and abs(r.IC) > 0.005]
    if ma_features:
        best_ma = ma_features[0]
        print(f"  {rec_num}. MA POSITION: '{best_ma.feature}' shows IC={best_ma.IC:+.4f}")
        print(f"     -> Price position relative to MA is predictive of forward returns")
        rec_num += 1

    # Check max_dd
    dd_features = [r for r in ic_df.itertuples() if "max_dd" in str(r.feature) and abs(r.IC) > 0.005]
    if dd_features:
        best_dd = dd_features[0]
        direction = "higher" if best_dd.IC > 0 else "lower"
        print(f"  {rec_num}. MAX DRAWDOWN: '{best_dd.feature}' shows IC={best_dd.IC:+.4f}")
        print(f"     -> {direction} recent max drawdown predicts better returns")
        rec_num += 1

    # Check existing signal features IC
    existing_features = ["score", "factor_value", "factor_quality",
                         "chan_divergence_strength", "chan_structure_score", "chan_buy_point"]
    for ef in existing_features:
        ef_row = ic_df[ic_df["feature"] == ef]
        if not ef_row.empty:
            row = ef_row.iloc[0]
            print(f"  {rec_num}. EXISTING FEATURE '{ef}': IC={row['IC']:+.4f} (baseline comparison)")
            rec_num += 1

    print(f"\n  Total time: {time.time()-t0:.0f}s ({(time.time()-t0)/60:.1f}min)")
    print("  Output saved to:", OUTPUT_DIR)
    print("  - enhanced_analysis_data.csv  (full dataset)")
    print("  - enhanced_ic_results.csv     (IC rankings)")
    if xgb is not None:
        print("  - enhanced_xgb_importance.csv (XGBoost feature importance)")
    print("\nDone.")


if __name__ == "__main__":
    main()
