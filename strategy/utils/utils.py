import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_signal_diagnosis(
    code,
    market_data,
    signal_engine,
    signal_store,
):
    df = market_data.copy()
    df["date"] = pd.to_datetime(df["datetime"]).dt.date

    fig, axes = plt.subplots(
        3, 1, figsize=(16, 10), sharex=True,
        gridspec_kw={"height_ratios": [3, 2, 1]}
    )

    # =========
    # Panel 1: Price
    # =========
    ax = axes[0]
    ax.plot(df["date"], df["close"], label="Close", color="black", linewidth=1)
    ax.plot(df["date"], signal_engine.ma_fast, label=f"MA{signal_engine.fast_ma}", color="blue", alpha=0.7)
    ax.plot(df["date"], signal_engine.ma_slow, label=f"MA{signal_engine.slow_ma}", color="orange", alpha=0.7)

    # ===== Signal =====
    buy_dates = []
    buy_prices = []
    sell_dates = []
    sell_prices = []
    scores = []
    buy_scores = []
    sell_scores = []
    buy_vols = []
    sell_vols = []

    for i, row in df.iterrows():
        sig = signal_store.get(code, row["date"])
        if sig is None:
            scores.append(0.0)
            continue
        scores.append(sig.score)
        if sig.buy:
            buy_dates.append(row["date"])
            buy_prices.append(row["close"])
            buy_scores.append(sig.score)
            buy_vols.append(sig.vol)
        if sig.sell:
            sell_dates.append(row["date"])
            sell_prices.append(row["close"])
            sell_scores.append(sig.score)
            sell_vols.append(sig.vol)


    ax.scatter(buy_dates, buy_prices, marker="^", color="green", label="Buy", s=80)
    ax.scatter(sell_dates, sell_prices, marker="v", color="red", label="Sell", s=80)

    ax.set_title(f"{code} Price & Signal")
    ax.legend()
    ax.grid(True)

    # =========
    # Panel 2: Score
    # =========
    ax = axes[1]
    ax.plot(df["date"], scores, color="purple", label="Score")
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)

    # 可选：画“弱信号区间”
    band = 0.03
    ax.axhline(band, color="gray", linestyle=":", alpha=0.6)
    ax.axhline(-band, color="gray", linestyle=":", alpha=0.6)

    ax.scatter(buy_dates, buy_scores, marker="^", color="green", label="Buy", s=80)
    ax.scatter(sell_dates, sell_scores, marker="v", color="red", label="Sell", s=80)

    ax.set_title("Signal Score")
    ax.legend()
    ax.grid(True)

    # =========
    # Panel 3: Vol
    # =========
    ax = axes[2]
    ax.plot(df["date"], signal_engine.vol, color="brown", label="Volatility")
    ax.scatter(buy_dates, buy_vols, marker="^", color="green", label="Buy", s=80)
    ax.scatter(sell_dates, sell_vols, marker="v", color="red", label="Sell", s=80)
    ax.set_title("Volatility")
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.show()

