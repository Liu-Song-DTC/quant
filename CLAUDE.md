# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A quantitative trading system for A-shares (Chinese stock market) built with Python. The system implements signal-based trading strategies with backtesting capabilities using the Backtrader framework.

## Commands

### Run Backtest
```bash
cd strategy && python bt_execution.py
```

### Download/Update Stock Data
```bash
cd data && python data_manager.py
```

## Architecture

### Core Flow
```
StockDataManager (data fetch) → Strategy (signal generation) → BacktraderExecution (backtest)
```

### Key Components

**strategy/core/strategy.py** - `Strategy` class orchestrates the entire pipeline:
- Generates market regime indicators from index data (sh000001)
- Delegates signal generation to `SignalEngine`
- Delegates position sizing to `PortfolioConstructor`

**strategy/core/signal_engine.py** - `SignalEngine` generates trading signals using multiple sub-strategies:
- Trend following (MA crossovers)
- Mean reversion (RSI, Bollinger Bands)
- Momentum
- Volume-price analysis
- Volatility-based signals
- Signals are combined with regime-aware weighting and smoothing

**strategy/core/portfolio.py** - `PortfolioConstructor` converts signals to positions:
- Risk budget allocation based on signal score and volatility
- Target volatility control
- Gross exposure limits based on market regime (1.0/0.6/0.3 for up/neutral/down)
- Drawdown-based defensive mode
- Gradual position entry/exit (entry_speed, exit_speed parameters)

**strategy/core/signal_store.py** - `SignalStore` caches signals by (code, date) tuple

**strategy/bt_execution.py** - Backtrader integration layer:
- `BacktraderExecution` wraps the strategy for Backtrader
- Handles order management, A-share 100-share lot sizing
- Configurable parameters: CASH, COMMISSION, MAX_POSITION, REBALANCE_DAYS

**data/data_manager.py** - `StockDataManager` handles data acquisition:
- Uses akshare for A-share data (with proxy support)
- Downloads raw, qfq (forward-adjusted), hfq (backward-adjusted) price data
- Creates Backtrader-compatible CSV files
- Includes data quality validation and filtering (ST exclusion, market cap, etc.)

### Data Paths
- Raw data: `data/stock_data/raw_data/{symbol}/`
- Backtrader data: `data/stock_data/backtrader_data/{symbol}_qfq.csv`
- Index: sh000001 (Shanghai Composite)

### Signal Data Structure
`Signal` dataclass: `buy`, `sell`, `score`, `risk_vol`
