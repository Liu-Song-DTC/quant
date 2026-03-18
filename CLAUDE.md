# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A quantitative trading system for A-shares (Chinese stock market) built with Python. The system implements signal-based trading strategies with backtesting capabilities using the Backtrader framework.

## Commands

### Run Backtest
```bash
cd strategy && python bt_execution.py
# or use the shell script
cd strategy && ./run_bt.sh
```

### Run Factor Validation
```bash
cd strategy && python analysis/factor_validator.py
```

### Run Rolling Signal Validation
```bash
cd strategy && python analysis/rolling_signal_validator.py
```

### Download/Update Stock Data
```bash
cd data && python data_manager.py
```

### Configuration
All parameters are defined in `strategy/config/factor_config.yaml`:
- `backtest` - 回测参数 (cash, commission, slippage, max_position, rebalance_days, num_workers)
- `portfolio` - 组合参数 (max_position, target_volatility, entry_speed, exit_speed, stop losses)
- `detailed_industries` - 行业分类定义
- `signal_thresholds` - 买入/卖出阈值
- `technical_weights` - 技术面因子权重
- `regime_weights` - 市场状态动态权重
- `fundamental_weights` - 基本面因子配置

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
- Integrates style factors (small-cap/large-cap, value/growth rotation)
- Supports industry-specific factor selection via `industry_factor_config.py`

**strategy/core/portfolio.py** - `PortfolioConstructor` converts signals to positions:
- Risk budget allocation based on signal score and volatility
- Target volatility control
- Gross exposure limits based on market regime (1.0/0.6/0.3 for up/neutral/down)
- Drawdown-based defensive mode
- Gradual position entry/exit (entry_speed, exit_speed parameters)

**strategy/core/market_regime_detector.py** - `MarketRegimeDetector` identifies market states:
- Detects bull, bear, and neutral regimes
- Uses index technical indicators for regime classification
- Provides regime-aware factor weighting

**strategy/core/industry_factor_config.py** - Industry-specific factor configuration:
- Based on factor validation results
- Each industry has optimized factor selection (top1 or weighted method)
- Supports 14 industry categories

**strategy/core/signal_store.py** - `SignalStore` caches signals by (code, date) tuple

**strategy/core/factor_library.py** - Factor calculation library:
- Unified factor calculation interface
- Used by both signal engine and validation scripts
- Consistent 120-day lookback, 20-day forward period

**strategy/analysis/factor_validator.py** - Factor validation:
- Validates individual factor IC/IR/win rate
- Per-industry factor ranking
- Results saved to `factor_validation_results/`

**strategy/analysis/rolling_signal_validator.py** - Signal validation:
- Uses SignalEngine to generate real signals
- Validates signal quality (IC, IR, win rate)
- Per-industry and overall market validation

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
- Fundamental data: `data/stock_data/fundamental_data/`
- Index: sh000001 (Shanghai Composite)

### Signal Data Structure
`Signal` dataclass: `buy`, `sell`, `score`, `factor_value`, `risk_vol`, `risk_extreme`
