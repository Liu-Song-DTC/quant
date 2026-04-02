# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A quantitative trading system for A-shares (Chinese stock market) built with Python. The system implements signal-based trading strategies with backtesting capabilities using the Backtrader framework.

**Goal: Sharpe Ratio > 1.0**

## Commands

### Run Full Pipeline
```bash
# 1. Run backtest (generates backtest_signals.csv, portfolio_selections.csv)
cd strategy && python bt_execution.py

# 2. Prepare validation data (calculates future_ret)
cd strategy && python analysis/signal_validator.py

# 3. Run analysis (produces IC metrics, signal quality, recommendations)
cd strategy && python analysis/analysis_framework.py
```

### Download/Update Stock Data
```bash
cd data && python data_manager.py
```

### Configuration
All parameters are defined in `strategy/config/factor_config.yaml`:
- `factor_mode` - dynamic/fixed/both (controls dynamic factor selection)
- `dynamic_factor` - train_window_days, forward_period, top_n_factors, min_ic_dates, ic_decay_factor
- `backtest` - cash, commission, slippage, max_position, rebalance_days, num_workers
- `portfolio` - max_position, target_volatility, entry_speed, exit_speed, stop losses
- `signal` - buy_threshold, sell_threshold

## Architecture

### Data Flow
```
factor_preparer.py → factor_df (all stocks × all dates)
                          ↓
              DynamicFactorSelector → IC validation → select top factors
                          ↓
              signal_engine.py → signals
                          ↓
              portfolio.py → selections
                          ↓
              bt_execution.py → backtest results
```

### Key Components

**strategy/core/factor_calculator.py** - Unified factor calculation:
- Single source of truth for all factor calculations
- Used by both signal_engine and factor_preparer
- Functions: `calculate_indicators()`, `compute_composite_factors()`

**strategy/core/factor_preparer.py** - Factor data precomputation:
- Prepares factor_df for dynamic factor selection
- Uses multiprocessing for parallel computation
- Only runs when `factor_mode != 'fixed'`

**strategy/core/signal_engine.py** - `SignalEngine` + `DynamicFactorSelector`:
- `SignalEngine`: generates trading signals using selected factors
- `DynamicFactorSelector`: walk-forward IC validation to select best factors per industry
- Industry mapping via `industry_mapping.py`

**strategy/core/portfolio.py** - `PortfolioConstructor` converts signals to positions:
- Risk budget allocation based on signal score and volatility
- Target volatility control
- Market regime-aware exposure (1.0/0.6/0.3 for bull/neutral/bear)

**strategy/core/market_regime_detector.py** - `MarketRegimeDetector`:
- Detects bull, bear, neutral regimes from index data (sh000001)
- Provides regime-aware factor weighting

**strategy/bt_execution.py** - Backtrader integration:
- `BacktraderExecution` wraps strategy for Backtrader
- Handles A-share 100-share lot sizing, order management

### Analysis Framework

**strategy/analysis/signal_validator.py** - Data preparation:
- Calculates future_ret for backtest signals
- Outputs: `validation_results.csv`

**strategy/analysis/analysis_framework.py** - Unified analysis:
- Module 1: Data quality check
- Module 2: Factor layer (dynamic factor usage, IC by factor count/industry)
- Module 3: Signal layer (buy/sell accuracy, signal IC)
- Module 4: Portfolio layer (industry distribution, concentration)
- Module 5: Temporal stability (yearly IC, IR)
- Module 6: Optimization recommendations

### Key Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| IC | >5% | Spearman correlation between factor rank and future return |
| IR | >0.5 | IC mean / IC std |
| Buy Accuracy | >55% | % of buy signals with positive future return |
| Sharpe | >1.0 | Strategy Sharpe ratio |

### Data Paths
- Raw data: `data/stock_data/raw_data/{symbol}/`
- Backtrader data: `data/stock_data/backtrader_data/{symbol}_qfq.csv`
- Fundamental data: `data/stock_data/fundamental_data/`
- Validation results: `strategy/rolling_validation_results/`

### Signal Data Structure
`Signal` dataclass: `buy`, `sell`, `score`, `factor_value`, `factor_name`, `industry`

### Factor Naming Convention
- `DYN_{industry}_{n}F_F` - Dynamic factor (walk-forward selected, n factors with fundamentals)
- `IND_{industry}_{type}` - Static industry factor
- `V41` - Fallback factor (used when data insufficient, idx < 60)
- `_T` suffix - Technical-only factor (no fundamental data)
