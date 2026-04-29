"""资本敏感性测试: 不同资金下策略表现 (优化版: 因子+信号只算一次)"""
import sys, os
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'strategy'))

import multiprocessing
multiprocessing.set_start_method('fork', force=True)

import pandas as pd
import numpy as np
import backtrader as bt
from tqdm import tqdm
from collections import defaultdict

import strategy.bt_execution as bte
from strategy.core.strategy import Strategy
from strategy.core.fundamental import FundamentalData
from strategy.core.factor_preparer import prepare_factor_data
from strategy.core.signal_store import SignalStore
from strategy.core.config_loader import load_config
from strategy.core.industry_mapping import INDUSTRY_KEYWORDS
from strategy.bt_execution import BacktraderExecution

CASH_LEVELS = [10000, 20000, 30000, 50000, 100000, 200000, 300000]
END_DATE = '2025-12-31'  # 与基线回测对齐

config = load_config()
DATA_PATH = str(ROOT / 'data' / 'stock_data' / 'backtrader_data') + '/'
FUNDAMENTAL_PATH = str(ROOT / 'data' / 'stock_data' / 'fundamental_data') + '/'
COMMISSION = config.get('backtest.commission', 0.0015)
PERC = config.get('backtest.slippage', 0.0015)
REBALANCE_DAYS = config.get('backtest.rebalance_days', 20)
bte.REBALANCE_DAYS = REBALANCE_DAYS
NUM_WORKERS = min(config.get('backtest.num_workers', 4), 4)

# ============================================================
# Step 1: 加载数据和生成信号 (只做一次)
# ============================================================
print("=" * 60)
print("Step 1: 加载数据 + 生成信号 (所有资金档位共用)")
print("=" * 60)

stock_codes = []
stock_data_dict = {}
for f in os.listdir(DATA_PATH):
    if f.endswith('_qfq.csv'):
        name = f[:-8]
    elif f.endswith('_hfq.csv'):
        name = f[:-8]
    else:
        continue
    if name != 'sh000001':
        stock_codes.append(name)
    data = pd.read_csv(DATA_PATH + f, parse_dates=['datetime'])
    data = data[data['datetime'] <= END_DATE]
    stock_data_dict[name] = data

dates = set()
for data in stock_data_dict.values():
    dates.update(data['datetime'])
calendar_index = pd.DatetimeIndex(sorted(dates))

# 基本面
fundamental_data = FundamentalData(FUNDAMENTAL_PATH, stock_codes)

# Market regime
strategy_for_prep = Strategy(init_cash=100000, fundamental_data=fundamental_data)
if "sh000001" in stock_data_dict:
    strategy_for_prep.generate_market_regime(stock_data_dict["sh000001"])
    strategy_for_prep.signal_engine.set_market_regime(strategy_for_prep.index_data)

# Factor preparation (reweight mode)
factor_mode = config.config.get('factor_mode', 'both')
if factor_mode != 'fixed':
    print(f"Factor mode: {factor_mode}")
    factor_df, industry_codes, all_dates = prepare_factor_data(
        stock_data_dict, fundamental_data, INDUSTRY_KEYWORDS, NUM_WORKERS)
    strategy_for_prep.set_factor_data(factor_df, industry_codes)
    if hasattr(strategy_for_prep.signal_engine, 'dynamic_factor_selector') and \
       strategy_for_prep.signal_engine.dynamic_factor_selector.enabled:
        strategy_for_prep.signal_engine.dynamic_factor_selector.precompute_all_factor_selections(
            progress_callback=lambda c, t: None, num_workers=NUM_WORKERS)

# Signal generation
stock_codes_list = [n for n in stock_data_dict if n != "sh000001"]
for code in tqdm(stock_codes_list, desc="generating signals"):
    if code in stock_data_dict:
        strategy_for_prep.generate_signal(code, stock_data_dict[code])
print(f"Signals: {len(strategy_for_prep.signal_store._store)} entries")

# ============================================================
# Step 2: 对每个资金档位跑 Backtrader
# ============================================================
results = []
for cash in CASH_LEVELS:
    n_pos = max(3, min(int(cash / 10000), 15))
    print(f"\n{'='*60}")
    print(f"CASH={cash:,}  max_position={n_pos}")
    print(f"{'='*60}")

    cerebro = bt.Cerebro()
    cerebro.broker.setcash(cash)
    cerebro.broker.setcommission(commission=COMMISSION)
    cerebro.broker.set_slippage_perc(perc=PERC)

    strategy = Strategy(init_cash=cash, fundamental_data=fundamental_data)
    # Inject precomputed signals, index data, market regime
    strategy.signal_store = strategy_for_prep.signal_store
    strategy.index_data = strategy_for_prep.index_data
    strategy.signal_engine.market_regime_data = \
        strategy_for_prep.signal_engine.market_regime_data

    # Add data feeds
    price_cols = ['open', 'high', 'low', 'close']
    for name, data in tqdm(stock_data_dict.items(), desc="adding datafeeds", leave=False):
        if name == "sh000001":
            continue
        d = data.set_index('datetime')
        d = d.reindex(calendar_index)
        d[price_cols] = d[price_cols].ffill()
        if 'volume' in d.columns:
            d['volume'] = d['volume'].fillna(0)
        datafeed = bt.feeds.PandasData(dataname=d)
        cerebro.adddata(datafeed, name=name)

    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='_SharpeRatio', riskfreerate=0.02)
    cerebro.addanalyzer(bt.analyzers.AnnualReturn, _name='_AnnualReturn')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='_DrawDown')
    cerebro.addstrategy(BacktraderExecution, real_strategy=strategy)

    try:
        result = cerebro.run()
        strat = result[0]

        sharpe = strat.analyzers._SharpeRatio.get_analysis().get('sharperatio', None)
        annual = strat.analyzers._AnnualReturn.get_analysis()
        dd = strat.analyzers._DrawDown.get_analysis()
        max_dd = dd.get('max', {}).get('drawdown', 0) if isinstance(dd, dict) else 0
        final_value = cerebro.broker.getvalue()

        r = {
            'cash': cash, 'max_position': n_pos, 'sharpe': sharpe,
            'annual_return': annual, 'max_drawdown': max_dd,
            'final_value': final_value, 'total_return': (final_value - cash) / cash,
        }
        results.append(r)
        print(f"  Sharpe: {sharpe:.3f}" if sharpe else "  Sharpe: N/A")
        print(f"  Total Return: {r['total_return']:.1%}  MaxDD: {max_dd:.1%}")

    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback
        traceback.print_exc()

# ============================================================
# Summary
# ============================================================
print(f"\n{'='*70}")
print(f"{'Cash':>10}  {'Pos':>3}  {'Sharpe':>8}  {'Return':>8}  {'MaxDD':>8}  {'Final':>12}")
print(f"{'-'*70}")
for r in results:
    s = r['sharpe']
    sharpe_str = f"{s:.3f}" if s else "N/A"
    print(f" ¥{r['cash']:>9,}  {r['max_position']:>3}  {sharpe_str:>8}  {r['total_return']:>7.1%}  {r['max_drawdown']:>7.1%}  ¥{r['final_value']:>11,.0f}")

print(f"\n公式: max_position = max(3, min(int(total_equity/10000), 15))")
