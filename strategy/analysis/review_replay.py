#!/usr/bin/env python3
"""Replay a frozen signal run and verify context identity before risk experiments.

Old runs without execution_context.pkl reconstruct the original pre-execution
inputs, then MUST reproduce the full curve byte-for-byte before experiments.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd


def load_helper(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod; spec.loader.exec_module(mod)
    return mod


def replay(run, risk_limits=False, label=None, calendar_cadence=False, legacy_market_context=False,
           latest_execution=False):
    started = time.perf_counter()
    run = Path(run).resolve()
    source_root = Path(__file__).resolve().parents[2]
    if not (run / 'manifest.json').exists():
        raise ValueError('expected an isolated run manifest')
    os.environ['QUANT_ALT_NO_AUTOREFRESH'] = '1'
    os.environ['QUANT_ACCOUNT_LIMITS'] = '1' if risk_limits else '0'
    sys.path.insert(0, str(run / 'strategy'))
    import bt_execution as engine
    if latest_execution:
        import core
        core.__path__.append(str(source_root / 'strategy/core'))
        load_helper('core.portfolio', source_root / 'strategy/core/portfolio.py')
        engine = load_helper('review_latest_engine', source_root / 'strategy/bt_execution.py')
    strategy_module = load_helper('core.review_strategy', source_root / 'strategy/core/strategy.py')
    Strategy = strategy_module.Strategy
    from core.fundamental import FundamentalData
    from core.industry_mapping import INDUSTRY_KEYWORDS

    signals = run / 'strategy/rolling_validation_results/backtest_signals.csv'
    if not signals.exists():
        raise FileNotFoundError('signal generation has not completed')
    context_module = load_helper('review_context', source_root / 'strategy/core/execution_context.py')
    context = context_module.load_context(signals)
    membership = engine._load_pool_membership()
    union = set().union(*membership.values()) if membership else None
    stock_map = {}
    for item in os.listdir(engine.DATA_PATH):
        if item.startswith('._') or not item.endswith(('_qfq.csv', '_hfq.csv')):
            continue
        code = item[:-8]
        if union is None or code in union | {'sh000001', '000001', 'sh000852', '399006'}:
            stock_map[code] = os.path.join(engine.DATA_PATH, item)
    fd = FundamentalData(engine.FUNDAMENTAL_PATH, list(stock_map))
    strategy = Strategy(engine.CASH, fundamental_data=fd)
    if context is None:
        if risk_limits:
            raise RuntimeError('risk experiment requires a successful identity replay first')
        idx = pd.read_csv(stock_map['sh000001'], parse_dates=['datetime'])
        idx = idx[(idx['datetime'] >= engine.FROMDATE) & (idx['datetime'] <= engine.TODATE)]
        small = pd.read_csv(stock_map['sh000852'], parse_dates=['datetime']) if 'sh000852' in stock_map else None
        growth = pd.read_csv(stock_map['399006'], parse_dates=['datetime']) if '399006' in stock_map else None
        strategy.generate_market_regime(idx, small_cap_df=small, growth_df=growth)
        dates = sorted(idx['datetime'].tolist())
        early_path = Path(engine.DATA_PATH).parent / 'raw_data/000001/qfq.csv'
        if early_path.exists():
            early = pd.to_datetime(pd.read_csv(early_path, usecols=['日期'])['日期'])
            dates = sorted(set(dates) | {d for d in early if pd.Timestamp(engine.FROMDATE) - pd.Timedelta(days=730) <= d < pd.Timestamp(engine.FROMDATE)})
        midpoint = dates[len(dates) // 2]
        groups = {name: [] for name in INDUSTRY_KEYWORDS}
        for code in stock_map:
            try:
                industry = fd.get_industry(code, midpoint)
                if industry is None and len(dates) > 200:
                    for date in reversed(dates[-200:]):
                        industry = fd.get_industry(code, date)
                        if industry:
                            break
                if industry:
                    for name, keywords in INDUSTRY_KEYWORDS.items():
                        if any(word in str(industry) for word in keywords):
                            groups[name].append(code); break
            except Exception:
                # 与原始 prepare_factor_data 的未匹配行业路径一致。
                continue
        strategy.industry_stock_counts = {k: len(v) for k, v in groups.items()}
    else:
        strategy.index_data = context['index_data']
        strategy.industry_stock_counts = context['industry_stock_counts']

    experiment = risk_limits or calendar_cadence or legacy_market_context or latest_execution
    if experiment:
        proof = run / 'replays/identity/summary.json'
        if not proof.exists() or not json.loads(proof.read_text()).get('curve_byte_identical'):
            raise RuntimeError('experiments require a successful identity replay first')
    if calendar_cadence:
        import core.trading_calendar as calendar_module
        calendar_module.sessions_elapsed = lambda start, end, calendar=None: (pd.Timestamp(end) - pd.Timestamp(start)).days
    if legacy_market_context:
        idx = pd.read_csv(stock_map['sh000001'], parse_dates=['datetime'])
        idx = idx[(idx['datetime'] >= engine.FROMDATE) & (idx['datetime'] <= engine.TODATE)]
        strategy.generate_market_regime(idx)
        strategy.industry_stock_counts = {}

    store_module = load_helper('core.review_signal_store', source_root / 'strategy/core/signal_store.py')
    strategy.signal_store = store_module.SignalStore()
    strategy.signal_store.finalize(signals, membership_map=membership)
    label = label or ('_'.join(name for enabled, name in [(risk_limits, 'risk_limits'), (calendar_cadence, 'calendar'), (legacy_market_context, 'legacy_context'), (latest_execution, 'latest_execution')] if enabled) if experiment else 'identity')
    if experiment and label == 'identity':
        raise ValueError('an experiment cannot overwrite identity proof')
    if Path(label).name != label:
        raise ValueError('invalid replay label')
    out = run / 'replays' / label / 'rolling_validation_results'
    out.mkdir(parents=True, exist_ok=True)
    engine.__file__ = str(out.parent / 'bt_execution.py')  # output routing only
    result = engine._vectorized_backtest(strategy, fd, engine.FROMDATE, engine.TODATE, engine.CASH)
    pd.DataFrame(result['selections'], columns=['date', 'code', 'weight', 'score']).to_csv(out / 'portfolio_selections.csv', index=False)
    pd.DataFrame(result['realized_trades']).to_csv(out / 'trade_realized.csv', index=False)
    summary = {'label': label, 'elapsed_seconds': time.perf_counter() - started,
               'final_nav': result['final_value'], 'sharpe': result['sharpe'],
               'max_drawdown': result['max_drawdown']}
    if latest_execution:
        summary['execution_source_sha256'] = {
            str(path): hashlib.sha256((source_root / path).read_bytes()).hexdigest()
            for path in ['strategy/bt_execution.py', 'strategy/core/portfolio.py',
                         'strategy/core/strategy.py', 'strategy/core/signal_store.py']}
        reference = run / 'replays/calendar/rolling_validation_results/equity_curve.csv'
        if reference.exists():
            summary['calendar_reference_curve_byte_identical'] = (
                reference.read_bytes() == (out / 'equity_curve.csv').read_bytes())
    if not experiment and label == 'identity':
        original = signals.with_name('equity_curve.csv')
        observed = out / 'equity_curve.csv'
        equal = hashlib.sha256(original.read_bytes()).digest() == hashlib.sha256(observed.read_bytes()).digest()
        summary['curve_byte_identical'] = equal
        if not equal:
            a, b = pd.read_csv(original), pd.read_csv(observed)
            summary['max_nav_difference'] = float(np.max(np.abs(a['nav'] - b['nav'])))
            (out.parent / 'summary.json').write_text(json.dumps(summary, indent=2))
            raise RuntimeError(f'identity replay failed: {summary}')
        context_module.save_context(strategy, signals)
    (out.parent / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    return out


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True)
    p.add_argument('--risk-limits', action='store_true')
    p.add_argument('--label')
    p.add_argument('--calendar-cadence', action='store_true')
    p.add_argument('--legacy-market-context', action='store_true')
    p.add_argument('--latest-execution', action='store_true')
    args = p.parse_args()
    replay(args.run, args.risk_limits, args.label, args.calendar_cadence, args.legacy_market_context,
           args.latest_execution)
