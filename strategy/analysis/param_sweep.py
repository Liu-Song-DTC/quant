#!/usr/bin/env python
"""参数扫描 — 复用已有信号, 只跑 vectorized backtest。

用法:
    # 1. 先跑一次全量回测生成 signals.csv
    # 2. 然后:
    python analysis/param_sweep.py --signals rolling_validation_results/backtest_signals.csv

输出:
    rolling_validation_results/param_sweep_results.csv  (每行一个参数组合的指标)
"""

import os, sys, json, argparse
import pandas as pd
import numpy as np
from datetime import date as date_type
from collections import deque
from copy import deepcopy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.config_loader import load_config
from core.portfolio import PortfolioConstructor
from core.signal_store import SignalStore
from core.signal import Signal
from core.strategy import Strategy


def load_signals_from_csv(csv_path: str) -> SignalStore:
    """从 CSV 加载信号到 SignalStore (复用 bt_execution 的 SignalStore.finalize)"""
    store = SignalStore()
    store.finalize(csv_path)
    return store


def run_backtest_from_signals(signals_csv: str, portfolio_overrides: dict,
                               fromdate: str = '2021-01-01',
                               todate: str = '2026-05-30') -> dict:
    """从已有信号CSV运行向量化回测, 仅覆盖组合参数。

    portfolio_overrides 示例:
        {'min_confidence': 0.80, 'max_positions': 6, 'position_stop_loss': 0.10}
    """
    # 注入参数到 YAML 配置 (内存级别, 不写盘)
    cfg = load_config()
    for key, value in portfolio_overrides.items():
        if key in ('min_confidence', 'min_absolute_score', 'min_rank_pct'):
            cfg.set(f'portfolio.selection.{key}', value)
        elif key in ('max_positions',):
            cfg.set(f'portfolio.params.{key}', value)
        elif key in ('position_stop_loss', 'portfolio_stop_loss'):
            cfg.set(f'portfolio.{key}', value)

    # 加载信号
    store = load_signals_from_csv(signals_csv)

    # 构造最小 Strategy + Portfolio
    strategy = Strategy(init_cash=100000)
    strategy.signal_store = store
    strategy.portfolio = PortfolioConstructor()

    # 加载价格数据 (与 bt_execution._vectorized_backtest 相同逻辑)
    DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             '..', 'data', 'stock_data', 'backtrader_data') + '/'
    FROMDATE = fromdate
    TODATE = todate
    CASH = cfg.get('backtest.cash', 100000.0)
    COMMISSION = cfg.get('backtest.commission', 0.0001)
    STAMP_TAX = cfg.get('backtest.stamp_tax', 0.0005)

    # 从信号CSV获取股票清单
    sig_df = pd.read_csv(signals_csv, usecols=['code'], dtype={'code': str})
    stock_codes_all = sorted(sig_df['code'].str.zfill(6).unique().tolist())
    print(f"[参数扫描] 信号: {len(sig_df)}条, {len(stock_codes_all)}只股票")

    # 加载价格数据
    stock_codes = []
    for c in stock_codes_all:
        fp = os.path.join(DATA_PATH, f'{c}_qfq.csv')
        if not os.path.exists(fp):
            fp = os.path.join(DATA_PATH, f'{c}_hfq.csv')
        if os.path.exists(fp):
            stock_codes.append(c)
    print(f"[参数扫描] 价格数据: {len(stock_codes)}只股票")

    calendar = pd.bdate_range(start=FROMDATE, end=TODATE)
    n_dates = len(calendar)
    code_to_idx = {c: i for i, c in enumerate(stock_codes)}

    # 加载价格矩阵
    close_px = np.full((n_dates, len(stock_codes)), np.nan, dtype=np.float32)
    volume_m = np.zeros_like(close_px)
    for j, code in enumerate(stock_codes):
        fp = os.path.join(DATA_PATH, f'{code}_qfq.csv')
        if not os.path.exists(fp):
            fp = os.path.join(DATA_PATH, f'{code}_hfq.csv')
        if not os.path.exists(fp):
            continue
        df = pd.read_csv(fp, parse_dates=['datetime'], index_col='datetime',
                        usecols=['datetime', 'close', 'volume'])
        df = df[df.index >= pd.Timestamp(FROMDATE)]
        df = df[df.index <= pd.Timestamp(TODATE)]
        df = df.reindex(calendar)
        close_px[:, j] = df['close'].ffill().values.astype(np.float32)
        volume_m[:, j] = df['volume'].fillna(0).values.astype(np.float32)
    del df

    # 流动性过滤
    daily_value = close_px * volume_m
    dv = pd.DataFrame(daily_value, index=calendar, columns=stock_codes)
    avg_daily_value = dv.rolling(20, min_periods=5).mean().values
    tradable_mat = (~np.isnan(close_px)) & (close_px > 2.0) & (volume_m > 100) & (avg_daily_value > 5e6)

    # ADV 矩阵
    _vol_df = pd.DataFrame(volume_m, index=calendar, columns=stock_codes)
    _adv_mat = _vol_df.rolling(20, min_periods=5).mean().shift(1).values
    _adv_mat = np.nan_to_num(_adv_mat, nan=0.0)

    # 冲击成本
    cm_config = cfg.get('cost_model', {})
    impact_enabled = cm_config.get('impact_cost_enabled', True)
    impact_base = cm_config.get('impact_cost_base', 0.0003)
    impact_exp = cm_config.get('impact_cost_exponent', 0.5)
    impact_min = cm_config.get('min_impact_cost', 0.00005)

    def _impact(adv, size):
        if not impact_enabled or adv <= 0:
            return 0.0
        return max(impact_base * (abs(size) / max(adv, 1)) ** impact_exp, impact_min)

    cash = float(CASH)
    positions = np.zeros(len(stock_codes), dtype=np.int32)
    nav = np.full(n_dates, np.nan)
    daily_ret = np.full(n_dates - 1, np.nan)
    cost_tracker = {}
    tplus1 = cfg.get('cost_model', {}).get('t_plus_1_enabled', True)
    _today_buys = set()

    for i in range(n_dates):
        date = calendar[i].date()
        px_today = close_px[i]
        ok = tradable_mat[i]
        _today_buys.clear()

        pos_value = float(np.dot(positions.astype(np.float64), np.nan_to_num(px_today, 0)))
        nav[i] = cash + pos_value
        if i > 0:
            daily_ret[i - 1] = (nav[i] - nav[i - 1]) / max(nav[i - 1], 1.0)

        universe = [c for c in stock_codes if ok[code_to_idx.get(c, -1)]]
        prices = {c: float(px_today[code_to_idx[c]]) for c in universe if code_to_idx.get(c) is not None}
        cur_pos = {}
        for j in range(len(stock_codes)):
            if positions[j] > 0 and ok[j]:
                cur_pos[stock_codes[j]] = float(positions[j]) * float(px_today[j])

        try:
            target = strategy.generate_positions(
                date=date, universe=universe, current_positions=cur_pos,
                cash=cash, prices=prices, cost=cost_tracker)
        except Exception:
            target = {}

        for code, tv in target.items():
            j = code_to_idx.get(code)
            if j is None or not ok[j]:
                continue
            px = float(px_today[j])
            buy_px = px
            if i > 0:
                prev_px = close_px[i-1, j]
                if not np.isnan(prev_px) and prev_px > 0 and (px / prev_px - 1) > 0.095:
                    buy_px = px * 1.03
            target_shares = int(tv / buy_px / 100) * 100
            curr_shares = int(positions[j])
            diff = target_shares - curr_shares
            if diff >= 100:
                impact = _impact(_adv_mat[i, j], diff)
                cost = diff * buy_px * (1.0 + COMMISSION + impact)
                if cost <= cash:
                    cash -= cost
                    positions[j] = target_shares
                    if tplus1:
                        _today_buys.add(code)
            elif diff <= -100:
                if tplus1 and code in _today_buys:
                    continue
                sell_px = px
                if i > 0:
                    prev_px = close_px[i-1, j]
                    if not np.isnan(prev_px) and prev_px > 0 and (px / prev_px - 1) < -0.095:
                        sell_px = px * 0.97
                impact = _impact(_adv_mat[i, j], abs(diff))
                cash += abs(diff) * sell_px * (1.0 - COMMISSION - STAMP_TAX - impact)
                positions[j] = target_shares

    # 计算指标
    total_return = nav[-1] / CASH - 1.0
    daily_ret_clean = np.nan_to_num(daily_ret, 0.0)
    mean_daily = np.mean(daily_ret_clean)
    std_daily = np.std(daily_ret_clean)
    sharpe = mean_daily / max(std_daily, 1e-10) * np.sqrt(252)

    peak = np.maximum.accumulate(nav)
    dd = (nav - peak) / peak
    max_dd = float(np.min(dd))

    annual_rets = {}
    for yr in sorted(set(d.year for d in calendar)):
        mask = np.array([d.year == yr for d in calendar])
        yr_nav = nav[mask]
        if len(yr_nav) > 1:
            annual_rets[yr] = float(yr_nav[-1] / yr_nav[0] - 1.0)

    return {
        'sharpe': round(sharpe, 4),
        'max_drawdown': round(abs(max_dd), 4),
        'total_return': round(total_return, 4),
        'annual_returns': annual_rets,
        'mean_daily': round(mean_daily, 6),
        'std_daily': round(std_daily, 6),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--signals', required=True, help='backtest_signals.csv 路径')
    parser.add_argument('--fromdate', default='2021-01-01')
    parser.add_argument('--todate', default='2026-05-30')
    args = parser.parse_args()

    # ── 参数网格 ──
    sweeps = [
        # 1. 单变量扫参
        {'name': 'max_positions=4',  'overrides': {'max_positions': 4}},
        {'name': 'max_positions=5',  'overrides': {'max_positions': 5}},
        {'name': 'max_positions=6',  'overrides': {'max_positions': 6}},
        {'name': 'max_positions=8',  'overrides': {'max_positions': 8}},
        {'name': 'min_confidence=0.70', 'overrides': {'min_confidence': 0.70}},
        {'name': 'min_confidence=0.75', 'overrides': {'min_confidence': 0.75}},
        {'name': 'min_confidence=0.80', 'overrides': {'min_confidence': 0.80}},
        {'name': 'min_confidence=0.85', 'overrides': {'min_confidence': 0.85}},
        {'name': 'min_confidence=0.90', 'overrides': {'min_confidence': 0.90}},
        {'name': 'abs_score=-0.05',  'overrides': {'min_absolute_score': -0.05}},
        {'name': 'abs_score=0.0',    'overrides': {'min_absolute_score': 0.0}},
        {'name': 'abs_score=0.05',   'overrides': {'min_absolute_score': 0.05}},
        {'name': 'stop_loss=0.07',   'overrides': {'position_stop_loss': 0.07}},
        {'name': 'stop_loss=0.08',   'overrides': {'position_stop_loss': 0.08}},
        {'name': 'stop_loss=0.10',   'overrides': {'position_stop_loss': 0.10}},
        {'name': 'stop_loss=0.12',   'overrides': {'position_stop_loss': 0.12}},
        {'name': 'pfolio_stop=0.08', 'overrides': {'portfolio_stop_loss': 0.08}},
        {'name': 'pfolio_stop=0.10', 'overrides': {'portfolio_stop_loss': 0.10}},
        {'name': 'pfolio_stop=0.12', 'overrides': {'portfolio_stop_loss': 0.12}},
    ]

    print(f"参数扫描: {len(sweeps)} 组")
    results = []
    for i, s in enumerate(sweeps):
        print(f"\n[{i+1}/{len(sweeps)}] {s['name']}...")
        metrics = run_backtest_from_signals(args.signals, s['overrides'],
                                            args.fromdate, args.todate)
        row = {'name': s['name'], **s['overrides'], **metrics}
        # 展开年收益
        for yr, ret in metrics.pop('annual_returns', {}).items():
            row[f'ret_{yr}'] = round(ret, 4)
        row.update(metrics)
        results.append(row)

    # 输出
    df = pd.DataFrame(results)
    out_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_path = os.path.join(out_dir, 'rolling_validation_results', 'param_sweep_results.csv')
    df.to_csv(out_path, index=False)
    print(f"\n结果已保存: {out_path}")

    # 简要排名
    print("\n=== Top 10 by Sharpe ===")
    top = df.sort_values('sharpe', ascending=False).head(10)
    for _, r in top.iterrows():
        print(f"  {r['name']:30s}  Sharpe={r['sharpe']:.4f}  MaxDD={r['max_drawdown']:.2%}  "
              f"Ret={r['total_return']:.2%}")


if __name__ == '__main__':
    main()
