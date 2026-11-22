"""Shared execution-universe arithmetic for backtest and live close decisions.

This module owns only the existing price/volume/liquidity eligibility gate.
Signal-store calendar membership and the shared ST timeline belong to callers.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .stock_pool import _is_non_stock_series


@dataclass(frozen=True)
class UniverseMatrices:
    tradable: np.ndarray
    raw_px: np.ndarray
    daily_value: np.ndarray
    avg_daily_value: np.ndarray
    vwap_px: np.ndarray


def compute_tradable_universe(close_px, volume_m, amount_m, raw_close_m,
                              calendar, stock_codes):
    """Apply the engine's original vectorized gate, preserving operation order.

    Inputs are the engine's float32 time-by-code matrices, already aligned to
    the real exchange calendar: close ffilled, volume/amount missing as zero,
    raw close NOT ffilled. Do not cast inputs, move casts, shorten the rolling
    history, or replace pandas rolling: those changes can alter boundary bits.
    """
    _proxy_value = close_px * volume_m * 100.0
    daily_value = np.where(amount_m > 0, amount_m, _proxy_value).astype(np.float32)
    dv = pd.DataFrame(daily_value, index=calendar, columns=stock_codes)
    avg_daily_value = dv.rolling(20, min_periods=5).mean().shift(1).values
    _vwap_px = np.where((amount_m > 0) & (volume_m > 0),
                        amount_m / np.maximum(volume_m * 100.0, 1.0),
                        close_px).astype(np.float32)
    _raw_px = np.where(np.isnan(raw_close_m), _vwap_px, raw_close_m)
    tradable = (~np.isnan(close_px)) & (_raw_px > 2.0) & (volume_m > 100) & (avg_daily_value > 5e6)
    return UniverseMatrices(tradable, _raw_px, daily_value, avg_daily_value, _vwap_px)


@dataclass(frozen=True)
class LiveUniverse:
    """One completed session's gate and prices (including ineligible stocks).

    ``codes`` is the sorted list passing the shared gate. Price maps cover all
    loaded stock series, so callers can also value holdings; raw_prices uses
    the same per-cell raw-close/VWAP fallback as the engine and can be NaN.
    ``diagnostics['per_code']`` explains each included/excluded stock's gate;
    ``non_stock_codes`` lists content-classified non-stock series separately.
    Prices here are decision closes, not quotes for next-session execution.
    """
    codes: list[str]
    raw_prices: dict[str, float]
    qfq_prices: dict[str, float]
    diagnostics: dict


def load_live_universe(data_path, raw_path, calendar, fromdate, target_date):
    """Load the engine universe as of target_date without current-pool filters.

    ``calendar`` must contain real, unique, ordered exchange sessions and the
    exact target session. All sessions from fromdate through target_date are
    used, retaining pandas rolling's original accumulation history. Each file
    is classified on its entire known prefix BEFORE calendar reindex/float32,
    just like the engine, never on only the trailing liquidity window.

    Missing amount cells (or an absent amount column) become zero and use the
    original proxy. Malformed/missing raw close sources fall back to VWAP, as
    in the default QFQ engine arm, with failures exposed in diagnostics.
    This loader does not implement the optional raw_execution arm.
    """
    sessions = pd.DatetimeIndex(pd.to_datetime(calendar))
    if (sessions.hasnans or sessions.has_duplicates or sessions.tz is not None
            or not sessions.is_monotonic_increasing
            or not sessions.equals(sessions.normalize())):
        raise ValueError('Universe requires unique, ordered local exchange dates')
    start, target = pd.Timestamp(fromdate), pd.Timestamp(target_date)
    sessions = sessions[(sessions >= start) & (sessions <= target)]
    if sessions.empty or sessions[-1] != target:
        raise ValueError(f'Universe calendar does not contain target session: {target_date}')

    data_path, raw_path = Path(data_path), Path(raw_path)
    source_codes = sorted(p.name.replace('_qfq.csv', '') for p in data_path.iterdir()
                          if p.name.endswith('_qfq.csv') and p.name != 'sh000001_qfq.csv'
                          and not p.name.startswith('._'))
    codes, raw_prices, qfq_prices = [], {}, {}
    excluded, raw_errors, per_code = [], {}, {}
    for code in source_codes:
        df = pd.read_csv(data_path / f'{code}_qfq.csv',
                         usecols=lambda c: c in ('datetime', 'close', 'volume', 'amount'))
        df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
        df = df.dropna(subset=['datetime']).set_index('datetime')
        df = df[(df.index >= start) & (df.index <= target)]
        if _is_non_stock_series(df):
            excluded.append(code)
            continue
        if 'amount' not in df.columns:
            df['amount'] = np.nan

        # A single column retains exactly the engine's dtype and rolling order
        # without allocating the full all-market time-by-code matrix for live.
        raw_close_m = np.full((len(sessions), 1), np.nan, dtype=np.float32)
        raw_file = raw_path / code / 'none.csv'
        if raw_file.exists():
            try:
                rd = pd.read_csv(raw_file, encoding='utf-8-sig')
                rd.columns = [str(c).strip().lstrip('\ufeff') for c in rd.columns]
                if '日期' in rd.columns and '收盘' in rd.columns:
                    rc = pd.Series(pd.to_numeric(rd['收盘'], errors='coerce').values,
                                   index=pd.DatetimeIndex(pd.to_datetime(rd['日期'], errors='coerce')))
                    rc = rc[rc.index.notna()]
                    rc = rc[~rc.index.duplicated()]
                    rc = rc[(rc.index >= start) & (rc.index <= target)]
                    raw_close_m[:, 0] = rc.reindex(sessions).values.astype(np.float32)
                else:
                    raw_errors[code] = 'missing 日期/收盘 columns'
            except Exception as exc:
                raw_errors[code] = f'{type(exc).__name__} {exc}'
        df = df.reindex(sessions)
        close_px = df['close'].ffill().values.astype(np.float32).reshape(-1, 1)
        volume_m = df['volume'].fillna(0).values.astype(np.float32).reshape(-1, 1)
        amount_m = pd.to_numeric(df['amount'], errors='coerce').fillna(0).values.astype(np.float32).reshape(-1, 1)
        result = compute_tradable_universe(close_px, volume_m, amount_m, raw_close_m,
                                           sessions, [code])
        eligible = bool(result.tradable[-1, 0])
        if eligible:
            codes.append(code)
        raw_prices[code] = float(result.raw_px[-1, 0])
        qfq_prices[code] = float(close_px[-1, 0])
        per_code[code] = {
            'tradable': eligible,
            'qfq_close': qfq_prices[code],
            'raw_price': raw_prices[code],
            'raw_close_present': bool(~np.isnan(raw_close_m[-1, 0])),
            'raw_file_present': raw_file.exists(),
            'volume': float(volume_m[-1, 0]),
            'amount': float(amount_m[-1, 0]),
            'daily_value': float(result.daily_value[-1, 0]),
            'avg_daily_value': float(result.avg_daily_value[-1, 0]),
            'close_present': bool(~np.isnan(close_px[-1, 0])),
            'price_above_2': bool(result.raw_px[-1, 0] > 2.0),
            'volume_above_100': bool(volume_m[-1, 0] > 100),
            'avg_value_above_5m': bool(result.avg_daily_value[-1, 0] > 5e6),
        }
    return LiveUniverse(codes, raw_prices, qfq_prices, {
        'fromdate': str(start.date()), 'target_date': str(target.date()),
        'calendar_sessions': len(sessions), 'source_codes': source_codes,
        'non_stock_codes': excluded, 'raw_read_errors': raw_errors,
        'per_code': per_code,
    })
