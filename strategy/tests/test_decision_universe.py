"""Independent comparisons against the pre-extraction vectorized engine.

No engine import, signal generation, network, or full backtest is needed.
"""
from pathlib import Path
import textwrap

import numpy as np
import pandas as pd
import pytest

from core.decision_universe import compute_tradable_universe, load_live_universe
from core.stock_pool import _is_non_stock_series


def original_gate(close_px, volume_m, amount_m, raw_close_m, calendar, stock_codes):
    # Frozen arithmetic from bt_execution.py before extraction. Intentionally
    # independent: comparisons must not invoke the production helper twice.
    _proxy_value = close_px * volume_m * 100.0
    daily_value = np.where(amount_m > 0, amount_m, _proxy_value).astype(np.float32)
    dv = pd.DataFrame(daily_value, index=calendar, columns=stock_codes)
    avg_daily_value = dv.rolling(20, min_periods=5).mean().shift(1).values
    _vwap_px = np.where((amount_m > 0) & (volume_m > 0),
                        amount_m / np.maximum(volume_m * 100.0, 1.0),
                        close_px).astype(np.float32)
    _raw_px = np.where(np.isnan(raw_close_m), _vwap_px, raw_close_m)
    tradable = (~np.isnan(close_px)) & (_raw_px > 2.0) & (volume_m > 100) & (avg_daily_value > 5e6)
    return dict(tradable=tradable, raw_px=_raw_px, daily_value=daily_value,
                avg_daily_value=avg_daily_value, vwap_px=_vwap_px)


def assert_bits(actual, expected):
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize('layout', ['C', 'F'])
def test_full_mask_and_intermediate_arrays_are_bitwise_original(layout):
    rng = np.random.default_rng(46556)
    shape = (137, 23)
    close = np.array(rng.uniform(.01, 200, shape), dtype=np.float32, order=layout)
    volume = np.array(rng.uniform(0, 20000, shape), dtype=np.float32, order=layout)
    amount = np.array(rng.uniform(0, 1e7, shape), dtype=np.float32, order=layout)
    raw = np.array(rng.uniform(1.9, 2.1, shape), dtype=np.float32, order=layout)
    close[::13, ::3] = np.nan
    amount[::7] = 0
    amount[::17, ::2] = np.nan
    volume[::11] = 0
    raw[::3] = np.nan
    raw[::29, 0] = -0.
    calendar = pd.bdate_range('2025-01-02', periods=shape[0])
    codes = [str(i) for i in range(shape[1])]
    with np.errstate(invalid='ignore'):
        expected = original_gate(close, volume, amount, raw, calendar, codes)
        actual = compute_tradable_universe(close, volume, amount, raw, calendar, codes)
    for name, value in expected.items():
        assert_bits(getattr(actual, name), value)
    assert actual.daily_value.dtype == actual.raw_px.dtype == np.float32
    assert actual.avg_daily_value.dtype == np.float64


def test_strict_boundaries_float32_rounding_minimum_history_and_shift():
    calendar = pd.bdate_range('2026-01-05', periods=27)
    close = np.full((27, 8), 10, dtype=np.float32)
    volume = np.full_like(close, 10000)
    amount = np.full_like(close, 6e6)
    raw = close.copy()
    raw[-1, 0] = 2.
    raw[-1, 1] = 2. + 1e-8  # casts back to exactly 2 before the gate
    raw[-1, 2] = np.nextafter(np.float32(2), np.float32(3))
    volume[-1, 3] = 100
    volume[-1, 4] = np.nextafter(np.float32(100), np.float32(101))
    amount[:, 5] = 5e6
    amount[-1, 5] = 1e9  # today's amount must not lift yesterday's mean
    amount[:, 6] = np.nextafter(np.float32(5e6), np.float32(6e6))
    amount[-1, 6] = 1
    amount[:, 7] = 4e6
    amount[-1, 7] = 1e9
    result = compute_tradable_universe(close, volume, amount, raw, calendar, list('abcdefgh'))
    assert result.tradable[-1].tolist() == [False, False, True, False, True, False, True, False]
    assert not result.tradable[:5].any()
    assert result.tradable[5, 0]
    assert result.avg_daily_value[-1, 5] == 5e6
    assert result.avg_daily_value[-1, 6] == float(amount[0, 6])
    assert result.avg_daily_value[-1, 7] == 4e6


def reference_engine_load(data, raw, calendar, start, target):
    """Original matrix loader, including classification before reindex/cast."""
    codes = sorted(p.name.replace('_qfq.csv', '') for p in data.iterdir()
                   if p.name.endswith('_qfq.csv') and p.name != 'sh000001_qfq.csv'
                   and not p.name.startswith('._'))
    shape = (len(calendar), len(codes))
    close_px = np.full(shape, np.nan, dtype=np.float32)
    volume_m = np.zeros_like(close_px)
    amount_m = np.zeros_like(close_px)
    raw_close_m = np.full_like(close_px, np.nan)
    excluded = []
    for j, code in enumerate(codes):
        df = pd.read_csv(data / f'{code}_qfq.csv')
        df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
        df = df.dropna(subset=['datetime']).set_index('datetime')
        df = df[(df.index >= start) & (df.index <= target)]
        if _is_non_stock_series(df):
            excluded.append(code)
            continue
        path = raw / code / 'none.csv'
        if path.exists():
            try:
                rd = pd.read_csv(path, encoding='utf-8-sig')
                rd.columns = [str(c).strip().lstrip('\ufeff') for c in rd.columns]
                if '日期' in rd and '收盘' in rd:
                    rc = pd.Series(pd.to_numeric(rd['收盘'], errors='coerce').values,
                                   index=pd.DatetimeIndex(pd.to_datetime(rd['日期'], errors='coerce')))
                    rc = rc[rc.index.notna()]
                    rc = rc[~rc.index.duplicated()]
                    rc = rc[(rc.index >= start) & (rc.index <= target)]
                    raw_close_m[:, j] = rc.reindex(calendar).values.astype(np.float32)
            except Exception:
                pass  # engine's default raw-reader failure -> VWAP
        df = df.reindex(calendar)
        close_px[:, j] = df['close'].ffill().values.astype(np.float32)
        volume_m[:, j] = df['volume'].fillna(0).values.astype(np.float32)
        amount_m[:, j] = pd.to_numeric(df['amount'], errors='coerce').fillna(0).values.astype(np.float32)
    keep = [j for j, code in enumerate(codes) if code not in excluded]
    codes = [codes[j] for j in keep]
    arrays = [x[:, keep] for x in (close_px, volume_m, amount_m, raw_close_m)]
    return codes, arrays, original_gate(*arrays, calendar, codes), excluded


@pytest.fixture
def market(tmp_path):
    data, raw = tmp_path / 'qfq', tmp_path / 'raw'
    data.mkdir()
    raw.mkdir()
    # A weekday holiday is absent; loader must use this real-calendar fixture.
    calendar = pd.bdate_range('2026-08-03', periods=71).delete(34)
    start, target = calendar[0], calendar[-1]

    def write(code, *, close=10., volume=10000., amount=1e7, raw_close=10., dates=None):
        dates = calendar if dates is None else dates
        df = pd.DataFrame({'datetime': dates, 'open': close, 'low': close,
                           'close': close, 'volume': volume, 'amount': amount})
        df.to_csv(data / f'{code}_qfq.csv', index=False)
        if raw_close is not None:
            folder = raw / code
            folder.mkdir(exist_ok=True)
            pd.DataFrame({'日期': dates, '收盘': raw_close}).to_csv(folder / 'none.csv', index=False)

    write('sh000001')
    write('000001', close=1.)  # raw eligibility must not use frozen QFQ close
    write('000002', raw_close=2.)
    vol = np.full(len(calendar), 10000.); vol[-1] = 100
    write('000003', volume=vol)
    write('000004', amount=np.nan, raw_close=None)
    write('000005', dates=calendar[:-1])  # actual suspended/missing session
    amt = np.full(len(calendar), 4e6); amt[-1] = 1e9
    write('000006', close=5., amount=amt)
    amt = np.full(len(calendar), 6e6); amt[-1] = 1
    write('000007', amount=amt)
    write('000008', amount=2e6, raw_close=None)  # VWAP=2 (not QFQ=10)
    write('000009')
    path = raw / '000009' / 'none.csv'
    rd = pd.read_csv(path); rd.loc[len(rd)-1, '收盘'] = 2.
    pd.concat([rd, pd.DataFrame({'日期': [str(target.date()), 'invalid'], '收盘': [10., 100.]})]).to_csv(path, index=False)
    write('000010')
    (raw / '000010' / 'none.csv').write_text('date,close\n2026-01-01,10\n')
    write('000012', dates=calendar[-4:])
    write('000013', close=1., raw_close=None)
    write('000014', close=np.nan)
    # Full known prefix is non-stock, even though its last 20 rows look stock.
    amt = np.full(len(calendar), 1e4); amt[-20:] = 1e7
    write('000020', amount=amt)
    write('000021', amount=1e4, dates=calendar[-19:])  # <20 valid observations: retain
    write('000022', amount=5e6)  # classifier ratio == .5: retain
    write('000023', amount=np.nextafter(5e6, 0))  # ratio below .5 before float32: exclude
    write('sh000852', amount=1e4)  # no name-based stock-only prefix filter
    write('399006', amount=1e4)
    write('000099', amount=1e7)  # future low-ratio data must not classify today
    path = data / '000099_qfq.csv'
    future = pd.read_csv(path).iloc[:].copy()
    future['datetime'] = pd.bdate_range(target + pd.Timedelta(days=1), periods=len(future))
    future['amount'] = 1
    pd.concat([pd.read_csv(path), future]).to_csv(path, index=False)
    (data / '._bad_qfq.csv').write_text('invalid ignored file')
    return data, raw, calendar, start, target, write


def test_live_latest_matches_original_all_column_engine_bitwise(market):
    data, raw, calendar, start, target, _ = market
    codes, arrays, expected, excluded = reference_engine_load(data, raw, calendar, start, target)
    actual = load_live_universe(data, raw, calendar, start, target)
    assert actual.codes == [code for j, code in enumerate(codes) if expected['tradable'][-1, j]]
    assert actual.diagnostics['non_stock_codes'] == excluded == ['000008', '000020', '000023', '399006', 'sh000852']
    for j, code in enumerate(codes):
        row = actual.diagnostics['per_code'][code]
        for name in ('raw_px', 'daily_value', 'avg_daily_value'):
            value = actual.raw_prices[code] if name == 'raw_px' else row[name]
            assert_bits(np.asarray(value, dtype=expected[name].dtype), expected[name][-1, j])
        assert_bits(np.float32(actual.qfq_prices[code]), arrays[0][-1, j])
        assert row['tradable'] == bool(expected['tradable'][-1, j])
    assert actual.codes == ['000001', '000004', '000007', '000010', '000013', '000099']
    assert actual.qfq_prices['000005'] == 10.
    assert actual.diagnostics['per_code']['000005']['volume'] == 0
    assert actual.diagnostics['per_code']['000005']['amount'] == 0
    assert not actual.diagnostics['per_code']['000005']['raw_close_present']
    assert actual.raw_prices['000013'] == 10. and actual.qfq_prices['000013'] == 1.
    assert actual.diagnostics['per_code']['000006']['avg_daily_value'] == 4e6
    assert actual.diagnostics['per_code']['000007']['avg_daily_value'] == 6e6
    assert '000010' in actual.diagnostics['raw_read_errors']


def test_missing_amount_column_raw_failure_and_vwap_boundary(market):
    data, raw, calendar, start, target, write = market
    write('000011', raw_close=None)
    path = data / '000011_qfq.csv'
    pd.read_csv(path).drop(columns='amount').to_csv(path, index=False)
    # Only 19 rows so the non-stock classifier cannot suppress the VWAP gate.
    write('000018', amount=2e6, raw_close=None, dates=calendar[-19:])
    write('000019')
    (raw / '000019' / 'none.csv').write_bytes(b'\xff\xfe\x00')
    result = load_live_universe(data, raw, calendar, start, target)
    assert '000011' in result.codes
    assert result.diagnostics['per_code']['000011']['daily_value'] == 1e7
    assert result.raw_prices['000018'] == 2. and '000018' not in result.codes
    assert '000019' in result.codes and '000019' in result.diagnostics['raw_read_errors']


def test_live_uses_fromdate_cutoff_before_classification(market):
    data, raw, calendar, _, target, _ = market
    result = load_live_universe(data, raw, calendar, calendar[-20], target)
    assert '000020' not in result.diagnostics['non_stock_codes']
    assert '000020' in result.codes


@pytest.mark.parametrize('kind', ['duplicate', 'unordered', 'missing_target', 'nat', 'intraday', 'timezone'])
def test_live_requires_real_calendar_target_and_valid_sessions(market, kind):
    data, raw, calendar, start, target, _ = market
    invalid = {
        'duplicate': calendar.append(calendar[-1:]),
        'unordered': calendar[::-1], 'missing_target': calendar[:-1],
        'nat': calendar.append(pd.DatetimeIndex([pd.NaT])),
        'intraday': calendar + pd.Timedelta(hours=1),
        'timezone': calendar.tz_localize('Asia/Shanghai'),
    }[kind]
    with pytest.raises(ValueError, match='calendar|exchange'):
        load_live_universe(data, raw, invalid, start, target)


def test_engine_local_call_preserves_original_mask_and_log(capsys):
    # Execute only the authorized gate snippet; importing the engine would
    # initialize unrelated strategies and cannot serve as an isolated test.
    source = (Path(__file__).parents[1] / 'bt_execution.py').read_text()
    begin = source.index('    from core.decision_universe import compute_tradable_universe')
    end = source.index('    execution_available =', begin)
    calendar = pd.bdate_range('2026-01-05', periods=25)
    close = np.full((25, 2), 10., dtype=np.float32)
    raw = close.copy(); raw[-1, 1] = np.nan
    scope = dict(close_px=close, volume_m=close * 1000, amount_m=close * 1e6,
                 raw_close_m=raw, calendar=calendar, stock_codes=['000001', '000002'], np=np, pd=pd)
    expected = original_gate(close, scope['volume_m'], scope['amount_m'], raw, calendar, scope['stock_codes'])
    exec(compile(textwrap.dedent(source[begin:end]), '<engine universe gate>', 'exec'), scope)
    assert_bits(scope['tradable'], expected['tradable'])
    assert capsys.readouterr().out == '2元闸口径(#348): 真实收盘 49 格 / 兜底VWAP 1 格\n'
