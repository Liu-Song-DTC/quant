"""
信号系统优化 - 改进信号生成逻辑
使用更合理的信号阈值和组合方式
"""

import os
import sys
import numpy as np
import pandas as pd
from collections import defaultdict

BASE_DIR = '/Users/litiancheng01/code/ltc/quant'
sys.path.insert(0, BASE_DIR)

DATA_PATH = os.path.join(BASE_DIR, 'data/stock_data')
BACKTRADER_PATH = os.path.join(DATA_PATH, 'backtrader_data')
FUNDAMENTAL_PATH = os.path.join(DATA_PATH, 'fundamental_data')


# 简化的行业因子配置（使用最稳定的因子）
INDUSTRY_FACTORS = {
    '银行Ⅱ': {
        'factors': ['price_ma_30', 'vol_10', 'atr_ratio_10'],
        'direction': {'price_ma_30': -1, 'vol_10': -1, 'atr_ratio_10': -1},
    },
    '贵金属': {
        'factors': ['vol_10', 'vol_5', 'vol_20'],
        'direction': {'vol_10': 1, 'vol_5': 1, 'vol_20': 1},
    },
    '消费电子': {
        'factors': ['rsi_14', 'rsi_20', 'vol_10'],
        'direction': {'rsi_14': 1, 'rsi_20': 1, 'vol_10': -1},
    },
    '电池': {
        'factors': ['price_ma_60', 'atr_ratio_10', 'vol_20'],
        'direction': {'price_ma_60': -1, 'atr_ratio_10': -1, 'vol_20': -1},
    },
    '工业金属': {
        'factors': ['rsi_14', 'rsi_20', 'price_pos_10'],
        'direction': {'rsi_14': -1, 'rsi_20': -1, 'price_pos_10': -1},
    },
    '基础建设': {
        'factors': ['rsi_14', 'bb_width_20', 'vol_20'],
        'direction': {'rsi_14': -1, 'bb_width_20': -1, 'vol_20': -1},
    },
    '电网设备': {
        'factors': ['rsi_20', 'atr_10', 'ma_all'],
        'direction': {'rsi_20': 1, 'atr_10': 1, 'ma_all': 1},
    },
    '光伏设备': {
        'factors': ['atr_10', 'vol_10', 'bb_width_20'],
        'direction': {'atr_10': 1, 'vol_10': 1, 'bb_width_20': 1},
    },
    '通信设备': {
        'factors': ['vol_20', 'bb_width_30', 'rsi_20'],
        'direction': {'vol_20': 1, 'bb_width_30': 1, 'rsi_20': 1},
    },
    '电力': {
        'factors': ['price_ma_30', 'rsi_14', 'trend_30'],
        'direction': {'price_ma_30': -1, 'rsi_14': -1, 'trend_30': -1},
    },
    '元件': {
        'factors': ['rsi_20', 'price_ma_60', 'atr_ratio_10'],
        'direction': {'rsi_20': 1, 'price_ma_60': -1, 'atr_ratio_10': -1},
    },
    '半导体': {
        'factors': ['vol_10', 'vol_5', 'vol_20'],
        'direction': {'vol_10': 1, 'vol_5': 1, 'vol_20': 1},
    },
}


def load_data(code, ndays=400):
    filepath = os.path.join(BACKTRADER_PATH, f'{code}_qfq.csv')
    if not os.path.exists(filepath):
        return None
    try:
        df = pd.read_csv(filepath)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values('datetime').tail(ndays)
        if len(df) < 200:
            return None
        return df
    except Exception:
        return None


def get_industry(code):
    filepath = os.path.join(FUNDAMENTAL_PATH, f'{code}.csv')
    if not os.path.exists(filepath):
        return None
    try:
        df = pd.read_csv(filepath)
        if '所处行业' in df.columns and len(df) > 0:
            ind = df.iloc[0]['所处行业']
            return str(ind).strip() if pd.notna(ind) else None
    except Exception:
        return None


def _momentum(arr, p):
    result = np.zeros_like(arr, dtype=float)
    result[:] = np.nan
    result[p:] = arr[p:] / (arr[:-p] + 1e-10) - 1
    return result


def _rsi(arr, p):
    delta = np.diff(arr, prepend=arr[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(p).mean()
    avg_loss = pd.Series(loss).rolling(p).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    return (100 - (100 / (1 + rs))).fillna(50).values


def calc_factors(df):
    close = df['close'].values
    high = df['high'].values if 'high' in df.columns else close
    low = df['low'].values if 'low' in df.columns else close
    volume = df['volume'].values if 'volume' in df.columns else np.ones(len(close))

    factors = {}

    for p in [5, 10, 20, 30]:
        factors[f'ret_{p}'] = _momentum(close, p)
        factors[f'mom_{p}'] = _momentum(close, p)
        returns = np.diff(close, prepend=close[0]) / (close + 1e-10)
        factors[f'vol_{p}'] = pd.Series(returns).rolling(p).std().values

    for p in [6, 10, 14, 20, 24]:
        factors[f'rsi_{p}'] = _rsi(close, p)

    for p in [10, 20, 30]:
        ma = pd.Series(close).rolling(p).mean()
        std = pd.Series(close).rolling(p).std()
        upper = ma + 2 * std
        lower = ma - 2 * std
        factors[f'bb_width_{p}'] = ((upper - lower) / (ma + 1e-10)).fillna(0).values

    for p in [5, 10, 20, 30, 60]:
        ma = pd.Series(close).rolling(p).mean()
        factors[f'price_ma_{p}'] = (close / (ma + 1e-10) - 1).fillna(0).values

    ma5 = pd.Series(close).rolling(5).mean()
    ma20 = pd.Series(close).rolling(20).mean()
    ma60 = pd.Series(close).rolling(60).mean()
    factors['ma_all'] = ((ma5 > ma20) & (ma20 > ma60)).astype(float)

    for p in [5, 10, 20, 30]:
        ma = pd.Series(close).rolling(p).mean()
        factors[f'trend_{p}'] = ((close - ma) / (ma + 1e-10)).fillna(0).values

    for p in [5, 10, 20]:
        high_p = pd.Series(high).rolling(p).max()
        low_p = pd.Series(low).rolling(p).min()
        factors[f'price_pos_{p}'] = ((close - low_p) / (high_p - low_p + 1e-10)).fillna(0.5).values

    for p in [5, 10]:
        vol_ma = pd.Series(volume).rolling(p).mean()
        factors[f'vol_ratio_{p}'] = (volume / (vol_ma + 1e-10)).fillna(1).values

    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = 0
    for p in [10, 20]:
        atr = pd.Series(tr).rolling(p).mean()
        factors[f'atr_{p}'] = atr.fillna(0).values
        factors[f'atr_ratio_{p}'] = (atr / (close + 1e-10)).fillna(0).values

    return factors


def normalize_factor(factor_values):
    """标准化因子"""
    arr = np.array(factor_values)
    arr = arr[~np.isnan(arr)]
    if len(arr) < 10:
        return factor_values
    mean = np.mean(arr)
    std = np.std(arr)
    if std < 1e-10:
        return factor_values
    return (factor_values - mean) / std


def generate_signal_v2(industry, factors, date_idx):
    """改进的信号生成"""
    if industry not in INDUSTRY_FACTORS:
        return 0

    config = INDUSTRY_FACTORS[industry]
    factor_list = config['factors']
    directions = config['direction']

    scores = []
    for factor in factor_list:
        if factor in factors:
            value = factors[factor][date_idx]
            if not np.isnan(value):
                direction = directions.get(factor, 1)
                # 标准化
                norm_value = normalize_factor(factors[factor])[date_idx]
                scores.append(norm_value * direction)

    if not scores:
        return 0

    # 综合得分 - 使用中位数更稳健
    avg_score = np.median(scores)

    # 信号阈值（更严格）
    if avg_score > 0.5:
        return 1  # 强买入
    elif avg_score < -0.5:
        return -1  # 强卖出
    elif avg_score > 0.2:
        return 0.5  # 弱买入
    elif avg_score < -0.2:
        return -0.5  # 弱卖出
    else:
        return 0


def rolling_validate_v2(lookback=150, forward=20, step=30, n_iterations=4):
    """滚动验证"""
    print(f"\n滚动验证: lookback={lookback}, forward={forward}, step={step}")

    codes = []
    for f in os.listdir(BACKTRADER_PATH):
        if f.endswith('_qfq.csv') and f != 'sh000001_qfq.csv':
            codes.append(f.replace('_qfq.csv', ''))

    stock_industry = {}
    for code in codes:
        ind = get_industry(code)
        if ind and ind in INDUSTRY_FACTORS:
            stock_industry[code] = ind

    stock_data = {}
    for code in stock_industry.keys():
        df = load_data(code)
        if df is not None and len(df) >= lookback + forward + 50:
            stock_data[code] = df

    print(f"有效股票: {len(stock_data)}")

    results = []

    for iteration in range(n_iterations):
        start_idx = iteration * step
        end_idx = start_idx + lookback
        val_start = end_idx
        val_end = val_start + forward

        signals = []
        returns = []
        signal_strengths = []

        for code, df in stock_data.items():
            industry = stock_industry[code]
            close = df['close'].values

            if len(close) < val_end:
                continue

            factors = calc_factors(df)
            signal_date_idx = end_idx - 1
            signal = generate_signal_v2(industry, factors, signal_date_idx)

            if signal == 0:
                continue

            future_ret = (close[val_start] / close[signal_date_idx] - 1) if signal_date_idx < val_start else 0

            signals.append(signal)
            returns.append(future_ret)
            signal_strengths.append(abs(signal))

        if len(signals) < 20:
            continue

        signals = np.array(signals)
        returns = np.array(returns)
        signal_strengths = np.array(signal_strengths)

        # 按信号强度分层
        strong_mask = signal_strengths >= 0.5
        weak_mask = (signal_strengths > 0) & (signal_strengths < 0.5)

        print(f"\n--- 第{iteration+1}轮 ---")
        print(f"  总信号: {len(signals)} (强:{strong_mask.sum()}, 弱:{weak_mask.sum()})")

        # 整体统计
        all_win = (returns > 0).mean()
        all_ret = returns.mean()
        print(f"  整体: 胜率{all_win:.1%}, 平均收益{all_ret:+.2%}")

        # 强信号
        if strong_mask.sum() > 5:
            strong_ret = returns[strong_mask]
            strong_win = (strong_ret > 0).mean()
            strong_avg = strong_ret.mean()
            print(f"  强信号: 胜率{strong_win:.1%}(n={strong_mask.sum()}), 收益{strong_avg:+.2%}")

        # 弱信号
        if weak_mask.sum() > 5:
            weak_ret = returns[weak_mask]
            weak_win = (weak_ret > 0).mean()
            weak_avg = weak_ret.mean()
            print(f"  弱信号: 胜率{weak_win:.1%}(n={weak_mask.sum()}), 收益{weak_avg:+.2%}")

        results.append({
            'n_signals': len(signals),
            'n_strong': strong_mask.sum(),
            'all_win': all_win,
            'all_ret': all_ret,
            'strong_win': (returns[strong_mask] > 0).mean() if strong_mask.sum() > 0 else 0,
            'strong_ret': returns[strong_mask].mean() if strong_mask.sum() > 0 else 0,
        })

    return results


def main():
    print("="*70)
    print("信号系统优化验证")
    print("="*70)

    results = rolling_validate_v2()

    print("\n" + "="*70)
    print("汇总")
    print("="*70)

    for r in results:
        print(f"信号数:{r['n_signals']:>3} 强:{r['n_strong']:>2} 胜率:{r['all_win']:>6.1%} 收益:{r['all_ret']:>+7.2%}")

    avg_win = np.mean([r['all_win'] for r in results])
    avg_ret = np.mean([r['all_ret'] for r in results])
    avg_strong = np.mean([r['strong_ret'] for r in results if r['n_strong'] > 0])

    print(f"\n平均: 胜率{avg_win:.1%}, 平均收益{avg_ret:+.2%}")
    if avg_strong != 0:
        print(f"强信号平均收益: {avg_strong:+.2%}")


if __name__ == '__main__':
    main()
