"""
简化信号系统 - 基于全市场有效的因子
不使用复杂的行业配置，使用最稳健的通用因子
"""

import os
import sys
import numpy as np
import pandas as pd
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = '/Users/litiancheng01/code/ltc/quant'
sys.path.insert(0, BASE_DIR)

DATA_PATH = os.path.join(BASE_DIR, 'data/stock_data')
BACKTRADER_PATH = os.path.join(DATA_PATH, 'backtrader_data')


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
    except:
        return None


def _momentum(arr, p):
    result = np.zeros_like(arr, dtype=float)
    result[:] = np.nan
    result[p:] = arr[p:] / (arr[:-p] + 1e-10) - 1
    return result


def calc_factors(df):
    close = df['close'].values
    volume = df['volume'].values if 'volume' in df.columns else np.ones(len(close))

    factors = {}

    # 最稳健的因子：动量 + 波动率
    for p in [5, 10, 20, 30]:
        factors[f'ret_{p}'] = _momentum(close, p)
        returns = np.diff(close, prepend=close[0]) / (close + 1e-10)
        factors[f'vol_{p}'] = pd.Series(returns).rolling(p).std().values

    # 成交量
    for p in [5, 10]:
        vol_ma = pd.Series(volume).rolling(p).mean()
        factors[f'vol_ratio_{p}'] = (volume / (vol_ma + 1e-10)).fillna(1).values

    return factors


def rolling_validate_simple(lookback=150, forward=20, step=30, n_iterations=5):
    """简化的滚动验证"""
    print(f"\n简化信号验证: lookback={lookback}, forward={forward}, step={step}")
    print("="*70)

    codes = []
    for f in os.listdir(BACKTRADER_PATH):
        if f.endswith('_qfq.csv') and f != 'sh000001_qfq.csv':
            codes.append(f.replace('_qfq.csv', ''))

    print(f"股票数: {len(codes)}")

    stock_data = {}
    for code in codes:
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

        # 计算样本内因子统计
        all_ret = []
        all_vol = []

        for code, df in stock_data.items():
            close = df['close'].values
            if len(close) < end_idx:
                continue
            factors = calc_factors(df)

            for i in range(30, end_idx):
                if not np.isnan(factors['ret_20'][i]):
                    all_ret.append(factors['ret_20'][i])
                if not np.isnan(factors['vol_20'][i]):
                    all_vol.append(factors['vol_20'][i])

        if len(all_ret) < 100:
            continue

        # 计算阈值
        ret_median = np.median(all_ret)
        vol_median = np.median(all_vol)
        ret_75 = np.percentile(all_ret, 75)
        vol_75 = np.percentile(all_vol, 75)

        # 生成信号
        signals = []
        returns = []

        for code, df in stock_data.items():
            close = df['close'].values
            if len(close) < val_end:
                continue

            factors = calc_factors(df)
            signal_idx = end_idx - 1

            ret_20 = factors['ret_20'][signal_idx]
            vol_20 = factors['vol_20'][signal_idx]

            if np.isnan(ret_20) or np.isnan(vol_20):
                continue

            # 简单策略：动量>75分位 AND 波动率>中位数 -> 买入
            # 动量<25分位 AND 波动率>中位数 -> 卖出

            # 重新计算分位
            ret_25 = np.percentile(all_ret, 25)

            if ret_20 > ret_75 and vol_20 > vol_median:
                signal = 1
            elif ret_20 < ret_25 and vol_20 > vol_median:
                signal = -1
            else:
                signal = 0

            if signal == 0:
                continue

            future_ret = (close[val_start] / close[signal_idx] - 1)

            signals.append(signal)
            returns.append(future_ret)

        if len(signals) < 20:
            continue

        signals = np.array(signals)
        returns = np.array(returns)

        buy_mask = signals == 1
        sell_mask = signals == -1

        buy_ret = returns[buy_mask]
        sell_ret = returns[sell_mask]

        buy_win = (buy_ret > 0).mean() if len(buy_ret) > 0 else 0
        sell_win = (sell_ret < 0).mean() if len(sell_ret) > 0 else 0

        print(f"\n--- 第{iteration+1}轮 [{start_idx}:{end_idx}] -> [{val_start}:{val_end}] ---")
        print(f"  买入: n={len(buy_ret)}, 胜率={buy_win:.1%}, 平均收益={buy_ret.mean():+.2%}")
        print(f"  卖出: n={len(sell_ret)}, 胜率={sell_win:.1%}, 平均收益={sell_ret.mean():+.2%}")
        print(f"  整体: n={len(signals)}, 胜率={(returns > 0).mean():.1%}, 平均收益={returns.mean():+.2%}")

        results.append({
            'n_buy': len(buy_ret),
            'n_sell': len(sell_ret),
            'buy_win': buy_win,
            'sell_win': sell_win,
            'buy_ret': buy_ret.mean() if len(buy_ret) > 0 else 0,
            'sell_ret': sell_ret.mean() if len(sell_ret) > 0 else 0,
            'all_win': (returns > 0).mean(),
            'all_ret': returns.mean()
        })

    # 汇总
    print("\n" + "="*70)
    print("汇总")
    print("="*70)

    for r in results:
        print(f"买:{r['n_buy']:>2} 卖:{r['n_sell']:>2} | 买胜率:{r['buy_win']:>5.1%} 收益:{r['buy_ret']:>+6.2%} | 整体胜率:{r['all_win']:>5.1%} 收益:{r['all_ret']:>+6.2%}")

    avg_buy_win = np.mean([r['buy_win'] for r in results])
    avg_buy_ret = np.mean([r['buy_ret'] for r in results])
    avg_all_win = np.mean([r['all_win'] for r in results])
    avg_all_ret = np.mean([r['all_ret'] for r in results])

    print(f"\n平均: 买入胜率{avg_buy_win:.1%}, 买入收益{avg_buy_ret:+.2%}")
    print(f"      整体胜率{avg_all_win:.1%}, 整体收益{avg_all_ret:+.2%}")

    return results


if __name__ == '__main__':
    print("="*70)
    print("简化信号系统验证")
    print("="*70)

    results = rolling_validate_simple()
