"""
信号系统构建与验证 - 滚动验证信号质量
基于行业因子构建信号，验证样本外表现
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


# 行业最优因子配置（基于之前滚动验证结果）
INDUSTRY_FACTORS = {
    '银行Ⅱ': {
        'factors': ['trend_30', 'price_ma_30', 'ret_30', 'rsi_14', 'vol_10'],
        'direction': {'trend_30': -1, 'price_ma_30': -1, 'ret_30': -1, 'rsi_14': -1, 'vol_10': -1},  # 反转
    },
    '贵金属': {
        'factors': ['ma_cross_10_60', 'ret_30', 'mom_30', 'price_ma_60', 'vol_20'],
        'direction': {'ma_cross_10_60': 1, 'ret_30': 1, 'mom_30': 1, 'price_ma_60': 1, 'vol_20': 1},  # 趋势
    },
    '消费电子': {
        'factors': ['bb_pos_30', 'price_ma_60', 'price_pos_20', 'rsi_20', 'vol_10'],
        'direction': {'bb_pos_30': 1, 'price_ma_60': -1, 'price_pos_20': -1, 'rsi_20': 1, 'vol_10': -1},  # 混合
    },
    '电池': {
        'factors': ['ma_cross_10_60', 'price_ma_60', 'price_ma_120', 'vol_20', 'atr_ratio_20'],
        'direction': {'ma_cross_10_60': -1, 'price_ma_60': -1, 'price_ma_120': -1, 'vol_20': -1, 'atr_ratio_20': -1},  # 反转
    },
    '工业金属': {
        'factors': ['price_ma_60', 'ma_cross_10_60', 'ma_golden_20_60', 'rsi_14', 'bb_width_20'],
        'direction': {'price_ma_60': -1, 'ma_cross_10_60': -1, 'ma_golden_20_60': -1, 'rsi_14': -1, 'bb_width_20': -1},  # 反转
    },
    '基础建设': {
        'factors': ['rsi_14', 'rsi_20', 'bb_width_20', 'vol_20', 'atr_20'],
        'direction': {'rsi_14': -1, 'rsi_20': -1, 'bb_width_20': -1, 'vol_20': -1, 'atr_20': -1},  # 反转
    },
    '电网设备': {
        'factors': ['price_ma_60', 'ma_all', 'vol_30', 'atr_10', 'rsi_20'],
        'direction': {'price_ma_60': 1, 'ma_all': 1, 'vol_30': 1, 'atr_10': 1, 'rsi_20': 1},  # 趋势
    },
    '光伏设备': {
        'factors': ['price_ma_60', 'price_ma_120', 'ma_cross_10_60', 'vol_10', 'bb_width_30'],
        'direction': {'price_ma_60': 1, 'price_ma_120': 1, 'ma_cross_10_60': 1, 'vol_10': 1, 'bb_width_30': 1},  # 趋势
    },
    '通信设备': {
        'factors': ['bb_width_30', 'vol_20', 'vol_30', 'rsi_20', 'ma_cross_10_20'],
        'direction': {'bb_width_30': 1, 'vol_20': 1, 'vol_30': 1, 'rsi_20': 1, 'ma_cross_10_20': 1},  # 趋势
    },
    '电力': {
        'factors': ['trend_30', 'price_ma_30', 'rsi_14', 'rsi_20', 'bb_pos_30'],
        'direction': {'trend_30': -1, 'price_ma_30': -1, 'rsi_14': -1, 'rsi_20': -1, 'bb_pos_30': -1},  # 反转
    },
    '元件': {
        'factors': ['price_ma_60', 'ma_cross_10_60', 'price_ma_30', 'rsi_20', 'bb_pos_30'],
        'direction': {'price_ma_60': -1, 'ma_cross_10_60': -1, 'price_ma_30': 1, 'rsi_20': 1, 'bb_pos_30': 1},  # 混合
    },
    '半导体': {
        'factors': ['ret_30', 'mom_30', 'price_ma_60', 'vol_10', 'rsi_14'],
        'direction': {'ret_30': 1, 'mom_30': 1, 'price_ma_60': 1, 'vol_10': 1, 'rsi_14': 1},  # 趋势
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


# 因子计算函数
def _shift(arr, p):
    result = np.zeros_like(arr, dtype=float)
    result[p:] = arr[:-p]
    result[:p] = np.nan
    return result


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
        factors[f'bb_pos_{p}'] = ((close - lower) / (upper - lower + 1e-10)).fillna(0.5).values

    for p in [5, 10, 20, 30, 60, 120]:
        ma = pd.Series(close).rolling(p).mean()
        factors[f'price_ma_{p}'] = (close / (ma + 1e-10) - 1).fillna(0).values

    ma5 = pd.Series(close).rolling(5).mean()
    ma10 = pd.Series(close).rolling(10).mean()
    ma20 = pd.Series(close).rolling(20).mean()
    ma60 = pd.Series(close).rolling(60).mean()

    factors['ma_golden_5_10'] = (ma5 > ma10).astype(float)
    factors['ma_golden_10_20'] = (ma10 > ma20).astype(float)
    factors['ma_golden_20_60'] = (ma20 > ma60).astype(float)
    factors['ma_all'] = ((ma5 > ma20) & (ma20 > ma60)).astype(float)

    for fast, slow in [(5, 10), (5, 20), (10, 20), (10, 60)]:
        ma_fast = pd.Series(close).rolling(fast).mean()
        ma_slow = pd.Series(close).rolling(slow).mean()
        factors[f'ma_cross_{fast}_{slow}'] = (ma_fast / (ma_slow + 1e-10) - 1).fillna(0).values

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


def generate_signal(industry, factors, date_idx):
    """生成信号"""
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
                scores.append(value * direction)

    if not scores:
        return 0

    # 综合得分
    avg_score = np.mean(scores)

    # 信号阈值
    if avg_score > 0.1:
        return 1  # 买入信号
    elif avg_score < -0.1:
        return -1  # 卖出信号
    else:
        return 0  # 无信号


def rolling_validate_signals(lookback=120, forward=20, step=30, n_iterations=4):
    """滚动验证信号质量"""
    print(f"\n信号质量滚动验证: lookback={lookback}, forward={forward}, step={step}")
    print("="*70)

    # 加载股票
    codes = []
    for f in os.listdir(BACKTRADER_PATH):
        if f.endswith('_qfq.csv') and f != 'sh000001_qfq.csv':
            codes.append(f.replace('_qfq.csv', ''))

    # 获取行业
    stock_industry = {}
    for code in codes:
        ind = get_industry(code)
        if ind and ind in INDUSTRY_FACTORS:
            stock_industry[code] = ind

    # 预加载数据
    stock_data = {}
    for code in stock_industry.keys():
        df = load_data(code)
        if df is not None and len(df) >= lookback + forward + 50:
            stock_data[code] = df

    print(f"有效股票: {len(stock_data)}")

    all_results = []

    for iteration in range(n_iterations):
        start_idx = iteration * step
        end_idx = start_idx + lookback
        val_start = end_idx
        val_end = val_start + forward

        print(f"\n--- 第{iteration+1}轮: 信号生成[{start_idx}:{end_idx}], 验证[{val_start}:{val_end}] ---")

        signals = []
        returns = []

        for code, df in stock_data.items():
            industry = stock_industry[code]
            close = df['close'].values

            if len(close) < val_end:
                continue

            factors = calc_factors(df)

            # 生成信号（样本内最后一天）
            signal_date_idx = end_idx - 1
            signal = generate_signal(industry, factors, signal_date_idx)

            if signal == 0:
                continue

            # 计算未来收益
            future_ret = (close[val_start] / close[signal_date_idx] - 1) if signal_date_idx < val_start else 0

            signals.append(signal)
            returns.append(future_ret)

        if len(signals) < 20:
            print(f"  信号不足，跳过")
            continue

        signals = np.array(signals)
        returns = np.array(returns)

        # 买入信号统计
        buy_mask = signals == 1
        sell_mask = signals == -1

        n_buy = buy_mask.sum()
        n_sell = sell_mask.sum()

        buy_ret = returns[buy_mask] if n_buy > 0 else np.array([0])
        sell_ret = returns[sell_mask] if n_sell > 0 else np.array([0])

        # 胜率
        buy_win_rate = (buy_ret > 0).mean() if n_buy > 0 else 0
        sell_win_rate = (sell_ret < 0).mean() if n_sell > 0 else 0

        # 平均收益
        avg_buy_ret = buy_ret.mean() if n_buy > 0 else 0
        avg_sell_ret = sell_ret.mean() if n_sell > 0 else 0

        # 整体
        all_ret = returns.mean()
        all_win_rate = (returns > 0).mean()

        print(f"  信号数: 买入{n_buy}, 卖出{n_sell}, 总{len(signals)}")
        print(f"  买入: 胜率{buy_win_rate:.1%}, 平均收益{avg_buy_ret:+.2%}")
        print(f"  卖出: 胜率{sell_win_rate:.1%}, 平均收益{avg_sell_ret:+.2%}")
        print(f"  整体: 胜率{all_win_rate:.1%}, 平均收益{all_ret:+.2%}")

        all_results.append({
            'iteration': iteration + 1,
            'n_signals': len(signals),
            'n_buy': n_buy,
            'n_sell': n_sell,
            'buy_win_rate': buy_win_rate,
            'sell_win_rate': sell_win_rate,
            'avg_buy_ret': avg_buy_ret,
            'avg_sell_ret': avg_sell_ret,
            'all_win_rate': all_win_rate,
            'all_ret': all_ret
        })

    return all_results


def main():
    print("="*70)
    print("信号系统构建与验证")
    print("="*70)

    print("\n行业因子配置:")
    for industry, config in INDUSTRY_FACTORS.items():
        print(f"  {industry}: {config['factors'][:3]}...")

    results = rolling_validate_signals()

    # 汇总
    print("\n" + "="*70)
    print("信号质量汇总")
    print("="*70)

    print(f"\n{'轮次':<6} {'信号数':<8} {'买入':<6} {'卖出':<6} {'胜率':<8} {'买入收益':<10} {'卖出收益':<10}")
    print("-"*70)

    for r in results:
        print(f"{r['iteration']:<6} {r['n_signals']:<8} {r['n_buy']:<6} {r['n_sell']:<6} {r['all_win_rate']:<8.1%} {r['avg_buy_ret']:<10.2%} {r['avg_sell_ret']:<10.2%}")

    # 平均
    avg_signals = np.mean([r['n_signals'] for r in results])
    avg_win_rate = np.mean([r['all_win_rate'] for r in results])
    avg_buy_ret = np.mean([r['avg_buy_ret'] for r in results])
    avg_sell_ret = np.mean([r['avg_sell_ret'] for r in results])
    avg_all_ret = np.mean([r['all_ret'] for r in results])

    print("-"*70)
    print(f"{'平均':<6} {avg_signals:<8.0f} {'':<6} {'':<6} {avg_win_rate:<8.1%} {avg_buy_ret:<10.2%} {avg_sell_ret:<10.2%}")

    print("\n" + "="*70)
    print("结论")
    print("="*70)

    if avg_win_rate >= 0.55 and avg_all_ret > 0:
        print(f"✓ 信号有效: 胜率={avg_win_rate:.1%}, 平均收益={avg_all_ret:+.2%}")
    elif avg_win_rate >= 0.50:
        print(f"○ 信号一般: 胜率={avg_win_rate:.1%}, 平均收益={avg_all_ret:+.2%}")
    else:
        print(f"✗ 信号无效: 胜率={avg_win_rate:.1%}, 平均收益={avg_all_ret:+.2%}")


if __name__ == '__main__':
    main()
