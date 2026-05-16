"""
行业因子深度挖掘 - 尝试更多因子类型和参数组合

目标：
1. 尝试更多因子类型（资金流、形态、量价等）
2. 尝试不同参数组合
3. 针对每个行业找到最佳因子
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


def load_stock_data(code, ndays=180):
    """加载股票数据"""
    filepath = os.path.join(BACKTRADER_PATH, f'{code}_qfq.csv')
    if not os.path.exists(filepath):
        return None
    try:
        df = pd.read_csv(filepath)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values('datetime').tail(ndays)
        if len(df) < 80:
            return None
        return df
    except Exception:
        return None


def get_industry(code):
    """获取行业"""
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


# ============ 因子计算函数库 ============

def calc_returns(arr, period):
    """收益率"""
    result = np.zeros_like(arr, dtype=float)
    result[:] = np.nan
    result[period:] = (arr[period:] / arr[:-period]) - 1
    return result


def calc_volatility(arr, period):
    """波动率"""
    returns = np.diff(arr, prepend=arr[0]) / (arr + 1e-10)
    result = pd.Series(returns).rolling(period).std().values
    return result


def calc_rsi(arr, period):
    """RSI"""
    delta = np.diff(arr, prepend=arr[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(period).mean().values
    avg_loss = pd.Series(loss).rolling(period).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1 + rs))


def calc_ma(arr, period):
    """移动平均"""
    return pd.Series(arr).rolling(period).mean().values


def calc_ema(arr, span):
    """指数移动平均"""
    return pd.Series(arr).ewm(span=span, adjust=False).mean().values


def calc_bb(arr, period=20):
    """布林带"""
    ma = pd.Series(arr).rolling(period).mean()
    std = pd.Series(arr).rolling(period).std()
    upper = ma + 2 * std
    lower = ma - 2 * std
    width = (upper - lower) / (ma + 1e-10)
    position = (arr - lower) / (upper - lower + 1e-10)
    return width.values, position.fillna(0.5).values


def calc_momentum(arr, period):
    """动量"""
    result = np.zeros_like(arr, dtype=float)
    result[:] = np.nan
    result[period:] = arr[period:] / (arr[:-period] + 1e-10) - 1
    return result


def calc_acceleration(arr):
    """价格加速度 (二阶导数)"""
    returns = np.diff(arr, prepend=arr[0]) / (arr + 1e-10)
    accel = np.diff(returns, prepend=returns[0])
    return accel


def calc_volume_ratio(volume, periods=[3, 5, 10, 20]):
    """成交量比率"""
    result = {}
    for p in periods:
        ma = pd.Series(volume).rolling(p).mean()
        result[f'vol_ratio_{p}'] = (volume / (ma + 1e-10)).fillna(1).values
    return result


def calc_money_flow(close, volume, period=5):
    """资金流因子 (简单版)"""
    returns = np.diff(close, prepend=close[0]) / (close + 1e-10)
    money_flow = returns * volume
    flow_ma = pd.Series(money_flow).rolling(period).mean()
    flow_std = pd.Series(money_flow).rolling(period).std()
    z_score = (money_flow - flow_ma) / (flow_std + 1e-10)
    return z_score.fillna(0).values


def calc_price_volume_corr(close, volume, period=20):
    """价格成交量相关性"""
    result = np.zeros_like(close, dtype=float)
    result[:] = np.nan
    for i in range(period, len(close)):
        if i >= period:
            c = close[i-period:i]
            v = volume[i-period:i]
            if len(c) > 5 and np.std(c) > 0 and np.std(v) > 0:
                result[i] = np.corrcoef(c, v)[0, 1]
    return result


def calc_trend_strength(close, period=20):
    """趋势强度"""
    ma = pd.Series(close).rolling(period).mean()
    trend = (close - ma) / (ma + 1e-10)
    return trend.fillna(0).values


def calc_high_low_ratio(close, period=20):
    """收盘价与区间高低比"""
    high = pd.Series(close).rolling(period).max()
    low = pd.Series(close).rolling(period).min()
    ratio = (close - low) / (high - low + 1e-10)
    return ratio.fillna(0.5).values


def calc_turnover_change(volume, period=5):
    """换手率变化"""
    vol_ma = pd.Series(volume).rolling(period).mean()
    turnover_change = np.diff(vol_ma, prepend=vol_ma[0]) / (vol_ma + 1e-10)
    return turnover_change


def calc_price_gap(arr, period=1):
    """跳空因子"""
    gap = arr - np.roll(arr, period)
    gap[:period] = np.nan
    return gap


# ============ 行业因子分析 ============

def analyze_single_industry(industry, stocks, forward_period=20):
    """深入分析单个行业的最佳因子"""
    print(f"\n{'='*60}")
    print(f"行业: {industry} ({len(stocks)}只股票)")
    print(f"{'='*60}")

    all_data = []

    for code in stocks:
        df = load_stock_data(code, ndays=200)
        if df is None or len(df) < 80:
            continue

        close = df['close'].values
        high = df['high'].values if 'high' in df.columns else close
        low = df['low'].values if 'low' in df.columns else close
        volume = df['volume'].values if 'volume' in df.columns else np.ones(len(close))

        # 计算所有因子
        factors = {}

        # 收益率因子
        for p in [1, 2, 3, 5, 10, 20]:
            factors[f'ret_{p}'] = calc_returns(close, p)

        # 波动率因子
        for p in [3, 5, 10, 20, 30]:
            factors[f'vol_{p}'] = calc_volatility(close, p)

        # RSI因子
        for p in [4, 6, 8, 10, 12, 14, 20]:
            factors[f'rsi_{p}'] = calc_rsi(close, p)

        # 动量因子
        for p in [3, 5, 8, 10, 15, 20, 30]:
            factors[f'mom_{p}'] = calc_momentum(close, p)

        # 布林带因子
        for p in [10, 20, 30]:
            bb_width, bb_pos = calc_bb(close, p)
            factors[f'bb_width_{p}'] = bb_width
            factors[f'bb_pos_{p}'] = bb_pos

        # 均线因子
        for p in [5, 10, 20, 30, 60]:
            ma = calc_ma(close, p)
            factors[f'price_ma_{p}'] = (close / (ma + 1e-10)) - 1  # 相对位置
            if p >= 10:
                ma_short = calc_ma(close, p//2)
                factors[f'ma_cross_{p}'] = (ma_short / (ma + 1e-10)) - 1

        # 趋势强度
        for p in [5, 10, 20, 30]:
            factors[f'trend_{p}'] = calc_trend_strength(close, p)

        # 高低价位
        for p in [5, 10, 20, 30]:
            factors[f'highlow_{p}'] = calc_high_low_ratio(close, p)

        # 成交量因子
        vol_ratios = calc_volume_ratio(volume, [3, 5, 10, 20])
        factors.update(vol_ratios)

        # 资金流因子
        for p in [3, 5, 10]:
            factors[f'money_flow_{p}'] = calc_money_flow(close, volume, p)

        # 换手率变化
        factors['turnover_change'] = calc_turnover_change(volume, 5)

        # 计算未来收益
        fwd_ret = np.zeros_like(close, dtype=float)
        fwd_ret[:] = np.nan
        for i in range(len(close) - forward_period):
            fwd_ret[i] = close[i + forward_period] / close[i] - 1

        # 收集有效数据
        for i in range(40, len(df) - forward_period):
            if np.isnan(fwd_ret[i]) or abs(fwd_ret[i]) > 1.0:  # 过滤异常值
                continue
            row = {'forward_return': fwd_ret[i]}
            for fname, fvals in factors.items():
                if i < len(fvals) and not np.isnan(fvals[i]):
                    row[fname] = fvals[i]
            all_data.append(row)

    if len(all_data) < 150:
        print(f"  数据不足: {len(all_data)} 样本")
        return None

    df = pd.DataFrame(all_data)
    print(f"  有效样本: {len(df)}")

    # 计算每个因子的IC
    factor_cols = [c for c in df.columns if c != 'forward_return']
    ic_results = {}

    for fc in factor_cols:
        valid = ~(df[fc].isna() | df['forward_return'].isna())
        if valid.sum() < 30:
            continue
        try:
            ic = np.corrcoef(df.loc[valid, fc], df.loc[valid, 'forward_return'])[0, 1]
            if not np.isnan(ic):
                ic_results[fc] = ic
        except Exception:
            continue

    # 按绝对值排序
    sorted_factors = sorted(ic_results.items(), key=lambda x: abs(x[1]), reverse=True)

    print(f"\n  Top 20 因子 (按|IC|排序):")
    for i, (fname, ic) in enumerate(sorted_factors[:20]):
        direction = "+" if ic > 0 else "-"
        print(f"    {i+1:2d}. {fname:25s}: IC={ic:+.4f} ({direction})")

    # 按因子类型汇总
    print(f"\n  因子类型汇总:")

    categories = {
        '收益率': [k for k in ic_results.keys() if k.startswith('ret_')],
        '波动率': [k for k in ic_results.keys() if k.startswith('vol_') and 'rsi' not in k],
        'RSI': [k for k in ic_results.keys() if k.startswith('rsi_')],
        '动量': [k for k in ic_results.keys() if k.startswith('mom_')],
        '布林带': [k for k in ic_results.keys() if k.startswith('bb_')],
        '均线': [k for k in ic_results.keys() if 'ma_' in k],
        '趋势': [k for k in ic_results.keys() if k.startswith('trend_')],
        '高低': [k for k in ic_results.keys() if k.startswith('highlow_')],
        '成交量': [k for k in ic_results.keys() if 'vol_' in k],
        '资金流': [k for k in ic_results.keys() if 'money' in k or 'turnover' in k],
    }

    category_best = {}
    for cat, cols in categories.items():
        if cols:
            best = max(cols, key=lambda x: abs(ic_results.get(x, 0)))
            best_ic = ic_results.get(best, 0)
            category_best[cat] = (best, best_ic)
            print(f"    {cat:8s}: {best:20s} IC={best_ic:+.4f}")

    return {
        'sample_count': len(df),
        'all_factors': ic_results,
        'top_factors': sorted_factors[:10],
        'category_best': category_best,
    }


def main():
    print("="*70)
    print("行业因子深度挖掘 - 寻找各行业最佳因子")
    print("="*70)

    # 获取股票列表
    codes = []
    for f in os.listdir(BACKTRADER_PATH):
        if f.endswith('_qfq.csv') and f != 'sh000001_qfq.csv':
            codes.append(f.replace('_qfq.csv', ''))

    print(f"\n股票数量: {len(codes)}")

    # 按行业分组
    industry_stocks = defaultdict(list)
    for code in codes:
        ind = get_industry(code)
        if ind:
            industry_stocks[ind].append(code)

    print(f"行业数量: {len(industry_stocks)}")

    # 按股票数排序
    sorted_ind = sorted(industry_stocks.items(), key=lambda x: len(x[1]), reverse=True)

    # 分析主要行业
    results = {}
    target_industries = []

    # 重点分析行业
    for ind, stocks in sorted_ind:
        if len(stocks) >= 8:  # 至少8只股票
            target_industries.append((ind, stocks))
        if len(target_industries) >= 8:  # 最多8个行业
            break

    for ind, stocks in target_industries:
        result = analyze_single_industry(ind, stocks, forward_period=20)
        if result:
            results[ind] = result

    # 汇总
    print("\n" + "="*70)
    print("各行业最佳因子汇总")
    print("="*70)

    for ind, r in results.items():
        print(f"\n{ind}:")
        print(f"  样本数: {r['sample_count']}")
        print(f"  Top5因子:")
        for fname, ic in r['top_factors'][:5]:
            print(f"    {fname}: IC={ic:+.4f}")

    # 保存结果
    import json
    output = {}
    for ind, r in results.items():
        output[ind] = {
            'sample_count': r['sample_count'],
            'top_factors': [(f, float(i)) for f, i in r['top_factors'][:10]],
            'category_best': {k: (v[0], float(v[1])) for k, v in r['category_best'].items()},
        }

    output_path = os.path.join(BASE_DIR, 'strategy/industry_best_factors.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n结果已保存到: {output_path}")


if __name__ == '__main__':
    main()
