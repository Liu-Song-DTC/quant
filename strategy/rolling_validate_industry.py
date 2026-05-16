"""
因子IC滚动验证 - 分行业版本
每个行业单独做滚动验证
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


# ==================== 因子计算函数 ====================

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


def calc_all_factors(df):
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

    for p in [5, 10, 20]:
        factors[f'log_ret_{p}'] = np.log(close / _shift(close, p))

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
        factors[f'atr_ratio_{p}'] = (atr / (close + 1e-10)).fillna(0).values

    return factors


def calculate_ic(df_data):
    cols = [c for c in df_data.columns if c not in ['fwd', 'code', 'industry']]
    ics = {}
    for c in cols:
        valid = ~(df_data[c].isna() | df_data['fwd'].isna())
        if valid.sum() < 15:
            continue
        try:
            ic = np.corrcoef(df_data.loc[valid, c], df_data.loc[valid, 'fwd'])[0, 1]
            if not np.isnan(ic):
                ics[c] = ic
        except Exception:
            continue
    return ics


def rolling_validate_industry(industry, stock_codes, lookback=120, forward=20, step=30, n_iterations=4):
    """单个行业的滚动验证"""
    # 预加载数据
    stock_data = {}
    for code in stock_codes:
        df = load_data(code)
        if df is not None and len(df) >= lookback + forward + 50:
            stock_data[code] = df

    if len(stock_data) < 3:
        return None

    results = []

    for iteration in range(n_iterations):
        start_idx = iteration * step
        end_idx = start_idx + lookback
        val_start = end_idx
        val_end = val_start + forward

        # 样本内
        data_in = []
        for code, df in stock_data.items():
            close = df['close'].values
            if len(close) < val_end:
                continue
            factors = calc_all_factors(df)

            fwd = np.zeros_like(close, dtype=float)
            fwd[:] = np.nan
            for i in range(len(close) - forward):
                fwd[i] = close[i + forward] / close[i] - 1

            for i in range(30, end_idx):
                if np.isnan(fwd[i]) or abs(fwd[i]) > 1.0:
                    continue
                row = {'fwd': fwd[i], 'code': code}
                for k, v in factors.items():
                    if i < len(v) and not np.isnan(v[i]):
                        row[k] = v[i]
                data_in.append(row)

        # 样本外
        data_out = []
        for code, df in stock_data.items():
            close = df['close'].values
            if len(close) < val_end:
                continue
            factors = calc_all_factors(df)

            fwd = np.zeros_like(close, dtype=float)
            fwd[:] = np.nan
            for i in range(len(close) - forward):
                fwd[i] = close[i + forward] / close[i] - 1

            for i in range(val_start, val_end):
                if np.isnan(fwd[i]) or abs(fwd[i]) > 1.0:
                    continue
                row = {'fwd': fwd[i], 'code': code}
                for k, v in factors.items():
                    if i < len(v) and not np.isnan(v[i]):
                        row[k] = v[i]
                data_out.append(row)

        if len(data_in) < 100 or len(data_out) < 30:
            continue

        df_in = pd.DataFrame(data_in)
        df_out = pd.DataFrame(data_out)

        ics_in = calculate_ic(df_in)
        ics_out = calculate_ic(df_out)

        if not ics_in:
            continue

        top5_in = sorted(ics_in.items(), key=lambda x: abs(x[1]), reverse=True)[:5]

        # 样本外IC
        ic_out_list = []
        for k, v_in in top5_in:
            v_out = ics_out.get(k, 0)
            if v_out != 0:
                ic_out_list.append(abs(v_out))

        if ic_out_list:
            results.append({
                'iteration': iteration + 1,
                'sample_in': len(data_in),
                'sample_out': len(data_out),
                'avg_ic_in': np.mean([abs(v) for k, v in top5_in]),
                'avg_ic_out': np.mean(ic_out_list),
                'top5_in': top5_in,
                'ics_out': ics_out
            })

    return results


def main():
    print("="*70)
    print("因子IC滚动验证 - 分行业")
    print("="*70)

    # 加载股票
    codes = []
    for f in os.listdir(BACKTRADER_PATH):
        if f.endswith('_qfq.csv') and f != 'sh000001_qfq.csv':
            codes.append(f.replace('_qfq.csv', ''))

    # 行业分组
    industry_stocks = defaultdict(list)
    for code in codes:
        ind = get_industry(code)
        if ind:
            industry_stocks[ind].append(code)

    sorted_ind = sorted(industry_stocks.items(), key=lambda x: len(x[1]), reverse=True)

    print(f"\n股票总数: {len(codes)}")
    print(f"行业数量: {len(industry_stocks)}")

    # 主要行业
    main_industries = [(ind, stocks) for ind, stocks in sorted_ind if len(stocks) >= 5]

    print(f"\n主要行业数量: {len(main_industries)}")

    # 分行业滚动验证
    all_results = {}

    for industry, stocks in main_industries:
        print(f"\n{'='*60}")
        print(f"行业: {industry} ({len(stocks)}只)")
        print(f"{'='*60}")

        results = rolling_validate_industry(industry, stocks)

        if not results or len(results) < 2:
            print("  数据不足，跳过")
            continue

        all_results[industry] = results

        print(f"\n  {'轮次':<6} {'样本内':<8} {'样本外':<8} {'样本内|IC|':<12} {'样本外|IC|':<12}")
        print("  " + "-"*60)

        for r in results:
            print(f"  {r['iteration']:<6} {r['sample_in']:<8} {r['sample_out']:<8} {r['avg_ic_in']:<12.4f} {r['avg_ic_out']:<12.4f}")

        avg_in = np.mean([r['avg_ic_in'] for r in results])
        avg_out = np.mean([r['avg_ic_out'] for r in results])

        print("  " + "-"*60)
        print(f"  {'平均':<6} {'':<8} {'':<8} {avg_in:<12.4f} {avg_out:<12.4f}")

        # 结论
        if avg_out >= 0.08:
            status = "✓✓有效"
        elif avg_out >= 0.04:
            status = "✓有效"
        elif avg_out >= 0.02:
            status = "○一般"
        else:
            status = "✗无效"

        print(f"  结论: {status}")

    # 汇总
    print("\n" + "="*70)
    print("分行业汇总")
    print("="*70)

    print(f"\n{'行业':<15} {'股票数':<8} {'样本内|IC|':<12} {'样本外|IC|':<12} {'结论':<10}")
    print("-"*65)

    valid_industries = []

    for industry, results in all_results.items():
        n_stocks = len(industry_stocks.get(industry, []))
        avg_in = np.mean([r['avg_ic_in'] for r in results])
        avg_out = np.mean([r['avg_ic_out'] for r in results])

        if avg_out >= 0.08:
            status = "✓✓有效"
            valid_industries.append(industry)
        elif avg_out >= 0.04:
            status = "✓有效"
            valid_industries.append(industry)
        elif avg_out >= 0.02:
            status = "○一般"
        else:
            status = "✗无效"

        print(f"{industry:<15} {n_stocks:<8} {avg_in:<12.4f} {avg_out:<12.4f} {status:<10}")

    # 统计
    n_total = len(all_results)
    n_valid = len(valid_industries)

    print("-"*65)
    print(f"总计: {n_total}个行业, {n_valid}个有效 ({n_valid/n_total*100:.0f}%)")

    # 推荐行业
    print("\n" + "="*70)
    print("推荐使用行业（样本外IC>4%）")
    print("="*70)

    for industry in valid_industries:
        results = all_results[industry]
        avg_out = np.mean([r['avg_ic_out'] for r in results])
        n_stocks = len(industry_stocks.get(industry, []))

        # 找出稳定的Top因子
        top_factors = defaultdict(list)
        for r in results:
            for k, v in r['top5_in']:
                top_factors[k].append(abs(r['ics_out'].get(k, 0)))

        # 计算每个因子的平均样本外IC
        factor_scores = {k: np.mean(v) for k, v in top_factors.items() if len(v) >= 2}
        stable_factors = sorted(factor_scores.items(), key=lambda x: x[1], reverse=True)[:3]

        print(f"\n【{industry}】({n_stocks}只) IC={avg_out:.2%}")
        print(f"  稳定因子: ", end="")
        for f, score in stable_factors:
            print(f"{f}({score:.2%}) ", end="")
        print()


if __name__ == '__main__':
    main()
