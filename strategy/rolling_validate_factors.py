"""
因子IC滚动验证 - 更稳健的验证方法
每次用过去N天挖掘因子，在后续M天验证，滚动多次
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
    """加载更多天数"""
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
    """计算所有扩展因子"""
    close = df['close'].values
    high = df['high'].values if 'high' in df.columns else close
    low = df['low'].values if 'low' in df.columns else close
    volume = df['volume'].values if 'volume' in df.columns else np.ones(len(close))
    open_arr = df['open'].values if 'open' in df.columns else close

    factors = {}

    # 核心因子
    for p in [5, 10, 20, 30]:
        factors[f'ret_{p}'] = _momentum(close, p)
        factors[f'mom_{p}'] = _momentum(close, p)
        factors[f'vol_{p}'] = pd.Series(np.diff(close, prepend=close[0]) / (close + 1e-10)).rolling(p).std().values

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

    # ATR
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
    """计算IC"""
    cols = [c for c in df_data.columns if c not in ['fwd', 'code', 'industry']]
    ics = {}
    for c in cols:
        valid = ~(df_data[c].isna() | df_data['fwd'].isna())
        if valid.sum() < 20:
            continue
        try:
            ic = np.corrcoef(df_data.loc[valid, c], df_data.loc[valid, 'fwd'])[0, 1]
            if not np.isnan(ic):
                ics[c] = ic
        except Exception:
            continue
    return ics


def rolling_validation(lookback=120, forward=20, step=20, n_iterations=5):
    """滚动验证"""
    print(f"\n滚动验证参数: lookback={lookback}, forward={forward}, step={step}")
    print("="*70)

    # 加载所有股票
    codes = []
    for f in os.listdir(BACKTRADER_PATH):
        if f.endswith('_qfq.csv') and f != 'sh000001_qfq.csv':
            codes.append(f.replace('_qfq.csv', ''))

    print(f"\n加载 {len(codes)} 只股票数据...")

    # 预加载所有股票数据
    all_stock_data = {}
    for code in codes:
        df = load_data(code)
        if df is not None and len(df) >= lookback + forward + 50:
            all_stock_data[code] = {
                'df': df,
                'industry': get_industry(code)
            }

    print(f"有效股票: {len(all_stock_data)}")

    # 滚动验证
    results = []

    for iteration in range(n_iterations):
        # 计算窗口位置
        start_idx = iteration * step
        end_idx = start_idx + lookback
        val_start = end_idx
        val_end = val_start + forward

        print(f"\n--- 第{iteration+1}轮: 样本内[{start_idx}:{end_idx}], 样本外[{val_start}:{val_end}] ---")

        # 收集样本内数据
        data_in = []
        for code, data in all_stock_data.items():
            df = data['df']
            industry = data['industry']
            if df is None or len(df) < val_end:
                continue

            close = df['close'].values
            factors = calc_all_factors(df)

            fwd = np.zeros_like(close, dtype=float)
            fwd[:] = np.nan
            for i in range(len(close) - forward):
                fwd[i] = close[i + forward] / close[i] - 1

            for i in range(40, end_idx):
                if np.isnan(fwd[i]) or abs(fwd[i]) > 1.0:
                    continue
                row = {'fwd': fwd[i], 'code': code, 'industry': industry}
                for k, v in factors.items():
                    if i < len(v) and not np.isnan(v[i]):
                        row[k] = v[i]
                data_in.append(row)

        # 收集样本外数据
        data_out = []
        for code, data in all_stock_data.items():
            df = data['df']
            industry = data['industry']
            if df is None or len(df) < val_end:
                continue

            close = df['close'].values
            factors = calc_all_factors(df)

            fwd = np.zeros_like(close, dtype=float)
            fwd[:] = np.nan
            for i in range(len(close) - forward):
                fwd[i] = close[i + forward] / close[i] - 1

            for i in range(val_start, val_end):
                if np.isnan(fwd[i]) or abs(fwd[i]) > 1.0:
                    continue
                row = {'fwd': fwd[i], 'code': code, 'industry': industry}
                for k, v in factors.items():
                    if i < len(v) and not np.isnan(v[i]):
                        row[k] = v[i]
                data_out.append(row)

        if len(data_in) < 500 or len(data_out) < 500:
            print(f"  数据不足，跳过")
            continue

        # 计算IC
        df_in = pd.DataFrame(data_in)
        df_out = pd.DataFrame(data_out)

        ics_in = calculate_ic(df_in)
        ics_out = calculate_ic(df_out)

        # 取Top10因子
        top10_in = sorted(ics_in.items(), key=lambda x: abs(x[1]), reverse=True)[:10]

        print(f"  样本内: {len(df_in)}, 样本外: {len(df_out)}")

        # 输出Top5
        print(f"  样本内Top5: ", end="")
        for k, v in top10_in[:5]:
            print(f"{k[:15]}{v:+.3f} ", end="")
        print()

        print(f"  样本外Top5: ", end="")
        top5_out = sorted(ics_out.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        for k, v in top5_out:
            print(f"{k[:15]}{v:+.3f} ", end="")
        print()

        # 计算Top10在样本外的IC
        ic_out_top10 = []
        for k, v_in in top10_in:
            v_out = ics_out.get(k, 0)
            if v_out != 0:
                ic_out_top10.append(v_out)

        if ic_out_top10:
            avg_ic_out = np.mean(ic_out_top10)
            results.append({
                'iteration': iteration + 1,
                'sample_in': len(df_in),
                'sample_out': len(df_out),
                'avg_ic_in': np.mean([abs(v) for k, v in top10_in]),
                'avg_ic_out': np.mean([abs(v) for v in ic_out_top10]),
                'top10_ic_out': ic_out_top10
            })
            print(f"  Top10平均|IC|: 样本内{np.mean([abs(v) for k,v in top10_in]):.4f} -> 样本外{avg_ic_out:.4f}")

    return results


def main():
    print("="*70)
    print("因子IC滚动验证 - 稳健性检验")
    print("="*70)

    # 测试不同参数
    configs = [
        {'lookback': 120, 'forward': 20, 'step': 20, 'n_iterations': 5},
    ]

    for config in configs:
        results = rolling_validation(**config)

    # 汇总
    print("\n" + "="*70)
    print("滚动验证汇总")
    print("="*70)

    print(f"\n{'轮次':<6} {'样本内':<10} {'样本外':<10} {'样本内|IC|':<12} {'样本外|IC|':<12}")
    print("-"*60)

    all_ic_out = []
    for r in results:
        print(f"{r['iteration']:<6} {r['sample_in']:<10} {r['sample_out']:<10} {r['avg_ic_in']:<12.4f} {r['avg_ic_out']:<12.4f}")
        all_ic_out.extend(r['top10_ic_out'])

    print("-"*60)
    print(f"{'平均':<6} {'':<10} {'':<10} {np.mean([r['avg_ic_in'] for r in results]):<12.4f} {np.mean([r['avg_ic_out'] for r in results]):<12.4f}")

    print("\n" + "="*70)
    print("结论")
    print("="*70)

    avg_ic_out = np.mean([r['avg_ic_out'] for r in results])
    if avg_ic_out >= 0.04:
        print(f"✓ 因子有效: 滚动样本外IC = {avg_ic_out:.4f} >= 4%")
    elif avg_ic_out >= 0.02:
        print(f"○ 因子一般: 滚动样本外IC = {avg_ic_out:.4f} >= 2%")
    else:
        print(f"✗ 因子可能无效: 滚动样本外IC = {avg_ic_out:.4f} < 2%")

    # 对比Top因子
    print("\n" + "="*70)
    print("各轮次Top3因子样本外IC")
    print("="*70)

    for r in results:
        print(f"\n第{r['iteration']}轮:")
        top3 = sorted(zip(['因子1','因子2','因子3'], r['top10_ic_out'][:3]), key=lambda x: abs(x[1]), reverse=True)
        for name, ic in top3:
            print(f"  {name}: {ic:+.4f}")


if __name__ == '__main__':
    main()
