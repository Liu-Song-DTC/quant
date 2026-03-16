"""
因子IC验证 - 样本内/样本外验证
把数据分成两半，前一半挖掘因子，后一半验证
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
FUNDAMENTAL_PATH = os.path.join(DATA_PATH, 'fundamental_data')


def load_data(code, ndays=250):
    """加载更多天数，方便切分"""
    filepath = os.path.join(BACKTRADER_PATH, f'{code}_qfq.csv')
    if not os.path.exists(filepath):
        return None
    try:
        df = pd.read_csv(filepath)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values('datetime').tail(ndays)
        if len(df) < 150:
            return None
        return df
    except:
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
    except:
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

    # 1. 收益率因子
    for p in [1, 2, 3, 5, 8, 10, 15, 20, 30]:
        factors[f'ret_{p}'] = _momentum(close, p)

    # 2. 对数收益率
    for p in [5, 10, 20]:
        factors[f'log_ret_{p}'] = np.log(close / _shift(close, p))

    # 3. 波动率因子
    returns = np.diff(close, prepend=close[0]) / (close + 1e-10)
    for p in [3, 5, 10, 20, 30]:
        factors[f'vol_{p}'] = pd.Series(returns).rolling(p).std().values

    # 波动率变化
    for p in [5, 10]:
        vol_p = factors[f'vol_{p}']
        vol_ma = pd.Series(vol_p).rolling(p).mean()
        factors[f'vol_chg_{p}'] = (vol_p / (vol_ma + 1e-10) - 1).fillna(0).values

    # 波动率突破
    for p in [10, 20]:
        vol = factors[f'vol_{p}']
        vol_max = pd.Series(vol).rolling(p*2).max()
        factors[f'vol_break_{p}'] = (vol / (vol_max + 1e-10)).fillna(0).values

    # 4. RSI多周期
    for p in [4, 6, 8, 10, 12, 14, 16, 20, 24]:
        factors[f'rsi_{p}'] = _rsi(close, p)

    # 5. 动量多周期
    for p in [3, 5, 8, 10, 12, 15, 20, 30, 60]:
        factors[f'mom_{p}'] = _momentum(close, p)

    # 动量加速
    for p in [5, 10]:
        mom = factors[f'mom_{p}']
        mom_ma = pd.Series(mom).rolling(p).mean()
        factors[f'mom_acc_{p}'] = (mom / (mom_ma + 1e-10) - 1).fillna(0).values

    # 6. 布林带因子
    for p in [10, 20, 30]:
        ma = pd.Series(close).rolling(p).mean()
        std = pd.Series(close).rolling(p).std()
        upper = ma + 2 * std
        lower = ma - 2 * std
        factors[f'bb_width_{p}'] = ((upper - lower) / (ma + 1e-10)).fillna(0).values
        factors[f'bb_pos_{p}'] = ((close - lower) / (upper - lower + 1e-10)).fillna(0.5).values
        factors[f'bb_upper_{p}'] = ((close - upper) / (upper - lower + 1e-10)).fillna(0).values

    # 7. 均线因子
    for p in [5, 10, 20, 30, 60, 120]:
        ma = pd.Series(close).rolling(p).mean()
        factors[f'price_ma_{p}'] = (close / (ma + 1e-10) - 1).fillna(0).values

    # 均线多头排列
    ma5 = pd.Series(close).rolling(5).mean()
    ma10 = pd.Series(close).rolling(10).mean()
    ma20 = pd.Series(close).rolling(20).mean()
    ma60 = pd.Series(close).rolling(60).mean()

    factors['ma_golden_5_10'] = (ma5 > ma10).astype(float)
    factors['ma_golden_10_20'] = (ma10 > ma20).astype(float)
    factors['ma_golden_5_20'] = (ma5 > ma20).astype(float)
    factors['ma_golden_20_60'] = (ma20 > ma60).astype(float)
    factors['ma_all'] = ((ma5 > ma20) & (ma20 > ma60)).astype(float)

    # 均线交叉
    for fast, slow in [(5, 10), (5, 20), (10, 20), (10, 60), (20, 60)]:
        ma_fast = pd.Series(close).rolling(fast).mean()
        ma_slow = pd.Series(close).rolling(slow).mean()
        factors[f'ma_cross_{fast}_{slow}'] = (ma_fast / (ma_slow + 1e-10) - 1).fillna(0).values

    # 8. 趋势强度
    for p in [5, 10, 20, 30]:
        ma = pd.Series(close).rolling(p).mean()
        factors[f'trend_{p}'] = ((close - ma) / (ma + 1e-10)).fillna(0).values

    # 9. 价位因子
    for p in [5, 10, 20, 30, 60]:
        high_p = pd.Series(high).rolling(p).max()
        low_p = pd.Series(low).rolling(p).min()
        factors[f'price_pos_{p}'] = ((close - low_p) / (high_p - low_p + 1e-10)).fillna(0.5).values
        factors[f'high_pos_{p}'] = ((high - low_p) / (high_p - low_p + 1e-10)).fillna(0.5).values

    # 10. 成交量因子
    for p in [3, 5, 10, 20]:
        vol_ma = pd.Series(volume).rolling(p).mean()
        factors[f'vol_ratio_{p}'] = (volume / (vol_ma + 1e-10)).fillna(1).values

    # 成交量突破
    for p in [5, 10]:
        vol = volume
        vol_max = pd.Series(vol).rolling(p*2).max()
        factors[f'vol_break_{p}'] = (vol / (vol_max + 1e-10)).fillna(0).values

    # 11. 资金流因子
    for p in [3, 5, 10]:
        money = close * volume
        money_ma = pd.Series(money).rolling(p).mean()
        factors[f'money_flow_{p}'] = (money / (money_ma + 1e-10) - 1).fillna(0).values

    # 12. ATR因子
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = 0

    for p in [5, 10, 14, 20]:
        atr = pd.Series(tr).rolling(p).mean()
        factors[f'atr_{p}'] = atr.fillna(0).values
        factors[f'atr_ratio_{p}'] = (atr / (close + 1e-10)).fillna(0).values

    # 13. 威廉指标
    for p in [10, 20]:
        high_p = pd.Series(high).rolling(p).max()
        low_p = pd.Series(low).rolling(p).min()
        williams = -100 * ((high_p - close) / (high_p - low_p + 1e-10))
        factors[f'williams_{p}'] = williams.fillna(-50).values

    # 14. KDJ因子
    for p in [9, 14]:
        low_p = pd.Series(low).rolling(p).min()
        high_p = pd.Series(high).rolling(p).max()
        k = 100 * ((close - low_p) / (high_p - low_p + 1e-10))
        k = k.fillna(50)
        d = k.rolling(3).mean()
        j = 3 * k - 2 * d
        factors[f'kdj_k_{p}'] = k.fillna(50).values
        factors[f'kdj_d_{p}'] = d.fillna(50).values
        factors[f'kdj_j_{p}'] = j.fillna(50).values

    # 15. MACD因子
    ema12 = pd.Series(close).ewm(span=12).mean()
    ema26 = pd.Series(close).ewm(span=26).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9).mean()
    hist = macd - signal
    factors['macd'] = macd.fillna(0).values
    factors['macd_signal'] = signal.fillna(0).values
    factors['macd_hist'] = hist.fillna(0).values

    # 16. 跳空因子
    for p in [1, 2, 3]:
        gap = close - _shift(open_arr, p)
        gap = np.where(np.isnan(gap), 0, gap)
        factors[f'gap_{p}'] = gap
        factors[f'gap_up_{p}'] = (gap > 0).astype(float)
        factors[f'gap_down_{p}'] = (gap < 0).astype(float)

    # 17. 换手率变化
    for p in [5, 10]:
        vol_ma = pd.Series(volume).rolling(p).mean().values
        vol_chg = np.zeros(len(volume), dtype=float)
        vol_chg[:] = np.nan
        for i in range(p, len(volume)):
            if vol_ma[i-p] != 0:
                vol_chg[i] = (vol_ma[i] - vol_ma[i-p]) / (vol_ma[i-p] + 1e-10)
        factors[f'turnover_chg_{p}'] = vol_chg

    # 18. 波动率倾斜
    for p in [10, 20]:
        close_std = pd.Series(close).rolling(p).std()
        returns_std = pd.Series(returns).rolling(p).std()
        factors[f'vol_skew_{p}'] = (close_std / (returns_std + 1e-10)).fillna(0).values

    return factors


def calculate_ic(df_data, forward=20):
    """计算IC"""
    cols = [c for c in df_data.columns if c not in ['fwd', 'code']]
    ics = {}
    for c in cols:
        valid = ~(df_data[c].isna() | df_data['fwd'].isna())
        if valid.sum() < 30:
            continue
        try:
            ic = np.corrcoef(df_data.loc[valid, c], df_data.loc[valid, 'fwd'])[0, 1]
            if not np.isnan(ic):
                ics[c] = ic
        except:
            continue
    return ics


def main():
    print("="*70)
    print("因子IC验证 - 样本内/样本外对比")
    print("="*70)

    # 加载所有股票
    codes = []
    for f in os.listdir(BACKTRADER_PATH):
        if f.endswith('_qfq.csv') and f != 'sh000001_qfq.csv':
            codes.append(f.replace('_qfq.csv', ''))

    print(f"\n股票总数: {len(codes)}")

    # 获取行业信息
    industry_stocks = defaultdict(list)
    for code in codes:
        ind = get_industry(code)
        if ind:
            industry_stocks[ind].append(code)

    # 只分析主要行业
    sorted_ind = sorted(industry_stocks.items(), key=lambda x: len(x[1]), reverse=True)
    main_industries = [(ind, stocks) for ind, stocks in sorted_ind if len(stocks) >= 5]

    # 加载所有股票数据
    forward = 20
    split_point = 100  # 前100天样本内，后面的样本外

    all_data_in = []  # 样本内
    all_data_out = []  # 样本外
    industry_data_in = defaultdict(list)
    industry_data_out = defaultdict(list)

    print("\n加载数据...")

    for code in codes:
        df = load_data(code)
        if df is None:
            continue

        industry = get_industry(code)
        if not industry:
            continue

        factors = calc_all_factors(df)
        close = df['close'].values

        if len(close) < split_point + forward + 40:
            continue

        # 未来收益
        fwd = np.zeros_like(close, dtype=float)
        fwd[:] = np.nan
        for i in range(len(close) - forward):
            fwd[i] = close[i + forward] / close[i] - 1

        # 样本内：前100天
        for i in range(40, split_point):
            if np.isnan(fwd[i]) or abs(fwd[i]) > 1.0:
                continue
            row = {'fwd': fwd[i], 'code': code, 'industry': industry}
            for k, v in factors.items():
                if i < len(v) and not np.isnan(v[i]):
                    row[k] = v[i]
            all_data_in.append(row)
            industry_data_in[industry].append(row)

        # 样本外：后100天
        for i in range(split_point, len(close) - forward):
            if np.isnan(fwd[i]) or abs(fwd[i]) > 1.0:
                continue
            row = {'fwd': fwd[i], 'code': code, 'industry': industry}
            for k, v in factors.items():
                if i < len(v) and not np.isnan(v[i]):
                    row[k] = v[i]
            all_data_out.append(row)
            industry_data_out[industry].append(row)

    print(f"样本内数据: {len(all_data_in)}")
    print(f"样本外数据: {len(all_data_out)}")

    # 计算样本内IC
    df_in = pd.DataFrame(all_data_in)
    ics_in_all = calculate_ic(df_in)
    sorted_in_all = sorted(ics_in_all.items(), key=lambda x: abs(x[1]), reverse=True)

    # 计算样本外IC - 全部股票
    df_out = pd.DataFrame(all_data_out)
    ics_out_all = calculate_ic(df_out)
    sorted_out_all = sorted(ics_out_all.items(), key=lambda x: abs(x[1]), reverse=True)

    print("\n" + "="*70)
    print("【全部股票】样本内 vs 样本外 IC对比")
    print("="*70)

    print(f"\n样本内 Top10 因子 (n={len(df_in)}):")
    for i, (k, v) in enumerate(sorted_in_all[:10]):
        print(f"  {i+1}. {k:25s}: {v:+.4f}")

    print(f"\n样本外 Top10 因子 (n={len(df_out)}):")
    for i, (k, v) in enumerate(sorted_out_all[:10]):
        # 查找样本内IC
        in_ic = ics_in_all.get(k, 0)
        decay = (v - in_ic) / abs(in_ic) * 100 if in_ic != 0 else 0
        print(f"  {i+1}. {k:25s}: {v:+.4f} (样本内:{in_ic:+.4f}, 衰减:{decay:+.1f}%)")

    # 按行业对比
    print("\n" + "="*70)
    print("【分行业】样本内 vs 样本外 IC对比")
    print("="*70)

    industry_results = []

    for industry, _ in main_industries:
        if len(industry_data_in.get(industry, [])) < 100:
            continue
        if len(industry_data_out.get(industry, [])) < 100:
            continue

        df_in_ind = pd.DataFrame(industry_data_in[industry])
        df_out_ind = pd.DataFrame(industry_data_out[industry])

        ics_in = calculate_ic(df_in_ind)
        ics_out = calculate_ic(df_out_ind)

        # 取Top3因子对比
        top3_in = sorted(ics_in.items(), key=lambda x: abs(x[1]), reverse=True)[:3]

        print(f"\n【{industry}】样本内:{len(df_in_ind)} 样本外:{len(df_out_ind)}")
        print(f"  样本内 Top3:")
        for k, v in top3_in:
            print(f"    {k:25s}: {v:+.4f}")

        print(f"  样本外 (对比Top3):")
        for k, v_in in top3_in:
            v_out = ics_out.get(k, 0)
            decay = (v_out - v_in) / abs(v_in) * 100 if v_in != 0 else 0
            status = "✓" if abs(v_out) >= 0.04 else "✗"
            print(f"    {k:25s}: 样本内{v_in:+.4f} -> 样本外{v_out:+.4f} ({decay:+.1f}%) {status}")

        # 记录结果
        avg_decay = np.mean([(ics_out.get(k, 0) - v) / abs(v) * 100 if v != 0 else 0 for k, v in top3_in])
        industry_results.append({
            'industry': industry,
            'top3_in': [(k, v) for k, v in top3_in],
            'avg_ic_in': np.mean([v for _, v in top3_in]),
            'avg_ic_out': np.mean([ics_out.get(k, 0) for k, _ in top3_in]),
            'avg_decay': avg_decay
        })

    # 汇总
    print("\n" + "="*70)
    print("汇总：样本外IC有效性")
    print("="*70)

    print(f"\n{'行业':<12} {'样本内IC':>12} {'样本外IC':>12} {'衰减':>10} {'结论':<10}")
    print("-"*60)

    for r in industry_results:
        in_ic = r['avg_ic_in']
        out_ic = r['avg_ic_out']
        decay = r['avg_decay']

        if abs(out_ic) >= 0.04:
            conclusion = "有效✓"
        elif abs(out_ic) >= 0.02:
            conclusion = "一般○"
        else:
            conclusion = "失效✗"

        print(f"{r['industry']:<12} {in_ic:>+12.4f} {out_ic:>+12.4f} {decay:>+9.1f}% {conclusion:<10}")

    # 计算平均
    avg_in = np.mean([r['avg_ic_in'] for r in industry_results])
    avg_out = np.mean([r['avg_ic_out'] for r in industry_results])
    avg_decay = np.mean([r['avg_decay'] for r in industry_results])

    print("-"*60)
    print(f"{'平均':<12} {avg_in:>+12.4f} {avg_out:>+12.4f} {avg_decay:>+9.1f}%")

    print("\n" + "="*70)
    print("结论")
    print("="*70)

    if avg_out >= 0.04:
        print(f"✓ 分行业因子仍然有效 (样本外IC={avg_out:.4f})")
    elif avg_out >= 0.02:
        print(f"○ 分行业因子效果一般 (样本外IC={avg_out:.4f})")
    else:
        print(f"✗ 分行业因子可能过拟合 (样本外IC={avg_out:.4f})")

    # 对比不分行业
    top3_all = sorted(ics_in_all.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
    out_ic_all = np.mean([ics_out_all.get(k, 0) for k, _ in top3_all])

    print(f"\n对比: 不分行业因子样本外IC = {out_ic_all:.4f}")


if __name__ == '__main__':
    main()
