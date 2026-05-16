"""
全股票因子挖掘 - 使用全部数据计算IC，避免过拟合
按波动率水平分类（高/中/低）
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


def load_data(code, ndays=200):
    filepath = os.path.join(BACKTRADER_PATH, f'{code}_qfq.csv')
    if not os.path.exists(filepath):
        return None
    try:
        df = pd.read_csv(filepath)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values('datetime').tail(ndays)
        if len(df) < 100:
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

    # 创新高/新低
    for p in [20, 60]:
        high_p = pd.Series(high).rolling(p).max()
        low_p = pd.Series(low).rolling(p).min()
        factors[f'new_high_{p}'] = (close >= high_p).astype(float)
        factors[f'new_low_{p}'] = (close <= low_p).astype(float)

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

    # 主力资金
    for p in [5, 10]:
        close_ma = pd.Series(close).rolling(p).mean()
        money_flow = (close > close_ma).astype(float) * volume
        money_ma = pd.Series(money_flow).rolling(p).sum()
        factors[f'主力资金_{p}'] = (money_flow / (money_ma + 1e-10)).fillna(0).values

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


def classify_stock(df, code):
    """
    根据股票特征自动分类
    使用过去60天的波动率和价格特征
    """
    if df is None or len(df) < 60:
        return '其他'

    close = df['close'].values
    returns = np.diff(close, prepend=close[0]) / (close + 1e-10)
    vol_60 = pd.Series(returns).rolling(60).std().mean()

    # 波动率分类
    if vol_60 > 0.04:
        return '高波动'
    elif vol_60 > 0.025:
        return '中波动'
    else:
        return '低波动'


def analyze_group(group_name, all_data, forward=20):
    """分析单个分组"""
    if len(all_data) < 200:
        return None

    df = pd.DataFrame(all_data)
    print(f"\n  样本: {len(df)}")

    # IC计算
    cols = [c for c in df.columns if c not in ['fwd', 'code']]
    ics = {}
    for c in cols:
        valid = ~(df[c].isna() | df['fwd'].isna())
        if valid.sum() < 50:
            continue
        try:
            ic = np.corrcoef(df.loc[valid, c], df.loc[valid, 'fwd'])[0, 1]
            if not np.isnan(ic):
                ics[c] = ic
        except Exception:
            continue

    # 排序
    sorted_factors = sorted(ics.items(), key=lambda x: abs(x[1]), reverse=True)

    print(f"\n  Top 20 因子 (按|IC|):")
    for i, (k, v) in enumerate(sorted_factors[:20]):
        sig = "✓✓✓" if abs(v) >= 0.08 else "✓✓" if abs(v) >= 0.04 else "✓"
        print(f"    {i+1:2d}. {k:25s}: {v:+.4f} {sig}")

    return {'sample': len(df), 'factors': sorted_factors[:30], 'ics': ics}


def main():
    print("="*70)
    print("全股票因子挖掘 - 使用全部数据计算IC")
    print("="*70)

    # 加载所有股票
    codes = []
    for f in os.listdir(BACKTRADER_PATH):
        if f.endswith('_qfq.csv') and f != 'sh000001_qfq.csv':
            codes.append(f.replace('_qfq.csv', ''))

    print(f"\n股票数量: {len(codes)}")

    # 加载所有股票数据并计算因子
    all_stock_data = []
    forward = 20

    for code in codes:
        df = load_data(code)
        if df is None:
            continue

        factors = calc_all_factors(df)
        close = df['close'].values

        # 未来收益
        fwd = np.zeros_like(close, dtype=float)
        fwd[:] = np.nan
        for i in range(len(close) - forward):
            fwd[i] = close[i + forward] / close[i] - 1

        # 分类
        group = classify_stock(df, code)

        # 收集数据
        for i in range(40, len(df) - forward):
            if np.isnan(fwd[i]) or abs(fwd[i]) > 1.0:
                continue
            row = {'fwd': fwd[i], 'code': code, 'group': group}
            for k, v in factors.items():
                if i < len(v) and not np.isnan(v[i]):
                    row[k] = v[i]
            all_stock_data.append(row)

    print(f"总样本: {len(all_stock_data)}")

    # 转为DataFrame
    all_df = pd.DataFrame(all_stock_data)

    # 按分组分析
    groups = all_df['group'].unique()
    print(f"\n分组: {groups}")

    results = {}

    for group in groups:
        if group == '其他':
            continue
        print(f"\n{'='*60}")
        print(f"分组: {group}")
        print(f"{'='*60}")

        group_data = all_df[all_df['group'] == group].to_dict('records')
        result = analyze_group(group, group_data, forward)
        if result:
            results[group] = result

    # 全部数据分析
    print(f"\n{'='*60}")
    print(f"全部股票汇总")
    print(f"{'='*60}")
    result_all = analyze_group('全部', all_stock_data, forward)
    if result_all:
        results['全部'] = result_all

    # 汇总
    print("\n" + "="*70)
    print("汇总: 各分组最高IC因子")
    print("="*70)

    for name, r in results.items():
        print(f"\n{name}:")
        top5 = r['factors'][:5]
        for k, v in top5:
            print(f"  {k}: {v:+.4f}")

    # 保存
    import json
    output = {}
    for name, r in results.items():
        output[name] = {
            'sample': r['sample'],
            'top_factors': [(k, float(v)) for k, v in r['factors'][:30]]
        }

    out_path = os.path.join(BASE_DIR, 'strategy/all_stock_factors_ic.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存到: {out_path}")


if __name__ == '__main__':
    main()
