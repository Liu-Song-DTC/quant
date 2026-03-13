# test_single_factors.py
"""
测试单个因子和组合因子的IC
"""
import sys
sys.path.insert(0, '.')
import pandas as pd
import numpy as np
import os

DATA_PATH = '../data/stock_data/backtrader_data/'


def load_all_data():
    """加载所有数据"""
    all_items = [f for f in os.listdir(DATA_PATH)
                if f.endswith('_qfq.csv') and f != 'sh000001_qfq.csv'][:30]

    all_data = []
    for item in all_items:
        code = item.replace('_qfq.csv', '')
        try:
            df = pd.read_csv(DATA_PATH + item, parse_dates=['datetime'])
            if len(df) < 120:
                continue
            df['code'] = code
            all_data.append(df)
        except:
            continue

    return pd.concat(all_data, ignore_index=True)


def calc_indicators(df):
    """计算技术指标"""
    close = df['close'].values

    # 动量
    mom_20 = close / np.roll(close, 20) - 1
    mom_10 = close / np.roll(close, 10) - 1
    mom_5 = close / np.roll(close, 5) - 1
    mom_3 = close / np.roll(close, 3) - 1

    # 组合动量：多周期平均
    combo_mom = (mom_20 + mom_10 + mom_5) / 3

    # 趋势动量：只看短期上涨
    trend_mom = np.where(mom_10 > 0, mom_20, mom_20 * 0.5)

    # 改进版趋势动量：10日动量>0时用20日，否则用20日*0.3
    trend_mom_v2 = np.where(mom_10 > 0, mom_20, mom_20 * 0.3)

    # 双趋势动量：10日>0且20日>0
    dual_trend = np.where((mom_10 > 0) & (mom_20 > 0), mom_20, mom_20 * 0.5)

    # 强势股动量：连续上涨
    strong_mom = np.where(mom_5 > 0, mom_20 * 1.2, mom_20 * 0.8)

    # 三重趋势动量：5日>0 且 10日>0 且 20日>0
    triple_trend = np.where((mom_5 > 0) & (mom_10 > 0) & (mom_20 > 0), mom_20, mom_20 * 0.4)

    # 连续上涨强化：5日>0时给20日动量加成
    momentum_boost = np.where(mom_5 > 0, mom_20 * 1.3, mom_20 * 0.7)

    # 趋势动量V3：10日>0用20日，否则用20日*0.2（更保守）
    trend_mom_v3 = np.where(mom_10 > 0, mom_20, mom_20 * 0.2)

    # 趋势动量V4：只用10日
    trend_mom_v4 = np.where(mom_10 > 0, mom_10, mom_10 * 0.5)

    # 趋势动量V5：5日>0用20日*1.5
    trend_mom_v5 = np.where(mom_5 > 0, mom_20 * 1.5, mom_20 * 0.5)

    # 趋势动量V6：10日>0用20日*1.2
    trend_mom_v6 = np.where(mom_10 > 0, mom_20 * 1.2, mom_20 * 0.2)

    # 趋势动量V7：只用正向动量
    trend_mom_v7 = np.where(mom_10 > 0, mom_20, 0)

    # 趋势动量V8：10日>0且5日>0用20日
    trend_mom_v8 = np.where((mom_10 > 0) & (mom_5 > 0), mom_20, mom_20 * 0.3)

    # 趋势动量V9：10日>0用20日*1.5
    trend_mom_v9 = np.where(mom_10 > 0, mom_20 * 1.5, mom_20 * 0.2)

    # 趋势动量V10：10日>0用20日*1.3
    trend_mom_v10 = np.where(mom_10 > 0, mom_20 * 1.3, mom_20 * 0.15)

    # 趋势动量V11：10日>0用20日*1.4
    trend_mom_v11 = np.where(mom_10 > 0, mom_20 * 1.4, mom_20 * 0.1)

    # 趋势动量V12：10日>0用20日*1.6
    trend_mom_v12 = np.where(mom_10 > 0, mom_20 * 1.6, mom_20 * 0.1)

    # 趋势动量V13：10日>0用20日*1.25
    trend_mom_v13 = np.where(mom_10 > 0, mom_20 * 1.25, mom_20 * 0.12)

    # 趋势动量V14：10日>0用20日*1.35
    trend_mom_v14 = np.where(mom_10 > 0, mom_20 * 1.35, mom_20 * 0.18)

    # 趋势动量V15：10日>0用20日*1.7
    trend_mom_v15 = np.where(mom_10 > 0, mom_20 * 1.7, mom_20 * 0.1)

    # 趋势动量V16：10日>0用20日*1.8
    trend_mom_v16 = np.where(mom_10 > 0, mom_20 * 1.8, mom_20 * 0.08)

    # 趋势动量V17：10日>0用20日*2.0
    trend_mom_v17 = np.where(mom_10 > 0, mom_20 * 2.0, mom_20 * 0.05)

    # 未来收益
    future_ret = np.roll(close, -20) / close - 1

    return {
        'mom_20': mom_20,
        'mom_10': mom_10,
        'mom_5': mom_5,
        'mom_3': mom_3,
        'combo_mom': combo_mom,
        'trend_mom': trend_mom,
        'trend_mom_v2': trend_mom_v2,
        'dual_trend': dual_trend,
        'strong_mom': strong_mom,
        'triple_trend': triple_trend,
        'momentum_boost': momentum_boost,
        'trend_mom_v3': trend_mom_v3,
        'trend_mom_v4': trend_mom_v4,
        'trend_mom_v5': trend_mom_v5,
        'trend_mom_v6': trend_mom_v6,
        'trend_mom_v7': trend_mom_v7,
        'trend_mom_v8': trend_mom_v8,
        'trend_mom_v9': trend_mom_v9,
        'trend_mom_v10': trend_mom_v10,
        'trend_mom_v11': trend_mom_v11,
        'trend_mom_v12': trend_mom_v12,
        'trend_mom_v13': trend_mom_v13,
        'trend_mom_v14': trend_mom_v14,
        'trend_mom_v15': trend_mom_v15,
        'trend_mom_v16': trend_mom_v16,
        'trend_mom_v17': trend_mom_v17,
        'future_ret': future_ret
    }


def calc_ic(values, returns):
    """计算IC"""
    valid = (~np.isnan(values)) & (~np.isnan(returns)) & (np.abs(values) < 1) & (np.abs(returns) < 1)
    if valid.sum() < 100:
        return 0
    return np.corrcoef(values[valid], returns[valid])[0, 1]


if __name__ == "__main__":
    print("加载数据...")
    df = load_all_data()
    print(f"总数据量: {len(df)}")

    print("\n计算指标...")
    ind = calc_indicators(df)

    print("\n" + "="*50)
    print("单个因子和组合因子IC测试")
    print("="*50)

    factors = {
        '20日动量': ind['mom_20'],
        '10日动量': ind['mom_10'],
        '5日动量': ind['mom_5'],
        '3日动量': ind['mom_3'],
        '组合动量(20+10+5)/3': ind['combo_mom'],
        '趋势动量(上涨时)': ind['trend_mom'],
        '趋势动量V2(弱)': ind['trend_mom_v2'],
        '双趋势(10日>0且20日>0)': ind['dual_trend'],
        '强势股动量': ind['strong_mom'],
        '三重趋势动量': ind['triple_trend'],
        '动量加成': ind['momentum_boost'],
        '趋势动量V3(更弱)': ind['trend_mom_v3'],
        '趋势动量V4(10日)': ind['trend_mom_v4'],
        '趋势动量V5(5日加成)': ind['trend_mom_v5'],
        '趋势动量V6(10日*1.2)': ind['trend_mom_v6'],
        '趋势动量V7(只取正)': ind['trend_mom_v7'],
        '趋势动量V8(10日&5日)': ind['trend_mom_v8'],
        '趋势动量V9(10日*1.5)': ind['trend_mom_v9'],
        '趋势动量V10(10日*1.3)': ind['trend_mom_v10'],
        '趋势动量V11(10日*1.4)': ind['trend_mom_v11'],
        '趋势动量V12(10日*1.6)': ind['trend_mom_v12'],
        '趋势动量V13(10日*1.25)': ind['trend_mom_v13'],
        '趋势动量V14(10日*1.35)': ind['trend_mom_v14'],
        '趋势动量V15(10日*1.7)': ind['trend_mom_v15'],
        '趋势动量V16(10日*1.8)': ind['trend_mom_v16'],
        '趋势动量V17(10日*2.0)': ind['trend_mom_v17'],
    }

    for name, values in factors.items():
        ic = calc_ic(values, ind['future_ret'])
        status = "✓" if abs(ic) > 0.04 else " "
        print(f"{status} {name}: IC = {ic*100:.2f}%")
