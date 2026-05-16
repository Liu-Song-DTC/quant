"""
行业因子深度挖掘分析

目标：
1. 针对每个行业深入挖掘有效因子
2. 验证因子质量（IC、IR）
3. 区分技术因子和基本面因子
4. 找出行业专属的有效因子组合
"""

import os
import sys
import numpy as np
import pandas as pd
from collections import defaultdict

BASE_DIR = '/Users/litiancheng01/code/ltc/quant'
sys.path.insert(0, BASE_DIR)

# 配置路径
DATA_PATH = os.path.join(BASE_DIR, 'data/stock_data')
BACKTRADER_PATH = os.path.join(DATA_PATH, 'backtrader_data')
FUNDAMENTAL_PATH = os.path.join(DATA_PATH, 'fundamental_data')


def load_stock_price_data(code, ndays=120):
    """加载股票价格数据"""
    filepath = os.path.join(BACKTRADER_PATH, f'{code}_qfq.csv')
    if not os.path.exists(filepath):
        return None
    try:
        df = pd.read_csv(filepath)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values('datetime').tail(ndays)
        if len(df) < 60:
            return None
        return df
    except Exception:
        return None


def get_stock_industry(code):
    """获取股票行业"""
    filepath = os.path.join(FUNDAMENTAL_PATH, f'{code}.csv')
    if not os.path.exists(filepath):
        return None
    try:
        df = pd.read_csv(filepath)
        if '所处行业' in df.columns and len(df) > 0:
            industry = df.iloc[0]['所处行业']
            if pd.notna(industry):
                return str(industry).strip()
        return None
    except Exception:
        return None


def get_stock_fundamental(code, date):
    """获取基本面数据"""
    filepath = os.path.join(FUNDAMENTAL_PATH, f'{code}.csv')
    if not os.path.exists(filepath):
        return {}
    try:
        df = pd.read_csv(filepath)
        if '数据可用日期' not in df.columns:
            return {}

        date_str = date.replace('-', '') if isinstance(date, str) else date.strftime('%Y%m%d')
        available = df[df['数据可用日期'] <= date_str]
        if len(available) == 0:
            return {}
        latest = available.iloc[0]

        result = {}
        # ROE
        if '净资产收益率' in latest:
            roe = latest['净资产收益率']
            try:
                result['roe'] = float(str(roe).strip('%')) / 100 if pd.notna(roe) else None
            except Exception:
                result['roe'] = None

        # 净利润增长
        if '净利润-同比增长' in latest:
            pg = latest['净利润-同比增长']
            try:
                result['profit_growth'] = float(str(pg).strip('%')) / 100 if pd.notna(pg) else None
            except Exception:
                result['profit_growth'] = None

        # 营收增长
        if '营业总收入-同比增长' in latest:
            rg = latest['营业总收入-同比增长']
            try:
                result['revenue_growth'] = float(str(rg).strip('%')) / 100 if pd.notna(rg) else None
            except Exception:
                result['revenue_growth'] = None

        return result
    except Exception:
        return {}


# ============ 因子计算函数 ============

def calc_volatility(arr, period):
    """波动率因子"""
    returns = np.diff(arr, prepend=arr[0]) / arr
    result = np.zeros_like(arr, dtype=float)
    result[:] = np.nan
    for i in range(period, len(arr)):
        result[i] = np.std(returns[i-period:i])
    return result


def calc_rsi(arr, period):
    """RSI因子"""
    delta = np.diff(arr, prepend=arr[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

    avg_gain = np.zeros_like(arr, dtype=float)
    avg_loss = np.zeros_like(arr, dtype=float)
    for i in range(period, len(arr)):
        avg_gain[i] = np.mean(gain[i-period:i])
        avg_loss[i] = np.mean(loss[i-period:i])

    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1 + rs))


def calc_momentum(arr, period):
    """动量因子"""
    result = np.zeros_like(arr, dtype=float)
    result[:] = np.nan
    result[period:] = (arr[period:] / arr[:-period]) - 1
    return result


def calc_bb_width(arr, period=20):
    """布林带宽度因子"""
    middle = pd.Series(arr).rolling(period).mean()
    std = pd.Series(arr).rolling(period).std()
    upper = middle + 2 * std
    lower = middle - 2 * std
    width = (upper - lower) / (middle + 1e-10)
    return width.fillna(0).values


def calc_volume_ratio(volume, period=5):
    """成交量突破因子"""
    vol_ma = pd.Series(volume).rolling(period).mean()
    ratio = volume / (vol_ma + 1e-10)
    return ratio.fillna(1).values


def calc_price_position(arr, period=20):
    """价格在区间位置"""
    high = pd.Series(arr).rolling(period).max()
    low = pd.Series(arr).rolling(period).min()
    pos = (arr - low) / (high - low + 1e-10)
    return pos.fillna(0.5).values


def calc_ma(arr, period):
    """移动平均"""
    return pd.Series(arr).rolling(period).mean().fillna(method='bfill').values


# ============ 行业因子分析 ============

def analyze_industry_factors(industry, stock_codes, forward_period=10):
    """分析单个行业的所有因子有效性"""
    print(f"\n{'='*60}")
    print(f"行业分析: {industry} ({len(stock_codes)}只股票)")
    print(f"{'='*60}")

    all_data = []

    for code in stock_codes:
        df = load_stock_price_data(code, ndays=150)
        if df is None or len(df) < 60:
            continue

        close = df['close'].values
        volume = df['volume'].values if 'volume' in df.columns else np.ones(len(close))

        # 计算技术因子
        factors = {
            # 波动率因子
            'vol_5': calc_volatility(close, 5),
            'vol_10': calc_volatility(close, 10),
            'vol_20': calc_volatility(close, 20),

            # RSI因子
            'rsi_6': calc_rsi(close, 6),
            'rsi_8': calc_rsi(close, 8),
            'rsi_10': calc_rsi(close, 10),
            'rsi_14': calc_rsi(close, 14),

            # 动量因子
            'mom_3': calc_momentum(close, 3),
            'mom_5': calc_momentum(close, 5),
            'mom_10': calc_momentum(close, 10),
            'mom_20': calc_momentum(close, 20),

            # 布林带
            'bb_width': calc_bb_width(close, 20),

            # 成交量
            'vol_ratio': calc_volume_ratio(volume, 5),
            'vol_ratio_10': calc_volume_ratio(volume, 10),

            # 价格位置
            'price_pos': calc_price_position(close, 20),

            # 均线
            'ma5_above_ma20': (calc_ma(close, 5) > calc_ma(close, 20)).astype(float),
        }

        # 计算未来收益
        fwd_ret = np.zeros_like(close, dtype=float)
        fwd_ret[:] = np.nan
        for i in range(len(close) - forward_period):
            fwd_ret[i] = close[i + forward_period] / close[i] - 1

        # 收集数据
        for i in range(30, len(df) - forward_period):
            if np.isnan(fwd_ret[i]):
                continue

            row = {'code': code, 'date_idx': i, 'forward_return': fwd_ret[i]}

            # 基本面因子 (简化处理)
            date = df['datetime'].iloc[i]
            fund = get_stock_fundamental(code, date)
            row['roe'] = fund.get('roe') if fund.get('roe') else np.nan
            row['profit_growth'] = fund.get('profit_growth') if fund.get('profit_growth') else np.nan
            row['revenue_growth'] = fund.get('revenue_growth') if fund.get('revenue_growth') else np.nan

            # 技术因子
            for fname, fvals in factors.items():
                if i < len(fvals):
                    row[fname] = fvals[i]

            all_data.append(row)

    if len(all_data) < 100:
        print(f"  数据不足，跳过")
        return None

    data_df = pd.DataFrame(all_data)
    print(f"  有效样本: {len(data_df)}")

    # 计算各因子的IC
    factor_names = [c for c in data_df.columns if c not in ['code', 'date_idx', 'forward_return']]

    ic_results = {}
    for fname in factor_names:
        valid = ~(data_df[fname].isna() | data_df['forward_return'].isna())
        if valid.sum() < 20:
            continue

        ic = np.corrcoef(data_df.loc[valid, fname], data_df.loc[valid, 'forward_return'])[0, 1]
        if not np.isnan(ic):
            ic_results[fname] = ic

    # 排序并输出
    sorted_factors = sorted(ic_results.items(), key=lambda x: abs(x[1]), reverse=True)

    print(f"\n  因子IC排名 (前15):")
    for i, (fname, ic) in enumerate(sorted_factors[:15]):
        direction = "+" if ic > 0 else "-"
        print(f"    {i+1:2d}. {fname:20s}: IC={ic:+.4f} ({direction})")

    # 分类汇总
    tech_factors = {k: v for k, v in ic_results.items()
                   if any(x in k for x in ['vol', 'rsi', 'mom', 'bb', 'vol_', 'price', 'ma'])}
    fund_factors = {k: v for k, v in ic_results.items()
                   if any(x in k for x in ['roe', 'profit', 'revenue'])}

    print(f"\n  技术因子统计: {len(tech_factors)}个, 平均|IC|={np.mean([abs(v) for v in tech_factors.values()]):.4f}")
    print(f"  基本面因子统计: {len(fund_factors)}个, 平均|IC|={np.mean([abs(v) for v in fund_factors.values()]):.4f}" if fund_factors else "  基本面因子: 数据不足")

    return {
        'stock_count': len(stock_codes),
        'sample_count': len(data_df),
        'all_factors': ic_results,
        'tech_factors': tech_factors,
        'fund_factors': fund_factors,
        'top5': [f[0] for f in sorted_factors[:5]],
        'top5_ic': {f[0]: f[1] for f in sorted_factors[:5]},
    }


def main():
    print("="*70)
    print("行业因子深度挖掘分析")
    print("="*70)

    # 获取所有股票
    codes = []
    for f in os.listdir(BACKTRADER_PATH):
        if f.endswith('_qfq.csv') and f != 'sh000001_qfq.csv':
            code = f.replace('_qfq.csv', '')
            codes.append(code)

    print(f"\n发现 {len(codes)} 只股票")

    # 按行业分组
    industry_stocks = defaultdict(list)
    for code in codes:
        ind = get_stock_industry(code)
        if ind:
            industry_stocks[ind].append(code)

    print(f"发现 {len(industry_stocks)} 个行业")

    # 按股票数量排序，只分析样本足够的行业
    sorted_industries = sorted(industry_stocks.items(), key=lambda x: len(x[1]), reverse=True)

    # 分析主要行业
    results = {}
    for industry, stocks in sorted_industries:
        if len(stocks) >= 5:  # 至少5只股票
            result = analyze_industry_factors(industry, stocks, forward_period=10)
            if result:
                results[industry] = result

    # 汇总分析
    print("\n" + "="*70)
    print("汇总分析")
    print("="*70)

    # 找出跨行业有效的因子
    all_tech_factors = defaultdict(list)
    all_fund_factors = defaultdict(list)

    for ind, r in results.items():
        for fname, ic in r['tech_factors'].items():
            all_tech_factors[fname].append(ic)
        for fname, ic in r['fund_factors'].items():
            all_fund_factors[fname].append(ic)

    # 计算平均IC
    avg_tech = {k: np.mean(v) for k, v in all_tech_factors.items()}
    avg_fund = {k: np.mean(v) for k, v in all_fund_factors.items()}

    print("\n跨行业平均IC (技术因子):")
    sorted_tech = sorted(avg_tech.items(), key=lambda x: abs(x[1]), reverse=True)
    for fname, ic in sorted_tech[:10]:
        print(f"  {fname:20s}: {ic:+.4f}")

    print("\n跨行业平均IC (基本面因子):")
    if avg_fund:
        sorted_fund = sorted(avg_fund.items(), key=lambda x: abs(x[1]), reverse=True)
        for fname, ic in sorted_fund[:5]:
            print(f"  {fname:20s}: {ic:+.4f}")
    else:
        print("  基本面因子数据不足")

    # 各行业最佳因子汇总
    print("\n各行业推荐因子:")
    print("-"*70)
    for ind in sorted(results.keys(), key=lambda x: results[x]['sample_count'], reverse=True):
        r = results[ind]
        top3 = r['top5'][:3]
        print(f"  {ind:15s}: {', '.join(top3)}")

    # 保存详细结果
    import json
    output = {}
    for ind, r in results.items():
        output[ind] = {
            'stock_count': r['stock_count'],
            'sample_count': r['sample_count'],
            'top5': r['top5'],
            'top5_ic': {k: float(v) for k, v in r['top5_ic'].items()},
            'all_factors': {k: float(v) for k, v in r['all_factors'].items()},
        }

    output_path = os.path.join(BASE_DIR, 'strategy/industry_factors_deep_analysis.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n详细结果已保存到: {output_path}")


if __name__ == '__main__':
    main()
