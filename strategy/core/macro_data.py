"""宏观数据获取与处理 — M1/社融/两融作为市场状态领先指标

核心逻辑(来自动量因子框架):
- M1增速回升 → 信用扩张, 行情启动前兆
- 社融增量回升 → 流动性充裕
- 两融余额趋势 → 市场情绪温度计
- 这三个指标通常领先市场价格1-3个月
"""

import pandas as pd
import numpy as np
import os
import json

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MACRO_DIR = os.path.join(PROJECT_DIR, 'data', 'macro')
os.makedirs(MACRO_DIR, exist_ok=True)


def download_macro_data():
    """下载M1/社融/两融历史数据"""
    import akshare as ak

    # 1. M1/M2
    m1 = ak.macro_china_money_supply()
    m1['date'] = pd.to_datetime(m1['月份'].str.replace('年', '-').str.replace('月份', '-01'))
    m1 = m1.sort_values('date')
    m1[['date', '货币(M1)-同比增长', '货币和准货币(M2)-同比增长']].to_csv(
        os.path.join(MACRO_DIR, 'm1_m2.csv'), index=False)
    print(f'M1/M2: {len(m1)}条, {m1.date.min().date()} ~ {m1.date.max().date()}')

    # 2. 社融
    sf = ak.macro_china_shrzgm()
    sf['date'] = pd.to_datetime(sf['月份'].astype(str).str[:4] + '-' + sf['月份'].astype(str).str[4:] + '-01')
    sf = sf.sort_values('date')
    sf[['date', '社会融资规模增量', '其中-人民币贷款']].to_csv(
        os.path.join(MACRO_DIR, 'social_financing.csv'), index=False)
    print(f'社融: {len(sf)}条, {sf.date.min().date()} ~ {sf.date.max().date()}')

    # 3. 两融余额
    try:
        mr = ak.stock_margin_sse(start_date='20180101')
        mr['date'] = pd.to_datetime(mr['信用交易日期'])
        daily = mr.groupby('date')['融资余额'].sum().reset_index()
        daily.to_csv(os.path.join(MACRO_DIR, 'margin_balance.csv'), index=False)
        print(f'两融: {len(daily)}天')
    except Exception as e:
        print(f'两融下载失败: {e}')


def load_macro_data():
    """加载宏观数据, 返回统一DataFrame"""
    m1 = pd.read_csv(os.path.join(MACRO_DIR, 'm1_m2.csv'), parse_dates=['date'])
    sf = pd.read_csv(os.path.join(MACRO_DIR, 'social_financing.csv'), parse_dates=['date'])
    mr = pd.read_csv(os.path.join(MACRO_DIR, 'margin_balance.csv'), parse_dates=['date'])

    # 合并到月频
    m1['month'] = m1['date'].dt.to_period('M')
    sf['month'] = sf['date'].dt.to_period('M')
    mr['month'] = mr['date'].dt.to_period('M')

    monthly = m1[['month', '货币(M1)-同比增长']].copy()
    monthly = monthly.merge(sf[['month', '社会融资规模增量']], on='month', how='left')
    monthly = monthly.merge(
        mr.groupby('month')['融资余额'].mean().reset_index(), on='month', how='left')

    # 计算变化率
    monthly['M1_yoy'] = monthly['货币(M1)-同比增长']
    monthly['SF_ma3'] = monthly['社会融资规模增量'].rolling(3).mean()
    monthly['margin_ma3'] = monthly['融资余额'].rolling(3).mean()
    monthly['margin_chg_3m'] = monthly['margin_ma3'].pct_change(3)

    # 信号: M1加速 + 社融改善 + 两融回升 → 行情前兆
    monthly['m1_accel'] = monthly['M1_yoy'].diff(3)  # M1同比加速
    monthly['sf_improve'] = monthly['SF_ma3'].diff(3) > 0  # 社融改善
    monthly['bull_signal'] = (
        (monthly['m1_accel'] > 0) &
        (monthly['sf_improve']) &
        (monthly['margin_chg_3m'] > -0.05)
    ).astype(int)

    return monthly


def is_macro_bull_confirmed(date, macro_df=None):
    """连续2个月bull_signal(本月+上月)才确认, 过滤2022的2个月分散假阳性"""
    if macro_df is None:
        macro_df = load_macro_data()
    month = pd.Period(date.strftime('%Y-%m'), freq='M')
    idx = macro_df[macro_df['month'] == month].index
    if len(idx) == 0:
        return False
    idx = idx[0]
    if idx == 0:
        return False
    # 本月+上月都是bull才确认
    return (macro_df.iloc[idx]['bull_signal'] == 1 and
            macro_df.iloc[idx-1]['bull_signal'] == 1)


def get_macro_regime(date, macro_df=None):
    """根据宏观数据判断市场状态倾向

    Returns:
        str: 'bullish' / 'neutral' / 'bearish'
    """
    if macro_df is None:
        macro_df = load_macro_data()

    month = pd.Period(date.strftime('%Y-%m'), freq='M')
    row = macro_df[macro_df['month'] == month]
    if row.empty:
        return 'neutral'

    row = row.iloc[0]
    m1_accel = row.get('m1_accel', 0)
    sf_ok = row.get('sf_improve', False)
    margin_ok = row.get('margin_chg_3m', -1) > -0.05

    if m1_accel > 0.5 and sf_ok and margin_ok:
        return 'bullish'
    elif m1_accel < -2.0 and row.get('M1_yoy', 0) > 0:  # M1从正转负才真恶化, 已在负值=已price in
        return 'bearish'
    return 'neutral'


if __name__ == '__main__':
    download_macro_data()
    df = load_macro_data()
    print(df.tail(10))
    print(f'\nbull_signal days: {df.bull_signal.sum()}/{len(df)}')
