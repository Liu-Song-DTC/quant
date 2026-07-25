"""宏观数据获取与处理 — 内盘×外盘共振框架

内盘(决定能不能涨): M1/社融/PPI/两融
外盘(决定什么时候涨): Fed利率/美元指数

基于《A股动量因子》框架:
- 内盘优先级: 信用扩张 > 企业盈利 > 市场情绪
- 外盘: Fed利率周期 > 美元强弱
- 共振规则: 内盘+外盘同向=趋势行情, 方向背离=震荡
"""

import pandas as pd
import numpy as np
import os

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MACRO_DIR = os.path.join(PROJECT_DIR, 'data', 'macro')
os.makedirs(MACRO_DIR, exist_ok=True)


def download_macro_data():
    """下载内盘+外盘宏观数据"""
    import akshare as ak

    # === 内盘 ===
    # M1/M2
    m1 = ak.macro_china_money_supply()
    m1['date'] = pd.to_datetime(m1['月份'].str.replace('年', '-').str.replace('月份', '-01'))
    m1 = m1.sort_values('date')
    m1[['date', '货币(M1)-同比增长', '货币和准货币(M2)-同比增长']].to_csv(
        os.path.join(MACRO_DIR, 'm1_m2.csv'), index=False)
    print(f'M1/M2: {len(m1)}条')

    # 社融
    sf = ak.macro_china_shrzgm()
    sf['date'] = pd.to_datetime(sf['月份'].astype(str).str[:4] + '-' + sf['月份'].astype(str).str[4:] + '-01')
    sf = sf.sort_values('date')
    sf[['date', '社会融资规模增量', '其中-人民币贷款']].to_csv(
        os.path.join(MACRO_DIR, 'social_financing.csv'), index=False)
    print(f'社融: {len(sf)}条')

    # 两融
    try:
        mr = ak.stock_margin_sse(start_date='20180101')
        mr['date'] = pd.to_datetime(mr['信用交易日期'])
        daily = mr.groupby('date')['融资余额'].sum().reset_index()
        daily.to_csv(os.path.join(MACRO_DIR, 'margin_balance.csv'), index=False)
        print(f'两融: {len(daily)}天')
    except Exception as e:
        print(f'两融下载失败: {e}')

    # PPI (企业盈利代理)
    try:
        ppi = ak.macro_china_ppi_yearly()
        ppi['date'] = pd.to_datetime(ppi['日期'])
        ppi = ppi[['date', '今值', '前值']].dropna(subset=['今值'])
        ppi.to_csv(os.path.join(MACRO_DIR, 'ppi.csv'), index=False)
        print(f'PPI: {len(ppi)}条')
    except Exception as e:
        print(f'PPI失败: {e}')

    # === 外盘 ===
    # Fed利率
    try:
        fed = ak.macro_bank_usa_interest_rate()
        fed['date'] = pd.to_datetime(fed['日期'])
        fed = fed[['date', '今值', '前值']].dropna(subset=['今值'])
        fed.to_csv(os.path.join(MACRO_DIR, 'fed_rate.csv'), index=False)
        print(f'Fed利率: {len(fed)}条')
    except Exception as e:
        print(f'Fed失败: {e}')


def load_macro_data():
    """加载内盘+外盘宏观数据, 返回统一DataFrame"""
    m1_df = pd.read_csv(os.path.join(MACRO_DIR, 'm1_m2.csv'), parse_dates=['date'])
    sf_df = pd.read_csv(os.path.join(MACRO_DIR, 'social_financing.csv'), parse_dates=['date'])
    mr_df = pd.read_csv(os.path.join(MACRO_DIR, 'margin_balance.csv'), parse_dates=['date'])

    m1_df['month'] = m1_df['date'].dt.to_period('M')
    sf_df['month'] = sf_df['date'].dt.to_period('M')
    mr_df['month'] = mr_df['date'].dt.to_period('M')

    monthly = m1_df[['month', '货币(M1)-同比增长']].copy()
    monthly = monthly.merge(sf_df[['month', '社会融资规模增量']], on='month', how='left')
    monthly = monthly.merge(mr_df.groupby('month')['融资余额'].mean().reset_index(), on='month', how='left')

    # PPI
    ppi_path = os.path.join(MACRO_DIR, 'ppi.csv')
    if os.path.exists(ppi_path):
        ppi_df = pd.read_csv(ppi_path, parse_dates=['date'])
        ppi_df['month'] = ppi_df['date'].dt.to_period('M')
        ppi_monthly = ppi_df.groupby('month')['今值'].mean().reset_index()
        ppi_monthly.columns = ['month', 'PPI']
        monthly = monthly.merge(ppi_monthly, on='month', how='left')

    # Fed利率
    fed_path = os.path.join(MACRO_DIR, 'fed_rate.csv')
    if os.path.exists(fed_path):
        fed_df = pd.read_csv(fed_path, parse_dates=['date'])
        fed_df['month'] = fed_df['date'].dt.to_period('M')
        fed_monthly = fed_df.groupby('month')['今值'].agg(['mean', 'last']).reset_index()
        fed_monthly.columns = ['month', 'fed_rate_mean', 'fed_rate']
        monthly = monthly.merge(fed_monthly[['month', 'fed_rate']], on='month', how='left')
        monthly['fed_rate'] = monthly['fed_rate'].ffill()

    # 计算内盘指标
    monthly['M1_yoy'] = monthly['货币(M1)-同比增长']
    monthly['SF_ma3'] = monthly['社会融资规模增量'].rolling(3).mean()
    monthly['margin_ma3'] = monthly['融资余额'].rolling(3).mean()
    monthly['margin_chg_3m'] = monthly['margin_ma3'].pct_change(3)
    monthly['m1_accel'] = monthly['M1_yoy'].diff(3)
    monthly['sf_improve'] = monthly['SF_ma3'].diff(3) > 0

    # M1-M2剪刀差: 框架核心 — 由负转正=行情高发区间
    if '货币和准货币(M2)-同比增长' in monthly.columns:
        monthly['M2_yoy'] = monthly['货币和准货币(M2)-同比增长']
        monthly['m1_m2_scissor'] = monthly['M1_yoy'] - monthly['M2_yoy']
        monthly['scissor_turn_positive'] = (
            (monthly['m1_m2_scissor'] > -1.0) &
            (monthly['m1_m2_scissor'].shift(3) < -2.0)
        )  # 剪刀差从<-2收敛到>-1=触底回升

    # PPI改善 (三个月均值上升)
    if 'PPI' in monthly.columns:
        monthly['ppi_ma3'] = monthly['PPI'].rolling(3).mean()
        monthly['ppi_improve'] = monthly['ppi_ma3'].diff(3) > 0
    else:
        monthly['ppi_improve'] = False

    # 内盘bull: M1加速 + 社融改善 + 两融不暴跌
    monthly['bull_signal'] = (
        (monthly['m1_accel'] > 0) &
        (monthly['sf_improve']) &
        (monthly['margin_chg_3m'] > -0.05)
    ).astype(int)

    # 内盘bear: M1严重减速
    monthly['bear_signal'] = (
        (monthly['m1_accel'] < -1.0) |
        ((~monthly['sf_improve']) & (monthly['margin_chg_3m'] <= -0.05))
    ).astype(int)

    # 外盘: Fed加息周期=压制, 降息周期=利好
    if 'fed_rate' in monthly.columns:
        monthly['fed_change'] = monthly['fed_rate'].diff(6)  # 6个月利率变化
        monthly['fed_easing'] = monthly['fed_change'] <= -0.25  # 降息周期
        monthly['fed_tightening'] = monthly['fed_change'] >= 0.5  # 加息周期
    else:
        monthly['fed_easing'] = False
        monthly['fed_tightening'] = False

    # 四级共振信号
    monthly['resonance'] = 'neutral'
    monthly.loc[(monthly['bull_signal'] == 1) & (monthly['fed_easing']), 'resonance'] = 'bull_bull'
    monthly.loc[(monthly['bull_signal'] == 1) & (monthly['fed_tightening']), 'resonance'] = 'bull_bear'
    monthly.loc[(monthly['bear_signal'] == 1) & (monthly['fed_easing']), 'resonance'] = 'bear_bull'
    monthly.loc[(monthly['bear_signal'] == 1) & (monthly['fed_tightening']), 'resonance'] = 'bear_bear'

    return monthly


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
    scissor_turn = row.get('scissor_turn_positive', False)

    # 剪刀差触底回升=框架定义的行情启动信号
    if scissor_turn and sf_ok:
        return 'bullish'
    elif m1_accel > 0.5 and sf_ok and margin_ok:
        return 'bullish'
    elif m1_accel < -1.0 and not sf_ok:
        return 'bearish'  # M1加速跌+社融不改善=确认恶化
    return 'neutral'


def get_macro_severity(date, macro_df=None):
    """返回bearish深度: 'deep'(m1_accel<-2) / 'shallow' / 'none'"""
    if macro_df is None:
        macro_df = load_macro_data()
    month = pd.Period(date.strftime('%Y-%m'), freq='M')
    row = macro_df[macro_df['month'] == month]
    if row.empty:
        return 'none'
    row = row.iloc[0]
    if row.get('m1_accel', 0) < -2.0:
        return 'deep'
    elif row.get('m1_accel', 0) < -1.0:
        return 'shallow'
    return 'none'


if __name__ == '__main__':
    import sys
    if 'download' in sys.argv:
        download_macro_data()
    df = load_macro_data()
    print(df[['month', 'M1_yoy', 'm1_accel', 'bull_signal', 'bear_signal', 'fed_rate', 'PPI']].tail(15))
    print(f'\nbull: {df.bull_signal.sum()} bear: {df.bear_signal.sum()}')
    print(f'resonance: {df.resonance.value_counts().to_dict()}')
