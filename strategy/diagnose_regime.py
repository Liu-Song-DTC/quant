# diagnose_regime.py
"""
市场状态判断诊断工具
验证在某种状态期间，市场的平均表现
"""
import pandas as pd
import numpy as np


def diagnose_regime_in_period(index_df: pd.DataFrame, regime_series: list) -> dict:
    """
    诊断市场状态判断的质量

    看看在判断为某种状态后，接下来 N 天的累计收益
    """
    df = index_df.copy()
    df['regime'] = regime_series

    results = {}

    # 计算接下来N天的累计收益
    for look_ahead in [5, 10, 20]:
        returns = []
        for i in range(len(df) - look_ahead):
            ret = df['close'].iloc[i + look_ahead] / df['close'].iloc[i] - 1
            returns.append((df['regime'].iloc[i], ret))

        ret_df = pd.DataFrame(returns, columns=['regime', 'return'])

        for regime in [-1, 0, 1]:
            regime_name = {-1: '熊市', 0: '震荡市', 1: '牛市'}[regime]
            subset = ret_df[ret_df['regime'] == regime]['return']
            if regime_name not in results:
                results[regime_name] = {}
            results[regime_name][f'{look_ahead}天累计'] = subset.mean()

    return results


def diagnose_yearly(index_df: pd.DataFrame, regime_series: list) -> pd.DataFrame:
    """分年份诊断"""
    df = index_df.copy()
    df['regime'] = regime_series
    df['year'] = df['datetime'].dt.year

    # 计算每日收益率
    df['daily_return'] = df['close'].pct_change()

    yearly_stats = []
    for year in sorted(df['year'].unique()):
        year_data = df[df['year'] == year]

        # 该年的收益
        year_return = year_data['close'].iloc[-1] / year_data['close'].iloc[0] - 1

        # 各状态天数
        regime_counts = year_data['regime'].value_counts()

        # 各状态期间的收益
        bull_return = year_data[year_data['regime'] == 1]['daily_return'].mean() * 20  # 近似月化
        bear_return = year_data[year_data['regime'] == -1]['daily_return'].mean() * 20
        side_return = year_data[year_data['regime'] == 0]['daily_return'].mean() * 20

        yearly_stats.append({
            'year': year,
            '总收益': year_return,
            '牛市天数': regime_counts.get(1, 0),
            '熊市天数': regime_counts.get(-1, 0),
            '震荡市天数': regime_counts.get(0, 0),
            '牛市期间收益(20日)': bull_return if not np.isnan(bull_return) else 0,
            '熊市期间收益(20日)': bear_return if not np.isnan(bear_return) else 0,
            '震荡期间收益(20日)': side_return if not np.isnan(side_return) else 0,
        })

    return pd.DataFrame(yearly_stats)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')

    from core.market_regime_detector import MarketRegimeDetector

    # 加载数据
    DATA_PATH = '../data/stock_data/backtrader_data/'
    index_data = pd.read_csv(DATA_PATH + 'sh000001_qfq.csv', parse_dates=['datetime'])

    # 使用检测器生成状态
    detector = MarketRegimeDetector()
    result = detector.generate(index_data)

    # 诊断未来收益
    results = diagnose_regime_in_period(index_data, result['regime'].tolist())

    print("="*60)
    print("市场状态判断质量诊断（判断后未来N天累计收益）")
    print("="*60)

    for regime_name, stats in results.items():
        print(f"\n【{regime_name}】")
        for period, ret in stats.items():
            print(f"  {period}: {ret:.2%}")

    print("\n" + "="*60)
    print("判断标准：")
    print("  牛市判断正确 -> 未来收益应该 > 0")
    print("  熊市判断正确 -> 未来收益应该 < 0")
    print("  震荡市判断正确 -> 未来收益应该接近 0")
    print("="*60)

    # 分年份诊断
    print("\n\n" + "="*60)
    print("分年份统计")
    print("="*60)

    yearly_df = diagnose_yearly(index_data, result['regime'].tolist())
    print(yearly_df.to_string(index=False))
