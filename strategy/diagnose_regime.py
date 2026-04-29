# diagnose_regime.py
"""
市场状态判断诊断工具
验证在某种状态期间，市场的平均表现
"""
import pandas as pd
import numpy as np


def diagnose_regime_comprehensive(index_df: pd.DataFrame, regime_series: list) -> dict:
    """
    综合诊断 - 多维度评价体系
    1. 方向准确率
    2. 相对基准超额收益
    3. 统计显著性（t检验）
    4. 信息比率
    """
    df = index_df.copy()
    df['regime'] = regime_series

    # 计算不同持有期的收益
    results = {}

    for look_ahead in [5, 10, 20]:
        # 构建收益表
        data = []
        for i in range(len(df) - look_ahead):
            ret = df['close'].iloc[i + look_ahead] / df['close'].iloc[i] - 1
            data.append({'regime': df['regime'].iloc[i], 'return': ret})

        ret_df = pd.DataFrame(data)

        # 计算各项指标
        results[look_ahead] = {}

        for regime in [-1, 0, 1]:
            regime_name = {-1: '熊市', 0: '震荡市', 1: '牛市'}[regime]
            subset = ret_df[ret_df['regime'] == regime]['return']

            if len(subset) == 0:
                continue

            mean_ret = subset.mean()
            std_ret = subset.std()
            count = len(subset)

            # 方向准确率：收益符号是否正确
            if regime == 1:  # 牛市
                direction_acc = (subset > 0).mean()
                expected_sign = 1
            elif regime == -1:  # 熊市
                direction_acc = (subset < 0).mean()
                expected_sign = -1
            else:  # 震荡市
                direction_acc = (subset.abs() < 0.02).mean()  # 接近0
                expected_sign = 0

            # 相对基准（随机选择）的超额收益
            baseline_ret = ret_df['return'].mean()
            excess_ret = mean_ret - baseline_ret

            # t检验（简化版）
            from scipy import stats
            if count > 10 and std_ret > 0:
                t_stat, p_value = stats.ttest_1samp(subset, 0)
            else:
                t_stat, p_value = 0, 1

            results[look_ahead][regime_name] = {
                '样本数': count,
                '平均收益': mean_ret,
                '标准差': std_ret,
                '方向准确率': direction_acc,
                '超额收益': excess_ret,
                't统计量': t_stat,
                'p值': p_value,
                '显著': '***' if p_value < 0.01 else '**' if p_value < 0.05 else '*' if p_value < 0.1 else ''
            }

    return results


def diagnose_regime_in_period(index_df: pd.DataFrame, regime_series: list) -> dict:
    """
    诊断市场状态判断的质量 - 增强版

    1. 不同持有期的收益（1,3,5,10,20天）
    2. 状态持续验证（状态持续N天才计入统计）
    """
    df = index_df.copy()
    df['regime'] = regime_series

    results = {}

    # 1. 不同持有期的收益
    print("\n" + "="*60)
    print("【不同持有期收益分析】")
    print("="*60)

    for look_ahead in [1, 3, 5, 10, 20]:
        returns = []
        for i in range(len(df) - look_ahead):
            ret = df['close'].iloc[i + look_ahead] / df['close'].iloc[i] - 1
            returns.append((df['regime'].iloc[i], ret))

        ret_df = pd.DataFrame(returns, columns=['regime', 'return'])

        print(f"\n持有{look_ahead}天:")
        for regime in [-1, 0, 1]:
            regime_name = {-1: '熊市', 0: '震荡市', 1: '牛市'}[regime]
            subset = ret_df[ret_df['regime'] == regime]['return']
            count = len(subset)
            mean_ret = subset.mean() if count > 0 else 0
            print(f"  {regime_name}: 样本{count:4d}个, 平均收益{mean_ret:7.2%}")

    # 2. 状态持续验证
    print("\n" + "="*60)
    print("【状态持续验证】- 只统计状态连续保持N天的情况")
    print("="*60)

    for persist_days in [1, 3, 5]:
        print(f"\n状态持续{persist_days}天以上:")

        # 计算状态持续天数
        persist_count = 0
        for i in range(len(df) - persist_days):
            # 检查接下来persist_days天是否都是同一状态
            regime = df['regime'].iloc[i]
            is_persistent = all(df['regime'].iloc[i:i+persist_days] == regime)

            if is_persistent:
                ret = df['close'].iloc[i + persist_days] / df['close'].iloc[i] - 1

                if regime not in results:
                    results[regime] = {}
                if f'persist_{persist_days}' not in results[regime]:
                    results[regime][f'persist_{persist_days}'] = []
                results[regime][f'persist_{persist_days}'].append(ret)

        for regime in [-1, 0, 1]:
            regime_name = {-1: '熊市', 0: '震荡市', 1: '牛市'}[regime]
            key = f'persist_{persist_days}'
            if regime in results and key in results[regime]:
                subset = results[regime][key]
                count = len(subset)
                mean_ret = np.mean(subset) if count > 0 else 0
                print(f"  {regime_name}: 样本{count:4d}个, 平均收益{mean_ret:7.2%}")

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

    # 综合诊断
    results = diagnose_regime_comprehensive(index_data, result['regime'].tolist())

    print("="*70)
    print("【综合诊断报告】- 多维度评价体系")
    print("="*70)

    for look_ahead, regime_results in results.items():
        print(f"\n持有期: {look_ahead}天")
        print("-"*70)
        print(f"{'状态':<8} {'样本数':>8} {'平均收益':>10} {'方向准确率':>10} {'超额收益':>10} {'p值':>8} {'显著':>6}")
        print("-"*70)

        for regime_name, stats in regime_results.items():
            print(f"{regime_name:<8} {stats['样本数']:>8} {stats['平均收益']:>10.2%} "
                  f"{stats['方向准确率']:>10.2%} {stats['超额收益']:>10.2%} "
                  f"{stats['p值']:>8.3f} {stats['显著']:>6}")

    print("\n" + "="*70)
    print("【判断标准说明】")
    print("  方向准确率: 预测方向正确的概率")
    print("  超额收益: 相对于随机选择的超额收益")
    print("  p值: 统计显著性 (<0.05为显著, <0.01为高度显著)")
    print("  *** p<0.01, ** p<0.05, * p<0.1")
    print("="*70)

    # 测试动量分数
    print("\n\n" + "="*70)
    print("【动量分数诊断】- 连续值信号")
    print("="*70)

    # 使用动量分数构建信号
    momentum_signals = []
    for i in range(len(result)):
        score = result['momentum_score'].iloc[i]
        # 将连续值转为离散信号
        if score > 0.3:
            sig = 1
        elif score < -0.3:
            sig = -1
        else:
            sig = 0
        momentum_signals.append(sig)

    results_mom = diagnose_regime_comprehensive(index_data, momentum_signals)

    for look_ahead, regime_results in results_mom.items():
        print(f"\n持有期: {look_ahead}天")
        print("-"*70)
        for regime_name, stats in regime_results.items():
            if stats['样本数'] > 0:
                print(f"{regime_name:<8} 样本{stats['样本数']:>4} 平均{stats['平均收益']:>8.2%} 方向{stats['方向准确率']:>7.2%} p={stats['p值']:.3f} {stats['显著']}")

    # 输出示例数据
    print("\n\n" + "="*70)
    print("【信号输出示例】- 最后10天")
    print("="*70)
    sample = result[['datetime', 'close', 'regime', 'confidence', 'momentum_score', 'trend_score', 'volatility', 'is_extreme']].tail(10)
    print(sample.to_string(index=False))

    # 分年份诊断
    print("\n\n" + "="*70)
    print("分年份统计")
    print("="*70)

    yearly_df = diagnose_yearly(index_data, result['regime'].tolist())
    print(yearly_df.to_string(index=False))
