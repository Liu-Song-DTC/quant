# diagnose_position.py
"""
仓位控制诊断
验证仓位决策的质量
"""
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '.')

from core.diagnostics import PositionDiagnostics


def diagnose_position_quality():
    """诊断仓位决策质量"""
    print("加载数据...")

    # 模拟仓位数据 - 实际应该从回测结果中提取
    DATA_PATH = '../data/stock_data/backtrader_data/'

    # 读取指数数据
    index_data = pd.read_csv(DATA_PATH + 'sh000001_qfq.csv', parse_dates=['datetime'])

    # 加载策略结果（模拟数据）
    # 这里我们用市场状态来模拟仓位
    from core.market_regime_detector import MarketRegimeDetector

    detector = MarketRegimeDetector()
    index_with_regime = detector.generate(index_data)

    # 定义仓位策略
    def get_position(regime, consecutive_losses=0):
        if regime == 1:
            return 1.0
        elif regime == 0:
            if consecutive_losses >= 2:
                return 0.25
            elif consecutive_losses >= 1:
                return 0.45
            else:
                return 0.7
        else:
            return 0.0

    # 模拟每日仓位
    positions = []
    for i in range(len(index_with_regime)):
        regime = index_with_regime.iloc[i]['regime']
        # 模拟连续亏损天数（实际应从真实回测获取）
        consecutive_losses = i % 5  # 简化模拟
        pos = get_position(regime, consecutive_losses)
        positions.append(pos)

    positions = pd.Series(positions)

    # 计算组合收益（简化：假设仓位与指数收益成正比）
    index_data['return'] = index_data['close'].pct_change()
    # 仓位调整后的收益 = 仓位 * 市场收益
    adjusted_returns = positions.shift(1) * index_data['return']

    # 去掉NaN
    valid = adjusted_returns.notna()
    positions = positions[valid]
    adjusted_returns = adjusted_returns[valid]

    print("\n" + "="*60)
    print("仓位控制诊断")
    print("="*60)

    report = PositionDiagnostics.diagnose_positions(
        positions,
        adjusted_returns.shift(-20),  # 20天后收益
        "当前仓位策略"
    )
    report.print_report()

    # 分析高仓位时期的实际表现
    print("\n" + "="*60)
    print("分仓位水平分析")
    print("="*60)

    for threshold in [0.7, 0.5, 0.3]:
        high = adjusted_returns[positions >= threshold]
        low = adjusted_returns[positions < threshold]

        if len(high) > 0:
            print(f"\n高仓位(>={threshold}):")
            print(f"  样本数: {len(high)}")
            print(f"  平均收益: {high.mean():.4%}")

        if len(low) > 0:
            print(f"\n低仓位(<{threshold}):")
            print(f"  样本数: {len(low)}")
            print(f"  平均收益: {low.mean():.4%}")

    # 分析不同市场状态的仓位
    print("\n" + "="*60)
    print("分市场状态仓位")
    print("="*60)

    regimes = index_with_regime['regime'].values[:len(positions)]
    positions_df = pd.DataFrame({
        'position': positions.values,
        'return': adjusted_returns.values,
        'regime': regimes
    })

    for regime, name in [(-1, "熊市"), (0, "震荡市"), (1, "牛市")]:
        subset = positions_df[positions_df['regime'] == regime]
        if len(subset) > 0:
            print(f"\n{name}:")
            print(f"  平均仓位: {subset['position'].mean():.2%}")
            print(f"  平均收益: {subset['return'].mean():.4%}")


if __name__ == "__main__":
    diagnose_position_quality()
