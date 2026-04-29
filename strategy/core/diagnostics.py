# core/diagnostics.py
"""
系统诊断框架
每个模块都可以单独验证质量，不只看端到端效果
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Callable


class DiagnosticReport:
    """诊断报告"""
    def __init__(self, module_name: str):
        self.module_name = module_name
        self.metrics = {}
        self.passed = True

    def add_metric(self, name: str, value: float, threshold: float = None, direction: str = "higher"):
        """添加指标"""
        self.metrics[name] = {
            'value': value,
            'threshold': threshold,
            'direction': direction,  # "higher" = 越大越好, "lower" = 越小越好
        }
        if threshold is not None:
            if direction == "higher":
                ok = value >= threshold
            else:
                ok = value <= threshold
            if not ok:
                self.passed = False

    def print_report(self):
        print(f"\n{'='*60}")
        print(f"诊断报告: {self.module_name}")
        print(f"{'='*60}")

        for name, m in self.metrics.items():
            value = m['value']
            if abs(value) < 1:
                value_str = f"{value:.2%}"
            else:
                value_str = f"{value:.2f}"

            status = "✓" if m['threshold'] is None or (
                (m['direction'] == "higher" and value >= m['threshold']) or
                (m['direction'] == "lower" and value <= m['threshold'])
            ) else "✗"

            thresh_str = f" (阈值: {m['threshold']:.2%})" if m['threshold'] else ""
            print(f"  {status} {name}: {value_str}{thresh_str}")

        print(f"{'='*60}")
        return self.passed


class FactorDiagnostics:
    """
    因子诊断
    验证某个因子的预测能力
    """

    @staticmethod
    def diagnose_factor(
        factor_values: pd.Series,  # 因子值
        future_returns: pd.Series,  # 未来收益
        name: str = "因子"
    ) -> DiagnosticReport:
        """
        诊断因子质量

        Args:
            factor_values: 因子值（越高越好）
            future_returns: 未来收益
            name: 因子名称

        Returns:
            DiagnosticReport
        """
        report = DiagnosticReport(f"因子: {name}")

        # 去除NaN
        valid = factor_values.notna() & future_returns.notna()
        fv = factor_values[valid]
        fr = future_returns[valid]

        if len(fv) < 100:
            print(f"警告: 数据点太少 ({len(fv)})")
            return report

        # 1. IC (Information Coefficient) - 因子与收益的相关性
        # A股市场技术面因子IC通常在2-4%，目标3%+
        ic = fv.corr(fr)
        report.add_metric("IC (相关系数)", ic, threshold=0.03, direction="higher")

        # 2. 分组回测 - 高因子组 vs 低因子组
        fv_quantile = pd.qcut(fv, q=5, labels=False, duplicates='drop')

        returns_by_group = []
        for q in range(5):
            group_returns = fr[fv_quantile == q]
            if len(group_returns) > 0:
                returns_by_group.append(group_returns.mean())

        if len(returns_by_group) >= 2:
            spread = returns_by_group[-1] - returns_by_group[0]  # 高-低
            report.add_metric("多空收益差 (20日)", spread, threshold=0.0, direction="higher")

        # 3. 因子有效性比例 - 因子值top20%的日子，未来收益>0的比例
        top_ratio = fr[fv >= fv.quantile(0.8)].mean()
        report.add_metric("高因子组平均收益", top_ratio, threshold=0.0, direction="higher")

        return report

    @staticmethod
    def diagnose_signal(
        signals: pd.Series,  # 信号 (1=买入, 0=持有, -1=卖出)
        future_returns: pd.Series,  # 未来收益
        name: str = "信号"
    ) -> DiagnosticReport:
        """
        诊断信号质量

        Args:
            signals: 信号序列
            future_returns: 未来收益
            name: 信号名称

        Returns:
            DiagnosticReport
        """
        report = DiagnosticReport(f"信号: {name}")

        # 去除NaN
        valid = signals.notna() & future_returns.notna()
        sig = signals[valid]
        fr = future_returns[valid]

        if len(sig) < 100:
            return report

        # 买入信号后的收益
        buy_signals = sig == 1
        if buy_signals.sum() > 0:
            buy_returns = fr[buy_signals]
            report.add_metric("买入信号后平均收益 (20日)", buy_returns.mean(), threshold=0.0, direction="higher")
            report.add_metric("买入信号胜率", (buy_returns > 0).mean(), threshold=0.5, direction="higher")

        # 卖出信号后的收益
        sell_signals = sig == -1
        if sell_signals.sum() > 0:
            sell_returns = fr[sell_signals]
            report.add_metric("卖出信号后平均收益 (20日)", sell_returns.mean(), threshold=0.0, direction="lower")

        return report


class MarketRegimeDiagnostics:
    """
    市场状态诊断
    验证市场状态判断是否正确
    """

    @staticmethod
    def diagnose_regime(
        regime_series: pd.Series,  # 市场状态 (-1=熊, 0=震荡, 1=牛)
        index_data: pd.DataFrame,  # 包含close价格
        look_ahead: int = 20
    ) -> DiagnosticReport:
        """诊断市场状态判断质量"""
        report = DiagnosticReport("市场状态判断")

        df = pd.DataFrame({
            'regime': regime_series,
            'close': index_data['close']
        })

        # 计算未来收益
        df['future_return'] = df['close'].shift(-look_ahead) / df['close'] - 1

        # 去掉NaN
        df = df.dropna(subset=['future_return'])

        for regime, name in [(-1, "熊市"), (0, "震荡市"), (1, "牛市")]:
            subset = df[df['regime'] == regime]['future_return']

            if len(subset) > 0:
                avg_ret = subset.mean()
                win_rate = (subset > 0).mean()

                if regime == 1:  # 牛市
                    report.add_metric(f"{name}判断后收益({look_ahead}日)", avg_ret, threshold=0.0, direction="higher")
                elif regime == -1:  # 熊市
                    report.add_metric(f"{name}判断后收益({look_ahead}日)", avg_ret, threshold=0.0, direction="lower")
                else:  # 震荡
                    report.add_metric(f"{name}判断后收益({look_ahead}日)", avg_ret)

        return report


class PositionDiagnostics:
    """
    仓位诊断
    验证仓位决策的质量
    """

    @staticmethod
    def diagnose_positions(
        positions: pd.Series,  # 仓位 (0-1)
        future_returns: pd.Series,  # 组合未来收益
        name: str = "仓位"
    ) -> DiagnosticReport:
        """诊断仓位决策质量"""
        report = DiagnosticReport(f"仓位: {name}")

        valid = positions.notna() & future_returns.notna()
        pos = positions[valid]
        fr = future_returns[valid]

        if len(pos) < 100:
            return report

        # 高仓位时的收益
        high_pos = fr[pos > 0.7]
        low_pos = fr[pos < 0.3]

        if len(high_pos) > 0:
            report.add_metric("高仓位(>70%)时收益", high_pos.mean(), threshold=0.0, direction="higher")
        if len(low_pos) > 0:
            report.add_metric("低仓位(<30%)时收益", low_pos.mean())

        # 仓位与收益的相关性
        corr = pos.corr(fr)
        report.add_metric("仓位-收益相关性", corr, threshold=0.0, direction="higher")

        return report


# 便捷函数
def run_full_diagnostics(
    index_data: pd.DataFrame,
    regime_series: pd.Series,
    factor_data: pd.DataFrame,  # 因子数据
    positions: pd.Series = None
):
    """运行完整诊断"""
    print("\n" + "="*60)
    print("系统完整诊断报告")
    print("="*60)

    # 1. 市场状态诊断
    regime_report = MarketRegimeDiagnostics.diagnose_regime(
        regime_series, index_data, look_ahead=20
    )
    regime_report.print_report()

    # 2. 因子诊断
    for col in factor_data.columns:
        factor = factor_data[col]
        future_ret = factor_data['close'].shift(-20) / factor_data['close'] - 1
        factor_report = FactorDiagnostics.diagnose_factor(factor, future_ret, col)
        factor_report.print_report()

    print("\n✓ 诊断完成")
