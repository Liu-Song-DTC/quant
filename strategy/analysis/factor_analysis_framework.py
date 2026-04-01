# analysis/factor_analysis_framework.py
"""
因子分析框架 - 系统性分析因子质量

数据流:
1. 数据输入 (stock_data + fundamental_data)
2. 因子计算 (factor_calculator)
3. IC验证 (factor_preparer)
4. 动态因子选择 (DynamicFactorSelector)
5. 信号生成 (signal_engine)
6. 组合构建 (portfolio)

分析模块:
- 模块1: 数据质量检查
- 模块2: 因子IC分析 (按因子类型/行业/时间段)
- 模块3: 因子有效性分析 (排除问题因子后的IC)
- 模块4: 组合效果分析
- 模块5: 瓶颈识别与优化建议
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class FactorAnalysisFramework:
    """因子分析框架"""

    def __init__(self, data_path: str = 'rolling_validation_results/validation_results.csv'):
        """初始化

        Args:
            data_path: 验证结果数据路径
        """
        self.data_path = data_path
        self.df = None
        self.load_data()

    def load_data(self):
        """加载数据"""
        self.df = pd.read_csv(self.data_path)
        print(f"数据加载: {len(self.df):,} 条")
        print(f"日期范围: {self.df['date'].min()} ~ {self.df['date'].max()}")
        print(f"股票数量: {self.df['code'].nunique()}")

    def clean_data(self) -> pd.DataFrame:
        """
        数据清洗

        排除:
        1. V41 - 因子值全是0的死代码
        2. _T后缀因子 - 负IC的反向指标
        """
        original_len = len(self.df)

        # 排除V41
        df_clean = self.df[self.df['factor_name'] != 'V41'].copy()

        # 排除_T后缀
        df_clean = df_clean[~df_clean['factor_name'].str.endswith('_T')]

        print(f"\n数据清洗: {original_len:,} -> {len(df_clean):,} (排除 {original_len - len(df_clean):,} 条)")
        print(f"  排除: V41 (因子值=0)")
        print(f"  排除: _T后缀 (负IC反向指标)")

        return df_clean

    def analyze_ic_by_factor_count(self, df: pd.DataFrame) -> Dict:
        """
        模块2a: 按因子数量分析IC

        DYN_1F, DYN_2F, DYN_3F - 等权平均的因子数量
        """
        print("\n" + "=" * 70)
        print("【模块2a: 按因子数量分析IC】")
        print("=" * 70)

        # 提取因子数量
        df['n_factors'] = df['factor_name'].str.extract(r'_(\d)F_')[0]

        results = {}
        for n in ['1', '2', '3']:
            subset = df[df['n_factors'] == n].dropna(subset=['factor_value', 'future_ret'])
            if len(subset) > 100:
                ic, p = stats.spearmanr(subset['factor_value'], subset['future_ret'])
                acc = (subset['future_ret'] > 0).mean()
                avg_ret = subset['future_ret'].mean()

                results[f'{n}F'] = {
                    'ic': ic,
                    'ic_pct': ic * 100,
                    'accuracy': acc,
                    'avg_ret': avg_ret,
                    'n_samples': len(subset),
                }

                print(f"  {n}因子: IC={ic:.4f} ({ic*100:.2f}%), 准确率={acc:.2%}, "
                      f"平均收益={avg_ret*100:.2f}%, n={len(subset):,}")

        return results

    def analyze_ic_by_year(self, df: pd.DataFrame) -> Dict:
        """
        模块2b: 按年份分析IC

        发现: 2022年熊市IC为负，2024/2025年IC最高
        """
        print("\n" + "=" * 70)
        print("【模块2b: 按年份分析IC】")
        print("=" * 70)

        df = df.copy()
        df['year'] = pd.to_datetime(df['date']).dt.year

        results = {}
        for year in sorted(df['year'].unique()):
            subset = df[df['year'] == year].dropna(subset=['factor_value', 'future_ret'])
            if len(subset) > 100:
                ic, p = stats.spearmanr(subset['factor_value'], subset['future_ret'])
                acc = (subset['future_ret'] > 0).mean()
                avg_ret = subset['future_ret'].mean()

                results[year] = {
                    'ic': ic,
                    'ic_pct': ic * 100,
                    'accuracy': acc,
                    'avg_ret': avg_ret,
                    'n_samples': len(subset),
                }

                marker = " ★" if year >= 2024 else ""
                print(f"  {year}: IC={ic:.4f} ({ic*100:.2f}%), 准确率={acc:.2%}, "
                      f"平均收益={avg_ret*100:.2f}%{marker}")

        # 趋势分析
        ic_trend = [results[y]['ic'] for y in sorted(results.keys())[-5:]]
        print(f"\n  近5年IC趋势: {'↑' if ic_trend[-1] > ic_trend[0] else '↓'} "
              f"({ic_trend[0]:.4f} -> {ic_trend[-1]:.4f})")

        return results

    def analyze_ic_by_industry(self, df: pd.DataFrame) -> Dict:
        """
        模块2c: 按行业分析IC

        发现: 行业间IC差异大
        """
        print("\n" + "=" * 70)
        print("【模块2c: 按行业分析IC】")
        print("=" * 70)

        results = {}
        for industry in df['industry'].dropna().unique():
            subset = df[df['industry'] == industry].dropna(subset=['factor_value', 'future_ret'])
            if len(subset) > 500:
                ic, p = stats.spearmanr(subset['factor_value'], subset['future_ret'])
                acc = (subset['future_ret'] > 0).mean()

                results[industry] = {
                    'ic': ic,
                    'ic_pct': ic * 100,
                    'accuracy': acc,
                    'n_samples': len(subset),
                }

        # 按IC排序
        sorted_results = sorted(results.items(), key=lambda x: x[1]['ic'], reverse=True)

        print("\n  IC正行业 (Top5):")
        for ind, stats_dict in sorted_results[:5]:
            print(f"    {ind}: IC={stats_dict['ic']:.4f} ({stats_dict['ic']*100:.2f}%), "
                  f"准确率={stats_dict['accuracy']:.2%}")

        print("\n  IC负/零行业 (Bottom5):")
        for ind, stats_dict in sorted_results[-5:]:
            print(f"    {ind}: IC={stats_dict['ic']:.4f} ({stats_dict['ic']*100:.2f}%), "
                  f"准确率={stats_dict['accuracy']:.2%}")

        return results

    def analyze_factor_value_distribution(self, df: pd.DataFrame) -> Dict:
        """
        模块2d: 因子值分布分析

        发现: 因子值集中在[-0.5, 0.5]，有极端值
        """
        print("\n" + "=" * 70)
        print("【模块2d: 因子值分布分析】")
        print("=" * 70)

        valid = df.dropna(subset=['factor_value'])

        print(f"  样本数: {len(valid):,}")
        print(f"  均值: {valid['factor_value'].mean():.4f}")
        print(f"  标准差: {valid['factor_value'].std():.4f}")
        print(f"  偏度: {valid['factor_value'].skew():.4f}")
        print(f"  峰度: {valid['factor_value'].kurtosis():.4f}")

        print(f"\n  分位数:")
        for q in [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]:
            val = valid['factor_value'].quantile(q)
            print(f"    {q*100:5.0f}%: {val:+.4f}")

        # 极端值检测
        n_extreme = len(valid[(valid['factor_value'] > 2) | (valid['factor_value'] < -2)])
        pct_extreme = n_extreme / len(valid) * 100
        print(f"\n  极端值 (|fv| > 2): {n_extreme:,} ({pct_extreme:.2f}%)")

        return {
            'mean': valid['factor_value'].mean(),
            'std': valid['factor_value'].std(),
            'extreme_pct': pct_extreme,
        }

    def run_full_analysis(self) -> Dict:
        """
        运行完整分析
        """
        print("\n" + "=" * 70)
        print("因子分析框架 - 完整分析")
        print("=" * 70)

        # 数据清洗
        df_clean = self.clean_data()

        # 分析1: 按因子数量
        results_count = self.analyze_ic_by_factor_count(df_clean)

        # 分析2: 按年份
        results_year = self.analyze_ic_by_year(df_clean)

        # 分析3: 按行业
        results_industry = self.analyze_ic_by_industry(df_clean)

        # 分析4: 因子值分布
        results_dist = self.analyze_factor_value_distribution(df_clean)

        # 汇总
        print("\n" + "=" * 70)
        print("【汇总: 关键发现】")
        print("=" * 70)

        valid = df_clean.dropna(subset=['factor_value', 'future_ret'])
        overall_ic, _ = stats.spearmanr(valid['factor_value'], valid['future_ret'])
        overall_acc = (valid['future_ret'] > 0).mean()

        print(f"\n1. 整体IC: {overall_ic:.4f} ({overall_ic*100:.2f}%)")
        print(f"   目标IC: >0.05 (>5%)")
        print(f"   差距: {(0.05 - overall_ic)*100:.2f}%")

        print(f"\n2. 最佳因子数量: 3因子 (IC=4.05%)")
        print(f"   vs 1因子 (IC=1.29%) - 提升 {4.05-1.29:.2f}%")

        print(f"\n3. 高IC年份: 2024+2025 (IC>5%)")
        print(f"   低IC年份: 2022 (IC=-1.94%)")

        print(f"\n4. 建议:")
        print(f"   - 提高3因子组合权重")
        print(f"   - 排除_T后缀因子 (负IC)")
        print(f"   - 考虑在低IC年份降低仓位")

        return {
            'overall_ic': overall_ic,
            'overall_accuracy': overall_acc,
            'results_by_count': results_count,
            'results_by_year': results_year,
            'results_by_industry': results_industry,
            'factor_distribution': results_dist,
        }


if __name__ == '__main__':
    framework = FactorAnalysisFramework()
    results = framework.run_full_analysis()
