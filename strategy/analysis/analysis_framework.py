# analysis/analysis_framework.py
"""
量化策略分析框架 - 统一入口

设计目标：
1. 统一入口，一键运行全部分析
2. 分层分析：因子层 → 信号层 → 组合层 → 时序分析
3. 输出结构化结论和优化建议

数据来源：
- rolling_validation_results/backtest_signals.csv（信号数据）
- rolling_validation_results/portfolio_selections.csv（选股结果）
- rolling_validation_results/validation_results.csv（IC验证结果）

分析模块：
1. 数据质量检查
2. 因子层分析（动态因子选择效果）
3. 信号层分析（买卖信号质量）
4. 组合层分析（持仓分布）
5. 时序稳定性分析
6. 优化建议

使用方式：
    python analysis/analysis_framework.py
"""

import os
import sys
import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Optional
from dataclasses import dataclass, field

# 路径设置
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STRATEGY_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(STRATEGY_DIR)
sys.path.insert(0, STRATEGY_DIR)


# ==================== 配置 ====================

# 目标阈值
IC_TARGET = 0.05       # 5% - 整体IC目标
IR_TARGET = 0.5       # 0.5 - IR目标
ACCURACY_TARGET = 0.55  # 55% - 买卖信号准确率目标


# ==================== 工具函数 ====================

def safe_spearmanr(x: pd.Series, y: pd.Series) -> tuple:
    """安全计算Spearman相关系数"""
    mask = ~(x.isna() | y.isna() | np.isinf(x) | np.isinf(y))
    if mask.sum() < 10:
        return np.nan, np.nan
    return stats.spearmanr(x[mask], y[mask])


# ==================== 主分析框架类 ====================

class AnalysisFramework:
    """
    统一分析框架

    分析流程：
    1. 数据质量检查 - 确保数据可用
    2. 因子层分析 - 动态因子选择是否有效
    3. 信号层分析 - 买卖信号质量
    4. 组合层分析 - 持仓分布
    5. 时序分析 - IC稳定性
    6. 优化建议 - 基于分析结果
    """

    def __init__(self, data_dir: str = None):
        """
        初始化分析框架

        Args:
            data_dir: 数据目录，默认使用 rolling_validation_results/
        """
        if data_dir is None:
            data_dir = os.path.join(STRATEGY_DIR, 'rolling_validation_results')

        self.data_dir = data_dir

        # 加载数据
        self.signals_df = None
        self.selections_df = None
        self.validation_df = None

        self._load_data()

        # 分析结果
        self.factor_results = {}
        self.signal_results = {}
        self.portfolio_results = {}
        self.temporal_results = {}

    def _load_data(self):
        """加载所有数据源"""
        signals_path = os.path.join(self.data_dir, 'backtest_signals.csv')
        selections_path = os.path.join(self.data_dir, 'portfolio_selections.csv')
        validation_path = os.path.join(self.data_dir, 'validation_results.csv')

        if os.path.exists(signals_path):
            self.signals_df = pd.read_csv(signals_path)
            print(f"✓ 加载信号数据: {len(self.signals_df):,} 条")
            print(f"  日期范围: {self.signals_df['date'].min()} ~ {self.signals_df['date'].max()}")

        if os.path.exists(selections_path):
            self.selections_df = pd.read_csv(selections_path)
            print(f"✓ 加载选股数据: {len(self.selections_df):,} 条")

        if os.path.exists(validation_path):
            self.validation_df = pd.read_csv(validation_path)
            print(f"✓ 加载验证数据: {len(self.validation_df):,} 条")

        if self.signals_df is None and self.selections_df is None:
            raise FileNotFoundError(f"未找到数据文件，请先运行回测")

    # ==================== 1. 数据质量检查 ====================

    def check_data_quality(self) -> Dict:
        """
        模块1: 数据质量检查

        检查：
        1. NaN/Inf 比例
        2. 极端值比例
        3. 数据完整性
        """
        print("\n" + "=" * 70)
        print("【模块1: 数据质量检查】")
        print("=" * 70)

        results = {}

        for name, df in [("信号数据", self.signals_df),
                         ("选股数据", self.selections_df),
                         ("验证数据", self.validation_df)]:
            if df is None:
                continue

            print(f"\n{name}:")
            quality = {}

            # 数值列检测
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if col in ['date', 'datetime']:
                    continue
                nan_pct = df[col].isna().mean() * 100
                if nan_pct > 0:
                    status = "✓" if nan_pct < 5 else "⚠" if nan_pct < 20 else "✗"
                    print(f"  {col}: NaN={nan_pct:.1f}% {status}")
                    quality[col] = nan_pct

            # 极端值检测
            if 'factor_value' in df.columns:
                valid = df['factor_value'].dropna()
                if len(valid) > 0:
                    extreme_pct = ((valid > 2) | (valid < -2)).mean() * 100
                    status = "✓" if extreme_pct < 5 else "⚠"
                    print(f"  factor_value极端值: {extreme_pct:.2f}% {status}")
                    quality['factor_value_extreme'] = extreme_pct

            results[name] = quality

        return results

    # ==================== 2. 因子层分析 ====================

    def analyze_factor_layer(self) -> Dict:
        """
        模块2: 因子层分析

        分析：
        1. 动态因子选择分布（是否使用动态因子）
        2. 不同因子类型的IC差异
        3. 因子值分布
        """
        print("\n" + "=" * 70)
        print("【模块2: 因子层分析 - 动态因子选择效果】")
        print("=" * 70)

        results = {}

        if self.validation_df is None:
            print("  (无验证数据)")
            return results

        df = self.validation_df.copy()

        # 1. 动态因子选择分布
        print("\n2.1 动态因子选择分布:")
        df['is_dynamic'] = df['factor_name'].str.startswith('DYN_')
        df['is_fallback'] = df['factor_name'] == 'V41'

        dyn_pct = df['is_dynamic'].mean() * 100
        fallback_pct = df['is_fallback'].mean() * 100

        status = "✓" if dyn_pct > 80 else "⚠" if dyn_pct > 50 else "✗"
        print(f"  动态因子使用率: {dyn_pct:.1f}% {status}")
        print(f"  Fallback(V41)使用率: {fallback_pct:.1f}%")

        results['dynamic_usage_pct'] = dyn_pct

        # 2. 按因子数量分析IC
        print("\n2.2 按因子数量分析IC:")
        df['n_factors'] = df['factor_name'].str.extract(r'_(\d)F_')[0]

        # 排除V41和_T后缀
        df_clean = df[(~df['is_fallback']) & (~df['factor_name'].str.endswith('_T'))]

        count_results = {}
        for n in ['1', '2', '3']:
            subset = df_clean[df_clean['n_factors'] == n].dropna(subset=['factor_value', 'future_ret'])
            if len(subset) > 100:
                ic, p = safe_spearmanr(subset['factor_value'], subset['future_ret'])
                acc = (subset['future_ret'] > 0).mean()
                count_results[f'{n}F'] = {'ic': ic, 'accuracy': acc, 'n': len(subset)}
                marker = " ★" if ic == max(r['ic'] for r in count_results.values()) else ""
                print(f"  {n}因子: IC={ic:.4f} ({ic*100:.2f}%), 准确率={acc:.2%}, n={len(subset):,}{marker}")

        results['ic_by_factor_count'] = count_results

        # 3. 因子IC质量
        print("\n2.3 因子IC质量（按因子名）:")
        factor_ic = {}
        for fn in df_clean['factor_name'].unique():
            if pd.isna(fn):
                continue
            subset = df_clean[df_clean['factor_name'] == fn].dropna(subset=['factor_value', 'future_ret'])
            if len(subset) > 100:
                ic, _ = safe_spearmanr(subset['factor_value'], subset['future_ret'])
                if not np.isnan(ic):
                    factor_ic[fn] = ic

        # 按IC排序
        sorted_factors = sorted(factor_ic.items(), key=lambda x: x[1], reverse=True)

        print("  IC最高 (Top5):")
        for fn, ic in sorted_factors[:5]:
            status = "✓" if ic >= IC_TARGET else "⚠" if ic >= 0.02 else "✗"
            print(f"    {fn}: IC={ic:.4f} ({ic*100:.2f}%) {status}")

        print("  IC最低 (Bottom5):")
        for fn, ic in sorted_factors[-5:]:
            status = "✓" if ic >= IC_TARGET else "⚠" if ic >= 0.02 else "✗"
            print(f"    {fn}: IC={ic:.4f} ({ic*100:.2f}%) {status}")

        results['factor_ic'] = factor_ic

        # 4. 整体IC
        valid = df_clean.dropna(subset=['factor_value', 'future_ret'])
        if len(valid) > 100:
            overall_ic, _ = safe_spearmanr(valid['factor_value'], valid['future_ret'])
            results['overall_ic'] = overall_ic
            status = "✓" if abs(overall_ic) >= IC_TARGET else "⚠" if abs(overall_ic) >= 0.02 else "✗"
            print(f"\n  整体IC: {overall_ic:.4f} ({overall_ic*100:.2f}%) {status}")

        self.factor_results = results
        return results

    # ==================== 3. 信号层分析 ====================

    def analyze_signal_layer(self) -> Dict:
        """
        模块3: 信号层分析

        分析：
        1. 买卖信号分布
        2. 买入信号质量（准确率、平均收益）
        3. 卖出信号质量
        4. 信号IC
        """
        print("\n" + "=" * 70)
        print("【模块3: 信号层分析 - 买卖信号质量】")
        print("=" * 70)

        results = {}

        # 使用 validation_df（有future_ret）进行信号分析
        # 回退到 signals_df 如果 validation_df 不可用
        df = self.validation_df if self.validation_df is not None else self.signals_df
        if df is None:
            print("  (无信号数据)")
            return results

        df = df.copy()

        # 信号分布
        total = len(df)
        buy_n = (df['buy'] == 1).sum() if 'buy' in df.columns else 0
        sell_n = (df['sell'] == 1).sum() if 'sell' in df.columns else 0
        no_signal = total - buy_n - sell_n

        results['distribution'] = {
            'total': total, 'buy': buy_n, 'sell': sell_n, 'no_signal': no_signal
        }

        print(f"\n3.1 信号分布:")
        print(f"  总观测: {total:,}")
        print(f"  买入信号: {buy_n:,} ({buy_n/total*100:.1f}%)")
        print(f"  卖出信号: {sell_n:,} ({sell_n/total*100:.1f}%)")
        print(f"  无信号: {no_signal:,} ({no_signal/total*100:.1f}%)")

        # 买入信号质量
        if 'buy' in df.columns:
            buy_signals = df[df['buy'] == 1].dropna(subset=['future_ret'])
            if len(buy_signals) > 0:
                buy_acc = (buy_signals['future_ret'] > 0).mean()
                buy_avg = buy_signals['future_ret'].mean()
                buy_median = buy_signals['future_ret'].median()

                results['buy'] = {
                    'accuracy': buy_acc,
                    'avg_return': buy_avg,
                    'median_return': buy_median,
                    'count': len(buy_signals)
                }

                status = "✓" if buy_acc >= ACCURACY_TARGET else "⚠"
                print(f"\n3.2 买入信号质量 {status}:")
                print(f"  数量: {len(buy_signals):,}")
                print(f"  准确率: {buy_acc:.2%}")
                print(f"  平均收益: {buy_avg*100:.2f}%")
                print(f"  中位数收益: {buy_median*100:.2f}%")

        # 卖出信号质量
        if 'sell' in df.columns:
            sell_signals = df[df['sell'] == 1].dropna(subset=['future_ret'])
            if len(sell_signals) > 0:
                sell_acc = (sell_signals['future_ret'] < 0).mean()
                sell_avg = sell_signals['future_ret'].mean()

                results['sell'] = {
                    'accuracy': sell_acc,
                    'avg_return': sell_avg,
                    'count': len(sell_signals)
                }

                status = "✓" if sell_acc >= ACCURACY_TARGET else "⚠"
                print(f"\n3.3 卖出信号质量 {status}:")
                print(f"  数量: {len(sell_signals):,}")
                print(f"  准确率: {sell_acc:.2%}")
                print(f"  平均收益: {sell_avg*100:.2f}%")

        # 信号IC
        valid = df.dropna(subset=['factor_value', 'future_ret'])
        if len(valid) > 100:
            ic, p = safe_spearmanr(valid['factor_value'], valid['future_ret'])
            results['signal_ic'] = ic
            status = "✓" if abs(ic) >= IC_TARGET else "⚠" if abs(ic) >= 0.02 else "✗"
            print(f"\n3.4 信号IC: {ic:.4f} ({ic*100:.2f}%) {status}")

        self.signal_results = results
        return results

    # ==================== 4. 组合层分析 ====================

    def analyze_portfolio_layer(self) -> Dict:
        """
        模块4: 组合层分析

        分析：
        1. 持仓分布
        2. 行业分布
        3. 选股集中度
        """
        print("\n" + "=" * 70)
        print("【模块4: 组合层分析 - 持仓分布】")
        print("=" * 70)

        results = {}

        if self.selections_df is None:
            print("  (无选股数据)")
            return results

        df = self.selections_df.copy()

        # 基本统计
        n_dates = df['date'].nunique()
        n_stocks = df['code'].nunique()
        avg_positions = len(df) / n_dates if n_dates > 0 else 0

        results['basic'] = {
            'n_dates': n_dates,
            'n_stocks': n_stocks,
            'avg_positions': avg_positions
        }

        print(f"\n4.1 选股统计:")
        print(f"  调仓次数: {n_dates}")
        print(f"  覆盖股票: {n_stocks}")
        print(f"  平均持仓: {avg_positions:.1f} 只/次")

        # 行业分布
        if 'industry' in df.columns:
            ind_dist = df['industry'].value_counts()
            top3_pct = ind_dist.head(3).sum() / len(df) * 100
            top5_pct = ind_dist.head(5).sum() / len(df) * 100

            results['industry_distribution'] = ind_dist.to_dict()
            results['top3_concentration'] = top3_pct
            results['top5_concentration'] = top5_pct

            print(f"\n4.2 行业分布 (Top5):")
            for ind, count in ind_dist.head(5).items():
                pct = count / len(df) * 100
                print(f"  {ind}: {count} ({pct:.1f}%)")

            print(f"\n  前3行业集中度: {top3_pct:.1f}%")
            print(f"  前5行业集中度: {top5_pct:.1f}%")

            if top3_pct > 60:
                print("  ⚠ 集中度过高，建议增加行业分散")

        # 权重分布
        if 'weight' in df.columns:
            weight_std = df['weight'].std()
            weight_max = df['weight'].max()
            weight_mean = df['weight'].mean()

            results['weight_stats'] = {
                'std': weight_std,
                'max': weight_max,
                'mean': weight_mean
            }

            print(f"\n4.3 权重分布:")
            print(f"  平均权重: {weight_mean:.4f}")
            print(f"  权重标准差: {weight_std:.4f}")
            print(f"  最大权重: {weight_max:.4f}")

        # 选股频率
        stock_freq = df['code'].value_counts()
        top_stock_pct = stock_freq.head(1).iloc[0] / len(df) * 100 if len(stock_freq) > 0 else 0
        results['top_stock_frequency'] = top_stock_pct
        print(f"\n4.4 最常选入股票: {stock_freq.index[0] if len(stock_freq) > 0 else 'N/A'} ({top_stock_pct:.1f}%)")

        self.portfolio_results = results
        return results

    # ==================== 5. 时序稳定性分析 ====================

    def analyze_temporal_stability(self) -> Dict:
        """
        模块5: 时序稳定性分析

        分析：
        1. IC随时间的稳定性
        2. 不同年份表现差异
        3. IC趋势
        """
        print("\n" + "=" * 70)
        print("【模块5: 时序稳定性分析 - IC稳定性】")
        print("=" * 70)

        results = {}

        df = self.validation_df if self.validation_df is not None else self.signals_df
        if df is None:
            print("  (无数据)")
            return results

        df = df.copy()
        df['year'] = pd.to_datetime(df['date']).dt.year

        # 按年份计算IC
        yearly_ic = {}
        for year in sorted(df['year'].unique()):
            subset = df[df['year'] == year].dropna(subset=['factor_value', 'future_ret'])
            if len(subset) > 100:
                ic, _ = safe_spearmanr(subset['factor_value'], subset['future_ret'])
                if not np.isnan(ic):
                    yearly_ic[year] = ic

        results['yearly_ic'] = yearly_ic

        print(f"\n5.1 年度IC:")
        ic_values = list(yearly_ic.values())
        for year, ic in sorted(yearly_ic.items()):
            marker = " ★" if ic >= IC_TARGET else " ⚠" if ic < 0 else ""
            print(f"  {year}: {ic:.4f} ({ic*100:.2f}%){marker}")

        # 趋势分析
        if len(yearly_ic) >= 2:
            recent_years = sorted(yearly_ic.keys())[-5:]
            ic_trend = [yearly_ic[y] for y in recent_years]
            trend_direction = "↑" if ic_trend[-1] > ic_trend[0] else "↓"
            ic_mean = np.mean(ic_trend)
            ic_std = np.std(ic_trend)
            ir = ic_mean / (ic_std + 1e-10)

            results['trend'] = {
                'direction': trend_direction,
                'mean_ic': ic_mean,
                'std_ic': ic_std,
                'ir': ir
            }

            print(f"\n5.2 近{len(recent_years)}年IC趋势: {trend_direction}")
            print(f"  均值: {ic_mean:.4f}")
            print(f"  标准差: {ic_std:.4f}")
            print(f"  IR: {ir:.4f}")

            status = "✓" if ir >= IR_TARGET else "⚠" if ir >= 0.3 else "✗"
            print(f"  IR目标: >{IR_TARGET:.2f} {status}")

        # 月度IC波动（如果有足够数据）
        if len(df) > 1000:
            df['month'] = pd.to_datetime(df['date']).dt.to_period('M')
            monthly_ic = {}
            for month in sorted(df['month'].unique())[-12:]:  # 最近12个月
                subset = df[df['month'] == month].dropna(subset=['factor_value', 'future_ret'])
                if len(subset) > 50:
                    ic, _ = safe_spearmanr(subset['factor_value'], subset['future_ret'])
                    if not np.isnan(ic):
                        monthly_ic[str(month)] = ic

            if monthly_ic:
                results['monthly_ic'] = monthly_ic
                print(f"\n5.3 最近月份IC (Top3/Bottom3):")
                sorted_monthly = sorted(monthly_ic.items(), key=lambda x: x[1], reverse=True)
                for m, ic in sorted_monthly[:3]:
                    print(f"  {m}: {ic:.4f}")
                print("  ...")
                for m, ic in sorted_monthly[-3:]:
                    print(f"  {m}: {ic:.4f}")

        self.temporal_results = results
        return results

    # ==================== 6. 优化建议 ====================

    def generate_recommendations(self) -> List[str]:
        """
        模块6: 基于分析结果生成优化建议
        """
        print("\n" + "=" * 70)
        print("【模块6: 优化建议汇总】")
        print("=" * 70)

        recommendations = []

        # 1. 基于因子层分析
        if self.factor_results:
            dyn_pct = self.factor_results.get('dynamic_usage_pct', 0)
            if dyn_pct < 80:
                recommendations.append(f"动态因子使用率仅{dyn_pct:.1f}%，低于80%，建议检查动态因子选择逻辑")

            ic_by_count = self.factor_results.get('ic_by_factor_count', {})
            if ic_by_count:
                best_n = max(ic_by_count.items(), key=lambda x: x[1].get('ic', 0))
                if best_n[1]['ic'] > 0.03:
                    recommendations.append(f"最优因子数量: {best_n[0]} (IC={best_n[1]['ic']:.4f})，可考虑提高该组合权重")

            overall_ic = self.factor_results.get('overall_ic', 0)
            if abs(overall_ic) < 0.02:
                recommendations.append(f"整体IC仅{overall_ic:.4f}，低于2%阈值，建议全面检查因子质量")

        # 2. 基于信号层分析
        if self.signal_results:
            buy_acc = self.signal_results.get('buy', {}).get('accuracy', 0)
            if buy_acc < 0.5:
                recommendations.append(f"买入信号准确率仅{buy_acc:.2%}，低于50%，建议优化买入信号逻辑")

            signal_ic = self.signal_results.get('signal_ic', 0)
            if abs(signal_ic) < 0.02:
                recommendations.append(f"信号IC仅{signal_ic:.4f}，建议检查因子与信号转换逻辑")

        # 3. 基于组合层分析
        if self.portfolio_results:
            top3 = self.portfolio_results.get('top3_concentration', 0)
            if top3 > 60:
                recommendations.append(f"前3行业集中度{top3:.1f}%过高，建议增加行业分散")

        # 4. 基于时序分析
        if self.temporal_results:
            yearly_ic = self.temporal_results.get('yearly_ic', {})
            negative_years = [y for y, ic in yearly_ic.items() if ic < 0]
            if negative_years:
                recommendations.append(f"负IC年份: {negative_years}，建议在这些年份降低仓位或排除相关行业")

        # 默认建议
        if not recommendations:
            recommendations.append("各项指标正常，维持当前策略")

        print()
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")

        return recommendations

    # ==================== 运行全部分析 ====================

    def run_full_analysis(self) -> Dict:
        """
        运行完整分析流程

        Returns:
            Dict: 包含所有分析结果
        """
        print("\n" + "#" * 70)
        print("# 量化策略分析框架 - 完整分析")
        print("#" * 70)

        all_results = {}

        # 1. 数据质量
        all_results['data_quality'] = self.check_data_quality()

        # 2. 因子层
        all_results['factor_layer'] = self.analyze_factor_layer()

        # 3. 信号层
        all_results['signal_layer'] = self.analyze_signal_layer()

        # 4. 组合层
        all_results['portfolio_layer'] = self.analyze_portfolio_layer()

        # 5. 时序稳定性
        all_results['temporal'] = self.analyze_temporal_stability()

        # 6. 优化建议
        all_results['recommendations'] = self.generate_recommendations()

        # 打印汇总
        self._print_summary(all_results)

        return all_results

    def _print_summary(self, results: Dict):
        """打印分析汇总"""
        print("\n" + "=" * 70)
        print("【分析汇总 - 关键指标】")
        print("=" * 70)

        print(f"\n目标阈值: IC>{IC_TARGET:.0%}, IR>{IR_TARGET:.2f}, 准确率>{ACCURACY_TARGET:.0%}")

        # 因子层
        if 'factor_layer' in results:
            fl = results['factor_layer']
            dyn_pct = fl.get('dynamic_usage_pct', 0)
            overall_ic = fl.get('overall_ic', 0)
            print(f"\n因子层:")
            print(f"  动态因子使用率: {dyn_pct:.1f}%")
            print(f"  整体IC: {overall_ic:.4f} ({overall_ic*100:.2f}%)")

            ic_by_count = fl.get('ic_by_factor_count', {})
            if ic_by_count:
                best = max(ic_by_count.items(), key=lambda x: x[1].get('ic', 0))
                print(f"  最优因子组合: {best[0]} (IC={best[1]['ic']:.4f})")

        # 信号层
        if 'signal_layer' in results:
            sl = results['signal_layer']
            print(f"\n信号层:")
            if 'buy' in sl:
                print(f"  买入准确率: {sl['buy']['accuracy']:.2%}")
            if 'signal_ic' in sl:
                print(f"  信号IC: {sl['signal_ic']:.4f}")

        # 组合层
        if 'portfolio_layer' in results:
            pl = results['portfolio_layer']
            print(f"\n组合层:")
            if 'top3_concentration' in pl:
                print(f"  前3行业集中度: {pl['top3_concentration']:.1f}%")

        # 时序
        if 'temporal' in results:
            t = results['temporal']
            if 'trend' in t:
                print(f"\n稳定性:")
                print(f"  近5年IR: {t['trend']['ir']:.4f}")


if __name__ == '__main__':
    framework = AnalysisFramework()
    results = framework.run_full_analysis()
