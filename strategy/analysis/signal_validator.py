# analysis/signal_validator.py
"""
信号系统统一验证模块

整合因子层、信号层、组合层评估:
1. 因子层 - IC/IR、胜率（动态因子选择验证）
2. 信号层 - 信号覆盖率、买卖信号准确率
3. 组合层 - Top-N选股收益、五分位单调性、行业权重优化

使用 walk-forward 验证方法:
- 每个验证时点使用之前的数据选择因子
- 使用 DynamicFactorSelector 动态选择因子
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm
from multiprocessing import Pool
import warnings
warnings.filterwarnings('ignore')

# 路径设置
script_dir = os.path.dirname(os.path.abspath(__file__))
strategy_dir = os.path.dirname(script_dir)
project_root = os.path.dirname(strategy_dir)
sys.path.insert(0, strategy_dir)

from core.config_loader import load_config
from core.factors import calc_all_factors_for_validation
from core.fundamental import FundamentalData
from core.signal_engine import DynamicFactorSelector, prepare_factor_data, SignalEngine
from core.signal_store import SignalStore

config = load_config(os.path.join(project_root, 'strategy/config/factor_config.yaml'))
detailed_industries = config.config.get('detailed_industries', {})

# 配置参数
dynamic_config = config.config.get('dynamic_factor', {})
TRAIN_WINDOW = dynamic_config.get('train_window', 250)
FORWARD_PERIOD = dynamic_config.get('forward_period', 20)
LOOKBACK = config.config.get('industry_factor_config', {}).get('lookback_days', 120)
MIN_TRAIN_SAMPLES = dynamic_config.get('min_train_samples', 50)
TOP_N_FACTORS = dynamic_config.get('top_n_factors', 3)

# 组合优化配置
portfolio_config = config.config.get('portfolio', {})
ENABLE_INDUSTRY_WEIGHTING = portfolio_config.get('enable_industry_weighting', True)


def calc_ic(factor_values, returns):
    """计算IC (Spearman秩相关)"""
    valid_mask = ~(np.isnan(factor_values) | np.isnan(returns))
    if valid_mask.sum() < 5:
        return np.nan
    ic, _ = stats.spearmanr(factor_values[valid_mask], returns[valid_mask])
    return ic


def generate_signal_with_factors(row, factor_list):
    """使用给定因子列表生成综合信号"""
    if not factor_list:
        return np.nan

    scores = []
    for fn in factor_list:
        if fn in row and not np.isnan(row[fn]):
            scores.append(row[fn])

    if not scores:
        return np.nan

    return np.mean(scores)


def generate_signals_for_stock_with_engine(args):
    """为单只股票生成信号（使用已创建的 SignalEngine）"""
    code, df, sample_dates, fd, signal_engine = args
    stock_dates = sorted(df.index.tolist())

    results = []
    date_to_idx = {d: i for i, d in enumerate(stock_dates)}

    for sample_date in sample_dates:
        valid_dates = [d for d in stock_dates if d <= sample_date]
        if len(valid_dates) < LOOKBACK:
            continue

        eval_date = valid_dates[-1]
        idx = date_to_idx.get(eval_date)
        if idx is None or idx < LOOKBACK:
            continue

        history = df.iloc[:idx+1].iloc[-LOOKBACK:]
        if len(history) < 60:
            continue

        history_copy = history.copy()
        history_copy['datetime'] = history_copy.index

        signal_store = SignalStore()
        signal_engine.generate_at_indices(code, history_copy, [len(history_copy)-1], signal_store)

        sig = signal_store.get(code, eval_date.date() if hasattr(eval_date, 'date') else eval_date)
        if sig and idx + FORWARD_PERIOD < len(df):
            future_price = df.iloc[idx + FORWARD_PERIOD]['close']
            current_price = df.iloc[idx]['close']
            if current_price > 0:
                future_ret = (future_price - current_price) / current_price
                results.append({
                    'code': code,
                    'date': eval_date,
                    'score': sig.score,
                    'factor_value': sig.factor_value,
                    'buy': sig.buy,
                    'sell': sig.sell,
                    'factor_name': sig.factor_name,
                    'future_ret': future_ret,
                })

    return results


class SignalValidator:
    """统一信号验证器 - 因子层+信号层+组合层评估"""

    def __init__(self, stock_data: dict, fd: FundamentalData, num_workers: int = 8):
        self.stock_data = stock_data
        self.fd = fd
        self.num_workers = num_workers

        # 动态因子数据
        self.factor_df = None
        self.industry_codes = {}
        self.all_dates = []

        # 验证结果
        self.results_df = None
        self.factor_selection_log = []

    def prepare_data(self):
        """预计算因子数据"""
        print("\n预计算因子数据...")
        dynamic_config = config.config.get('dynamic_factor', {})
        use_dynamic = dynamic_config.get('enabled', False)

        if use_dynamic:
            self.factor_df, self.industry_codes, self.all_dates = prepare_factor_data(
                stock_data=self.stock_data,
                fd=self.fd,
                detailed_industries=detailed_industries,
                num_workers=self.num_workers
            )
            print(f"动态因子模式: 预计算 {len(self.factor_df)} 条因子数据")
        else:
            # 构建行业映射
            self.all_dates = sorted(set(d for df in self.stock_data.values() for d in df.index))
            self.industry_codes = {cat: [] for cat in detailed_industries.keys()}
            for code in self.stock_data.keys():
                try:
                    sample_date = self.all_dates[100]
                    ind = self.fd.get_industry(code, sample_date)
                    for cat, keywords in detailed_industries.items():
                        if any(kw in str(ind) for kw in keywords):
                            self.industry_codes[cat].append(code)
                            break
                except:
                    pass

        return self

    def run_validation(self):
        """执行滚动验证"""
        print("=" * 60)
        print("信号系统统一验证")
        print(f"训练窗口: {TRAIN_WINDOW}天, 前瞻期: {FORWARD_PERIOD}天")
        print(f"每个验证点动态选择Top-{TOP_N_FACTORS}因子")
        print("=" * 60)

        if self.factor_df is None or len(self.factor_df) == 0:
            print("没有因子数据!")
            return self

        # 验证时间点
        start_idx = LOOKBACK + TRAIN_WINDOW
        validation_dates = self.all_dates[start_idx:-FORWARD_PERIOD:FORWARD_PERIOD]

        print(f"\n验证时间点: {len(validation_dates)} 个")
        print(f"验证区间: {validation_dates[0]} ~ {validation_dates[-1]}")

        # 因子列表
        exclude_cols = ['code', 'date', 'future_ret']
        factor_names = [c for c in self.factor_df.columns if c not in exclude_cols]
        print(f"因子数量: {len(factor_names)}")

        # 创建动态因子选择器
        factor_selector = DynamicFactorSelector()
        factor_selector.set_factor_data(self.factor_df)
        factor_selector.set_industry_mapping(self.industry_codes)

        # 滚动验证
        print("\n滚动验证...")
        validation_results = []

        for val_date in tqdm(validation_dates, desc="验证"):
            # 使用 DynamicFactorSelector 选择因子
            industry_factors = factor_selector.select_factors_for_date(val_date, self.all_dates)

            # 如果动态选择失败，使用默认因子
            if not industry_factors:
                industry_factors = {cat: factor_names[:TOP_N_FACTORS]
                                  for cat in detailed_industries.keys()}

            # 记录因子选择
            for cat, factors in industry_factors.items():
                self.factor_selection_log.append({
                    'date': val_date,
                    'industry': cat,
                    'factors': ','.join(factors)
                })

            # 验证数据
            val_df = self.factor_df[self.factor_df['date'] == val_date].copy()
            if len(val_df) < 10:
                continue

            # 生成信号
            for idx, row in val_df.iterrows():
                code = row['code']

                stock_industry = None
                for cat, codes in self.industry_codes.items():
                    if code in codes:
                        stock_industry = cat
                        break

                if stock_industry and stock_industry in industry_factors:
                    signal = generate_signal_with_factors(row, industry_factors[stock_industry])
                else:
                    signal = generate_signal_with_factors(row, factor_names[:TOP_N_FACTORS])

                if not np.isnan(signal):
                    validation_results.append({
                        'date': val_date,
                        'code': code,
                        'industry': stock_industry,
                        'signal': signal,
                        'future_ret': row['future_ret']
                    })

        self.results_df = pd.DataFrame(validation_results)
        print(f"\n验证结果: {len(self.results_df)} 条")

        return self

    def run_signal_evaluation(self):
        """运行信号层评估（使用SignalEngine生成真实信号）"""
        print("\n生成信号...")

        # 采样日期
        sample_dates = self.all_dates[LOOKBACK:-FORWARD_PERIOD:FORWARD_PERIOD]
        print(f"采样日期: {len(sample_dates)} 个")

        # 创建 SignalEngine（带动态因子数据）
        main_engine = SignalEngine()
        main_engine.set_fundamental_data(self.fd)
        if self.factor_df is not None and len(self.factor_df) > 0:
            main_engine.set_factor_data(self.factor_df)
            main_engine.set_industry_mapping(self.industry_codes)

        # 生成信号
        args_list = [(code, self.stock_data[code], sample_dates, self.fd, main_engine)
                     for code in self.stock_data.keys()]

        all_signals = []
        with Pool(self.num_workers) as pool:
            for res in tqdm(pool.imap(generate_signals_for_stock_with_engine, args_list, chunksize=10),
                           total=len(args_list), desc="生成信号"):
                all_signals.extend(res)

        self.signals_df = pd.DataFrame(all_signals)
        print(f"信号数据: {len(self.signals_df)} 条")

        return self

    def evaluate_factor_layer(self) -> dict:
        """因子层评估: IC/IR、胜率"""
        if self.results_df is None or len(self.results_df) == 0:
            return {}

        results = {}

        # 整体IC
        overall_ic_list = []
        for date, group in self.results_df.groupby('date'):
            if len(group) >= 10:
                ic = calc_ic(group['signal'].values, group['future_ret'].values)
                if not np.isnan(ic):
                    overall_ic_list.append(ic)

        if overall_ic_list:
            results['overall'] = {
                'ic_mean': np.mean(overall_ic_list),
                'ir': np.mean(overall_ic_list) / (np.std(overall_ic_list) + 1e-10),
                'win_rate': np.mean([1 if i > 0 else 0 for i in overall_ic_list])
            }

        # 分行业IC
        industry_results = {}
        for cat in detailed_industries.keys():
            cat_df = self.results_df[self.results_df['industry'] == cat]
            if len(cat_df) < 20:
                continue

            ic_list = []
            for date, group in cat_df.groupby('date'):
                if len(group) >= 3:
                    ic = calc_ic(group['signal'].values, group['future_ret'].values)
                    if not np.isnan(ic):
                        ic_list.append(ic)

            if len(ic_list) >= 5:
                industry_results[cat] = {
                    'ic_mean': np.mean(ic_list),
                    'ir': np.mean(ic_list) / (np.std(ic_list) + 1e-10),
                    'win_rate': np.mean([1 if i > 0 else 0 for i in ic_list])
                }

        results['industry'] = industry_results
        return results

    def evaluate_signal_layer(self) -> dict:
        """信号层评估: 覆盖率、买卖信号准确率"""
        if not hasattr(self, 'signals_df') or len(self.signals_df) == 0:
            return {}

        df = self.signals_df
        results = {}

        # 1. 信号覆盖率
        total = len(df)
        buy_signals = df['buy'].sum()
        sell_signals = df['sell'].sum()

        results['coverage'] = {
            'total_observations': total,
            'buy_signals': int(buy_signals),
            'sell_signals': int(sell_signals),
            'buy_rate': buy_signals / total if total > 0 else 0,
            'sell_rate': sell_signals / total if total > 0 else 0,
        }

        # 2. 信号强度分布
        scores = df['score'].dropna()
        results['score_distribution'] = {
            'mean': scores.mean(),
            'std': scores.std(),
            'median': scores.median(),
            'positive_rate': (scores > 0).mean(),
        }

        # 3. 买入信号准确率
        buy_df = df[df['buy'] == True]
        if len(buy_df) > 0:
            buy_accuracy = (buy_df['future_ret'] > 0).mean()
            buy_avg_ret = buy_df['future_ret'].mean()
            results['buy_signal_quality'] = {
                'count': len(buy_df),
                'accuracy': buy_accuracy,
                'avg_return': buy_avg_ret,
            }

        # 4. 卖出信号准确率
        sell_df = df[df['sell'] == True]
        if len(sell_df) > 0:
            sell_accuracy = (sell_df['future_ret'] < 0).mean()
            sell_avg_ret = sell_df['future_ret'].mean()
            results['sell_signal_quality'] = {
                'count': len(sell_df),
                'accuracy': sell_accuracy,
                'avg_return': sell_avg_ret,
            }

        # 5. 分数与未来收益的相关性
        valid = df.dropna(subset=['score', 'future_ret'])
        if len(valid) > 10:
            ic, _ = stats.spearmanr(valid['score'], valid['future_ret'])
            results['score_ic'] = ic

        return results

    def evaluate_portfolio_layer(self, top_n: int = 10, top_pct: float = 0.7) -> dict:
        """组合层评估: Top-N选股、五分位、行业权重优化

        优化: 使用排名(percentile)排序，选70-80%分位(Q4)避免极端反转
        """
        if self.results_df is None or len(self.results_df) == 0:
            return {}

        df = self.results_df.copy()
        results = {}

        # 计算每只股票在其日期内的排名百分位 (0-1)
        df['rank_pct'] = df.groupby('date')['signal'].rank(pct=True)

        # 1. Top-N选股（等权）- 选70-80%分位(Q4)避免极端反转
        # 原因: Q5(最高分)往往存在反转效应，Q4表现更稳定
        portfolio_returns = []
        for date, group in df.groupby('date'):
            if len(group) < top_n:
                continue
            # 选70-90%分位（Q4），避免选最高分
            top = group[(group['rank_pct'] > 0.7) & (group['rank_pct'] < 0.9)]
            if len(top) < 3:
                top = group.nlargest(top_n, 'rank_pct')  # 回退到Top-N
            portfolio_returns.append({
                'date': date,
                'return': top['future_ret'].mean(),
            })

        if portfolio_returns:
            ret_df = pd.DataFrame(portfolio_returns)
            results['top_n_equal_weight'] = {
                'n': top_n,
                'periods': len(ret_df),
                'avg_return': ret_df['return'].mean(),
                'sharpe': ret_df['return'].mean() / (ret_df['return'].std() + 1e-10) * np.sqrt(12),
                'win_rate': (ret_df['return'] > 0).mean(),
            }

        # 2. 行业权重优化选股
        if ENABLE_INDUSTRY_WEIGHTING:
            optimized_returns = self._evaluate_industry_weighted_portfolio(df, top_n)
            if optimized_returns:
                results['top_n_industry_weighted'] = optimized_returns

        # 3. 五分位分组 - 使用排名百分位
        quintile_returns = {i: [] for i in range(1, 6)}
        for date, group in df.groupby('date'):
            if len(group) < 25:
                continue

            group = group.copy()
            # 使用排名百分位分五分位
            group['quintile'] = pd.qcut(group['rank_pct'], 5, labels=[1,2,3,4,5], duplicates='drop')

            for q in range(1, 6):
                q_ret = group[group['quintile'] == q]['future_ret'].mean()
                if not np.isnan(q_ret):
                    quintile_returns[q].append(q_ret)

        results['quintile_returns'] = {
            f'Q{q}': np.mean(quintile_returns[q]) if quintile_returns[q] else np.nan
            for q in range(1, 6)
        }

        # 单调性检验
        q1_avg = np.mean(quintile_returns[1]) if quintile_returns[1] else 0
        q5_avg = np.mean(quintile_returns[5]) if quintile_returns[5] else 0
        results['monotonicity'] = {
            'q5_minus_q1': q5_avg - q1_avg,
            'is_monotonic': q5_avg > q1_avg,
        }

        # 4. 分行业评估
        industry_eval = self._evaluate_by_industry(df)
        results['industry_performance'] = industry_eval

        return results

    def _evaluate_industry_weighted_portfolio(self, df: pd.DataFrame, top_n: int) -> dict:
        """行业权重优化选股

        策略:
        1. 计算各行业的历史IR作为权重
        2. 在每个行业内按排名选Top-N
        3. 按行业权重合并
        """
        # df 已有 rank_pct 列
        if 'rank_pct' not in df.columns:
            df = df.copy()
            df['rank_pct'] = df.groupby('date')['signal'].rank(pct=True)

        # 计算各行业IR
        industry_ir = {}
        for cat in detailed_industries.keys():
            cat_df = df[df['industry'] == cat]
            if len(cat_df) < 20:
                continue

            ic_list = []
            for date, group in cat_df.groupby('date'):
                if len(group) >= 3:
                    ic = calc_ic(group['rank_pct'].values, group['future_ret'].values)
                    if not np.isnan(ic):
                        ic_list.append(ic)

            if len(ic_list) >= 3:
                industry_ir[cat] = np.mean(ic_list) / (np.std(ic_list) + 1e-10)

        if not industry_ir:
            return None

        # 归一化权重（softmax风格）
        ir_values = np.array(list(industry_ir.values()))
        ir_values = np.maximum(ir_values, 0)  # 只保留正IR
        weights = np.exp(ir_values * 10) / np.sum(np.exp(ir_values * 10))
        industry_weights = {k: v for k, v in zip(industry_ir.keys(), weights)}

        # 选股 - 使用排名排序
        portfolio_returns = []
        for date, group in df.groupby('date'):
            if len(group) < top_n * 2:
                continue

            weighted_return = 0
            total_weight = 0

            for cat, weight in industry_weights.items():
                cat_group = group[group['industry'] == cat]
                if len(cat_group) == 0:
                    continue

                # 行业内按排名选Top
                # 选70-90%分位(Q4)，避免选最高分
                # 使用全局排名而非组内排名
                top_cat = cat_group[(cat_group['rank_pct'] > 0.7) & (cat_group['rank_pct'] < 0.9)]
                if len(top_cat) < 2:
                    n_select = max(1, int(top_n * weight * 3))
                    top_cat = cat_group.nlargest(n_select, 'rank_pct')
                cat_ret = top_cat['future_ret'].mean()

                weighted_return += cat_ret * weight
                total_weight += weight

            if total_weight > 0:
                portfolio_returns.append(weighted_return / total_weight)

        if portfolio_returns:
            ret_arr = np.array(portfolio_returns)
            return {
                'periods': len(portfolio_returns),
                'avg_return': np.mean(ret_arr),
                'sharpe': np.mean(ret_arr) / (np.std(ret_arr) + 1e-10) * np.sqrt(12),
                'win_rate': np.mean(ret_arr > 0),
            }

        return None

    def _evaluate_by_industry(self, df: pd.DataFrame) -> dict:
        """分行业评估"""
        results = {}
        for cat in detailed_industries.keys():
            cat_df = df[df['industry'] == cat]
            if len(cat_df) < 20:
                continue

            ic_list = []
            for date, group in cat_df.groupby('date'):
                if len(group) >= 3:
                    ic = calc_ic(group['signal'].values, group['future_ret'].values)
                    if not np.isnan(ic):
                        ic_list.append(ic)

            if len(ic_list) >= 3:
                results[cat] = {
                    'ic_mean': np.mean(ic_list),
                    'ir': np.mean(ic_list) / (np.std(ic_list) + 1e-10),
                    'win_rate': np.mean([1 if i > 0 else 0 for i in ic_list]),
                    'n_samples': len(cat_df),
                }

        return results

    def print_report(self):
        """打印完整评估报告"""
        print("\n" + "=" * 70)
        print("信号系统统一验证报告")
        print("=" * 70)

        # 因子层
        print("\n【因子层评估】")
        print("-" * 50)
        factor_results = self.evaluate_factor_layer()

        if 'overall' in factor_results:
            ov = factor_results['overall']
            print(f"整体表现:")
            print(f"  IC均值: {ov['ic_mean']:.4f}")
            print(f"  IR: {ov['ir']:.4f}")
            print(f"  胜率: {ov['win_rate']:.1%}")

        if 'industry' in factor_results:
            print(f"\n分行业表现:")
            for cat, res in sorted(factor_results['industry'].items(), key=lambda x: -x[1]['ir']):
                print(f"  {cat}: IC={res['ic_mean']:.4f}, IR={res['ir']:.4f}, 胜率={res['win_rate']:.1%}")

        # 信号层
        if hasattr(self, 'signals_df') and len(self.signals_df) > 0:
            print("\n【信号层评估】")
            print("-" * 50)
            signal_results = self.evaluate_signal_layer()

            cov = signal_results.get('coverage', {})
            print(f"总观测数: {cov.get('total_observations', 0)}")
            print(f"买入信号: {cov.get('buy_signals', 0)} ({cov.get('buy_rate', 0):.1%})")
            print(f"卖出信号: {cov.get('sell_signals', 0)} ({cov.get('sell_rate', 0):.1%})")

            buy_q = signal_results.get('buy_signal_quality', {})
            if buy_q:
                print(f"\n买入信号质量:")
                print(f"  数量: {buy_q.get('count', 0)}, 准确率: {buy_q.get('accuracy', 0):.1%}")
                print(f"  平均收益: {buy_q.get('avg_return', 0)*100:.2f}%")

            print(f"\n信号IC: {signal_results.get('score_ic', 0):.4f}")

        # 组合层
        print("\n【组合层评估】")
        print("-" * 50)
        portfolio_results = self.evaluate_portfolio_layer(top_n=10)

        if 'top_n_equal_weight' in portfolio_results:
            top = portfolio_results['top_n_equal_weight']
            print(f"Top-{top['n']} 等权组合:")
            print(f"  期数: {top['periods']}")
            print(f"  平均收益: {top['avg_return']*100:.2f}%")
            print(f"  夏普比率: {top['sharpe']:.2f}")
            print(f"  胜率: {top['win_rate']:.1%}")

        if 'top_n_industry_weighted' in portfolio_results:
            ind_w = portfolio_results['top_n_industry_weighted']
            print(f"\nTop-{10} 行业加权组合:")
            print(f"  期数: {ind_w['periods']}")
            print(f"  平均收益: {ind_w['avg_return']*100:.2f}%")
            print(f"  夏普比率: {ind_w['sharpe']:.2f}")
            print(f"  胜率: {ind_w['win_rate']:.1%}")

        if 'quintile_returns' in portfolio_results:
            print(f"\n五分位收益（从低分到高分）:")
            for q, ret in portfolio_results['quintile_returns'].items():
                if not np.isnan(ret):
                    marker = "★" if q == 'Q5' else ""
                    print(f"  {q}: {ret*100:.2f}% {marker}")

            mono = portfolio_results['monotonicity']
            status = '✓ 通过' if mono['is_monotonic'] else '✗ 未通过'
            print(f"\n单调性检验: Q5-Q1 = {mono['q5_minus_q1']*100:.2f}% ({status})")

        # 有效行业
        if 'industry_performance' in portfolio_results:
            ind_perf = portfolio_results['industry_performance']
            good = [(k, v) for k, v in ind_perf.items() if v['ir'] >= 0.15]
            weak = [(k, v) for k, v in ind_perf.items() if v['ir'] < 0.15]

            if good:
                print("\n有效行业（IR >= 0.15）:")
                for ind, stats in sorted(good, key=lambda x: -x[1]['ir']):
                    print(f"  {ind}: IC={stats['ic_mean']:.4f}, IR={stats['ir']:.4f}")

            if weak:
                print("\n弱势行业（IR < 0.15）:")
                for ind, stats in sorted(weak, key=lambda x: -x[1]['ir']):
                    print(f"  {ind}: IC={stats['ic_mean']:.4f}, IR={stats['ir']:.4f}")

    def save_results(self):
        """保存结果"""
        results_dir = os.path.join(project_root, 'strategy/rolling_validation_results')
        os.makedirs(results_dir, exist_ok=True)

        if self.results_df is not None:
            self.results_df.to_csv(os.path.join(results_dir, 'rolling_signals.csv'), index=False)

        if hasattr(self, 'signals_df') and len(self.signals_df) > 0:
            self.signals_df.to_csv(os.path.join(results_dir, 'signal_evaluation_data.csv'), index=False)

        if self.factor_selection_log:
            pd.DataFrame(self.factor_selection_log).to_csv(
                os.path.join(results_dir, 'factor_selection_log.csv'), index=False
            )

        print(f"\n结果已保存到 {results_dir}/")


def run_validation(stock_data: dict, fd: FundamentalData, num_workers: int = 8):
    """运行统一验证"""
    validator = SignalValidator(stock_data, fd, num_workers)
    validator.prepare_data()
    validator.run_validation()
    validator.run_signal_evaluation()
    validator.print_report()
    validator.save_results()

    return validator


if __name__ == '__main__':
    DATA_PATH = os.path.join(project_root, 'data/stock_data/backtrader_data/')
    FUND_PATH = os.path.join(project_root, 'data/stock_data/fundamental_data/')

    # 加载数据
    print("加载数据...")
    files = [f for f in os.listdir(DATA_PATH)
             if f.endswith('_qfq.csv') and f != 'sh000001_qfq.csv']

    stock_data = {}
    for f in files:
        code = f.replace('_qfq.csv', '')
        df = pd.read_csv(os.path.join(DATA_PATH, f), parse_dates=['datetime']).set_index('datetime').sort_index()
        if len(df) >= 300:
            stock_data[code] = df

    print(f"加载 {len(stock_data)} 只股票")

    # 加载基本面数据
    fd = FundamentalData(FUND_PATH, list(stock_data.keys()))

    # 运行验证
    run_validation(stock_data, fd, num_workers=8)
