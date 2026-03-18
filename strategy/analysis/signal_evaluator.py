# analysis/signal_evaluator.py
"""
信号系统分层评估模块

评估层次:
1. 因子层 - IC/IR（由 rolling_signal_validator.py 负责）
2. 信号层 - 信号质量、覆盖率、准确率
3. 组合层 - Top-N选股收益、持仓分析
4. 执行层 - 交易成本、换手率影响（由回测负责）

本模块专注于 信号层 + 组合层 的评估
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
from core.signal_engine import SignalEngine, prepare_factor_data
from core.signal_store import SignalStore
from core.fundamental import FundamentalData

config = load_config(os.path.join(project_root, 'strategy/config/factor_config.yaml'))
detailed_industries = config.config.get('detailed_industries', {})


def generate_signals_for_stock_with_engine(args):
    """为单只股票生成信号（使用已创建的 SignalEngine）"""
    code, df, sample_dates, fd, signal_engine = args
    stock_dates = sorted(df.index.tolist())

    results = []
    date_to_idx = {d: i for i, d in enumerate(stock_dates)}

    for sample_date in sample_dates:
        valid_dates = [d for d in stock_dates if d <= sample_date]
        if len(valid_dates) < 120:
            continue

        eval_date = valid_dates[-1]
        idx = date_to_idx.get(eval_date)
        if idx is None or idx < 120:
            continue

        history = df.iloc[:idx+1].iloc[-120:]
        if len(history) < 60:
            continue

        history_copy = history.copy()
        history_copy['datetime'] = history_copy.index

        signal_store = SignalStore()
        signal_engine.generate_at_indices(code, history_copy, [len(history_copy)-1], signal_store)

        sig = signal_store.get(code, eval_date.date() if hasattr(eval_date, 'date') else eval_date)
        if sig and idx + 20 < len(df):
            future_price = df.iloc[idx + 20]['close']
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


def generate_signals_for_stock(args):
    """为单只股票生成信号"""
    code, df, sample_dates, fd, signal_engine = args
    stock_dates = sorted(df.index.tolist())

    if signal_engine is None:
        signal_engine = SignalEngine()
        signal_engine.set_fundamental_data(fd)

    results = []
    date_to_idx = {d: i for i, d in enumerate(stock_dates)}

    for sample_date in sample_dates:
        valid_dates = [d for d in stock_dates if d <= sample_date]
        if len(valid_dates) < 120:
            continue

        eval_date = valid_dates[-1]
        idx = date_to_idx.get(eval_date)
        if idx is None or idx < 120:
            continue

        history = df.iloc[:idx+1].iloc[-120:]
        if len(history) < 60:
            continue

        history_copy = history.copy()
        history_copy['datetime'] = history_copy.index

        signal_store = SignalStore()
        signal_engine.generate_at_indices(code, history_copy, [len(history_copy)-1], signal_store)

        sig = signal_store.get(code, eval_date.date() if hasattr(eval_date, 'date') else eval_date)
        if sig and idx + 20 < len(df):
            future_price = df.iloc[idx + 20]['close']
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


class SignalEvaluator:
    """信号系统评估器"""

    def __init__(self, signals_df: pd.DataFrame):
        """
        Args:
            signals_df: 包含 code, date, score, factor_value, buy, sell, future_ret 的DataFrame
        """
        self.df = signals_df

    def evaluate_signal_layer(self) -> dict:
        """
        信号层评估
        - 信号覆盖率
        - 信号强度分布
        - 买卖信号准确率
        """
        results = {}

        # 1. 信号覆盖率
        total = len(self.df)
        buy_signals = self.df['buy'].sum()
        sell_signals = self.df['sell'].sum()

        results['coverage'] = {
            'total_observations': total,
            'buy_signals': int(buy_signals),
            'sell_signals': int(sell_signals),
            'buy_rate': buy_signals / total if total > 0 else 0,
            'sell_rate': sell_signals / total if total > 0 else 0,
        }

        # 2. 信号强度分布
        scores = self.df['score'].dropna()
        results['score_distribution'] = {
            'mean': scores.mean(),
            'std': scores.std(),
            'median': scores.median(),
            'q25': scores.quantile(0.25),
            'q75': scores.quantile(0.75),
            'positive_rate': (scores > 0).mean(),
        }

        # 3. 买入信号准确率
        buy_df = self.df[self.df['buy'] == True]
        if len(buy_df) > 0:
            buy_accuracy = (buy_df['future_ret'] > 0).mean()
            buy_avg_ret = buy_df['future_ret'].mean()
            results['buy_signal_quality'] = {
                'count': len(buy_df),
                'accuracy': buy_accuracy,  # 正收益占比
                'avg_return': buy_avg_ret,
                'median_return': buy_df['future_ret'].median(),
            }
        else:
            results['buy_signal_quality'] = {'count': 0, 'accuracy': 0, 'avg_return': 0}

        # 4. 卖出信号准确率（卖出后下跌才算对）
        sell_df = self.df[self.df['sell'] == True]
        if len(sell_df) > 0:
            sell_accuracy = (sell_df['future_ret'] < 0).mean()  # 卖出后下跌
            sell_avg_ret = sell_df['future_ret'].mean()
            results['sell_signal_quality'] = {
                'count': len(sell_df),
                'accuracy': sell_accuracy,
                'avg_return': sell_avg_ret,  # 应该为负才好
            }
        else:
            results['sell_signal_quality'] = {'count': 0, 'accuracy': 0, 'avg_return': 0}

        # 5. 分数与未来收益的相关性
        valid = self.df.dropna(subset=['score', 'future_ret'])
        if len(valid) > 10:
            ic, _ = stats.spearmanr(valid['score'], valid['future_ret'])
            results['score_ic'] = ic
        else:
            results['score_ic'] = np.nan

        return results

    def evaluate_portfolio_layer(self, top_n: int = 10) -> dict:
        """
        组合层评估
        - Top-N选股收益
        - 分组收益（五分位）
        - 多空收益
        """
        results = {}

        # 1. 按日期分组，每期选Top-N
        portfolio_returns = []
        bottom_returns = []

        for date, group in self.df.groupby('date'):
            if len(group) < top_n * 2:
                continue

            sorted_df = group.sort_values('score', ascending=False)
            top = sorted_df.head(top_n)
            bottom = sorted_df.tail(top_n)

            portfolio_returns.append({
                'date': date,
                'top_return': top['future_ret'].mean(),
                'bottom_return': bottom['future_ret'].mean(),
                'long_short': top['future_ret'].mean() - bottom['future_ret'].mean(),
                'n_stocks': len(group),
            })

        if portfolio_returns:
            ret_df = pd.DataFrame(portfolio_returns)

            results['top_n_portfolio'] = {
                'n': top_n,
                'periods': len(ret_df),
                'avg_return': ret_df['top_return'].mean(),
                'std_return': ret_df['top_return'].std(),
                'sharpe': ret_df['top_return'].mean() / (ret_df['top_return'].std() + 1e-10) * np.sqrt(12),
                'win_rate': (ret_df['top_return'] > 0).mean(),
                'max_return': ret_df['top_return'].max(),
                'min_return': ret_df['top_return'].min(),
            }

            results['bottom_n_portfolio'] = {
                'n': top_n,
                'avg_return': ret_df['bottom_return'].mean(),
                'win_rate': (ret_df['bottom_return'] > 0).mean(),
            }

            results['long_short'] = {
                'avg_return': ret_df['long_short'].mean(),
                'std_return': ret_df['long_short'].std(),
                'sharpe': ret_df['long_short'].mean() / (ret_df['long_short'].std() + 1e-10) * np.sqrt(12),
                'win_rate': (ret_df['long_short'] > 0).mean(),
            }

        # 2. 五分位分组
        quintile_returns = {i: [] for i in range(1, 6)}

        for date, group in self.df.groupby('date'):
            if len(group) < 25:
                continue

            group = group.copy()
            group['quintile'] = pd.qcut(group['score'], 5, labels=[1,2,3,4,5], duplicates='drop')

            for q in range(1, 6):
                q_ret = group[group['quintile'] == q]['future_ret'].mean()
                if not np.isnan(q_ret):
                    quintile_returns[q].append(q_ret)

        results['quintile_returns'] = {
            f'Q{q}': np.mean(quintile_returns[q]) if quintile_returns[q] else np.nan
            for q in range(1, 6)
        }

        # 3. 单调性检验（Q5应该 > Q1）
        q1_avg = np.mean(quintile_returns[1]) if quintile_returns[1] else 0
        q5_avg = np.mean(quintile_returns[5]) if quintile_returns[5] else 0
        results['monotonicity'] = {
            'q5_minus_q1': q5_avg - q1_avg,
            'is_monotonic': q5_avg > q1_avg,
        }

        return results

    def evaluate_by_industry(self, industry_map: dict) -> dict:
        """分行业评估"""
        self.df['industry'] = self.df['code'].map(industry_map)

        results = {}
        for industry in self.df['industry'].dropna().unique():
            ind_df = self.df[self.df['industry'] == industry]
            if len(ind_df) < 50:
                continue

            # 信号IC
            ic_list = []
            for date, group in ind_df.groupby('date'):
                if len(group) >= 5:
                    valid = group.dropna(subset=['score', 'future_ret'])
                    if len(valid) >= 3:
                        ic, _ = stats.spearmanr(valid['score'], valid['future_ret'])
                        if not np.isnan(ic):
                            ic_list.append(ic)

            if ic_list:
                results[industry] = {
                    'ic_mean': np.mean(ic_list),
                    'ir': np.mean(ic_list) / (np.std(ic_list) + 1e-10),
                    'win_rate': np.mean([1 if i > 0 else 0 for i in ic_list]),
                    'n_signals': len(ind_df),
                }

        return results

    def print_report(self, industry_map: dict = None):
        """打印评估报告（A股版本：不含做空、杠杆相关指标）"""
        print("=" * 70)
        print("信号系统分层评估报告（A股）")
        print("=" * 70)

        # 信号层
        signal_eval = self.evaluate_signal_layer()

        print("\n【信号层评估】")
        print("-" * 50)
        cov = signal_eval['coverage']
        print(f"总观测数: {cov['total_observations']}")
        print(f"买入信号: {cov['buy_signals']} ({cov['buy_rate']:.1%})")
        print(f"卖出信号: {cov['sell_signals']} ({cov['sell_rate']:.1%})")

        print(f"\n信号分数分布:")
        dist = signal_eval['score_distribution']
        print(f"  均值: {dist['mean']:.4f}, 标准差: {dist['std']:.4f}")
        print(f"  中位数: {dist['median']:.4f}, Q25-Q75: [{dist['q25']:.4f}, {dist['q75']:.4f}]")
        print(f"  正分数占比: {dist['positive_rate']:.1%}")

        buy_q = signal_eval['buy_signal_quality']
        print(f"\n买入信号质量:")
        print(f"  数量: {buy_q['count']}, 准确率: {buy_q['accuracy']:.1%}")
        print(f"  平均收益: {buy_q['avg_return']*100:.2f}%")

        print(f"\n信号IC: {signal_eval['score_ic']:.4f}")

        # 组合层
        portfolio_eval = self.evaluate_portfolio_layer(top_n=10)

        print("\n【组合层评估】")
        print("-" * 50)

        if 'top_n_portfolio' in portfolio_eval:
            top = portfolio_eval['top_n_portfolio']
            print(f"Top-{top['n']} 组合（纯多头）:")
            print(f"  期数: {top['periods']}")
            print(f"  平均收益: {top['avg_return']*100:.2f}%")
            print(f"  夏普比率: {top['sharpe']:.2f}")
            print(f"  胜率: {top['win_rate']:.1%}")
            print(f"  最大单期收益: {top['max_return']*100:.2f}%")
            print(f"  最大单期亏损: {top['min_return']*100:.2f}%")

            # A股不能做空，Bottom组合仅作为对照参考
            bottom = portfolio_eval['bottom_n_portfolio']
            print(f"\nBottom-{bottom['n']} 对照组（仅参考，A股不可做空）:")
            print(f"  平均收益: {bottom['avg_return']*100:.2f}%")

        print(f"\n五分位收益（从低分到高分）:")
        for q, ret in portfolio_eval['quintile_returns'].items():
            if not np.isnan(ret):
                marker = "★" if q == 'Q5' else ""
                print(f"  {q}: {ret*100:.2f}% {marker}")

        mono = portfolio_eval['monotonicity']
        status = '✓ 通过' if mono['is_monotonic'] else '✗ 未通过'
        print(f"\n单调性检验: Q5-Q1 = {mono['q5_minus_q1']*100:.2f}% ({status})")

        # 分行业
        if industry_map:
            industry_eval = self.evaluate_by_industry(industry_map)

            print("\n【分行业评估】")
            print("-" * 50)

            # 分类显示
            good_industries = []
            weak_industries = []

            for ind, stats in sorted(industry_eval.items(), key=lambda x: -x[1]['ir']):
                if stats['ir'] >= 0.15:
                    good_industries.append((ind, stats))
                else:
                    weak_industries.append((ind, stats))

            if good_industries:
                print("有效行业（IR >= 0.15）:")
                for ind, stats in good_industries:
                    print(f"  {ind}: IC={stats['ic_mean']:.4f}, IR={stats['ir']:.4f}, 胜率={stats['win_rate']:.1%}")

            if weak_industries:
                print("\n弱势行业（IR < 0.15，建议减配或不做）:")
                for ind, stats in weak_industries:
                    print(f"  {ind}: IC={stats['ic_mean']:.4f}, IR={stats['ir']:.4f}, 胜率={stats['win_rate']:.1%}")


def run_evaluation(stock_data: dict, fd: FundamentalData, num_workers: int = 8):
    """运行评估"""
    print("=" * 60)
    print("信号系统评估")
    print("=" * 60)

    # 检查动态因子配置
    dynamic_config = config.config.get('dynamic_factor', {})
    use_dynamic = dynamic_config.get('enabled', False)

    # 预计算因子数据（用于动态因子选择）
    factor_data = None
    industry_codes = {}
    all_dates = []

    if use_dynamic:
        print("动态因子模式已启用，预计算因子数据...")
        factor_data, industry_codes, all_dates = prepare_factor_data(
            stock_data=stock_data,
            fd=fd,
            detailed_industries=detailed_industries,
            num_workers=num_workers
        )

    # 生成采样日期
    all_dates_set = set()
    for df in stock_data.values():
        all_dates_set.update(df.index.tolist())

    common_dates = sorted(all_dates_set)[120:-20]
    sample_dates = common_dates[::20]  # 每20天采样一次，与调仓周期一致

    print(f"采样日期: {len(sample_dates)} 个")

    # 创建带有因子数据的 SignalEngine（用于动态因子选择）
    # 注意：因子数据需要在每个 worker 进程中可用
    # 由于多进程无法传递大型 DataFrame，这里使用单进程生成信号
    print("\n生成信号...")

    # 创建主进程 SignalEngine（包含因子数据）
    main_engine = SignalEngine()
    main_engine.set_fundamental_data(fd)
    if use_dynamic and factor_data is not None and len(factor_data) > 0:
        main_engine.set_factor_data(factor_data)
        main_engine.set_industry_mapping(industry_codes)

    # 单进程生成信号（避免多进程传递 factor_data）
    args_list = [(code, stock_data[code], sample_dates, fd, main_engine) for code in stock_data.keys()]

    all_signals = []
    from tqdm import tqdm
    for res in tqdm(map(generate_signals_for_stock_with_engine, args_list),
                   total=len(args_list), desc="计算信号"):
        all_signals.extend(res)

    all_signals = []
    with Pool(num_workers) as pool:
        for res in tqdm(pool.imap(generate_signals_for_stock, args_list, chunksize=10),
                       total=len(args_list), desc="计算信号"):
            all_signals.extend(res)

    signals_df = pd.DataFrame(all_signals)
    print(f"信号数据: {len(signals_df)} 条")

    if len(signals_df) == 0:
        print("没有信号数据!")
        return

    # 行业映射
    industry_map = {}
    for code in stock_data.keys():
        try:
            ind = fd.get_industry(code, sample_dates[0])
            for cat, keywords in detailed_industries.items():
                if any(kw in str(ind) for kw in keywords):
                    industry_map[code] = cat
                    break
        except:
            pass

    # 评估
    evaluator = SignalEvaluator(signals_df)
    evaluator.print_report(industry_map)

    # 保存结果
    results_dir = os.path.join(project_root, 'strategy/rolling_validation_results')
    os.makedirs(results_dir, exist_ok=True)
    signals_df.to_csv(os.path.join(results_dir, 'signal_evaluation_data.csv'), index=False)
    print(f"\n数据已保存到 {results_dir}/signal_evaluation_data.csv")

    return evaluator


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
        if len(df) >= 200:
            stock_data[code] = df

    print(f"加载 {len(stock_data)} 只股票")

    # 加载基本面数据
    fd = FundamentalData(FUND_PATH, list(stock_data.keys()))

    # 运行评估
    run_evaluation(stock_data, fd, num_workers=8)
