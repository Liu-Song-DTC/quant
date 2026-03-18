# analysis/rolling_signal_validator.py
"""
滚动信号验证模块 (Walk-Forward Validation)

这是正确的样本外验证方法:
- 每个验证时点只使用该时点之前的数据选择因子
- 使用 signal_engine.DynamicFactorSelector 动态选择因子

参数从 factor_config.yaml 读取:
- dynamic_factor.train_window: 训练窗口 (默认250天)
- dynamic_factor.forward_period: 前瞻期 (默认20天)
- dynamic_factor.top_n_factors: 选择Top-N因子 (默认3)
- dynamic_factor.min_train_samples: 最少训练样本数 (默认50)
- dynamic_factor.min_ic_dates: 最少有效IC天数 (默认5)
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
from core.signal_engine import DynamicFactorSelector

config = load_config(os.path.join(project_root, 'strategy/config/factor_config.yaml'))
detailed_industries = config.config.get('detailed_industries', {})

# 配置参数 - 从配置文件读取
dynamic_config = config.config.get('dynamic_factor', {})
TRAIN_WINDOW = dynamic_config.get('train_window', 250)
FORWARD_PERIOD = dynamic_config.get('forward_period', 20)
LOOKBACK = config.config.get('industry_factor_config', {}).get('lookback_days', 120)
MIN_TRAIN_SAMPLES = dynamic_config.get('min_train_samples', 50)
TOP_N_FACTORS = dynamic_config.get('top_n_factors', 3)


def calc_ic(factor_values, returns):
    """计算IC (Spearman秩相关)"""
    valid_mask = ~(np.isnan(factor_values) | np.isnan(returns))
    if valid_mask.sum() < 5:
        return np.nan
    ic, _ = stats.spearmanr(factor_values[valid_mask], returns[valid_mask])
    return ic


def process_stock_factors(args):
    """计算单只股票所有日期的因子值"""
    code, df, all_sample_dates, fd = args
    stock_dates = sorted(df.index.tolist())

    results = []
    for sample_date in all_sample_dates:
        valid_dates = [d for d in stock_dates if d <= sample_date]
        if len(valid_dates) < LOOKBACK:
            continue

        eval_date = valid_dates[-1]
        idx = stock_dates.index(eval_date)

        if idx < LOOKBACK:
            continue

        history = df.iloc[:idx+1].iloc[-LOOKBACK:]
        if len(history) < 60:
            continue

        # 计算因子
        factors = calc_all_factors_for_validation(
            history['close'].values,
            history['high'].values if 'high' in history.columns else history['close'].values,
            history['low'].values if 'low' in history.columns else history['close'].values,
            history['volume'].values if 'volume' in history.columns else np.ones(len(history)),
            fundamental_data=fd,
            code=code,
            eval_date=eval_date
        )

        row = {'code': code, 'date': eval_date}
        for fn, vals in factors.items():
            if hasattr(vals, '__len__') and len(vals) > 0:
                val = vals[-1]
            else:
                val = vals
            if val is not None and not np.isnan(val):
                row[fn] = float(val)

        # 计算未来收益
        if idx + FORWARD_PERIOD < len(df):
            future_price = df.iloc[idx + FORWARD_PERIOD]['close']
            current_price = df.iloc[idx]['close']
            if current_price > 0:
                row['future_ret'] = (future_price - current_price) / current_price
                results.append(row)

    return results


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


def run_validation(stock_data: dict, fd: FundamentalData, num_workers: int = 8):
    """执行滚动验证"""
    print("=" * 60)
    print("滚动信号验证 (Walk-Forward)")
    print(f"训练窗口: {TRAIN_WINDOW}天, 前瞻期: {FORWARD_PERIOD}天")
    print(f"每个验证点动态选择Top-{TOP_N_FACTORS}因子")
    print("=" * 60)

    # 获取所有日期
    all_dates = set()
    for df in stock_data.values():
        all_dates.update(df.index.tolist())
    all_dates = sorted(all_dates)

    # 验证时间点
    start_idx = LOOKBACK + TRAIN_WINDOW
    validation_dates = all_dates[start_idx:-FORWARD_PERIOD:FORWARD_PERIOD]

    print(f"验证时间点: {len(validation_dates)} 个")
    print(f"验证区间: {validation_dates[0]} ~ {validation_dates[-1]}")

    # 预计算因子
    print("\n计算因子...")
    all_sample_dates = all_dates[LOOKBACK:-FORWARD_PERIOD:5]

    args_list = [(code, stock_data[code], all_sample_dates, fd) for code in stock_data.keys()]

    all_factor_data = []
    with Pool(num_workers) as pool:
        for res in tqdm(pool.imap(process_stock_factors, args_list, chunksize=10),
                       total=len(args_list), desc="计算因子"):
            all_factor_data.extend(res)

    factor_df = pd.DataFrame(all_factor_data)
    print(f"因子数据: {len(factor_df)} 条")

    if len(factor_df) == 0:
        print("没有因子数据!")
        return

    # 因子列表
    exclude_cols = ['code', 'date', 'future_ret']
    factor_names = [c for c in factor_df.columns if c not in exclude_cols]
    print(f"因子数量: {len(factor_names)}")

    # 行业映射
    industry_codes = {cat: [] for cat in detailed_industries.keys()}
    for code in stock_data.keys():
        try:
            ind = fd.get_industry(code, all_dates[100])
            for cat, keywords in detailed_industries.items():
                if any(kw in str(ind) for kw in keywords):
                    industry_codes[cat].append(code)
                    break
        except:
            pass

    # 创建动态因子选择器
    factor_selector = DynamicFactorSelector()
    factor_selector.set_factor_data(factor_df)
    factor_selector.set_industry_mapping(industry_codes)

    # 滚动验证
    print("\n滚动验证...")
    validation_results = []
    factor_selection_log = []

    for val_date in tqdm(validation_dates, desc="滚动验证"):
        # 使用 DynamicFactorSelector 选择因子
        industry_factors = factor_selector.select_factors_for_date(val_date, all_dates)

        # 如果动态选择失败，使用默认因子
        if not industry_factors:
            industry_factors = {cat: factor_names[:TOP_N_FACTORS]
                              for cat in detailed_industries.keys()}

        # 记录因子选择
        for cat, factors in industry_factors.items():
            factor_selection_log.append({
                'date': val_date,
                'industry': cat,
                'factors': ','.join(factors)
            })

        # 验证数据
        val_df = factor_df[factor_df['date'] == val_date].copy()
        if len(val_df) < 10:
            continue

        # 生成信号
        for idx, row in val_df.iterrows():
            code = row['code']

            stock_industry = None
            for cat, codes in industry_codes.items():
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

    results_df = pd.DataFrame(validation_results)
    print(f"\n验证结果: {len(results_df)} 条")

    # 评估
    print("\n" + "=" * 60)
    print("验证结果")
    print("=" * 60)

    # 整体IC
    overall_ic_list = []
    for date, group in results_df.groupby('date'):
        if len(group) >= 10:
            ic = calc_ic(group['signal'].values, group['future_ret'].values)
            if not np.isnan(ic):
                overall_ic_list.append(ic)

    if overall_ic_list:
        print(f"\n整体表现:")
        print(f"  IC均值: {np.mean(overall_ic_list):.4f}")
        print(f"  IR: {np.mean(overall_ic_list) / (np.std(overall_ic_list) + 1e-10):.4f}")
        print(f"  胜率: {np.mean([1 if i > 0 else 0 for i in overall_ic_list]):.1%}")

    # 分行业
    print(f"\n分行业表现:")
    industry_results = {}
    for cat in detailed_industries.keys():
        cat_df = results_df[results_df['industry'] == cat]
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

    for cat, res in sorted(industry_results.items(), key=lambda x: -x[1]['ir']):
        print(f"  {cat}: IC={res['ic_mean']:.4f}, IR={res['ir']:.4f}, 胜率={res['win_rate']:.1%}")

    # Top-N选股
    print(f"\n全市场Top-10选股:")
    portfolio_returns = []

    for date, group in results_df.groupby('date'):
        if len(group) < 10:
            continue
        top_n = group.nlargest(10, 'signal')
        portfolio_returns.append(top_n['future_ret'].mean())

    if portfolio_returns:
        print(f"  平均收益: {np.mean(portfolio_returns) * 100:.2f}%")
        print(f"  夏普比率: {np.mean(portfolio_returns) / (np.std(portfolio_returns) + 1e-10) * np.sqrt(12):.2f}")
        print(f"  正收益占比: {np.mean([1 if r > 0 else 0 for r in portfolio_returns]):.1%}")

    # 保存
    results_dir = os.path.join(project_root, 'strategy/rolling_validation_results')
    os.makedirs(results_dir, exist_ok=True)

    results_df.to_csv(os.path.join(results_dir, 'rolling_signals.csv'), index=False)
    pd.DataFrame(factor_selection_log).to_csv(
        os.path.join(results_dir, 'factor_selection_log.csv'), index=False
    )

    print(f"\n结果已保存到 {results_dir}/")

    return results_df


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
