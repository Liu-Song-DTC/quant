# analysis/rolling_signal_validator.py
"""
滚动信号验证模块

使用实际SignalEngine验证信号质量:
- 回看期: 120天
- 展望期: 20天
- 采样频率: 每20天（减少计算量）

评估:
- 总体IC/IR/胜率
- 分行业信号质量
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

# 添加路径
script_dir = os.path.dirname(os.path.abspath(__file__))
strategy_dir = os.path.dirname(script_dir)
project_root = os.path.dirname(strategy_dir)
sys.path.insert(0, strategy_dir)

from core.config_loader import load_config
from core.signal_engine import SignalEngine
from core.fundamental import FundamentalData
from core.signal_store import SignalStore

config = load_config(os.path.join(project_root, 'strategy/config/factor_config.yaml'))
detailed_industries = config.config.get('detailed_industries', {})


def generate_signals_for_stock(args):
    """为单只股票生成信号"""
    code, df, sample_dates, fd = args
    stock_dates = sorted(df.index.tolist())

    # 在每个进程中创建SignalEngine（无法跨进程传递）
    signal_engine = SignalEngine()
    signal_engine.set_fundamental_data(fd)

    results = []
    # 找到每个采样日期对应的索引
    date_to_idx = {d: i for i, d in enumerate(stock_dates)}

    for sample_date in sample_dates:
        valid_dates = [d for d in stock_dates if d <= sample_date]
        if len(valid_dates) < 120:
            continue
        eval_date = valid_dates[-1]
        idx = date_to_idx.get(eval_date)
        if idx is None or idx < 120:
            continue

        # 提取120天历史窗口（与因子验证一致）
        history = df.iloc[:idx+1].iloc[-120:]
        if len(history) < 60:
            continue

        # 添加datetime列
        history_copy = history.copy()
        history_copy['datetime'] = history_copy.index

        # 生成信号
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
                    'score': sig.factor_value,  # 使用原始因子值，与因子验证一致
                    'buy': sig.buy,
                    'factor_name': sig.factor_name,
                    'future_ret': future_ret,
                })

    return results


def run_validation(stock_data: dict, fd: FundamentalData, num_workers: int = 8):
    """运行验证"""
    print("=" * 60)
    print("滚动信号验证 - 使用SignalEngine")
    print(f"回看期: 120天, 展望期: 20天")
    print("=" * 60)

    # 生成采样日期（每20天，减少计算量）
    all_dates = set()
    for df in stock_data.values():
        all_dates.update(df.index.tolist())

    common_dates = sorted(all_dates)[120:-20]
    sample_dates = common_dates[::5]  # 每5天采样，与因子验证一致

    print(f"采样日期数量: {len(sample_dates)}")

    print("\n生成信号...")
    codes = list(stock_data.keys())
    args_list = [(code, stock_data[code], sample_dates, fd) for code in codes]

    all_signals = []
    with Pool(num_workers) as pool:
        for res in tqdm(pool.imap(generate_signals_for_stock, args_list, chunksize=10), total=len(args_list), desc="计算信号"):
            all_signals.extend(res)

    signals_df = pd.DataFrame(all_signals)
    print(f"信号数据: {len(signals_df)} 条")

    if len(signals_df) == 0:
        print("没有信号数据!")
        return

    # 验证
    valid = signals_df.dropna(subset=['score', 'future_ret'])
    print(f"有效数据: {len(valid)} 条")

    # 1. 总体
    print("\n" + "=" * 60)
    print("总体信号验证")
    print("=" * 60)

    # 对信号分数按行业做 Rank 标准化（与因子验证一致）
    valid = valid.copy()
    valid['industry'] = valid['factor_name'].str.extract(r'IND_([^_]+)')

    ic_list = []
    for date, group in valid.groupby('date'):
        if len(group) >= 10:
            # 对每个行业内的分数做 Rank 百分位标准化
            fv_rank = np.zeros(len(group))
            fr = group['future_ret'].values

            for i, (idx, row) in enumerate(group.iterrows()):
                ind_name = row['industry']
                ind_mask = group['industry'] == ind_name
                ind_scores = group.loc[ind_mask, 'score'].values
                # 百分位排名
                n = len(ind_scores)
                if n > 1:
                    rank = np.argsort(np.argsort(ind_scores)) / (n - 1)
                else:
                    rank = np.array([0.5])
                # 找到当前股票在行业内的排名
                local_idx = list(ind_scores).index(row['score'])
                fv_rank[i] = rank[local_idx]

            valid_mask = ~np.isnan(fv_rank) & ~np.isnan(fr)
            if valid_mask.sum() >= 5:
                ic, _ = stats.spearmanr(fv_rank[valid_mask], fr[valid_mask])
                if not np.isnan(ic):
                    ic_list.append(ic)

    if ic_list:
        ic_mean = np.mean(ic_list)
        ic_std = np.std(ic_list)
        ir = ic_mean / (ic_std + 1e-10)
        win_rate = np.mean([1 if i > 0 else 0 for i in ic_list])

        print(f"IC均值: {ic_mean:.4f} ({ic_mean*100:.2f}%)")
        print(f"IC标准差: {ic_std:.4f}")
        print(f"IR: {ir:.4f}")
        print(f"胜率: {win_rate:.2%}")
        print(f"验证期数: {len(ic_list)}")

    # 2. 分行业（行业内Rank标准化）
    print("\n" + "=" * 60)
    print("分行业验证 (与因子验证一致)")
    print("=" * 60)

    industry_results = {}
    for cat in detailed_industries.keys():
        cat_codes = []
        for code in codes:
            try:
                ind = fd.get_industry(code, '2024-01-01')
                if any(kw in str(ind) for kw in detailed_industries[cat]):
                    cat_codes.append(code)
            except:
                pass

        if len(cat_codes) < 5:
            continue

        cat_df = valid[valid['code'].isin(cat_codes)]
        if len(cat_df) < 10:
            continue

        ic_list = []
        for date, group in cat_df.groupby('date'):
            if len(group) >= 3:
                # 直接用原始分数（与因子验证一致）
                fv = group['score'].values
                fr = group['future_ret'].values
                valid_mask = ~(np.isnan(fv) | np.isnan(fr))
                if valid_mask.sum() >= 2:
                    ic, _ = stats.spearmanr(fv[valid_mask], fr[valid_mask])
                    if not np.isnan(ic):
                        ic_list.append(ic)

        if ic_list:
            industry_results[cat] = {
                'ic_mean': np.mean(ic_list),
                'ic_std': np.std(ic_list),
                'ir': np.mean(ic_list) / (np.std(ic_list) + 1e-10),
                'win_rate': np.mean([1 if i > 0 else 0 for i in ic_list]),
                'n_periods': len(ic_list),
            }

    for cat, st in sorted(industry_results.items(), key=lambda x: -x[1].get('ir', 0)):
        print(f"{cat}: IC={st['ic_mean']:.4f}, IR={st['ir']:.4f}, 胜率={st['win_rate']:.1%}")

    # 3. 因子统计
    print("\n" + "=" * 60)
    print("因子使用统计")
    print("=" * 60)

    if 'factor_name' in signals_df.columns:
        factor_counts = signals_df['factor_name'].value_counts().head(10)
        print("Top10因子:")
        for name, count in factor_counts.items():
            print(f"  {name}: {count}")

    # 保存结果
    results_dir = os.path.join(project_root, 'strategy/rolling_validation_results')
    os.makedirs(results_dir, exist_ok=True)

    signals_df.to_csv(os.path.join(results_dir, 'signal_validation_results.csv'), index=False)

    rows = []
    if ic_list:
        rows.append({'category': '总体', 'ic_mean': ic_mean, 'ic_std': ic_std,
                     'ir': ir, 'win_rate': win_rate, 'n_periods': len(ic_list)})

    for cat, st in industry_results.items():
        rows.append({'category': cat, **st})

    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(results_dir, 'signal_quality_report.csv'), index=False)

    print(f"\n结果已保存到 {results_dir}/")

    return signals_df


if __name__ == '__main__':
    DATA_PATH = os.path.join(project_root, 'data/stock_data/backtrader_data/')
    FUND_PATH = os.path.join(project_root, 'data/stock_data/fundamental_data/')

    # 加载股票数据
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
    print("加载基本面数据...")
    fd = FundamentalData(FUND_PATH, list(stock_data.keys()))

    # 运行验证
    run_validation(stock_data, fd, num_workers=8)
