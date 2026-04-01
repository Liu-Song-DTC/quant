#!/usr/bin/env python
"""诊断动态因子IC质量问题 - 系统性分析"""

if __name__ == '__main__':
    import sys
    sys.path.insert(0, '/Users/litiancheng01/code/ltc/quant/strategy')

    import pandas as pd
    import numpy as np
    from scipy import stats
    from core.factor_preparer import prepare_factor_data
    from core.fundamental import FundamentalData
    from core.industry_mapping import INDUSTRY_KEYWORDS
    from core.config_loader import load_config
    import os

    # 配置
    config = load_config()
    data_path = '/Users/litiancheng01/code/ltc/quant/data/stock_data/backtrader_data/'
    fundamental_path = '/Users/litiancheng01/code/ltc/quant/data/stock_data/fundamental_data/'

    # 加载数据
    stock_files = [f.replace('_hfq.csv', '') for f in os.listdir(data_path) if f.endswith('_hfq.csv') and not f.startswith('sh')]
    stock_codes = [s for s in stock_files if s != 'sh000001'][:200]  # 用200只股票

    print("=== 1. 加载数据 ===")
    fd = FundamentalData(fundamental_path, stock_codes)
    print(f"基本面数据: {len(fd.stock_data)} 只股票")

    stock_data_dict = {}
    for code in stock_codes:
        try:
            df = pd.read_csv(f'{data_path}{code}_hfq.csv', parse_dates=['datetime'])
            stock_data_dict[code] = df
        except:
            pass
    print(f"价格数据: {len(stock_data_dict)} 只股票")

    print("\n=== 2. 准备因子数据 ===")
    factor_df, industry_codes, all_dates = prepare_factor_data(
        stock_data_dict, fd, INDUSTRY_KEYWORDS, num_workers=2
    )
    print(f"factor_df: {len(factor_df)} 行")
    print(f"all_dates: {len(all_dates)} 个日期")

    # 打印行业分布
    print("\n=== 3. 行业股票分布 ===")
    for ind, codes in sorted(industry_codes.items()):
        print(f"  {ind}: {len(codes)} 只")

    # 关键参数
    train_window_days = config.get('dynamic_factor', {}).get('train_window_days', 120)
    forward_period = config.get('dynamic_factor', {}).get('forward_period', 20)
    min_ic_dates = config.get('dynamic_factor', {}).get('min_ic_dates', 5)
    ic_decay_factor = config.get('dynamic_factor', {}).get('ic_decay_factor', 1.0)

    print(f"\n关键参数: train_window={train_window_days}, forward={forward_period}, min_ic_dates={min_ic_dates}, decay={ic_decay_factor}")

    # 诊断：手动计算每个行业的因子IC质量
    print("\n=== 4. 因子IC质量诊断（按行业） ===")

    exclude_cols = ['code', 'date', 'future_ret', 'industry']
    factor_names = [c for c in factor_df.columns if c not in exclude_cols]
    print(f"可用因子: {len(factor_names)} 个")

    # 用一个典型日期测试（选择数据充足的日期）
    test_dates = [all_dates[500], all_dates[1000]]
    ic_stats_by_factor = {}

    for val_date in test_dates:
        print(f"\n--- 测试日期: {val_date} ---")
        val_idx = all_dates.index(val_date)
        train_start_idx = max(0, val_idx - train_window_days)
        train_start = all_dates[train_start_idx]
        train_end = val_date - pd.Timedelta(days=forward_period)

        train_df = factor_df[(factor_df['date'] >= train_start) & (factor_df['date'] < train_end)]
        print(f"训练窗口: {train_start} 到 {train_end}, 共 {len(train_df)} 条记录")

        for industry, codes in list(industry_codes.items())[:3]:
            if not codes:
                continue
            ind_df = train_df[train_df['code'].isin(codes)]
            if len(ind_df) < 30:
                print(f"  {industry}: 样本不足 ({len(ind_df)})")
                continue

            dates_sorted = sorted(ind_df['date'].unique())
            print(f"\n  {industry}: {len(ind_df)} 条记录, {len(dates_sorted)} 个日期")

            # 计算每个因子的IC质量
            for fn in factor_names[:5]:
                if fn not in ind_df.columns:
                    continue

                ic_list = []
                for date in dates_sorted:
                    group = ind_df[ind_df['date'] == date]
                    if len(group) >= 3:
                        valid_mask = ~(np.isnan(group[fn].values) | np.isnan(group['future_ret'].values))
                        if valid_mask.sum() >= 3:
                            ic, _ = stats.spearmanr(group[fn].values[valid_mask], group['future_ret'].values[valid_mask])
                            if not np.isnan(ic):
                                ic_list.append(ic)

                if len(ic_list) >= min_ic_dates:
                    # 计算IC指标
                    n = len(ic_list)
                    if ic_decay_factor < 1.0:
                        weights = np.array([ic_decay_factor ** (n - i - 1) for i in range(n)])
                        weights = weights / weights.sum()
                        ic_mean = np.sum(np.array(ic_list) * weights)
                    else:
                        ic_mean = np.mean(ic_list)

                    ic_std = np.std(ic_list) + 1e-10
                    ir = ic_mean / ic_std
                    ic_signs = np.sign(ic_list)
                    ic_stability = np.abs(np.sum(ic_signs)) / len(ic_signs)
                    t_stat = ic_mean / (ic_std / np.sqrt(n))
                    combined_ir = ir * (0.5 + 0.5 * ic_stability)

                    print(f"    {fn}: IC_mean={ic_mean:.4f}, IR={ir:.4f}, stability={ic_stability:.2f}, t={t_stat:.2f}, combined={combined_ir:.4f}")

                    # 记录统计
                    key = fn
                    if key not in ic_stats_by_factor:
                        ic_stats_by_factor[key] = []
                    ic_stats_by_factor[key].append({
                        'ic_mean': ic_mean, 'ir': ir, 'stability': ic_stability,
                        't_stat': t_stat, 'combined_ir': combined_ir
                    })

    print("\n=== 5. IC质量分布汇总 ===")
    for fn, stats_list in ic_stats_by_factor.items():
        avg_combined = np.mean([s['combined_ir'] for s in stats_list])
        avg_stability = np.mean([s['stability'] for s in stats_list])
        print(f"  {fn}: avg_combined_ir={avg_combined:.4f}, avg_stability={avg_stability:.2f}")

    print("\n=== 6. 问题诊断 ===")
    # 检查combined_ir异常高的原因
    high_ir_count = 0
    low_stability_count = 0
    for fn, stats_list in ic_stats_by_factor.items():
        for s in stats_list:
            if s['ir'] > 2.0:
                high_ir_count += 1
            if s['stability'] < 0.5:
                low_stability_count += 1

    print(f"IR > 2.0 的因子-日期组合: {high_ir_count}")
    print(f"stability < 0.5 的因子-日期组合: {low_stability_count}")
    print("\n如果IR普遍很高但stability低，说明IC方向不稳定但幅度大")
    print("如果IR高且stability也高，说明IC确实好")

    print("\n=== 诊断完成 ===")
