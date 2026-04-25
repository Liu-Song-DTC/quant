"""
诊断回测Sharpe(0.283) vs 离线模拟(0.77)的差距原因

重点检查:
1. 信号分布: buy/sell/factor_value/score的统计
2. 组合选股: rank_pct>0.5的股票特征
3. 卖出逻辑: sell信号触发频率、score<0触发频率
4. 非再平衡日的行为: 是否有主动卖出
5. 离线模拟 vs 实际回测的关键差异
"""

import pandas as pd
import numpy as np
import os

# === 加载数据 ===
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
results_dir = os.path.join(base_dir, 'rolling_validation_results')

signals_path = os.path.join(results_dir, 'backtest_signals.csv')
selections_path = os.path.join(results_dir, 'portfolio_selections.csv')

if not os.path.exists(signals_path):
    print(f"错误: 找不到 {signals_path}")
    print("请先运行回测: cd strategy && python bt_execution.py")
    exit(1)

signals = pd.read_csv(signals_path)
print(f"信号数据: {len(signals)} 条")

selections = None
if os.path.exists(selections_path):
    selections = pd.read_csv(selections_path)
    print(f"选股数据: {len(selections)} 条")

# === 1. 信号分布分析 ===
print("\n" + "="*60)
print("1. 信号分布分析")
print("="*60)

# buy/sell 统计
total = len(signals)
buy_count = signals['buy'].sum()
sell_count = signals['sell'].sum()
neither = total - buy_count  # sell只是buy的子集

print(f"总信号: {total}")
print(f"buy=True: {buy_count} ({100*buy_count/total:.1f}%)")
print(f"sell=True: {sell_count} ({100*sell_count/total:.1f}%)")
print(f"buy=False: {total - buy_count} ({100*(total-buy_count)/total:.1f}%)")

# factor_value 分布
fv = signals['factor_value'].dropna()
print(f"\nfactor_value 分布:")
print(f"  mean: {fv.mean():.4f}")
print(f"  std: {fv.std():.4f}")
print(f"  min: {fv.min():.4f}")
print(f"  max: {fv.max():.4f}")
print(f"  >0 占比: {(fv > 0).mean()*100:.1f}%")
print(f"  >0.5 占比: {(fv > 0.5).mean()*100:.1f}%")
print(f"  <-0.15 占比: {(fv < -0.15).mean()*100:.1f}%")

# score 分布
sc = signals['score'].dropna()
print(f"\nscore 分布:")
print(f"  mean: {sc.mean():.4f}")
print(f"  std: {sc.std():.4f}")
print(f"  min: {sc.min():.4f}")
print(f"  max: {sc.max():.4f}")
print(f"  >0 占比: {(sc > 0).mean()*100:.1f}%")
print(f"  <0 占比: {(sc < 0).mean()*100:.1f}%")

# === 2. 每日截面分析 ===
print("\n" + "="*60)
print("2. 每日截面分析")
print("="*60)

# 按日期分组
signals['date'] = pd.to_datetime(signals['date'])
daily_stats = signals.groupby('date').agg(
    n_signals=('code', 'count'),
    n_buy=('buy', 'sum'),
    n_sell=('sell', 'sum'),
    fv_mean=('factor_value', 'mean'),
    fv_median=('factor_value', 'median'),
    score_mean=('score', 'mean'),
).reset_index()

print(f"每日信号数: mean={daily_stats['n_signals'].mean():.0f}, "
      f"min={daily_stats['n_signals'].min():.0f}, max={daily_stats['n_signals'].max():.0f}")
print(f"每日buy数: mean={daily_stats['n_buy'].mean():.0f}")
print(f"每日sell数: mean={daily_stats['n_sell'].mean():.0f}")
print(f"每日fv_mean: {daily_stats['fv_mean'].mean():.4f}")

# === 3. 组合选股分析 ===
print("\n" + "="*60)
print("3. 组合选股分析 (portfolio_selections)")
print("="*60)

if selections is not None and len(selections) > 0:
    selections['date'] = pd.to_datetime(selections['date'])
    daily_sel = selections.groupby('date').agg(
        n_selected=('code', 'count'),
        avg_weight=('weight', 'mean'),
    ).reset_index()

    print(f"选股日期数: {len(daily_sel)}")
    print(f"每日选股数: mean={daily_sel['n_selected'].mean():.1f}, "
          f"min={daily_sel['n_selected'].min():.0f}, max={daily_sel['n_selected'].max():.0f}")
    print(f"平均权重: {daily_sel['avg_weight'].mean():.4f}")

    # 选股的score分布
    sel_scores = selections['score'].dropna()
    print(f"\n选股score分布:")
    print(f"  mean: {sel_scores.mean():.4f}")
    print(f"  >0 占比: {(sel_scores > 0).mean()*100:.1f}%")
    print(f"  <0 占比: {(sel_scores < 0).mean()*100:.1f}%")

    # 行业分布
    if 'industry' in selections.columns:
        ind_dist = selections['industry'].value_counts()
        print(f"\n行业分布 (top 10):")
        for ind, cnt in ind_dist.head(10).items():
            print(f"  {ind}: {cnt}")

    # 检查选股和信号的对应关系
    # 选中的股票，其factor_value是否真的高？
    sel_with_signals = selections.merge(
        signals[['date', 'code', 'factor_value', 'score', 'buy', 'sell']],
        on=['date', 'code'],
        how='left'
    )
    if 'factor_value_y' in sel_with_signals.columns:
        sel_fv = sel_with_signals['factor_value_y'].dropna()
        print(f"\n选中股票的factor_value:")
        print(f"  mean: {sel_fv.mean():.4f}")
        print(f"  median: {sel_fv.median():.4f}")
        print(f"  >0: {(sel_fv > 0).mean()*100:.1f}%")

    # 检查rank_pct (如果有)
    if 'rank_pct' in selections.columns:
        rp = selections['rank_pct'].dropna()
        print(f"\n选中股票的rank_pct:")
        print(f"  mean: {rp.mean():.4f}")
        print(f"  median: {rp.median():.4f}")
        print(f"  >0.5: {(rp > 0.5).mean()*100:.1f}%")
        print(f"  >0.7: {(rp > 0.7).mean()*100:.1f}%")

# === 4. 卖出逻辑分析 ===
print("\n" + "="*60)
print("4. 卖出逻辑分析")
print("="*60)

# sell信号触发频率
sell_signals = signals[signals['sell'] == True]
print(f"sell=True 信号数: {len(sell_signals)} ({100*len(sell_signals)/total:.1f}%)")

# score<0 的频率（也会触发卖出）
neg_score = signals[signals['score'] < 0]
print(f"score<0 信号数: {len(neg_score)} ({100*len(neg_score)/total:.1f}%)")

# factor_value < -0.15 的频率
neg_fv = signals[signals['factor_value'] < -0.15]
print(f"factor_value<-0.15 信号数: {len(neg_fv)} ({100*len(neg_fv)/total:.1f}%)")

# 每日卖出信号数
daily_sell = signals.groupby('date').agg(
    n_sell=('sell', 'sum'),
    n_neg_score=('score', lambda x: (x < 0).sum()),
    n_neg_fv=('factor_value', lambda x: (x < -0.15).sum()),
).reset_index()

print(f"\n每日卖出信号数: mean={daily_sell['n_sell'].mean():.1f}")
print(f"每日score<0数: mean={daily_sell['n_neg_score'].mean():.1f}")
print(f"每日fv<-0.15数: mean={daily_sell['n_neg_fv'].mean():.1f}")

# === 5. 关键差异：离线模拟 vs 实际回测 ===
print("\n" + "="*60)
print("5. 离线模拟 vs 实际回测 关键差异")
print("="*60)

print("""
离线模拟假设:
- 每20天再平衡
- 选中rank_pct>0.5的top10股票
- 等权持仓
- 无交易成本
- 无100股最小手数限制

实际回测:
- 20天再平衡 + 日常止损/止盈
- sell信号 + score<0触发卖出
- 风险调整权重（非等权）
- 有交易成本 (0.15%佣金 + 0.15%滑点)
- 100股最小手数（小资金下影响大）
- REBALANCE_THRESHOLD: 变化<2.5%总资产不调仓
- entry_speed/exit_speed: 渐进进出
- KEEP_RANK_BUFFER: 保留区域
- INDUSTRY_DISCOUNT: 行业权重调整
- max_single_weight: 0.10 单只上限
- position_stop_loss: 0.12 个股止损
- portfolio_stop_loss: 0.08 组合止损
""")

# === 6. 模拟离线选股逻辑验证 ===
print("\n" + "="*60)
print("6. 模拟离线选股逻辑 (rank_pct>0.5 top10)")
print("="*60)

# 对每个日期，模拟portfolio的选股逻辑
simulated_selections = []
for date, group in signals.groupby('date'):
    # 只看有factor_value的
    valid = group.dropna(subset=['factor_value'])
    if len(valid) == 0:
        continue

    # 计算rank_pct
    valid = valid.copy()
    valid['rank_pct'] = valid['factor_value'].rank(pct=True)

    # 过滤rank_pct > 0.5
    qualified = valid[valid['rank_pct'] > 0.5]
    if len(qualified) == 0:
        qualified = valid[valid['rank_pct'] > 0.3]

    # 按factor_value排序
    qualified = qualified.sort_values('factor_value', ascending=False)

    # 行业均衡选股 (每行业最多2只)
    industry_count = {}
    selected = []
    for _, row in qualified.iterrows():
        ind = row.get('industry', 'default')
        if industry_count.get(ind, 0) >= 2:
            continue
        selected.append(row)
        industry_count[ind] = industry_count.get(ind, 0) + 1
        if len(selected) >= 10:
            break

    for row in selected:
        simulated_selections.append({
            'date': date,
            'code': row['code'],
            'factor_value': row['factor_value'],
            'rank_pct': row['rank_pct'],
            'score': row['score'],
            'industry': row.get('industry', ''),
        })

sim_df = pd.DataFrame(simulated_selections)
if len(sim_df) > 0:
    print(f"模拟选股: {len(sim_df)} 条, {sim_df['date'].nunique()} 个日期")
    print(f"每日选股数: {sim_df.groupby('date').size().mean():.1f}")
    print(f"选中股票rank_pct: mean={sim_df['rank_pct'].mean():.4f}, median={sim_df['rank_pct'].median():.4f}")
    print(f"选中股票factor_value: mean={sim_df['factor_value'].mean():.4f}")

    # 对比实际选股
    if selections is not None and len(selections) > 0:
        # 看实际选中的股票在模拟中的rank
        sel_dates = set(selections['date'].unique())
        sim_in_sel = sim_df[sim_df['date'].isin(sel_dates)]
        print(f"\n同日期模拟选股rank_pct: mean={sim_in_sel['rank_pct'].mean():.4f}")

# === 7. 核心问题总结 ===
print("\n" + "="*60)
print("7. 核心问题诊断总结")
print("="*60)

# 检查: buy=True的比例是否过高
buy_rate = buy_count / total
if buy_rate > 0.9:
    print(f"[问题1] buy=True比例={100*buy_rate:.1f}% 过高 → 组合层rank_pct选股有效，但非再平衡日缺乏主动卖出机制")

# 检查: sell信号触发率
sell_rate = sell_count / total
if sell_rate < 0.05:
    print(f"[问题2] sell=True比例={100*sell_rate:.1f}% 过低 → 因子值很少<-0.15，持仓无法主动退出")

# 检查: score<0的比例
neg_score_rate = len(neg_score) / total
if neg_score_rate < 0.3:
    print(f"[问题3] score<0比例={100*neg_score_rate:.1f}% → portfolio.build()中score<0的卖出条件几乎不触发")

# 检查: rank_pct分布
if len(sim_df) > 0:
    rp_mean = sim_df['rank_pct'].mean()
    if rp_mean < 0.7:
        print(f"[问题4] 选中股票rank_pct均值={rp_mean:.3f} → 可能不够'精尖'，包含太多rank接近0.5的边缘股")

# 检查: factor_value分布是否过于集中
fv_std = fv.std()
fv_mean = fv.mean()
if fv_std < 0.3:
    print(f"[问题5] factor_value标准差={fv_std:.4f} → 区分度不够，rank_pct接近随机")

# 核心建议
print("\n=== 核心建议 ===")
print("""
1. 卖出逻辑问题:
   - 当前sell仅看factor_value<-0.15，触发极少
   - score<0的条件中，score=factor_value*0.7+动量*0.3，动量项推高score
   - 建议：卖出条件改为factor_value<0（而非<-0.15），或rank_pct<0.3

2. 非再平衡日行为:
   - 20天才再平衡选股，中间19天只能靠sell/score<0止损退出
   - 如果卖出信号几乎不触发，中间19天持仓完全不动
   - 即使因子值恶化，也要等到再平衡日才调整

3. 组合权重:
   - max_single_weight=0.10 限制了单只权重
   - INDUSTRY_DISCOUNT 对某些行业打折
   - 这些调整可能降低了离线模拟中"等权top10"的效果

4. 建议优化方向:
   A) 让卖出更敏感：factor_value<0就标记sell
   B) 减少再平衡间隔（或让非再平衡日也有调整能力）
   C) 简化权重：用等权而非复杂的风险调整权重
""")
