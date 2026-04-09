#!/usr/bin/env python
import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_csv('rolling_validation_results/validation_results.csv')
df['date'] = pd.to_datetime(df['date'])
df['score_rank'] = df.groupby('date')['score'].rank(ascending=False, pct=True)

print('=== 选股策略对比 ===')

# 策略对比
strategies = [
    ('策略A: 选Top10%', 'score_rank <= 0.10'),
    ('策略B: 选Top20%', 'score_rank <= 0.20'),
    ('策略C: 选Top30%', 'score_rank <= 0.30'),
    ('策略D: 选10-30%', 'score_rank > 0.10 and score_rank <= 0.30'),
    ('策略E: 选20-40%', 'score_rank > 0.20 and score_rank <= 0.40'),
    ('策略F: 选30-50%', 'score_rank > 0.30 and score_rank <= 0.50'),
    ('策略G: 选40-60%', 'score_rank > 0.40 and score_rank <= 0.60'),
]

results = []
for name, cond in strategies:
    sub = df.query(cond)
    if len(sub) > 100:
        acc = (sub['future_ret'] > 0).mean()
        avg_ret = sub['future_ret'].mean()
        ic, _ = stats.spearmanr(sub['score'], sub['future_ret'])
        results.append({
            'strategy': name,
            'accuracy': acc,
            'return': avg_ret,
            'ic': ic,
            'n': len(sub)
        })

# 打印结果
for r in results:
    print(f"{r['strategy']}: 准确率={r['accuracy']*100:.2f}%, 收益={r['return']*100:.2f}%, IC={r['ic']*100:.2f}%")

# 计算最佳策略的年化收益（假设每次持仓10只）
print('\n=== 策略F(30-50%)详细分析 ===')
sub = df.query('score_rank > 0.30 and score_rank <= 0.50')
# 每天的股票数
daily_counts = sub.groupby('date').size()
print(f'每天可选股票数: 均值={daily_counts.mean():.1f}, 范围={daily_counts.min()}-{daily_counts.max()}')

# 如果每次选10只，每天都能满足吗？
print(f'能满足选10只的天数比例: {(daily_counts >= 10).mean()*100:.1f}%')