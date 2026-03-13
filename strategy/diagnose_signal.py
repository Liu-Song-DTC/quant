# 诊断脚本：检查信号分布
import pandas as pd
import numpy as np
import os
from core.signal_engine import SignalEngine
from core.signal_store import SignalStore

DATA_PATH = "../data/stock_data/backtrader_data/"

# 只分析少量股票
sample_stocks = ['600000', '600016', '600028', '600036', '600050']

signal_engine = SignalEngine()
signal_store = SignalStore()

# 生成信号
for item in os.listdir(DATA_PATH):
    if not item.endswith('.csv'):
        continue
    code = item[:-8]
    if code not in sample_stocks:
        continue
    if code == 'sh000001':
        continue

    df = pd.read_csv(DATA_PATH + item, parse_dates=['datetime'])
    signal_engine.generate(code, df, signal_store)

# 统计信号分布
buy_count = 0
sell_count = 0
neutral_count = 0
scores = []
sell_scores = []

for (code, date), sig in signal_store._store.items():
    if code not in sample_stocks:
        continue
    scores.append(sig.score)
    if sig.buy:
        buy_count += 1
    elif sig.sell:
        sell_count += 1
        sell_scores.append(sig.score)
    else:
        neutral_count += 1

print(f"股票: {sample_stocks}")
print(f"总信号数: {len(scores)}")
print(f"买入信号: {buy_count} ({buy_count/len(scores)*100:.1f}%)")
print(f"卖出信号: {sell_count} ({sell_count/len(scores)*100:.1f}%)")
print(f"中性信号: {neutral_count} ({neutral_count/len(scores)*100:.1f}%)")
print(f"平均分数: {np.mean(scores):.3f}")
print(f"分数>0.25: {sum(1 for s in scores if s > 0.25)}")
print(f"分数>0.30: {sum(1 for s in scores if s > 0.30)}")
print(f"分数>0.35: {sum(1 for s in scores if s > 0.35)}")
print(f"卖出信号分数: {sell_scores[:10]}")  # 前10个卖出信号
