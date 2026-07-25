# 7/25 调优计划

## 当前状态
- 基准: commit 8e0f082, 收益 390%, Sharpe 1.54, 2022 -4.62%, 2023 -2.32%
- 全禁止损版: 收益 250%, Sharpe 1.35, 仍远低于390%
- 当前未提交改动在 portfolio.py / bt_execution.py / industry_chain.py
- 目标: 确认收益差异根因, 在390%基础上增量改善

## 7/24 主要发现

### Bug修复
1. _regime 未定义 (603/613行) → 已修复
2. candidates 未定义 (1931行) → 已修复  
3. cost_tracker 从未填充 → 所有止损循环被跳过 → 看似390%实则在裸奔

### 止损迭代结论
- peak_trail 收紧→过早止盈, 收益崩溃(210次→390→95%)
- cost_stop 12%→同样问题(75次@-9.6%)
- 最终: 全禁(NORM cost_stop=99%, peak_trail仅BEAR), 仍无法恢复390%
- **未解问题**: 逻辑上应与基线等价, 但收益仅217%

### 选股分析
- B0 (无结构) score=0.318 > B4+ score=0.279
- 信号引擎质量OK (入选 vs 被拒 score差距显著)
- Q1 2021 零信号 → 数据冷启动限制, 非bug

### 行业链
- BOM 命中率 28% → 已扩充至153概念+65 NO_CHAIN
- 命中率低不是性能瓶颈, 风格概念无链是正常的

## 明日Step 1: A/B对比定位差异

```bash
# 1. Stash 当前改动
git stash

# 2. 确认 390% 基线仍可复现
/quant/.venv/bin/python strategy/bt_execution.py

# 3. 逐个恢复改动, 每次跑回测:
#    a. cost_tracker alone (bt_execution.py)
#    b. + bear_risk 去 trend 要求  
#    c. + no_chan_penalty -0.20→-0.15
#    d. + peak_trail 仅 BEAR
#    e. + cost_stop 99%

# 4. 找到导致收益下降的具体改动
```

## 待解决问题
- 2022/2023 仍需正收益
- cost_tracker 填充后如何利用而不是伤害收益
- bear_risk_fast 期间的仓位管理
