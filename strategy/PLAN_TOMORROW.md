# 明日执行计划

## 当前状态

- 季度配置文件需要重新生成（已修复 YAML 序列化 bug）
- 因子计算缓存已就位：`cache/full_factor_data.parquet`
- 8 worker 并行 IC + 因子选择，22 季度约 5h

## 执行步骤

### Step 1: 季度滚动标定
```bash
cd /mnt/d/quant/strategy
python analysis/quarterly_calibrate.py
```
预计 ~5h。缓存命中跳过 Phase 1，直接进入 Phase 2。

### Step 2: 验证配置文件
```bash
python -c "
import yaml
for q in ['2021Q1','2021Q2','2022Q1','2023Q1','2024Q1','2025Q1','2026Q1']:
    with open(f'config/quarterly_factors/{q}.yaml') as f:
        c = yaml.safe_load(f)
    print(f'{q}: {len(c[\"industry_factors\"])} industries OK')
"
```
确保每个季度 YAML 都能正常加载。

### Step 3: 回测
```bash
rm -f rolling_validation_results/backtest_signals.csv
python bt_execution.py
```
预计 ~1.5h。

### Step 4: 检查结果
```bash
tail -40 logs/bt_execution_*.log | grep -A30 "向量化回测结果"
```
关注：Sharpe、年化收益、最大回撤、买入准确率。

## 预期

- bug 修复 + 新因子 + 季度滚动标定 + ML 季度训练 + 真实交易日历
- Sharpe 目标 > 0.5
- 2022 年熊市防御改善
