# 调优实验计划 — 2026-06-22 → 2026-06-23

## 当日发现（6/22）

### 1. CSV 序列化 bug
- 同一套信号：fresh generation → Sharpe 1.30，fast mode (CSV加载) → 0.93
- 根因未定位，疑似 bool 列解析或列序问题
- **规避方案：每次回测前删除 signal CSV，强制 fresh generation**

### 2. 门控不生效
- B3 gate: 无论阈值 1.00 还是 2.50，Sharpe 始终 0.8646
- B2/B3 信号占比完全相同 (B2=1.0%, B3=2.3%)
- **疑似根因**: `result['b3_breakout_vol_ratio']` 等字段在 indicator 输出中不存在，默认值为 0.0，导致所有条件永远不满足
- 后果: B3 被 gate 拒绝后，由龙虎榜 `dt_sig > 0.3` 重新买入 → B3 信号质量退化

### 3. Gate wiring 回归
- clean HEAD (f92a263): Sharpe 1.30
- gate wiring: Sharpe 0.86 (2025年尤其明显: 27% → 10%)

---

## 明日计划 (6/23)

### P0: 修复门控不生效（代码改动）

1. **验证 B3 字段是否存在**
   - 在 `_vectorized_score_assembly` 中打印 `ind` dict 的 key 列表
   - 检查 `b3_breakout_vol_ratio`, `b3_pullback_vol_ratio`, `b3_pullback_shallowness`, `b3_trend_rank` 是否存在
   
2. **如果字段不存在**，从 `ind` 中正确的 key 名提取：
   - 搜索 chan_theory.py 中这些字段的实际 key 名
   - 修正 `_safe_get_arr` 调用

3. **如果字段存在**，检查默认值和实际值分布：
   - 打印 B3 信号的实际字段值
   - 据此调整 gate 阈值

4. **验证门控生效后再重新跑 Phase 1 实验**

### P1: Phase 1 实验（仅 config 改动）

前提：门控已修复并生效

| Exp | 改动 | 说明 |
|-----|------|------|
| 1A | vol_breakout: 1.30→1.15 | 保守放松 |
| 1B | vol_breakout: 1.30→1.00 | 中等放松 |
| 2A | min_conditions: 2→1 | 条件数放松 |
| 2B | min_cond=1 + vol=1.15 | 组合 |
| 3A | B2 conf: 0.30→0.25 | B2 置信度放松 |
| 4 | 取最优组合 | |

### P2: Phase 2 准备（代码改动）

- B1 gate 接线: `factor_config.yaml` +15行, `signal_engine.py` +20行
- 新增字段: min_b1_confidence, min_div_strength, min_v_reversal_dd, min_gap_up

---

## 关键文件

| 文件 | 状态 |
|------|------|
| `strategy/core/signal_engine.py` | 已改动（B2/B3 gate + result fields） |
| `strategy/config/factor_config.yaml` | 已改动（b3_filter 阈值） |
| `data/data_manager.py` | 已改动（pd.to_numeric fix） |

## 回测 SOP
```bash
# 1. 删除旧信号，强制 fresh gen
rm -f strategy/rolling_validation_results/backtest_signals.csv

# 2. 修改 config（如需）
vim strategy/config/factor_config.yaml

# 3. 跑回测
cd strategy && python bt_execution.py 2>&1 | tee bt_execution_$(date +%Y%m%d_%H%M%S).log

# 4. 记录结果
grep -A 10 "向量化回测结果\|买入信号分布" <logfile>
```
