# 实验计划 — 2026-06-23

## Exp 0 结果: Gate Fix 回测

| 指标 | 基线 (f92a263) | Exp 0 (gate fix) | 变化 |
|------|---------------|-----------------|------|
| Sharpe | **1.30** | **0.72** | -45% |
| 总收益 | 59.83% | 28.46% | -52% |
| 最大回撤 | 13.56% | 13.77% | ≈ |
| 选股次数 | 448 | 452 | ≈ |
| 买入成交率 | ~100% | 87.0% (194/223) | -13% |

### 根因分析: B3 Gate 通过率仅 1.1%

各买点 buy=True 率 (信号CSV):

| 买点 | 总信号 | buy=True | 通过率 | 选股占比 |
|------|--------|----------|--------|---------|
| B1 | 105,218 | 34,558 | 32.8% | 13.8% |
| B2 | 18,580 | 5,537 | **29.8%** | 2.0% |
| **B3** | 19,237 | **209** | **1.1%** | 0.0% |
| B4 | 390,645 | 321,190 | 82.2% | 43.3% |
| B5 | 29,347 | 23,946 | 81.6% | 2.3% |
| BP6 | 86,134 | 58,735 | 68.2% | 21.6% |
| BP7 | 52,391 | 16,567 | 31.6% | 1.6% |
| BP8 | 77,721 | 47,332 | 60.9% | 11.5% |

**B3 被完全摧毁**: 仅 209/19,237 通过 gate，选股 0 只 B3 (选股中的4只B3全部 buy=False)。

**选股中 32.5% (147/452) 是 buy=False**: 其中 88.4% 无买点(chan_buy_point=0)，依赖龙虎榜/背离信号兜底。

### B3 Gate 逻辑审视

```python
# 当前 gate: 4选2，阈值偏高
_pass = 0
if _b3_vol_brk >= 1.3:       # 突破量比 >= 1.3
    _pass += 1
if 0 < _b3_vol_pb <= 0.85:    # 回调量比 <= 0.85
    _pass += 1
if _b3_shallow >= 0.5:        # 回调浅度 >= 0.5
    _pass += 1
if trend_or_div and _b3_trend_ok:  # 趋势确认
    _pass += 1
if _pass < 2: buy = False     # 需要 >= 2 条件
```

**问题**: B3 信号的字段值分布未知，阈值可能完全脱离实际分布。

---

## 修正后的实验计划

### P0: B3 Gate 紧急修复（必须先做）

**目标**: 恢复 B3 信号，gate 通过率目标 > 30%

#### Exp 0A — 禁用 B3 Gate
- `b3_filter.enabled: false`
- 验证: B3 信号回归 ~2.3%，Sharpe 回升
- 如果 Sharpe > 1.0 → gate 本身是问题根因

#### Exp 0B — B3 Gate: min_conditions 2→1
- `gate_min_conditions: 2 → 1`
- 1.1% → 预期 10-20% 通过率

#### Exp 0C — B3 Gate: vol_breakout 1.3→0.8 + min_cond=1
- 大幅降低量比门槛
- 预期 20-40% 通过率

#### Exp 0D — B3 Gate: 打印实际字段值分布
- 在 gate 处加 print 输出 B3 信号的 `vol_brk`, `vol_pb`, `shallow`, `trend_ok` 分布
- 基于实际数据设定合理阈值

### P1: B2 Gate 调优

**目标**: B2 通过率 29.8% → 50-70%

#### Exp 1A — B2 min_confidence: 0.30→0.15
- 保守放半

#### Exp 1B — B2 min_confidence: 0.30→0.05
- 几乎不禁 (只过滤 conf≈0 的)

### P2: 选股质量修复

**问题**: 147/452 (32.5%) 选股是 buy=False，平均权重仅 0.0315 (vs 0.0929)

**根因**: 候选池 buy=True 股票不足时，portfolio 降级到无结构但有 divergence 的兜底股票

#### Exp 2A — 提高候选池数量下限
- 当候选 < N 时跳过该调仓日，而非降级选 buy=False 股票

### P3: 龙虎榜 + Gate 执行顺序修正

**当前**: 龙虎榜(628) → B3 gate(630) → B2 gate(649)
- 龙虎榜设置 buy=True，然后 gate 立即覆盖为 False
- **龙虎榜无法拯救被 gate 拒绝的好信号**

**修正**: B3 gate(先) → 龙虎榜(后) → B2 gate(后)
- gate 先拒绝低质量信号
- 龙虎榜可重新激活有机构支撑的 B3
- B2 置信度 gate 最后检查

---

## 执行优先级

```
P0: Exp 0A (禁用B3 gate) → 快速验证 gate 是否是 Sharpe 下降根因
  ├── 如果 Sharpe > 1.0 → P0B, P0C, P0D (逐步修复 gate)
  └── 如果 Sharpe < 0.8 → 回退所有 gate 代码，从 clean HEAD 开始
P1: Exp 1A, 1B (B2 relax)
P2: Exp 2A (选股质量)
P3: 龙虎榜顺序修正 (代码改动)
```

## 关键文件

| 文件 | 改动类型 |
|------|---------|
| `strategy/config/factor_config.yaml` | config 改动 |
| `strategy/core/signal_engine.py` | gate 顺序修正 |
| `strategy/core/portfolio.py` | 选股质量修复 |

## 回测 SOP

```bash
rm -f strategy/rolling_validation_results/backtest_signals.csv
# 修改 config（如需）
vim strategy/config/factor_config.yaml
# 跑回测
cd strategy && python bt_execution.py 2>&1 | tee bt_execution_expX_$(date +%Y%m%d_%H%M%S).log
```
