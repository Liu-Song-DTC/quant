# 量化选股系统完整调优实验计划 — 2026-06-21

## 基线

回测: `bt_execution_20260620_162149.log`
Sharpe 1.30 | 总收益 59.83% | 最大回撤 13.56% | 选股 448 次

---

## 一、8 买点代码审计总览

| 买点 | 占比 | Config段 | 加载? | 使用? | 硬编码参数数 | 优先级 |
|------|------|---------|-------|-------|-------------|--------|
| B1 一买 | 6.0% | b1_gate | ✅ | **❌** | ~8 | P1 |
| B2 二买 | 1.0% | b2_gate | ✅ | **✅ 已修复** | ~6 | 本轮 |
| B3 三买 | 2.3% | b3_filter | ✅ | **✅ 已修复** | ~6 | 本轮 |
| B4 回调 | 60.3% | **无** | N/A | N/A | ~10 | P0 |
| B5 启动 | 4.3% | **无** | N/A | N/A | ~8 | P2 |
| BP6 回踩 | 10.7% | **无** | N/A | N/A | ~6 | P1 |
| BP7 震荡 | 2.8% | **无** | N/A | N/A | ~5 | P3 |
| BP8 突破 | 8.9% | **无** | N/A | N/A | ~7 | P1 |

**已完成的基础修复：**
- B3: `b3_filter` config 接线 + 量能/浅度/趋势多条件门控 (signal_engine.py)
- B2: `b2_gate.min_confidence` 门控接线 (signal_engine.py)
- B3 数据字段: `b3_breakout_vol_ratio`, `b3_pullback_vol_ratio`, `b3_pullback_shallowness`, `b3_trend_rank`, `second_buy_confidence` 加入 result 传递链
- data_manager.py: `pd.to_numeric()` 修复 akshare 字符串兼容

---

## 二、全部实验列表（共 6 个 Phase, 21 个实验）

---

### Phase 0: 基线验证 (1 实验)

**前提**: 不改任何阈值，确认 gate 接线不破坏系统。

#### Exp 0 — 纯基线
- 操作: 直接跑 `python bt_execution.py`
- 预期: Sharpe≈1.30, B2≈1.0%, B3≈2.3%

---

### Phase 1: B2/B3 调优 (6 实验)

**前提**: B2/B3 gate 已接线，只需改 config 值。

#### Exp 1A — B3 突破量比保守放松
- 改: `gate_vol_breakout_min: 1.30 → 1.15`
- 固定: gate_min_conditions=2, gate_confirm=true

#### Exp 1B — B3 突破量比中等放松
- 改: `gate_vol_breakout_min: 1.30 → 1.00`

#### Exp 2A — B3 条件数放松
- 改: `gate_min_conditions: 2 → 1`

#### Exp 2B — B3 条件数+量比组合
- 改: `gate_min_conditions: 2 → 1` + `gate_vol_breakout_min: 1.15`

#### Exp 3A — B2 置信度保守放松
- 改: `b2_gate.min_confidence: 0.30 → 0.25`
- 注: 硬编码 0.35，config 0.30 生效后会先过滤掉 conf<0.30 的 B2，改 0.25 等同不限制

#### Exp 3B — B2 分型质量放松
- 改: `chan_theory.py` 两处 `bottom_fractal_quality < 0.30` → `0.25`
- 注: 这是代码改动，需要重新生成信号

#### Exp 4 — B2+B3 最优组合
- 取 Exp1-3 中各自表现最好的参数组合

---

### Phase 2: B1 gate 接线 + 实验 (代码改动 + 2 实验)

**前提**: `signal_engine.py:274-275` 已加载 `chan_b1_min_bottom_div` (0.15) 和 `chan_b1_min_fx_vol_spike` (1.5)，但从未用于过滤。

**实现改动 (1 次性代码修改):**
1. `factor_config.yaml` b1_gate 下新增字段:
   - `min_b1_confidence: 0.25` — B1 最低置信度
   - `min_div_strength: 0.10` — 底背离强度阈值
   - `min_v_reversal_dd: 0.15` — V型反转最低跌幅
   - `min_gap_up: 0.02` — 缺口幅度阈值
2. `signal_engine.py`: bp_buy==1 时增加 B1 质量门控

#### Exp 5A — B1 启用门控
- 操作: 接线后不改阈值跑一次
- 预期: B1≈5-6% (比基线略少，过滤低质量)

#### Exp 5B — B1 放松门槛
- 改: `min_b1_confidence: 0.20`, `min_div_strength: 0.08`

---

### Phase 3: B4 添加 config + 实验 (代码改动 + 4 实验)

**前提**: B4 占 60% 信号，所有阈值硬编码在 `chan_theory.py:1117-1158`。

**需提取的硬编码参数:**
| 参数 | 当前值 | 含义 | 位置 |
|------|--------|------|------|
| `b4_ma_bull_required` | true | 需 MA20>MA60 | chan_theory:1131 |
| `b4_mom_5d_max` | 0.03 | 5日涨幅上限 | :1135 |
| `b4_dist_ma20_min` | -0.05 | 距MA20下限 | :1139 |
| `b4_dist_ma20_max` | 0.08 | 距MA20上限 | :1139 |
| `b4_vol_contraction_ratio` | 1.1 | 缩量比例 | :1146 |
| `b4_mom_60d_max` | 0.50 | 60日涨幅上限 | :1150 |
| `b4_mom_20d_max` | 0.25 | 20日涨幅上限 | :1152 |
| `b4_min_confidence` | 0.20 | 最低置信度 | :1158 |

**实现改动 (1 次性代码修改):**
1. `factor_config.yaml` 新增 `b4_gate:` 配置段（含上表所有参数）
2. `signal_engine.py`: 加载 `b4_gate`，bp_buy==4 时应用 B4 质量门控
   - 均线距离检查
   - 缩量确认 (已有 vol data)
   - 力竭检查
   - 置信度门控

#### Exp 6A — B4 启用门控
- 操作: 接线后不改阈值
- 预期: B4 从 60% 降到 45-55% (过滤低质量回调)

#### Exp 6B — B4 收紧回调范围
- 改: `b4_dist_ma20_min: -0.03`, `b4_dist_ma20_max: 0.05` (收窄)

#### Exp 6C — B4 放松回调范围
- 改: `b4_dist_ma20_min: -0.08`, `b4_dist_ma20_max: 0.12` (放宽)

#### Exp 6D — B4 收紧力竭
- 改: `b4_mom_60d_max: 0.35`, `b4_mom_20d_max: 0.18`

---

### Phase 4: BP6/BP8 添加 config + 实验 (代码改动 + 4 实验)

**BP6 硬编码参数** (`chan_theory.py:1204-1240`):
| 参数 | 当前值 | 含义 |
|------|--------|------|
| `b6_near_ma20_pct` | 0.03 | MA20附近范围 |
| `b6_near_ma60_pct` | 0.05 | MA60附近范围 |
| `b6_had_rally_pct` | 1.05 | 曾有上涨确认 |
| `b6_bounce_3d_min` | -0.02 | 3日反弹下限 |
| `b6_vol_ok_ratio` | 1.2 | 回踩量比上限 |
| `b6_min_confidence` | 0.20 | 最低置信度 |

**BP8 硬编码参数** (`chan_theory.py:1274-1307`):
| 参数 | 当前值 | 含义 |
|------|--------|------|
| `b8_vol_contract_ratio` | 0.70 | ATR收缩比例 |
| `b8_tight_range_pct` | 0.12 | 窄幅盘整上限 |
| `b8_breakout_vol_ratio` | 1.3 | 突破量比 |
| `b8_price_breakout_min` | 0.01 | 突破涨幅下限 |
| `b8_min_confidence` | 0.15 | 最低置信度 |

**实现改动 (1 次性代码修改):**
1. `factor_config.yaml` 新增 `b6_gate:` 和 `b8_gate:` 配置段
2. `signal_engine.py`: 加载并应用

#### Exp 7A — BP6 启用门控
- 操作: 接线后不改阈值

#### Exp 7B — BP6 收紧均线距离
- 改: `b6_near_ma20_pct: 0.02`

#### Exp 8A — BP8 启用门控
- 操作: 接线后不改阈值

#### Exp 8B — BP8 收紧突破量比
- 改: `b8_breakout_vol_ratio: 1.5`

---

### Phase 5: B5/BP7 添加 config + 实验 (代码改动 + 2 实验)

**B5 硬编码参数** (`chan_theory.py:1160-1202`):
| 参数 | 当前值 | 含义 |
|------|--------|------|
| `b5_mom_5d_min` | 0.005 | 5日动量下限 |
| `b5_mom_5d_max` | 0.10 | 5日动量上限 |
| `b5_vol_confirm_ratio` | 0.85 | 量确认比例 |
| `b5_mom_10d_max` | 0.18 | 10日涨幅上限 |
| `b5_had_pullback_ratio` | 0.92 | 曾有回落确认 |
| `b5_min_confidence` | 0.15 | 最低置信度 |

**BP7 硬编码参数** (`chan_theory.py:1242-1272`):
| 参数 | 当前值 | 含义 |
|------|--------|------|
| `b7_near_zd_pct` | 0.03 | ZD附近范围 |
| `b7_pivot_width_min` | 0.03 | 中枢最小宽度 |
| `b7_window_bars` | 30 | 中枢完成后窗口 |
| `b7_min_confidence` | 0.20 | 最低置信度 |

**实现改动 (1 次性代码修改):**
1. `factor_config.yaml` 新增 `b5_gate:` 和 `b7_gate:` 配置段
2. `signal_engine.py`: 加载并应用

#### Exp 9A — B5 启用门控
- 操作: 接线后不改阈值

#### Exp 10A — BP7 启用门控
- 操作: 接线后不改阈值

---

### Phase 6: 全买点组合最优 (2 实验)

#### Exp 11 — 各买点最优参数单独组合
- 取 Phase 1-5 各自最优参数

#### Exp 12 — 全局微调
- 在 Exp11 基础上，看 B2/B3 占比是否达 8-12%，否则进一步放松

---

## 三、实验执行清单（按顺序）

```
Phase 0:  [Exp0]
Phase 1:  [Exp1A] [Exp1B] [Exp2A] [Exp2B] [Exp3A] [Exp3B] [Exp4]
Phase 2:  [实现 B1 gate]  [Exp5A] [Exp5B]
Phase 3:  [实现 B4 gate]  [Exp6A] [Exp6B] [Exp6C] [Exp6D]
Phase 4:  [实现 BP6/BP8 gate] [Exp7A] [Exp7B] [Exp8A] [Exp8B]
Phase 5:  [实现 B5/BP7 gate] [Exp9A] [Exp10A]
Phase 6:  [Exp11] [Exp12]
```

---

## 四、代码改动量预估

| Phase | 文件 | 改动 |
|-------|------|------|
| 0 | 无 | — |
| 1 | factor_config.yaml | 改已有字段值 |
| 2 | factor_config.yaml + signal_engine.py | +15 config行 +20 代码行 |
| 3 | factor_config.yaml + signal_engine.py | +15 config行 +30 代码行 |
| 4 | factor_config.yaml + signal_engine.py | +25 config行 +30 代码行 |
| 5 | factor_config.yaml + signal_engine.py | +20 config行 +20 代码行 |
| 6 | factor_config.yaml | 改已有字段值 |

---

## 五、结果记录表

| Exp | Phase | 参数改动 | Sharpe | 总收益% | 回撤% | B1% | B2% | B3% | B4% | B5% | BP6% | BP7% | BP8% | 选股次数 |
|-----|-------|---------|--------|---------|-------|-----|-----|-----|-----|-----|------|------|------|---------|
| 基线 | — | — | 1.30 | 59.83 | 13.56 | 6.0 | 1.0 | 2.3 | 60.3 | 4.3 | 10.7 | 2.8 | 8.9 | 448 |
| 0 | 0 | gate接线 | | | | | | | | | | | | |
| 1A | 1 | vol_brk=1.15 | | | | | | | | | | | | |
| 1B | 1 | vol_brk=1.00 | | | | | | | | | | | | |
| 2A | 1 | min_cond=1 | | | | | | | | | | | | |
| 2B | 1 | min_cond=1+vol=1.15 | | | | | | | | | | | | |
| 3A | 1 | B2 conf=0.25 | | | | | | | | | | | | |
| 3B | 1 | fx_quality=0.25 | | | | | | | | | | | | |
| 4 | 1 | 最优组合 | | | | | | | | | | | | |
| 5A | 2 | B1 gate on | | | | | | | | | | | | |
| 5B | 2 | B1 relax | | | | | | | | | | | | |
| 6A | 3 | B4 gate on | | | | | | | | | | | | |
| 6B | 3 | B4 tight | | | | | | | | | | | | |
| 6C | 3 | B4 loose | | | | | | | | | | | | |
| 6D | 3 | B4 exhaustion | | | | | | | | | | | | |
| 7A | 4 | BP6 gate on | | | | | | | | | | | | |
| 7B | 4 | BP6 tight | | | | | | | | | | | | |
| 8A | 4 | BP8 gate on | | | | | | | | | | | | |
| 8B | 4 | BP8 tight | | | | | | | | | | | | |
| 9A | 5 | B5 gate on | | | | | | | | | | | | |
| 10A | 5 | BP7 gate on | | | | | | | | | | | | |
| 11 | 6 | 各自最优 | | | | | | | | | | | | |
| 12 | 6 | 全局微调 | | | | | | | | | | | | |
