# D:\quant 实盘前最终审查报告

**审查日期**: 2026-05-27
**审查性质**: 实盘上线前全面审计
**审查结论**: **当前状态不可直接实盘，需修复 3 个阻断性问题**

---

## 🔴 阻断性问题（必须修后才能实盘）

### 1. 持仓状态字典每天被清空 → 止损/止盈全线失效

**根因**: `runner.py:366-380` 每天创建全新的 `SignalRunner` → 全新的 `Strategy` → 全新的 `PortfolioConstructor`。以下 6 个跟踪字典全部重置为空 `{}`：

| 字典 | 依赖的机制 | 失效后果 |
|------|-----------|---------|
| `_entry_dates` | 时间止损(持仓天数检查) | **永不自损** — 每天都是"第一天" |
| `_peak_prices` | 移动止损(最高价回撤) | **永不自损** — 没有峰值可追踪 |
| `_entry_reasons` | 入场理由消失退出 | **永不触发** — 没有理由记录 |
| `_mr_exit_cooldown` | 均值回归退出冷却期 | **绕过冷却** — 刚卖就买回 |
| `_entry_reason_lost_count` | 3天确认期 | **绕过确认** — 立即退出 |
| `_post_sell_tracking` | S2卖出后趋势恶化跟踪 | **失效** — 无跟踪数据 |

**这意味着**: 回测中有止损保护，实盘中完全没有。回测的 Sharpe/回撤数据会明显优于实盘。

**修复方向**: 将 6 个字典序列化到 `portfolio_state.json` (或 `.internal.json`)，在 `PortfolioConstructor.__init__` 中恢复，在 `runner.py` 的每个环节结束时保存。

### 2. 非调仓日触发任意卖出 → 全仓清仓

**根因**: `portfolio.py:1363-1366`

```python
if not rebalance and not stop_loss_sells and not chan_force_sells:
    adjusted = deepcopy(current_positions)  # 保持现有持仓
```

当 `rebalance=False` 但有一个止损触发时，`adjusted` 初始化为空 `{}`。之后的循环只处理 `desired_value` 中的条目。`Recommender` 看到持仓不在返回结果中，生成**全部卖出**建议。

**影响**: 某只股票触发 15% 止损 → 所有 6 只持仓全部被建议卖出。这是灾难性的。

**修复方向**: `adjusted` 始终从 `deepcopy(current_positions)` 开始，然后只覆盖有卖出信号的标的。

### 3. `monitor.save_report()` 是死代码

**根因**: `signal_runner.py:403-405`

```python
return {...}        # ← return 先执行
monitor.save_report()  # ← 永远不会执行
```

监控数据从未被持久化。之前的修复写了方法但调用位置放错了。

---

## 🟡 高优先级（应在上线前修）

### 4. `portfolio_state.json` 无自动更新

**根因**: `runner.py:426` — 打印提示 "执行后请手动更新 portfolio_state.json"。`PortfolioState.update_after_trade()` 方法存在但从未被调用。

**影响**: 每次运行读取的是**过时**的仓位/现金数据（上次手动更新时的值）。如果用户忘了手动更新，系统会基于错误的状态生成信号。

### 5. 89% 的异常被静默吞掉

实盘路径（`strategy/core/` + `trade/`）中 **35 个 try/except 块**，只有 4 个打印了 traceback。其余 31 个是 `pass`、`return None` 或 `return default`。

**关键风险点**:
- `signal_engine.py:2252` — 动态因子选择失败静默返回 None
- `fundamental.py:56` — CSV 读取损坏静默缓存空 DataFrame
- `signal_engine.py:2454` — 基本面过滤异常静默返回 False

**影响**: 实盘中出现数据异常时，信号可能基于错误/缺失数据生成，而没有告警。

### 6. 缺少自动化调度

No cron, no systemd timer, no internal scheduler. 必须每天手动 `python main.py run`。忘了一次 = 错过一天调仓。

### 7. `stock_pool.py` 缺少流动性过滤

文档声称有换手率和成交额过滤，但代码中未实现。ST 检测依赖 `stock_list.csv` 存在（缺失时静默返回空集）。

---

## 🟢 次要问题（可以上线后修）

### 8. Chan 强制卖出 vs 止损卖出行为不一致

`stop_loss_sells` → 立即全清 (line 1373)
`chan_force_sells` → 走 exit_speed 路径，默认只卖 50% (line 1384)

两个都被标注为"强制"，但行为不同。

### 9. `_should_rebalance` 依赖不存在的字段

读取 `last_rebalance_date` 但该字段从未被写入 `portfolio_state.json`。结果：每次运行都触发调仓（因为找不到上次调仓日期）。非调仓日的逻辑从未被测试。

### 10. 状态在卖出前就被清除

卖出逻辑中先 `pop(_entry_dates)` 再执行卖出操作。如果卖出失败（如涨跌停），跟踪信息已丢失。

---

## 📊 评分

| 维度 | 之前 | 现在 | 变化 |
|------|------|------|------|
| 策略设计 | 8 | 8 | — |
| 因子体系 | 7 | 8 | +1 (修复分散化bug) |
| 风控机制 | 7 | **4** | -3 (实盘中止损失效) |
| 回测框架 | 6 | 6 | — |
| 代码质量 | 7 | 6 | -1 (静默异常吞掉) |
| **总分** | 7.0 | **6.4** | **状态下降** |

评分下降不是因为代码变差了，而是因为实盘路径与回测路径的差异被暴露了出来——回测中有效的机制在实盘中因为状态不持久而静默失效。

---

## 🔧 修复优先级（上线前必须完成）

### 今天必须修

1. **持久化 6 个状态字典** — `portfolio_state.json` 增加字段或新增 `portfolio_tracking.json`
2. **修复非调仓日全仓清仓 bug** — `adjusted` 始终从 `deepcopy(current_positions)` 开始
3. **fix `monitor.save_report()` 位置** — 移到 `return` 之前

### 上线前修

4. **自动更新 `portfolio_state.json`** — runner.py 调用 `PortfolioState.update_after_trade()`
5. **关键路径加 error logging** — 至少 `_select_factor`、`_get_fundamental_score` 失败时记录 warning

### 上线后尽快修

6. 添加 cron / systemd timer 自动调度
7. stock_pool 补充流动性过滤
8. 统一 chan_force_sells 和 stop_loss_sells 的退出行为

---

*报告生成时间：2026-05-27*
*审查范围：实盘路径全链路审计*
