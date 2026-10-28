# P0-c 交付: 已提交源码的封存全链复现 (2026-10-28)

按交接计划 §1.6 交付格式；对应 §3 任务 2「在独立封存臂验新成交报表」。

## 问题与成因

10-27 23:06 跑出的研究复现锚（`.review_runs/ml_required_coverage_20261027/`，权益
882,519.85）用的是**当时的源码**；此后当天到 10-28 上午又落了 7 个提交（含
`9ca562b` 覆盖语义显式化、`831fa5b` 实际/目标敞口分列、`45cf979` 补齐模拟成交明细、
以及本轮 P0-a 的 `3071932`）。**这些提交之后的已提交源码从未做过全链封存验证**：
计划 §5 P1 明写「新版本尚未完成全链封存回测；不能把旧研究臂的报表当成新版本的
端到端证据」。⇒ 不知道"已提交版本"是否仍逐字节复现锚。

## 证据路径/版本

| 项 | 值 |
|---|---|
| 候选跑 | `.review_runs/p0c_committed_1028/`（13:29 启动，13:35 冻结源码） |
| 源码快照 | 主树 `strategy/**`（= commit `3071932`；核对：`bt_execution.py`/`core/factor_preparer.py`/`core/ml_predictor.py`/`core/forward_targets.py`/`core/fundamental.py`/`core/signal_engine.py`/`config/factor_config.yaml` 与 HEAD **逐文件 sha256 相同**） |
| 数据 | `data_catalog_quarantine_20261027/data`（sealed_content_snapshot） |
| `data_sha256` | `7b99ea25ba95637c4a425e57f24ee3f46ef2f173c9e268d52bcbd996050cb273` ⇒ **与锚相同** |
| `concept_boundary` | `first_snapshot_proxy_before_first_version` ⇒ 与锚相同 |
| `effective_config_sha256` | `83168b90…`（锚为 `e1c230cb…`）⇒ 差异**仅 2 行注释**（`coverage_policy: require_prediction` 前后加了说明），解析后 YAML 相同 |
| 环境 | python 3.14 / numpy 2.4.6 / pandas 3.0.3 / xgboost 3.3.0；8 workers |
| `production_outputs_reused` | false ⇒ 未复用任何生产信号/预测/指标 |

**源码差异**（锚 vs 本跑，逐文件 sha256）：11 个文件不同，其中**行为相关** 4 个 ——
`bt_execution.py`、`core/factor_preparer.py`、`core/forward_targets.py`、`core/ml_predictor.py`
（今日 7 提交）；另 7 个为测量/门禁/配置注释/测试。

## 同快照全量对照（含失败项）

| 产物 | SHA256 | 结论 |
|---|---|---|
| `equity_curve.csv` | 相同 ✓ | 逐日权益逐位一致 |
| `trade_realized.csv` | 相同 ✓ | 161 笔已平仓逐位一致 |
| `execution_fills.csv` | 相同 ✓ | 2,343 笔模拟成交逐位一致 |
| `portfolio_selections.csv` | 相同 ✓ | 组合选择逐位一致 |
| `regime_state.csv` | **不同**（仅列差异） | 旧 `exposure/n_pos`（**目标**组合）原样搬到 `target_exposure/target_n_pos`（逐位相等）；`exposure/n_pos` 改为**实际成交后**持仓 ⇒ 这正是本次要交付的报表修正 |

指标（用与锚同口径的 `review_metrics` 计算）：

| 指标 | 本跑 | 锚 |
|---|---|---|
| 最终权益 | **882,519.8499688671** | 882,519.8499688671 |
| 年化 / Sharpe / 全期最大回撤 | 25.41% / **1.2255** / **−21.3615%** | 同 |
| 已成交笔数 / 已平仓笔数 | 2,343 / 161 | 同 |
| 已平仓净现金流胜率 | **52.80%** | 同 |
| 佣金 / 印花税 / 价格调整成本 | 25,080.76 / 16,751.07 / 50,162.25 | 同 |
| 年化单边换手 | 8.025 | 同 |
| 净盈亏比 / 剔最佳笔后胜率 | 1.946 / 52.50% | 同 |

⇒ **已提交源码逐字节复现研究锚**；今日 7 个提交（含信号指纹内容化、覆盖语义显式化、
实际/目标敞口分列、成交明细补齐）在封存快照上**净值中性**。

### 实际敞口独立重放（计划 §3 任务 2 的第二项）

`strategy/analysis/audit_executed_exposure.py`（只读，用 2,343 笔成交独立重放现金/股数/敞口）：

```
days 1405 | fills 2343 | max_nav_abs_error 0.0217 元 | min_cash 1,833.80
max_actual_exposure 0.99711 | days_actual_exposure_over_one 0
max_reported_target_exposure 1.49734 | days_target_exposure_over_one 72
max_reported_actual_exposure_diff 3.6e-08
```

⇒ 新 `regime_state.csv` 的**实际**敞口与逐日重放一致（最大差 3.6e-08）；实际敞口
**从未超过 100%**，而**目标**敞口有 72/1405 天 >100%（最高 149.73%）⇒ 旧报表的
"杠杆"是目标口径造成的假象，不是借贷。

## 单测与数据探针

- 主树全量单测 **126 项全绿**（已提交代码，无研究臂）。
- `strategy/analysis/compare_sealed_runs_1028.py`（本次新增）：先整文件 SHA256，
  不同则给行数、首行差异位置、列差异 —— 用于"追第一笔不同信号"而非只看最终权益。
- `strategy/analysis/probe_feature_availability_1028.py` + `docs/feature_availability_audit_20261028.md`：
  ML 特征合同审计（配置 62 ⇒ 实际 30）。

## 实盘一致性检查

- 生产与封存跑同源：`factor_preparer`/`bt_execution`/`ml_predictor` 均为已提交版本；
  生产链 `run_daily_live.sh` 的信号指纹（内容摘要）与封存跑同一算法 ⇒ 生产若复用旧信号
  会被指纹拦住。
- 生产信号当前仍是 **10-26 臂口径**（见 `docs/work_site_lock_20261028.md` §3b）⇒
  今晚链以 `--force-full` 重建，回退到已提交口径。

## 可回滚提交/推送状态

- 交付提交：`3071932`（P0-a）、`f0b0b53`（特征审计 + 现场锁定）、`bc1b20f`（对照工具），
  均已 push 到 `origin/main`（`831fa5b..f0b0b53`）。
- 本跑为只读研究目录，可整体删除；研究臂在 `research/uncommitted-arms-1028`（可回滚源）。

## 剩余风险

1. **锚仍是研究口径**：概念首版代理（2026-05-31 前借用首版名单）⇒ 882,519.85 是
   **工程回归基线**，不是严格 PIT 收益证明（计划 §5 P0-c 未闭合）。
2. **配置用注释区分身份**：锚与本跑的 `factor_config.yaml` 仅差 2 行注释 ⇒ 配置文件
   的**内容哈希**不同（`effective_config_sha256` 不可直接跨跑比对），应改为解析后
   YAML 的规范化哈希。
3. `manifest.json` **未记录 git SHA 与 dirty diff**（计划 §5 P1 要求）⇒ 本次用
   逐文件 sha256 人工核对源码身份；工具应补这两个字段。
4. 全链封存验证只在**这一份快照**上做过；换数据窗口（首版以后严格区间、纸面交易）
   才能回答投资有效性问题。
