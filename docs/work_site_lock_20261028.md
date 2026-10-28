# 工作现场与实验身份锁定 (2026-10-28, 交接计划 §3.1)

对象仓库：`/mnt/d/quant`（主树，出单区）。研究产物在 `.review_runs/`（主树内），
隔离实现用 quant 自身的 worktree `/mnt/d/quant-w1`（分支 `feat/listing-reconciliation`）。
**不用** `/mnt/d/quant-r`（10-24 起的旧研究区，detached 于 ff53295，已停用）。

## 1. 现场快照

```
git status -sb  →  ## main...origin/main [ahead 3]
git log -1      →  042ccb4 Expand end-to-end quant review with reproduced data gate gap
```

- 已跟踪未提交：**33** 个文件
- 未跟踪：**560** 项
- 洁净闸（`strategy/run_daily_live.sh` 第 0 步）：白名单外已跟踪改动 ⇒ **拒绝出单**；
  白名单 = `data/**`、`.arms/**`、`factor_config.yaml` 仅 todate 行。
  ⇒ **当前主树按每日链会被洁净闸阻断**（这是 P0-b 的实质）。

## 2. 改动归属分类（逐文件）

| 归属 | 路径 | 说明 |
|---|---|---|
| 生产数据（刷新产物，二进制） | `data/alternative_data/*.pkl`(8)、`data/concept_hist.pkl`、`data/concept_map_history.pkl`、`data/stock_concept_map.pkl` | 每日刷新产物；白名单内 |
| 研究/门禁强化（**上一模型未提交**） | `data/akshare_kline_downloader.py`、`data/build_events_table.py`、`data/fetch_actions_table.py`、`data/fetch_sina_factors.py`、`data/freeze_st_timeline.py`、`data/guard_degenerate_prices.py`、`data/market_codes.py`、`data/rebuild_table_qfq.py`、`data/refresh_all.sh` + 新文件 `data/guard_security_identity.py` | 证券身份守卫（交易所限定 quote id、响应自校验、退市码不再用数字黑名单）；**已提交的 refresh_all.sh 不含这道守卫** ⇒ 已提交版自洽但缺守卫 |
| 研究（ML/基本面/评分，未提交） | `strategy/bt_execution.py`、`core/factor_calculator.py`、`core/factor_preparer.py`、`core/fundamental.py`、`core/signal_engine.py`、`generate_trade_orders.py`、`snapshot_data.py`、`analysis/*`(5)、`tests/test_review_regressions.py`；未跟踪新模块 `core/fundamental_freshness.py`、`core/fundamental_units.py`、`core/price_frame.py`、`core/st_history.py`、`core/calibration_windows.py` + 12 个新测试 | 与"单位修正臂/统一评分臂/增长压缩臂"对应（研究候选，**不得直接进生产**） |
| 本次交付（P0-a） | `data/listing_reconciliation.py`、`data/check_listing_reconciliation.py`、`data/probe_listing_gap_1028.py`、`data/stock_data/listing_reconciliation_allowlist.txt`、`strategy/tests/test_listing_reconciliation.py`、以及 `data/akshare_kline_downloader.py` / `data/refresh_all.sh` / `data/rebuild_table_qfq.py` / `strategy/check_data_freshness.py` 的增量 | 见 `docs/listing_reconciliation_p0a_20261028.md` |
| 研究台账（白名单内） | `.arms/**`（491 项未跟踪 + winrate_PLAN.md 已跟踪） | 不入库亦不阻断 |

**已提交版可独立启动**（P0-b 的前置）：`git show HEAD:data/refresh_all.sh` 引用的 14 个
`.py` 全部在 HEAD 中被跟踪 ⇒ 不依赖任何未提交/未跟踪文件（缺的是证券身份守卫）。

## 3. 正在执行的任务（锁定时）

| 任务 | 位置 | 状态 |
|---|---|---|
| `code_repro_1028`：当前工作区源码 + 封存数据快照的重算 | `.review_runs/code_repro_1028/` | 运行中（因子计算 57%） |
| `arm_base_1026`：当前代码 + **生产实时数据**的基线复现 | `.review_runs/arm_base_1026/` | 已完成：NAV **842,623**（≠ 封存锚 882,519.85；数据态不同，不可直接比较） |

⇒ 同一时点只允许一个全量重算（15GB 内存，历史上有过 OOM）。

## 4. 三臂审计表（§3.1 的"源码差异—配置差异—缓存—交易差异"）

工具：`.arms/probe/arm_audit_table_1028.py`（可复跑）。

| 臂 | 目录 | 权益 | Sharpe | MDD | 笔数 | T+3 | T+5 |
|---|---|---|---|---|---|---|---|
| 锚（基线） | `ml_required_coverage_20261027` | 882,519.85 | 1.2255 | −21.36% | 161 | 50.93% | 45.96% |
| 统一评分 | `fundamental_score_contract_20261027` | 861,569.19 | 1.3485 | −16.98% | 162 | 45.06% | 40.12% |
| 增长压缩 | `fundamental_score_growth_contract_20261027` | 404,336.48 | 0.5472 | −31.78% | 169 | 47.93% | 39.64% |

**身份**（manifest，三臂一致）：

| 项 | 锚 | 统一评分 | 增长压缩 |
|---|---|---|---|
| 数据源 | `data_catalog_quarantine_20261027/data`（sealed_content_snapshot） | 同左 | 同左 |
| data_sha256 | 7b99ea25… | **同左** | **同左** |
| concept_boundary | first_snapshot_proxy_before_first_version | 同左 | 同左 |
| effective_config_sha256 | e1c230cb… | 484c608f… | **484c608f…（同统一评分）** |
| 源码差异（相对锚） | — | 27 改 / 128 增 / 122 删 | 同上 + `core/fundamental_units.py` |

⇒ **增长压缩臂相对统一评分臂的唯一差异在代码**（`core/fundamental.py`、
`core/factor_calculator.py` + 新模块 `core/fundamental_units.py`），**不在 yaml 配置**
（两者 effective_config_sha256 相同）⇒ 属单变量臂 ✓；但它把权益打到 404,336
⇒ **按计划 §0 视为失败候选，禁止合并生产** ✓（已遵守：未进主干）。

**交易差异**（按 (code, entry_date) 对齐）：

| 臂 | 笔数 | 仅锚有 | 仅本臂有 | 共有但收益不同 |
|---|---|---|---|---|
| 统一评分 | 162 | 71 | 72 | 76 |
| 增长压缩 | 169 | 121 | 129 | 38 |

⇒ 两个臂都**重排了约四成选股**（不是边际改动）；增长压缩臂的换手更高、尾部更差
（最深回撤 −31.78% 落在 **2026-09-28**，峰在 2026-05-11 ⇒ 崩在最近一段）。

**回撤归因**：锚 −21.36% @2024-08-14（峰 2023-07-04）；统一评分 −16.98% @2023-06-08（峰
2023-02-15）⇒ 它的"回撤改善"来自**不同的回撤窗口**，不是同一窗口里的改善；
增长压缩 −31.78% @2026-09-28 ⇒ 失败集中在最近一段。

**结论**：两臂都不满足"整体目标"（权益或胜率倒退）⇒ 只能作为负结果登记；
锚 882,519.85 继续作为工程回归基线（不是严格 PIT 收益证明，见计划 §5 P0-c）。
