# 工作现场与实验身份锁定 (2026-10-28, 交接计划 §3.1)

对象仓库：`/mnt/d/quant`（主树，出单区）。研究产物在 `.review_runs/`（主树内），
研究区 = worktree `/mnt/d/quant-rs`（分支 `research/uncommitted-arms-1028`）。
`/mnt/d/quant-w1`（`feat/listing-reconciliation`）= P0-a 的对照分支，已被 `3071932` 取代。
**不用** `/mnt/d/quant-r`（10-24 起的旧研究区，detached 于 ff53295，已停用）。

## 1. 现场快照（15:10 更新）

```
git status -sb  →  ## main...origin/main (已同步)
git log -1      →  51ac47d docs: 现场锁定补 P0-c 结果与今晚链口径
```

- 已跟踪未提交：**20** 个文件（**全部在 `data/**` 白名单内**）
- 未跟踪：**536** 项（`.arms/**` 为主 + `data/` 历史脚本）
- **洁净闸：白名单外已跟踪改动 = 0 ⇒ 出单前置条件已恢复**（P0-b 已闭合）

## 2. 改动归属分类（逐文件）

| 归属 | 路径 | 说明 |
|---|---|---|
| 生产数据（刷新产物，二进制） | `data/alternative_data/*.pkl`(8)、`data/concept_hist.pkl`、`data/concept_map_history.pkl`、`data/stock_concept_map.pkl` | 每日刷新产物；白名单内 |
| **数据层脚本（未提交，白名单内 ⇒ 不拦链）** | `data/akshare_kline_downloader.py`、`build_events_table.py`、`fetch_actions_table.py`、`fetch_sina_factors.py`、`freeze_st_timeline.py`、`guard_degenerate_prices.py`、`market_codes.py`、`rebuild_table_qfq.py`、`refresh_all.sh` + 未跟踪 `guard_security_identity.py`、`record_st_status.py`、`audit_security_identity.py`、`audit_fundamental_inputs.py`、`quarantine_security_sources.py`、`update_actions_table.py` | **不回退**：这些脚本正是 10-26 刷新所用的版本，`market_codes.is_stock_code(code, raw_dir)` 会改变重建符号集 ⇒ 回退后下次刷新会产出**不同的价格目录**（改写历史）。晋升（=提交进 main）属 D1 工作包，需先验证自洽与守卫无误报 |
| 研究臂（**已移出生产树**） | 原 13 个 `strategy/**` 文件 + 26 个未跟踪研究模块/测试 | 已原样保存到研究区 `research/uncommitted-arms-1028`（`2e21784` + `7a6aa3c`），主树已恢复为 HEAD |
| 本次交付（P0-a，已提交 `3071932`） | `data/listing_reconciliation.py`、`data/check_listing_reconciliation.py`、`data/probe_listing_gap_1028.py`、`data/stock_data/listing_reconciliation_allowlist.txt`、`strategy/tests/test_listing_reconciliation.py` + `akshare_kline_downloader.py`/`refresh_all.sh`/`rebuild_table_qfq.py`/`check_data_freshness.py` 的增量 | 见 `docs/listing_reconciliation_p0a_20261028.md` |
| 研究台账（白名单内） | `.arms/**` | 不入库亦不阻断 |

**已提交版可独立启动**：`git show HEAD:data/refresh_all.sh` 引用的 `.py` 全部在 HEAD
中被跟踪 ⇒ 不依赖任何未提交/未跟踪文件。

## 3. 正在执行的任务（锁定时）

| 任务 | 位置 | 状态 |
|---|---|---|
| `p0c_committed_1028`：**已提交源码** + 封存数据快照（计划 §3 任务 2） | `.review_runs/p0c_committed_1028/` | **完成**（13:29→14:58）：四份产物 SHA256 与锚逐字节一致；NAV 882,519.8499688671 / Sharpe 1.2255 / MDD −21.36% / 2,343 成交 / 161 已平仓 / 胜率 52.80%；实际敞口与逐日重放差 3.6e-08 ⇒ 详见 `docs/p0c_committed_source_20261028.md` |
| 今晚链（17:40 排程） | `logs/chain_20261028.log` | 已排程：`--force-full` + `NAVLAW_REBASELINE` ⇒ 强制用**已提交口径**重建信号（不复用 10-26 的臂口径信号） |
| `code_repro_1028`：脏工作区源码 + 封存快照 | `.review_runs/code_repro_1028/` | **已停**（ML 4/24）—— 前提（脏源码）已随 P0-b 消失，其结论由三臂审计表覆盖 |
| `arm_base_1026`：当前代码 + 生产实时数据的基线复现 | `.review_runs/arm_base_1026/` | 已完成：NAV **842,623**（≠ 封存锚 882,519.85；数据态不同，不可直接比较） |

⇒ 同一时点只允许一个全量重算（15GB 内存，历史上有过 OOM）。

## 3b. 更正: 10-26 的生产链**没有**被失败臂污染（先前推断有误）

初查曾推断"10-26 06:52 那次 `--force-full` 信号生成用了增长压缩臂代码 ⇒ 生产信号是
失败候选口径"。**该推断被产物级证据推翻**：

| 产物 | 生产(10-26) vs 锚 | 结论 |
|---|---|---|
| `equity_curve.csv` | SHA256 `cd2747c9…` **完全相同** | 逐日权益一致 |
| `trade_realized.csv` | `dd584425…` **完全相同** | 161 笔已平仓一致 |
| `portfolio_selections.csv` | `a8993056…` **完全相同** | 组合选择一致 |
| `regime_state.csv` | `3896d12b…` **完全相同**（旧 4 列形式） | 报表一致 |

⇒ 10-26 06:52 时工作区的代码与锚**净值等价**（臂代码是 10-27 才在研究区搭起来的，
晚于那次链）。**没有发生失败候选进生产的事故**；当时的 navlaw REBASELINE 记录来自
方案A改动包落地（口径变更），不是臂污染。

仍然成立的两点：

1. **前向风险是真的**：如果此刻直接跑链，工作区的臂代码会被用上（增长臂在封存快照上
   NAV 404,336）⇒ 生产树必须先清干净（10-28 已做：研究臂移到 `/mnt/d/quant-rs`）。
2. **今晚仍需 `--force-full`**：理由不是"防污染"，而是 ①生产信号只到 2026-10-23（陈旧，
   明日名单需要 10-28 收盘后的新数据）②生产 `regime_state.csv` 仍是旧的"目标口径"四列，
   新的实际/目标分列要等今晚重生成才进生产；③信号指纹已变（`3d263728|0` → `9975ff9e`），
   本来就会触发重生成，但熊市空仓前置闸会**整段跳过**重生成 ⇒ 必须显式 `--force-full`。

## 3c. P0-b 处置记录（13:20–13:35）

1. 备份：`.arms/backups/uncommitted_1028/tracked.patch`（1494 行）+ 状态清单。
2. 研究区：新建 worktree `/mnt/d/quant-rs`（分支 `research/uncommitted-arms-1028`），
   sparse-checkout `/*` + `!/data/`，`data/` 改为实体脚本 + 共享数据产品的符号链接。
3. 搬入：13 个 `strategy/**` 改动（`git apply --include='strategy/*'`）+ 26 个未跟踪
   研究模块/测试 + `data/` 脚本改动 ⇒ 提交 `2e21784`、`7a6aa3c`。
4. 主树：`git checkout -- strategy/`（13 文件）+ 删除 26 个未跟踪研究文件 ⇒
   `strategy/**` 与 HEAD 逐字节一致。
5. 验证：洁净闸白名单外改动 = 0；全量单测 **126 项全绿**（比之前少 46 项 = 被移走的
   研究测试）；`build_st_filter` 实测 000004 正确判 ST、600519/000001 不判；
   `check_data_freshness` 的上市名册对账在真实数据上 rc=0（含 301718 例外）。

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

## 3d. 今晚链（10-28 17:40）的口径与守门

已排程：`NAVLAW_REBASELINE="…P0-b…"` + `bash strategy/run_daily_live.sh --force-full`
⇒ 日志 `logs/chain_20261028.log`。

必带 `--force-full` 的原因：熊市 + 空仓时前置闸会**跳过信号重生成**（省 80 分钟），
而生产信号只到 2026-10-23（陈旧）⇒ 跳过就没有基于 10-28 收盘数据的新名单；
且新的"实际/目标敞口分列"报表要等重生成才进生产。
`--force-full` 只多跑信号生成，订单仍为空（熊市 + 0 持仓 ⇒ 0 单）。

今晚链会用到两处**未提交**的工作区脚本（data/ 白名单内，当前流水线的实际版本）：

1. `data/refresh_all.sh` 的 **1b 证券身份检查**（`data/guard_security_identity.py`）——
   已在当前数据上实测通过（rc=0），不会误拦；
2. `data/refresh_all.sh` 的 **3c 上市名册对账**（P0-a，本次交付）——工作区版本的允许清单
   路径原写成 `$SD/stock_data/listing_reconciliation_allowlist.txt`（多一层目录 ⇒ 文件
   不存在 ⇒ 301718 会被判缺股 ⇒ 链在换名前中止），**已修为**
   `$SD/listing_reconciliation_allowlist.txt`（与已提交版本一致）。

## 3e. 当前回测口径（2026-10-28 生产产物 = 研究锚，逐字节）

区间 2021-01-04 → 2026-10-23（1,405 交易日），初始 25 万：

| 指标 | 值 |
|---|---|
| 最终权益 | **882,519.85 元**（累计 **+253.01%**，年化 **+25.39%**） |
| Sharpe / Sortino / Calmar | 1.2251 / 1.6051 / 1.188 |
| 全期最大回撤 | **−21.36%**（谷 2024-08-14，峰 2023-07-04） |
| 年化波动 / 日胜率 | 20.14% / 37.94% |
| 逐年收益 | 2021 +56.77%(S 2.47) / 2022 +14.32%(1.04) / 2023 +5.17%(0.42) / 2024 +14.64%(0.74) / 2025 +40.18%(1.52) / 2026 +13.21%(1.08) |
| 组合 | 101 次调仓，平均持仓 1.59 只，top1 平均权重 24.9%，年化换手 19.0 倍 |
| 交易 | 已平仓 161 笔，净现金流胜率 **52.80%**，净盈亏比 1.946 |
| 基准 | 指数年化 +0.94%、MDD −27.27% ⇒ 超额年化 **+24.45%**，beta 0.59，IR 1.18 |
| ML | 3,243,907 条预测全有效，验证 IC 0.0917 |

⚠ 口径边界：概念首版代理（2026-05-31 前借用首版名单）⇒ 这是**工程回归基线**，
不是严格 PIT 收益证明；2021–2026 已被反复用于研究，不能当"未用样本外"。

## 3f. 事故记录: P0-b 搬走研究臂时误搬了生产数据链的依赖 (18:26)

**经过**：17:40 排程的链在**第 1 步就失败退出**（rc=1）：
`data/akshare_kline_downloader.py:301` → `data/record_st_status.py:6`
→ `from core.st_history import record_observations`
→ `ModuleNotFoundError: No module named 'core.st_history'`。

**根因**：P0-b 清理时把 26 个未跟踪文件**全部**当作研究残码搬走，其中
`strategy/core/st_history.py` 其实是**生产数据链的依赖**（下载器 → record_st_status →
core.st_history），而 `data/record_st_status.py` 是未提交的下载器配套件（白名单内，
不拦链）。⇒ 只看"未跟踪 + 在 strategy/ 下"就判为研究残码，漏了跨目录依赖。

**处置**：从研究区 `2e21784` 恢复 `strategy/core/st_history.py` 到主树（未跟踪），
18:29 重启链（`logs/chain_20261028_r2.log`）。
**失败是 fail-safe**：刷新第一步就崩 ⇒ raw/价格目录未被写、信号未重生成、订单未出。

**教训（写进铁律）**：
1. 搬走任何文件前，必须扫**现存代码的 import**（不只是 `strategy/` 内）：
   `data/*.py` 也会 import `core.*`（本次即 `data/record_st_status.py` → `core.st_history`）。
2. 未跟踪文件不等于研究残码：**生产链的未提交依赖必须显式登记**（本次补记：
   `data/record_st_status.py` + `strategy/core/st_history.py` 是数据层未提交件）。
3. 该依赖仍是"未提交 + 主树依赖"的状态 ⇒ 脆（一次 `git clean -xdf` 就会打断链）
   ⇒ 属 D1 晋升项：要么把 ST 观测记录这组一起提交进 main，要么把下载器的
   `record_st_status` 集成撤掉（退回已提交版下载器）。
