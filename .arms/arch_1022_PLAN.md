# 架构面 arch_1022 — 底层架构优化计划 (2026-10-22)

用户裁定: "继续寻找可能的优化点，可以从底层架构开始"。
本文件只记录**架构层**缺口与优先级; 旋钮面/因子面已闭合项不在此重开。

---

## 0. 已发现并量化的架构缺口

### A. 实盘组合状态跨日丢失 (已定位, P0)
`PortfolioConstructor` 是有状态类, 类里已有 `save_tracking_state/restore_tracking_state`,
但现行 `generate_trade_orders.py` 每次进程只注入 `_last_rebalance_date`, 其余全丢:

| 状态 | 回测 | 实盘(旧) | 生产绑定面 |
|---|---|---|---|
| `current_exposure` 平滑 | 跨日连续 | 每日回 1.0 | **死变量(2026-10-22 查证): 1663-1665 只写不读, 权重全部用 `target_exposure`。臂 sm70/sm85 逐位=基线已证实** |
| 波动率降仓(20日窗) | 每日喂收益 | `_daily_returns` 空 | 16/140 调仓日绑定 (均值 0.979, min 0.75) |
| 组合止损(peak/drawdown) | 连续 | peak 每日重置, dd≡0 | 20/147 调仓日绑定 |
| 连亏熔断 CLB | 连续 | `_prev_equity=None` | 全史 0 次触发 (惰性, 但必须保持同构) |
| 入场日期/min_hold | 连续 | 空 | 依赖 `_entry_dates/_position_entry_dates` |

近似反事实探针 (`.arms/probe/livestate_cf_1022.py`, 等比账簿近似, 非正式臂):
- ~~只复现平滑重置: +261k~~ **作废(2026-10-22 10:05)**: 探针把 `current_exposure`
  当成实际敞口, 但代码里它只写不读(死变量); 正式臂 sm70/sm85 逐位=基线已反证.
- 全部 live 语义 (平滑重置+控制失效): **1,403,992** (+420k) — 其中的平滑成分作废,
  只剩 vol/stop/CLB/入场日期通道, 以 livestate 臂为准.
- 仅组合止损: 982,877 (−1.5k, 近中性); 仅波动率降仓: 1,095,517 (+111k, 但 tvol 轴已闭合)
⇒ 实盘状态缺口**不在平滑**; 正式结论以 livestate 臂为准。

### B. 实盘资本不复利 (已定位, P0)
旧实盘: `total_equity = 静态 --cash(50万)`, `cash = 50万 − Σ持仓['amount']`,
`amount` = **下单时市值**, 之后从不更新. 后果:
1. 浮盈浮亏不进入目标仓位分母; 2. 平仓只按 `amount` 回款, 已实现盈亏丢失;
3. 账户涨 ~50% 后 1.5× 静态资金守卫误报.
回测: `nav = cash + Σshares×close` 复利. 这是"实盘=回测"的资本维度缺口.
本机生产回测 250k→984k (3.94×), 实盘若不复利, 目标仓位占真实账户比例会持续下降.

### C. 0.98 目标敞口闸缺失 (已定位, P0)
回测 `bt_execution.py:2459-2469`: 目标持仓总市值 > 98%×(现金+持仓市值) 时**等比缩放**.
实盘旧路径只在买单层用 100% 可用现金兜底 ⇒ 满仓时多 2pp 且不约束目标总市值.

### D. 执行层重叠: 日内 stop/tp overlay (P3)
实盘 `trade_orders.json` 输出 `stop_loss_price/take_profit_price` 给 miniqmt 盘中执行,
回测只按日线 open/close 出场. 这是实盘独有的一条决策路径, 无回测对照;
需要明确"overlay 是执行保护还是策略的一部分", 否则实盘收益不可复现.

### E. 分段建仓 tranche (P1, 候选)
`tranche_enabled=false` 已有实现 (bt_execution 7825 附近 yaml + 2075/2293-2393 代码),
交易级证据: 限价成交 98% 更优, 3/5/8% 折价下 +1.84~4.17pp 半仓增益; 从未组合层验证.
实盘路径未实现. 属执行层, 不改信号; 但改 `bt_execution.py` 会触发信号指纹失效,
需安排一次 ~80min 重生成 (或先做不改文件的 env 钩子方案, 见 P1).

### F. 现金收益/逆回购 (P4)
`repo_sweep` 只给建议, 不进入净值; 回测无现金利息. 量级小 (年化 ~1-2% × 闲置比例),
但"实盘=回测"要求明确口径: 建议把现金收益记为**回测外增量**, 不并入策略净值.

---

## 1. 优先级与动作

### P0 (已完成代码, 待实盘首日验证)
1. `core/live_account.py`: 账户台账 `live_account.json`
   - `nav(t)=cash+Σshares×close`; 日收益喂 `portfolio.update_returns`;
   - 成交按 订单价±slippage + commission/stamp 估算记账;
   - 同日重跑拒绝二次记账 (`--force-rerun` 仅重写订单);
   - 历史 `--date` 写保护 (目标日≠信号最新日 ⇒ 拒绝非 dry-run 写入).
2. `generate_trade_orders.py`:
   - `restore_tracking_state/save_tracking_state` 接入 `live_portfolio_state.json`;
   - `total_equity=NAV`, `cash=台账现金`, `current_positions` 用**收盘市值**;
   - `amount` 统一为最新收盘市值;
   - 0.98 目标敞口闸 (与回测同构);
   - 脏数据守卫改按上期 NAV, 保留 `--force-reset`.
3. 待办: `save/restore_tracking_state` 补 `_position_entry_dates`/`_selection_history`
   (min_hold/rank_decay 状态; 当前类未持久化).

### P1 分段建仓 tranche 组合层臂
- 先加 env 钩子 `QUANT_TRANCHE=1` (不改 yaml, 避免 backtest 节进指纹),
  跑 1 臂; 采纳线: ΔNAV≥150k + 逐年同号 + 胜率不劣化 + ex-top 仍胜.
- 采纳后再实现实盘限价二段.

### P2 敞口平滑面判读 (arch_1022 臂)
- 面 w∈{0.15,0.50,0.70,0.85,1.00} + smonly/livestate 诊断;
- 若 sm100/smonly 显著优于生产, 说明"平滑"本身在拖收益; 但注意 tvol 轴闭合的教训,
  需逐年同号 + 胜率 + ex-top 全过, 否则不动生产.

### P3 日内 overlay 口径
- 量化 overlay 触发率/收益贡献 (需要日内或次日高低数据); 明确是否保留.

### P4 现金收益口径
- 只记账、不进策略净值; 或在 live_account 单列 `cash_yield`.

### P5 信号存储/磁盘
- RVD 下 ~75GB 历史 `backtest_signals.csv.<tag>` 副本; 建议裁 10/17 前 (~45GB),
  **需用户批准** (删除实验证据). 不在本次自动执行.

---

## 2. 预注册判据 (所有正式臂)
- id0 逐位复现 `984331.4340947575 / 350208d1`;
- "复用已有信号" + 信号 CSV mtime/md5 不变;
- ΔNAV ≥ +20k 才判读, 采纳线 150k (噪声地板);
- 逐年同号或单年 |Δ|<60% 总 Δ;
- 短(1/5/20日)长(月/年)胜率都不劣化, McNemar 只算不一致期;
- ex-top: 剥离臂独有 top1 贡献后仍 > 基线;
- Jaccard: 纯权重轴应=1.000; 状态轴可变.
