# 2026-09-30 执行手册 (20天程序收口日, 2026-09-21预写)

当日三件套: **锚点重对账 → V2 OOS审计 → Q4标定**, 之后恢复实盘流。
全程串行, 全程 `QUANT_ALT_NO_AUTOREFRESH=1`, 所有运行显式 `/mnt/d/quant/.venv/bin/python`。

## 0. 前置 (用户侧, 9/30收盘后)

1. Windows下载器拉到 2026-09-30 (全量K线+另类数据)。
2. `cd /mnt/d/quant && ./data/refresh_all.sh` (或等价) — 受控更新, 运行中零自动刷新。
3. `cd strategy && /mnt/d/quant/.venv/bin/python check_data_freshness.py` — 门禁必须过。
4. 预期变化: 数据态 9/17 → 9/30, 锚点必然漂移 (池滑动+历史更正双因素, 先例: 每次刷新
   −5万~−13万级漂移), 漂移必须归因, 不默认接受。

## A. 锚点重对账 (约90min全链, 一次)

```bash
cd /mnt/d/quant/strategy
QUANT_ALT_NO_AUTOREFRESH=1 /mnt/d/quant/.venv/bin/python bt_execution.py
```
- 新锚点四指标 vs 9/17态 **732,689/193.08%/1.1961/17.92%** (fp 6f1cb6b1|0)。
- **期望分解 (9/22预登记)**: 9/30总漂移 = 已知成分 + 数据刷新残差 —
  | 成分 | 期望 | 来源 |
  |---|---|---|
  | A.5印花税 | −32.3k (−4.41%±1%) | 反事实700,382跨进程双复现(9/21-22) |
  | 池迁移(9/30月边界) | ≈0 (新入4/退出6, 留存99.9%) | 9/22预估计探针(代理下界) |
  | 数据刷新9/17→9/30 | 先例−50k~−130k | 归因三件套量化 |
  | **期望带** | **≈570k~650k** | 组合 |
  裁决: 实际锚点落在期望带外 OR **扣减已知成分后的残差** >±3万 → 归因三件套
  ①pool_flip_report ②K线重拉检查 ③另类数据diff — 参照 9/14晚流归因协议
  (单日期全截面raw重算+中性化复现)。±3万阈值只作用于残差, 不作用于总漂移
  (印花税−32k是诚实修正, 非漂移)。
- 产出新sidecar fp; 归档 equity_curve 为9/30基线。
- **分析件同步重生成 (9/22发现stale隐患)**: validation_results.csv 曾自9/6起
  stale (9/17/9/20流未重跑signal_validator) — 9/30全链后必须重跑
  `analysis/signal_validator.py` + `analysis/analysis_framework.py`, 否则分析层
  读的是旧信号态的概念归属/factor_name。

## A.5 成本修复 (9/30重锚顺带, 成本审计9/21-22裁决)

在A的全链跑之前落码(印花税唯一factual项, 其余待成交反馈):

1. **印花税分档 (马上做, 逐行补丁配方)** — probe_stamp_tax 探针已验证的 `_STAMP_ARR[i]`
   补丁永久化 (恒等烟测逐位复现732,689; 反事实700,382跨进程两次逐位一致)。
   三处编辑 (bt_execution.py, 行号=9/22代码态, 漂移则按内容定位):
   - **a. 插入分档数组** — line 1526 `n_dates = len(calendar)` 之后插入:
     `_STAMP_ARR = np.where(calendar.values < np.datetime64('2023-08-28'), 0.001, 0.0005).astype(np.float64)  # 印花税分档: 2023-08-28前万10后万5 (9/22成本审计)`
     对齐自证: 循环 `for i in tqdm(range(n_dates))` 的 i 与 calendar 同序
     (line 1656 `date = calendar[i].date()`); np 已导入 (line 1695 在用 np.isnan)。
   - **b. 替换两处卖单结算** — line 1699 与 1731 的
     `(1.0 - COMMISSION - STAMP_TAX - impact)` → `(1.0 - COMMISSION - _STAMP_ARR[i] - impact)`
     (全文件 STAMP_TAX 引用共2处, 均在 _vectorized_backtest 内, 替换后模块级常量闲置)。
   - **c. 常量注释+删死代码** — line 44 `STAMP_TAX` 注释改为
     "万5扁平默认 — 9/22成本审计后由 _vectorized_backtest 内 _STAMP_ARR 日期分档取代,
     仅作yaml兼容读取"; line 45 `EFFECTIVE_COMM = COMMISSION + STAMP_TAX * 0.5` 整行删除
     (定义后零引用, 注释"买入万1"与万5实测口径矛盾)。
   预期新锚点 ≈ −4.4%±1% vs 732,689口径 (算术−1.45% + 2022熊窗乘数)。
   **这不是策略劣化而是诚实修正, 不得触发"回滚"裁决**; 归因时把此成分从数据漂移
   中分解出来 (参照成本审计响应面: 扁平万10→万5 = +6.1%的反方向同构)。
2. **佣金 (待成交反馈)**: 万5 → 用户实际费率 (fill_cost_feedback报告的佣金gap
   给出实测值; 9/30前无成交流水则按万2中心或暂缓, 不得拍脑袋)。
3. **滑点 (待成交反馈)**: 保持万10, 待D.5反馈回路积累后在下一次重锚校准
   (不对称分解探针9/22的卖/买侧弹性定校准优先序)。
4. fp变化→信号重生成与数据刷新共用同一次全链 (零额外代价, 见A)。

## A.6 架构审计落地项 (9/22 yaml×census对账裁决, 可选, 与A.5同窗口)

审计文档 `analysis/yaml_census_audit_20260922.md`。裁决: 架构合理, 无alpha臂;
以下全部**零行为影响**修复, 与A.5共用同一次fp变更窗口:

1. **季度诊断可观测性**: `_QUARTER_DIAG`计数在4-worker多进程内累积后随进程消亡,
   bt_execution.py:1170主进程打印恒空 → 生产日志从未出现"行业名查找"行。
   fix: worker聚合diag传回主进程(共享Queue或每worker落临时文件聚合)。
2. **死键注释归档**: 9个死yaml键+6个死旋钮(审计文档§1 A/B组)加注释"已归档9/22",
   不删yaml键(指纹内节)。
3. **双源参数合一**: `indicator_params` yaml节 vs `get_default_params()`硬编码
   二选一(值已逐位一致, 建议删硬编码统一读yaml — 或反之, 只留一处)。
4. **日志语义注释**: "固定行业因子49.5%/固定默认因子50.5%"是计数器语义
   (入口计数vs P0命中), 非覆盖度; 真实兜底率0.6%。加注释澄清防误读。
5. **勿做**: P0中性/牛市切季度权重 — 已探针否决(IC配对差−0.0175, t=−5.4)。
   勿做: IC预计算门控 — 已证伪, 代码早已在`if use_dynamic`内正确门控。
   勿做: λ-blend权重/牛熊分支去分支 — 探针9/22: blend在严格中性日纯噪声
   (t=−0.37), 牛/熊分支在2026干净窗生产配置全占优(审计§7)。

## B. V2 OOS审计 (只读, 先于Q4标定)

```bash
cd /mnt/d/quant/strategy
QUANT_ALT_NO_AUTOREFRESH=1 /mnt/d/quant/.venv/bin/python analysis/v2_oos_runbook_0930.py
```
- Phase 0 前提自检 → Phase 1 尾部因子值 → Phase 2 Q3 IC审计 → Phase 3 标定重核
  → Phase 4 Q3实现审计 → Phase 5 决策清单 (不自动跑全链)。
- 读数纪律: **按年分解** (F3/F4教训: 时段集中优势不可外推); 加权IC与符号一致率
  双口径 (9/17早读P2'模板); P3'标定diff按数据修订分级 (良性修订 vs 权重翻转)。

## C. Q4标定 (写盘, 在B之后)

```bash
cd /mnt/d/quant/strategy
QUANT_ALT_NO_AUTOREFRESH=1 /mnt/d/quant/.venv/bin/python analysis/calib_2026Q4_tail.py
```
- 只写 2026Q4.yaml + index.yaml 追加条目, 其余23个季度文件零扰动。
- **焦点纪律(9/22审计探针裁决)**: 季度标定的真实增益=新概念覆盖(仅季度概念
  季度权重IC +0.2423 vs 默认+0.0089, t=5.07) + 熊分支 + 概念解析gating;
  存量交集概念的中性/牛市权重重标定不被消费且IC更差 — 标定时优先新概念。
- **PIT gate + 薄样本回退 (9/22覆盖预检, 已在calib_2026Q4_tail.py内)**:
  标定估计样本按 concept_inception gated (与生产消费一致 — 中报4概念此前96%
  样本在概念成立之前=前视归属稀释); gated天数<120的概念回退2026Q3权重
  (窗口1193日的10%; Q3权重经23日OOS验证+0.2423, 薄样本重标定方差高且无
  OOS余量)。**9/22干跑实证(阈值60): 4/2026年份概念(gated 101-119日)拿到fresh
  重标定且坍缩为volatility单因子/整因子被弃 = 噪声签名; 120纳入Q3先验。
  9/30投影: 中报4≈56日/一季报预增≈116日/电池技术≈128日 → 120边界=10%份额
  处。** 回退/保留清单脚本内打印, 核对后入档; 写盘后验证器复核<120天概念
  无fresh权重。
- **写盘后跑预检** (q4_calib_preflight_20260930.py, 只读): 覆盖结构/覆盖缺口
  (Q3 buys概念无配置者将落P3兜底)/bear刷新量/因子宇宙校验 — 四张表入档。
- **写盘后必须再跑一次全链裁决** (E-D2纪律: quarterly_factors 在信号指纹内,
  信号必重生成): 四指标 vs 9/30基线对账, 铁律裁决 — 采纳或硬回退全局权重。

## D. 实盘流恢复 (用户核对后)

1. **已知阻塞 (9/17遗留)**: current_positions.json 为 21只/993,544 > 1.5×50万 →
   0h现金闸拒绝出单 (设计行为)。用户必须核对真实账户持仓后二选一:
   手动修正持仓文件 / 确认无真实持仓后 `--force-reset`。**此步不可跳过, 不可代核**。
2. 出单: `QUANT_ALT_NO_AUTOREFRESH=1 /mnt/d/quant/.venv/bin/python generate_trade_orders.py`
3. 逆回购sweep激活决策 (预备已落地 ff49580, 默认关闭):
   `live_monitoring.repo_sweep.enabled: false → true` — 闲置现金≈56.9% → GC001隔夜
   ≈0.91% NAV/年; QMT侧以实际可用余额为准, 股票单全部成交后最后下 (卖出204001.SH,
   1000元整数倍)。建议先观察1-2个交易日的出单-成交对账再开启。
4. 出单后核对: trade_orders.json vs 账户实际成交 (价格滑点/部分成交/涨停买不进)。
5. **成交成本反馈 (每日, 成本审计裁决落地件)**: QMT成交流水导出CSV后跑
   ```bash
   /mnt/d/quant/.venv/bin/python strategy/tools/fill_cost_feedback.py --fills 成交.csv
   ```
   逐笔对齐 raw none.csv 开盘基准 → 真实滑点/佣金/印花分档差额 + slip_rate建议
   (p90×1.5缓冲) + 响应面NAV映射。样本积累后, 在下一次自然重锚窗口按报告建议
   校准 yaml slippage/commission (与印花税修复同一窗口, 零额外代价)。

## E. 对账检查点清单

- [ ] fp sidecar 与当前代码指纹一致 (bt_execution._signal_code_fingerprint)
- [ ] 数据态=9/30 (数据新鲜度门禁过)
- [ ] 9/30基线四指标+年份分解入档 (MORNING_REPORT模板)
- [ ] Q4标定后四指标裁决入档
- [ ] 持仓文件与真实账户对账闭环 (D.1阻塞解除)
- [ ] 首周成交反馈回路跑通 (D.5, 每日QMT流水→fill_cost_feedback)

## 回滚预案

- Q4标定裁决失败 → 硬回退全局权重 (E-K1先例: 原22季度文件零扰动, 7/1前净值逐位一致)。
- 9/30数据异常 → 数据态回退9/17, 按 9/14隔离重建协议处理, 不在此手册范围自动执行。

## F. 排练记录 (2026-09-21 冷跑, 数据态9/17冻结)

全链路冷跑排练完成, 两条路径验证, 零副作用 (trade_orders.json/current_positions.json
mtime未动, fp sidecar 6f1cb6b1|0 未动, 无新strategy改动):

1. **新鲜度门禁路径** (期望拒绝): `generate_trade_orders.py --dry-run` →
   正确拒绝: K线max 2026-09-17 < 期望≥2026-09-18, 2个硬指标滞后 — 9/30用户
   下载+refresh_all.sh后此门自动解除。
2. **跳过门禁全上游排练**: `--dry-run --skip-freshness` →
   SignalStore 6,658,619行 / 池5188文件→5126成员 / ST排除161 / 科创板排除595 /
   regime=0 — 全链健康, 无隐藏地雷。
3. **0h闸拒单路径** (期望拒绝, 与用户9/30所见一致): 拒单诊断精确输出
   "持仓成本合计993,544 > 1.5×账户资金500,000" + 二选一指引(手动修正持仓文件 /
   --force-reset)。→ D.1阻塞在9/30仍会以同一方式触发, 用户核对真实持仓后解除。
4. 排练结论: 9/30唯一人因前置=用户下载数据+核对持仓文件; 系统侧全链已验证。
