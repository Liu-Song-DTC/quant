# ML 层静态审查 (2026-10-29, 交接计划 §3 任务3 / §6)

方式：**只读代码静态追踪**（未跑回测；10-28 夜间的生产链占着机器）。
清单：`docs/source_inventory_ml_20261029.csv`（8 行，按计划 §6 的字段：跟踪状态/入口调用者/依赖数据/
配置项/缓存键/输入时点/失败语义/对应测试）。

一句话结论：**ML 层的训练与缓存机制比预期扎实（内容寻址的 chunk 缓存、purge、成员校验、
固化预处理都在），但"配置默认值"和"实盘模型来源"两处埋着会静默改变口径的坑。**

## 1. 链路与口径（先确认事实）

```
bt_execution.py:1100-1310  分块滚动训练
  ├ 标签: future_ret 的**截面排名**(percentile−0.5)  ml_predictor.py:335-342
  ├ 训练/验证切分: 日期 80/20 + label_end_date purge   ml_predictor.py:463-472
  ├ 交叉特征对: 用**训练段**标签可得行选 top-8 ⇒ 28 对  ml_predictor.py:101-106,358-359
  ├ 截面 z-score(市场级列豁免, 且同日期跨股不一致直接抛) ml_predictor.py:256-273
  ├ Ensemble=3 种子; val_ic=逐日 Spearman IC 均值      ml_predictor.py:403-445
  └ chunk 缓存: 每边界 K线逐行 md5 聚合 + 基本面/概念 pkl + config节 + 代码指纹 + 运行时版本
                                                        bt_execution.py:371-444
signal_engine.py:1533-1578  融合
  ├ 预测按"当日或之前最近一条"前向填充(searchsorted right−1)  :1549
  ├ ml_normalized = tanh(3·pred); 仅 |·|>0.01 的行参与加权      :1560,1574
  └ require_prediction ⇒ valid &= ml_prediction_available       :1570-1571
```

**PIT 判定**：标签只用未来收益做训练目标、特征只用当日及之前、验证段标签不参与选特征、
purge 用 `label_end_date` ⇒ **训练/预测路径未见前视**。实盘订单路径读的是回测信号 CSV
（`generate_trade_orders.py:59`）⇒ 实盘与回测的 ML 口径**一致**（铁律成立）。

## 2. 发现（按严重度）

### M-1【高】关键口径键的**代码默认值 ≠ 生产配置** ⇒ 配置键丢失会静默换口径

| 键 | 代码默认 | 生产配置 | 丢失后的后果 |
|---|---|---|---|
| `ml.coverage_policy` | `factor_fallback`（`signal_engine.py:410`） | `require_prediction` | **计划 §0 记载的变体：NAV 555,364 vs 882,520（−37%）** |
| `ml.blend_weight` | `0.50`（`signal_engine.py:409`、`ml_predictor.py:72`） | `0.4` | 因子分被稀释，权重曲线单峰在 0.4 |
| `ml.min_val_ic` | `0.03`（`bt_execution.py:1244`） | `0.02` | chunk 准入变严 → 复用链变长 |

配置文件里这三项都有值，所以**当前生产没踩**；但它们是"少一行就变口径、且不报错"的形状。
建议（M1）：把这三项改为**缺失即报错**（fail-closed），或在引擎初始化时断言"生产值"并登记。

### M-2【中】缺列零填充 ⇒ 模型输入里"缺失"与"真实 0"不可区分

- 配置白名单 62 项，**管线只产出 30 项**（`docs/feature_availability_audit_20261028.md`）；
  缺的 32 项在训练时被剔除、在预测期被**填 0**（`ml_predictor.py:502-510`）。
- 训练期特征 NaN 一律 `fillna(0.0)`（`ml_predictor.py:374`）⇒ 缺失=截面均值。
- 已有 `use_fill_indicator` 机制（识别被填格、剔除池外掩码行、加 0/1 指示位），**默认关**
  ⇒ 现在模型**不知道**哪些值是填出来的。
- 已知副作用：填充实为"稀释基本面噪声"，去掉后 NAV 882,520→930,615，但 3 日胜率 −6.2pp
  ⇒ 填充是承重件 ⇒ 正确方向是**让缺失可区分**，不是删填充。
- 建议（M1）：开 `use_fill_indicator` 跑同快照两臂，**三指标齐看**（权益/MDD/3-5 日胜率），
  再按 §1.2 判定；不要只比权益。

### M-3【中】复用链可以无限延续（陈旧模型长期供数）

`bt_execution.py:1245-1274`：验证 IC < `min_val_ic` 时
①无上一有效模型 ⇒ **跳过整个 chunk**（该段日期无预测）
②复用上一有效模型，连续复用计数 < `max_consecutive_reuse`(3) ⇒ 继续复用
③超限且新模型 IC>0 ⇒ 采用新模型（`reuse_forced`）
④超限且新模型 IC≤0 ⇒ **继续复用旧模型且计数不复位**（`reuse_chain_exceeded`）

⇒ 只要新模型一直 IC≤0，一个很旧的模型可以**无限期**供数，且没有任何上限告警。
建议：给"超限仍复用"加硬上限 + 显式告警计数（写进 navlaw/台账）。

### M-4【中】`require_prediction` 下"无预测"= 该股当日被剔除

`signal_engine.py:1570-1571`：`valid &= ml_prediction_available`。若无预测（chunk 被跳过 /
新股未覆盖 / 前向填充未命中），该股当日**直接不可交易** ⇒ 股票池被系统性削掉，且原因不在
数据层。计划 §0 提到"简化覆盖逻辑而不保留该条件"→ 555,364 ⇒ 这条是**口径变量**，不是 bug。
建议（M1）：量化"被覆盖规则剔除的日数×只数"，登记为口径披露项。

### M-5【低-中】regime 交互特征依赖输入列顺序

`ml_predictor.py:151-152`：`self._regime_feature_sources = list(base_features[:10])`
⇒ 参与 regime 交互的 10 个特征由**因子表的列顺序**决定。上游新增/重排一列 ⇒ 静默换掉
这 10 个特征（训练时固化并入库 ✓，但"选谁"本身依赖列序）。建议：改为按**固定名单**取。

### M-6【低】交叉特征对每次重训会漂移

top-8 由训练段 IC 选出 ⇒ 每个 chunk 的 28 对可能不同 ⇒ 模型不可比（已由 chunk 缓存
sha256 绑定，生产上不会混用 ✓），但"跨 chunk 的特征重要性"不可直接横比。

### M-7【低】`ml_active = |tanh(3·pred)| > 0.01` ⇒ 近零预测被当作"无 ML"

真实的小预测被忽略 ⇒ ML 在低信号日不参与打分；副作用上**部分掩盖**了"零预测 vs 缺失"的
混淆（M-2），但也意味着 ML 的影响面比"覆盖率"看起来小。

### M-8【中】实盘模型加载路径是**死代码**，且带着一个陈旧季度模型

`strategy/core/live_init.py:103-125` 加载 `models/xgb_<year>Q<n>.json`（季度模型），
**全仓无调用方**（只有 `bt_execution.py:2` 的注释要求"务必同步更新"）。
- 那些季度模型最后写于 **2026-07-23** ⇒ 若有人复活这条路径，会静默用 3 个月前的模型；
- 更危险：按 `require_prediction`，该路径 `ml_prediction_available` 恒为 False ⇒
  `valid` 会被清空 ⇒ 零信号；
- 且 `bt_execution.py:2` 的注释要求人工同步两份实现 ⇒ 漂移风险。

生产订单路径读回测信号 CSV（`generate_trade_orders.py:59`）⇒ **实盘口径不受影响**。
建议：把 `live_init.py` 标注为死路径或删除（需用户裁定），并去掉那句人工同步义务。

## 3. 测试覆盖

7 个 ML 测试：`test_ml_chunk_identity`（缓存身份）、`test_ml_combination_rules`（融合规则）、
`test_ml_feature_contract`（特征合同）、`test_ml_market_scope`（市场级特征）、
`test_ml_objective_probe`（目标）、`test_ml_preprocessing_contract`（固化预处理）、
`test_ml_signal_validity`（信号有效性）。
**缺口**：M-1（配置默认值漂移）、M-3（复用链上限）、M-4（覆盖剔除的量化）、
M-8（死路径）**无对应测试**。

## 4. 建议的下一步（M1 内，均须同快照两臂 + §1.2 口径）

1. 把 `coverage_policy` / `blend_weight` / `min_val_ic` 改成缺失即报错（M-1）——纯工程，
   零收益影响，先做。
2. 量化"覆盖剔除"与"复用链"的实际影响面（M-3/M-4）——用已封存跑的产物做只读统计即可，
   不需重训。
3. `use_fill_indicator` 的 A/B（M-2）——需要一次全量重训，按 §1.2 走。
4. 死路径处置（M-8）——需要用户裁定。

## 5. 剩余风险

- 本次为**静态**审查：未跑任何臂 ⇒ 上述"后果"沿用计划 §0 已有的实测数字，未新增实测证据。
- 因子表（`future_ret`/`label_end_date`）由因子层生产 ⇒ 标签的时点正确性属**因子层**，
  在下一节单独审。
- 成交层（撮合/涨跌停/T+1/费用）与风险层（敞口/集中度/熊市闸）未在本次范围内。
