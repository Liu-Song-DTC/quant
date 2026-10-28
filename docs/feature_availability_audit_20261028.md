# 特征可用时点审计 (2026-10-28, 交接计划 §3 任务 3)

工具：`strategy/analysis/probe_feature_availability_1028.py`（只读，可复跑）
清单：`docs/feature_availability_audit_1028.csv`（62 行 = 配置白名单全量）
数据：封存因子表 `.review_runs/code_repro_1028/strategy/cache/factor_df_5325s_821d_151560b4.parquet`
（3,664,020 行，2020-01 ~ 2026-10，5,325 只）

## 1. 结论：配置声明 62 项，模型实际只见到 30 项

| 项 | 数量 | 证据 |
|---|---|---|
| 配置白名单 `ml.feature_whitelist` | 62 | `strategy/config/factor_config.yaml:7764` |
| 实际进入模型的基础特征 | **30** | 探针复算 + 运行日志 `[ML][FEATURE_CONTRACT] 配置=62 可用=30` |
| **声明但管线从不产出** | **32** | 同上 `缺列=[…]` |

32 项缺列（训练时被剔除、预测时按 `ml_predictor.py:505-510` **零填充**）：
`fund_pe`、`volume_divergence`、`relative_volume_ratio`、`volume_trend`、`overnight_gap_5d`、
`amplitude_5d`、`amplitude_expansion`、`limit_up_proximity`、`limit_down_proximity`、
`volume_price_resonance`、`volume_dry_up`、`dt_signal`、`northbound_signal`、`margin_signal`、
`profit_growth_accel`、`revenue_growth_accel`、`days_since_report`、`earnings_yield`、
`fund_cf_quality`、`debt_ratio`、`log_total_assets`、`log_revenue`、`log_operating_cf`、
`fund_roic`、`fund_net_margin`、`vol_regime`、`turnover_regime`、`market_breadth`、
`market_momentum_ratio`、`market_volatility`、`market_dry_up_ratio`、`market_fear_index`

⇒ **不能说当前模型已使用宏观、财报新鲜度、北向/两融、涨跌停邻近度**。这些名字只存在于
配置里。配置与实现不一致本身就是缺陷：审计/复核者读配置会得到错误的模型信息集。

## 2. 进入模型的 30 项按数据来源分类

| 类别 | 数量 | 特征 | 真实可用时点 |
|---|---|---|---|
| 量价 | 24 | `trend_lowvol` `momentum_reversal` `mom_x_lowvol_{20_20,20_10,10_10,10_20}` `mom_diff_{5_20,10_20}` `momentum_acceleration` `volatility_skew` `low_downside` `skewness_20` `tail_risk` `bb_width_20` `ret_vol_ratio_20` `ema20_slope` `ma_alignment` `trend_mom_v24` `relative_strength` `bb_rsi_combo` `vol_price_breakout` `intraday_ret` `gap_ratio` `V41_RSI_915` | **T 日收盘可知 ⇒ T+1 开盘可用**（`gap_ratio`/`intraday_ret` 开盘/日内可知）⇒ 未发现前视 |
| **量价推断**（名字误导） | 2 | `smart_money_flow`、`earnings_quality` | T 日收盘 |
| 基本面 | 4 | `fund_roe`、`fund_gross_margin`、`fund_cf_to_profit`、`tech_fund_combo` | 报告期 + 披露截止日（季度末+62 天）⇒ 由"数据可用日期"闸控 ⇒ PIT 成立 |

**量价推断的两项必须改名或改口径**（计划 F1「研究资金流须区分真实观测与量价推断」）：

- `smart_money_flow`：`core/capital_flow.py:7` 自述"**量价关系推断**主力意图"，
  实现是 `factor_calculator.py:1279 _calc_smart_money_flow(returns, vol)` ⇒
  **不是**龙虎榜/资金流的真实观测。
- `earnings_quality`：`factor_calculator.py:589` = `inv_turnover × 1/(1+|vol20|×10)` ⇒
  **纯量价代理**，与"盈利质量"无关（计划 §0 已指出，此处给出代码级证据）。

## 3. 缺失结构（决定"缺失如何表示"）

| 特征 | 全期缺失率 | 逐年（2020→2026） | 结构判定 |
|---|---|---|---|
| `fund_cf_to_profit` | **21.7%** | 21.9 / 16.2 / 19.0 / 21.2 / 23.1 / 24.1 / 25.6 % | **结构性缺失**（现金流科目本身缺失），逐年略升 ⇒ 缺失**不是**信息 |
| `fund_gross_margin` | 3.6% | 5.4 → 2.0 % | 覆盖逐年改善 |
| `fund_roe` | 2.1% | 3.7 → 0.9 % | 覆盖逐年改善 |
| 其余 27 项 | **0.0%** | 全 0 | 无缺失 |

- 缺失的表示：因子层写 NaN ⇒ 截面中性化按**当日均值**填充 ⇒ 模型看到的是"当日平均"。
  **缺失与真实值在模型输入里不可区分**（缺失指示位机制已实现但**默认关闭**，
  `ml_predictor.use_fill_indicator`）。
- 已有独立证据（此前实验）：均值填充的作用实为"稀释基本面噪声"（把填充行钉在截面中位），
  去掉填充后 NAV 由 882,520 升到 930,615，但 3 日胜率 −6.2pp ⇒ 填充是**承重件**，
  不能简单删。⇒ 正确的方向是**让缺失可区分**（开启缺失指示位或改为"拒绝/降权"），
  而不是继续用均值填充。

## 4. 未覆盖项（按计划 §3 任务 3 的优先级）

计划点名的"公告后真实可用的财报边际/预告、行业成交额份额与广度、美债/汇率/全市场波动率"
**目前都不在模型的 30 项里**：财报边际对应 `profit_growth_accel`/`revenue_growth_accel`/
`days_since_report`（缺列），行业广度对应 `market_breadth`（缺列），宏观对应
`market_fear_index`/`vol_regime`（缺列）。⇒ 这些是**待新增**的真实可用特征，不是已生效的能力。

## 5. 与本审计相关的失败模式

1. **配置—实现不一致**：白名单 62 vs 实际 30 ⇒ 任何"模型用了 X"的陈述都必须以运行日志
   的 `FEATURE_CONTRACT` 为准，不可以配置为准。
2. **缺列零填充**：预测期若某列整体缺失，会被静默填 0（与"真实预测为 0"不可区分），
   与计划 §5 P1「零值与真实预测混淆」同源；本审计确认该路径存在。
3. **名字误导**：`smart_money_flow`/`earnings_quality` 不是其名字所指的观测量。
4. **基本面缺失不可区分**：结构性缺失率最高 25.6%，模型无从分辨。

## 6. 剩余风险与下一步

- 本审计只覆盖 **ML 的基础特征**；交叉特征（28 项）与状态交互（10 项）由
  `ml_predictor.prepare_features` 在训练时按当日截面排名/regime 生成，
  其时点继承自基础特征 ⇒ 未单独展开。
- 线性/组合评分另消费 10 个因子（见 CSV `in_linear_score=Y`），其时点同样为 T 日收盘。
- 下一步（工作包 F1/M1）：先修"配置—实现不一致"（把 32 项缺列从白名单移除或补齐实现），
  再对新增的真实可用特征（财报边际、行业广度、宏观预警）做**无前视**探针，
  不得凭故事声称领先价格。
