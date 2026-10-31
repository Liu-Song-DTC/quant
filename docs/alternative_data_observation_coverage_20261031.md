# 另类数据历史观测边界与市场信号消融实验

`strategy/analysis/audit_alt_observation_coverage.py` 只读当前
`data/alternative_data/alt_versions/*.json` 中的首次观测日；
针对 2021-01-01—2026-10-28 研究窗口，六种缓存（龙虎榜及历史、北向、
两融、减持记录及计划）的最早留痕均为 **2026-10-29**，
即 `6/6` 在该窗口结束前**没有可验证的供应商观测版本**。
本地审计产物：`.review_runs/data_post_refresh_20261029/alt_observation_coverage_20261031.json`。

这里需要区分三件事：数据表的一条事件日期不证明供应商在该日期已提供
这条记录；2026-10-29 之后开始留存版本，使之后的刷新可审计，却
不能追溯证明 2021—2026-10-28 历史版本；
`resolve_asof(..., prehistory_proxy=True)` 在首次留存之前返回第一版，
这是**带事后信息风险的代理口径**，不能标为严格 point-in-time。
调用点里龙虎榜单日信号设置了 `prehistory_proxy=False`，而龙虎榜历史、
北向、两融、减持记录及计划仍默认使用首版代理；各调用点仍按事件日期
过滤，因此不能仅从本报告断定具体历史交易受影响的金额/方向。
2026-10-29 完成的旧封存回测创建于版本清单纳入快照之前，
其 `data/alternative_data/` 没有 `alt_versions/`：**不能事后拿当前版本清单
给那个旧 run 补签“当时严格 PIT”的证明**。未来 run 的快照脚本
会对版本清单及归档内容做指纹。

只有 2026-10-29 以后取得的版本能按实际留存日回放；更早日期沿用
**第一版历史序列按数据日期过滤**，作为用户确定的回测基线。将首次
留存前的北向、两融信号置零只是消融研究，不是真实历史数据的替代品，
也不能用置零臂判断此前实际供应商版本。减持记录和计划属于持仓风险
过滤，不能把“无可核验历史快照”直接当作“历史没有减持”而放松风控。

运行核验：

```bash
.venv/bin/python -B strategy/analysis/audit_alt_observation_coverage.py \
  --root data/alternative_data \
  --start 2021-01-01 --end 2026-10-28 \
  --out .review_runs/data_post_refresh_20261029/alt_observation_coverage_20261031.json
```

## 同一批历史入选日的首版代理影响探针（2026-10-31）

`probe_alt_preobservation_materiality.py` 在**已封存 run 的 161 条入选日期/代码**上，
只读查询**当前留存的 2026-10-29 首版另类数据快照**：北向代理信号
161/161 非零，两融代理信号 143/161 非零（2021 年 18 条均为零）；
减持记录与计划在已入选股票上 0/161 命中，但在 101 个入选日上，
全市场减持风险名单和计划名单都非空。报告为
`.review_runs/data_post_refresh_20261029/alt_preobservation_materiality_20261031.json`。
已入选股票零命中**不能**证明风险过滤未发挥作用：过滤本来就在入选前
发生，必须进一步分析当日入选前候选池。探针把旧入选决策与后补首版
快照结合，仅用于确定研究优先级；它不是对旧 run 当时另类信号的重算，
不产生收益结论，也不证明这些历史事件当时已被供应商观察。

已新增默认关闭的研究开关 `QUANT_ALT_ZERO_PREOBSERVED_MARKET=1`：
仅北向与两融市场信号在首次留存版本之前返回零，其余逻辑（特别是
减持过滤）保持不变。此开关**仅用于研究消融**，不作为切换生产口径的
选优开关；历史按日期读取与后续版本化快照的默认逻辑不变。

## 封存双臂结果（2026-10-31）

运行目录分别为 `.review_runs/alt_preobs_proxy_20261031` 与
`.review_runs/alt_preobs_zero_market_20261031`。两臂 34,171 个封存数据
文件的内容指纹相同（`09f50b1cb532d0c6da0785d1471f039c08bc10906b6bf4d34c7f3048bbc026a3`），
源代码及有效配置指纹相同；各自 `corrected_input_stamp.json` 的
`unchanged=true`，仅置零臂的研究开关在运行环境指纹中存在。

| 指标 | 首版历史序列按日期读取 | 首次留存前两信号置零 |
| --- | ---: | ---: |
| 最终权益 | 882,519.85 | 783,658.68 |
| Sharpe | 1.2242 | 1.1236 |
| 最大回撤 | 21.36% | 16.77% |
| 已平仓净现金流胜率 | 52.80% (161 笔) | 50.96% (157 笔) |
| 成交入场后 1 日扣费盯市胜率 | 50.93% | 48.41% |
| 成交入场后 3 日扣费盯市胜率 | 50.93% | 47.77% |
| 成交入场后 5 日扣费盯市胜率 | 45.96% | 42.04% |

短期标签从实际 BUY 成交成本起算，但期末是盯市收盘价，跌停等无法退出的
样本仍保留，**不代表可成交卖出收益**。置零使收益和胜率降低、最大回撤
减小；它说明策略对市场级偏移有一定敏感性，不能证明历史供应商版本的
真实性，也不能以哪个结果更好决定是否使用历史数据。按用户确定的口径
保持默认按数据日期读取；后续数据刷新继续留存不可变版本。研究到此收口，
优先处理原价成交与公司行动账本。

```bash
.venv/bin/python -B strategy/analysis/probe_alt_preobservation_materiality.py \
  --selections .review_runs/data_post_refresh_20261029/strategy/rolling_validation_results/portfolio_selections.csv \
  --alt-root data/alternative_data \
  --out .review_runs/data_post_refresh_20261029/alt_preobservation_materiality_20261031.json
```
