# 另类数据历史观测边界：先核证，再做严格实验臂

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

下一步用独立封存数据臂逐项比较首版代理与“首次观测之前缺失”的严格
口径。严格臂需先处理北向、两融等无快照时的 `None` 返回；减持记录和
减持计划属于持仓风险过滤，当前 `PortfolioConstructor` 在数据读取失败时
主动拒绝空数据放行，不能把“无可信历史快照”直接替换成“历史无减持”
而偷偷放松风控。应先取得当期可核验的历史公告来源，或在研究臂中
明确区分未知与空集并展示两种边界，再按实际特征调用记录影响的股票、
日期、入选和 NAV 差异。比较前先封存相同信号输入及全成本成交口径，
不能把未知历史信息直接当作没有该信息，也不能声称严格臂必然提高收益。

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
减持过滤）保持不变。下一步应在**新封存的同一批当前输入**上做
首版代理 vs 置零的双臂完整回测，并明确标记为“市场信号可归因实验”，
而非全另类数据 PIT 证明；两臂分别记录逐日净值、成交、入选、1/3/5 日
胜率及回撤，收益没改善则保持生产开关关闭。

```bash
.venv/bin/python -B strategy/analysis/probe_alt_preobservation_materiality.py \
  --selections .review_runs/data_post_refresh_20261029/strategy/rolling_validation_results/portfolio_selections.csv \
  --alt-root data/alternative_data \
  --out .review_runs/data_post_refresh_20261029/alt_preobservation_materiality_20261031.json
```
