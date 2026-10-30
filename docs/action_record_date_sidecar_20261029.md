# 股权登记日：单独保留，不改变现行前复权

项目使用的 `akshare.stock_fhps_detail_em` 响应已经带 `股权登记日`，但原抓取只缓存 `(code, ex_date, s, D)` 并将它聚合为价格事件，丢弃了登记日。研究原始价格、现金股息和送转时，必须能区分持有人权益登记日与除权日。

新抓取会为**成功获取的非空响应**额外保存 `data/alternative_data/.ca_record_parts/{code}.pkl`，每行 `(code, ex_date, record_date)`。原 `.ca_parts` 结构、`corporate_actions.pkl`、融合 `events_table.pkl` 和 QFQ 递推不变。日期列缺失、登记日晚于或等于除权日、文件保存失败时，侧车记录缺失并打印告警；现行价格事件照旧。上游随后修订已记录日期会告警并保留首次观测，新补充的事件追加；旧日期此前未知而新查到时补上。

**覆盖率当前未知**：原有缓存只有四字段，新侧车在下一轮每只股票实际从接口成功获取时逐步产生；不能把没有侧车的事件按默认日期填充。不能把这个字段作为选股特征，因为它可能来自回溯下载，缺少当时公告的可信发布时间。下一步应按封存持仓核对 21 条除权事件各自的登记日来源、红利实际可用时点、送转可交易日期，补齐后才能验算原始成交实验臂；配股、特殊分派和新浪只提供乘数的事件仍要另行处理。

## 当前封存回测覆盖率（只读）

按 `strategy/rolling_validation_results/trade_realized.csv` 与 2026-10-27 封存 `events_table.pkl` 核对：161 笔已平仓交易里 20 笔跨 21 条除权事件；21 条的现金/送转比例均可分解，但登记日侧车尚未建立，**0/21 有可核验的登记日，21/21 待补**。这不是 21 条已确认没有登记日，而是旧抓取缓存把上游字段丢弃了。用 `strategy/analysis/audit_action_record_coverage.py` 复查，输出 JSON 和逐事件 CSV；补取后要用相同封存事件快照复查，不能把新事件与旧回测混为同一基线。

```bash
.venv/bin/python -B strategy/analysis/audit_action_record_coverage.py \
  --trades strategy/rolling_validation_results/trade_realized.csv \
  --events .review_runs/data_catalog_quarantine_20261027/data/alternative_data/events_table.pkl \
  --record-parts data/alternative_data/.ca_record_parts \
  --out /tmp/quant_action_held_event_coverage_20261029.json
```

## 公告逐事件校核示例

深交所披露的 [300953 震裕科技 2025 年度权益分派实施公告（2026-05-20）](https://disc.static.szse.cn/download/disc/disk03/finalpage/2026-05-20/09250f51-0f0e-40fb-a2a8-565be42f46ba.PDF)：股权登记日 2026-05-27，除权除息、现金红利发放、无限售转增股份上市日均为 2026-05-28；参与分配股东每股现金 0.30 元、每股转增 0.4 股。公告为**市场除权参考价**按含回购股份的总股本折算为现金 0.2984469 元/股、转增 0.3979292 股/股。股东权益金额与交易所参考价公式是两个不同口径，不可把后者的比例直接记入持仓现金和股数；也不能因为公告写「红利发放日」就断言当日开盘时券商账户已有可买入现金。此例仅人工核实一条事件，**不代表 21 条全部已核实**。

沪市 [603444 吉比特 2025 年度权益分派实施公告（2026-05-14）](https://dataclouds.cninfo.com.cn/shgonggao/hsomarket/2026/20260513/4c79a3df94f64cef90ff8ffeb1741db7.PDF) 也出现两套数：登记日 2026-05-20、除息和现金红利发放日 2026-05-21；参与分红股东每股获现金 7.00 元，而因回购股份不参与分红，**市场除息参考价**的每股摊薄额是 6.9723 元。个人股息税随持股期限在卖出时另行结算，不能直接按该参考价倒推出持仓净现金。本例仍未验证经纪商的日内资金可用时点。

2026-10-29 只读核验：实时调用 `akshare.stock_fhps_detail_em(symbol='300953')` 返回 2026-05-28 事件的 `股权登记日=2026-05-27`、每 10 股转增 4、派现金 3 元；与上方深交所公告一致。这仅核验一个接口样本，未将该查询写入生产事件或封存回测快照。

## 独立回填入口

`backfill_action_record_dates.py` 只对封存交易持仓期间确实跨过的事件调用东财详情，先核对接口响应中全部对应的除权日都存在且登记日有效，才为该证券写侧车；未匹配的证券保留失败状态，原生产价格及其历史缓存不改写。可先用 `/tmp` 做单票/少量探针，不把探针结果直接接入回测。

```bash
.venv/bin/python -B strategy/analysis/backfill_action_record_dates.py \
  --trades strategy/rolling_validation_results/trade_realized.csv \
  --events .review_runs/data_catalog_quarantine_20261027/data/alternative_data/events_table.pkl \
  --record-parts /tmp/quant_record_parts_held_20261029 \
  --codes 300953,603444 --workers 2 \
  --out /tmp/quant_record_backfill_20261029.json
```

## 实测回填进度与剩余来源

2026-10-29 用独立 `/tmp/quant_record_parts_held_20261029` 试跑 19 只持仓涉事件股票。接口与封存除权日对齐后，**20/21 条事件**取得上游登记日；唯一未写侧车的是 300580（东财两次连接超时，故留作缺失，未按规则硬猜）。公司发布的 [300580 贝斯特 2022 年度权益分派实施公告（2023-06-14）](https://static.cninfo.com.cn/finalpage/2023-06-14/1217053542.PDF) 说明该事件登记日 2023-06-20、除权日 2023-06-21、每 10 股转增 5 股和现金 4 元。它提供人工独立证据，但该日期**未写进假冒东财源的侧车**。后续可以带来源列单独管理公告证据，再与机器侧车交叉确认。

有些来源会瞬时不可用，回填请求因此设连接 6 秒、读取 18 秒的超时，失败记录在 `/tmp/quant_record_backfill_all_20261029.json`；成功侧车保留，重试会跳过已覆盖事件。

对唯一接口超时的 300580，发行人公告核实的日期另存于 `docs/action_record_date_issuer_verified_20261029.csv`，包括公告原文地址。审计命令可以通过 `--manual-records` 接入，逐行列出 `record_date_source` 与 `issuer_notice_url`；接口侧车未写入人工推断。若后来接口数据与公告日期冲突，审计会直接失败，提醒人工处理。此表仅用作**事后事件结算研究**，不能作为当时选股可用数据。

```bash
.venv/bin/python -B strategy/analysis/audit_action_record_coverage.py \
  --trades strategy/rolling_validation_results/trade_realized.csv \
  --events .review_runs/data_catalog_quarantine_20261027/data/alternative_data/events_table.pkl \
  --record-parts /tmp/quant_record_parts_held_20261029 \
  --manual-records docs/action_record_date_issuer_verified_20261029.csv \
  --out /tmp/quant_action_held_event_issuer_coverage_20261029.json
```

登记日的覆盖率不等于已经核实实际每股到手现金、股息税、转增股可交易日或券商资金可用时点；这些条件必须分别核验后才能公平对比 QFQ 模拟与原始价成交。

按上面的只读命令实测：封存 161 笔已平仓交易中，20 笔跨 21 条事件；登记日 21/21（接口 20 条、发行人公告单独补 1 条），事件现金/股数拆分 21/21。下一步对账须按持仓实际份额核算现金、转增股份和对应卖出资格，不能将此覆盖率视为回测收益已修复。

融合事件表的 `src=em_repD` 是为了拟合行情供应商复权曲线而修正了 `D`，不能拿修正值当实际到手红利；`src=sina` 只有乘数，也缺少现金和送转拆分；冻结值同样需要核实原始公告。审计现新增 `terms_from_unmodified_provider` 和对应缺口数量。当前生产事件表有 `em_repD` 14 条，但封存持仓涉及的 21 条均为 `em`，没有触碰这 14 条；`em` 也只能说明未经这条融合逻辑修订，不能代替发行人公告逐条验真。

## 原始价成交研究：发行人条款核对

`docs/action_terms_issuer_verified_20261029.csv` 将已查到原公告的 **4 条**持仓事件按实际股东口径保存：300580、300953 的送转数及新增股份可交易日，以及 603444、600858 的现金；四条均含股权登记日、现金发放日和原公告。`audit_issuer_settlement_terms.py` 只读核对发行人数据与封存事件表中的每股送转数、现金、登记日；日期/数值冲突一律报错，不自动修订生产事件、QFQ 或净值。

```bash
.venv/bin/python -B strategy/analysis/audit_issuer_settlement_terms.py \
  --held /tmp/quant_action_held_event_issuer_coverage_20261029.csv \
  --issuer docs/action_terms_issuer_verified_20261029.csv \
  --out /tmp/quant_issuer_terms_held_20261029.json
```

截至 2026-10-30，对 2026-10-29 封存样本的事后核验结果：**4/21 条已有发行人公告核实实际权益和日期（其中两条送转），17/21 条仍只有上游信息**；第三条持仓送转 300416 尚未核实原公告。现金发放“日期”不能推断开盘时可动用资金；个人投资者股息税还可能在卖出时扣除。故当前不得宣称 21 条事件全部具备原始价回放条件，更不能用四条公告核验结果推断策略收益。

生产 `data/refresh_fetch.sh` 现在在融合事件表生成之后、QFQ 重建之前执行同一审计的 `--events` 模式，针对这 4 条已核实发行人条款检查“事件仍在、每股送转数和股东现金仍一致”；已有登记日侧车时还会逐事件核对它与公告日期，侧车未生成则明确计为 0 而不冒充已交叉核验。此前生产事件表对三条公告实测通过 `3/3`，侧车目录当时尚未建立，日期交叉核验为 `0/3`；新增 600858 后已对当前事件和侧车重新执行守卫：`4/4` 条款一致、`4/4` 供应商登记日与公告日期一致；这属于事后当前数据的核验，不能回填到旧封存输入。事件被上游撤回或改值、已有侧车与公告冲突会使刷新停止，避免继续重建生产价格。它**没有**替换原始事件源或将公告数据送入选股信号；其余 17 条的来源真实性仍待核验，也不能保证接口未来永不修订。新鲜度门禁和逐日净值输入指纹仍须独立通过，才能运行当日出单。

独立研究侧车 `/tmp/quant_record_parts_held_20261029` 此前可完成 `2/3` 条“东财登记日 vs 发行人公告”交叉核验；300580 仍缺东财侧车，所以 `2/3` 不等于生产已经拥有这两条侧车。生产下一次成功获取含事件详情时才会按股票生成侧车并触发同一交叉核验。
