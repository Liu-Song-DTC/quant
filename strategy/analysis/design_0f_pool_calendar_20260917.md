# 0f 池口径修复设计 (2026-09-17, R2b+因子细胞diff实证后)

## 实证基础 (双探针已跑)

1. **R2b=912,019** ≈ R1复现 → 2×主导项=池滑动(2674 vs 2689信号层1.9×放大), todate日推进零贡献。
2. **因子细胞diff**: 同池异todate公共细胞**逐位全等**(PIT干净); 同todate异池**97.2%分歧**
   (128列全波及, 全史2020-01-10起, 指数行也变)。
3. **机制**: factor_neutralization(enabled/industry+market_cap)在factor_preparer内对
   **as-of-todate池**做逐日截面行业均值剥离+市值OLS → 池成员任何翻转(9/14→9/15有~31码)
   → 全截面中性化基准漂移 → 所有因子值变化 → 截面rank/ML/动态因子选择全量重算 →
   饱和评分承重墙放大(±方向不稳定) → 锚点彩票。

## 设计: 季度日历池 (pool_calendar) — 实现口径已定 (masked季度成员制)

**池成员按季度日历生效, 季度内固定; 因子矩阵按(code,date)掩码成员资格**:
- `pool(t) = get_stock_pool(todate=≤t的最近季度末)` — as-of语义, 与现逻辑同准则
  (近20日成交额+价格阈值+60bar最小长度+ST/退市排除+北交所排除), 只是todate换成季度边界。
- **masked制**: factor_df中 (code,date) 若 code ∉ pool(边界≤date) → 因子值置NaN,
  中性化/rank/ML/动态因子选择全部自动继承(notna过滤)。码的全史因子仍在矩阵中,
  只是成员区间外被掩码 — 消除"9/15入池的码向2020截面注入因子值"的非PIT成分。
- 边界列表 = prev(FROMDATE) 到 prev(TODATE) 的全部季度末 (~22个)。
- 新上市: 下季度边界入池; 退市: 留在池但无bar自然无交易 ✓。
- 指数行(sh000001/sh000852/000001/399006): 始终成员, 不掩码。

## 修复面 (全部非豁免, 需全链重生成验证)

| 文件 | 改动 |
|---|---|
| core/stock_pool.py | ① `_quarter_boundary_prev(t)` ② `get_pool_membership_map(boundaries, data_dir)`: **单遍扫描**(每文件读datetime/close/amount/volume一次, 逐边界求as-of成员资格) — 避免22×get_stock_pool的22次全文件扫读; 缓存到 strategy/cache/pool_membership_{data_fp}.parquet (按K线数据指纹, 同数据态复用) |
| bt_execution.py | 两处(477/1936): get_stock_pool→**union池**过滤stock_file_map/stock_codes(任何边界曾入池的码都保留全史); 把membership_map传入factor_preparer; **信号消费按日membership闸**(加载/复用signals CSV后按(code,date)过滤, date∉成员区间→丢弃行, 消除"union扩大后历史非成员期买入") |
| core/factor_preparer.py | prepare_factor_data新增membership_map参数: factor_data组装后、噪声删除/中性化(line~640)前, 按边界区间掩码(向量化, ~分钟级); 指数码豁免(sh000001/sh000852/000001/399006, 与bt_execution 479行现有约定一致) |
| generate_trade_orders.py | universe过滤→get_pool_membership_map([prev(target_date)]) + **实盘安全滤镜**: 最新数据近20日成交额≥板块阈值且价≥2的再检查(日历成员近期失流动性→排除买单+警示) |
| config/factor_config.yaml | stock_pool节新增 `pool_calendar: quarterly`(quarterly/off, off=现状) |

### 信号层membership闸 (2026-09-17补充)

union池使stock_file_map扩至"任何边界曾入池"的码 → 这些码的全史信号都会进入信号CSV。
逐码指标计算本身PIT诚实, 但**非成员日期的买入**是universe选择问题: 与实盘错配
(实盘universe=target_date日历池+安全滤镜)。故信号消费端必须按日闸:
`code ∈ pool(prev_quarter_boundary ≤ signal_date)`, 否则该行信号丢弃。
**闸点=core/signal_store.py `finalize()`**(复用路径bt_execution:1969与重生成路径:1098
的唯一共同choke point): 新增可选参数 `membership_map=None` (pool_calendar=off时不传,
行为与现状逐位一致)。实现=按date→boundary分组, 每boundary一次向量化isin过滤
(~22组操作, 700K行毫秒级)。指数码(sh000001/sh000852/000001/399006)恒成员。
预期效果: 历史非成员期信号清零, 只在季度内成员区间出单 — 与实盘同构。

因子缓存键含池keys→键变化→全量重算(一次性, 预期, ~40min); 信号代码指纹含
bt_execution/factor_preparer→信号全重生成(一次性, ~1.2h)。总计一次性~2h。

### 指纹/缓存失效链 (2026-09-17核对)

- **stock_pool.py 必须加入 factor_preparer._FACTOR_CODE_FILES**: 现不在列表 —
  membership逻辑变更会使掩码内容变化, 但因子缓存键不含stock_pool.py → 旧缓存静默复用
  (344复现陷阱同型)。加入后因子缓存自动失效重算, 方向安全。
- **signal_store.py 不必入指纹**: 闸在finalize()消费时应用, 磁盘CSV内容不变(未过滤),
  复用路径每次加载都重施闸 → 闸代码变更无需信号重生成。
- **membership parquet缓存键**: data_fp(_data_fingerprint复用, fd=None) + boundaries
  + min_price/流动性阈值 + stock_pool.py代码hash。任一变化→重算(~分钟级)。

## 验证协议 (明天执行)

1. 代码改动+`pool_calendar: quarterly` → 全链重生成冷跑 → 四指标 = 新日历池锚点。
2. **稳定性验证(核心)**: 同数据态连跑两天(或模拟: 删1日数据重跑) → 锚点必须±微小漂移
   (≤0.1%量级, 无K线/基本面变化=应该逐位一致), 不再2×。此验证不过=设计失败。
3. reconcile_anchor + pool_flip_report照常; 季度边界日预期有正常翻转(全年4次, 有预告)。
4. 采纳标准: 锚点稳定性+四指标年度分解可解释; 与1,728,548/1,671,592的直接四指标对比
   不是采纳依据(旧锚点本身带彩票+前视成分), 但按年度分解对账必须可解释。
   注意与0g的叠加: 0g(dragon_tiger OFF, −57k全在2026)先落地, 0f的年度对账基准
   用OFF态基线(1,671,592), 避免artifact混入0f解读。

## 风险与边界

- 中性化基准在季度边界仍会跳(4次/年, 有限量级) — 可接受, 是诚实低频再平衡;
- 实盘与回测池完全同构(同日历), 消除"回测买了实盘买不到"的错配; 安全滤镜只减不加;
- 该修复是**系统性正确性修复**(刷新彩票根因), 非收益优化 — 锚点数值会变是预期的,
  变的方向由日历池的承重墙响应决定, 裁决口径=稳定性+四指标年度可解释性。

## 排序

0g(dragon_tiger OFF, 进行中) → C5六臂 → C1三臂 → **0f实现(明天上午)** → 批次2。
0f在Lane 0优先级最高, 且影响所有后续校准实验的基线(先定池再调参)。
