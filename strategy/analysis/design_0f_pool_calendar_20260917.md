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

## 0f-v2 daily粒度 (2026-09-18补充, 探针驱动)

**季度0f冷跑结果 (9/17数据态)**: NAV 439,541 / +75.82% / Sharpe 0.6726 / MDD 23.13%,
买入624, 选股539. membership闸丢弃3,442,089行(55%). 确定性验证✓ (复用路径逐位一致).
年度 vs OFF低分支(981,379): 2021 −14.55pp / 2022 −1.63pp / 2023 −5.15pp /
2024 −33.05pp / 2025 −38.11pp / 2026 −12.71pp — 损伤集中在2024-25.

**粒度探针 (probe_0f_granularity_20260918.py)**:
- A. 死股续命: 26.3%的0f买单在买日∉daily池 — 但前瞻收益普遍优于daily合法买单
  (2022 fwd20 +4.34 vs −1.86; 2023 +9.15 vs +2.71; 2024 +4.83 vs +3.60;
   2025 +8.10 vs +5.69) — 季度退出滞后并非拖累, 反而是波动热门股的续命增益。
- B. 锁定热度: 2025有175只码daily先入池、季度池等待, 中位锁窗42天、
  锁窗涨幅mean +16.9% (56只≥20%); 2026有62只 (+8.0%). 季度池错过确定性行情。
- 净方向不确定(A减分/B加分) → 需要cold run bracket裁决, 故实现daily模式。

**实现 (已入库)**: stock_pool._daily_boundaries (pd.date_range, 2816边界 2019-01-02..
2026-09-17); bt_execution._load_pool_membership支持quarterly|daily (缓存键含mode);
generate_trade_orders daily=target_date当日as-of (与get_stock_pool(todate=target_date)
逐位等价 — 烟测验证: 2026-09-17双方2623只零对称差); yaml pool_calendar: daily。
掩码/闸/union/指纹链全部boundary-count无关, 无需改动。

**成本**: 单遍扫描~4min(缓存后秒级, 5.2M成员记录, parquet ~60MB); 掩码循环2816×np.where
~2-3min; 闸~3-4min; union 5095只(含单日闪现码). 信号重生成~3h(一次性)。

**裁决口径**: daily vs quarterly 四指标+年度分解 bracket; 稳定性=同数据态determinism
重跑逐位一致(机制已证); 采纳方=诚实基线, 与OFF锚点(981k/1.67M)对比仅用于年度对账。

## 0f-v3 跨模式缓存污染bug + raw缓存重构 (2026-09-18 下午, daily重跑前)

**bug**: daily首次cold run (0918b) 产出 353,124/41.25%/0.4642/36.62%, 但结果**无效**。
因子缓存键(_cache_key_str)只含股票列表+日期+参数+数据/代码指纹, **不含池模式** —
掩码只在计算路径、中性化之前应用并随缓存保存 → quarterly run(01:57)建的缓存被daily
run静默复用: 因子层=quarterly掩码, 只有SignalStore闸是daily。hybrid="worst of both":
锁定热度码(B)在因子层被杀 + 死股续命码(A)在闸层被杀, 2021 −20.24%即两效应叠加签名。
log铁证: 0918b无"因子数据日历池掩码"print(缓存命中早return), quarterly log有
(掩码1,847,786行)。

**修复 (factor_preparer.py v3)**: 缓存只存raw(掩码/中性化/rank前) — 计算块抽为
`_compute_factor_data_raw` helper; 掩码+中性化+rank移到缓存命中路径之后, 每次运行按
membership_map应用。收益: 池模式/边界变化只需重算掩码+中性化(~分钟级), 无需重算因子
(~3h); 跨模式污染结构上不可能。附带修复: tmp_factor symlink悬空(WSL重启清/tmp)恢复
(重建目标目录), 清理不再rmdir。
验证信标: **每次运行(含缓存命中)必须出现"因子数据日历池掩码: 边界N个"print**。
IC缓存(ic_cache_*, 键仅config hash)生产路径从不调用(零文件零print), 留档不修。

**daily重跑 (0918d, fresh raw计算~3h)**: 完成后按v2裁决口径执行daily vs quarterly
bracket。参考: quarterly 439,541 (年度 −8.10/−7.48/+8.63/+6.43/+43.01/+25.79);
hybrid 353,124 仅作bug取证, 不入裁决。

## 0f-v3.5 bug裁决更正: 0918b实为纯daily, 353,124诚实基线认证 (2026-09-18 晚)

上文"0918b=hybrid无效"判定**证伪**。0918d完成后取证链闭环:
1. **缓存键含yaml**: `_code_fingerprint`覆盖`config/factor_config.yaml`, quarter→daily
   切yaml即换键 → daily运行结构上不可能命中quarterly缓存。
2. **0918b命中的缓存ef39daf2(01:57)是首次daily尝试(02:11 log)建的**: 该log有daily
   掩码print(边界2816个, 掩码1,937,851行 — 与0918d同数), 保存终态缓存后被WSL重启
   杀死; 0918b随后命中该**daily掩码**缓存 → 0918b=纯daily, 非hybrid。
3. **跨实现逐位一致=确定性认证**: 0918b(旧缓存命中路径) ≡ 0918d(v3全冷重算路径):
   四指标+六年度+avg_IC 0.1041+preds 3,093,843+闸计数(丢弃3,647,530/保留2,919,323)
   +选股529次全同 — 比det重跑更强的复现证据(两套代码路径收敛)。
4. 教训: "命中者无掩码print"只证明走缓存分支, 不证明缓存内容 — 判缓存看**建造者
   log**。v3重构保留: 键缺池模式是真实隐患(yaml是唯一分隔, 若池模式改运行时参数
   则复发), 信标print成为每次运行强制验证项。

**0f诚实基线定案 (9/17数据态)**:
| 配置 | NAV | 收益 | Sharpe | MDD |
|------|-----|------|--------|-----|
| OFF锚点 | 981,379 | 292.55% | 1.4864 | 18.77% |
| quarterly诚实 | 439,541 | 75.82% | 0.6726 | 23.13% |
| **daily诚实(认证)** | **353,124** | **41.25%** | **0.4642** | **36.62%** |

年分解 (quarterly vs daily): 2021 −8.10/−20.24, 2022 −7.48/−10.06, 2023 +8.63/−5.24,
2024 +6.43/+25.04, 2025 +43.01/+52.26, 2026 +25.79/+9.39。
探针A/B在真实回测中均兑现: daily早准入热门码(锁窗+16.9%)在2024/2025全胜;
quarterly粘性(死股续命)在2021-23/2026占优, 四指标合计quarterly 4-0全胜daily。
granularity bracket待monthly(0f-v3 arm A)补齐第三点后按spec取四指标最优者为
floor-relax臂底座; daily的活faithfulness(零入场滞后)作为次级判据记录。

## 0f-v3.6 臂序列工程化: raw缓存键池节豁免 + monthly上线 (2026-09-18 晚)

**raw缓存键豁免池配置**: v3缓存只存raw(掩码前) — 池成员资格不参与raw计算,
但代码指纹含stock_pool.py与yaml全文 → 每次granularity/relax臂切换(改yaml
pool_calendar或pool_relax_*)都会作废raw缓存→重算raw(~15-30min)。修复:
新增`_raw_code_fingerprint()`(raw缓存键专用) — 豁免stock_pool.py文件与
yaml的stock_pool节(`_RAW_YAML_SKIP_SECTIONS = 组合/执行层节 ∪ {stock_pool}`,
`_yaml_raw_digest`); 信号代码指纹**不含豁免**(池模式变化必须重生成信号 —
池节仍计入_yaml_stripped_digest)。烟测: 仅池节内容不同时raw yaml摘要逐位
不变(8caa457c)。收益: B/C/D臂及一切granularity重跑 raw缓存秒级命中。

**DrvFs元数据staleness观察**: monthly冷跑启动时stale检查算得信号fp 1c96da24,
2分钟后prepare_factor_data算得0a2b35ba(与启动前独立进程一致) — WSL2 DrvFs
元数据缓存短暂陈旧导致同进程两次调用fp不同。判定实际运行代码态看log证据
("代码指纹(raw)"新print=当前代码, monthly边界=新代码)而非单次fp值;
设计天然fail-safe(fp不一致→强制重生成, 两个值均≠sidecar故重生成决策不变)。
副作用: sidecar记录finalize时刻fp, 若finalize时遇陈旧视图则下次运行多花
一次重生成(安全损失~3h, 非正确性)。

**monthly臂上线 (0918a)**: `_month_boundary_prev`/`_monthly_boundaries`
(93边界, 2018-12-31..2026-08-31) + bt_execution/generate_trade_orders
monthly分支 + yaml pool_calendar: monthly。烟测全过: 边界函数6断言+
93边界全月末单调+真实membership构建171,567条(缓存键94635f773433,
quarter-end边界成员数与quarterly跑逐位一致=交叉验证)。union池5051只
(< daily 5095 ✓ 月边界⊂日边界)。冷跑日志
logs/bt_execution_0f_monthly_0918a.log (raw缓存因数据指纹66a2a376不同而
fresh计算, 预期~15min; 此后掩码beacon应为"边界93个")。

## 0f-v3后续方向: 池层流动性地板松弛 (探针驱动, 待0f基线定案后设计)

OFF−honest ≈ 540-630k NAV = 系统最大单一alpha源(刷新彩票)。彩票两成分:
(a) pre-admission动量微盘 — 地板卡住的热门码(探针B: 2025锁定175码锁窗+16.9%);
(b) 幸存者选择(不可诚实回收)。方向: as-of daily池 + 动量感知地板松弛
(20d amount ≥ floor/2 且动量条件, 全as-of无前视, 实盘可实现) + 入场粘性
(N天再评估, 诚实化复现死股续命alpha)。bracket候选: floor/2, floor/4,
floor/2+stickiness30d, floor/2+stickiness60d。每臂一冷跑~4h严格串行。

## 0f-v3.7 granularity bracket定案: monthly 3-1胜出, 定为后续臂底座 (2026-09-18 深夜)

monthly冷跑 (0918a, 93边界) 完成: **529,676 / +111.87% / 0.8767 / 30.91%**,
买入639, 选股540. membership闸丢弃3,623,190行(55.6%, 介于daily 3,647,530与
quarterly 3,442,089之间=边界数单调✓)。全链beacon齐: raw缓存05d0e100 fresh算
(数据fp 66a2a376), 掩码93边界1,927,448行, avg_IC 0.0881, preds 3,074,556。

| 配置 | NAV | 收益 | Sharpe | MDD | 收益/MDD |
|------|-----|------|--------|-----|----------|
| daily诚实 | 353,124 | 41.25% | 0.4642 | 36.62% | 1.13 |
| **monthly** | **529,676** | **111.87%** | **0.8767** | **30.91%** | **3.62** |
| quarterly | 439,541 | 75.82% | 0.6726 | 23.13% | 3.28 |

- vs daily: 4-0全占优 (monthly严格支配 — 月末粘性优于逐日churn).
- vs quarterly: 3-1 (NAV +90,135/+36.05pp/+0.2041; MDD −7.78pp).
- 年分解 (monthly vs quarterly): 2021 −16.26/−8.10, 2022 −9.59/−7.48,
  2023 +10.52/+8.63, 2024 +8.95/+6.43, **2025 +81.23/+43.01**, 2026 +25.04/+25.79。
  monthly 2025年近乎翻倍quarterly且超daily(+52.26) — 月末粘性(锁窗中位42天≈2个
  月边界)同时获得探针A续命与探针B锁热度的双重增益; 2026与quarterly持平;
  代价=2021-22少防守(MDD 30.91 vs 23.13, 全曲线最深在2021-22腿)。
- **裁决 (spec_0f_v3_floor_relax条款)**: granularity轴四指标最优=monthly
  (排序和5 < quarterly 7 < daily 12), **monthly=臂B/C/D底座**。MDD差距移交
  C5缓冲臂队列后期攻击。

臂B (monthly+floor/2+momentum 0.10) 即刻上线; 臂C (monthly+floor/2无门) 其后。
