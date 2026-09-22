# yaml×census 全量对账 + 评分配置路径审计 (2026-09-22)

用户9/18指令的架构审查落地件: "目前架构是否合理, 有没有优化的地方"。
方法: 36个零文档参数逐键追reader → 分组裁决; 评分配置消费链逐行追读 →
验证CSV+factor_df离线探针裁决。**零生产改动, fp 6f1cb6b1|0 未动。**

## 1. 36参数分组裁决

| 组 | 参数 | 裁决 | 证据 |
|---|---|---|---|
| A 死yaml键 | chan_gate_enabled, chan_gate_min_count, buy_sell_points, divergence_threshold, consolidation_discount, alignment_boost, top_divergence_exit, volume_divergence_exit, zhongyin_penalty | 零reader, yaml在fp内**不可删**, 仅文档化 | 全仓grep零命中 |
| A' 双源镜像 | `indicator_params`整节 | factor_preparer走硬编码`get_default_params()`(factor_calculator.py:1085), signal_engine:324读yaml — **值逐位一致**, 双源真相=架构债, 当前零影响 | 两处值比对全等 |
| B 死旋钮 | ti_boost_scale(signal_engine:296), chan_bonus_buy_point(portfolio:290), chan_bonus_sl2(289), fv_low/fv_high/fv_exposure_min/fv_exposure_max(portfolio:187-190), use_static_candidates(selector:408), evolution_guard模块 | 赋值后零读; 全并入census死旋钮名录 | 逐键read trace |
| C 机制关闭 | dynamic_factor整节(train_window/forward_period/top_n_factors/min_ic_dates/ic_decay_factor/min_train_samples/min_factor_count/min_factor_families/extra_candidate_factors) | factor_mode=fixed → enabled=False, 生产日志 DYN命中 0/6,658,619 | 生产日志line 63 |
| D 实践惰性 | composite_bias(IND_信号=0), V41(311,340行0买卖), volatility_adaptive_mult(回测cost={}), 退出栈全套(exit_mode='simple'三层闸) | census已裁决或本审计量化 | census文档 |
| E 实活已覆盖 | entry_chan_gate(E-N5), chan_div两阈值(sell侧66事件量化), mtf折扣(机制级), sell_threshold(末班8+11笔), indicator_params(见A') | 全部已有bracket/量化记录 | census+执行层末班 |
| F 实活+本审计新量化 | **全局industry_factors(377概念)** vs **季度标定(401概念)** | 见§2-4: 现状架构经IC探针裁决为**合理** | 本审计探针 |

## 2. 评分配置消费链 (fixed模式全链)

`_select_factor` → fixed模式 → `_calculate_default_factor`:
- **P0**: `_get_specific_industry`(概念解析: STOCK_CONCEPT_MAP风格过滤 + inception PIT gate + 季度配置成员资格) → 熊/弱趋势(regime==-1或trend<-0.05)读**季度bear_factors**; 中性/牛市读**全局factors/bull_factors**。
- **P1**: 仅当P0未命中 — `FactorLibrary.get_scoring_factors(fallback_config=季度配置)` → 季度factors(权重归一)。
- **P2**: IC磁盘缓存 — 生产无ic_cache_*.pkl文件 → 恒空。
- **P3**: 硬编码4因子(trend_lowvol/relative_strength/low_downside/momentum_reversal)。

验证CSV路径实测: **见§8** (9/22生产ground-truth交叉验证, 取代本节早期基于stale
CSV的43.7%/35.9%/8.7%/11.8%数字 — 该批数字的validation_results.csv自9/6起stale)。

## 3. 关键证伪: "预计算浪费算力"不成立

前一分析段曾认为 bt_execution.py:919 无条件跑IC预计算(固定模式纯浪费)。
**证伪**: 该调用位于 `if use_dynamic:` 块内(bt_execution.py:874, use_dynamic = factor_mode != 'fixed');
live_init.py:94同样门控`factor_mode != 'fixed'`。生产日志0处预计算进度标记+"因子选择分布(总计1)"
来自backtest_diagnostics记录器而非预计算。**代码早已正确门控, 无修复必要。**

## 4. 核心裁决: 全局权重 vs 季度权重 IC探针 (probe_qweight_coverage_20260922.py)

问题: 全局配置自2026-03-20未实质更新(git -S), 365交集概念的中性/牛市评分跑旧权重 —
是否应把P0切到季度权重?

方法: factor_df缓存(3.5M行, 生产评分同源rank值)上复刻生产概念解析(风格过滤+inception gate+
季度成员资格), 对可切换集逐日Spearman IC(score vs future_ret), 687交易日配对。

| 权重 | 逐日IC mean | t | 正比例 |
|---|---|---|---|
| 全局 factors | **+0.0464** | 16.6 | 73.9% |
| 季度 factors | +0.0290 | 10.1 | 65.1% |
| 配对差 季度−全局 | **−0.0175** | **−5.4** | 40.8% |

分年配对差: 2021 −0.0285 / 2022 −0.0284 / 2023 −0.0051 / 2024 −0.0179 / 2025 −0.0148 / **2026 −0.0063**。
块bootstrap(block=20, B=2000, seed=42, n=687): 配对差95%CI **[−0.0313, −0.0030]**,
P(mean>0)=0.0120 — 统计上排除零, 季度权重切换不是噪声级劣化。
分年CI: 2021 [−0.0576, +0.0035] P=0.038 / 2022 [−0.0565, −0.0083] P=0.0005 /
2023 [−0.0326, +0.0217] / 2024 [−0.0417, +0.0239] / 2025 [−0.0474, +0.0069] /
2026 [−0.0883, +0.0745] — 2021/2022单独可辨(劣化), 2023-26不可辨但点估计全负。
诚实注: 2021-25全局占优部分因hindsight-fit(全局配置在2026-03生成, 携带全窗口知识);
但2026是干净窗口(两配置都先于2026固定) — 季度权重仍输(点估计负, CI不可辨)。

**裁决: P0中性/牛市切季度权重 = 负期望, 方向关闭。全局旧权重不是缺陷, 是特性。**
(呼应全程序一贯证据: 简单老机制胜过滚动重标定 — 与F-2/F-3/E-seq1同签名。)

## 5. 季度标定的真实价值 = 新概念覆盖

仅季度概念(36个存量+2026中报类新概念)上, 季度权重 vs P3硬编码默认:
**季度 +0.2423 (t=7.25, 95.7%正) vs 默认 +0.0089 (t=0.22), 配对差 +0.2334 (t=5.07)** —
窗口=2026年23个交易日, 对2026Q3文件为**OOS**(标定至6/30, 测试7/1~9/16)。

结论: 季度标定通道的真实增益 = **新概念权重 + 熊分支bear_factors + 概念解析gating**
(消解E-K1悖论: 冷跑+60,652的增益来自这三通道, 而非存量概念权重刷新)。
**9/30 Q4标定执行建议: 焦点=全局缺失的新概念; 存量交集概念的中性/牛市权重重标定
不改善评分(甚至不被消费) — 不必花力气。**

## 6. 架构合理性结论 + 遗留修复候选 (全部零行为影响)

架构裁决: **合理**。评分配置双层(全局静态+季度滚动)是自洽设计; 全局层扛中性/牛市、
季度层覆盖新概念与熊分支, 经IC探针验证优于反事实替代。

9/30窗口(与A.5印花税共用一次fp重锚)可选修复:
1. **季度诊断可观测性**: `_QUARTER_DIAG`计数在多进程worker内累积后随进程消亡,
   主进程bt_execution.py:1170打印恒为0 — 生产日志从未出现"行业名查找"行。
   fix: worker结束时聚合diag传回主进程(或spawn前存文件)。零行为影响。
2. **死键/死旋钮清理**: A/B组(约20个)注释归档 — 已文档化, 删除需在fp豁免节外,
   不删yaml键(指纹内节)。
3. **双源参数合一**: indicator_params删yaml节或删硬编码, 二选一。零行为影响。
4. 生产日志"固定行业因子49.5%/固定默认因子50.5%"是计数器语义artifact
   (fixed_default=入口计数, fixed_industry=P0命中), **非覆盖度指标** — 真实覆盖度
   见§2(仅0.6%真兜底)。注释澄清即可。

**不排新alpha臂** — 与census终态一致, 本审计方向探针(季度权重切换)是最后一个
疑似headroom, 已量化为负期望关闭。

## 7. 分支原位+λ-blend连续体裁决 (probe_qweight_blend_bear_20260922.py, 零生产改动)

补三个悬疑点, 使评分配置裁决完备。regime/trend_score从冻结指数CSV经
MarketRegimeDetector精确重建(bt_execution.py:592-631同款), 熊/弱趋势日331日,
牛日251日, 严格中性日~800日。

**(A) λ-blend连续体**: score_λ=(1−λ)·score_g+λ·score_q (权重均归一≈1, 凸组合)。
非熊日表面: λ∈{0,.25,.5,.75,1} → IC +0.0441/+0.0486/+0.0490/+0.0412/+0.0289 —
表面非单调, λ=0.5处bump +0.0049 (t=2.10, CI[−0.0040,+0.0157]含零, P=0.88, 分年5/6正)。
**bump定位**: 全在牛日 (+0.0081, t=2.99; 2026 +0.0279); **严格中性日(生产实际消费
全局权重的域)上blend=纯噪声** (−0.0015, t=−0.37, 2026 −0.0034)。牛日生产消费的是
bull权重而非中性权重, 该bump不是生产域内的内点最优。**λ-blend方向关闭**。

**(B) 熊日原位**: 生产=季度bear IC +0.0427, 6/6年为正; 反事实=全局中性 +0.0489
(配对差−0.0062, t=−1.69)。但反事实不自洽: 全局中性权重是2026-03全窗hindsight-fit,
从未在熊日服务过 — 用事后拟合权重当反事实必然高估。**干净窗2026: 生产bear +0.0779
vs 反事实+0.0652, 配对差+0.0127生产占优**。裁决: 熊分支保留, 获正证据背书。

**(C) 牛日原位**: 生产=全局bull IC +0.0350, 6/6年为正; 反事实=全局中性 +0.0446
(配对差−0.0095, t=−2.04)。同样受hindsight-fit污染(全局bull与neutral同批拟合, 但
反事实在2026干净窗骤降: 2026 bull +0.0381 vs neutral +0.0131, 配对差+0.0250生产
占优, n=28)。裁决: 牛分支保留。备记: 若未来重访牛日评分, λ=0.5 blend(bull⊗季度)
是leading candidate(2026 +0.0279), 但CI含零, 低于行动阈值。

**终局**: 评分链三分支(中性/牛/熊)在唯一干净窗(2026)生产配置全部占优; 全部候选
替代(季度切换/λ-blend/去分支)已裁决关闭。评分配置链审计完备闭合。

## 8. 生产ground-truth交叉验证 (probe_factor_name_crosscheck_20260922.py, 9/22)

动机: §4-7的IC裁决全部建立在离线概念解析复刻上 — 复刻若与生产消费链不一致,
所有结论失效。用生产validation CSV的 factor_name 列做ground truth验证。

前置修复: §2引用的validation_results.csv自9/6起stale (9/17/9/20流从未重跑
signal_validator) — 9/22重生成 (6,580,994行, buys 1,223,334, 2021-01-04~2026-09-03),
runbook A节新增强制重生成条目。

方法: 逐buy行读生产factor_name → 剥除末token尾缀flag (`_[A-Z0-9]+$`, 含R2=bp2通道
数字尾缀; 因子名全小写无数字故安全) → tokens与分支预测配置的有序前缀比对。
分支预测=离线复刻 (STYLE_KW过滤+inception gate+季度成员资格+首个命中;
熊日qb→gb→g→q / 牛日gbull / 中性g / 仅季度概念P1→q)。

**结果**:
- 概念解析命中 97.0% (1,186,817/1,223,334)
- 分来源吻合率: 中性g **93.9%** / 牛gbull **90.4%** / 熊qb **94.0%** (熊日子组同)
- 未吻合构成: 特殊通道single-token名(REV60/REV/MOM等)全样本6.7%;
  真不吻合仅5,919 (0.5%), 其中30.4%全体token落在同概念配置并集内=跨集加成注入;
  残余≈0.35% (加成池外注入/季度边界) — **离线复刻与生产消费链吻合 ≥99.5%**

**路径census (新鲜, 取代§2)**:
- qb 43.1% / gbull 32.5% / g 15.3% / special 6.7% / q(P1) 1.8% / unknown 0.5% / P3-default 0.1%
- 加未解析组36,517行 (3.0% buys, 无概念配置→真P3兜底, factor_name=P3四元组)
  → **真P3兜底率 ≈3.1% buys** (早期0.6%为stale CSV低估)
- 2026年 mean future_ret by path: qb +0.0134 / q +0.0171 / special +0.0058 /
  gbull +0.0020 / g −0.0187 / P3-default **−0.0686** — 路径间原始均值受样本群差异
  影响不作裁决依据; 但P3兜底行2026均值−6.9%显著为负, 排队探针:
  **P3兜底信号是否实际进入组合层 (buy阈值与portfolio_selections交叉)** —
  若进入, "P3兜底信号禁买/加严阈值"成为新候选臂。

**裁决**: 离线概念解析复刻与生产消费链ground-truth吻合≥99.5% — §4-7全部IC探针
(季度权重切换/λ-blend/bear-bull分支)的结论建立在生产忠实的基础上。评分链审计完备闭合。

### 8.1 执行层两条排队探针的闭合 (9/22)

**(a) P3兜底执行探针 (probe_p3_fallback_execution_20260922.py)** — P3兜底buy
36,579行 (3.0%), 但**0/570执行** (同日join 0, 陈信号15日兜底 0)。原因: P3行
score当日百分位中位42.6 vs 被执行行96.5 — 结构性出局。且P3行future_ret
+0.0024 vs 全buy +0.0094, 即使执行也只是轻微劣化。"P3禁买/加严"臂=零行为
变化 (census死旋钮同源), 无需冷跑。**方向关闭。**

**(b) 执行集路径归因 (probe_exec_path_attribution_20260922.py)** — 570执行行的
评分路径: gbull 40.2% / qb 32.1% / g 23.3% / special 2.5% / q 1.4% (全buy层
43.1%/32.5%/15.3% — g被超选、qb被低选, 与牛日信号在敞口限制下更易入选一致)。
执行集Σw×fr(20日近似)全正: g +0.394 / gbull +0.385 / qb +0.309 / special
+0.076 / q +0.048 / unknown −0.015 — 无毒性路径。2026 g路径−0.053由2/5、5/28、
6/29三批中性日入场承担(市场时序, 非因子集缺陷 — §4 IC已证全局中性权重是该域
最优)。**执行层无新臂, 归因存档。**

## 9. Q4标定新概念覆盖预检 + PIT gate落地 (probe_q4_newconcept_coverage_20260922.py)

§5焦点纪律的补充证据: 36仅季度概念在标定窗口(2021-10-01+, 1193天)的估计
样本量, 两口径对比 — A=标定实际(offline_calibration无gate, 每概念ALL所属股票)
vs B=生产消费(inception PIT gate)。

- 全部36概念通过min_codes=20 (最小毛发医疗20股22.9k stock-days);
- **中报4概念(扭亏/预减/预增/首亏) gate_share仅0.040-0.041** — 96%标定样本
  在概念成立(2026-07, gated仅47天)之前 = 前视归属稀释: 2021年的股票-日被打上
  "2026中报预增"标签参与权重估计;
- 中度稀释: 光刻机0.643 / 脑机接口0.689 / 液冷服务器0.687 / 超导0.745 / 供销社0.791;
- 其余概念gate_share=1.0 (inception早于窗口)。

**落地 (9/22)**:
1. offline_calibration.py: `calibrate_industry_regime`/`select_best_factors` 新增
   可选 `concept_inception` 参数 — 传入时每概念估计样本只含inception之后的行
   (与生产signal_engine一致); None=旧行为。合成烟测3断言过 (gate ON n_dates
   60→30 / 空inception==OFF / select路径正常)。
2. calib_2026Q4_tail.py: 加载concept_inception接入两调用; **薄样本回退规则** —
   gated天数<120的概念回退2026Q3权重 (窗口1193日的10%; Q3权重经§5的23日
   OOS验证+0.2423, 薄样本gated重标定方差高且无OOS余量, 用已验证先验胜于
   噪声重标定)。**9/22干跑实证(阈值60): 4/2026年份概念(gated 101-119日, 份额
   8.5-9.7%)拿到fresh重标定且产物不稳定 — 大盘价值/行业龙头坍缩为volatility
   单因子, 一季报预增fund_pg_improve整因子被弃 = 薄样本噪声签名; 120把它们
   纳入Q3先验。9/30投影: 中报4≈56日, 一季报预增≈116日, 电池技术≈128日 →
   120把边界放在10%份额处(4.7-9.7%回退, 10.7%重标定), 相同行为带=[120,~128];
   60/100会放一季报档去噪声重标定。** 回退/保留清单脚本内打印, 9/30写盘后
   验证器(verify_q4_dryrun_20260922.py)复核<120天概念无fresh权重。
3. 裁决权不变: 9/30写盘后全链冷跑四指标 (runbook C)。

**9/22干跑执行+验证记录** (CALIB_OUTPUT_DIR=/tmp重定向, env已核验
QUANT_ALT_NO_AUTOREFRESH=1, 生产config零写入 — /tmp季度副本与生产md5逐位一致):
- 5.56M行×5128股×1193日; PIT gate 163概念加载; 402概念标定(neutral/bull/bear
  各402); 概念churn: +代糖/彩票/退税商店, −昨日炸板/昨日触板 (min_codes=20
  池滑动, 正常)。
- 回退表(启动阈值60): 中报4(47日)+玻璃基板(46日) 5概念copied; 2026Q4.yaml
  +index(24季度)写入/tmp成功。
- 验证器9项: 8过1报 — 中报4 Q4==Q3逐位一致✓/日志copied清单✓(np.str_ regex
  修正后)/权重和✓/保存确认✓; check#4报7概念(101-119日)fresh=**60阈值启动的
  干跑artifact, 恰实证120裁决** (见上第2条坍缩证据) — 9/30正式运行预期零违规,
  验证器将强制执行。机制链路(加载→gated→回退→保存→索引)全通, 干跑目的达成。

**注意**: 上述改变仅影响2026Q4文件内容 (quarterly_factors在信号指纹内→信号
必重生成, 已有全链裁决兜底), 23个存量季度文件与生产732,689零扰动。

## 10. BEAR部分敞口层级探针 (probe_bear_partial_tier_20260922.py, 9/22)

审计链最后一环: 9/21 regime量化留下一个未测方向 — 早触发已否决(H5/H6/E-N12),
再入场提速已采纳(E-A4), 但**层级本身(bear_risk日exposure 0.0→0.3)从未bracket**。
9/21文档的"25簇68%零保护+现金期仅−0.11%"表面证据曾暗示部分层级为正期望。

方法: 冻结指数经MarketRegimeDetector重建bear_risk (自检: 266日≈260 ✓),
25个spell逐段测spell期指数持有收益 (0.3层级贡献=0.3×spell_ret), 分年合计。

**结果: 指数级0.3层级全期 −6.75pp NAV — 方向否决, 免冷跑。**
- 16/25 spell为负, 负spell代价合计−11.31pp; 2022 −6.4pp / 2023 −3.1pp /
  2024 +0.4pp / 2026 +2.4pp — 结构熊年(2022/2023)现金期继续跌, 修正年(2026)
  现金期上涨, 层级是落在结构熊侧的坏硬币。
- 表面证据反转: "68%零保护"是簇数口径; 持有期口径(层级臂的真实暴露)下
  spell收益均值−0.9%, 且**spell结束后fwd20均值−0.4%** (9/21所列+5.4~13.8%
  反弹仅存在于5个大簇的精选口径, 全簇口径V腿≈0)。
- 9/21"当前calibration=趋势确认后才全清是最不坏点"获第三角度加强:
  早触发(出口)✗ + 部分持有(层级)✗ + 延迟回补(入口, E-N12)✗ —
  三方向全负, 二进制全清+确认后恢复=该机制唯一存活形态。
- 2026签名(熊簇期间市场+7.9%)提示2026-08簇"实际保护"是小型代价事件;
  该信息与E-A4的再入场提速互补, 不构成新臂。

**归档: regime机制三方向(触发时点/层级/回补)全部量化闭合。**

## 11. 敞口阶梯全量化 (9/22, 生产faithful regime_state.csv × 重建trend/ivr)

§10只测了bear rung; 本节目的是把 portfolio.py:996-1031 整条敞口机制逐级量化
(数据源: regime_state.csv末行NAV=732,688.93逐位=锚点, BEAR 260日与文档全等 —
确认生产faithful, 非实验臂产物)。base_exposure=0.85。

| 机制 | 值 | 绑定实测 | 裁决 |
|---|---|---|---|
| bear rung | 0.0 | 260日 (97.7% BEAR日) | §10: 0.3层级−6.75pp否决, 二进制全清存活 |
| weak rung (0.5×base) | 0.425 | **0日** — trend_score量化档{−1,−0.5,0,0.5,1}, 0.5入full档, (0,0.5)空集 | **死代码**, 归档 |
| neutral/dip rungs (0.3×base) | 0.255 | ≈19日贴cap; trend==0仅60日且全在2021Q1零信号期(需求=0) | 需求约束压倒cap, 无可辨识prize |
| 缩量cap | 0.35/0.50 | **ivr<0.5: 0日(死代码); 0.5-0.7: 8日** (fwd20 +3.07%) | "无量无行情"实际8日, 死旋钮归档 |
| Chan强买点floor | 0.60/0.45/0.15 | 上界≈53/35/6日 (暴露∈cap±0.025) | 方向=系统自身信号背书(E-A4同向), 达≤94日×0.1级, 免臂 |
| full rung | 0.85 | 87日 (6.3%) | base_exposure已9/9 bracket(1.0铁律否决) |
| vol_scale/tvol | 0.28/0.75 | 批4e已bracket | 闭 |
| emergency/stop-loss | 0.65/0.08 | C12已bracket | 闭 |

结论: **敞口机制八级全部量化闭合** — 四级绑定(0.85/0.0/vol/emergency)全已有
bracket或否决记录, 四级近死(weak/neutral/缩量/Chan floor仅上界94日)。
"非BEAR日敞口0.68-0.71"的真相=需求驱动+平滑惯性, 非任何未测阶梯在起作用。
市场择时类旋钮在该系统的否决纪录再添一笔(与S批次/bear层级/H5/H6/C12同签名)。

## §12 买入侧三层门控闭合 (9/22, probe_buyth_pct + probe_entrygate_margin, 免冷跑)

**动态阈值死旋钮**: 执行buy 562/569可复算p30(探针slice索引bug已修, 首跑0/569
为artifact非机制), 生产行pct=0.3在真实边界b=0.8/1.0切**0**执行buy(自检
ratio<0.8=0), 提档pct0.4-0.7×b1.0切2-51行且切集weight×fwd**全为正**
(+0.0059~+0.1106) → 提档=砍正收益笔, 无候选臂。静态buy_threshold同理:
执行buy score min=0.0701 vs floor乘数0.15 → 需X>0.47才触首行, 此前池收缩
已灾难(0.7档切40.9%全buy行)。机制本身被验证: 全buy行ratio梯度单调
([0,1)+0.68%→[2,inf)+0.96% fwd), 执行集ratio≥1.0全过(69% floor绑定
p30≤0→实际门=score≥0)。

**初始门控零绑定**: 执行buy score min 0.070 vs 门槛−0.05(裕量+0.120);
dist_ma20 max 0.247 vs 默认容差0.30(裕量0.053); 0行超0.40自检过。30%执行buy
低于MA20靠结构豁免=最佳带(fwd+2.5%, wfr+3.94)非泄漏; dist≥15%追高带仅11行
wfr−0.30(≈0.04% NAV<CI最小可辨, max_dist收紧臂免冷跑关闭)。全buy行score
十分位fwd单调(+0.63%→+1.82%)=rank排序有效。门控仅候选过滤, 选择权在
portfolio rank层 = 设计意图实况。

## §13 月度季节性扩展 + P3覆盖缺口 (9/22, 免冷跑)

**月度表(执行buy wfr)**: 10月+2.10/3月+2.10/4月+1.48强; 8月−0.70/12月−0.36/
5月−0.01弱 — 8/9月负与S批次结论一致, **12月负为新发现**(fwd−0.44%, 全buy行
fwd+0.03%也为最弱月)但年收益≈−0.36 wfr≈+0.05% NAV<CI, 且S批次日历敞口臂
已否决 → 关闭为census证据。

**P3覆盖缺口(40申万标签, 2.5% Q3 buys)**: 缺口信号有肉(fwd+1.01% vs有配置
+0.12%, 日期中性+0.84% vs −0.02%)但P3打分水平压低(0.270 vs 0.352)→执行集
仅1/565; 收割上界≈+0.08% NAV/年<CI, 与Q4共用窗口破坏单变量归因 → 关闭。
列为未来自然重锚窗口零成本顺风候选。

## §14 Q3→Q4 bear刷新幅度 (9/22干跑产物, 9/30风险预期)

交集399概念: bear因子集不同267(67%)/相同132; 同集bear权重|Δ|均值0.050
中位0.040 p90 0.109 max 0.356 — 温和刷新非重写(典型bear权重0.2-0.5, |Δ|
≈10-20%相对)。中性因子集不同294/399但**不被消费**(P0读全局), 无害。
9/30的Q4赌注风险画像=中低, 四指标裁决兜底不变。

## §15 9/22晚 census完备性代码级复核 + 9/30窗口de-risk (day-8收口)

### 15.1 剩余零覆盖旋钮代码级死链确认 (census"旋钮空间耗尽"主张复核)

对program doc census表之外的零提及yaml键逐一代码追踪 (portfolio.py消费链):
- **trailing_stop_by_buy_point**: 三消费点全部双闸死 — 1842(成本循环guard `cost[code]`
  有效 + `exit_mode=='simple'` continue) / 1964(`exit_mode!='simple'`整块gate) /
  2128(显式`exit_mode!='simple'`)。program doc line 743"全部设计关闭"主张
  **代码级证实**; M3探针fwd20=+2.34%是反事实模拟非生产事件, 无矛盾。
- **sector_signal_density_weight 0.6 / sector_momentum_weight 0.4**: 读取后零消费 → 死键。
- **mom_60d_fomo_threshold/mult + warn_threshold/mult**: 读取后零消费 → 死键
  (E-N15高位动量入场被否决后回滚, 旋钮留在yaml但无消费链)。
- **isolated_b3_penalty −0.08**: 读取后零消费 → 死键。
- **consecutive_loss_breaker (enabled=true)**: C10探针已量化 (154/154选股行clb=0,
  零触发) — live-armed但经验上惰性, E-H6覆盖语义 ✓。
- **volatility_control blend/lookback**: C11死旋钮 (config_loader只映射enabled) ✓。
- **max_adaptive_stop_mult / profit_lock_pct / win_floor**: 均位于exit_mode='simple'
  或cost={}闸后 → 死链 ✓。
**结论**: census终局表+本轮代码级复核, 组合层每个key要么已bracket/量化, 要么代码级
证实死链 — 冻结态旋钮空间确认穷尽。

### 15.2 A.5印花税补丁配方机械验证 (9/30免现场调试)

runbook A.5逐行配方对/tmp副本全量应用: 4断言(锚点唯一性)全过 + py_compile PASS。
行号锚点(44/45/1526/1656/1695/1699/1731)与当前代码态零漂移; STAMP_TAX全文件引用
= 定义+EFFECTIVE_COMM+2卖单结算, 与配方"共2处使用"自洽。9/30窗口补丁为纯机械操作。

### 15.3 9/30月度日历池迁移预估计 (probe_pool_transition_0930_20260922.py)

生产口径 (pool_calendar=monthly, relax_floor=0.05, momentum=None) 用数据至9/17
评估2026-09-30边界成员资格 (代理下界, 实际9/30还见9/18-9/29新数据):
- 8/31边界(当前生效) 5128 → 9/30代理 5126: 新入4 / 退出6 / 留存5122 (99.9%)。
- **池迁移成分对9/30锚点漂移的贡献≈0** — 与9/15危机(−690k池滑动)结构性不同
  (日历池月边界 vs as-of每日重估)。9/30归因分解: 漂移 = 数据刷新delta +
  A.5印花税(−4.4%±1%), 池成分可忽略。若实际漂移显著偏离此分解 → 立即深查
  数据刷新层 (9/15危机同款取证路径)。

## §16 缠论census终局: 信号引擎键家族活性分类 + bp门绑定性量化 (9/22, day-8)

signal_engine.py 是alpha心脏, 此前census只覆盖portfolio层 — 本节约束全信号CSV
(rolling_validation_results/backtest_signals.csv, 40列×6.66M行, SignalStore dtype
表证实含全部缠论组件列) 免冷跑量化 + 代码消费链追踪。

### 16.1 新死键归档 (读取后零消费, 代码级证实)

- **ti_boost_scale 2.0 / ti_boost_magnitude 0.12 / ti_adaptive_scale 0.3 /
  ti_adaptive_max_adjust 0.12 / signal_confidence_baseline 0.2**:
  signal_engine.py:296-300读取后全库零引用 → 死键×5。
- **trend_sell_threshold_strong/weak + trend_sell_ti_threshold/ti_relax**:
  :279-282读取后零引用 → 死键×4。
- **divergence bottom/top_div_threshold 0.3/0.3**: :346-347读取后零引用 →
  死键×2 (b2门实际消费chan_b2_min_div_strength=0.0)。
- **portfolio.signal_boost 六乘数 (bottom_divergence_mult 1.15 / top_divergence
  _mult 0.78 / alignment_boost 0.06 / zhongyin_penalty 0.9 / pivot_breakout
  _buy_mult 1.08 / pivot_breakout_sell_mult 0.82)**: 全生产代码零消费 (仅
  config_loader默认值+evolution_guard范围表) → 死键×6。yaml值与config_loader
  默认不同=历史调参残留, 消费代码已随机制演进移除。
- **bp4_gate.min_signal_level=3 惰性**: 全量30,411个bp4 buys全部sl=4,
  边际桶0行 → 门值从不绑定 (b4真实过滤器=min_confidence 0.2+tt≥0)。
- **b2_gate.min_div_strength=0.0**: 值上无牙 (一切div≥0通过)。
- **b3_filter.enabled=false**: 关闭态。
- **resonance五键**: 计数只进tag/CSV字段, portfolio/bt_execution零下游消费
  → 末班八项"名字装饰"裁决获代码级证实 ✓。

### 16.2 活门绑定性量化 (probe_bpgate_binding/tail_20260922.py)

买入点门家族 = signal引擎唯一census未量化活旋钮。生产值: b1 sl≥2+底背离
≥0.15+量峰≥1.5 / b4 sl≥3+conf≥0.2 / bp5,7,8 sl≥2 / bp6 reject_all。

**绑定量级**: bp1/bp5/bp7/bp8 buys的48-54%坐边际桶(sl=min) — 门真实承重。

**边际桶vs内桶fwd差 (日期中性化, 年度分解)**:
| bp | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|---|
| bp1 | −0.002 | −0.005 | −0.008 | +0.001 | −0.005 | −0.014 |
| bp5 | −0.005 | +0.002 | −0.001 | −0.008 | −0.007 | −0.017 |
| bp7 | −0.023 | −0.014 | −0.012 | +0.007 | −0.000 | −0.024 |
| bp8 | +0.001 | −0.005 | −0.005 | −0.000 | −0.005 | −0.010 |

方向稳定5/6年负, bp7幅度最大(−1.1~−2.4pp)。

**执行集cut-set (真实砍集, 决定性)**:
- 执行bp1边际桶 n=43 fwd +0.0278/胜率53% — 正! score已选好 → 收紧砍赢家
- 执行bp5边际桶 n=12 fwd +0.0168/胜率67% — 正 → 同上关闭
- 执行bp8边际桶 n=10 fwd +0.0164/胜率70% — 正 → 同上关闭 (池级负均值由
  未执行行拖累, E-N7教训镜像)
- **执行bp7边际桶 n=18 fwd −0.0086/胜率39%/权重2.34 vs 内桶+0.0223/58% —
  唯一真负cut-set**; 且执行边际桶score 0.792≈内桶0.805 — score对bp7的
  signal_level正交信息失明 (池级sl=2 score均值0.271甚至>sl=3的0.264)
  → **bp7臂已启动: min_signal_level 2→3, 后台be942rzgw, 裁决四指标vs
  732,689/193.08%/1.1961/17.92%**。

**b1 min_bottom_div 0.15**: 边际带[0.15,0.25) fwd +0.0278≈内带+0.0265 —
门不切割质量分层 → 收紧无cut-set, 关闭。

**剩余生成态旋钮终局**: b1 min_bottom_div + min_fx_vol_spike 读取后零消费
(:872-877 b1门仅查sl) → 死键×2入档; b4 min_confidence 0.2活(消费buy_confidence)
但执行b4仅15行 = cut-set亚CI → 关闭。

### 16.3 P3缺口标签诊断终局 (probe_p3_gap_label_diag_20260922.py)

缺口40标签粒度=申万一级/二级混合 (银行一级/化学制品二级…), 非映射bug:
98个缺口code全panel零标签漂移 (98/98仅缺口标签) — 这些股的概念从未进
配置, P3兜底恰当。**年度分解杀死缺口headroom**: 缺口buy ex_fwd vs有配置
2021 −0.0049/2022 +0.0013/2023 −0.0039/2024 −0.0017/2025 −0.0043/
2026 −0.0020 — 5/6年负, Q3 2026的+0.84%是2.5个月小样本瞬态。§13关闭
(+0.08% NAV/年<CI)获年度分解加强, 维持关闭; 未来自然重锚窗口候选保留。

## 17. 中枢追高过滤臂 E-ZG1 (2026-09-22, 探针→设计)

### 17.1 发现链 (probe_exec_discriminator → probe_pivot_proximity)

执行集565行全40列特征扫 → chan_pivot_zg/zd 原始价格是唯一6/6年负
Spearman候选。但原始价格混入低价股效应 (阶段5d/6已覆盖分桶)。归一化
复算分离效应 (probe_pivot_proximity_20260922.py, qfq close复算):

- **dist_zg = close/zg − 1**: rho −0.161, 6/6年负 (2021 −0.239/2022 −0.192/
  2023 −0.202/2024 −0.164/2025 −0.157/2026 −0.087); 高三分位 vs 低三分位
  fwd差 −3.4pp → **结构效应真, 非价格层代理**。
- zone_pos (0=贴下沿/1=顶住上沿): rho −0.149 6/6年负, 一致性印证。
- **执行集close≥zg (上沿上方追买) n=121/463 (26.1%), fwd −2.38% vs
  close<zg +0.49%, 胜率分化显著** — 加权cut-set −0.315 w×fwd。
- zone_pos≥2/3 n=172 fwd −2.1%/胜率33%。
- **与score完全正交** (部分相关 −0.162, 即score对追高失明 — bp7同型签名)。

### 17.2 臂设计

```python
# 新入场过滤: close≥中枢上沿 → 拒 (仅新入场, 持仓keep-alive不受影响)
_zg = self._nan_safe(getattr(sig, 'chan_pivot_zg', float('nan')))
if _zg > 0 and price > _zg:
    _rej['pivot_chase'] = _rej.get('pivot_chase', 0) + 1
    continue
```

- 位置: portfolio.py候选循环 `if code not in current_positions:` 块内,
  no_chan_penalty之后、candidates.append之前 (patch已py_compile过,
  /tmp/portfolio.py.zgarm_20260922.py)。
- **fp豁免**: portfolio.py不在信号指纹 → bp7臂结束后恢复yaml, 换
  patched portfolio.py跑bt_execution: 信号复用生产态(指纹不变),
  仅回测重跑 (~40min)。
- 预计影响: 执行buy 463→~342行 (−26%), 其中fwd −2.38%的尾桶被剔除。
  2026年执行集里若追高占比高则2026改善大 (probe显示2026 dist_zg rho
  最弱 −0.087 → 注意2026可能gain有限)。
- 变体备查: zone_pos≥2/3阈值版 (n=172 cut更大但同源)。

### 17.3 裁决标准

四指标 vs 732,689/193.08%/1.1961/17.92%, 加年度分解 (重点看2026
是否正贡献: 探针2026 rho最弱是主要风险) + 执行集替换质量。

### 17.4 regime交互分解 (2026翻转=噪声, 无条件臂成立)

regime∈{−1熊,0震荡,1牛} (MarketRegimeDetector复算, 463/463匹配):

- 熊: chase −0.77% vs keep +0.29% (差−1.1pp, n=33/90)
- **震荡: chase −4.44% vs keep +1.27% (差−5.7pp, n=34/104, 最强)**
- 牛: chase −2.07% vs keep +0.07% (差−2.1pp, n=54/148)

追高惩罚三regime全负 — 震荡市最重 (区间顶买入=买在箱体上沿, 反转系统
DNA)。2026翻转非regime交互: 2026牛n=16 chase 7行 fwd −0.17% vs keep
−4.87% 双负小样本 (sub-CI)。**结论: 无条件臂成立, 不需要regime条件变体。**
2026年翻转按噪声处理, 四指标裁决时以总体+年度分解为准。

### 17.5 执行集次强特征关闭 (免冷跑, 量化入档)

probe_exec_discriminator全表复核 (n=565执行行):

- **volume_ratio**: rho −0.040 5/6年负, 三分位差 −1.2pp — 已部分消费
  (放量下跌复合条件+ bp3缩量惩罚), 残余边际与mom/dist追高家族重叠 →
  无独立臂, 关闭。
- **mtf_discount_factor/avg_trend_strength/mtf_alignment_score**: rho
  +0.014/+0.007/+0.005, 3-4/6年 — sub-CI (差距<0.5pp), 关闭。
  mtf_discount在signal_engine读入但**未应用于score** (pre_discount_score
  为快照命名, 生产score已是blend), 字段仅CSV存证 — 死字段入档。
- **max_dd_20d**: rho +0.078 4/2年 — 反转DNA已由评分链覆盖, 无臂。
- **bp9执行坏 (n=11 fwd −3.79%/18%)**: sub-CI + E-N7已bracket过
  bp4/8/9类级加成全否决 → 关闭。
- **exhaustion_risk**: 7唯一值, 差距−0.2pp, sub-CI → 关闭。

执行集特征空间至此穷尽: 唯一6/6年稳定新候选 = 中枢追高(E-ZG1)。

### 17.6 排名层复核: score不仅失明, 还轻度偏好追高 (bp7同型签名强化)

chase行score均值 0.762 > keep 0.716, 当日排名均值2.21≈keep 2.27,
末三位占比13.2% < keep 16.1% — 追高行**不是低分填充物**, 反而挤占
prime槽位 (rank 1-2)。过滤后的替换者来自下一档score候选 = probe中的
keep行 (同rank水平fwd更优)。→ 加权cut-set (−0.315 w×fwd) 是**下界**,
替换者质量优势使实际效应可能2-3×于下界估计。与bp7执行集同签名:
score对结构信息(中枢位置)失明。

### 17.7 FAST期0.80排名地板关闭 (量化惰性, 免冷跑)

`_eff_min_rank = max(self.min_rank_pct, 0.80)` (FAST分支硬编码): FAST日
仅cap2+fast_min_score 0.30+0.80排名地板三重闸。FAST窗5.5年≈12-15段
(E-A4证据), 每段入口决策≤2槽 → 0.80地板总bind <40次入场, 且与
fast_min_score重叠。全程序CI下sub-CI → 关闭, 入死旋钮普查。

### 17.8 排名段死读×3入档 (A.6零行为修复队列+3)

portfolio排名段(line ~1195-1230) "恢复2.04的12条件"扣分制:
`vol_regime`(1204)/`div_strength_pen`(1226)/`gbc_pen`(1227) 读取后
零消费 (全文件唯一出现点=读取行) — 原2.04阶梯的残留读取, 与census
signal_engine 18+死键同型。零行为 (删除=无NAV变化), 列入9/30 A.6
零行为修复队列, 不单独跑臂。

### 17.9 bp7臂sl3冷跑裁决: 4-0全胜但超预期幅值 → bracket邻居sl4 (9/22晚)

**结果**: 767,530/207.01%/1.2655/16.02% vs 锚732,689/193.08%/1.1961/17.92%
— NAV +34,841 (+4.76%) / +13.93pp / Sharpe +0.0694 / MDD −1.90pp, 4-0全胜。
年分解: 2021 −3.52 / 2022 +4.39 / 2023 −0.84 / 2024 −5.73 / 2025 +4.30 / 2026 +7.24
(3正3负, 净+4.84pp)。

**幅值远超切集预期**: 探针切集=sl2执行边际桶18行 fwd −0.0086 (sub-CI)。
+4.76% NAV 不可能来自18行直损 — 实为**排名涟漪**: bp7+sl2行失0.06加分
→ 全局排序重排 → 边际槽换人, 与E-ZG1的rank-replacement逻辑同型。
切集是下界而非预期值, 且涟漪方向(换入者更优)事前不可测 — 臂跑是唯一裁决。

**运行有效性核验**: alt数据pkl mtime全部≤9/20 (零写入, kill-switch生效),
fp aaa78ec4 vs 生产6f1cb6b1 差异=仅yaml 7611行 sl 2→3, ML avg_IC 0.1121
与生产逐位同。对比干净。

**4-0≠充分处置**: 单臂4-0不满足采纳门槛 — 需响应面形状+复现+年分解权衡。
sl4 bracket邻居臂已launch (fp 9b6f838c, ~23:00裁决)。判定树:
  sl4 < sl3 → sl3为候选内点峰 (再补sl3复现run);
  sl4 > sl3 → 继续sl5;
  同时关注2024 −5.73pp红旗: 若sl4把2024拉回 → sl3的2024损失=噪声明证;
  若sl4 2024继续恶化 → 收紧方向对2024系统有害, 采纳需年分解权衡门。
生产yaml备份 /tmp/factor_config.yaml.prodBp7_20260922.bak (7611=2) 未动。

### 17.10 bp7臂sl4冷跑裁决: 4-0胜且逐年支配sl3, 响应面单调爬升 (9/22夜)

**结果**: 853,076/241.23%/1.3551/15.99% vs 锚732,689/193.08%/1.1961/17.92%
— NAV +120,387 (+16.43%) / +48.15pp / Sharpe +0.1590 / MDD −1.93pp, 4-0全胜。
年分解: 2021 −0.10 / 2022 +4.40 / 2023 −0.46 / 2024 −3.56 / 2025 +4.35 / 2026 +14.78
(3正3负, 净+19.41pp) — **每一年都优于sl3对应年份** (sl3: −3.52/+4.39/−0.84/−5.73/+4.30/+7.22)。
2024红旗半解除: sl3的2024 −5.73pp收窄至−3.56pp, 2021 −3.52→−0.10接近归零。

**响应面单调**: sl2(锚) 732,689 < sl3 767,530 < sl4 853,076, 且增量加速
(sl2→3 +34,841, sl3→4 +85,546) — 边际切集随阈值上升越来越负, 无内点峰迹象。

**关键事实**: 全信号signal_level max=4 (值域0-4)。sl4=仅保留最强bp7行
(2703行/1051日, 日均2.6)。**sl5=整族摘除** (b7_reject恒真, 与bp6_reject_all同型)。
判定树sl4>sl3 → 继续sl5已launch (fp待记, ~02:30裁决)。

**运行有效性核验**: alt数据pkl mtime零写入, fp 9b6f838c vs 生产6f1cb6b1
差异=仅yaml 7611行 sl 2→4, ML avg_IC 0.1121逐位同。

**2026 +14.78pp注记**: 全增益+48.15pp中2026占14.78pp。0g教训
(贡献集中2026+artifact区重合→怀疑)不适用: bp7门消费的K线/结构字段
无2026前视artifact源, 且剔除2026后仍有+33.37pp全史净增益。
若sl5裁决后进入采纳序列, 2026贡献单独入档复核。

### 17.11 bp7臂sl5裁决+flat-top闭合: 采纳sl4内点 (9/23凌晨)

**sl5结果**: 852,976/241.19%/1.3557/15.95% vs 锚732,689 — NAV +120,287
(+16.42%) / +48.11pp / Sharpe +0.1596 / MDD −1.97pp, 4-0全胜。
年分解: 2021 −0.33 / 2022 +4.84 / 2023 −1.26 / 2024 −3.34 / 2025 +4.67 / 2026 +15.07。

**sl5≈sl4 (顶flat-top, 曲线级)**: ΔNAV −100 (−0.01%) / 收益 −0.04pp /
Sharpe +0.0006 / MDD −0.04pp; 1385对齐日中有差异日1143但最大|ΔNAV|仅5,737
— 两曲线全程贴合, 差异微观。响应面: sl2(732,689) < sl3(767,530) <
**sl4(853,076) ≈ sl5(852,976)**: 单调爬升后顶部变平。

**判定树结论 — 采纳sl4 (flat-top内点, 非极端角点)**:
① sl5=整族摘除对sl4零可测增益 — 摘除整类无代价说法不成立;
② sl4零成本保留bp7最强片 (sl=4行, 2703笔/1051日) — 若该片历史WR 55.6%
对未来有任何正外推, sl4优于sl5; 若外推为零, 两者相等 → sl4弱占优;
③ 2023年分解 sl4(−0.46pp) 优于 sl5(−1.26pp);
④ 幅度全史净增益: sl4 +48.15pp中2026占14.78pp, 剔除2026仍有+33.37pp。

**度量层CI (paired block bootstrap, 2000×块10)**: sl4 ΔNAV +120,387
95%CI [−75,611, +479,076] 含0 → 单路径口径不可辨 — 与全程序CI结论一致
(9/21收口: 单路径ΔNAV无一统计可辨, 多层校验=统计必需)。本bracket承重=
四指标+逐年支配+单调响应面+机制链(Gate-1类级grade+排名涟漪, bp7切集
fwd −0.0086/39%为全类唯一负切集)+flat-top形状。

**运行有效性核验**: alt数据pkl mtime零写入(kill-switch生效), 仅yaml 7611
sl 4→5, ML avg_IC 0.1121逐位同; BP7 gate日志 70,189行(3.9%)被拒=sl5
整族摘除语义正确。

**采纳序列**: yaml 7611已改回sl4(注释终版) → sl4复现run已launch
(/tmp/bp7_sl4_repro_run.log, PID 207076, ~3.5h) — 双目的: 确定性复现
(9/14标准: 全冷重算逐位一致)+生产态再生(sidecar fp=yaml fp)。复现
通过→新生产锚 853,076/241.23%/1.3551/15.99%, fp=sl4 fp; 随后#211
门控家族分桶探针(在sl4态上测其余类下一切集, 判E-K1sl/gate爬升臂)。

**sl5臂生产备份**: /tmp/factor_config.yaml.prodBp7_20260922.bak(7611=2)
保持不动; sl4/sl5曲线备份 /tmp/bp7_sl4_arm_curve_20260922.csv /
/tmp/bp7_sl5_arm_curve_20260922.csv。

### 17.11续: sl4复现裁决通过 + 生产采纳 (9/23 06:22)

**复现冷跑** (02:40:48启动, 全链信号重生成+回测, 06:22完成, fp 12c2f8b3|0):

| 指标 | 复现 | 期望(sl4臂) | Δ |
|---|---|---|---|
| NAV | 853,076 | 853,076 | -0 |
| 收益 | 241.23% | 241.23% | +0.0004pp |
| Sharpe | 1.3551 | 1.3551 | +0.000014 |
| MDD | 15.99% | 15.99% | -0.0025pp |

- 曲线逐日diff: 1385日对齐, 不同日0, 最大|Δ|0 → 与9/22臂逐位一致.
- 信号生成: SignalStore finalized 6,463,982行, membership闸丢弃194,637行非成员.
- BP7门: 70,189行(3.9%)过sl4闸.
- **生产采纳: min_signal_level=4 (yaml 7611), 新生产锚 853,076/241.23%/1.3551/15.99%, fp 12c2f8b3|0** (注释改动→stripped digest变化, 与9/22臂fp 9b6f838c不同).
- 年分解(sl4 vs 732,689): 2021 −0.10 / 2022 +4.40 / 2023 −0.46 / 2024 −3.56 / 2025 +4.35 / 2026 +14.78pp.
- Paired CI: ΔNAV +120,387 95%CI [−75,611, +479,076]含0 → 统计不可辨; 承重=四指标+年分解+单调响应面(sl2 732,689<sl3 767,530<sl4 853,076≈sl5 852,976 flat-top)+机制链(rank-ripple).
- 曲线备份: /tmp/bp7_sl4_repro_curve_20260923.csv (复现) vs /tmp/bp7_sl4_arm_curve_20260922.csv (臂) 全等.
