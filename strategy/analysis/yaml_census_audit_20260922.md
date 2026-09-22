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
   gated天数<60的概念回退2026Q3权重 (中报4: 47天<60→回退; Q3权重经§5的23日
   OOS验证+0.2423, 56日gated重标定方差高且无OOS余量, 用已验证先验胜于噪声重标定)。
   回退/保留清单脚本内打印。
3. 裁决权不变: 9/30写盘后全链冷跑四指标 (runbook C)。

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
