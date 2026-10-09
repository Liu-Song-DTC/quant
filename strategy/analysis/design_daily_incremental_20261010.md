# 日常增量流水线设计 (A+C项) — 2026-10-10 定稿

## 目标与现状

用户裁定: 实盘流周一10/12晚恢复; 性能队列 B→D→F→E→G→A+C; 日常流目标 ≤30min。

全链现状 (10/9晚 20:54 run 实测, 数据态10/8, todate=10/9, P=8; 相界=落盘文件mtime):

| 相 | 时长 | 备注 |
|---|---|---|
| 数据刷新 (refresh_all.sh) | ~20min | 下载器/概念/另类数据 |
| 因子准备 (prep) | ~9min | 数据刷新→因子缓存失效→全量重算 (8 workers) |
| ML滚动训练 | ~24min | 24 chunks 全量重训 (冷) |
| 信号生成 (池相) | ~32min | 5349股全历史 chan+评分 (P=8; 1864ms/股单线程, 61.7% eff→378ms/股墙钟) |
| 回测 | ~21min | Backtrader 1394天 (含reindex/组合/妖股名单) |
| 出单 | ~2min | run_live Step 3 |

bt_execution 总计 ≈ 77min (ML冷) / ≈ 55min (ML缓存命中, 见下)。

## 已落地: ML chunk跨日缓存 (F/A+C核心)

- **chunk网格不变性**: chunk = `_pred_dates`(factor_df日期≥pred_start)上的index网格
  (`chunk_starts = range(0, len(_pred_dates), _retrain_freq)`)。日历锚定append-only
  → 新交易日只延长**末尾chunk**, 既有chunk的日期窗/purge/train_mask/模型全不变。
- **缓存键 = 内容指纹按chunk边界切分** (mtime+size和"去末行hash"均被否决:
  每日数据刷新改写全部K线文件→mtime指纹跨日全失效; 追加后旧末行进入hash体→窗口平移):
  - 每K线文件每行md5, 按 `date ≤ chunk.last_pred_date` 聚合 → 追加的新bar(日期>边界)
    不影响既有chunk的digest; 历史修正(≤边界的行变化)才失效。
  - margin_daily同构按边界切分; 基本面/题材pkl全局mtime; ML相关config节
    (ml/margin_features/dynamic_factor json) — 否则改feature_whitelist/min_val_ic
    会静默服务陈旧模型 (2026-10-10补上); 信号代码指纹。
  - 键含chunk身份 (chunk_start/first_pred_date/last_pred_date/train_window/fp_days)。
- **val_ic裁决链按outcome重放**: trained_good/reuse_prev/reuse_forced/
  reuse_chain_exceeded 存manifest, 重放路径维护 _val_ics/_last_good_predictor/
  _consecutive_reuse 与训练路径逐位一致。
- **不完整模型防护**: load_model静默跳过缺失ensemble成员是生产真bug —
  `_ml_chunk_load` 完整性校验 (无_ensemble_models/feature_cols → RuntimeError →
  回退全量重训), 烟测覆盖。
- **验证链**: run2冷(锚点复现+建缓存) → run3热(全命中, 逐位一致) — 通过后commit。

## 决策矩阵

| 项 | 决策 | 理由 |
|---|---|---|
| ML跨日缓存 | **采纳** | 24min→~2min, 验证链跑通后生效 |
| num_workers 8→12 | **采纳(待池相恒等验证)** | 池相32→30min(探针335ms/股@P=12 vs 378@P=8); 每股独立→结果逐位不变, 池相恒等跑一遍信号CSV对比即可 |
| prep增量 | **暂缓** | 中性化/rank已证实逐日截面(append下历史行稳定) ✓, 但factor_df含缠论结构字段(状态累积, 右端笔/线段可被新bar修订) → 需每股状态持久化+恢复, identity风险=池相同款monster; 省7min不抵风险 |
| 池相增量 | **暂缓(monster)** | 每股成本=全历史indicators一次算(1.8s/股单线程)。尾日期行需截断序列重算→更贵; 唯一省法=每股indicator数组缓存+chan状态机序列化恢复 — 右端修订语义+identity风险双重monster |
| backtest跳过/解耦 | **暂缓** | 每晚回测=数据重锚铁律的执行者(9/14撕裂/9/15危机均靠它发现); 用户次日手动执行订单 → 锚点当晚落盘即可, 解耦省21min但变更面大(run_live编排+--backtest-only模式) |
| latest_only池 | **关闭** | 每股成本由全历史indicators主导(chan状态必需), 只省_select_factor链 |

## ≤30min目标的结构性裁决

实测分解: 数据刷新20 + prep 9 + ML 2(缓存) + 池 30(P=12) + 回测 21 + 出单 2 ≈ 84min。
要 ≤30min 必须: 数据20→10 + prep→0(全缓存) + 池→3(状态序列化monster)
+ 回测→0(跳过=违反重锚铁律) — 三monster+一铁律违规, 与correctness-first风险预算冲突。
**裁决: 目标不可达, 诚实交付** — ML缓存+P=12(77→53min bt_execution),
日常流总墙钟 ≈ 73min(数据20+53)。若未来做池相状态序列化, 先过identity验证框架再议。

## 周一 10/12 实盘流恢复

全链运行 (数据刷新日=重锚日, 四指标铁律豁免仅限数据重锚)。ML缓存指纹append稳定
→ 刷新日也命中 (新bar日期>既有chunk边界) — 周一 bt_execution ≈ 55min 含全量回测+锚点。
出单门禁不变 (全量K线+白名单+新鲜度)。

## 附带: H-sb臂关闭 (冗余裁决)

用户问"买点类型多重确认是否提高胜率" → 240股53,264 buy行探针:
线段级确认(second_buy_point)是强信号 (bp2 100%是sb=1, 64.4%胜率 vs 全buy 50.0%),
**但该经济内容已被 E-K1 bp2_score_boost=0.45 定价** (sb=1中91%是bp=2)。
残余单元 (bp4×sb1 +16pp, 样本n=59; bp3×sb1 n=16) 过薄不达铁律臂标准。
笔级(bi)确认加成~0.9pp, 三源齐确认64.04% (n=648) 与 chan+线段 64.48% 无增量。
结论: 多重确认=单类确认已定价, 关闭无臂测。
