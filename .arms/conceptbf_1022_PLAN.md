# conceptbf_1022 — #366 概念历史回补数据态臂 (2026-10-22)

## 状态
- 08:42 启动 (`run_conceptbf_1022.sh`), 08:52 因子缓存 `5be32292`, 09:18 ML 24 chunks
  (avg_IC 0.1139) 完成, 09:18→ 生成信号 (8 workers). ETA ~10:10 出净值.
- 基线: 984,331.4340947575 / equity md5 350208d1 / signals 94bca0cc / 1403 日.
- 生产产物在 `.arms/prodguard/conceptbf_1022/`; 臂产物将归档 `.arms/armout/conceptbf_1022/`.

## 已确认的机制语义 (读码, 决定怎么判读)
1. `backtest_factors` (fixed 生产) **不含** concept_heat; ML whitelist 也**不含**
   ⇒ 回补热值**不进评分**.
2. `signal_confidence += gate_quality*0.4` 但该字段**无消费方** (死字段).
3. 活通道 = gate_q 的两处消费:
   (i) signal_engine 卖出: `sell_confirmed |= (gate_q<0.55 and score<0)` 与
       `(score<-0.20 and gate_q<0.75)` ⇒ 2021-22 会**多出卖点**;
   (ii) portfolio 硬门槛: NEW gate_q≥0.65 / HOLD≥0.50 ⇒ 会**少入场** (HOLD 不受影响: clip 下限 0.5).
   买点(`buy`)本身不用 gate_q ⇒ 买入侧信号格只有 `gate_quality` 列改写.
   生产 2021-22 heat 恒 0.5 ⇒ G3=0.6; 回补后 heat≈0.01 ⇒ G3=0.4 ⇒ gate_q −0.0909.
4. **ML 是确定性的且与 concept_heat 无关**: 本臂 24 chunks 的样本数/验证IC/avg_IC/preds
   与生产 10-21 22:26 运行**逐位相同** ⇒ 2023+ 信号应**逐格相同** (不是"浮点噪声").
   判据(b) 因此更硬: >2023-04 的改写格应为 0; 有任何非零必须查 (不得用噪声搪塞).

## 臂结束后 (按顺序)
```bash
cd /mnt/d/quant
# 1. 确认 arm_end 输出 & 生产 md5 还原
cat .arms/logs/conceptbf_1022.log | tail -20
md5sum strategy/rolling_validation_results/{equity_curve.csv,backtest_signals.csv}
md5sum data/concept_hist.pkl    # 必须 c1a4d896… 且 mtime 21:55:56

# 2. 给判读脚本准备 tag 副本 (RVD 加 tag 文件, 不动生产名)
R=strategy/rolling_validation_results; A=.arms/armout/conceptbf_1022
for f in equity_curve.csv portfolio_selections.csv trade_realized.csv backtest_report.json; do
  cp -p $R/$f $R/$f.base_bf1022; cp -p $A/$f $R/$f.conceptbf_1022; done

# 3. 净值/胜率/票级α
.venv/bin/python .arms/probe/judge_census_1020.py   # JUDGE_BASE=base_bf1022 JUDGE_TAGS=conceptbf_1022
.venv/bin/python .arms/probe/winrate_arms.py base_bf1022 conceptbf_1022

# 4. 信号逐格 (3.2GB, 每份 ~3-5min)
.venv/bin/python .arms/sigcells.py $R/backtest_signals.csv .arms/sigdig/base_bf1022.pkl
.venv/bin/python .arms/sigcells.py $A/backtest_signals.csv .arms/sigdig/arm_bf1022.pkl
.venv/bin/python .arms/probe/bfsig_bucket_1022.py .arms/sigdig/base_bf1022.pkl .arms/sigdig/arm_bf1022.pkl

# 5. gate 反事实 (机制核对, 给理论入场拦截数)
.venv/bin/python .arms/probe/bfgate_cf_1022.py .arms/prodguard_backup_signals.csv
#   ↑ 生产信号在 arm_end 后回到 RVD, 直接用 $R/backtest_signals.csv 即可
```

## 预注册判据 (结果前写定, 不因结果改)
(a) 机制: 改写格集中在 <2023-01-03; 2023-01..04 = EMA 尾随应小; >2023-04 只允许
    ML/score 浮点微移 (量级 |Δscore| 小, 非 buy 翻转).
(b) 收益: ΔNAV ≥ 150k 地板 (或 ≥20k 且有跨年/票级一致性), CAGR/Sharpe 同看.
(c) 胜率: 1/5/20 日 + 月/年 + 逐笔, 短期与长期都不劣化 (McNemar 只算不一致期).
(d) 逐年同号 或 最大单年 |Δ| < 60% 总Δ.
(e) ex-top: 臂独有 top1 剥离后仍 > 基线.
(f) 票级 α: 臂独有 fwd10 α ≥ 基线独有 (n≥30).
(g) 机制方向: 若净效应 = 少入场, 检查被拦买点的事后收益 (若被拦的更差 ⇒ 修复为正).

## 如果采纳 (尚未决策)
1. 复现性: 重跑 `.arms/probe/concept_backfill_1022.py` 确认同一批行 (md5 092e30b6).
2. 接线: 把回补并入 `data/refresh_concept_hist_0903.py` (或在 data/ 加一次性 backfill 脚本),
   refresh 是 append-only (`tail = date > last_d`) ⇒ 已入库的 2021-22 行不会被改写.
   首行守卫只对 |r|>0.30 置零 ⇒ 我们保留的 2021-01-04 真实收益 (|r|≤0.087) 安全.
3. 重跑生产 bt 验证与臂逐位相同 (同数据+同代码) → 铸新锚 + 台账 + 记忆.
4. ⚠ PIT 残余风险: 东财板块 K 线的历史值是否用**当时**成分股计算 (非当前成分回算)
   无法直接证明; 但本臂效应 98.7% 来自"缺数据默认档 0.6 → 真实分布档 0.4"的水平校准,
   对 heat 的具体数值不敏感 ⇒ 风险主要落在 1.3% 高热度尾部的入场差异.
5. 替代方案 (若只是想修默认档): 无数据的 heat 默认 0.5→0.4 (G3 0.6→0.4) 一行代码,
   无需供应商数据/PIT 风险; 若臂为大正效应, 应跑这个"默认档臂"看能否复现.

---

## 结果 (2026-10-22 10:12 臂结束) — 判决: 不采纳 (零决策影响)

### 1. 净值面: 逐位等于生产
| 产物 | md5 | vs 生产 |
|---|---|---|
| equity_curve | 350208d1 | 逐位相同 (NAV 984,331.4340947575) |
| portfolio_selections | f4cdeb65 | 相同 |
| trade_realized | 6f0c8b3c | 相同 |
| regime_state | 89f86c5d | 相同 |
| yaogu_watchlist | a290cb9d | 相同 |
| backtest_signals | fa833261 | 不同 (生产 94bca0cc) — **唯一差异层** |

### 2. 信号逐格 (sigcells + sigcmp + bfsig_bucket)
- 6,847,747 行 / 5,321 股 / 1,403 日; 共有格全等集合, 新增/消失 **0**
- 内容改写 **1,690,061 格 = 24.68%**; 41 列里只有 2 列变:
  **gate_quality 428 日 (30.5%) / sell 218 日 (15.5%)**; `buy`/`score`/`ml_score` 恒等
- 日期桶: `<2023-01-03` **1,689,981 / 2,177,529 = 77.61%** (gate_quality 425d, sell 218d)
  · `2023-01-03..2023-04-03` **80 / 292,785 = 0.03%** (gate_quality 3d, EMA 尾随)
  · `>2023-04-03` **0 / 4,377,433 = 0.00%**
- 逐年改写: 2021 = 682,750 / 2022 = 1,007,231 / 2023 = 80
- ML 确定性已核: 24 chunks 样本数/验证 IC/avg_IC 0.1139/preds 3,227,351 与生产 10-21 运行**逐位相同**
  ⇒ 判据(a) 完全通过, 桶3 的 0 是硬结论, 不是"噪声很小".

### 3. 为什么零决策影响 (反事实 bfgate_cf_1022)
- 2021-22: buy 行 **383,814 个, 被 0.65 硬门槛新拦 0 个** (买点 gq 分布本就高, 扣 0.0909 后仍全过);
  全格层面新被拦 78,631 格 (3.61%) 但**无一落在 buy 行**; HOLD 0.50 门槛 **0 格**被拦 (clip 下限 0.5)
- Δgq 均值 **−0.0706** (G3 0.6→0.4, 覆盖 78% 格); 2021-22 生产 heat 恒 0.5 的前提成立 ⇒ 桶1 反事实有效
- sell 218 日翻转: **trades md5 逐位相同** ⇒ 无任何持仓股的卖出决策被改变 (翻转都在非持仓票)
- ⚠ 探针 2023Q1/after 桶的 287/29 个"被拦买点"是**上界伪影**: 它统一假设 old G3=0.6,
  而生产从 2023-01-03 起已有真实 heat; 实际臂 2023Q1 仅 80 格改写、之后 0 格 ⇒ 该两桶数字作废.

### 4. 判决与理由
**#366 关闭: 数据可得、回补真实且定位精确, 但零决策影响 ⇒ 不采纳.**
- fixed 模式 `backtest_factors`/ML whitelist 均不含 concept_heat; 活通道只有 gate_q 的二元门槛
- 2021-22 买点 0 个被拦 ⇒ "补真实 heat" 与 "缺数据默认 0.5" 在本数据态里对成交**完全等价**
- 故也不再跑"默认档 0.5→0.4"替代臂: 门槛不 bind, 必然同样逐位相同 (可证, 非猜测)
- 不采纳的代价为零; 采纳则白引入供应商历史 PIT 不确定性 + 数据量 ⇒ 候选文件留作审计证据

### 5. 产物与运维
- 逐格证据: `.arms/sigdig/base_bf1022.pkl` / `arm_bf1022.pkl` (93MB×2)
- 臂归档 `.arms/armout/conceptbf_1022/` (信号 md5 fa833261; 大 CSV 于磁盘清理时删除, 其余小产物保留)
- 生产已 bit-for-bit 还原: equity 350208d1 / signals 94bca0cc / concept_hist c1a4d896 (mtime 21:55:56)
- ⚠ `.signal_code_fp` 侧车仍是 `0998289a|0` (早前臂污染的旧值): 今晚 todate 前进本来就要重生成信号,
  零额外代价; 仅"同 todate 重跑"会多付 ~78min. 不做手工改写 (侧车手改有误用旧信号的风险).
