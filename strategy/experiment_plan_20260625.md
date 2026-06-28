# 实验计划 — 2026-06-25（晚间更新）

**状态**: 代码改动完成，v5回测未跑完。明天继续。

---

## 今日回测结果

| Exp | 说明 | Sharpe | 总收益 | 最大回撤 |
|-----|------|--------|--------|----------|
| WF基线 | ML时间隔离, 旧代码 | **-0.27** | -19.6% | 44.9% |
| 优化v1 | 门控+Layer1-3 | **-0.28** | -19.8% | 45.0% |

WF ML IC=0.097（前视偏差消除后），vs 旧IC=0.176。

---

## 关键数据发现

### 1. 评分系统整体IC≈0
- 旧score与future_ret的Spearman IC = -0.0012（随机）
- 评分十分位呈倒U型：中位(D5) WR=57.7%，两端(D1/D10) WR=42%/48%
- **结论：评分公式在系统性奖励错误的股票**

### 2. SHARPE因子是毒瘤
- SHARPE: 136K信号, WR=45.2%, AvgRet=-0.31%（亏损！）
- MOM: 121K信号, WR=56.0%, AvgRet=+1.71%
- REV: 18K信号, WR=61.7%, AvgRet=+3.42%（最佳）
- SHARPE占信号量50%但平均亏钱 → 这是Sharpe为负的主因

### 3. FQG后缀决定生死
- `_SM` 后缀: WR=18.5% 毒药（已封禁）
- `_T` 后缀: WR=38.2% 差
- `_V` 后缀: WR=35.8% 差
- `_F` 后缀: WR=45-62% 中等
- `_FLA` 后缀: WR=52-83% 最佳

### 4. signal_level是唯一被验证的预测因子
- SL与WR/MR单调相关
- SL=3 WR=63% > SL=4 WR=59%（SL=4涨过头）
- SL=2 WR=53%, SL=0 WR=47%

### 5. 市场状态检测从未判牛
- 旧系统MOM=0 → 2021-2026牛市从未被检测到
- 中性(SHARPE)占52%信号，但SHARPE因子平均亏损
- MOM/REV/SHARPE是同一公式不同标签（权重略有不同）

---

## 今日代码改动汇总

### signal_engine.py
- `_calculate_default_factor` 重写：score = sl_score_map[SL] + trend_type × 0.03
- SL分数映射修正：SL=3(0.65) > SL=4(0.50) > SL=2(0.30) > SL=1(0.10) > SL=0(-0.15)
- 向量化快速路径同步更新
- BP0龙虎榜门控: SL>=2
- B4门控: SL>=3 + trend_type>=0
- BP5门控: SL>=2
- BP6门控: SL>=2 + trend_type>=0
- BP7门控: SL>=2

### portfolio.py
- Layer1: min_absolute_score_by_regime (bull 0.03/neutral 0.08/bear 0.15)
- Layer2: max_positions_by_regime (bull 6/neutral 4/bear 2)
- Layer3: portfolio_stop_loss_by_regime (bull 0.12/neutral 0.10/bear 0.07)
- max_per_industry=2（同行业上限）
- min_hold_days 15→10

### market_regime_detector.py
- 中性区间收窄: 5%→3%
- 牛市检测: 4路径(新增短期爆发+EMA20>60)，mom60 3%→1%
- 中性→熊市: 不确定时默认判熊（利用REV因子的高胜率）

### stock_pool.py
- ST股票过滤: 改用stock_list_full.csv（405只ST/退市）
- 流动性过滤收紧: 主板≥2亿 / 创业板≥1亿 / 科创板≥5000万

### bt_execution.py
- ML预测 pickle→parquet（69MB/worker → ~8KB按需读取）
- 8 workers（原来14 OOM → 2 慢 → 8 快）
- ML训练/预测集时间隔离

### factor_config.yaml
- ML blend_weight: 0.80→0.20
- 所有新门控配置（bp5/6/7_gate）
- 所有regime自适应参数

---

## 2026-06-26 改动 + 实验计划

### 今日代码改动

| 层级 | 改动 | 文件 |
|------|------|------|
| 卖出端 | 分级移动止盈(15%/30%/50%三档) + 死钱退出(30天\<2%) | portfolio.py |
| 买点门控 | BP6 TT默认0, BP8 开启+SL>=2 | signal_engine.py |
| 因子库 | FactorStore持久化 + FactorLibrary时变评分 | core/factor_library.py |
| 分析器 | 因子审计脚本(IC时间线/衰减/后缀/家族/行业覆盖) | analysis/factor_library_analyzer.py |
| 股票池 | ST改为逐日判断(不再静态排除) | stock_pool.py, bt_execution.py |
| 性能 | 涨跌停预计算移到股票池过滤之后 | bt_execution.py |

### 实验矩阵

| Exp | factor_mode | 评分 | ML blend | 卖出端 | 门控 | 目的 |
|-----|-------------|------|----------|--------|------|------|
| v5 | fixed | SL+trend | 0.20 | 旧(纯止损) | 当前 | **进行中** — SL定价基线 |
| v6 | **both** | SL+trend | 0.20 | 新(分级止盈) | 收紧 | 产出 factor_library.parquet + 验证卖出端 |
| v7 | both/dynamic | 待定(看因子审计) | 0.20 | 新 | 收紧 | 因子重建后首轮 |

### v5 → v6 关键变化

| 项目 | v5 | v6 |
|------|-----|-----|
| factor_mode | fixed | **both** |
| 因子选择 | 无(跳过DYN) | **启用**(产出评估数据) |
| 卖出端 | 只有硬止损+缠论止盈 | **新增**分级止盈+死钱退出 |
| BP6门控 | TT默认-1(禁用) | **TT>=0** |
| BP8门控 | 关闭 | **开启 SL>=2** |
| FactorLibrary | 无数据 | **积累 parquet** |

### v6 目标

1. **Sharpe**: 验证 v5→v6 方向（正负判断）
2. **factor_library.parquet**: 积累 ≥60 个评估窗口的因子数据，供因子审计
3. **卖出端**: 观察分级止盈是否减少"盈利回吐"
4. **门控收紧**: 信号量下降是否提升 WR

### v5 拿到结果后的决策树

```
v5 Sharpe > 0:
  → 直接开 v6 both（产出因子数据 + 验证卖出端）
  → 跑 factor_library_analyzer.py → 因子审计报告
  → 根据审计结果进入因子重建

v5 Sharpe 仍负:
  → 先关 ML（blend_weight=0）跑一次确认
  → 仍负 → 封禁 SHARPE 因子 + B4 SL>=4
  → 转正后开 v6 both

v5 Sharpe > 0.3:
  → v6 both 同时 ML 角色改为负向过滤
```
