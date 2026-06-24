# 实验计划 — 2026-06-25

**目标**: 零前视偏差下 Sharpe > 1.2, 最大回撤 < 10%

---

## 〇、明天第一件事

```bash
# 确认WF-ML回测跑通
cd /mnt/d/quant/strategy
rm -f rolling_validation_results/backtest_signals.csv
python bt_execution.py 2>&1 | tee logs/bt_wf_baseline_$(date +%Y%m%d_%H%M%S).log

# 拿到结果后 grep:
grep -A 15 "向量化回测结果" logs/bt_wf_baseline_*.log
grep "ML\|训练集\|预测集\|验证IC" logs/bt_wf_baseline_*.log
```

---

## 一、系统深层问题诊断

### 问题1: B4内部严重分化（不是简单的SL=0问题）

B4门控后（SL>=2）的结构：

| SL | 买入量 | 占B4 | WR | MR |
|----|--------|------|-----|-----|
| 2 | **65,403** | **47.8%** | 53.2% | +1.21% |
| 3 | 5,834 | 4.3% | **63.1%** | **+2.34%** |
| 4 | 38,475 | 28.1% | 58.6% | +2.07% |

**根因**: SL=2占了B4的一半，但WR仅53%（刚过55%目标线以下）。SL>=3质量高但只占32%。B4门控从SL>=1升到SL>=2过滤了20%最差的，但SL=2本身也只是"勉强不亏损"。

**额外发现**: B4中trend_type=-2有670个(WR=37%)且TT=2有19,630个(WR=51%)也在拖累。当前B4门控只看SL，没看trend_type。

### 问题2: BP0（龙虎榜独立买点）完全无门控

| 指标 | 数值 |
|------|------|
| BP0买入信号 | **22,356 (4.8%)** |
| 生成路径 | dt_sig > 0.3 → 绕过所有结构/因子/价格检查 |
| 门控状态 | **零过滤** |

这些是纯龙虎榜信号——机构大买就直接buy=True。没有任何结构要求、趋势确认、估值约束。如果龙虎榜数据有噪声（上榜≠真机构买入），22K信号会系统性污染选股池。

### 问题3: 市场状态识别偏向中性

| 因子 | 占比 | 适用市场 |
|------|------|---------|
| SHARPE | **51.7%** | 中性 |
| MOM | 37.5% | 牛市 |
| REV | 10.8% | 熊市 |

系统52%的时间认为市场是"中性"。但实际上A股极端化严重：要么牛要么熊。`market_regime_detector`的阈值可能设得太宽，导致大量"中性"误判。中性时使用SHARPE因子（动量/波动率比），这个因子在极端市场中效果存疑。

### 问题4: REV(反转)因子在牛市中产生信号

REV因子50,167个信号(10.8%)，本应在熊市使用。但2025年+79.88%的牛市中也有REV信号。说明要么：
- 市场状态检测在牛市中误判为熊市
- 或者REV因子也在牛市中触发（通过BP8等买点）

### 问题5: B2条件链过长 → 信号指数衰减

```
B2信号: 4,347 (0.9%!)
条件: B1已发生 → 次级别回调 → 不破B1低点 → MACD金叉确认
```

每多一个AND条件，候选集指数衰减。4个条件串行让B2几乎消失。实际B1发生后，次级别回调+不破前低已经是强确认，MACD要求可能是多余的。

### 问题6: 因子已死，但ML训练数据也有问题

- 因子IC=0.018 → 接近随机
- 274/377行业退化>20%
- ML IC=0.176 →唯一有效
- 但：ML用91个特征，大部分是退化的因子！
- ML从噪声中提取信号的能力取决于训练数据质量
- 如果因子特征本身在退化，ML也会跟着退化

### 问题7: 组合层过度集中

| 参数 | 当前值 | 问题 |
|------|--------|------|
| max_positions | 5 | 极度集中，单票20%权重 |
| target_volatility | 0.25 | 偏高，放大了选股错误的影响 |
| 行业约束 | 无 | 5只可能全在同一板块 |
| min_hold_days | 15 | 强制持有可能在下跌中无法止损 |
| position_stop_loss | 0.12 | 12%止损偏高 |

---

## 二、修正后的实验计划

### Phase 0: WF基线确立（必须先跑）

```
目的: 拿到零前视偏差的真实Sharpe
时间: ~2小时
关键指标: Sharpe, 年收益(2021-2026), 买点分布, ML IC
```

### Phase 1: 漏洞修补（低风险高收益）

#### Exp P1: BP0龙虎榜门控
**问题**: 22K无结构信号零过滤
**改动**: 龙虎榜独立买点也要求SL>=1（至少有结构确认）
```python
# signal_engine.py
if _dt_sig > 0.3 and not hard_reject and not _is_limit_down_stock:
    sl = int(result['signal_level'][i])
    if sl >= 1:  # 新增：龙虎榜也需要有结构
        buy = True
```
**文件**: `signal_engine.py` 行621-623
**风险**: 极低（SL=0的龙虎榜信号大概率也是噪声）

#### Exp P2: B4门控加入trend_type
**问题**: B4中TT=-2的670个(WR=37%)仍在污染
**改动**: B4门控从只看SL → SL>=2 AND TT>=0
```python
if buy and bp_buy == 4 and self.b4_gate_enabled:
    if _b4_sl < 2 or _b4_tt < 0:  # 新增TT条件
        buy = False
```
**文件**: `signal_engine.py` B4门控段
**风险**: 极低（TT=-2在B4中仅0.5%但WR=37%）

#### Exp P3: B4 SL>=3门控（渐进收紧）
**问题**: SL=2的WR=53%拖累B4平均
**改动**: B4 min_signal_level: 2 → 3
**影响**: 过滤47.8%的B4信号（SL=2），保留32%高质量信号
**风险**: B4信号减少50%可能降低绝对收益
**方案**: 先做P1+P2，如果B4占比仍>50%再做P3

### Phase 2: 买点均衡（中期优化）

#### Exp A1: B2简化条件链
**分析先行**:
```python
# 读取B2信号数据，分析各条件独立贡献
# 找出哪个条件过滤了最多"原本会盈利"的信号
```
**候选改动**: 去掉MACD确认（让回调+不破前低成为充分条件）
**目标**: B2占比 0.9% → 3-5%

#### Exp A2: BP7扩展
**分析先行**: 读懂BP7触发条件，找放宽点
**目标**: BP7占比 2.8% → 5-8%

#### Exp A3: B5门控
**同B4模式**: 分析SL vs WR → 加门控
**前提**: 先有WF回测的validation数据

### Phase 3: 收益平滑（结构改进）

#### Exp B1: 市场自适应
**改两个地方**:
1. `market_regime_detector.py`: 收窄"中性"区间，减少SHARPE因子误用
2. `portfolio.py`: 熊市自动降仓

#### Exp B2: 行业分散
**portfolio.py**: 同行业≤2只

### Phase 4: 因子重建

#### Exp C1: 因子重标定
```bash
cd strategy && python offline_calibration.py
```
**预期**: IC从0.018→0.03+

---

## 三、收益不稳定 — 专项解决方案

### 根因

```
熊市: MA20 < MA60 → B4(ma_bull条件)无法触发
         ↓
   只剩 BP6/BP7/BP0/少量B1 凑数
         ↓
   信号质量不够 → portfolio "降级使用" → 硬选不够格的股票
         ↓
   熊市亏损 + 牛市暴涨 = 收益极端不均衡
```

portfolio已有仓位控制基础设施(line 640-680)，但关键缺陷在 line 1036-1042：
```python
# 当前逻辑: 选不够min_pos时"降级使用", 从不空仓
if len(selected) < min_pos:
    shortage_ratio = max(0.3, len(selected) / max(min_pos, 1))
    target_exposure *= shortage_ratio  # 凑不齐也硬上, 只是降敞口
```

### 方案: 三层防御

#### Layer 1: 信号质量门槛随市场浮动

```yaml
# factor_config.yaml
min_absolute_score_by_regime:
  bull: 0.03      # 牛市: 低门槛, 广泛参与
  neutral: 0.08   # 震荡: 中等门槛
  bear: 0.15      # 熊市: 高门槛, 宁缺毋滥
```

```python
# portfolio.py 选股
threshold = min_score_by_regime.get(market_regime, 0.05)
qualified = [c for c in candidates if c['effective_score'] >= threshold]
# 熊市高门槛→通过少→可能0-2只→自然减仓或空仓
```

#### Layer 2: 最大持仓随市场浮动

```yaml
max_positions_by_regime:
  strong_bull: 8
  bull: 6
  neutral: 4
  bear: 2       # 熊市最多2只, 允许0只(空仓)
```

#### Layer 3: 止损线随市场收紧

```yaml
portfolio_stop_loss_by_regime:
  bull: 0.12
  neutral: 0.10
  bear: 0.07     # 熊市止损更紧
```

### 预期效果

| 年份 | 市场 | 当前 | Layer1+2后 |
|------|------|------|-----------|
| 2021 | 震荡 | ? | 小幅± |
| 2022 | 熊市 | ? | -3~-5% |
| 2023 | 震荡 | ? | 小幅+ |
| 2024 | 先跌后涨 | -2.92% | +0~5% |
| 2025 | 牛市 | +79.88% | +60~80% |
| 2026 | 震荡偏牛 | +19.27% | +15~25% |

### 代码改动量

| 改动 | 文件 | 行数 |
|------|------|------|
| 信号质量门槛×regime | `portfolio.py` + `factor_config.yaml` | ~30行 |
| 最大持仓×regime | `portfolio.py` + `factor_config.yaml` | ~15行 |
| 止损×regime | `portfolio.py` + `factor_config.yaml` | ~10行 |

**总计 ~55行改动**, 不需要修改signal_engine或chan_theory。

---

## 四、执行顺序

```
明天上午:
  [必做] WF基线回测 (2h)
  [必做] 分析WF结果, 特别是2021-2023熊市年表现

明天下午 — 按优先级:
  P0-漏洞:
    [必做] P1: BP0龙虎榜SL>=1门控 (30min)
    [必做] P2: B4门控加trend_type>=0 (10min)
  
  P0-收益平滑:
    [必做] Layer1: 信号质量门槛×regime (30min)
    [必做] Layer2: 最大持仓×regime (15min)
    [建议] Layer3: 止损×regime (10min)
  
  P1-因子:
    [建议] C1: 因子重标定 (30min, 可并行)
  
  P2-买点(如时间允许):
    [可选] P3: B4 SL>=3
    [可选] A1: B2条件简化
    [可选] A2: BP7扩展

后续:
  A3: B5门控
  B2: 行业分散
```

---

## 五、执行记录

| Exp | 日期 | 改动 | Sharpe | 收益 | B4% | 结论 |
|-----|------|------|--------|------|-----|------|
| WF基线 | 6/26 | 数据扩容+WF-ML | ? | ? | ? | — |
| P1 | | BP0龙虎榜SL>=1 | | | | |
| P2 | | B4+trend_type门控 | | | | |
| P3 | | B4 SL>=3 | | | | |
| C1 | | 因子重标定 | | | | |
