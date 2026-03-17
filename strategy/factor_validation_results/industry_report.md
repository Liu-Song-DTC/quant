# 分行业因子验证报告

## 各行业最优因子汇总

| 行业 | 最优因子 | IC | IR | 胜率 |
|------|----------|-----|-----|------|
| 电力设备 | fund_score | 7.79% | 0.380 | 64.8% |
| 新能源车/风电 | fund_profit_growth | 7.82% | 0.341 | 61.5% |
| 互联网/软件 | mom_x_lowvol_20_10 | 8.17% | 0.318 | 63.2% |
| 基建/地产/石油石化 | fund_profit_growth | 7.24% | 0.281 | 61.5% |
| 有色/钢铁/煤炭/建材 | fund_score | 5.44% | 0.267 | 59.3% |
| 自动化/制造 | fund_score | 7.17% | 0.248 | 59.2% |
| 电子 | fund_profit_growth | 5.07% | 0.247 | 59.5% |
| 半导体/光伏 | mom_x_lowvol_20_20 | 6.21% | 0.220 | 58.6% |
| 化工 | fund_profit_growth | 5.04% | 0.211 | 60.3% |
| 通信/计算机 | fund_score | 5.78% | 0.211 | 58.9% |
| 金融 | fund_score | 7.36% | 0.205 | 59.5% |
| 军工 | fund_profit_growth | 7.00% | 0.196 | 57.5% |
| 消费/传媒/农业/环保/医药 | bb_width | 5.45% | 0.194 | 58.3% |
| 交运 | mom_x_lowvol_20_20 | 6.19% | 0.162 | 57.0% |

## 关键发现

### 1. 行业通用因子
- **fund_score**: 5个行业最优（电力设备、有色/钢铁/煤炭/建材、自动化/制造、通信/计算机、金融）
- **fund_profit_growth**: 5个行业最优（新能源车/风电、基建/地产/石油石化、电子、化工、军工）

### 2. 行业专用因子
- **互联网/软件**: 动量×低波动最有效（mom_x_lowvol）
- **半导体/光伏**: mom_x_lowvol 系列
- **消费/医药**: 布林带宽度（bb_width）更有效
- **交运**: mom_x_lowvol_20_20

### 3. 因子推荐优先级

#### 第一梯队（IC > 7%）
- fund_profit_growth（利润增长）
- fund_score（综合评分）
- mom_x_lowvol_20_10/20_20（动量×低波动）

#### 第二梯队（IC 5-7%）
- fund_roe（净资产收益率）
- fund_eps（每股收益）
- fund_revenue_growth（营收增长）
- bb_width（布林带宽度）

#### 第三梯队（IC 3-5%，部分行业有效）
- mom_x_lowvol_10_10/10_20
- fund_cf_to_profit（现金流/利润）
- volatility_20（部分行业）

### 4. 无效/负面因子
- 量价背离系列（divergence_*）
- rsi_oversold_rebound
- fund_debt_ratio（除军工外多为负向）
- bb_volume_confirm
- trend_volatility_combo
