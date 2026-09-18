# 0f-v3 池层流动性地板松弛臂设计 (2026-09-18)

## 动机

OFF−honest池 ≈ 540-630k NAV (292.55%→~75%或更低) = 系统最大单一alpha源(刷新彩票)。
彩票成分分解:
- (a) **pre-admission动量微盘**: 今日流动但当年在地板下的码, OFF全史可买 —
  探针B量化: 2025年175只码被季度池锁定, 锁窗中位42天, 锁窗涨幅mean +16.9%
  (56只≥20%); 2026年62只(+8.0%)。daily池能提前准入这些码(daily 2024/2025优于
  quarterly, 同源证据), 但地板本身仍在2021/2023/2026截断alpha。
- (b) **幸存者选择**: 今日还活着的码 — 不可诚实回收, 接受损失。

方向: 在as-of honest框架内, 用**动量感知地板松弛**诚实回收(a)。
全部条件as-of评估, 无前视, 实盘get_stock_pool同规则可实现。

## 机制

`get_pool_membership_map`新增两参数 (yaml: stock_pool节):
- `pool_relax_floor: float = 1.0` — 流动性地板倍率 (0.5 = 地板减半:
  主板160M→80M, 创业板80M→40M, 科创40M→20M, volume回退同比例)
- `pool_relax_momentum: float|None = None` — 松弛准入动量门槛:
  20日close收益 ≥ 该值(如0.10)才允许按松弛地板准入; None=无门槛(纯地板松弛)

准入规则 (as-of边界b, 与现有准则串联):
1. 现有准则 (cut≥60, close≥2.0, 20d amount≥floor) → 成员 (不变)
2. 否则若 20d amount ≥ relax_floor×floor 且 (relax_momentum is None 或 20d ret ≥ relax_momentum) → 成员 (松弛准入)
3. 否则非成员

缓存键追加 relax_floor/relax_momentum (md5输入追加, 自动失效)。
掩码/闸/union自动继承新membership (消费端零改动)。

## 臂序列 (每臂一冷跑, 严格串行, 每臂~3.5-4h: mask~3min+中性化~3min+rank~40s
+ML~7min+信号重生成~2.5-3h+回测~15min; raw缓存命中)

| # | 臂 | 粒度 | relax_floor | relax_momentum | 假设 |
|---|----|------|-------------|----------------|------|
| A | monthly | monthly边界 | 1.0 | None | 粘性阶梯第三点(daily/quarterly之间), 完成granularity价值函数 |
| B | floor/2+gated | daily | 0.5 | 0.10 | 动量感知松弛 — 只放热门码进池(探针B的目标群体) |
| C | floor/2 | daily | 0.5 | None | 纯地板松弛 — 分离地板成本与动量门的价值 |
| D | 视A-C结果定 | 胜者粒度 | 胜者参数 | — | flat-top收口或关闭 |

裁决: 每臂 vs **daily纯基线**(0918d结果) 四指标铁律 + 年度分解;
铁律适用(这是策略变更不是诚实修复)。A臂同时完成monthly数据点, 其裁决对象
是granularity bracket(daily vs monthly vs quarterly, 三者均诚实) — 若monthly
非支配, granularity轴取四指标最优者为后续臂的底座。

## 优先级理由

按NAV杠杆: 池层彩票540k ≫ C5驱动(±50-100k) > C1 blend(±30-100k) > 批次2旋钮
(±10-50k)。即便诚实回收30% (~180k), 也超过其余队列总和。C5/C1顺延。
(0f-v3 A臂的monthly实现同时解答"粘性是否诚实可行" — 探针A死股续命增益的
诚实化版本, 若monthly优于daily则粘性方向成立, D臂可在胜者粒度上叠地板松弛。)

## 实现清单 (臂A起)

1. stock_pool.py: `_monthly_boundaries()` (~75边界) + membership参数扩展
   (relax_floor/relax_momentum) + 缓存键追加
2. bt_execution.py: `_load_pool_membership`支持 monthly 模式 (边界生成器分支+键)
3. yaml: pool_calendar / pool_relax_* 开关
4. generate_trade_orders: monthly = 上月未as-of (与quarterly同型)
5. 烟测: 边界数/缓存键/membership计数与手算一致
6. 冷跑A → 裁决 → B/C → D

## 数据态

全程9/17数据态不动 (bracket可比性); 实盘暂停中无干扰。
