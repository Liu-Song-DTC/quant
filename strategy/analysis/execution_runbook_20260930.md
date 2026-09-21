# 2026-09-30 执行手册 (20天程序收口日, 2026-09-21预写)

当日三件套: **锚点重对账 → V2 OOS审计 → Q4标定**, 之后恢复实盘流。
全程串行, 全程 `QUANT_ALT_NO_AUTOREFRESH=1`, 所有运行显式 `/mnt/d/quant/.venv/bin/python`。

## 0. 前置 (用户侧, 9/30收盘后)

1. Windows下载器拉到 2026-09-30 (全量K线+另类数据)。
2. `cd /mnt/d/quant && ./data/refresh_all.sh` (或等价) — 受控更新, 运行中零自动刷新。
3. `cd strategy && /mnt/d/quant/.venv/bin/python check_data_freshness.py` — 门禁必须过。
4. 预期变化: 数据态 9/17 → 9/30, 锚点必然漂移 (池滑动+历史更正双因素, 先例: 每次刷新
   −5万~−13万级漂移), 漂移必须归因, 不默认接受。

## A. 锚点重对账 (约90min全链, 一次)

```bash
cd /mnt/d/quant/strategy
QUANT_ALT_NO_AUTOREFRESH=1 /mnt/d/quant/.venv/bin/python bt_execution.py
```
- 新锚点四指标 vs 9/17态 **732,689/193.08%/1.1961/17.92%** (fp 6f1cb6b1|0)。
- 若漂移 >±3万: 归因三件套 — ①pool_flip_report (池成员翻转) ②K线重拉检查
  ③另类数据diff — 参照 9/14晚流归因协议 (单日期全截面raw重算+中性化复现)。
- 产出新sidecar fp; 归档 equity_curve 为9/30基线。

## B. V2 OOS审计 (只读, 先于Q4标定)

```bash
cd /mnt/d/quant/strategy
QUANT_ALT_NO_AUTOREFRESH=1 /mnt/d/quant/.venv/bin/python analysis/v2_oos_runbook_0930.py
```
- Phase 0 前提自检 → Phase 1 尾部因子值 → Phase 2 Q3 IC审计 → Phase 3 标定重核
  → Phase 4 Q3实现审计 → Phase 5 决策清单 (不自动跑全链)。
- 读数纪律: **按年分解** (F3/F4教训: 时段集中优势不可外推); 加权IC与符号一致率
  双口径 (9/17早读P2'模板); P3'标定diff按数据修订分级 (良性修订 vs 权重翻转)。

## C. Q4标定 (写盘, 在B之后)

```bash
cd /mnt/d/quant/strategy
QUANT_ALT_NO_AUTOREFRESH=1 /mnt/d/quant/.venv/bin/python analysis/calib_2026Q4_tail.py
```
- 只写 2026Q4.yaml + index.yaml 追加条目, 其余23个季度文件零扰动。
- **写盘后必须再跑一次全链裁决** (E-D2纪律: quarterly_factors 在信号指纹内,
  信号必重生成): 四指标 vs 9/30基线对账, 铁律裁决 — 采纳或硬回退全局权重。

## D. 实盘流恢复 (用户核对后)

1. **已知阻塞 (9/17遗留)**: current_positions.json 为 21只/993,544 > 1.5×50万 →
   0h现金闸拒绝出单 (设计行为)。用户必须核对真实账户持仓后二选一:
   手动修正持仓文件 / 确认无真实持仓后 `--force-reset`。**此步不可跳过, 不可代核**。
2. 出单: `QUANT_ALT_NO_AUTOREFRESH=1 /mnt/d/quant/.venv/bin/python generate_trade_orders.py`
3. 逆回购sweep激活决策 (预备已落地 ff49580, 默认关闭):
   `live_monitoring.repo_sweep.enabled: false → true` — 闲置现金≈56.9% → GC001隔夜
   ≈0.91% NAV/年; QMT侧以实际可用余额为准, 股票单全部成交后最后下 (卖出204001.SH,
   1000元整数倍)。建议先观察1-2个交易日的出单-成交对账再开启。
4. 出单后核对: trade_orders.json vs 账户实际成交 (价格滑点/部分成交/涨停买不进)。

## E. 对账检查点清单

- [ ] fp sidecar 与当前代码指纹一致 (bt_execution._signal_code_fingerprint)
- [ ] 数据态=9/30 (数据新鲜度门禁过)
- [ ] 9/30基线四指标+年份分解入档 (MORNING_REPORT模板)
- [ ] Q4标定后四指标裁决入档
- [ ] 持仓文件与真实账户对账闭环 (D.1阻塞解除)

## 回滚预案

- Q4标定裁决失败 → 硬回退全局权重 (E-K1先例: 原22季度文件零扰动, 7/1前净值逐位一致)。
- 9/30数据异常 → 数据态回退9/17, 按 9/14隔离重建协议处理, 不在此手册范围自动执行。
