# B批口径修复设计 (2026-08-31)

状态: 设计定稿, 待 A批基线(A-1/A-2)完成后实施。实施时用 apply_Bfixes.py 幂等补丁 + 配对验证。

## B1 另类数据 lag1 (signal_engine.py:1513-1533)

**泄漏**: 北向资金 D 日值当晚公布、两融 D 日值次晨公布、龙虎榜 D 日晚间公布,
但引擎循环把 D 日值直接加进 D 日收盘信号。provider 内部是 `date <= 查询日` as-of 口径,
所以修法=调用处传 D-1 (signal_engine 无 datetime import, 用 `pd.Timedelta` 即可):

- L1515: `nb = self._alt_data.get_northbound_signal(bar_date)` → `get_northbound_signal(bar_date - pd.Timedelta(days=1))`
- L1516: 同上 get_margin_signal
- L1530: `dt_sig = self._alt_data.get_dragon_tiger_signal(code, pd.to_datetime(dates[i]).date())` → `.date() - pd.Timedelta(days=1)` 注意 getter 内部 `str(date)[:10]` 兼容 Timedelta 减后的 date 对象

ML 路径已 T-1 安全, 不动。实盘影响: 8/31 早间运行时 8/28 值已公开, lag1 使实盘略保守, 换取回测口径诚实 (用户已同意)。

## B2 BOM as-of + 接线 (signal_engine.py:610,1125-1167,1635,953)

**现状**: `_get_bom_score(code)` 每代码一次标量、无日期参数; `row = fd[code].iloc[-1]` 取全历史最后一行;
且 `getattr(row, 'roe', ...)` 对中文列 DataFrame 永远取默认值 → 基本面部分恒为 0.10/0.30/0,
BOM 实际只随概念(静态)变。即: 泄漏路径被 getattr 失效意外"挡住", 但基本面接线是死的。

**修复** (FundamentalData 已有 `_get_available_data(code, date)` as-of 过滤, 直接复用):

1. `_get_bom_score(self, code, date=None)`:
   - date is None: 保留旧行为 (兼容)
   - date 给定: `df = fd._get_available_data(code, date)`, 取 `df.iloc[0]` (已按报告期降序);
     映射中文列: `roe = 净资产收益率/100 (str→float去%)`, `gross_margin = 销售毛利率/100`, market_cap=0;
     缓存键 `(code, date)`, diag 照旧
2. 主流程 Phase 0 (L610) 标量 → 逐bar数组:
   ```python
   bom_arr = np.full(n, 0.3)
   if code:
       for i in range(60, n):
           bom_arr[i] = self._get_bom_score(code, pd.to_datetime(dates[i]).date())
   ```
   成本: 缓存命中 O(1), 仅在报告变更日 miss (每代码全年~5-10次), 与既有 concept_heat 逐bar循环同量级
3. L618 传 `bom_score=bom_arr`; L1635 `bom_mult_arr = 0.7 + 0.6 * np.asarray(bom_score, dtype=float)`
4. L953 `bom_quality_score=float(bom_score)` → `float(bom_arr[i])`

## B3 T+1 开盘成交 (bt_execution.py:1296-1618)

**现状**: D 日收盘信号按 D 日收盘价成交 (T-close fill), 收益系统性偏乐观。

**修复**: 挂单模型 — D 日收盘生成 target, D+1 开盘成交:

1. 矩阵 (L1310-1326): 增加 `open_px` 矩阵, `usecols=['datetime','open','close','volume']`,
   open 同样 reindex(calendar)+ffill
2. 循环重构 (L1448-1608):
   - 循环体内, 在 nav[i] 计算**前**, 若 `_pending` 非空 → 用 `open_px[i]` 执行前一日挂单:
     - 卖出: `sell_px = open_px[i,j]*(1-slip)`; 跳空低开 >9.5% → ×0.97 (prev=close_px[i-1]);
       跌停拦阻用 `_limit_down[i,j]`; T+1 禁卖簿 `_today_buys` 在执行日清空重建
     - 买入: 涨停拦阻 `_limit_up[i,j]`; `buy_px = open_px[i,j]*(1+slip)`;
       跳空高开 >9.5% → ×1.03; 现金不足按比例缩放 (先卖后买, 卖在开盘释放现金);
       成交后 `_entry_dates[code] = calendar[i].date()` (以成交日为入场日),
       `_log_realized(code, calendar[i].date(), ...)` 同理
   - generate_positions(date=D) 仍在 D 日收盘调用 (组合状态/止损判定按 D 收盘), 产出存 `_pending = target`
   - 末日无 D+1 → 挂单作废; 首日无挂单
   - `_today_buys` 每日执行前 clear, 执行后记录当日新买 (禁当日卖)
3. 不改 selections 记录时点 (仍按信号日 D 记录)

**验证配对**: B-baseline (8/27池 + POOL_TODATE=2026-08-28 两抽取) vs A-baseline (A-1/A-2)。
预期: 收益/Sharpe 下降 (乐观偏差被移除), DD 略改善或持平; 买卖点时段错位一天。

## 实施顺序

1. A批基线出数后: apply_Bfixes.py (B1+B2+B3, 同 apply_Afixes.py 幂等+回滚模式)
2. py_compile + B-baseline 双抽取 (artifacts *.B1/*.B2, 日志 strategy/logs/)
3. 报用户: A 批 delta 表 + B 批 delta 表 (B 批为权威口径)
4. 之后才允许开调优实验 (队列: 方向H/熔断机制等在权威口径上重排)
