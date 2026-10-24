# 实盘运行手册 (runbook) — 报警处置 + 变更流程 (P0, 2026-10-24)

> 原则: **报警先问快照/数据拍在哪个事件之前, 权威对比 = 存档**, 不是记忆。
> 实盘出单铁律: 永不 `--skip-freshness`; 17:30 后运行; 不带病出单 (对不上=其中一个错了)。

## 1. 报警处置

### navlaw 净值铁律违反 (rc=2, 拦截出单)
1. 读 `strategy/logs/navlaw_YYYYMMDD.log` — 是 NAV 改写还是选股集合改写? 改写多少天?
2. 问: 今天数据刷新是否动过历史? 供应商重述 vs 真 bug。
3. 三板斧: ① OHLCV 逐位对比 (快照 `data/snapshots/daily/` 或 `.arms/datastate.py snap/diff`)
   ② navlaw 存档 `.bt_prev` 是权威昨日态 ③ 消费链查证 (改写是否传导到入池/成交)。
4. 判定=数据正当重述/重锚 → `NAVLAW_REBASELINE="<原因>"` 显式放行 (原因进日志)。
   判定=代码 bug → 修复, **绝不静默放行**; 需要今天出单且已向用户报告 → `NAVLAW_OVERRIDE="<原因>"` (明日继续报)。
5. 选股集合改写 (离散) 比 NAV 改写 (连续) 严重一级 — 必须逐日列出核实。

### histcells / sigcells 哨兵 (历史格摘要变化)
1. 先核对哨兵盯的文件是否还存在/改名 (教训: northbound 盯过死文件 `_daily.parquet`, 实读 `.pkl`)。
2. `diff` 输出按 (实体,日期) 格定位; 抽 3 格回 raw 数据人工复核。
3. 与快照对比 → 供应商重述 (登记) / 铁律违反 (修复)。

### 新鲜度门禁失败 (refresh_all.sh rc≠0)
1. 读 `logs/refresh_*.log` 最后 50 行 — 哪个子源失败。
2. 单源失败可当天跳过该源 (守卫家族会兜底报警), 但**核心 K 线失败必须修好再出单**。
3. 永远不要用 skip 参数绕过; 数据没到 = 当天不出单。

### 快照/datastate 失败 (链内 set +e, 不拦出单但必须处置)
1. `logs/snapshot_YYYYMMDD.log` — datastate 指纹失败多为 IO 超时 (NTFS 慢), 重跑一次。
2. 快照缺失当天 → 供应商重述时无"回到过去"能力, 手工补: `python strategy/snapshot_data.py`。

### 微信推送失败 (notify rc=2)
1. 出单不受影响 (订单文件已落盘); 当天人工看 `strategy/trade_orders.json`。
2. 检查 `strategy/config/secrets.env` 密钥 / 网络; Server酱 免费额度限额 (5条/日)。

### 熊市前置闸错误 (rc≥2)
1. `strategy/logs/precheck_YYYYMMDD.log`; 多为 regime 查表失败 — 检查指数数据新鲜度。

## 2. 变更流程 checklist (模板, 以 cat25 采纳 2026-10-24 为工作样例)

```
[1] 探针先行      .arms/ 预注册 (判据+配对样本+identity 烟测)         §43 臂梯 6 臂
[2] 判读          剂量单调性/胜率/ex-top/CI; 臂级唯一可信              MDD≤10% 不可达; cat25 唯一严格占优
[3] 用户裁定      改吧 → 生产 yaml (注释: 日期+依据+机制)              stock_stop_loss.enabled 等 3 键
[4] 生产复现      全量回测 (~80min; portfolio 层改动不 bust 信号指纹)   signals md5 a24bf8e2 不变 ✓
[5] 逐位验证      equity md5 对比 + signals md5 对比                   31bd8064 两轮一致 ✓
[6] navlaw        NAVLAW_REBASELINE="原因"; 检查签名: 0 选股集合改写    840d NAV / 0d selections ✓
[7] 锚+台账       anchor.json 新锚 (嵌 prev) + .arms/ledger.csv        anchor_1024 = 1,351,542.15
[8] 记忆+提交     memory topic + MEMORY.md 索引 + git commit           b860506
[9] 周日冻结链    10-25 官方全链重跑确认冻结
```

- 回测产物在信号指纹内的文件 (bt_execution.py 等) 改动 ⇒ 下一链付一次 ~80min 重生成, 属预期。
- 臂运行中**绝不编辑** portfolio.py / bt_execution.py。
- 已闭合的轴不要重开; 否决臂重验前先查锚点是否变过。

## 3. 每日节奏 (交易日)

```
16:00   Windows 计划任务更新 K 线 (QMT 客户端需登录) — 接入后生效
17:30+  run_daily_live.sh: 刷新→快照→前置闸→todate→bt(~80min)→报告→navlaw→出单→微信推送
次日早  微信推送即早报; 开盘按单执行 (09:25-09:26 撮合价, T+1, 不追价)
收盘后  positions_snapshot → true-up (桥接入后实测, 之前=估算)
```

## 4. 周末

- 周六: 链被 weekday guard 拒 (正常, 非故障)。
- 周日: 官方全链重跑 + 冻结确认 (10-25 起); 快照 weekly 层自动追加。
