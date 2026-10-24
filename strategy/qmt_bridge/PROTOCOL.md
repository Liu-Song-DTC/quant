# QMT 桥 — 文件握手协议 (P0, 2026-10-24)

> 前置: `VERIFY.md` 5 项全部 PASS 后才允许实现本协议。
> 架构: **大脑** = WSL `D:\quant` (信号/出单/守卫/账簿); **执行器** = QMT 客户端内置策略。
> 唯一通信介质 = 本目录下的文件 (双方可读写)。无网络通道, 无远程调用。

## 1. 文件清单与方向

| 文件 | 写方 | 读方 | 说明 |
|---|---|---|---|
| `orders_today.json` | 大脑 (generate_trade_orders 后) | QMT | 次日执行指令 + manifest |
| `bridge_state.txt` | 双方 (约定状态机) | 双方 | `PENDING_CONFIRM` → `ARMED` → `EXECUTED` |
| `fills_today.csv` | QMT (append-only) | 大脑 | 成交回报 → true-up |
| `positions_snapshot.csv` | QMT (收盘后) | 大脑 | 持仓快照 → 对账 |
| `deviation_log.csv` | QMT | 大脑 | 废单/未成交/拒单 (偏差账本) |

## 2. orders_today.json 结构

```json
{
  "run_id": "20261026",          // = 交易日, 幂等键: QMT 只执行未见过的 run_id
  "generated_at": "2026-10-25 21:14:33",
  "source_md5": "<trade_orders.json 的 md5>",
  "orders": [
    {"seq": 1, "stock_code": "688063.SH", "action": "open",
     "amount": 60000,             // 目标金额(元), QMT 侧换算整手
     "limit_price": null,         // null = 用 09:25-09:26 开盘撮合价
     "board": "STAR"}             // 科创板→200股整手; 主板→100股
  ],
  "guardrails": {"max_single_pct": 0.25, "max_day_pct": 0.60, "price_dev_max": 0.03}
}
```

大脑侧同目录写 `orders_today.md5`(指令文件 md5), QMT 读后校验一致才接受——防半写文件。

## 3. 状态机 (bridge_state.txt, 单行)

```
PENDING_CONFIRM   大脑写完订单后的初始态。QMT 看到此态 = 只读不执行。
ARMED             用户在 QMT 侧(或 make arm 式确认脚本)显式确认后写入。
                  QMT 只在 09:15 后读到 ARMED 才进入执行窗口。半自动裁定: 无确认不下单。
EXECUTED          当日窗口结束, QMT 写入 + 附 fills 行数摘要。run_id 记入 done_list。
```

- 09:30 仍未 ARMED → QMT 写 `deviation_log.csv` 一行 `SKIPPED_NOT_ARMED` 并放弃当日 (不追价不顺延)。
- `EXECUTED` 后大脑次日早报消费 fills + positions_snapshot 做 true-up。

## 4. fills_today.csv (append-only, 大脑不写)

```
run_id,seq,stock_code,order_type,price,volume,fill_id,order_status,ts
20261026,1,688063.SH,BUY,12.34,600,F20261026001,FILLED_ALL,09:25:07
```

- **幂等**: QMT 以 `(run_id, fill_id)` 去重, 重跑不重复 append。
- 废单/部分成交/拒单同样 append, `order_status` 区分 (`REJECTED` / `PARTIAL` / `UNFILLED_CANCEL`)。

## 5. QMT 侧执行自检 (下单前逐条, 任一不过 → 拒单+记偏差)

1. 整手: 科创板 ≥200 且增量 1 股递增可执行; 主板 100 的整数倍
2. 单票金额 ≤ 账户总资产 × 25%
3. 当日累计买入 ≤ 账户总资产 × 60%
4. 下单价 vs 大脑参考价(若有) 偏差 > 3% → 拒单
5. T+1 纪律: 只在 09:25-09:26 窗口下单, 开盘撮合价, **不追价不顺延不改单**

## 6. 对账闭环 (大脑侧, 次日)

- `positions_snapshot.csv` (QMT 收盘后写: stock_code,volume,cost_price,market_value,ts)
  → `live_state` true-up: 实测仓位 vs 计划账簿, 差异进早报披露。
- `fills_today.csv` → 成交均价 vs 回测假设(次日开盘价) 的滑点账本。
- 已知约束 (arch_1022): 券商回报接入前 true-up=估算; 本协议落地后变实测——**这是产品化最硬的洞, 本文件就是补它的设计**。

## 7. 时间轴 (交易日 D)

```
D-1 晚   大脑: run_daily_live.sh → trade_orders.json → 转译 orders_today.json (PENDING_CONFIRM)
         用户: 微信收到出单推送 → 决定是否确认
D 09:15  QMT 轮询: 校验 md5 → 等待 ARMED
D 09:25  ARMED 且窗口开 → 逐单执行 (自检 §5)
D 09:30  窗口关; 未执行部分记 deviation
D 15:05  QMT 写 positions_snapshot.csv → 大脑次日早报对账
```

## 8. 实施顺序 (裁定: 先模拟后真实)

1. VERIFY.md 5 项 PASS (probe_qmt.py, 模拟账户)
2. 按本协议实现 QMT 侧执行器 (读写/状态机/自检/回报落盘)
3. **2 周模拟盘**: 每日照走时间轴, fills/偏差/true-up 全链验证
4. 模拟盘零重大偏差 → 才接真实资金 (用户显式放行)
