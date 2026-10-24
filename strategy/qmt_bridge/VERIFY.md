# QMT 桥 — 第 0 步能力验证清单 (P0)

> 背景: 券商已撤 miniQMT 权限, 只允许 QMT 客户端内置策略。
> 架构裁定: 本系统 = 大脑 (WSL/D:\quant), QMT 内置策略 = 执行器+回报器。
> **任何下单代码落地之前, 本清单 5 项必须全部 PASS。** 每项有判据, 逐项记录日期。

## 运行方式

QMT 客户端 → 内置策略/模型研究 → 新建策略 → 粘贴 `probe_qmt.py` → 运行 (模拟盘账户)。
输出写 `D:\quant\strategy\qmt_bridge\probe_result.txt`。

## 验证项与判据

| # | 验证项 | 方法 | PASS 判据 |
|---|--------|------|-----------|
| 1 | xtquant 可用 | `from xtquant import xttrader` (或 passorder) | import 成功, 无权限报错 |
| 2 | 交易 API 权限 | `XtQuantTrader.connect()` 连模拟账户 | connect 回调 on_connected |
| 3 | 文件交换 | 读 `D:\quant\strategy\qmt_bridge\handshake.txt` 并写回 probe_result.txt | 双向读写成功, 内容一致 |
| 4 | 1 手模拟单 | passorder 买 1 手 (200 股科创板或 100 股主板) 最小金额标的 | 委托回报+成交回报回调触发 |
| 5 | 回报落盘 | on_trade_callback 写 fills_test.csv | CSV append 成功, 字段完整 (code/price/volume/fill_id/time) |

- 任一项 FAIL → 停, 把 probe_result.txt 给 Claude 分析, **不写桥**。
- 全部 PASS → 按 PROTOCOL.md 实现桥, 2 周模拟盘后才接真实资金。

## 已知约束 (来自项目史, 写桥时必须遵守)

- 科创板 200 股整手 + 1 股递增; 主板 100 股整手
- 回测口径 = 次日开盘价成交 ⇒ QMT 侧 09:25-09:26 按开盘撮合价下单, **不追价不顺延**
- 未成交/废单 = 记录进偏差账本, 每日早报披露 (实盘=回测的偏差必须可见)
- 一切执行前自检走 PROTOCOL.md 护栏 (单票≤25%、日总≤60%、价偏>3% 拒单)
