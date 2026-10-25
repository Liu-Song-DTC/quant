#!/bin/bash
# 交易日自动实盘选股 — 收盘数据齐备后运行
# 用法: 30 18 * * 1-5 /mnt/d/quant/strategy/cron_run.sh >> /mnt/d/quant/strategy/logs/cron.log 2>&1

set -e
cd /mnt/d/quant
echo "========================================"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 实盘选股启动"

# 使用 .venv Python (与回测一致)
bash strategy/run_daily_live.sh

echo "[$(date '+%Y-%m-%d %H:%M:%S')] 实盘选股完成"
