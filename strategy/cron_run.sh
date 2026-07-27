#!/bin/bash
# 交易日自动实盘选股 — 由 cron 在每工作日 16:00 触发
# 用法: 0 16 * * 1-5 /mnt/d/quant/strategy/cron_run.sh >> /mnt/d/quant/strategy/logs/cron.log 2>&1

set -e
cd /mnt/d/quant
echo "========================================"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 实盘选股启动"

# 使用 .venv Python (与回测一致)
/mnt/d/quant/.venv/bin/python strategy/run_live.py

echo "[$(date '+%Y-%m-%d %H:%M:%S')] 实盘选股完成"
