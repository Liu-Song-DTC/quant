#!/bin/bash
# 实盘每日选股 —— 由 cron 在交易日 18:00 自动执行

set -e
cd /mnt/d/quant
echo "========================================"
echo "实盘选股开始: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"

# 安静模式开关: 默认全速(按配置并行, 正常优先级); 显式设 STRATEGY_QUIET=1 = 安静(降并行度到2, nice 19)
# 注意: 不要用 taskset 锁少数核心——会把热聚到那几颗核上, 单核打满反而更烫, 风扇更响。
# 安静模式下让 2 个进程自然散布全核, 每颗核占用低、boost 低, 更安静。
if [ "${STRATEGY_QUIET:-0}" = "1" ]; then
    NICE=19
else
    NICE=0
fi
nice -n "$NICE" .venv/bin/python strategy/run_live.py \
    --date "$(date +%Y-%m-%d)" \
    --cash 300000 \
    --skip-data

echo "========================================"
echo "实盘选股完成: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"
