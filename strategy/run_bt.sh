#!/bin/bash
# 运行回测并保存结果

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_FILE="bt_result.log"

echo "运行回测..."
cd "$SCRIPT_DIR" || exit 1

# 运行回测并保存日志
python bt_execution.py 2>&1 | tee "$LOG_FILE"

echo ""
echo "========================================"
echo "回测完成！结果已保存到 $LOG_FILE"
echo "========================================"
