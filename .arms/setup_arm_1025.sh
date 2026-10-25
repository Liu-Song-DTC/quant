#!/bin/bash
# 臂环境搭建 (2026-10-25): 从生产树派生一个可复现的臂树。
# 关键: 信号指纹一致 ⇒ 臂直接复用基线的 signals CSV (组合层臂只需 ~10min,
#       信号层臂需要重生成 ~45min), raw 因子缓存复制一份避免污染生产缓存。
# 用法: setup_arm_1025.sh <arm_name> [--fresh-signals]
set -euo pipefail
PROD=/mnt/d/quant
ARM_NAME="$1"; shift || true
FRESH=0
[ "${1:-}" = "--fresh-signals" ] && FRESH=1
ARM=$PROD/.review_runs/$ARM_NAME

if [ ! -d "$ARM" ]; then
  echo "[setup] 新建臂树 $ARM"
  mkdir -p "$ARM/strategy/rolling_validation_results" "$ARM/strategy/logs"
  rsync -a --exclude 'rolling_validation_results/' --exclude 'logs/' \
        --exclude '__pycache__/' --exclude 'cache/' --exclude 'daily_review/' \
        --exclude 'live_charts/' --exclude 'backups/' \
        --exclude 'analysis/*.parquet' --exclude 'analysis/*.csv' --exclude 'analysis/*.pkl' \
        "$PROD/strategy/" "$ARM/strategy/"
  ln -sfn "$PROD/data" "$ARM/data"
  # raw 因子缓存 (指纹一致则命中; 复制而非软链, 避免臂写穿生产缓存)
  mkdir -p "$ARM/strategy/cache"
  for f in "$PROD"/strategy/cache/factor_df_*.parquet; do
    [ -e "$f" ] || continue
    cp -f "$f" "$ARM/strategy/cache/" 2>/dev/null || true
  done
  for f in "$PROD"/strategy/cache/pool_membership_*.parquet; do
    [ -e "$f" ] || continue
    cp -f "$f" "$ARM/strategy/cache/" 2>/dev/null || true
  done
fi

if [ "$FRESH" = "0" ]; then
  # 复用基线信号 (组合层臂): 3.2GB CSV + store parquet
  for f in backtest_signals.csv backtest_signals.csv.store.parquet backtest_signals.csv.store.json; do
    src=$PROD/strategy/rolling_validation_results/$f
    dst=$ARM/strategy/rolling_validation_results/$f
    if [ -e "$src" ] && [ ! -e "$dst" ]; then
      echo "[setup] 复制信号产物 $f ..."
      cp -f "$src" "$dst"
    fi
  done
fi
echo "[setup] 就绪: $ARM"
du -sh "$ARM"
