#!/bin/bash
# 信号层臂 (2026-10-25): 只改 ml 节 ⇒ raw 因子缓存命中(ml已在_RAW_YAML_SKIP_SECTIONS),
# 只需重训 ML + 重生成信号 + 回测 (~30-45min), 不触发全市场因子重算。
# 用法: setup_arm_ml_1025.sh <臂名> <python片段文件>   # 片段里直接改 yaml 文本
set -euo pipefail
PROD=/mnt/d/quant
ARM_NAME="$1"; PATCH_PY="$2"
ARM=$PROD/.review_runs/$ARM_NAME

if [ ! -d "$ARM" ]; then
  echo "[setup] 新建信号层臂树 $ARM"
  mkdir -p "$ARM/strategy/rolling_validation_results" "$ARM/strategy/logs"
  rsync -a --exclude 'rolling_validation_results/' --exclude 'logs/' \
        --exclude '__pycache__/' --exclude 'cache/' --exclude 'daily_review/' \
        --exclude 'live_charts/' --exclude 'backups/' \
        --exclude 'analysis/*.parquet' --exclude 'analysis/*.csv' --exclude 'analysis/*.pkl' \
        "$PROD/strategy/" "$ARM/strategy/"
  ln -sfn "$PROD/data" "$ARM/data"
  mkdir -p "$ARM/strategy/cache"
  for f in "$PROD"/strategy/cache/factor_df_*.parquet "$PROD"/strategy/cache/pool_membership_*.parquet; do
    [ -e "$f" ] || continue
    cp -f "$f" "$ARM/strategy/cache/" 2>/dev/null || true
  done
fi

/mnt/d/quant/.venv/bin/python "$PATCH_PY" "$ARM/strategy/config/factor_config.yaml" "${@:3}"
echo "[setup] 臂就绪: $ARM"
# 校验: ml 节改动**不得**进 raw 指纹 (否则要重算全市场因子), 但**必须**进信号指纹
/mnt/d/quant/.venv/bin/python - "$ARM/strategy/config/factor_config.yaml" <<'EOF'
import sys
sys.path.insert(0, '/mnt/d/quant/strategy')
from core.factor_preparer import _yaml_raw_digest, _yaml_stripped_digest
p = sys.argv[1]
print(f'[verify] 臂 raw digest   = {_yaml_raw_digest(p)}')
print(f'[verify] 生产 raw digest = {_yaml_raw_digest("/mnt/d/quant/strategy/config/factor_config.yaml")}')
print(f'[verify] 臂 信号 digest  = {_yaml_stripped_digest(p)}')
EOF
