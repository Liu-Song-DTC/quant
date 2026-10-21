#!/bin/bash
# 买点源级优化臂链 (2026-10-22), 基线 prod_live_1021 (NAV 984,331.43 / md5 350208d1).
# 用法: bash run_bp_arms_1022.sh base3 [min2 ...]
#   信号复用: 只改 portfolio 旋钮, 不触碰信号指纹; 每臂必须打印"复用已有信号".
#   每臂产物存档 .arms/armout/bp1022_<tag>/, 不覆盖其他臂.
set -e
cd /mnt/d/quant/strategy
PY=/mnt/d/quant/.venv/bin/python
S=/mnt/d/quant/strategy/rolling_validation_results
A=/mnt/d/quant/.arms/armout
mkdir -p "$A" logs
BASE_NAV_MD5=350208d13c79a27b2d1d548afeda5012
SIG_MD5=94bca0cc69be9f708e76364976c3957f

arm_env() {
  case "$1" in
    base3)    ;;
    min2)     export QUANT_BP_CONFIRM_MIN=2 ;;
    min4)     export QUANT_BP_CONFIRM_MIN=4 ;;
    reqs4_m3) export QUANT_BP_REQUIRE_S4=1 QUANT_BP_CONFIRM_MIN=3 ;;
    reqs4_m1) export QUANT_BP_REQUIRE_S4=1 QUANT_BP_CONFIRM_MIN=1 ;;
    w1112_m3) export QUANT_BP_WEIGHTS=1,1,1,2 QUANT_BP_CONFIRM_MIN=3 ;;
    w1112_m2) export QUANT_BP_WEIGHTS=1,1,1,2 QUANT_BP_CONFIRM_MIN=2 ;;
    *) echo "未知臂: $1"; exit 2 ;;
  esac
}

run_arm() {
  tag="$1"; arm_env "$tag"
  out="$A/bp1022_$tag"
  mkdir -p "$out"
  echo "[$(date '+%F %T')] ===== ARM $tag ====="
  # 清掉上一臂产物, 失败时不会把旧文件当本臂
  rm -f "$S/equity_curve.csv" "$S/portfolio_selections.csv" \
        "$S/trade_realized.csv" "$S/backtest_report.json" "$S/regime_state.csv"
  set +e
  $PY -u bt_execution.py > "logs/arm_bp1022_$tag.log" 2>&1
  rc=$?
  set -e
  if [ $rc -ne 0 ]; then echo "ARM $tag 失败 rc=$rc, 见 logs/arm_bp1022_$tag.log"; exit $rc; fi
  if ! grep -q "复用已有信号" "logs/arm_bp1022_$tag.log"; then
    echo "ARM $tag 未复用信号 — 拒绝, 见 logs/arm_bp1022_$tag.log"; exit 3
  fi
  sig_md5=$(md5sum "$S/backtest_signals.csv" | cut -d' ' -f1)
  eq_md5=$(md5sum "$S/equity_curve.csv" | cut -d' ' -f1)
  cp "$S/equity_curve.csv" "$out/equity_curve.csv"
  cp "$S/portfolio_selections.csv" "$out/portfolio_selections.csv"
  cp "$S/trade_realized.csv" "$out/trade_realized.csv"
  cp "$S/regime_state.csv" "$out/regime_state.csv" 2>/dev/null || true
  cp "logs/arm_bp1022_$tag.log" "$out/run.log"
  {
    echo "tag=$tag"
    echo "signal_md5=$sig_md5 (base $SIG_MD5)"
    echo "equity_md5=$eq_md5 (base $BASE_NAV_MD5)"
    echo "env=$(env | grep -E '^QUANT_BP' | sort | tr '\n' ' ')"
  } > "$out/meta.txt"
  echo "ARM $tag done: equity_md5=$eq_md5 signal_md5=$sig_md5"
  if [ "$sig_md5" != "$SIG_MD5" ]; then echo "WARN: 信号变更! $sig_md5 != $SIG_MD5"; exit 4; fi
  unset QUANT_BP_CONFIRM_MIN QUANT_BP_REQUIRE_S4 QUANT_BP_WEIGHTS
}

for tag in "$@"; do run_arm "$tag"; done
echo "[$(date '+%F %T')] 全部臂完成: $*"
