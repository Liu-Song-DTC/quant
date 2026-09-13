#!/bin/bash
# P0-1 Stage2 跑后恢复: 先把Stage2产物存入备份, 再复原生产 yaml + 6输出 + sidecar
# 幂等: 重复执行安全 (备份目录中的stage2_*文件会被覆盖为最新, 原文件被原样复原)
set -e
BK=/mnt/d/quant/strategy/backups/stage2_0913
RVR=/mnt/d/quant/strategy/rolling_validation_results
CFG=/mnt/d/quant/strategy/config/factor_config.yaml

echo "== 1. Stage2产物归档到备份 =="
for f in backtest_signals.csv .signal_code_fp equity_curve.csv regime_state.csv \
         portfolio_selections.csv trade_realized.csv; do
  if [ -f "$RVR/$f" ]; then
    cp "$RVR/$f" "$BK/stage2_$f"
    echo "  archived stage2_$f"
  fi
done

echo "== 2. 恢复生产文件 =="
for f in backtest_signals.csv .signal_code_fp equity_curve.csv regime_state.csv \
         portfolio_selections.csv trade_realized.csv; do
  cp "$BK/$f" "$RVR/$f"
done
cp "$BK/factor_config.yaml" "$CFG"
echo "  yaml + 7输出已复原"

echo "== 3. 验证 =="
grep -c "fundamental_data_deadline" "$CFG" || echo "  yaml已无deadline路径 ✓"
/usr/bin/md5sum "$CFG" "$BK/factor_config.yaml" | awk '{print $1}' | sort -u | wc -l | \
  xargs -I{} sh -c '[ {} -eq 1 ] && echo "  yaml与备份一致 ✓" || echo "  !!! yaml与备份不一致"'
cat "$RVR/.signal_code_fp"; echo "  <- sidecar (应与原18a40e81一致)"
