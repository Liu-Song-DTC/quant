#!/bin/bash
# 接力: 等 mp 面(DONE_mp)完成后自动启动 esp 面.
# 用法: setsid nohup bash /mnt/d/quant/.arms/chain_mp_esp.sh &
#
# ⚠ 修复(2026-10-16 11:25): 旧版守卫是 `pgrep -f run_arms6.sh` — 但 run_arms6.sh 要等
#   DONE_rebr 才由 chain_rebr_mp.sh 启动, 故脚本一启动守卫就为假 ⇒ 立即自杀(10:50:47 已发生).
#   正确守卫: 只在 **rebr 已完成(DONE_rebr存在)** 且 **两个 mp 驱动都不存活** 时才判定管道死亡.
set -u
S=/mnt/d/quant/.arms
LOG=$S/logs/chain_esp4.log
while [ ! -f $S/DONE_mp ]; do
  if [ -f $S/DONE_rebr ] \
     && ! pgrep -f "run_arms6.sh"   > /dev/null 2>&1 \
     && ! pgrep -f "chain_rebr_mp.sh" > /dev/null 2>&1; then
    echo "[$(date +%m-%d_%H:%M:%S)] !!DONE_rebr 已就绪但两个 mp 驱动都不存活 — 管道死亡, 中止接力" >> $LOG
    exit 1
  fi
  sleep 60
done
echo "[$(date +%m-%d_%H:%M:%S)] mp 面完成 ⇒ 启动 esp4 面" >> $LOG
bash $S/run_arms7.sh esp
