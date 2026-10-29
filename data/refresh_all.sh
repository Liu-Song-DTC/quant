#!/bin/bash
# 实盘选股前一键刷新全部数据 (串行, 禁并行防OOM)
# 用法: bash data/refresh_all.sh
# 2026-10-28 拆成两段(取数/校验), 本文件是兼容入口:
#   refresh_fetch.sh   步骤 1-8 取数(下载/事件表/复权重建/守卫/基本面/另类/概念/转债)
#   refresh_verify.sh  新鲜度门禁 + 历史格哨兵
# 拆分的理由: 校验段失败时重跑整条链会重复下载 5,394 只(约 50 分钟);
#   拆开后重跑校验段只需秒级(取数段另有幂等标记: 名册未变且已同步到同一目标交易日 ⇒ 跳过)。
set -e
set -o pipefail
export REFRESH_LOG=${REFRESH_LOG:-/mnt/d/quant/logs/refresh_$(date +%Y%m%d_%H%M%S).log}
bash /mnt/d/quant/data/refresh_fetch.sh
bash /mnt/d/quant/data/refresh_verify.sh
