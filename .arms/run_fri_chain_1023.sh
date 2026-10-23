#!/bin/bash
# 2026-10-23(周五) 晚间链 — 用户裁定: 今天不碰实盘; 周六日优化; 周一正式实盘.
# 目的:
#   ① 数据补齐到周五收盘 (10-22/10-23 两根K线, 上次刷新停在 10-21)
#   ② #344 首次自动 histcells diff (refresh_all 步骤9/10, 基线 base_1023pre)
#   ③ 指纹bust信号再生预付 (~85min; bust源=#348落地+alt告警编辑, 均已登记)
#   ④ todate=2026-10-23 生产回测
#   ⑤ navlaw 真闸: 对 anchor_1022 基线 (10-23 补铸), 期望共有段零改写
# 跳过: 出单 — 周日晚再出周一的单.
set -e
cd /mnt/d/quant
PY=.venv/bin/python
TODAY=2026-10-23

echo "===== 1/5 数据刷新 + 新鲜度门禁 + #344 histcells 自动diff ====="
bash data/refresh_all.sh

echo "===== 2/5 todate → $TODAY ====="
sed -i "s/^  todate: .*/  todate: '$TODAY'/" strategy/config/factor_config.yaml
grep -n "^  todate" strategy/config/factor_config.yaml

echo "===== 3/5 信号重生成+回测 (指纹bust预期内, ~85min) ====="
cd strategy
mkdir -p logs
../.venv/bin/python -u bt_execution.py > "logs/live_${TODAY//-/}.log" 2>&1
echo "回测完成 → logs/live_${TODAY//-/}.log"

echo "===== 4/5 回测富报告 ====="
../.venv/bin/python -u analysis/backtest_report.py | tee "logs/report_${TODAY//-/}.log"

echo "===== 5/5 navlaw 真闸 (基线=anchor_1022, 期望: 共有段 0 改写) ====="
set +e
../.venv/bin/python -u tools/navlaw.py watch --tail 10 --strict | tee "logs/navlaw_${TODAY//-/}.log"
_rc=${PIPESTATUS[0]}
set -e
echo "navlaw rc=$_rc (0=干净; 2=违反需排查 — 今晚不出单, 不拦任何东西, 周六判读)"

echo "===== 周五链完成 (未出单) ====="
