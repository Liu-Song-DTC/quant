#!/bin/bash
# 实盘选股前一键刷新全部数据 (串行, 禁并行防OOM)
# 用法: bash data/refresh_all.sh
# 说明: K线下载 2026-09-30 起由 akshare/efinance 中继代理链路替代 Windows xtquant (QMT权限撤销)
#       raw增量 → bt全量重建 → 基本面/另类/概念 → 转债sleeve → 新鲜度门禁
set -e
set -o pipefail
cd /mnt/d/quant
PY=/mnt/d/quant/.venv/bin/python
TS=$(date +%m%d)

echo "=== 1/8 K线增量 (akshare/efinance中继代理, EM侧复权变化自动全量重载) ==="
QUANT_ALT_NO_AUTOREFRESH=1 $PY -u data/akshare_kline_downloader.py 2>&1 | tail -6

echo "=== 2/8 bt文件重建 (raw_data → backtrader_data, 无停牌填充) ==="
(cd strategy && $PY convert_raw_to_backtrader.py --workers 12 2>&1 | tail -4)

echo "=== 3/8 基本面 (最近3报告期) ==="
$PY -u data/refresh_fundamental_recent.py 2>&1 | tail -5

echo "=== 4/8 另类数据 (龙虎榜/融资融券/北向/减持/解禁/业绩预告) ==="
$PY -u data/refresh_altdata_0902.py 2>&1 | tail -12

echo "=== 5/8 概念历史 (60板块日涨跌, 东财K线集群限流时自动轮换重试) ==="
# 2026-10-12: push2his/push2delay直连被远端切断(同概念映射), 迁中继代理版
$PY -u data/refresh_concept_hist_relay.py 2>&1 | tail -6

echo "=== 6/8 概念映射 (股票→概念, push2delay直连已断, 走中继代理) ==="
$PY -u data/refresh_concept_map_relay.py 2>&1 | tail -6

echo "=== 7/8 转债sleeve数据 (cb_daily/cb_value增量, 2026-10-07激活接线; 失败仅告警不阻塞股票链) ==="
# 2026-10-15 门控: live_monitoring.cb_sleeve.enabled=false 时整步跳过.
# 依据: 回测/信号链对 cb_daily/cb_value/cb_sleeve 零读取(grep core/*.py + bt_execution.py = 0 命中),
#       关闭态下 compute_cb_sleeve() 返回空块; 本步实测耗时 ~22分钟/天(1043只券×2端点+SLEEP=0.3).
# 重新开启无需补偿: cb_sleeve_refresh.py 的"增量"= 当天已刷过才跳过, 否则按券全量重下.
# 读不到配置时保守全刷(默认 true).
CB_ENABLED=$($PY -c "
import yaml
try:
    print(str(yaml.safe_load(open('strategy/config/factor_config.yaml'))
              .get('live_monitoring',{}).get('cb_sleeve',{}).get('enabled',False)).lower())
except Exception:
    print('true')
" 2>/dev/null || echo true)
if [ "$CB_ENABLED" = "true" ]; then
  QUANT_ALT_NO_AUTOREFRESH=1 $PY -u data/cb_sleeve_refresh.py 2>&1 | tail -6 || {
    echo "!!! cb刷新失败(不影响股票出单; cb_sleeve选择器对陈旧券自动跳过)"
  }
else
  echo "  [跳过] live_monitoring.cb_sleeve.enabled=false (省 ~22分钟; 开启后自动恢复全量增量)"
fi

echo "=== 8/8 qfq负价修复安全网 (EM加性前复权会产生负价~56只, sina乘性复权修复) ==="
$PY -u data/repair_qfq_negatives.py 2>&1 | tail -4

echo ""
echo "=== 刷新完成, 跑新鲜度门禁 ==="
$PY strategy/check_data_freshness.py || {
  echo "!!! 门禁未通过, 请检查上面输出 (K线增量/bt重建失败?)"
  exit 1
}
echo "=== 全部新鲜, 可以出单: cd strategy && $PY generate_trade_orders.py ==="

echo "=== 9/8(附加) 打新提醒 (新股/新债今日申购清单→strategy/ipo_today.json, 仅信息块; 失败不阻塞) ==="
QUANT_ALT_NO_AUTOREFRESH=1 $PY strategy/tools/ipo_reminder.py 2>&1 | tail -15 || {
  echo "!!! 打新提醒获取失败(不影响出单; 手动: $PY strategy/tools/ipo_reminder.py)"
}
