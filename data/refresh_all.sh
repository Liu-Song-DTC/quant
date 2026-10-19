#!/bin/bash
# 实盘选股前一键刷新全部数据 (串行, 禁并行防OOM)
# 用法: bash data/refresh_all.sh
# 说明: K线下载 2026-09-30 起由 akshare/efinance 中继代理链路替代 Windows xtquant (QMT权限撤销)
#       raw增量 → 事件表 → 表驱动复权重建 → 基本面/另类/概念 → 转债sleeve → 新鲜度门禁
set -e
set -o pipefail
cd /mnt/d/quant
PY=/mnt/d/quant/.venv/bin/python
SD=/mnt/d/quant/data/stock_data
# 每一步的**完整**输出落这里; 控制台仍只留尾部 N 行。
# ⚠ 2026-10-19: 原来直接 `2>&1 | tail -N` ⇒ 只留最后 N 行, 而 `SystemExit` 的错误
#   消息会被随后 atexit 的 6 条 resource_tracker 告警**挤出窗口** ⇒ **报错被吞**。
#   实测把一个"退市股在东财无页 ⇒ 落盘拒绝"的**死锁**藏了整条链(排查 3 小时)。
#   `tail` 只该用来**缩短控制台**, 不该用来**丢弃证据**; `set -o pipefail` 保证
#   退出码仍如实传播(cmd 失败 ⇒ 整条管道非零)。
mkdir -p /mnt/d/quant/logs
REFRESH_LOG=/mnt/d/quant/logs/refresh_$(date +%Y%m%d_%H%M%S).log
echo "完整日志: $REFRESH_LOG"

echo "=== 1/9 K线增量 (akshare/efinance中继代理, EM侧复权变化自动全量重载) ==="
QUANT_ALT_NO_AUTOREFRESH=1 $PY -u data/akshare_kline_downloader.py 2>&1 | tee -a "$REFRESH_LOG" | tail -6

# ── 2/9 事件表 ────────────────────────────────────────────────────────────────
# 复权口径 = **表驱动乘性**(2026-10-17 接入). 旧链路 `convert_raw_to_backtrader.py`
# 直通东财**加性**前复权(qfq=(raw−C)/Π), 在非除权日不保收益:
#   实测失真 `ret_q/ret_n = raw·Π/qfq` 随累计分红单调增 ⇒ amp>1.10 占 13.9%, 尾部 7.2×;
#   合成一次未来分红 ⇒ **97.01% 历史日漂移**(乘性 0.05%)。且东财每次除权**整条重锚**
#   ⇒ 违反铁律"更新今天不得改写历史"。
# 新口径: W(0)=raw(0); 事件日 W(t)=W(t−1)·((1+s)·raw(t)+D)/raw(t−1); 否则 W(t−1)·raw(t)/raw(t−1)。
#   非事件日 ⇒ 复权收益 ≡ 裸收益(**恒等式**, 零自由参数)。
echo "=== 2/9 公司行动表 (东财 stock_fhps_detail_em) + 新浪乘性因子 → 融合事件表 ==="
# 东财表**不是完备事件源**(结构性不覆盖配股/重整转增; 漏"特别股息"), 新浪乘性因子补洞。
# 两者融合规则见 data/build_events_table.py 的 docstring(逐事件, 无自由阈值)。
# ⚠ 每天都要拉: 事件**必须在除权日之前**进表, 否则事后补 = 改历史(锚漂移)。
#   未到期的除权日在重建里被 `pos >= n ⇒ continue` 忽略 ⇒ 提前进表零副作用。
$PY -u data/fetch_actions_table.py 2>&1 | tee -a "$REFRESH_LOG" | tail -6
$PY -u data/fetch_sina_factors.py 2>&1 | tee -a "$REFRESH_LOG" | tail -6
$PY -u data/build_events_table.py 2>&1 | tee -a "$REFRESH_LOG" | tail -10

# ── 3/9 表驱动复权重建 ────────────────────────────────────────────────────────
echo "=== 3/9 bt文件重建 (表驱动乘性复权, 建到临时目录后原子换名) ==="
BT_NEW=$SD/.bt_new
BT_OLD=$SD/.bt_prev
rm -rf "$BT_NEW"
$PY -u data/rebuild_table_qfq.py \
    --events /mnt/d/quant/data/alternative_data/events_table.pkl \
    --anchor frozen --out "$BT_NEW" 2>&1 | tee -a "$REFRESH_LOG" | tail -12
# 落盘守卫: 新目录必须与生产**符号集完全相同**(缺一个 = 该股票从池子里消失)
# ⚠ 2026-10-17: 守卫原来只查 `*_qfq.csv` **一个方向** ⇒ 生产目录里的**非 qfq 文件**
#   会在 `mv` 换名时被**静默丢弃**(换名是整目录替换, 不是合并)。
#   实测代价: 10-17 10:59 换名丢掉了 57 个 `*.preQfqFix*`(09-03 修复脚本留在目录里的
#   取证备份), 5,437 → 5,381, 守卫一声不吭。备份在 `.arms/snapshots/test_1016/` 与
#   `data/.preFullRefresh_0904/` 各有一份 ⇒ 未丢数据, 但**静默**本身是缺陷。
#   修法: 换名前把生产目录里所有非 `_qfq.csv` 文件**原样搬进新目录**(严格超集),
#   并大声列出来 —— 未知文件既不丢也不静默。
$PY - "$SD" "$BT_NEW" <<'EOF'
import glob, os, shutil, sys
sd, new = sys.argv[1], sys.argv[2]
prod = f'{sd}/backtrader_data'
p = {os.path.basename(x)[:-8] for x in glob.glob(f'{prod}/*_qfq.csv')}
t = {os.path.basename(x)[:-8] for x in glob.glob(f'{new}/*_qfq.csv')}
if p - t:
    raise SystemExit(f"[换名守卫] 新目录缺 {len(p - t)} 只: {sorted(p - t)[:10]} ⇒ 拒绝换名")
extra = sorted(f for f in os.listdir(prod)
               if not f.endswith('_qfq.csv') and os.path.isfile(f'{prod}/{f}'))
for f in extra:
    if not os.path.exists(f'{new}/{f}'):
        shutil.copy2(f'{prod}/{f}', f'{new}/{f}')
if extra:
    print(f"  ⚠ 生产目录有 {len(extra)} 个非 qfq 文件, 已原样搬入: {extra[:8]}"
          f"{' …' if len(extra) > 8 else ''}")
print(f"  符号集核对: {len(t)} 只, 与生产完全一致")
EOF
# 原子换名: mv 是原子的 ⇒ 任何时刻 backtrader_data 都指向完整目录。
# 快照(datastate.py)是**硬链接**, 换名不影响其 inode ⇒ 历史快照不被污染。
rm -rf "$BT_OLD"
mv "$SD/backtrader_data" "$BT_OLD"
if ! mv "$BT_NEW" "$SD/backtrader_data"; then
  echo "!!! 换名失败, 回滚"; mv "$BT_OLD" "$SD/backtrader_data"; exit 1
fi
echo "  已换名; 上一版留在 $BT_OLD (下次刷新时清除)"
# 表驱动生产哨兵 (2026-10-18): 遗留加性链路的守卫靠它判"这是生产库"。
# 放在 $SD **而不是** backtrader_data/ 里 —— 目录每次刷新整体换名, 放里面会被换掉。
# 与 qfq_anchor.json 并列(任一存在即触发守卫): 锚是业务文件(迁移可能正当删除),
# 本哨兵是专用标记, 无其它删除理由 ⇒ 删锚不会让守卫静默失效。
touch "$SD/.table_driven_prod"

echo "=== 4/9 基本面 (最近3报告期) ==="
$PY -u data/refresh_fundamental_recent.py 2>&1 | tee -a "$REFRESH_LOG" | tail -5

echo "=== 5/9 另类数据 (龙虎榜/融资融券/北向/减持/解禁/业绩预告) ==="
$PY -u data/refresh_altdata_0902.py 2>&1 | tee -a "$REFRESH_LOG" | tail -12

echo "=== 6/9 概念历史 (60板块日涨跌, 东财K线集群限流时自动轮换重试) ==="
# 2026-10-12: push2his/push2delay直连被远端切断(同概念映射), 迁中继代理版
$PY -u data/refresh_concept_hist_relay.py 2>&1 | tee -a "$REFRESH_LOG" | tail -6

echo "=== 7/9 概念映射 (股票→概念, push2delay直连已断, 走中继代理) ==="
$PY -u data/refresh_concept_map_relay.py 2>&1 | tee -a "$REFRESH_LOG" | tail -6

echo "=== 8/9 转债sleeve数据 (cb_daily/cb_value增量, 2026-10-07激活接线; 失败仅告警不阻塞股票链) ==="
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
  QUANT_ALT_NO_AUTOREFRESH=1 $PY -u data/cb_sleeve_refresh.py 2>&1 | tee -a "$REFRESH_LOG" | tail -6 || {
    echo "!!! cb刷新失败(不影响股票出单; cb_sleeve选择器对陈旧券自动跳过)"
  }
else
  echo "  [跳过] live_monitoring.cb_sleeve.enabled=false (省 ~22分钟; 开启后自动恢复全量增量)"
fi

# 原 8/8 `repair_qfq_negatives.py`(新浪乘性复权修 52 只负价)**已删除**(2026-10-17):
#   它是在**输出**上打补丁, 而表驱动重建从 (raw + 事件表) 重新构造 ⇒ 负价在输入侧就不存在。
#   留着它反而会把新浪口径混进表驱动产物(两种口径混用)。

echo ""
echo "=== 刷新完成, 跑新鲜度门禁 ==="
$PY strategy/check_data_freshness.py || {
  echo "!!! 门禁未通过, 请检查上面输出 (K线增量/bt重建失败?)"
  exit 1
}
echo "=== 全部新鲜, 可以出单: cd strategy && $PY generate_trade_orders.py ==="

echo "=== 9/9(附加) 打新提醒 (新股/新债今日申购清单→strategy/ipo_today.json, 仅信息块; 失败不阻塞) ==="
QUANT_ALT_NO_AUTOREFRESH=1 $PY strategy/tools/ipo_reminder.py 2>&1 | tee -a "$REFRESH_LOG" | tail -15 || {
  echo "!!! 打新提醒获取失败(不影响出单; 手动: $PY strategy/tools/ipo_reminder.py)"
}
