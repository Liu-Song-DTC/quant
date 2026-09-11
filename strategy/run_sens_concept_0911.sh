#!/bin/bash
# 阶段6b(2026-09-11): 概念前视敏感性双跑 — 量化"净值/残差alpha里未来概念占多少"
#   armA: 静态剥离未来概念(70+清单 + 预增/预盈/预减子串) — 保守上界(2023+概念
#         在2023-26的合法使用也被剥, 方向=高估污染影响)
#   armB: 概念映射全空 — 硬上界(概念路径整体关闭, 含合法概念alpha)
# 基线: 979,885/291.95%/1.4094/18.33%
# 协议: 备份map+anchors+signals+sidecar → armA全链(~90min, map mtime触发因子+信号
#       缓存失效) → 存臂产物 → armB全链 → trap恢复全部。串行, 不要并行(会oom)。
set -u
cd /mnt/d/quant
VENV=.venv/bin/python
MAP=data/stock_concept_map.pkl
RVR=strategy/rolling_validation_results
echo "=== 概念敏感性双跑 开始 $(date '+%H:%M:%S') ==="

# ---------- 0) 一次性备份 ----------
if [ ! -f "$MAP.preConcept0911" ]; then
    cp "$MAP" "$MAP.preConcept0911"
    echo "[backup] stock_concept_map.pkl -> .preConcept0911"
fi
if [ ! -f "$RVR/backtest_signals.preConcept0911.csv" ]; then
    cp "$RVR/backtest_signals.csv" "$RVR/backtest_signals.preConcept0911.csv"
    cp "$RVR/.signal_code_fp" "$RVR/.signal_code_fp.preConcept0911" 2>/dev/null
    cp "$RVR/portfolio_selections.csv" "$RVR/portfolio_selections.preConcept0911.csv"
    cp "$RVR/trade_realized.csv" "$RVR/trade_realized.preConcept0911.csv"
    cp "$RVR/equity_curve.csv" "$RVR/equity_curve.preConcept0911.csv" 2>/dev/null
    echo "[backup] signals+sidecar+anchors -> .preConcept0911"
fi

restore() {
    cp "$MAP.preConcept0911" "$MAP"
    echo "[restore] stock_concept_map.pkl"
    for f in backtest_signals portfolio_selections trade_realized equity_curve; do
        if [ -f "$RVR/$f.preConcept0911.csv" ]; then
            cp "$RVR/$f.preConcept0911.csv" "$RVR/$f.csv"
            echo "[restore] $f.csv"
        fi
    done
    [ -f "$RVR/.signal_code_fp.preConcept0911" ] && cp "$RVR/.signal_code_fp.preConcept0911" "$RVR/.signal_code_fp" && echo "[restore] .signal_code_fp"
}
trap restore EXIT

# ---------- 1) armA: 剥离未来概念 ----------
$VENV - <<'EOF'
import pickle
FUTURE = set('''ChatGPT概念 CPO概念 AIGC概念 AI智能体 AI手机 AI眼镜 AI制药（医疗） AI医疗 低空经济
人形机器人 数据要素 液冷服务器 液冷概念 华为昇腾 英伟达概念 Sora概念 合成生物 车路云 量子科技
卫星互联网 商业航天 可控核聚变 深海科技 固态电池 钙钛矿电池 HBM概念 玻璃基板 AI语料 AI应用
Kimi概念 文生视频 多模态AI 6G概念 脑机接口 室温超导 超导概念 新质生产力 飞行汽车 星闪概念
卫星通信 通感一体化 量子计算 低轨卫星 灵巧手 机器人执行器 算力概念 算力租赁 数据确权 数字中国
大模型概念 文心一言 通义千问 讯飞星火 智谱AI AI服务器 AIPC AI芯片 AI办公 AI教育 AI安全
液冷温控 硅光子 Chiplet概念 先进封装 CoWoS 减肥药 GLP-1概念 司美格鲁肽 微短剧 短剧游戏
出海概念 小米汽车 BC电池 半固态电池 液冷超充 新型工业化 石英砂概念 萝卜快跑 Robotaxi
MR混合现实 空间计算 苹果MR VisionPro 5.5G概念 英伟达产业链 液冷IDC'''.split())
BAD_KW = ('预增', '预盈', '预减')
raw = pickle.load(open('data/stock_concept_map.pkl', 'rb'))
n_before = sum(len(cs) for cs in raw.values())
out = {}
for code, cs in raw.items():
    keep = [c for c in cs if c not in FUTURE and not any(k in c for k in BAD_KW)]
    if keep:
        out[code] = keep
n_after = sum(len(cs) for cs in out.values())
pickle.dump(out, open('data/stock_concept_map.pkl', 'wb'))
print(f'[armA] 概念映射剥离: {len(raw)}->{len(out)} codes, 概念边 {n_before}->{n_after} '
      f'(剥 {100*(1-n_after/max(n_before,1)):.0f}%)')
EOF
echo "=== armA 全链跑 开始 $(date '+%H:%M:%S') ==="
$VENV strategy/bt_execution.py > strategy/logs/bt_execution_conceptA_0911.log 2>&1
RC=$?
echo "=== armA 结束 rc=$RC $(date '+%H:%M:%S') ==="
grep -E "最终净值|Sharpe|最大回撤" strategy/logs/bt_execution_conceptA_0911.log | tail -5
cp "$RVR/backtest_signals.csv" "$RVR/backtest_signals_conceptA.csv" 2>/dev/null && echo "[save] signals -> backtest_signals_conceptA.csv"

# ---------- 2) armB: 概念映射全空 ----------
$VENV - <<'EOF'
import pickle
pickle.dump({}, open('data/stock_concept_map.pkl', 'wb'))
print('[armB] 概念映射 -> 空字典')
EOF
echo "=== armB 全链跑 开始 $(date '+%H:%M:%S') ==="
$VENV strategy/bt_execution.py > strategy/logs/bt_execution_conceptB_0911.log 2>&1
RC=$?
echo "=== armB 结束 rc=$RC $(date '+%H:%M:%S') ==="
grep -E "最终净值|Sharpe|最大回撤" strategy/logs/bt_execution_conceptB_0911.log | tail -5
cp "$RVR/backtest_signals.csv" "$RVR/backtest_signals_conceptB.csv" 2>/dev/null && echo "[save] signals -> backtest_signals_conceptB.csv"
echo "=== 双跑完成 $(date '+%H:%M:%S') (trap恢复map+anchors+signals) ==="
