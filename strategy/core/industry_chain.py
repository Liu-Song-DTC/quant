# core/industry_chain.py
"""
产业链拓扑图 — 跟踪链条上下游传导

核心逻辑:
1. 每条产业链定义 上游→中游→下游
2. 每日计算各环节(概念板块)的近期涨跌幅
3. 检测已涨环节, 推算还没涨的下游 → 提前埋伏加分
"""

from typing import Dict, List, Tuple, Optional
import numpy as np

# 产业链拓扑定义 {名称: {"upstream": [...], "midstream": [...], "downstream": [...]}}
INDUSTRY_CHAINS: Dict[str, Dict[str, List[str]]] = {
    # ═══ AI/算力产业链 ═══
    "AI人工智能": {
        "算力基础设施": ["算力概念", "液冷概念", "光通信模块", "数据中心", "云计算", "东数西算"],
        "芯片与通信":  ["存储芯片", "通信技术", "5G概念", "国产芯片", "半导体概念"],
        "模型与平台":  ["AI应用", "DeepSeek概念", "国产软件", "信创", "大数据",
                      "机器视觉", "多模态AI"],
        "终端应用":    ["机器人概念", "人形机器人", "智能穿戴", "工业互联",
                      "智慧城市", "智能家居", "车联网(车路云)"],
    },
    # ═══ 半导体产业链 ═══
    "半导体": {
        "材料与设备":  ["光刻机(胶)", "EDA概念", "第三代半导体", "第四代半导体"],
        "设计与制造":  ["国产芯片", "存储芯片", "汽车芯片", "IGBT概念", "先进封装"],
        "消费电子":    ["消费电子概念", "智能穿戴", "小米概念", "虚拟现实"],
        "通信与物联网": ["5G概念", "物联网", "车联网(车路云)"],
    },
    # ═══ 新能源车产业链 ═══
    "新能源车": {
        "上游矿材":   ["锂矿概念", "稀土永磁", "新材料", "氢能源"],
        "电池技术":   ["锂电池概念", "固态电池", "钠离子电池", "电池技术",
                     "动力电池回收", "刀片电池"],
        "整车与部件":  ["新能源车", "特斯拉概念", "新能源"],
        "充电与运营":  ["充电桩", "换电概念", "储能概念"],
        "智能化":     ["无人驾驶", "车联网(车路云)", "汽车芯片"],
    },
    # ═══ 光伏/新能源产业链 ═══
    "光伏储能": {
        "上游材料":   ["新材料", "碳基材料"],
        "电池技术":   ["HJT电池", "钙钛矿电池", "光伏概念"],
        "储能系统":   ["储能概念", "固态电池", "电网概念", "智能电网"],
        "下游应用":   ["绿色电力", "节能环保", "风能", "核能核电"],
    },
    # ═══ 机器人/自动化产业链 ═══
    "机器人": {
        "核心零部件":  ["减速器", "传感器", "轮毂电机", "发电机概念"],
        "本体制造":   ["机器人概念", "人形机器人", "工业母机"],
        "AI与视觉":   ["机器视觉", "AI应用", "DeepSeek概念"],
        "工业应用":   ["工业互联", "专精特新"],
    },
    # ═══ 低空经济产业链 ═══
    "低空经济": {
        "材料与动力":  ["碳纤维", "新材料", "航天航空"],
        "飞行器":     ["低空经济", "无人机", "飞行汽车(eVTOL)"],
        "基础设施":   ["通用航空", "智慧城市", "5G概念"],
    },
    # ═══ 医药/创新药产业链 ═══
    "医药健康": {
        "研发与外包":  ["创新药", "CRO", "基因测序", "合成生物", "生物疫苗"],
        "器械与诊断":  ["医疗器械概念", "体外诊断概念"],
        "制药":       ["中药概念", "病原体防治"],
        "服务与商业":  ["互联医疗", "养老概念", "医美概念"],
    },
    # ═══ 军工/航天产业链 ═══
    "军工航天": {
        "军工电子":   ["军民融合", "军工"],
        "航空航天":   ["航天航空", "大飞机", "商业航天"],
        "船舶与装备":  ["船舶制造"],
        "安防与无人机": ["无人机", "安防概念"],
    },
    # ═══ 数字经济/信创产业链 ═══
    "数字经济": {
        "底层技术":   ["信创", "国产软件", "大数据", "区块链"],
        "基础设施":   ["数据中心", "云计算", "算力概念", "5G概念"],
        "金融科技":   ["互联网金融", "数字经济", "跨境支付"],
        "应用场景":   ["电商概念", "智慧城市", "虚拟现实"],
    },
}

# 所有已映射的概念名集合 (自动从映射文件和定义中收集)
_ALL_CHAIN_CONCEPTS: set = None


def get_all_chain_concepts() -> set:
    global _ALL_CHAIN_CONCEPTS
    if _ALL_CHAIN_CONCEPTS is not None:
        return _ALL_CHAIN_CONCEPTS
    _ALL_CHAIN_CONCEPTS = set()
    for chain in INDUSTRY_CHAINS.values():
        for segment in chain.values():
            _ALL_CHAIN_CONCEPTS.update(segment)
    return _ALL_CHAIN_CONCEPTS


def compute_chain_signals(
    concept_returns: Dict[str, float],
    top_n_per_chain: int = 3,
    auto_discovered_edges: Optional[Dict[str, List[Tuple[str, float]]]] = None,
) -> Dict[str, float]:
    """计算产业链传导信号（自适应阈值 + 分行业vol版）

    每产业链独立计算波动率基准，避免高波动行业(如AI)掩蔽低波动行业(如医药)的传导。
    支持自动发现的传导边作为补充信号。
    """
    signals: Dict[str, float] = {}

    # 全局波动率基准（兜底）
    ret_vals = np.array(list(concept_returns.values()))
    global_vol = float(np.std(ret_vals)) if len(ret_vals) > 3 else 0.02
    global_vol = max(global_vol, 0.008)

    for chain_name, chain in INDUSTRY_CHAINS.items():
        seg_names = list(chain.keys())
        if len(seg_names) < 2:
            continue

        segment_returns = {}
        segment_concepts = {}
        chain_all_rets = []
        for seg_name, seg_concepts_list in chain.items():
            seg_rets = [concept_returns.get(c, 0) for c in seg_concepts_list]
            if seg_rets:
                segment_returns[seg_name] = np.mean(seg_rets)
                chain_all_rets.extend(seg_rets)
            else:
                segment_returns[seg_name] = 0
            segment_concepts[seg_name] = seg_concepts_list

        # 分行业vol: 本产业链内概念的标准差，兜底到全局vol
        chain_vol = float(np.std(chain_all_rets)) if len(chain_all_rets) > 2 else global_vol
        chain_vol = max(chain_vol, 0.008)

        forward_threshold = chain_vol * 1.5
        reverse_threshold = chain_vol * 2.5
        # 跨级前向传导阈值（跨多环节需要更大价差）
        forward_skip_threshold = chain_vol * 2.2

        for i in range(len(seg_names)):
            for j in range(i + 1, len(seg_names)):
                early_seg = seg_names[i]
                late_seg = seg_names[j]
                spread = segment_returns[early_seg] - segment_returns[late_seg]
                skip_levels = j - i  # 跨环节数: 1=相邻, 2+=跨级

                if skip_levels == 1 and spread > forward_threshold:
                    bonus = min(spread / chain_vol * 0.10, 0.30)
                    for c in segment_concepts[late_seg]:
                        current = signals.get(c, 0)
                        signals[c] = max(current, bonus)
                elif skip_levels >= 2 and spread > forward_skip_threshold:
                    # 跨级传导: 更大价差要求, 更小加成(间接传导致信度低)
                    bonus = min(spread / chain_vol * 0.06, 0.18)
                    for c in segment_concepts[late_seg]:
                        current = signals.get(c, 0)
                        signals[c] = max(current, bonus)

        for i in range(len(seg_names) - 1, -1, -1):
            for j in range(i):
                late_seg = seg_names[i]
                early_seg = seg_names[j]
                spread = segment_returns[late_seg] - segment_returns[early_seg]
                if spread > reverse_threshold:
                    bonus = min(spread / chain_vol * 0.06, 0.25)
                    for c in segment_concepts[early_seg]:
                        current = signals.get(c, 0)
                        signals[c] = max(current, bonus)

    # 自动发现边: 高相关性 + 前导滞后关系 → 补充传导信号
    if auto_discovered_edges:
        for edge_name, edges in auto_discovered_edges.items():
            for target_concept, bonus in edges:
                current = signals.get(target_concept, 0)
                signals[target_concept] = max(current, bonus)

    return signals


def discover_chain_edges(
    concept_hist: Dict[str, np.ndarray],  # {concept_name: return_series}
    concept_names: List[str],
    min_correlation: float = 0.6,
    max_lag: int = 5,
    top_n: int = 20,
) -> Dict[str, List[Tuple[str, float]]]:
    """从历史数据自动发现产业链传导关系

    方法: 对每对概念计算滞后相关性，找出A领先B的关系。
    - 高正相关 (ρ > 0.6) 且 A(t) 与 B(t+k) 相关度最高 → A 领先 B

    Args:
        concept_hist: {concept_name: daily_return_array}
        concept_names: 要分析的概念列表
        min_correlation: 最低相关系数阈值
        max_lag: 最大滞后天数
        top_n: 返回前N个最强的领先-滞后关系

    Returns:
        {leader_name: [(follower_name, edge_strength), ...]}
    """
    edges: Dict[str, List[Tuple[str, float]]] = {}
    scored_edges: List[Tuple[str, str, float]] = []

    for i, lead_name in enumerate(concept_names):
        lead_rets = concept_hist.get(lead_name)
        if lead_rets is None or len(lead_rets) < 20:
            continue

        for j, follow_name in enumerate(concept_names):
            if i == j:
                continue
            follow_rets = concept_hist.get(follow_name)
            if follow_rets is None or len(follow_rets) < 20:
                continue

            # 确保长度一致
            min_len = min(len(lead_rets), len(follow_rets))
            if min_len < 20:
                continue
            lead = lead_rets[-min_len:]
            follow = follow_rets[-min_len:]

            # 同期相关性
            corr_0 = float(np.corrcoef(lead, follow)[0, 1])
            if np.isnan(corr_0) or corr_0 < min_correlation:
                continue

            # 滞后相关性: lead(t) vs follow(t+k) for k=1..max_lag
            best_lag = 0
            best_corr = corr_0
            for k in range(1, max_lag + 1):
                if min_len <= k:
                    break
                corr_k = float(np.corrcoef(lead[:-k], follow[k:])[0, 1])
                if np.isnan(corr_k):
                    continue
                if corr_k > best_corr:
                    best_corr = corr_k
                    best_lag = k

            # 存在显著领先关系: 最佳滞后k>0且相关性提升>0.05
            if best_lag > 0 and best_corr > corr_0 + 0.05 and best_corr > min_correlation:
                edge_strength = float(np.clip((best_corr - min_correlation) * 2.5, 0.05, 0.30))
                scored_edges.append((lead_name, follow_name, edge_strength))

    # 取top_n最强边
    scored_edges.sort(key=lambda x: -x[2])
    for lead, follow, strength in scored_edges[:top_n]:
        if lead not in edges:
            edges[lead] = []
        edges[lead].append((follow, strength))

    return edges


# ── 反向索引: concept → chain_name → set of all concepts in that chain ──
_CHAIN_INDEX: Dict[str, set] = None


def _build_chain_index() -> Dict[str, set]:
    """构建反向索引: concept_name → {all concepts in same chain}"""
    idx = {}
    for chain_name, tiers in INDUSTRY_CHAINS.items():
        for tier_name, concepts in tiers.items():
            for c in concepts:
                if c not in idx:
                    idx[c] = set()
                # 收集该链条所有概念
                for tier2, concepts2 in tiers.items():
                    idx[c].update(concepts2)
    return idx


def get_chain_concepts(concept_name: str) -> Optional[set]:
    """返回 concept_name 所属产业链的所有概念板块名称，找不到返回 None"""
    global _CHAIN_INDEX
    if _CHAIN_INDEX is None:
        _CHAIN_INDEX = _build_chain_index()
    return _CHAIN_INDEX.get(concept_name)
def get_chain_lead_score(
    code: str,
    stock_concepts: Dict[str, List[str]],
    chain_signals: Dict[str, float],
) -> float:
    """计算单只股票的产业链埋伏分数

    max替代mean避免"已涨+未涨"被平均掉; tanh(×1.5)避免早期饱和
    """
    concepts = stock_concepts.get(code, [])
    if not concepts or not chain_signals:
        return 0.0

    scores = [chain_signals.get(c, 0) for c in concepts]
    if not scores:
        return 0.0

    return float(np.tanh(max(scores) * 1.5))
