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
    "人工智能": {
        "upstream": ["算力概念", "光模块", "液冷散热",
                     "存储芯片", "数据中心", "通信技术",
                     "国产芯片", "半导体概念", "5G概念"],
        "midstream": ["大模型", "AI应用", "机器视觉", "多模态AI"],
        "downstream": ["智能驾驶", "机器人概念", "智能穿戴",
                       "工业互联", "智能家居"],
    },
    "半导体": {
        "upstream": ["光刻机", "半导体设备", "半导体材料", "EDA概念", "第三代半导体"],
        "midstream": ["国产芯片", "存储芯片", "汽车芯片", "先进封装", "IGBT概念"],
        "downstream": ["消费电子", "智能穿戴", "5G概念", "物联网"],
    },
    "新能源": {
        "upstream": ["锂矿", "钴矿", "稀土永磁", "正极材料",
                     "负极材料", "电解液", "电池材料"],
        "midstream": ["锂电池概念", "固态电池", "钠电池", "储能概念",
                     "电池回收", "电池技术", "刀片电池"],
        "downstream": ["新能源车", "充电桩", "换电概念", "汽车电子",
                       "汽车零部件", "无人驾驶"],
    },
    "光伏": {
        "upstream": ["光伏材料", "硅料", "银浆"],
        "midstream": ["光伏电池", "光伏组件", "光伏逆变器", "HIT电池", "钙钛矿电池"],
        "downstream": ["光伏电站", "BIPV概念", "储能概念"],
    },
    "机器人": {
        "upstream": ["减速器", "伺服系统", "电机", "控制器", "传感器"],
        "midstream": ["机器人概念", "工业母机", "机器视觉"],
        "downstream": ["工业互联", "智能物流", "智能制造"],
    },
    "低空经济": {
        "upstream": ["航天航空", "航空发动机", "碳纤维", "通航材料"],
        "midstream": ["低空经济", "无人机概念", "飞行汽车"],
        "downstream": ["通用航空", "物流概念", "智慧城市"],
    },
    "医药": {
        "upstream": ["创新药", "CRO概念", "基因测序", "生物疫苗", "合成生物"],
        "midstream": ["化学制药", "中药概念", "医疗器械概念", "体外诊断"],
        "downstream": ["医药商业", "互联医疗", "养老概念", "医美概念"],
    },
    "军工": {
        "upstream": ["军民融合", "军工电子", "航天航空", "船舶制造"],
        "midstream": ["大飞机", "北斗导航", "天基互联", "商业航天"],
        "downstream": ["无人机概念", "通用航空", "安防概念"],
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
) -> Dict[str, float]:
    """计算产业链传导信号

    Args:
        concept_returns: {概念名: 近期涨跌幅(小数)}
        top_n_per_chain: 每条链关注的"埋伏"概念数

    Returns:
        {概念名: 埋伏分数(0~1)}
    """
    signals: Dict[str, float] = {}

    for chain_name, chain in INDUSTRY_CHAINS.items():
        # 计算每个环节的平均涨跌幅
        segment_returns = {}
        segment_concepts = {}
        for seg_name, seg_concepts in chain.items():
            seg_rets = [concept_returns.get(c, 0) for c in seg_concepts]
            segment_returns[seg_name] = np.mean(seg_rets) if seg_rets else 0
            segment_concepts[seg_name] = seg_concepts

        # 按涨跌幅排序环节: 上游→中游→下游 → 上游→下游
        order = ["upstream", "midstream", "downstream"]

        # 如果上游涨了但下游没涨 → 埋伏下游
        for i in range(len(order)):
            for j in range(i + 1, len(order)):
                early_seg = order[i]
                late_seg = order[j]
                spread = segment_returns[early_seg] - segment_returns[later_seg]

                # 上游比下游领先3%以上 → 下游有补涨空间
                if spread > 0.03:
                    bonus = min(spread * 5, 0.3)  # max 0.3 bonus
                    for c in segment_concepts[late_seg]:
                        # 取已有值和新值的最大
                        current = signals.get(c, 0)
                        signals[c] = max(current, bonus)

        # 另一方向: 如下游暴涨但上游没动 → 上游有补库需求
        for i in range(len(order) - 1, -1, -1):
            for j in range(i):
                late_seg = order[i]
                early_seg = order[j]
                spread = segment_returns[late_seg] - segment_returns[early_seg]
                if spread > 0.05:
                    bonus = min(spread * 3, 0.25)
                    for c in segment_concepts[early_seg]:
                        current = signals.get(c, 0)
                        signals[c] = max(current, bonus)

    return signals


def get_chain_lead_score(
    code: str,
    stock_concepts: Dict[str, List[str]],
    chain_signals: Dict[str, float],
) -> float:
    """计算单只股票的产业链埋伏分数

    Args:
        code: 股票代码
        stock_concepts: {code: [概念列表]}
        chain_signals: 从 compute_chain_signals 返回的信号

    Returns:
        0~1 之间的埋伏分数
    """
    concepts = stock_concepts.get(code, [])
    if not concepts or not chain_signals:
        return 0.0

    scores = [chain_signals.get(c, 0) for c in concepts]
    if not scores:
        return 0.0

    return float(np.tanh(np.mean(scores) * 3.0))
