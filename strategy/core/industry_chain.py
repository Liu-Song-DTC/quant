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
        seg_names = list(chain.keys())  # ["算力基础设施","芯片与通信",...]
        if len(seg_names) < 2:
            continue

        # 计算每个环节的平均涨跌幅
        segment_returns = {}
        segment_concepts = {}
        for seg_name, seg_concepts_list in chain.items():
            seg_rets = [concept_returns.get(c, 0) for c in seg_concepts_list]
            segment_returns[seg_name] = np.mean(seg_rets) if seg_rets else 0
            segment_concepts[seg_name] = seg_concepts_list

        # 正向传导: 上游环节涨了但下游没涨 → 埋伏下游
        for i in range(len(seg_names)):
            for j in range(i + 1, len(seg_names)):
                early_seg = seg_names[i]
                late_seg = seg_names[j]
                spread = segment_returns[early_seg] - segment_returns[late_seg]

                # 上游比下游领先3%以上 → 下游有补涨空间
                if spread > 0.03:
                    bonus = min(spread * 5, 0.3)  # max 0.3 bonus
                    for c in segment_concepts[late_seg]:
                        # 取已有值和新值的最大
                        current = signals.get(c, 0)
                        signals[c] = max(current, bonus)

        # 逆向传导: 下游暴涨但上游没动 → 上游补库
        for i in range(len(seg_names) - 1, -1, -1):
            for j in range(i):
                late_seg = seg_names[i]
                early_seg = seg_names[j]
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
