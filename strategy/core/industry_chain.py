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
        "算力基础设施": ["算力概念", "光通信模块", "数据中心", "云计算", "东数西算", "人工智能"],
        "芯片与通信":  ["国产芯片", "半导体概念", "F5G概念", "PCB", "华为概念"],
        "模型与平台":  ["AI应用", "DeepSeek概念", "国产软件", "信创", "大数据",
                      "机器视觉", "AI智能体", "AI语料", "智谱AI概念"],
        "终端应用":    ["机器人概念", "人形机器人", "智能穿戴", "工业互联",
                      "智能家居", "AI眼镜", "虚拟现实", "物联网"],
    },
    # ═══ 半导体产业链 ═══
    "半导体": {
        "材料与设备":  ["半导体概念", "国产芯片", "LED概念"],
        "设计与制造":  ["国产芯片", "华为概念", "PCB", "3D打印"],
        "通信与物联网": ["F5G概念", "物联网", "光通信模块"],
    },
    # ═══ 新能源车产业链 ═══
    "新能源车": {
        "上游矿材":   ["稀土永磁", "新材料", "氢能源", "锂电池概念", "锂矿概念"],
        "电池技术":   ["钠离子电池", "电池技术", "动力电池回收", "电池"],
        "整车与部件":  ["新能源车", "新能源", "小米汽车", "汽车整车", "汽车零部件",
                      "汽车一体化压铸"],
        "充电与运营":  ["充电桩", "换电概念", "储能概念"],
        "智能化":     ["无人驾驶", "物联网", "智能穿戴"],
    },
    # ═══ 光伏/新能源产业链 ═══
    "光伏储能": {
        "上游材料":   ["新材料", "化工原料"],
        "发电技术":   ["光伏概念"],
        "储能系统":   ["储能概念", "电网概念", "虚拟电厂"],
        "下游应用":   ["绿色电力", "节能环保", "风能", "核能核电", "抽水蓄能",
                      "特高压", "氢能源", "生物质能发电", "可控核聚变"],
    },
    # ═══ 机器人/自动化产业链 ═══
    "机器人": {
        "核心零部件":  ["发电机概念", "工业母机"],
        "本体制造":   ["机器人概念", "人形机器人", "工业母机"],
        "AI与视觉":   ["机器视觉", "AI应用", "DeepSeek概念"],
        "工业应用":   ["工业互联", "工程机械概念", "新型工业化"],
    },
    # ═══ 低空经济产业链 ═══
    "低空经济": {
        "材料与动力":  ["新材料", "航天航空"],
        "飞行器":     ["低空经济", "航天航空"],
        "基础设施":   ["F5G概念", "通用航空"],
    },
    # ═══ 医药/创新药产业链 ═══
    "医药健康": {
        "研发与外包":  ["创新药", "CRO", "基因测序", "合成生物", "生物疫苗",
                      "CAR-T细胞疗法", "AI制药（医疗）"],
        "器械与诊断":  ["医疗器械概念", "医疗器械", "体外诊断概念"],
        "制药":       ["中药概念", "维生素", "独家药品",
                      "幽门螺杆菌概念", "流感", "肝炎概念", "减肥药"],
        "服务与商业":  ["互联医疗", "医美概念", "创新医疗服务", "养老概念",
                      "辅助生殖", "阿兹海默", "精准医疗", "免疫治疗"],
    },
    # ═══ 军工/航天产业链 ═══
    "军工航天": {
        "军工":       ["军工"],
        "航空航天":   ["航天航空"],
        "海工装备":   ["海工装备", "海洋经济"],
        "安防":       ["安防概念"],
    },
    # ═══ 数字经济/信创产业链 ═══
    "数字经济": {
        "底层技术":   ["信创", "国产软件", "大数据", "区块链", "数据确权", "数据要素"],
        "基础设施":   ["数据中心", "云计算", "算力概念", "F5G概念"],
        "金融科技":   ["互联网金融", "数字经济", "跨境支付", "蚂蚁概念"],
        "应用场景":   ["电商概念", "虚拟现实", "Web3.0"],
    },
    # ═══ 消费产业链 ═══
    "消费": {
        "白酒食品":   ["酿酒概念", "预制菜概念", "乳业", "调味品概念", "鸡肉概念",
                      "猪肉概念", "食品安全", "白酒", "味蕾经济"],
        "零售流通":   ["新零售", "社区团购", "内贸流通", "跨境电商", "快递概念",
                      "C2M概念", "拼多多概念"],
        "旅游文娱":   ["旅游概念", "旅游酒店", "文娱消费", "影视概念", "冰雪经济",
                      "体育产业", "宠物经济", "户外露营", "谷子经济"],
        "消费电子":   ["智能穿戴", "智能家居", "无线耳机", "电子烟", "小米汽车"],
    },
    # ═══ 资源/材料产业链 ═══
    "资源材料": {
        "有色金属":   ["黄金概念", "稀土永磁", "小金属概念", "稀缺资源"],
        "化工材料":   ["磷化工", "新材料", "复合集流体", "煤化工概念",
                      "化学纤维", "化工原料", "氟化工概念", "降解塑料",
                      "有机硅概念", "化学制品", "包装材料", "工业气体"],
        "电力能源":   ["抽水蓄能", "绿色电力", "核能核电", "特高压",
                      "油气资源", "氢能源", "虚拟电厂"],
        "农业":       ["农业种植", "粮食概念", "生态农业", "乡村振兴", "农药兽药",
                      "土地流转", "土壤修复"],
    },
    # ═══ 基建/海洋产业链 ═══
    "基建海洋": {
        "水利管网":   ["水利建设", "地下管网", "海绵城市"],
        "海洋经济":   ["海洋经济", "海工装备"],
        "通信基建":   ["光纤概念", "铜缆高速连接", "光通信模块"],
        "建筑工程":   ["工程建设", "新型城镇化", "建筑节能", "铁路基建",
                      "装配建筑", "磁悬浮概念", "工程机械概念"],
    },
}

# 宽基指数/风格/主题概念 — 无产业链上下游关系, 不触发缺失警告
_NO_CHAIN_CONCEPTS: set = {
    # 宽基/风格/主题概念
    "上证50_", "深证100R", "上证180_", "央视50_", "创业成份",
    "大盘价值", "中盘价值", "大盘成长", "中盘成长", "价值股", "红利股",
    "权重股", "周期股", "微利股", "微盘精选", "低价股", "宁组合", "特钢",
    "超级品牌", "茅指数", "证金持股", "中字头", "中特估", "行业龙头", "基金重仓",
    "AH股", "AB股", "IPO受益", "REITs概念",
    # 金融/投资类
    "反内卷概念", "超级电容", "参股券商", "参股银行", "参股保险",
    "参股期货", "参股新三板", "券商概念", "北交所概念", "化债(AMC)概念",
    "并购重组概念", "PPP模式",
    # 地区/政策类
    "湖北自贸", "海南自贸", "上海自贸", "京津冀", "成渝特区", "滨海新区",
    "东北振兴", "雄安新区", "粤港自贸", "中俄贸易概念", "沪企改革",
    "一带一路", "碳交易", "统一大市场",
    # 企业/事件类
    "东方财富热股", "阿里概念", "百度概念", "特斯拉概念", "英伟达概念",
    "股权激励", "网络游戏", "抖音概念(字节概念)",
    "快手概念", "小红书概念", "AIPC", "AIGC概念",
    # 业绩/财务类
    "2025三季报预增", "2025年报扭亏", "2025年报预减", "2026一季报预增",
    "昨日涨停", "昨日涨停_含一字", "昨日炸板", "昨日触板",
    "最近多板", "历史新高", "近期新高",
    # 无产业链的主题概念
    "冷链物流", "婴童概念", "职业教育", "病毒防治", "在线教育",
    "知识产权", "垃圾分类", "环境治理",
    "贬值受益", "网红经济", "地摊经济", "工业大麻", "ST股",
    "超跌股", "房屋检测", "新冠药物", "人脑工程",
    "元宇宙概念", "空气能热泵", "化妆品概念", "新消费",
    "特色药", "单抗概念", "病原体防治", "精准诊断",
    "零售概念", "数字货币", "碳纤维", "鸿蒙概念",
    "超超临界发电", "雅下水电概念", "股权转让",
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
    result = _CHAIN_INDEX.get(concept_name)
    if result is None and concept_name in _NO_CHAIN_CONCEPTS:
        return set()  # 宽基/主题概念无产业链, 返回空集等价于不限行业
    return result
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
