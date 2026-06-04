# core/bom_chain.py
"""
BOM (Bill of Materials) 产业链研究 — 寻找高壁垒、高利润、高市值个股

核心逻辑:
1. 产业链分段 → 识别每个环节的"卡脖子"位置
2. 基本面对齐 → PE/毛利率/ROE/市值 综合评估
3. 壁垒评分 → 行业集中度 + 技术壁垒 + 利润率 = 综合壁垒分
4. 选股输出 → 高壁垒段内的龙头个股排行

与 industry_chain.py 的联动:
- industry_chain: 动量层面的传导 (上游涨→埋伏下游)
- bom_chain: 质量层面的选股 (埋伏时要买下游里壁垒最高的)

三层筛选逻辑:
  Layer 1 - 壁垒识别: 该环节是否被少数公司垄断？ (集中度)
  Layer 2 - 利润验证: 龙头公司是否有持续高利润？ (毛利率/ROE)
  Layer 3 - 市值确认: 大市值=市场已验证的壁垒 (流动性+共识)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


# ═══════════════════════════════════════════════════════════════════
# 产业链 BOM 拓扑: 标注每个环节的壁垒特征
# ═══════════════════════════════════════════════════════════════════

@dataclass
class BOMSegment:
    """产业链环节的BOM特征"""
    name: str
    concepts: List[str]            # 对应的概念板块
    barrier_level: str = 'medium'  # low/medium/high/moat (护城河)
    margin_profile: str = 'medium'  # low/medium/high
    tech_dependency: str = 'none'  # none/import/self_sufficient/export
    typical_pe_range: Tuple[float, float] = (15, 40)  # 合理PE区间


# BOM增强版产业链定义 (在原有 industry_chain 基础上叠加基本面维度)
BOM_CHAINS: Dict[str, Dict[str, BOMSegment]] = {
    # ═══ 半导体产业链 — 最典型的卡脖子链条 ═══
    "半导体": {
        "设备与材料": BOMSegment(
            name="设备与材料",
            concepts=["光刻机(胶)", "EDA概念", "第三代半导体", "第四代半导体"],
            barrier_level="moat",         # ASML光刻机+EDA全球垄断
            margin_profile="high",        # 毛利率50-85%+
            tech_dependency="import",
            typical_pe_range=(40, 150),
        ),
        "芯片设计": BOMSegment(
            name="芯片设计",
            concepts=["国产芯片", "汽车芯片", "IGBT概念"],
            barrier_level="high",
            margin_profile="high",        # 毛利率40-60%
            tech_dependency="self_sufficient",
            typical_pe_range=(30, 80),
        ),
        "封装测试": BOMSegment(
            name="封装测试",
            concepts=["先进封装"],
            barrier_level="medium",
            margin_profile="medium",      # 毛利率15-25%
            tech_dependency="self_sufficient",
            typical_pe_range=(15, 35),
        ),
        "终端应用": BOMSegment(
            name="终端应用",
            concepts=["消费电子概念", "智能穿戴"],
            barrier_level="low",
            margin_profile="low",
            tech_dependency="self_sufficient",
            typical_pe_range=(10, 30),
        ),
    },

    # ═══ AI/算力产业链 ═══
    "AI人工智能": {
        "算力芯片与互联": BOMSegment(
            name="算力芯片与互联",
            concepts=["算力概念", "存储芯片", "光通信模块"],
            barrier_level="moat",         # NVIDIA GPU + 光模块800G
            margin_profile="high",
            tech_dependency="import",
            typical_pe_range=(40, 120),
        ),
        "算力基础设施": BOMSegment(
            name="算力基础设施",
            concepts=["液冷概念", "数据中心", "云计算", "东数西算"],
            barrier_level="high",
            margin_profile="medium",
            tech_dependency="self_sufficient",
            typical_pe_range=(25, 60),
        ),
        "模型与平台": BOMSegment(
            name="模型与平台",
            concepts=["AI应用", "DeepSeek概念", "国产软件", "信创", "大数据"],
            barrier_level="medium",
            margin_profile="high",        # 软件毛利率高
            tech_dependency="self_sufficient",
            typical_pe_range=(30, 80),
        ),
        "AI终端": BOMSegment(
            name="AI终端",
            concepts=["机器人概念", "人形机器人", "智能穿戴", "工业互联"],
            barrier_level="medium",
            margin_profile="medium",
            tech_dependency="self_sufficient",
            typical_pe_range=(25, 70),
        ),
    },

    # ═══ 新能源车产业链 ═══
    "新能源车": {
        "上游矿材": BOMSegment(
            name="上游矿材",
            concepts=["锂矿概念", "稀土永磁", "新材料"],
            barrier_level="high",         # 资源垄断 + 开采权壁垒
            margin_profile="high",
            tech_dependency="self_sufficient",
            typical_pe_range=(10, 30),
        ),
        "动力电池": BOMSegment(
            name="动力电池",
            concepts=["锂电池概念", "固态电池", "钠离子电池", "电池技术", "动力电池回收"],
            barrier_level="moat",         # 宁德时代/比亚迪寡头格局
            margin_profile="medium",      # 规模效应后利润率趋稳
            tech_dependency="self_sufficient",
            typical_pe_range=(15, 40),
        ),
        "整车制造": BOMSegment(
            name="整车制造",
            concepts=["新能源车", "特斯拉概念", "新能源"],
            barrier_level="medium",
            margin_profile="low",         # 价格战压缩利润
            tech_dependency="self_sufficient",
            typical_pe_range=(10, 35),
        ),
        "充电与运营": BOMSegment(
            name="充电与运营",
            concepts=["充电桩", "换电概念", "储能概念"],
            barrier_level="low",
            margin_profile="medium",
            tech_dependency="self_sufficient",
            typical_pe_range=(15, 40),
        ),
        "智能化": BOMSegment(
            name="智能化",
            concepts=["无人驾驶", "车联网(车路云)", "汽车芯片"],
            barrier_level="high",         # 技术密集
            margin_profile="high",
            tech_dependency="import",
            typical_pe_range=(35, 80),
        ),
    },

    # ═══ 医药健康产业链 ═══
    "医药健康": {
        "创新药研发": BOMSegment(
            name="创新药研发",
            concepts=["创新药", "CRO", "基因测序", "合成生物"],
            barrier_level="moat",         # 专利保护 + 研发周期10年+
            margin_profile="high",        # 毛利率80%+
            tech_dependency="self_sufficient",
            typical_pe_range=(30, 100),
        ),
        "医疗器械": BOMSegment(
            name="医疗器械",
            concepts=["医疗器械概念", "体外诊断概念"],
            barrier_level="high",
            margin_profile="high",        # 毛利率60%+
            tech_dependency="import",
            typical_pe_range=(25, 60),
        ),
        "制药": BOMSegment(
            name="制药",
            concepts=["中药概念", "病原体防治", "生物疫苗"],
            barrier_level="high",         # 品牌 + 配方壁垒(中药)
            margin_profile="high",
            tech_dependency="self_sufficient",
            typical_pe_range=(20, 50),
        ),
        "医疗服务": BOMSegment(
            name="医疗服务",
            concepts=["互联医疗", "养老概念", "医美概念"],
            barrier_level="low",
            margin_profile="medium",
            tech_dependency="self_sufficient",
            typical_pe_range=(20, 50),
        ),
    },

    # ═══ 光伏储能产业链 ═══
    "光伏储能": {
        "上游硅料": BOMSegment(
            name="上游硅料",
            concepts=["新材料", "碳基材料", "HJT电池", "钙钛矿电池"],
            barrier_level="high",         # 技术迭代快，新进入难
            margin_profile="high",        # 硅料涨价时暴利
            tech_dependency="self_sufficient",
            typical_pe_range=(8, 25),
        ),
        "光伏组件": BOMSegment(
            name="光伏组件",
            concepts=["光伏概念"],
            barrier_level="low",          # 产能过剩、同质化严重
            margin_profile="low",
            tech_dependency="self_sufficient",
            typical_pe_range=(8, 20),
        ),
        "储能系统": BOMSegment(
            name="储能系统",
            concepts=["储能概念", "固态电池", "电网概念", "智能电网"],
            barrier_level="high",
            margin_profile="medium",
            tech_dependency="self_sufficient",
            typical_pe_range=(15, 40),
        ),
        "下游电站": BOMSegment(
            name="下游电站",
            concepts=["绿色电力", "节能环保", "风能", "核能核电"],
            barrier_level="medium",
            margin_profile="medium",
            tech_dependency="self_sufficient",
            typical_pe_range=(10, 30),
        ),
    },

    # ═══ 机器人产业链 ═══
    "机器人": {
        "核心零部件": BOMSegment(
            name="核心零部件",
            concepts=["减速器", "传感器", "轮毂电机", "发电机概念"],
            barrier_level="moat",         # 谐波减速器日本垄断(哈默纳科)
            margin_profile="high",
            tech_dependency="import",
            typical_pe_range=(40, 100),
        ),
        "本体制造": BOMSegment(
            name="本体制造",
            concepts=["机器人概念", "人形机器人", "工业母机"],
            barrier_level="medium",
            margin_profile="medium",
            tech_dependency="self_sufficient",
            typical_pe_range=(25, 60),
        ),
        "AI与视觉": BOMSegment(
            name="AI与视觉",
            concepts=["机器视觉", "AI应用", "DeepSeek概念"],
            barrier_level="high",
            margin_profile="high",
            tech_dependency="import",
            typical_pe_range=(30, 80),
        ),
        "集成应用": BOMSegment(
            name="集成应用",
            concepts=["工业互联", "专精特新"],
            barrier_level="low",
            margin_profile="low",
            tech_dependency="self_sufficient",
            typical_pe_range=(15, 35),
        ),
    },

    # ═══ 低空经济产业链 ═══
    "低空经济": {
        "材料与动力": BOMSegment(
            name="材料与动力",
            concepts=["碳纤维", "新材料", "航天航空"],
            barrier_level="high",
            margin_profile="high",
            tech_dependency="import",
            typical_pe_range=(30, 80),
        ),
        "飞行器制造": BOMSegment(
            name="飞行器制造",
            concepts=["低空经济", "无人机", "飞行汽车(eVTOL)"],
            barrier_level="high",
            margin_profile="medium",
            tech_dependency="self_sufficient",
            typical_pe_range=(25, 70),
        ),
        "基础设施": BOMSegment(
            name="基础设施",
            concepts=["通用航空", "智慧城市", "5G概念"],
            barrier_level="medium",
            margin_profile="medium",
            tech_dependency="self_sufficient",
            typical_pe_range=(20, 50),
        ),
    },

    # ═══ 数字经济/信创产业链 ═══
    "数字经济": {
        "底层基础软件": BOMSegment(
            name="底层基础软件",
            concepts=["信创", "国产软件", "大数据"],
            barrier_level="moat",         # 操作系统/数据库垄断
            margin_profile="high",
            tech_dependency="import",
            typical_pe_range=(40, 120),
        ),
        "算力基础设施": BOMSegment(
            name="算力基础设施",
            concepts=["数据中心", "云计算", "算力概念", "5G概念"],
            barrier_level="high",
            margin_profile="medium",
            tech_dependency="self_sufficient",
            typical_pe_range=(20, 50),
        ),
        "金融科技": BOMSegment(
            name="金融科技",
            concepts=["互联网金融", "数字经济", "跨境支付"],
            barrier_level="medium",
            margin_profile="medium",
            tech_dependency="self_sufficient",
            typical_pe_range=(15, 40),
        ),
        "应用场景": BOMSegment(
            name="应用场景",
            concepts=["电商概念", "智慧城市", "虚拟现实"],
            barrier_level="low",
            margin_profile="low",
            tech_dependency="self_sufficient",
            typical_pe_range=(10, 30),
        ),
    },

    # ═══ 军工航天产业链 ═══
    "军工航天": {
        "军工电子": BOMSegment(
            name="军工电子",
            concepts=["军民融合", "军工"],
            barrier_level="moat",         # 军工资质壁垒极高
            margin_profile="high",
            tech_dependency="self_sufficient",
            typical_pe_range=(30, 80),
        ),
        "航空航天": BOMSegment(
            name="航空航天",
            concepts=["航天航空", "大飞机", "商业航天"],
            barrier_level="moat",
            margin_profile="medium",
            tech_dependency="import",
            typical_pe_range=(40, 100),
        ),
        "船舶与装备": BOMSegment(
            name="船舶与装备",
            concepts=["船舶制造"],
            barrier_level="high",
            margin_profile="low",
            tech_dependency="self_sufficient",
            typical_pe_range=(15, 40),
        ),
        "无人机与安防": BOMSegment(
            name="无人机与安防",
            concepts=["无人机", "安防概念"],
            barrier_level="high",
            margin_profile="medium",
            tech_dependency="self_sufficient",
            typical_pe_range=(20, 50),
        ),
    },
}


# ═══════════════════════════════════════════════════════════════════
# BOM 评分引擎
# ═══════════════════════════════════════════════════════════════════

# 壁垒等级 → 分数映射
BARRIER_SCORE = {
    'moat': 1.0,       # 护城河: 全球垄断或极高资质
    'high': 0.75,      # 高壁垒: 技术/资源/品牌垄断
    'medium': 0.50,    # 中等壁垒: 有一定门槛但竞争存在
    'low': 0.25,       # 低壁垒: 同质化竞争
}

MARGIN_SCORE = {
    'high': 1.0,
    'medium': 0.6,
    'low': 0.3,
}

TECH_SCORE = {
    'import': 0.8,          # 依赖进口 → 国产替代空间大(壁垒在供给端)
    'self_sufficient': 0.5, # 自给自足
    'export': 1.0,          # 出口竞争力 → 全球领先
    'none': 0.3,
}


def compute_bom_segment_score(seg: BOMSegment) -> float:
    """计算单个产业链环节的BOM综合质量分 [0, 1]"""
    barrier = BARRIER_SCORE.get(seg.barrier_level, 0.5)
    margin = MARGIN_SCORE.get(seg.margin_profile, 0.6)
    tech = TECH_SCORE.get(seg.tech_dependency, 0.5)
    # 壁垒40% + 利润率35% + 技术依赖25%
    return float(np.clip(barrier * 0.40 + margin * 0.35 + tech * 0.25, 0.0, 1.0))


def get_bom_segment_for_concept(concept_name: str) -> Optional[BOMSegment]:
    """根据概念名称查找其所属的BOM环节"""
    for chain_name, chain in BOM_CHAINS.items():
        for seg_name, seg in chain.items():
            if concept_name in seg.concepts:
                return seg
    return None


def compute_stock_bom_score(
    code: str,
    stock_concepts: Dict[str, List[str]],  # {code: [concept_names]}
    fundamentals: Optional[Dict[str, dict]] = None,  # {code: {pe, roe, gross_margin, market_cap}}
) -> Dict[str, float]:
    """计算单只股票的BOM质量评分

    Args:
        code: 股票代码
        stock_concepts: 股票→概念映射
        fundamentals: 基本面数据 (PE/ROE/毛利率/市值)

    Returns:
        {
            'bom_barrier_score': 壁垒分 [0, 1],
            'bom_profit_score': 利润分 [0, 1],
            'bom_quality_score': 综合质量分 [0, 1],
            'bom_moat_segments': 高壁垒段数量,
        }
    """
    concepts = stock_concepts.get(code, [])
    if not concepts:
        return {
            'bom_barrier_score': 0.3,
            'bom_profit_score': 0.3,
            'bom_quality_score': 0.3,
            'bom_moat_segments': 0,
        }

    # 找该股票涉及的所有BOM环节
    segment_scores = []
    moat_count = 0
    for c in concepts:
        seg = get_bom_segment_for_concept(c)
        if seg:
            seg_score = compute_bom_segment_score(seg)
            segment_scores.append(seg_score)
            if seg.barrier_level in ('moat', 'high'):
                moat_count += 1

    if not segment_scores:
        bom_barrier = 0.3
    else:
        # 取最高分 — 股票只要有一个卡脖子位置就值得高看
        bom_barrier = float(np.clip(max(segment_scores), 0.0, 1.0))

    # 利润分: 从基本面数据计算
    bom_profit = 0.5
    if fundamentals and code in fundamentals:
        fund = fundamentals[code]
        roe = fund.get('roe', 0.10)
        gm = fund.get('gross_margin', 0.30)
        # ROE>15% = 优质, 毛利率>40% = 高利润
        roe_score = np.clip(roe / 0.20, 0.0, 1.0)
        gm_score = np.clip(gm / 0.50, 0.0, 1.0)
        bom_profit = float(np.clip(roe_score * 0.5 + gm_score * 0.5, 0.0, 1.0))

    # 综合: 壁垒55% + 利润45%
    bom_quality = float(np.clip(bom_barrier * 0.55 + bom_profit * 0.45, 0.0, 1.0))

    return {
        'bom_barrier_score': bom_barrier,
        'bom_profit_score': bom_profit,
        'bom_quality_score': bom_quality,
        'bom_moat_segments': moat_count,
    }


def get_high_barrier_stocks(
    stock_concepts: Dict[str, List[str]],
    fundamentals: Optional[Dict[str, dict]] = None,
    min_quality: float = 0.55,
    min_market_cap: float = 50.0,  # 亿
) -> List[Tuple[str, float, Dict]]:
    """筛选高壁垒高利润个股

    Returns:
        [(code, bom_quality_score, detail_dict), ...] 按质量分降序
    """
    results = []
    for code in stock_concepts:
        scores = compute_stock_bom_score(code, stock_concepts, fundamentals)
        if scores['bom_quality_score'] < min_quality:
            continue
        if fundamentals and code in fundamentals:
            mc = fundamentals[code].get('market_cap', 0)
            if mc < min_market_cap:
                continue
        results.append((code, scores['bom_quality_score'], scores))

    results.sort(key=lambda x: -x[1])
    return results


def get_chain_choke_points(chain_name: str) -> List[Tuple[str, float]]:
    """获取某产业链的卡脖子环节列表

    Returns:
        [(segment_name, bom_score), ...] 按壁垒分降序
    """
    chain = BOM_CHAINS.get(chain_name, {})
    if not chain:
        return []
    result = []
    for seg_name, seg in chain.items():
        score = compute_bom_segment_score(seg)
        if seg.barrier_level in ('moat', 'high'):
            result.append((seg_name, score))
    result.sort(key=lambda x: -x[1])
    return result


def summarize_bom_landscape() -> str:
    """生成BOM产业链全景摘要"""
    lines = ["=== BOM产业链全景: 高壁垒环节 ==="]
    for chain_name, chain in BOM_CHAINS.items():
        choke_points = get_chain_choke_points(chain_name)
        if not choke_points:
            continue
        cp_str = ", ".join(f"{name}({score:.2f})" for name, score in choke_points[:3])
        lines.append(f"  {chain_name}: {cp_str}")
    return "\n".join(lines)
