# core/signal.py
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class Signal:
    """
    增强的策略信号 - 包含收益和风险信息
    """
    buy: bool = False
    sell: bool = False
    score: float = 0.0           # 交易分数（用于排序）

    # === 因子相关 ===
    factor_value: float = 0.0    # 原始因子值（用于IC计算）
    factor_name: str = ""         # 因子名称

    # === 行业信息 ===
    industry: str = ""            # 行业分类

    # === 风险信息 ===
    risk_vol: float = 0.0        # 波动率风险
    risk_regime: int = 0          # 市场状态风险: -1(熊), 0(震荡), 1(牛)
    risk_confidence: float = 0.0  # 市场判断置信度: 0-1
    risk_extreme: bool = False    # 是否处于极端状态

    # === 风险调整分数 ===
    adjusted_score: float = 0.0   # 风险调整后的分数

    # === 因子质量（用于动态阈值）===
    factor_quality: float = 0.0   # 动态因子质量分数

    # === 缠论信号 ===
    chan_divergence_type: str = ""           # 'none'/'bottom'/'top'/'hidden_bottom'/'hidden_top'
    chan_divergence_strength: float = 0.0    # 背离强度 [0, 1]
    chan_structure_score: float = 0.0        # 结构对齐分数 [-1, 1]

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            'buy': self.buy,
            'sell': self.sell,
            'score': self.score,
            'factor_value': self.factor_value,
            'factor_name': self.factor_name,
            'industry': self.industry,
            'risk_vol': self.risk_vol,
            'risk_regime': self.risk_regime,
            'risk_confidence': self.risk_confidence,
            'risk_extreme': self.risk_extreme,
            'adjusted_score': self.adjusted_score,
            'factor_quality': self.factor_quality,
            'chan_divergence_type': self.chan_divergence_type,
            'chan_divergence_strength': self.chan_divergence_strength,
            'chan_structure_score': self.chan_structure_score,
        }

    def get_risk_level(self) -> str:
        """获取风险等级描述"""
        if self.risk_regime == -1 and self.risk_confidence > 0.5:
            return "HIGH_RISK"  # 熊市高风险
        elif self.risk_extreme:
            return "EXTREME"    # 极端状态
        elif self.risk_regime == 1:
            return "LOW_RISK"    # 牛市低风险
        else:
            return "NORMAL"      # 正常
