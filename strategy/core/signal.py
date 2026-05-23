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

    # === 信号信心度 ===
    signal_confidence: float = 0.0    # 四重确认综合信心度 [0, 1]

    # === 缠论信号 ===
    chan_divergence_type: str = ""           # 'none'/'bottom'/'top'/'hidden_bottom'/'hidden_top'/'buy1'/'buy2'/'buy3'/'sell1'/'sell2'/'sell3'
    chan_divergence_strength: float = 0.0    # 背离强度 [0, 1]
    chan_structure_score: float = 0.0        # 结构对齐分数 [-1, 1]
    chan_buy_point: int = 0                  # 买点类型: 0/1/2/3 (来自chan_theory)
    chan_sell_point: int = 0                 # 卖点类型: 0/1/2/3 (来自chan_theory)
    signal_level: int = 0                    # 多级别确认: 3=双级别, 2=线段级, 1=笔级, 0=无

    # === 缠论走势类型与中枢 ===
    trend_type: int = 0                     # 走势类型: 2=上涨趋势, -2=下跌趋势, 1=盘整, 0=无
    chan_pivot_zg: float = float('nan')      # 中枢上沿
    chan_pivot_zd: float = float('nan')      # 中枢下沿
    chan_pivot_zz: float = float('nan')      # 中枢中轴

    # === 三系统共振 ===
    resonance_systems: int = 0               # 共振系统数: 0-3
    capital_flow_score: float = 0.0          # 系统2: 资金流向 [0,1]
    news_sentiment_score: float = 0.0        # 系统3: 资讯热点 [0,1]

    # === 当日数据 ===
    daily_return: float = 0.0               # 当日收益率 (选股日当天)
    volume_ratio: float = 1.0               # 当日量比 (vs 20日均量, 压缩值)

    # === 新因子：笔阶段+力竭+跳空+顶分型量能 ===
    exhaustion_risk: float = 0.0             # 力竭风险 [0, 1]
    gap_breakout_confirm: float = 0.0        # 跳空突破确认 [-1, 1]
    stroke_phase: float = 0.0                # 笔阶段 [-1, 1]
    top_fractal_volume: float = 0.0          # 顶分型量能背离 [-1, 1]

    # === 均线趋势（缠论买点的方向前提）===
    ma_trend_up: bool = False                # EMA20 > EMA60 均线多头排列

    # === 基本面排雷 ===
    profit_declining: bool = False            # 近两季度净利润同比持续下滑

    # === 预信号价格特征（用于均值回归过滤）===
    mom_60d: float = 0.0                     # 60日动量 (price vs 60d ago, %)
    dist_ma60: float = 0.0                   # 距MA60偏离 (%)
    max_dd_20d: float = 0.0                  # 20日最大回撤 (%, 负值)
    vol_regime: float = 1.0                  # 波动率区间 (短期vol/长期vol)

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
            'signal_confidence': self.signal_confidence,
            'chan_divergence_type': self.chan_divergence_type,
            'chan_divergence_strength': self.chan_divergence_strength,
            'chan_structure_score': self.chan_structure_score,
            'chan_buy_point': self.chan_buy_point,
            'chan_sell_point': self.chan_sell_point,
            'trend_type': self.trend_type,
            'resonance_systems': self.resonance_systems,
            'capital_flow_score': self.capital_flow_score,
            'news_sentiment_score': self.news_sentiment_score,
            'mom_60d': self.mom_60d,
            'dist_ma60': self.dist_ma60,
            'max_dd_20d': self.max_dd_20d,
            'vol_regime': self.vol_regime,
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
