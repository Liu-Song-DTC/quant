# core/multi_strategy.py
"""
多策略框架 — 趋势跟踪 / 均值回归 / 防御 三子策略并行评分。

设计: 不重构 portfolio.build()，仅作为后处理层调整单策略输出的权重。
默认关闭(enabled: false)，开启后按市场状态混合子策略评分。
"""

import numpy as np
from typing import Dict, List, Any, Optional
from .config_loader import load_config


class MultiStrategyWeights:
    """三子策略并行 → 按市场状态混合 → 输出调整后的股票权重"""

    def __init__(self):
        config = load_config()
        ms = config.get('multi_strategy', {}) if hasattr(config, 'get') else {}
        self.enabled = ms.get('enabled', False)

        # 子策略权重
        trend_cfg = ms.get('trend', {})
        rev_cfg = ms.get('reversion', {})
        def_cfg = ms.get('defensive', {})
        self.trend_w = trend_cfg.get('weight', 0.40)
        self.rev_w = rev_cfg.get('weight', 0.30)
        self.def_w = def_cfg.get('weight', 0.30)

        # 趋势策略参数
        self.min_chan_signal = trend_cfg.get('min_chan_signal', 2)

        # 均值回归策略参数
        self.max_exhaustion = rev_cfg.get('max_exhaustion', 0.30)

        # 防御策略参数
        self.def_dd_trigger = def_cfg.get('max_drawdown_trigger', 0.12)

        # 市场状态倾斜
        blending = ms.get('blending', {})
        self._blending = {
            'bull_tilt': blending.get('bull_tilt', 'trend'),
            'bear_tilt': blending.get('bear_tilt', 'defensive'),
        }

    def compute_weights(self, candidates: List[Dict], signal_store,
                        market_regime: int, drawdown: float) -> Dict[str, float]:
        """计算每个候选股票的多策略调整系数。

        Returns:
            {code: multiplier} — 1.0=不变, >1.0=加仓, <1.0=减仓, 0.0=拒绝
        """
        if not self.enabled or not candidates:
            return {}

        regime_tilt = {1: self._blending.get('bull_tilt'),
                       -1: self._blending.get('bear_tilt')}.get(market_regime)

        results = {}
        for c in candidates:
            code = c.get('code', '')
            sig = c.get('sig')
            if sig is None:
                results[code] = 1.0
                continue

            sl = getattr(sig, 'signal_level', 0)
            exhaustion = getattr(sig, 'exhaustion_risk', 0)
            bp = getattr(sig, 'chan_buy_point', 0)
            cs = getattr(sig, 'chan_sell_point', 0)
            tt = getattr(sig, 'trend_type', 0)

            # 趋势评分: Chan买点 + 多级别确认 + 趋势方向
            trend_score = 0.5
            if sl >= self.min_chan_signal or bp >= 2:
                trend_score = min(1.0, 0.5 + 0.25 * sl + 0.15 * (tt == 2))
            elif bp == 1 and sl >= 1:
                trend_score = 0.7

            # 均值回归评分: 力竭低 + B1买点 + 非追高
            if exhaustion <= self.max_exhaustion:
                rev_score = 0.6 + 0.4 * (1.0 - exhaustion / max(self.max_exhaustion, 0.01))
            elif bp == 1:
                rev_score = 0.6
            elif exhaustion <= 0.15:
                rev_score = 0.7
            else:
                rev_score = 0.3

            # 防御评分: 卖点少 + 力竭低 + 趋势不差
            def_score = 1.0
            if cs >= 3:
                def_score = 0.0
            elif cs >= 2:
                def_score = 0.3
            elif exhaustion > 0.4:
                def_score = 0.3
            elif cs == 1:
                def_score = 0.5
            elif tt == -2:
                def_score = 0.4

            # 混合
            if regime_tilt == 'trend':
                multiplier = self.trend_w * trend_score + self.rev_w * rev_score + self.def_w * def_score
            elif regime_tilt == 'defensive':
                multiplier = 0.3 * trend_score + 0.2 * rev_score + 0.5 * def_score
            else:
                multiplier = self.trend_w * trend_score + self.rev_w * rev_score + self.def_w * def_score

            results[code] = float(np.clip(multiplier, 0.0, 1.5))

        return results
