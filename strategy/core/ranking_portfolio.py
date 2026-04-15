# core/ranking_portfolio.py
"""
排名组合构建器 - 基于因子排名选股

核心思想:
- 因子值高 -> 排名高 -> 入选
- 直接利用IC的排序能力，而非阈值判断方向
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from copy import deepcopy
from .config_loader import load_config


class RankingPortfolio:
    """
    排名组合构建器

    简化版选股逻辑:
    1. 收集所有股票的因子值
    2. 截面标准化（排名百分位）
    3. 选择Top-N
    4. 风险调整权重
    """

    def __init__(self, config=None):
        self.config = config or load_config()

        portfolio_config = self.config.get('portfolio', {})
        self.max_position = portfolio_config.get('max_position', 10)
        self.target_volatility = portfolio_config.get('target_volatility', 0.20)

        backtest_config = self.config.get('backtest', {})
        self.commission = backtest_config.get('commission', 0.0015)
        self.slippage = backtest_config.get('slippage', 0.0015)

        # 行业因子配置
        self.industry_factors = self.config.get('industry_factors', {})

        # 记录
        self.last_selection = []
        self.equity_history = []

    def _get_factor_list(self, industry: str) -> List[str]:
        """获取行业的因子列表"""
        factors = self.industry_factors.get(industry, {}).get('factors', ['fund_score'])
        return factors if factors else ['fund_score']

    def build(
        self,
        date,
        universe: List[str],
        signal_store,
        cash: float,
        prices: Dict[str, float],
        current_positions: Dict[str, float],
        market_regime: int = 0,
        rebalance: bool = False,
    ) -> Dict[str, float]:
        """
        构建目标持仓

        Args:
            date: 当前日期
            universe: 可交易股票列表
            signal_store: 信号存储（包含factor_value等）
            cash: 现金
            prices: 当前价格
            current_positions: 当前持仓 {code: market_value}
            market_regime: 市场状态
            rebalance: 是否调仓日

        Returns:
            目标持仓 {code: market_value}
        """
        if not rebalance:
            return current_positions

        # 1. 收集所有股票的因子值
        candidates = []
        for code in universe:
            sig = signal_store.get(code, date)
            if sig is None:
                continue

            # 使用 factor_value 作为排名依据
            # 不再使用 buy/sell 阈值判断
            factor_value = getattr(sig, 'factor_value', None)
            if factor_value is None or np.isnan(factor_value):
                continue

            candidates.append({
                'code': code,
                'factor_value': factor_value,
                'volatility': getattr(sig, 'risk_vol', 0.03),
                'industry': getattr(sig, 'industry', 'default'),
                'score': getattr(sig, 'score', 0),
            })

        if not candidates:
            return {}

        # 2. 截面标准化 - 使用排名百分位
        factor_values = np.array([c['factor_value'] for c in candidates])
        # 排名百分位 (0-1)
        ranks = pd.Series(factor_values).rank(pct=True)

        for i, c in enumerate(candidates):
            c['rank_pct'] = ranks.iloc[i]

        # 3. 按排名选择Top-N
        candidates.sort(key=lambda x: x['rank_pct'], reverse=True)

        # 行业分散限制
        selected = []
        industry_count = {}
        industry_cap = 2  # 单个行业最多选2只

        for c in candidates:
            ind = c['industry']
            if industry_count.get(ind, 0) >= industry_cap:
                continue
            selected.append(c)
            industry_count[ind] = industry_count.get(ind, 0) + 1
            if len(selected) >= self.max_position:
                break

        if not selected:
            return {}

        # 4. 计算权重
        # 使用指数衰减权重
        weights = []
        for i, c in enumerate(selected):
            w = np.exp(-0.15 * i)  # rank 0: 1.0, rank 1: 0.86, ...
            weights.append(w)

        # 波动率调整
        for i, c in enumerate(selected):
            vol = max(0.01, min(1.0, c.get('volatility', 0.03)))
            vol_factor = min(1.0 / vol, 2.0)
            weights[i] *= vol_factor

        # 归一化
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]

        # 5. 计算目标市值
        total_equity = cash + sum(current_positions.values())

        # 根据市场状态调整仓位
        exposure = 1.0
        if market_regime == -1:  # 熊市
            exposure = 0.5
        elif market_regime == 0:  # 中性
            exposure = 0.8

        desired_value = {}
        for i, c in enumerate(selected):
            desired_value[c['code']] = weights[i] * total_equity * exposure

        # 记录选股结果
        self.last_selection = [
            {
                'date': date,
                'code': c['code'],
                'score': c['score'],
                'weight': weights[i],
                'industry': c['industry'],
                'rank_pct': c['rank_pct'],
            }
            for i, c in enumerate(selected)
        ]

        return desired_value
