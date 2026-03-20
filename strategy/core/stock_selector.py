# core/stock_selector.py
"""
统一选股模块

确保验证和回测使用完全相同的选股逻辑。
验证时模拟回测的选股过程，回测时直接调用同一模块。
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class StockSelection:
    """选股结果"""
    code: str
    score: float
    weight: float
    industry: str = ''
    rank_pct: float = 0.0


class StockSelector:
    """统一选股器

    验证和回测共用的选股逻辑：
    1. 筛选买入信号股票
    2. 按分数排序
    3. 选择Top-N股票
    4. 计算仓位权重
    """

    def __init__(
        self,
        max_position: int = 10,
        use_percentile: bool = False,  # 默认使用Top-N
        percentile_range: Tuple[float, float] = (0.7, 0.9),
    ):
        self.max_position = max_position
        self.use_percentile = use_percentile
        self.percentile_range = percentile_range

    def select(
        self,
        candidates: List[Dict],
        market_regime: int = 0,
    ) -> List[StockSelection]:
        """选择股票

        Args:
            candidates: 候选股票列表，每个元素包含:
                - code: 股票代码
                - score: 信号分数
                - risk_vol: 波动率
                - industry: 行业（可选）
                - risk_extreme: 是否极端状态（可选）
            market_regime: 市场状态 (-1熊市, 0震荡, 1牛市)

        Returns:
            选中的股票列表
        """
        if not candidates:
            return []

        # 1. 按分数排序
        sorted_candidates = sorted(candidates, key=lambda x: x['score'], reverse=True)

        # 2. 计算排名百分位
        n = len(sorted_candidates)
        for i, c in enumerate(sorted_candidates):
            c['rank_pct'] = 1.0 - i / max(n - 1, 1)

        # 3. 选股
        if self.use_percentile:
            # 分位选股（避免极端高分反转）
            low, high = self.percentile_range
            selected = [c for c in sorted_candidates if low < c['rank_pct'] <= high]
            if len(selected) < 3:
                # 回退到Top-N
                selected = sorted_candidates[:self.max_position]
        else:
            # Top-N选股
            selected = sorted_candidates[:self.max_position]

        # 限制数量
        selected = selected[:self.max_position]

        if not selected:
            return []

        # 4. 计算权重
        results = []
        total_weight = 0

        for c in selected:
            # 基础权重 = 分数（已归一化）
            base_weight = max(0.1, c['score'] + 0.5)  # 转换到正数区间

            # 波动率调整
            risk_vol = c.get('risk_vol', 0.2)
            risk_vol = max(0.01, min(1.0, risk_vol))
            vol_factor = min(1.0 / risk_vol, 3.0)

            # 极端状态调整
            extreme_factor = 0.7 if c.get('risk_extreme', False) else 1.0

            weight = base_weight * vol_factor * extreme_factor
            total_weight += weight

            results.append(StockSelection(
                code=c['code'],
                score=c['score'],
                weight=weight,
                industry=c.get('industry', ''),
                rank_pct=c['rank_pct'],
            ))

        # 归一化权重
        if total_weight > 0:
            for r in results:
                r.weight = r.weight / total_weight

        return results


def validate_with_backtest_logic(
    signals_df: pd.DataFrame,
    max_position: int = 10,
    forward_period: int = 20,
) -> Dict:
    """使用回测逻辑进行验证

    Args:
        signals_df: 信号数据，包含 date, code, score, future_ret 列
        max_position: 最大持仓数
        forward_period: 前瞻期

    Returns:
        验证结果
    """
    selector = StockSelector(max_position=max_position, use_percentile=False)

    results = []

    for date, group in signals_df.groupby('date'):
        # 构建候选列表
        candidates = []
        for _, row in group.iterrows():
            candidates.append({
                'code': row['code'],
                'score': row['score'],
                'risk_vol': row.get('risk_vol', 0.2),
                'industry': row.get('industry', ''),
                'future_ret': row['future_ret'],
            })

        # 选股
        selected = selector.select(candidates)

        if not selected:
            continue

        # 计算组合收益（加权）
        portfolio_ret = 0
        for s in selected:
            # 找到对应的future_ret
            for c in candidates:
                if c['code'] == s.code:
                    portfolio_ret += s.weight * c['future_ret']
                    break

        results.append({
            'date': date,
            'return': portfolio_ret,
            'n_stocks': len(selected),
        })

    if not results:
        return {}

    df = pd.DataFrame(results)

    return {
        'periods': len(df),
        'avg_return': df['return'].mean(),
        'std_return': df['return'].std(),
        'sharpe': df['return'].mean() / (df['return'].std() + 1e-10) * np.sqrt(252 / forward_period),
        'win_rate': (df['return'] > 0).mean(),
        'cumulative_return': (1 + df['return']).prod() - 1,
    }
