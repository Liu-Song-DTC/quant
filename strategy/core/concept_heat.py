# core/concept_heat.py
"""
题材热度因子 — 基于东方财富概念板块数据

两阶段设计:
1. 离线: stock_concept_map.pkl (股票→概念列表映射, 一次性采集)
2. 在线: stock_board_concept_name_em() (每日概念板块涨跌幅, 轻量调用)

计算: concept_heat = Σ(股票所属概念的涨跌幅 × 权重) / 概念数
"""

import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Optional


class ConceptHeatCalculator:
    """题材热度计算器"""

    def __init__(self, map_path: str = None):
        if map_path is None:
            map_path = str(Path(__file__).parent.parent.parent / "data" / "stock_concept_map.pkl")

        self._map_path = map_path
        self._stock_concepts: Dict[str, List[str]] = {}
        self._concept_stocks: Dict[str, List[str]] = {}
        self._concept_scores: Dict[str, float] = {}  # {概念名: 当日涨跌幅}
        self._loaded = False

    def load(self):
        """加载股票→概念映射"""
        if self._loaded:
            return
        with open(self._map_path, 'rb') as f:
            self._stock_concepts = pickle.load(f)

        # 构建反向索引: 概念→股票列表
        for code, concepts in self._stock_concepts.items():
            for c in concepts:
                self._concept_stocks.setdefault(c, []).append(code)

        self._loaded = True
        print(f"[ConceptHeat] 已加载 {len(self._stock_concepts)} 只股票的题材映射")

    def update_concept_scores(self, concept_returns: Dict[str, float]):
        """更新当日概念涨跌幅

        Args:
            concept_returns: {概念名: 涨跌幅(float)}
        """
        self._concept_scores = concept_returns

    def get_concept_heat(self, code: str) -> float:
        """获取单只股票的题材热度

        Returns:
            0~1 之间的热度得分
        """
        if not self._loaded:
            self.load()

        concepts = self._stock_concepts.get(code, [])
        if not concepts or not self._concept_scores:
            return 0.5  # 默认中等

        scores = []
        for c in concepts:
            ret = self._concept_scores.get(c)
            if ret is not None:
                # tanh 压缩涨跌幅到 (0,1)，涨幅越大热度越高
                scores.append(np.tanh(max(ret, 0) * 5.0))

        if not scores:
            return 0.5

        return float(np.clip(np.mean(scores), 0.0, 1.0))

    def get_all_concept_heat(self, codes: List[str]) -> np.ndarray:
        """批量获取股票题材热度

        Returns:
            numpy array [n] of heat scores 0~1
        """
        if not self._loaded:
            self.load()

        result = np.full(len(codes), 0.5)
        if not self._concept_scores:
            return result

        for i, code in enumerate(codes):
            result[i] = self.get_concept_heat(code)
        return result

    def get_top_concepts(self, n: int = 30) -> List[str]:
        """获取当日涨幅最大的前N个概念板块"""
        sorted_concepts = sorted(self._concept_scores.items(),
                                key=lambda x: -x[1])
        return [c for c, _ in sorted_concepts[:n]]

    def get_top_concept_stocks(self, n_concepts: int = 30) -> List[str]:
        """获取当日最热概念板块的成分股"""
        top = self.get_top_concepts(n_concepts)
        stocks = set()
        for c in top:
            stocks.update(self._concept_stocks.get(c, []))
        return list(stocks)


# 全局单例
_calculator: Optional[ConceptHeatCalculator] = None


def get_calculator() -> ConceptHeatCalculator:
    global _calculator
    if _calculator is None:
        _calculator = ConceptHeatCalculator()
        _calculator.load()
    return _calculator


def compute_concept_heat(close_arr, volume_surge, turnover_burst, limit_up_freq,
                         excess_return_5d=None, stock_codes=None, daily_concept_returns=None):
    """numpy 兼容接口 — 回退到本地合成版（当无概念数据时）"""
    n = len(close_arr)
    result = np.full(n, 0.5)

    try:
        calc = get_calculator()
        if daily_concept_returns:
            calc.update_concept_scores(daily_concept_returns)
        if stock_codes:
            return calc.get_all_concept_heat(stock_codes)
    except Exception:
        pass

    # Fallback: 本地合成版
    yaogu = (np.tanh(np.maximum(volume_surge, 0) * 1.5) +
             np.tanh(np.maximum(turnover_burst, 0) * 1.5) +
             np.tanh(np.maximum(limit_up_freq, 0) * 1.5)) / 3.0

    result = 0.4 * yaogu + 0.3 * 0.5 + 0.3 * np.clip(np.tanh(excess_return_5d * 3.0) if excess_return_5d is not None else 0.5, 0, 1)
    return np.clip(result, 0.0, 1.0)
