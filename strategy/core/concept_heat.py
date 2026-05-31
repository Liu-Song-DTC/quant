# core/concept_heat.py
"""
题材热度因子 — 基于东方财富概念板块数据 + 产业链传导

三层次信号:
1. 概念板块涨跌幅: stock_board_concept_name_em() 日更
2. 产业链传导: 上游涨→埋伏下游 (core/industry_chain.py)
3. 妖股信号: volume_surge + turnover_burst + limit_up_freq

回测: 使用 concept_hist.pkl 缓存的历史数据
实盘: 每日调用一次 stock_board_concept_name_em()
"""

import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd

from .industry_chain import compute_chain_signals, get_chain_lead_score


class ConceptHeatCalculator:
    """题材热度计算器 — 支持回测和实盘双模式"""

    def __init__(self, map_path: str = None):
        if map_path is None:
            map_path = str(Path(__file__).parent.parent.parent / "data" / "stock_concept_map.pkl")

        self._map_path = map_path
        self._stock_concepts: Dict[str, List[str]] = {}
        self._concept_stocks: Dict[str, List[str]] = {}
        self._concept_scores: Dict[str, float] = {}
        self._chain_signals: Dict[str, float] = {}
        self._concept_hist: Dict[str, pd.DataFrame] = {}
        self._loaded = False

    def load(self, hist_path: str = None):
        """加载股票→概念映射 + 历史概念数据"""
        if self._loaded:
            return

        with open(self._map_path, 'rb') as f:
            self._stock_concepts = pickle.load(f)

        for code, concepts in self._stock_concepts.items():
            for c in concepts:
                self._concept_stocks.setdefault(c, []).append(code)

        # 加载概念历史数据(回测用)
        if hist_path is None:
            hist_path = str(Path(__file__).parent.parent.parent / "data" / "concept_hist.pkl")
        try:
            with open(hist_path, 'rb') as f:
                self._concept_hist = pickle.load(f)
            print(f"[ConceptHeat] 已加载 {len(self._stock_concepts)} 股票映射 + {len(self._concept_hist)} 概念历史")
        except FileNotFoundError:
            print(f"[ConceptHeat] 已加载 {len(self._stock_concepts)} 股票映射 (无历史数据)")

        self._loaded = True

    def set_daily_data(self, date, concept_returns: Dict[str, float] = None):
        """设置当日概念涨跌幅并计算产业链传导

        - 回测模式: 从 concept_hist 按日期查找
        - 实盘模式: 传入 concept_returns
        """
        if concept_returns is not None:
            self._concept_scores = concept_returns
        elif self._concept_hist:
            # 回测: 从缓存历史中查找
            date_ts = pd.Timestamp(date)
            self._concept_scores = {}
            for name, df in self._concept_hist.items():
                row = df[df['date'] == date_ts]
                if len(row) > 0:
                    self._concept_scores[name] = float(row.iloc[0]['return'])
                else:
                    # 尝试最近日期
                    nearby = df[df['date'] <= date_ts]
                    if len(nearby) > 0:
                        self._concept_scores[name] = float(nearby.iloc[-1]['return'])

        # 计算产业链传导信号
        if self._concept_scores:
            self._chain_signals = compute_chain_signals(self._concept_scores)
        else:
            self._chain_signals = {}

    def get_concept_heat(self, code: str) -> float:
        """获取单只股票的题材热度 (概念涨幅 + 产业链埋伏)"""
        if not self._loaded:
            self.load()

        concepts = self._stock_concepts.get(code, [])
        if not concepts or not self._concept_scores:
            return 0.5

        # 维度1: 概念板块涨幅
        scores = []
        for c in concepts:
            ret = self._concept_scores.get(c)
            if ret is not None:
                scores.append(np.tanh(max(ret, 0) * 5.0))

        concept_score = float(np.clip(np.mean(scores), 0.0, 1.0)) if scores else 0.5

        # 维度2: 产业链埋伏加分
        chain_score = get_chain_lead_score(code, self._stock_concepts, self._chain_signals)

        # 融合: 70%概念涨幅 + 30%产业链埋伏
        return float(np.clip(0.7 * concept_score + 0.3 * chain_score, 0.0, 1.0))

    def get_all_concept_heat(self, codes: List[str], date=None) -> np.ndarray:
        """批量获取股票题材热度"""
        if not self._loaded:
            self.load()

        if date is not None:
            self.set_daily_data(date)

        result = np.full(len(codes), 0.5)
        if not self._concept_scores:
            return result

        for i, code in enumerate(codes):
            result[i] = self.get_concept_heat(code)
        return result

    def get_top_concepts(self, n: int = 30) -> List[str]:
        sorted_concepts = sorted(self._concept_scores.items(), key=lambda x: -x[1])
        return [c for c, _ in sorted_concepts[:n]]

    def get_chain_lead_concepts(self, min_score: float = 0.05) -> Dict[str, float]:
        """获取产业链埋伏信号中分数>threshold的概念"""
        return {k: v for k, v in sorted(self._chain_signals.items(),
                key=lambda x: -x[1]) if v > min_score}


# 全局单例
_calculator: Optional[ConceptHeatCalculator] = None

def get_calculator() -> ConceptHeatCalculator:
    global _calculator
    if _calculator is None:
        _calculator = ConceptHeatCalculator()
        _calculator.load()
    return _calculator


def compute_concept_heat(volume_surge, turnover_burst, limit_up_freq,
                         excess_return_5d=None, stock_codes=None, daily_concept_returns=None):
    """numpy兼容接口 — 本地合成版(fallback)"""
    n = len(volume_surge)
    result = np.full(n, 0.5)

    try:
        calc = get_calculator()
        if daily_concept_returns:
            calc.update_concept_scores(daily_concept_returns)
        if stock_codes:
            return calc.get_all_concept_heat(stock_codes)
    except Exception:
        pass

    yaogu = (np.tanh(np.maximum(volume_surge, 0) * 1.5) +
             np.tanh(np.maximum(turnover_burst, 0) * 1.5) +
             np.tanh(np.maximum(limit_up_freq, 0) * 1.5)) / 3.0
    result = 0.4 * yaogu + 0.3 * 0.5 + 0.3 * np.clip(
        np.tanh(excess_return_5d * 3.0) if excess_return_5d is not None else 0.5, 0, 1)
    return np.clip(result, 0.0, 1.0)
