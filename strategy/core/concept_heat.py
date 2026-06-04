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

from .industry_chain import compute_chain_signals, get_chain_lead_score, discover_chain_edges


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
        self._auto_edges: dict = {}  # 自动发现的产业链传导边
        self._auto_edges_computed = False
        self._scores_cache: Dict[str, tuple] = {}  # 日期→(concept_scores, chain_signals)
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

        # 自动发现产业链传导关系（从历史数据中挖掘）
        if len(self._concept_hist) >= 10:
            self._compute_auto_edges()
        self._loaded = True

    def _compute_auto_edges(self):
        """从历史概念数据自动发现传导关系

        使用3日滚动收益替代日频：单日噪音太大导致相关性不显著，
        3日聚合后信噪比提升，同时保留足够的时序分辨率。
        """
        if self._auto_edges_computed or len(self._concept_hist) < 10:
            return
        try:
            concept_3d_rets = {}
            for name, df in self._concept_hist.items():
                if len(df) < 20 or 'return' not in df.columns:
                    continue
                daily = df['return'].dropna().values
                if len(daily) < 20:
                    continue
                # 3日滚动累计收益: r[t-2]+r[t-1]+r[t] 的累积
                rolling_3d = np.array([
                    np.prod(1 + daily[max(0, i-2):i+1]) - 1
                    for i in range(2, len(daily))
                ])
                if len(rolling_3d) >= 10:
                    concept_3d_rets[name] = rolling_3d
            if len(concept_3d_rets) < 10:
                return
            names = list(concept_3d_rets.keys())
            self._auto_edges = discover_chain_edges(
                concept_3d_rets, names,
                min_correlation=0.35,  # 3日频阈值（日频0.55太高，周频0.30太低）
                max_lag=3,             # 3日频滞后1~3期 ≈ 3~9个自然日
                top_n=30,
            )
            self._auto_edges_computed = True
            total_edges = sum(len(v) for v in self._auto_edges.values())
            print(f"[ConceptHeat] 自动发现 {total_edges} 条产业链传导边 (来自 {len(names)} 个概念, 3日频)")
        except Exception:
            self._auto_edges = {}
            self._auto_edges_computed = True

    def set_daily_data(self, date, concept_returns: Dict[str, float] = None):
        """设置当日概念涨跌幅并计算产业链传导

        - 实盘模式: 优先从 data/concept_daily.csv 自动读取
        - 回测模式: 从 concept_hist 按日期查找
        - 手动模式: 传入 concept_returns
        """
        # 统一规范化日期键，确保所有路径使用一致缓存键
        cache_key = str(pd.Timestamp(date).date())
        # _scores_cache 已在 __init__ 初始化

        if concept_returns is not None:
            self._concept_scores = concept_returns
            self._chain_signals = compute_chain_signals(
                concept_returns, auto_discovered_edges=self._auto_edges
            )
            self._scores_cache[cache_key] = (concept_returns, self._chain_signals)
            return
        elif cache_key in self._scores_cache:
            self._concept_scores, self._chain_signals = self._scores_cache[cache_key]
            return
        else:
            # 尝试从当日概念数据CSV读取 (实盘 data_manager 已下载)
            if not hasattr(self, '_daily_csv_path'):
                self._daily_csv_path = Path(__file__).parent.parent.parent / "data" / "concept_daily.csv"
                self._daily_csv_exists = self._daily_csv_path.exists()
            if self._daily_csv_exists:
                try:
                    df = pd.read_csv(self._daily_csv_path)
                    if '涨跌幅' in df.columns and '板块名称' in df.columns:
                        self._concept_scores = dict(zip(
                            df['板块名称'], df['涨跌幅'].astype(float) / 100.0))
                except Exception:
                    self._daily_csv_exists = False

        if not self._concept_scores and self._concept_hist:
            # 回测: 从缓存历史中查找
            date_ts = pd.Timestamp(date)
            self._concept_scores = {}
            for name, df in self._concept_hist.items():
                row = df[df['date'] == date_ts]
                if len(row) > 0:
                    self._concept_scores[name] = float(row.iloc[0]['return'])
                else:
                    nearby = df[df['date'] <= date_ts]
                    if len(nearby) > 0:
                        self._concept_scores[name] = float(nearby.iloc[-1]['return'])

        # 计算产业链传导信号
        if self._concept_scores:
            self._chain_signals = compute_chain_signals(
                self._concept_scores, auto_discovered_edges=self._auto_edges
            )
        else:
            self._chain_signals = {}

        # 缓存结果供同日期其他股票复用（限300条防内存膨胀）
        # _scores_cache 已在 __init__ 初始化
        if len(self._scores_cache) > 300:
            oldest = min(self._scores_cache.keys())
            del self._scores_cache[oldest]
        self._scores_cache[cache_key] = (self._concept_scores, self._chain_signals)

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


def compute_concept_heat_for_date(date, codes: List[str]):
    """为指定日期批量计算股票的题材热度（回测用）

    Args:
        date: 日期 (str or Timestamp)
        codes: 股票代码列表

    Returns:
        np.ndarray of heat scores 0~1
    """
    calc = get_calculator()
    return calc.get_all_concept_heat(codes, date=date)
