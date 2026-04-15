# core/ranking_engine.py
"""
排名信号引擎 - 直接利用IC的排序能力

核心思想:
- IC衡量的是"因子值高的股票未来收益排名更高"
- 因此应该用排名选股，而非阈值判断方向

设计原则:
1. 简洁：去除复杂的阈值逻辑
2. 直接：因子值高 -> 排名高 -> 入选
3. 稳定：截面标准化消除极端值影响
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

from .config_loader import load_config
from .industry_mapping import INDUSTRY_KEYWORDS
from .factor_calculator import calculate_indicators, compute_composite_factors


@dataclass
class RankingSignal:
    """排名信号"""
    factor_value: float      # 因子值（截面标准化后）
    raw_factor_value: float  # 原始因子值
    factor_name: str         # 因子名称
    industry: str            # 行业
    volatility: float        # 波动率（用于风险调整）
    rank: int = 0            # 全市场排名
    industry_rank: int = 0   # 行业内排名


class RankingEngine:
    """
    排名信号引擎

    工作流程:
    1. 对每个调仓日，计算所有股票的因子值
    2. 截面标准化因子值
    3. 按因子值排名
    4. 返回排名结果供组合层使用
    """

    def __init__(self, config=None):
        self.config = config or load_config()

        # 加载行业因子配置
        self.industry_factors = self.config.get('industry_factors', {})

        # 获取因子列表
        self.backtest_factors = self.config.get('backtest_factors', [])

        # 动态因子配置
        self.factor_mode = self.config.get('factor_mode', 'fixed')
        self.dynamic_config = self.config.get('dynamic_factor', {})

        # 基本面数据
        self.fundamental_data = None
        self._init_fundamental()

    def _init_fundamental(self):
        """初始化基本面数据"""
        try:
            from .fundamental import FundamentalData
            import os
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            data_dir = os.path.join(os.path.dirname(base_dir), 'data/stock_data/fundamental_data')
            if os.path.exists(data_dir):
                self.fundamental_data = FundamentalData(data_dir, [])
        except Exception as e:
            print(f"[RankingEngine] 基本面数据初始化失败: {e}")

    def _get_industry(self, code: str, date) -> str:
        """获取股票行业"""
        if not self.fundamental_data or not code:
            return 'default'
        try:
            industry_name = self.fundamental_data.get_industry(code, date)
            if not industry_name:
                return 'default'

            # 映射到行业分类
            for category, keywords in INDUSTRY_KEYWORDS.items():
                for kw in keywords:
                    if kw in industry_name:
                        return category
            return 'default'
        except:
            return 'default'

    def _get_fundamental_factor_value(self, code: str, date, factor_name: str) -> float:
        """获取基本面因子值"""
        if not self.fundamental_data:
            return 0.0

        try:
            # 映射因子名到基本面字段
            factor_map = {
                'fund_score': 'score',
                'fund_profit_growth': 'profit_growth',
                'fund_revenue_growth': 'revenue_growth',
                'fund_roe': 'roe',
                'fund_cf_to_profit': 'cf_to_profit',
                'fund_gross_margin': 'gross_margin',
            }

            field = factor_map.get(factor_name)
            if not field:
                return 0.0

            # 获取基本面数据
            df = self.fundamental_data.get_fundamental_df(code)
            if df is None or len(df) == 0:
                return 0.0

            # 找到报告日期之前最近的数据
            if isinstance(date, str):
                date = pd.to_datetime(date)

            df = df[df['report_date'] <= date]
            if len(df) == 0:
                return 0.0

            row = df.iloc[-1]
            value = row.get(field, 0.0)

            # 标准化处理
            if field == 'score':
                return np.clip(value / 100, -1, 1)  # score是0-100
            else:
                return np.clip(np.tanh(value), -1, 1)  # 其他字段用tanh压缩

        except Exception as e:
            return 0.0

    def _calculate_factor_value(
        self,
        code: str,
        date,
        stock_data: pd.DataFrame,
        factor_list: List[str]
    ) -> Tuple[float, str, float]:
        """
        计算单个股票的因子值

        Returns:
            (factor_value, factor_name, volatility)
        """
        if len(stock_data) < 60:
            return 0.0, 'INSUFFICIENT_DATA', 0.03

        # 确保日期格式
        if isinstance(date, str):
            date = pd.to_datetime(date)

        # 找到对应日期的数据
        stock_data = stock_data.copy()
        if 'datetime' in stock_data.columns:
            stock_data['date'] = pd.to_datetime(stock_data['datetime'])
        elif 'date' in stock_data.columns:
            stock_data['date'] = pd.to_datetime(stock_data['date'])

        # 筛选到当前日期为止的数据
        hist_data = stock_data[stock_data['date'] <= date].copy()
        if len(hist_data) < 60:
            return 0.0, 'INSUFFICIENT_DATA', 0.03

        # 计算技术指标
        close = hist_data['close'].values
        high = hist_data['high'].values if 'high' in hist_data.columns else close
        low = hist_data['low'].values if 'low' in hist_data.columns else close
        volume = hist_data['volume'].values if 'volume' in hist_data.columns else np.ones(len(close))

        indicators = calculate_indicators(close, high, low, volume)

        # 计算波动率（用于风险调整）
        volatility = indicators.get('volatility_20', [0.03])[-1]
        if np.isnan(volatility) or volatility <= 0:
            volatility = 0.03

        # 获取行业
        industry = self._get_industry(code, date)

        # 选择因子列表
        if not factor_list:
            # 根据行业选择因子
            factor_list = self.industry_factors.get(industry, {}).get('factors', ['fund_score'])

        # 计算因子值
        factor_values = []
        for factor_name in factor_list:
            if factor_name.startswith('fund_'):
                # 基本面因子
                fv = self._get_fundamental_factor_value(code, date, factor_name)
            else:
                # 技术因子
                fv = self._get_technical_factor_value(indicators, factor_name)

            if not np.isnan(fv):
                factor_values.append(fv)

        if not factor_values:
            return 0.0, 'NO_FACTOR', volatility

        # 等权组合
        factor_value = np.mean(factor_values)
        factor_name = '+'.join(factor_list[:3]) if len(factor_list) <= 3 else f"{factor_list[0]}+{len(factor_list)}F"

        return factor_value, factor_name, volatility

    def _get_technical_factor_value(self, indicators: Dict, factor_name: str) -> float:
        """从技术指标中获取因子值"""
        if factor_name not in indicators:
            return np.nan

        value = indicators[factor_name][-1] if isinstance(indicators[factor_name], np.ndarray) else indicators[factor_name]

        if np.isnan(value):
            return np.nan

        return float(value)

    def compute_ranking(
        self,
        date,
        stock_data_dict: Dict[str, pd.DataFrame],
        current_positions: Dict[str, float] = None
    ) -> Dict[str, RankingSignal]:
        """
        计算截面排名

        Args:
            date: 当前日期
            stock_data_dict: {code: DataFrame} 股票数据字典
            current_positions: 当前持仓（用于持仓检查）

        Returns:
            {code: RankingSignal} 每个股票的排名信号
        """
        # 1. 计算所有股票的因子值
        results = {}

        for code, df in stock_data_dict.items():
            try:
                # 获取行业
                industry = self._get_industry(code, date)

                # 获取该行业的因子列表
                factor_list = self.industry_factors.get(industry, {}).get('factors', ['fund_score'])

                fv, fn, vol = self._calculate_factor_value(code, date, df, factor_list)

                if fn not in ['INSUFFICIENT_DATA', 'NO_FACTOR']:
                    results[code] = {
                        'factor_value': fv,
                        'factor_name': fn,
                        'industry': industry,
                        'volatility': vol,
                    }
            except Exception as e:
                continue

        if not results:
            return {}

        # 2. 截面标准化
        factor_values = np.array([r['factor_value'] for r in results.values()])

        # 使用排名百分位（更鲁棒）
        ranks = pd.Series(factor_values).rank(pct=True)
        codes = list(results.keys())

        for i, code in enumerate(codes):
            results[code]['raw_factor_value'] = results[code]['factor_value']
            results[code]['factor_value'] = ranks.iloc[i]  # 0-1之间的排名百分位

        # 3. 计算排名
        sorted_codes = sorted(codes, key=lambda c: results[c]['factor_value'], reverse=True)

        for rank, code in enumerate(sorted_codes):
            results[code]['rank'] = rank + 1

        # 4. 计算行业内排名
        industry_groups = {}
        for code in codes:
            ind = results[code]['industry']
            if ind not in industry_groups:
                industry_groups[ind] = []
            industry_groups[ind].append(code)

        for ind, ind_codes in industry_groups.items():
            ind_codes_sorted = sorted(ind_codes, key=lambda c: results[c]['factor_value'], reverse=True)
            for rank, code in enumerate(ind_codes_sorted):
                results[code]['industry_rank'] = rank + 1

        # 5. 构建返回结果
        signals = {}
        for code, data in results.items():
            signals[code] = RankingSignal(
                factor_value=data['factor_value'],
                raw_factor_value=data['raw_factor_value'],
                factor_name=data['factor_name'],
                industry=data['industry'],
                volatility=data['volatility'],
                rank=data['rank'],
                industry_rank=data['industry_rank'],
            )

        return signals

    def select_top_n(
        self,
        signals: Dict[str, RankingSignal],
        n: int = 10,
        industry_cap: int = 3
    ) -> List[Tuple[str, float, RankingSignal]]:
        """
        选择Top-N股票

        Args:
            signals: 排名信号字典
            n: 总选股数量
            industry_cap: 单个行业最多选几只

        Returns:
            [(code, weight, signal), ...]
        """
        # 按因子值排序
        sorted_items = sorted(signals.items(), key=lambda x: x[1].factor_value, reverse=True)

        selected = []
        industry_count = {}

        for code, sig in sorted_items:
            # 行业限制
            ind = sig.industry
            if industry_count.get(ind, 0) >= industry_cap:
                continue

            # 计算权重（排名越高权重越大）
            # 使用指数衰减权重
            rank = len(selected) + 1
            weight = np.exp(-0.15 * (rank - 1))  # rank 1: 1.0, rank 2: 0.86, rank 3: 0.74

            selected.append((code, weight, sig))
            industry_count[ind] = industry_count.get(ind, 0) + 1

            if len(selected) >= n:
                break

        # 归一化权重
        total_weight = sum(w for _, w, _ in selected)
        if total_weight > 0:
            selected = [(c, w / total_weight, s) for c, w, s in selected]

        return selected
