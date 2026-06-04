# core/sector_rotation.py
"""
板块轮动分析 — 行业动量排名 + 强弱识别

集成点:
- signal_runner 在信号生成后调用 compute()，传入 stock_data_dict 和 industry_codes
- portfolio 在选股评分时调用 get_sector_tilt()，给强势行业加分、弱势行业减分

方法:
- compute(): 计算各行业多周期动量排名
- get_sector_score(): 单只股票所属行业的动量分数 (-1~1, 截面rank)
- get_sector_tilt(): 组合层行业倾斜乘数 (0.85~1.15)
- get_strong/weak_sectors(): 领涨/领跌行业列表
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional


class SectorRotation:
    """行业轮动分析器 — 多周期动量 + 截面排名"""

    def __init__(self):
        self._sector_momentum: Dict[str, float] = {}   # {industry: composite_momentum}
        self._sector_rank: Dict[str, float] = {}        # {industry: rank_pct 0~1}
        self._sector_tilt: Dict[str, float] = {}        # {industry: tilt_multiplier}
        self._signal_density: Dict[str, float] = {}     # {industry: density}
        self._signal_density_rank: Dict[str, float] = {}  # {industry: density_rank}
        self._computed = False
        # 轮动速度追踪：记录每期的领涨行业top3，用于检测切换过快
        self._top_sectors_history: list = []  # [(date_str, [top3_industries]), ...]
        self._rotation_speed: float = 0.0     # 0=稳定, 1=极快切换
        self._last_compute_date: str = ""

    # ── 公共接口 ──

    def compute(
        self,
        stock_data_dict: dict,
        industry_codes: dict,  # {industry: [codes]}
        lookback_short: int = 10,
        lookback_mid: int = 20,
        lookback_long: int = 60,
        date_str: str = "",
    ):
        """计算各行业多周期动量排名

        Args:
            stock_data_dict: {code: DataFrame} 日线数据
            industry_codes: {industry_name: [code, ...]}
            lookback_short/mid/long: 短/中/长周期
            date_str: 日期标识，用于轮动速度追踪
        """
        if not stock_data_dict or not industry_codes:
            self._computed = False
            return

        # 构建 code → industry 反向映射
        code_to_ind = {}
        for ind, codes in industry_codes.items():
            for c in codes:
                code_to_ind[c] = ind

        sector_rets: Dict[str, List[float]] = {ind: [] for ind in industry_codes}

        for code, df in stock_data_dict.items():
            industry = code_to_ind.get(code)
            if industry is None:
                continue
            if len(df) < lookback_long:
                continue

            close = df['close'].values
            last = close[-1]

            # 短/中/长期动量
            if len(close) > lookback_short:
                mom_s = (last / close[-lookback_short] - 1) if close[-lookback_short] > 0 else 0
            else:
                mom_s = 0
            mom_m = (last / close[-lookback_mid] - 1) if close[-lookback_mid] > 0 else 0
            mom_l = (last / close[-lookback_long] - 1) if close[-lookback_long] > 0 else 0

            # 复合动量: 短40% + 中35% + 长25%
            composite = mom_s * 0.40 + mom_m * 0.35 + mom_l * 0.25
            sector_rets[industry].append(composite)

        # 行业均值 + 截面排名
        self._sector_momentum = {}
        for ind, rets in sector_rets.items():
            if rets:
                self._sector_momentum[ind] = float(np.median(rets))

        if not self._sector_momentum:
            self._computed = False
            return

        # 截面排名 (0~1)
        vals = np.array(list(self._sector_momentum.values()))
        ranks = pd.Series(vals).rank(pct=True).values
        self._sector_rank = {
            ind: float(ranks[i])
            for i, ind in enumerate(self._sector_momentum.keys())
        }

        # 行业倾斜乘数: rank_pct 线性映射到 [0.85, 1.15]
        self._sector_tilt = {}
        for ind, rp in self._sector_rank.items():
            self._sector_tilt[ind] = float(0.85 + rp * 0.30)

        self._computed = True

        # ── 轮动速度追踪 ──
        # 记录当日前3领涨行业，与上一次对比计算切换率
        if date_str:
            sorted_items = sorted(self._sector_rank.items(), key=lambda x: -x[1])
            top3 = [s[0] for s in sorted_items[:3]]
            self._top_sectors_history.append((date_str, top3))
            # 保留最近20期
            if len(self._top_sectors_history) > 20:
                self._top_sectors_history = self._top_sectors_history[-20:]
            # 计算轮动速度：连续两期top3的切换比例 (0=完全重合, 1=完全不同)
            if len(self._top_sectors_history) >= 2:
                prev_top3 = set(self._top_sectors_history[-2][1])
                curr_top3 = set(self._top_sectors_history[-1][1])
                n_changed = len(curr_top3 - prev_top3)
                self._rotation_speed = float(0.7 * self._rotation_speed + 0.3 * (n_changed / 3.0))
            self._last_compute_date = date_str

    def is_ready(self) -> bool:
        return self._computed

    def get_sector_score(self, code: str, industry: str = None) -> float:
        """单只股票的行业动量分数 (-1~1)，未计算时返回 0"""
        if not self._computed:
            return 0.0
        ind = industry if industry else ''
        if ind not in self._sector_momentum:
            return 0.0
        rp = self._sector_rank.get(ind, 0.5)
        return float(rp * 2 - 1)  # rank_pct → [-1, 1]

    def compute_signal_density(
        self,
        buy_signals: list,  # [(code, industry), ...]
        industry_stock_counts: dict,  # {industry: total_stocks_count}
    ):
        """计算各行业买入信号密度 — 领先指标

        信号密度 = 行业买入信号数 / 行业总股票数
        高密度意味着该行业正在形成买入共识，可能先于价格上涨

        Args:
            buy_signals: [(code, industry), ...] 当天触发买入的股票列表
            industry_stock_counts: {industry: count} 各行业股票池总数
        """
        if not buy_signals or not industry_stock_counts:
            self._signal_density = {}
            self._signal_density_rank = {}
            return

        buy_counts: Dict[str, int] = {}
        for _, ind in buy_signals:
            if ind:
                buy_counts[ind] = buy_counts.get(ind, 0) + 1

        self._signal_density = {}
        for ind, total in industry_stock_counts.items():
            buys = buy_counts.get(ind, 0)
            if total > 0:
                self._signal_density[ind] = buys / total

        if not self._signal_density:
            self._signal_density_rank = {}
            return

        # 截面排名
        vals = np.array(list(self._signal_density.values()))
        ranks = pd.Series(vals).rank(pct=True).values
        self._signal_density_rank = {
            ind: float(ranks[i])
            for i, ind in enumerate(self._signal_density.keys())
        }

        # 轮动速度追踪: 基于信号密度的top3变化
        sorted_items = sorted(self._signal_density_rank.items(), key=lambda x: -x[1])
        top3 = [s[0] for s in sorted_items[:3]]
        self._top_sectors_history.append(("", top3))
        if len(self._top_sectors_history) > 20:
            self._top_sectors_history = self._top_sectors_history[-20:]
        if len(self._top_sectors_history) >= 2:
            prev_top3 = set(self._top_sectors_history[-2][1])
            curr_top3 = set(self._top_sectors_history[-1][1])
            n_changed = len(curr_top3 - prev_top3)
            self._rotation_speed = float(0.7 * self._rotation_speed + 0.3 * (n_changed / 3.0))

    def get_composite_tilt(self, industry: str) -> float:
        """综合行业倾斜乘数: 动量(滞后)40% + 信号密度(领先)60%

        优先跟随信号密度，因为它是领先指标；动量作为验证。

        防反馈环: 单个行业tilt上限1.15（前版1.20），防止因子偏好被板块倾斜放大。
        轮动速度保护: 当领涨行业切换过快(rotation_speed>0.55)，市场无主线，tilt→1.0。
        """
        # 轮动速度过快 → 行业倾斜失效，返回中性
        if self._rotation_speed > 0.55:
            return 1.0

        momentum_tilt = self._sector_tilt.get(industry, 1.0)
        signal_rank = self._signal_density_rank.get(industry, 0.5)
        signal_tilt = float(0.85 + signal_rank * 0.30)  # rank→[0.85,1.15]

        composite = momentum_tilt * 0.40 + signal_tilt * 0.60
        # 收紧上限: 1.20→1.15 防反馈环
        return float(np.clip(composite, 0.80, 1.15))

    def get_signal_density_score(self, industry: str) -> float:
        """纯信号密度分数 (-1~1)"""
        if not hasattr(self, '_signal_density_rank') or not self._signal_density_rank:
            return 0.0
        rp = self._signal_density_rank.get(industry, 0.5)
        return float(rp * 2 - 1)

    def get_strong_sectors(self, top_n: int = 5) -> List[str]:
        """领涨行业 (基于综合评分)"""
        if not self._sector_momentum:
            return []
        # 综合评分 = 动量排名 + 信号密度排名
        composite = {}
        for ind in self._sector_momentum:
            momentum_rank = self._sector_rank.get(ind, 0.5)
            signal_rank = self._signal_density_rank.get(ind, 0.5)
            composite[ind] = momentum_rank * 0.4 + signal_rank * 0.6
        sorted_items = sorted(composite.items(), key=lambda x: -x[1])
        return [s[0] for s in sorted_items[:top_n]]

    def get_weak_sectors(self, bottom_n: int = 5) -> List[str]:
        """领跌行业"""
        if not self._sector_momentum:
            return []
        composite = {}
        for ind in self._sector_momentum:
            momentum_rank = self._sector_rank.get(ind, 0.5)
            signal_rank = self._signal_density_rank.get(ind, 0.5)
            composite[ind] = momentum_rank * 0.4 + signal_rank * 0.6
        sorted_items = sorted(composite.items(), key=lambda x: x[1])
        return [s[0] for s in sorted_items[:bottom_n]]

    def summary(self) -> dict:
        """返回轮动摘要，供监控/日志使用"""
        return {
            'computed': self._computed,
            'strong': self.get_strong_sectors(5),
            'weak': self.get_weak_sectors(5),
            'sector_count': len(self._sector_momentum),
            'signal_density_top': sorted(
                self._signal_density.items(), key=lambda x: -x[1]
            )[:5] if self._signal_density else [],
        }
