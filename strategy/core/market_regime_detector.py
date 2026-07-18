# core/market_regime_detector.py
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional
from collections import deque


@dataclass
class MarketRegimeInfo:
    """
    市场状态信息 - 包含多种信号供策略使用
    """
    regime: int           # 离散状态: -1(熊), 0(震荡), 1(牛)
    confidence: float     # 置信度: 0-1
    momentum_score: float  # 动量分数: -1到1
    trend_score: float    # 趋势分数: -1到1 (EMA排列)
    volatility: float     # 波动率
    is_extreme: bool      # 是否处于极端状态
    # 风格因子 (新增)
    style_regime: str     # 风格状态: 'large_cap', 'small_cap', 'value', 'growth', 'balanced'
    style_score: float    # 风格分数: -1(大盘/价值) 到 1(小盘/成长)
    size_score: float     # 大小盘分数: -1(大盘) 到 1(小盘)
    style_confidence: float  # 风格置信度: 0-1
    # 熊市风险信号 (新增)
    bear_risk: bool       # 熊市风险（用于风险管理，120日维度）
    bear_risk_fast: bool  # 快速熊市风险（60日维度，急跌检测）
    severe_bear: bool = False  # 持续熊市: 60日bear_risk密度>70%
    # 状态切换率
    regime_volatility: float = 0.0  # 0(稳定) ~ 1(混沌)，用于动态调仓频率

    def to_dict(self) -> dict:
        return {
            'regime': self.regime,
            'confidence': self.confidence,
            'momentum_score': self.momentum_score,
            'trend_score': self.trend_score,
            'volatility': self.volatility,
            'is_extreme': self.is_extreme,
            'style_regime': self.style_regime,
            'style_score': self.style_score,
            'size_score': self.size_score,
            'style_confidence': self.style_confidence,
            'bear_risk': self.bear_risk,
            'bear_risk_fast': self.bear_risk_fast,
            'severe_bear': self.severe_bear,
            'regime_volatility': self.regime_volatility,
        }


class MarketRegimeDetector:
    """
    市场状态检测器 - 输出多种信号
    """

    def __init__(self, config=None):
        self.config = config or {}
        self.index_data = None
        self._init_params()

    def _init_params(self):
        """初始化参数"""
        # 熊市阈值 (ExpB1: 收窄中性区间, 5%→3%, 减少SHARPE因子误用)
        self.mom5_bear = -0.03    # 5日动量 < -3%
        self.mom_bear = -0.03     # 20日动量 < -3%
        self.mom_bear_sustained = -0.06  # 持续熊市：20日动量 < -6%

        # 牛市阈值 (A股急涨急跌, 60日动量3%太高→MOM因子从未触发)
        self.mom_bull = 0.025     # 20日动量 > 2.5% (3%→2.5%)
        self.mom60_bull = 0.0     # 60日动量 > 0% (1%→0%: 任何正60日动量配合即可)

        # 极端波动阈值
        self.vol_extreme_high = 0.30
        self.vol_extreme_low = 0.15

        # 均线参数
        self.ema_short = 20
        self.ema_medium = 60
        self.ema_long = 120

        # 风格检测参数 (新增)
        # 大小盘轮动: 比较沪深300与中证1000的相对强弱
        self.size_lookback = 20  # 短期动量比较
        self.size_lookback_long = 60  # 长期动量比较
        self.size_threshold = 0.03  # 3%阈值判断大小盘

        # === 新增：熊市风险检测参数 ===
        # 用于风险管理的熊市检测（不同于短期超跌检测）
        self.bear_risk_ma_period = 120  # 均线周期
        self.bear_risk_drawdown = 0.10  # 回撤阈值 10% (原15%过严, 2%触发率)
        self.bear_risk_momentum = -0.05  # 120日动量阈值 -5% (原-10%)
        # === 快速熊市检测：60日维度，捕捉急跌（120日动量滞后）===
        self.bear_risk_momentum_fast = -0.03  # 60日动量阈值 -3% (原-6%)
        self.bear_risk_drawdown_fast = 0.07   # 60日回撤阈值 7% (原10%)

    def generate(self, index_df: pd.DataFrame,
                 small_cap_df: pd.DataFrame = None,      # 中证1000 (000852)
                 growth_df: pd.DataFrame = None):         # 创业板指 (399006)
        """
        生成市场状态序列 (Fix#6: 支持多指数风格检测)

        Args:
            index_df: 上证指数 (sh000001) — 主指数, 用于牛熊判断
            small_cap_df: 中证1000 — 用于大小盘风格判断
            growth_df: 创业板指 — 用于成长/价值风格判断
        """
        self.index_data = index_df.copy()
        self._calculate_indicators()

        # 预处理辅助指数
        self._small_cap_data = None
        self._growth_data = None
        if small_cap_df is not None:
            self._small_cap_data = self._prepare_aux_index(small_cap_df, len(self.index_data))
        if growth_df is not None:
            self._growth_data = self._prepare_aux_index(growth_df, len(self.index_data))

        # 生成完整信号
        regime_info_list = []
        for i in range(len(self.index_data)):
            info = self._detect_detailed(i)
            regime_info_list.append(info)

        # 展开为列
        self.index_data['regime'] = [r.regime for r in regime_info_list]
        self.index_data['confidence'] = [r.confidence for r in regime_info_list]
        self.index_data['momentum_score'] = [r.momentum_score for r in regime_info_list]
        self.index_data['trend_score'] = [r.trend_score for r in regime_info_list]
        self.index_data['volatility'] = [r.volatility for r in regime_info_list]
        self.index_data['is_extreme'] = [r.is_extreme for r in regime_info_list]
        self.index_data['style_regime'] = [r.style_regime for r in regime_info_list]
        self.index_data['style_score'] = [r.style_score for r in regime_info_list]
        self.index_data['size_score'] = [r.size_score for r in regime_info_list]
        self.index_data['style_confidence'] = [r.style_confidence for r in regime_info_list]
        # 熊市风险信号
        self.index_data['bear_risk'] = [r.bear_risk for r in regime_info_list]
        self.index_data['bear_risk_fast'] = [r.bear_risk_fast for r in regime_info_list]
        # 持续熊市: 60日滚动bear_risk密度>70% (2022:98%d, 2025:0d, 精准区分)
        br_series = self.index_data['bear_risk'].astype(float)
        br_density = br_series.rolling(60, min_periods=30).mean()
        self.index_data['severe_bear'] = br_density > 0.70
        # 状态切换率（后处理：基于已生成的regime序列计算20日切换频率）
        regime_vals = self.index_data['regime'].values
        regime_vol_list = []
        for i in range(len(regime_vals)):
            if i < 20:
                regime_vol_list.append(0.0)
            else:
                recent = regime_vals[i-20:i+1]
                changes = sum(1 for j in range(1, len(recent)) if recent[j] != recent[j-1])
                regime_vol_list.append(min(changes / 20 * 5, 1.0))
        self.index_data['regime_volatility'] = regime_vol_list

        # 市场量能环境: 成交额 vs 20日均量
        if 'volume' in self.index_data.columns:
            vol = self.index_data['volume']
            vol_ma20 = vol.rolling(20, min_periods=5).mean()
            vol_ratio = vol / (vol_ma20 + 1)
            self.index_data['index_volume_ratio'] = vol_ratio.values
        else:
            self.index_data['index_volume_ratio'] = 1.0

        return self.index_data

    def _calculate_indicators(self):
        """计算技术指标"""
        close = self.index_data['close']
        high = self.index_data['high']
        low = self.index_data['low']

        # EMA
        self.ema20 = close.ewm(span=self.ema_short).mean()
        self.ema60 = close.ewm(span=self.ema_medium).mean()
        self.ema120 = close.ewm(span=self.ema_long).mean()

        # 动量
        self.momentum = close / close.shift(20) - 1
        self.momentum_60 = close / close.shift(60) - 1
        self.momentum_10 = close / close.shift(10) - 1
        self.momentum_5 = close / close.shift(5) - 1
        self.momentum_120 = close / close.shift(120) - 1  # 新增120日动量

        # 波动率
        self.returns = close.pct_change()
        self.volatility = self.returns.rolling(20).std() * np.sqrt(252)

        # 风格指标 (新增)
        # 大小盘相对强弱: 使用短期动量差异
        self.size_momentum = self.momentum_10  # 个股相对于指数的动量
        self.size_momentum_long = self.momentum  # 长期动量

        # === 新增：熊市风险指标 ===
        # 计算回撤 (120日)
        self.rolling_max = close.rolling(window=120, min_periods=1).max()
        self.drawdown = (close - self.rolling_max) / self.rolling_max
        # 快速回撤 (60日) - 用于急跌检测
        self.rolling_max_60 = close.rolling(window=60, min_periods=1).max()
        self.drawdown_60 = (close - self.rolling_max_60) / self.rolling_max_60

    def _detect_detailed(self, i: int) -> MarketRegimeInfo:
        """
        详细检测 - 输出多种信号
        """
        if i < 60:
            return MarketRegimeInfo(
                regime=0, confidence=0.0, momentum_score=0.0,
                trend_score=0.0, volatility=0.0, is_extreme=False,
                style_regime='balanced', style_score=0.0, size_score=0.0, style_confidence=0.0,
                bear_risk=False, bear_risk_fast=False, severe_bear=False
            )

        # 获取各指标
        mom_5 = self.momentum_5.iloc[i]
        mom = self.momentum.iloc[i]
        mom_60 = self.momentum_60.iloc[i]
        vol = self.volatility.iloc[i]

        # EMA趋势分数
        ema20 = self.ema20.iloc[i]
        ema60 = self.ema60.iloc[i]
        ema120 = self.ema120.iloc[i]
        ema20_above_60 = ema20 > ema60
        ema60_above_120 = ema60 > ema120

        # === 动量分数 (-1 到 1) ===
        momentum_score = np.clip(mom / 0.15, -1.0, 1.0)

        # === 趋势分数 (-1 到 1) ===
        if ema20_above_60 and ema60_above_120:
            trend_score = 1.0
        elif not ema20_above_60 and not ema60_above_120:
            trend_score = -1.0
        elif ema20_above_60:
            trend_score = 0.5
        else:
            trend_score = -0.5

        # === 离散状态判断 ===
        regime = 0
        confidence = 0.0

        # 均线空头排列检测（熊市辅助判断）
        ema_bearish = not ema20_above_60 and not ema60_above_120

        # === 多周期动量确认（5日+20日+60日三周期对齐才判定）===
        # 熊市判断：要求至少2/3周期确认，且60日动量为负
        mom_signals_bear = sum([
            mom_5 < self.mom5_bear,
            mom < self.mom_bear,
            mom_60 < -0.02
        ])
        if mom_signals_bear >= 2 and mom_60 < 0:
            regime = -1
            confidence = 0.7 + 0.1 * mom_signals_bear
        elif mom < self.mom_bear_sustained and ema_bearish and mom_60 < 0:
            # 持续下跌 + 均线空头排列 = 熊市
            regime = -1
            confidence = 0.8

        # 牛市判断：多路径捕获（A股急涨特性，60日动量常滞后）
        elif mom > self.mom_bull and mom_60 > self.mom60_bull:
            regime = 1
            confidence = 1.0
        elif mom > self.mom_bull * 0.5 and ema20_above_60 and ema60_above_120:
            # 动量较好 + 均线多头排列 = 牛市
            regime = 1
            confidence = 0.7
        elif mom > self.mom_bull * 0.67 and ema20_above_60:
            # 短期趋势已确立, 无需等60>120 (捕捉V形反弹)
            regime = 1
            confidence = 0.55
        elif mom_5 > self.mom_bull * 1.5 and mom > self.mom_bull * 0.5:
            # 短期爆发: 5日动量强 + 20日确认 (捕捉急涨启动)
            regime = 1
            confidence = 0.45

        # === 极端状态判断 ===
        is_extreme = vol > self.vol_extreme_high or vol < self.vol_extreme_low

        # === 熊市风险检测（用于风险管理）===
        drawdown = self._safe_get(self.drawdown, i, 0)
        mom_120 = self._safe_get(self.momentum_120, i, 0)
        bear_risk = (drawdown < -self.bear_risk_drawdown and
                     mom_120 < self.bear_risk_momentum and
                     ema_bearish)
        # 快速熊市检测: 60日维度, 捕捉急跌（120日动量滞后，急跌中数周才触发）
        drawdown_60 = self._safe_get(self.drawdown_60, i, 0)
        mom_60_val = self._safe_get(self.momentum_60, i, 0)
        bear_risk_fast = (drawdown_60 < -self.bear_risk_drawdown_fast and
                          mom_60_val < self.bear_risk_momentum_fast)

        # === 中性→熊市: A股无真正中性, 中性信号WR=45.2%系统亏损 ===
        # 回测数据: REV因子(熊市) WR=61.7% Avg=+3.42% >> SHARPE(中性) WR=45.2%
        # 不确定时默认判熊, 利用REV因子的均值回归优势
        # 但偏多信号不再强制判熊 → 让MOM因子(WR=56%)有机会触发
        if regime == 0 and confidence == 0.0:
            bull_signals = sum([mom_5 > 0.01, mom > 0.01, mom_60 > 0.02, ema20_above_60])
            bear_signals = sum([mom_5 < -0.01, mom < -0.01, mom_60 < -0.02, not ema20_above_60])
            total_signals = max(bull_signals + bear_signals, 1)
            signal_agreement = abs(bull_signals - bear_signals) / total_signals
            if signal_agreement < 0.3:
                confidence = 0.5  # 完全分歧→真震荡, 保持中性
            elif bear_signals > bull_signals:
                # 偏空信号→判为熊市(REV因子, WR=61.7%)
                regime = -1
                confidence = 0.45
            elif bull_signals >= 3:
                # 偏多信号且多头信号≥3→弱牛市(MOM因子, WR=56%)
                regime = 1
                confidence = 0.35
            else:
                # 偏多但不够强→中性(不再强制判熊), 让结构信号自行筛选
                confidence = 0.30

        style_regime, style_score, size_score, style_confidence = self._detect_style_regime(i)

        return MarketRegimeInfo(
            regime=regime,
            confidence=confidence,
            momentum_score=momentum_score,
            trend_score=trend_score,
            volatility=vol,
            is_extreme=is_extreme,
            style_regime=style_regime,
            style_score=style_score,
            size_score=size_score,
            style_confidence=style_confidence,
            bear_risk=bear_risk,
            bear_risk_fast=bear_risk_fast,
        )

    def _prepare_aux_index(self, aux_df: pd.DataFrame, target_len: int):
        """对齐辅助指数到主指数的时间轴

        返回与主指数同长度的close/returns数组，日期不匹配处填NaN。
        """
        if aux_df is None or 'close' not in aux_df.columns:
            return None
        aux_dates = pd.to_datetime(aux_df['datetime'].values)
        main_dates = pd.to_datetime(self.index_data['datetime'].values)
        close_arr = np.full(target_len, np.nan)
        ret_arr = np.full(target_len, np.nan)
        date_to_idx = {str(d.date()): i for i, d in enumerate(aux_dates)}
        for i, d in enumerate(main_dates):
            key = str(d.date())
            if key in date_to_idx:
                j = date_to_idx[key]
                close_arr[i] = aux_df['close'].values[j]
                if j > 0:
                    prev = aux_df['close'].values[j - 1]
                    if prev > 0:
                        ret_arr[i] = (close_arr[i] - prev) / prev
        return {'close': close_arr, 'ret': ret_arr}

    def _detect_style_regime(self, i: int):
        """
        检测风格状态 (Fix#6: 支持多指数真实风格检测)

        当接入中证1000和创业板指数据时:
        - size_score: 中证1000 vs 上证指数 20日相对强弱 → 真实大小盘判断
        - style_score: 创业板指 vs 上证指数 20日相对强弱 → 真实成长/价值判断

        无辅助指数时回退到单指数动量代理。

        返回: (style_regime, style_score, size_score, confidence)
        """
        if i < 60:
            return 'balanced', 0.0, 0.0, 0.0

        # ── Fix#6: 使用辅助指数做真实风格检测 ──
        if self._small_cap_data is not None and not np.isnan(self._small_cap_data['ret'][i]):
            # 大小盘: 中证1000相对上证指数的超额收益
            sc_ret_20 = np.nanmean(self._small_cap_data['ret'][max(0,i-20):i+1])
            main_ret_20 = np.nanmean(self.returns[max(0,i-20):i+1])
            size_score = np.clip((sc_ret_20 - main_ret_20) / 0.02, -1.0, 1.0)
        else:
            # 回退: 单指数动量代理
            size_mom = self._safe_get(self.size_momentum, i, 0)
            size_score = np.clip(size_mom / self.size_threshold, -1.0, 1.0)

        if self._growth_data is not None and not np.isnan(self._growth_data['ret'][i]):
            # 成长/价值: 创业板指相对上证指数的超额收益
            gr_ret_20 = np.nanmean(self._growth_data['ret'][max(0,i-20):i+1])
            main_ret_20_gr = np.nanmean(self.returns[max(0,i-20):i+1])
            style_score = np.clip((gr_ret_20 - main_ret_20_gr) / 0.02, -1.0, 1.0)
        else:
            # 回退: 长期动量趋势
            size_mom_long = self._safe_get(self.size_momentum_long, i, 0)
            if size_mom_long > self.size_threshold:
                style_score = np.clip(size_mom_long / 0.1, 0.0, 1.0)
            elif size_mom_long < -self.size_threshold:
                style_score = np.clip(size_mom_long / 0.1, -1.0, 0.0)
            else:
                style_score = 0.0

        confidence = min(1.0, abs(size_score) * 2)

        # 风格状态判定 — 组合size+style，保留两个维度
        size_part = 'balanced'
        if size_score > 0.3:
            size_part = 'small_cap'
        elif size_score < -0.3:
            size_part = 'large_cap'

        style_part = ''
        if style_score > 0.3:
            style_part = '_growth'
        elif style_score < -0.3:
            style_part = '_value'

        style_regime = size_part + style_part

        return style_regime, style_score, size_score, confidence

    def _safe_get(self, series, idx, default=0.0):
        """安全获取序列值"""
        try:
            if series is None:
                return default
            val = series.iloc[idx]
            if pd.isna(val):
                return default
            return val
        except (IndexError, KeyError, AttributeError):
            return default

    # 兼容旧接口
    def _detect_single(self, i: int) -> int:
        """简化版判断 - 保持原有逻辑"""
        return self._detect_detailed(i).regime

    def get_info(self, date) -> Optional[MarketRegimeInfo]:
        """获取指定日期的市场状态信息"""
        if self.index_data is None:
            return None

        row = self.index_data[self.index_data['datetime'].dt.date == date]
        if row.empty:
            return None

        idx = row.index[0]
        return MarketRegimeInfo(
            regime=int(self.index_data.loc[idx, 'regime']),
            confidence=float(self.index_data.loc[idx, 'confidence']),
            momentum_score=float(self.index_data.loc[idx, 'momentum_score']),
            trend_score=float(self.index_data.loc[idx, 'trend_score']),
            volatility=float(self.index_data.loc[idx, 'volatility']),
            is_extreme=bool(self.index_data.loc[idx, 'is_extreme']),
            style_regime=str(self.index_data.loc[idx, 'style_regime']),
            style_score=float(self.index_data.loc[idx, 'style_score']),
            size_score=float(self.index_data.loc[idx, 'size_score']),
            style_confidence=float(self.index_data.loc[idx, 'style_confidence']),
            bear_risk=bool(self.index_data.loc[idx, 'bear_risk']),
            bear_risk_fast=bool(self.index_data.loc[idx, 'bear_risk_fast']),
            severe_bear=bool(self.index_data.loc[idx, 'severe_bear']) if 'severe_bear' in self.index_data.columns else False,
        )

    def get_regime(self, date) -> int:
        """获取指定日期的市场状态"""
        if self.index_data is None:
            return 0

        row = self.index_data[self.index_data['datetime'].dt.date == date]
        if row.empty:
            return 0
        return int(row['regime'].values[0])

    def get_regime_stats(self) -> dict:
        """获取各年份市场状态统计"""
        if self.index_data is None:
            return {}

        self.index_data['year'] = self.index_data['datetime'].dt.year
        regime_by_year = self.index_data.groupby('year')['regime'].value_counts().unstack(fill_value=0)

        stats = {}
        for year in regime_by_year.index:
            total = regime_by_year.loc[year].sum()
            bull = regime_by_year.loc[year].get(1, 0)
            bear = regime_by_year.loc[year].get(-1, 0)
            stats[year] = {
                'bull_days': bull,
                'bear_days': bear,
                'sideways_days': total - bull - bear,
                'bull_ratio': bull / total if total > 0 else 0,
            }
        return stats


# 便捷函数
def create_market_regime(index_df: pd.DataFrame, config=None) -> pd.DataFrame:
    """创建市场状态序列"""
    detector = MarketRegimeDetector(config)
    return detector.generate(index_df)
