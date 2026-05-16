#!/usr/bin/env python3
"""
量化系统每日复盘 & 自我进化引擎

设计理念（融合盘后复盘方法论）:
  复盘 ≠ 单纯看报告。复盘 = 体检诊断 → 发现偏差 → 驱动系统参数自调整。

复盘6步法（量化落地版）:
  Step 1: 大盘环境深度体检 — 指数状态、情绪周期、量价关系、风格偏向
  Step 2: 板块与主线逻辑拆解 — 行业趋势浓度、梯队完整度、资金流向
  Step 3: 涨停板与情绪监控 — 涨跌停统计、连板高度、赚钱效应
  Step 4: 持仓股与自选股排雷 — 关键支撑/均线破位检查、逻辑验证
  Step 5: 缠论结构全面扫描 — 走势类型、买卖点、背驰、多级别确认
  Step 6: 生成作战计划 + 系统自进化建议

使用方式:
  cd strategy && python analysis/chan_review.py                      # 复盘最新交易日
  cd strategy && python analysis/chan_review.py --date 2026-05-15    # 复盘指定日期
  cd strategy && python analysis/chan_review.py --top 20 --evolve    # 复盘+自动进化
"""

import os
import sys
import argparse
import json
import warnings
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STRATEGY_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(STRATEGY_DIR)
sys.path.insert(0, STRATEGY_DIR)

from core.chan_theory import (
    compute_enhanced_chan_output,
    analyze_bottom_fractal,
    detect_stroke_mmd,
    check_bi_trend_depletion,
    get_structure_stop_price,
)
from core.market_regime_detector import MarketRegimeDetector
from core.industry_mapping import INDUSTRY_KEYWORDS, get_industry_category
from core.chan_theory import _calc_ema

# ==================== 配置 ====================

DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'stock_data', 'backtrader_data')
INDEX_FILE = os.path.join(DATA_DIR, 'sh000001_qfq.csv')
CONFIG_PATH = os.path.join(STRATEGY_DIR, 'config', 'factor_config.yaml')
REVIEW_OUTPUT_DIR = os.path.join(STRATEGY_DIR, 'daily_review')
EVOLUTION_LOG = os.path.join(REVIEW_OUTPUT_DIR, 'evolution_history.json')

DEFAULT_MIN_VOLUME = 20_000_000   # 最小成交额2000万
DEFAULT_MIN_TURNOVER = 15_000_000  # 备用：成交额门槛

# 关键均线（排雷用）
KEY_MA_PERIODS = [5, 10, 20, 60]

# 情绪周期阈值
SENTIMENT_THRESHOLDS = {
    'ice': {'limit_up_max': 30, 'adv_pct_max': 0.25},        # 冰点: 涨停<30, 上涨占比<25%
    'repair': {'limit_up_range': (30, 60), 'adv_pct_range': (0.25, 0.45)},  # 修复
    'climax': {'limit_up_min': 60, 'adv_pct_min': 0.45},     # 高潮: 涨停>60, 上涨>45%
    'divergence': {'adv_pct_range': (0.30, 0.55)},            # 分歧: 涨跌互现
    'retreat': {'limit_up_max': 40, 'adv_pct_max': 0.35},     # 退潮: 涨停减少
}


# ==================== 数据结构 ====================

@dataclass
class MarketVitalSigns:
    """大盘体检报告"""
    date: str = ''

    # 指数状态
    index_name: str = '上证指数'
    index_close: float = 0.0
    index_change_pct: float = 0.0
    index_volume: str = ''  # 放量/缩量/平量
    volume_ratio: float = 1.0

    # 市场状态
    regime: str = ''          # 牛市/震荡/熊市
    regime_confidence: float = 0.0
    sentiment_cycle: str = ''  # 冰点/修复/高潮/分歧/退潮
    volatility_pct: float = 0.0
    momentum_score: float = 0.0

    # 赚钱效应
    advancing: int = 0
    declining: int = 0
    up_pct: float = 0.0
    limit_up_count: int = 0
    limit_down_count: int = 0
    limit_up_ratio: float = 0.0  # 涨停/跌停比

    # 风格偏向
    style_regime: str = ''
    bear_risk: bool = False

    # 指数共振
    index_divergence: bool = False  # 上证/深成/创业板是否分化

    # 综合判定
    summary: str = ''


@dataclass
class IndustryMainLine:
    """行业主线分析"""
    industry: str = ''
    stock_count: int = 0

    # 趋势分布
    up_trend_pct: float = 0.0
    down_trend_pct: float = 0.0
    consolidation_pct: float = 0.0

    # 缠论买点浓度
    b1_count: int = 0
    b2_count: int = 0
    b3_count: int = 0
    total_buy_signals: int = 0
    total_sell_signals: int = 0

    # 资金流向
    avg_volume_ratio: float = 1.0
    net_flow_score: float = 0.0

    # 梯队完整度 (量化近似)
    has_leader: bool = False      # 有龙头（涨>5%+放量>2倍均量）
    has_mid_cap: bool = False     # 有中军（涨>2%+成交额>行业平均）
    has_follower: bool = False    # 有跟风（行业内涨跌比>50%）
    formation_score: float = 0.0  # 梯队完整度 0-1

    # 综合评分
    hot_rank: int = 0
    avg_composite_score: float = 0.0
    summary: str = ''


@dataclass
class StockDiagnosis:
    """个股诊断（缠论 + 排雷）"""
    symbol: str
    name: str = ''
    industry: str = ''

    # 基础数据
    close: float = 0.0
    change_pct: float = 0.0
    volume_ratio: float = 1.0

    # 缠论结构
    trend_type: int = 0          # 2=上涨趋势, 1=盘整, -2=下跌趋势
    trend_strength: float = 0.0
    pivot_position: int = 0      # -1=中枢下, 0=中枢内, 1=中枢上
    pivot_zg: float = np.nan
    pivot_zd: float = np.nan
    stroke_direction: int = 0    # 当前笔方向
    segment_direction: int = 0   # 当前线段方向

    # 买卖点
    buy_point: int = 0
    sell_point: int = 0
    buy_confidence: float = 0.0
    sell_confidence: float = 0.0
    second_buy: bool = False
    second_buy_conf: float = 0.0
    confirmed_buy: bool = False
    confirmed_sell: bool = False
    signal_level: int = 0
    buy_strength: float = 0.0
    sell_strength: float = 0.0

    # 背驰
    bottom_divergence: bool = False
    top_divergence: bool = False

    # 排雷检查
    ma_breakdowns: List[str] = field(default_factory=list)   # 均线破位列表
    pivot_breakdown: bool = False   # 中枢下方运行
    volume_anomaly: str = ''        # 倍量/堆量/缩量
    bi_trend_depletion: bool = False

    # 均线排列
    alignment_score: float = 0.0

    # 综合评分 & 等级
    composite_score: float = 0.0
    opportunity_rank: str = ''     # A/B/C/D
    risk_level: str = ''           # 高危/注意/安全

    # 止损参考
    stop_loss_price: float = 0.0


@dataclass
class LimitUpDiagnosis:
    """涨停板缠论归因分析"""
    symbol: str
    name: str = ''
    industry: str = ''
    close: float = 0.0
    change_pct: float = 0.0
    volume_ratio: float = 1.0

    # 涨停前5日的缠论结构快照
    pre_trend_type: int = 0         # 涨停前的走势类型
    pre_pivot_position: int = 0     # 涨停前相对中枢位置
    pre_stroke_direction: int = 0   # 涨停前笔方向
    pre_buy_point: int = 0          # 涨停前是否有买点
    pre_buy_confidence: float = 0.0
    pre_bottom_divergence: bool = False  # 涨停前底背驰
    pre_alignment_score: float = 0.0
    pre_consolidation_days: int = 0      # 涨停前盘整天数（近似）

    # 涨停日的信号
    cur_buy_point: int = 0
    cur_buy_confidence: float = 0.0
    cur_bottom_divergence: bool = False

    # 结构特征（涨停日附近的）
    pivot_breakout: bool = False     # 是否突破中枢上沿
    pivot_zg: float = np.nan
    pivot_zd: float = np.nan
    above_pivot_pct: float = 0.0    # 收盘价超过ZG的百分比
    below_pivot_pct: float = 0.0    # 开盘前在ZD下方的百分比

    # 量能特征
    vol_spike: float = 1.0          # 涨停日相对5日均量
    bottom_fx_quality: float = 0.0  # 最近底分型质量

    # 涨停归因分类
    limit_up_type: str = ''         # 中枢突破板/底背驰反转板/二买确认板/趋势加速板/笔耗尽反转板/盘整突破板
    type_confidence: float = 0.0    # 归因置信度
    chan_explanation: str = ''      # 缠论解释


@dataclass
class LimitUpSummary:
    """涨停板共性总结"""
    date: str
    total_limit_up: int
    limit_up_stocks: List[LimitUpDiagnosis] = field(default_factory=list)

    # 按类型分布
    type_distribution: Dict[str, int] = field(default_factory=dict)
    type_avg_confidence: Dict[str, float] = field(default_factory=dict)

    # 共性特征
    common_traits: List[str] = field(default_factory=list)      # 共性发现
    industry_concentration: Dict[str, int] = field(default_factory=dict)  # 涨停行业集中度
    dominant_pattern: str = ''      # 今日主导涨停模式


@dataclass
class EvolutionPatch:
    """系统自进化补丁"""
    section: str          # 配置section
    key: str              # 配置key
    current_value: float
    suggested_value: float
    reason: str
    confidence: float     # 建议置信度 0-1
    urgency: str          # urgent/high/medium/low


@dataclass
class DailyReviewReport:
    """完整的每日复盘报告"""
    date: str
    market: MarketVitalSigns
    main_lines: List[IndustryMainLine]
    opportunities: Dict[str, List[StockDiagnosis]]
    risk_alerts: Dict[str, List[StockDiagnosis]]
    portfolio_suggestions: List[StockDiagnosis]
    limit_up_summary: Optional[LimitUpSummary] = None
    evolution_patches: List[EvolutionPatch] = field(default_factory=list)
    summary: Dict[str, any] = field(default_factory=dict)


# ==================== 复盘引擎 ====================

class QuantReviewEngine:
    """量化复盘引擎 — 体检诊断 → 机会发现 → 系统进化"""

    def __init__(self, data_dir: str = None, min_volume: float = None):
        self.data_dir = data_dir or DATA_DIR
        self.min_volume = min_volume or DEFAULT_MIN_VOLUME
        self.output_dir = REVIEW_OUTPUT_DIR
        os.makedirs(self.output_dir, exist_ok=True)

        self._price_cache = {}

    # ========== 数据层 ==========

    def _find_latest_date(self) -> str:
        import glob
        files = glob.glob(os.path.join(self.data_dir, '*_qfq.csv'))
        if not files:
            return datetime.now().strftime('%Y-%m-%d')
        for f in files:
            try:
                df = pd.read_csv(f)
                if 'date' in df.columns and len(df) > 0:
                    return str(df['date'].max())
            except Exception:
                continue
        return datetime.now().strftime('%Y-%m-%d')

    def _load_index_data(self, date: str, lookback: int = 250) -> pd.DataFrame:
        if not os.path.exists(INDEX_FILE):
            return pd.DataFrame()
        df = pd.read_csv(INDEX_FILE)
        date_col = 'datetime' if 'datetime' in df.columns else 'date'
        if date_col not in df.columns:
            return pd.DataFrame()
        if date_col != 'date':
            df = df.rename(columns={date_col: 'date'})
        df['date'] = df['date'].astype(str)
        return df[df['date'] <= date].tail(lookback).copy()

    def _load_stock_data(self, symbol: str, date: str, lookback: int = 250) -> Optional[pd.DataFrame]:
        cache_key = f'{symbol}_{date}_{lookback}'
        if cache_key in self._price_cache:
            return self._price_cache[cache_key]

        fp = os.path.join(self.data_dir, f'{symbol}_qfq.csv')
        if not os.path.exists(fp):
            return None
        try:
            df = pd.read_csv(fp)
            # 统一日期列名
            date_col = 'datetime' if 'datetime' in df.columns else 'date'
            if date_col != 'date':
                df = df.rename(columns={date_col: 'date'})
            if 'date' not in df.columns or len(df) < 60:
                return None
            df['date'] = df['date'].astype(str)
            df = df[df['date'] <= date].tail(lookback).copy()
            if len(df) < 60:
                return None

            # 标准化列名
            col_map = {}
            for std_col in ['open', 'high', 'low', 'close', 'volume']:
                for c in df.columns:
                    if c.lower() == std_col:
                        col_map[std_col] = c
                        break
            df = df.rename(columns={v: k for k, v in col_map.items()})

            # 成交额列（如果有）
            for c in ['turnover', 'amount', '成交额']:
                if c in df.columns:
                    col_map['turnover'] = c
                    break

            self._price_cache[cache_key] = df
            return df
        except Exception:
            return None

    def _list_available_stocks(self) -> List[str]:
        import glob
        files = glob.glob(os.path.join(self.data_dir, '*_qfq.csv'))
        skip_prefixes = ('sh000', 'sh399', 'sz399', 'sh688', 'bj')
        symbols = []
        for f in files:
            bn = os.path.basename(f)
            sym = bn.replace('_qfq.csv', '')
            # 跳过指数
            if any(sym.startswith(p) for p in skip_prefixes):
                continue
            # 只保留纯数字代码（6位）+ 可选sh/sz前缀
            code = sym[2:] if sym[:2] in ('sh', 'sz') else sym
            if code.isdigit() and len(code) == 6:
                symbols.append(sym)
        return sorted(symbols)

    def _get_stock_name(self, symbol: str) -> str:
        metadata_file = os.path.join(
            PROJECT_ROOT, 'data', 'stock_data', 'stock_metadata', 'stock_list.json'
        )
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, 'r') as f:
                    stocks = json.load(f)
                for s in stocks:
                    if str(s.get('symbol', '')) == symbol:
                        return s.get('name', '')
            except Exception:
                pass
        return ''

    # ========== Step 1: 大盘环境深度体检 ==========

    def examine_market_vitals(self, date: str, stock_list: List[str]) -> MarketVitalSigns:
        """大盘环境深度体检"""
        v = MarketVitalSigns(date=date)

        index_df = self._load_index_data(date)
        if len(index_df) < 60:
            v.summary = '数据不足，无法完成大盘体检'
            return v

        close = index_df['close'].values.astype(np.float64)
        v.index_close = float(close[-1])
        v.index_change_pct = float((close[-1] / close[-2] - 1) * 100) if len(close) >= 2 else 0.0

        # 成交量判断
        vol = index_df['volume'].values.astype(np.float64) if 'volume' in index_df.columns else np.ones(len(close))
        vol_ma20 = np.mean(vol[-21:-1]) if len(vol) >= 21 else vol[-1]
        v.volume_ratio = float(vol[-1] / vol_ma20) if vol_ma20 > 0 else 1.0
        if v.volume_ratio >= 1.3:
            v.index_volume = '放量'
        elif v.volume_ratio <= 0.7:
            v.index_volume = '缩量'
        else:
            v.index_volume = '平量'

        # 市场状态检测
        detector = MarketRegimeDetector()
        regime_df = detector.generate(index_df)
        last_row = regime_df.iloc[-1]
        regime_labels = {1: '牛市', 0: '震荡', -1: '熊市'}
        v.regime = regime_labels.get(int(last_row.get('regime', 0)), '未知')
        v.regime_confidence = float(last_row.get('confidence', 0))
        v.momentum_score = float(last_row.get('momentum_score', 0))
        v.volatility_pct = float(last_row.get('volatility', 0))
        v.style_regime = str(last_row.get('style_regime', ''))
        v.bear_risk = bool(last_row.get('bear_risk', False))

        # 赚钱效应统计（抽样300只）
        adv, dec, limit_up, limit_down = 0, 0, 0, 0
        for sym in stock_list[:300]:
            df = self._load_stock_data(sym, date, lookback=5)
            if df is None or len(df) < 2:
                continue
            chg = (df['close'].iloc[-1] / df['close'].iloc[-2] - 1)
            if chg > 0:
                adv += 1
            elif chg < 0:
                dec += 1
            if chg >= 0.098:
                limit_up += 1
            elif chg <= -0.098:
                limit_down += 1

        v.advancing = adv
        v.declining = dec
        v.up_pct = adv / max(adv + dec, 1) * 100
        v.limit_up_count = limit_up
        v.limit_down_count = limit_down
        v.limit_up_ratio = limit_up / max(limit_down, 1)

        # 情绪周期判断
        v.sentiment_cycle = self._classify_sentiment(limit_up, limit_down, v.up_pct / 100)

        # 指数共振（简化版：检查市场宽度）
        v.index_divergence = (v.up_pct < 45 and v.index_change_pct > 0.3) or \
                             (v.up_pct > 55 and v.index_change_pct < -0.3)

        # 综合判定
        parts = []
        parts.append(f"{v.regime}{'放量' if v.index_volume == '放量' else ''}")
        parts.append(f"情绪: {v.sentiment_cycle}")
        conditions = []
        if v.bear_risk:
            conditions.append("⚠熊市风险")
        if v.index_divergence:
            conditions.append("指数分化")
        if v.limit_up_ratio >= 3:
            conditions.append("赚钱效应强")
        elif v.limit_up_ratio <= 0.5:
            conditions.append("亏钱效应强")
        parts.append(' | '.join(conditions) if conditions else '正常')
        v.summary = '，'.join(parts)

        return v

    def _classify_sentiment(self, limit_up: int, limit_down: int, up_pct: float) -> str:
        """情绪周期分类: 冰点→修复→高潮→分歧→退潮"""
        if limit_up < 30 and up_pct < 0.25:
            return '冰点'
        if limit_up > 60 and up_pct > 0.50:
            return '高潮'
        if limit_down > limit_up * 0.6:
            return '退潮'
        if 0.30 <= up_pct <= 0.55 and abs(limit_up - limit_down) <= limit_up * 0.4:
            return '分歧'
        if 30 <= limit_up <= 60 and 0.25 <= up_pct <= 0.45:
            return '修复'
        return '分歧'

    # ========== Step 2: 板块与主线逻辑拆解 ==========

    def analyze_industry_main_lines(
        self, diagnoses: List[StockDiagnosis]
    ) -> List[IndustryMainLine]:
        """行业主线分析 — 包含梯队完整度"""
        ind_data = defaultdict(lambda: {
            'stocks': [], 'buys': 0, 'sells': 0,
            'b1': 0, 'b2': 0, 'b3': 0,
            'up_trend': 0, 'down_trend': 0, 'cons': 0,
            'vol_ratios': [], 'composites': [],
            'leaders': 0, 'mid_caps': 0, 'followers': 0,
        })

        for d in diagnoses:
            ind = d.industry or '其他'
            dd = ind_data[ind]
            dd['stocks'].append(d)
            if d.buy_point > 0:
                dd['buys'] += 1
            if d.sell_point > 0:
                dd['sells'] += 1
            if d.buy_point == 1:
                dd['b1'] += 1
            elif d.buy_point == 2:
                dd['b2'] += 1
            elif d.buy_point == 3:
                dd['b3'] += 1

            if d.trend_type == 2:
                dd['up_trend'] += 1
            elif d.trend_type == -2:
                dd['down_trend'] += 1
            else:
                dd['cons'] += 1

            dd['vol_ratios'].append(d.volume_ratio)
            dd['composites'].append(d.composite_score)

            # 梯队检测
            if d.change_pct > 5 and d.volume_ratio > 2:
                dd['leaders'] += 1
            if d.change_pct > 2 and d.volume_ratio > 1.5:
                dd['mid_caps'] += 1
            if d.change_pct > 0:
                dd['followers'] += 1

        results = []
        for ind, dd in ind_data.items():
            n = len(dd['stocks'])
            if n < 3:
                continue

            ml = IndustryMainLine(industry=ind, stock_count=n)
            ml.up_trend_pct = dd['up_trend'] / n * 100
            ml.down_trend_pct = dd['down_trend'] / n * 100
            ml.consolidation_pct = dd['cons'] / n * 100
            ml.b1_count = dd['b1']
            ml.b2_count = dd['b2']
            ml.b3_count = dd['b3']
            ml.total_buy_signals = dd['buys']
            ml.total_sell_signals = dd['sells']
            ml.avg_volume_ratio = np.mean(dd['vol_ratios']) if dd['vol_ratios'] else 1.0
            ml.avg_composite_score = np.mean(dd['composites']) if dd['composites'] else 0.0

            # 梯队完整度
            ml.has_leader = dd['leaders'] > 0
            ml.has_mid_cap = dd['mid_caps'] > 1
            ml.has_follower = (dd['followers'] / n) > 0.5 if n > 0 else False
            ml.formation_score = (
                (0.4 if ml.has_leader else 0) +
                (0.3 if ml.has_mid_cap else 0) +
                (0.3 if ml.has_follower else 0)
            )
            ml.net_flow_score = dd['buys'] - dd['sells']

            # 综合判定
            if ml.formation_score >= 0.7 and ml.total_buy_signals >= 3:
                ml.summary = '★ 强主线，梯队完整，可积极操作'
            elif ml.formation_score >= 0.4 and ml.total_buy_signals >= 1:
                ml.summary = '○ 次主线，有结构但梯队不全，谨慎参与'
            elif ml.total_sell_signals > ml.total_buy_signals:
                ml.summary = '✗ 资金出逃，回避'
            else:
                ml.summary = '— 无明显方向，观望'

            results.append(ml)

        results.sort(key=lambda x: (x.formation_score + x.avg_composite_score / 3), reverse=True)
        for i, r in enumerate(results):
            r.hot_rank = i + 1
        return results

    # ========== Step 3+5: 涨停监控 & 缠论扫描（合并） ==========

    def diagnose_stock(
        self, symbol: str, date: str, index_df: pd.DataFrame = None
    ) -> Optional[StockDiagnosis]:
        """个股全面诊断 — 缠论结构 + 排雷检查"""
        df = self._load_stock_data(symbol, date)
        if df is None or len(df) < 60:
            return None

        # 流动性过滤 — 优先使用amount列(成交额)，其次 volume*close
        if 'amount' in df.columns:
            avg_turnover = df['amount'].tail(20).mean()
        elif 'volume' in df.columns and 'close' in df.columns:
            avg_turnover = (df['volume'].tail(20) * df['close'].tail(20)).mean()
        else:
            avg_turnover = float('inf')
        if avg_turnover < self.min_volume:
                return None

        n = len(df)
        close = df['close'].values.astype(np.float64)
        high = df['high'].values.astype(np.float64)
        low = df['low'].values.astype(np.float64)
        volume = df['volume'].values.astype(np.float64) if 'volume' in df.columns else np.ones(n)

        # === 缠论增强输出 ===
        ema12 = _calc_ema(close, 12)
        ema26 = _calc_ema(close, 26)
        macd_line = ema12 - ema26
        macd_hist = 2 * (macd_line - _calc_ema(macd_line, 9))
        ema20 = _calc_ema(close, 20)
        ema60 = _calc_ema(close, 60)
        ema120 = _calc_ema(close, 120)

        chan = compute_enhanced_chan_output(
            close, high, low, ema20, ema60, ema120, macd_hist, volume,
        )

        idx = -1
        d = StockDiagnosis(symbol=symbol)
        d.name = self._get_stock_name(symbol)
        try:
            d.industry = get_industry_category(d.name)
        except Exception:
            d.industry = '其他'

        d.close = float(close[idx])
        d.change_pct = float((close[idx] / close[idx - 1] - 1) * 100) if idx - 1 >= -n else 0.0
        d.volume_ratio = float(volume[idx] / np.mean(volume[-21:-1])) if len(volume) >= 21 and np.mean(volume[-21:-1]) > 0 else 1.0

        # 缠论结构
        d.trend_type = int(chan['trend_type'][idx])
        d.trend_strength = float(chan['trend_strength'][idx])
        d.pivot_position = int(chan['pivot_position'][idx])
        d.pivot_zg = float(chan['pivot_zg'][idx]) if not np.isnan(chan['pivot_zg'][idx]) else np.nan
        d.pivot_zd = float(chan['pivot_zd'][idx]) if not np.isnan(chan['pivot_zd'][idx]) else np.nan
        d.stroke_direction = int(chan['stroke_direction'][idx])
        d.segment_direction = int(chan['segment_direction'][idx])

        # 买卖点
        d.buy_point = int(chan['buy_point'][idx])
        d.sell_point = int(chan['sell_point'][idx])
        d.buy_confidence = float(chan['buy_confidence'][idx])
        d.sell_confidence = float(chan['sell_confidence'][idx])
        d.second_buy = bool(chan.get('second_buy_point', np.zeros(n))[idx])
        d.second_buy_conf = float(chan.get('second_buy_confidence', np.zeros(n))[idx])
        d.confirmed_buy = bool(chan['confirmed_buy'][idx])
        d.confirmed_sell = bool(chan['confirmed_sell'][idx])
        d.signal_level = int(chan['signal_level'][idx])
        d.buy_strength = float(chan['buy_strength'][idx])
        d.sell_strength = float(chan['sell_strength'][idx])

        # 背驰
        d.bottom_divergence = bool(chan.get('hidden_bottom_divergence', np.zeros(n))[idx])
        d.top_divergence = bool(chan.get('hidden_top_divergence', np.zeros(n))[idx])

        # 均线排列
        d.alignment_score = float(chan['alignment_score'][idx])

        # 笔趋势耗尽
        d.bi_trend_depletion = bool(chan.get('bi_td', np.zeros(n))[idx])

        # === 排雷检查 ===
        # 1. 均线破位
        d.ma_breakdowns = []
        for ma_period in KEY_MA_PERIODS:
            ma = _calc_ema(close, ma_period)
            if not np.isnan(ma[idx]) and close[idx] < ma[idx] * 0.98:
                d.ma_breakdowns.append(f'跌破MA{ma_period}')

        # 2. 中枢破位
        if d.pivot_position == -1 and d.trend_type <= 0:
            d.pivot_breakdown = True

        # 3. 量异常
        if d.volume_ratio >= 2.5:
            d.volume_anomaly = '倍量'
        elif d.volume_ratio >= 1.5:
            d.volume_anomaly = '放量'
        elif d.volume_ratio <= 0.4:
            d.volume_anomaly = '缩量'
        else:
            d.volume_anomaly = '正常'

        # === 综合评分 ===
        d.composite_score = self._compute_composite(d)
        d.opportunity_rank = self._rank_opportunity(d)
        d.risk_level = self._assess_risk(d)
        d.stop_loss_price = float(chan.get('structure_stop_price', np.full(n, np.nan))[idx])
        if np.isnan(d.stop_loss_price):
            d.stop_loss_price = d.close * 0.93

        return d

    def _compute_composite(self, d: StockDiagnosis) -> float:
        """综合评分 [-1, +1]"""
        score = 0.0
        # 趋势 (25%)
        if d.trend_type == 2:
            score += 0.25
        elif d.trend_type == -2:
            score -= 0.25

        # 买卖点 (35%)
        if d.confirmed_buy:
            score += 0.35 * d.buy_strength
        elif d.confirmed_sell:
            score -= 0.35 * d.sell_strength
        elif d.buy_point > 0:
            score += 0.20 * d.buy_confidence
        elif d.sell_point > 0:
            score -= 0.20 * d.sell_confidence

        # 二买 (10%)
        if d.second_buy:
            score += 0.10 * d.second_buy_conf

        # 中枢位置 (10%)
        if d.pivot_position == 1:
            score += 0.08
        elif d.pivot_position == -1 and d.trend_type >= 0:
            score += 0.05

        # 均线对齐 (5%)
        score += d.alignment_score * 0.05

        # 排雷惩罚 (10%)
        if d.ma_breakdowns:
            score -= len(d.ma_breakdowns) * 0.05
        if d.pivot_breakdown:
            score -= 0.05
        if d.top_divergence:
            score -= 0.10

        # 笔趋势耗尽 (5%)
        if d.bi_trend_depletion and d.stroke_direction == -1:
            score += 0.05

        return float(np.clip(score, -1.0, 1.0))

    def _rank_opportunity(self, d: StockDiagnosis) -> str:
        """机会等级 A/B/C/D"""
        if d.confirmed_buy and d.signal_level >= 3 and d.buy_confidence >= 0.6 and d.composite_score >= 0.5:
            return 'A'
        if d.second_buy and d.second_buy_conf >= 0.6 and d.composite_score >= 0.4:
            return 'A'
        if d.buy_point >= 2 and d.buy_confidence >= 0.5 and d.composite_score >= 0.3:
            return 'B'
        if d.buy_point == 1 and d.composite_score >= 0.1:
            return 'C'
        if d.composite_score <= -0.5:
            return 'D-'
        return 'D'

    def _assess_risk(self, d: StockDiagnosis) -> str:
        """风险评估"""
        if d.confirmed_sell or d.sell_point >= 1:
            return '高危'
        if len(d.ma_breakdowns) >= 2 or d.pivot_breakdown:
            return '高危'
        if d.top_divergence or (d.bi_trend_depletion and d.stroke_direction == 1):
            return '注意'
        if d.trend_type == -2 and d.pivot_position <= 0:
            return '注意'
        return '安全'

    # ========== 机会 & 风险分类 ==========

    def classify_opportunities(
        self, diagnoses: List[StockDiagnosis]
    ) -> Dict[str, List[StockDiagnosis]]:
        """分类排序机会"""
        ops = {
            'A_strong_buy': [], 'B2_confirmed': [], 'B1_reversal': [],
            'B3_acceleration': [], 'double_confirmed': [], 'bottom_divergence': [],
        }
        for d in diagnoses:
            if d.opportunity_rank == 'A':
                ops['A_strong_buy'].append(d)
            if d.second_buy and d.second_buy_conf >= 0.45:
                ops['B2_confirmed'].append(d)
            if d.buy_point == 1 and d.buy_confidence >= 0.4:
                ops['B1_reversal'].append(d)
            if d.buy_point == 3:
                ops['B3_acceleration'].append(d)
            if d.confirmed_buy and d.signal_level >= 3:
                ops['double_confirmed'].append(d)
            if d.bottom_divergence and d.buy_point > 0:
                ops['bottom_divergence'].append(d)

        ops['A_strong_buy'].sort(key=lambda x: x.composite_score, reverse=True)
        ops['B2_confirmed'].sort(key=lambda x: x.second_buy_conf, reverse=True)
        ops['B1_reversal'].sort(key=lambda x: x.buy_confidence, reverse=True)
        ops['B3_acceleration'].sort(key=lambda x: x.buy_confidence, reverse=True)
        ops['double_confirmed'].sort(key=lambda x: x.buy_strength, reverse=True)
        ops['bottom_divergence'].sort(key=lambda x: x.buy_confidence, reverse=True)
        return ops

    def classify_risks(
        self, diagnoses: List[StockDiagnosis]
    ) -> Dict[str, List[StockDiagnosis]]:
        """分类排序风险"""
        risks = {
            'S1_top_reversal': [], 'S3_breakdown': [],
            'confirmed_sell': [], 'top_divergence': [],
            'ma_breakdown': [], 'trend_depletion': [],
        }
        for d in diagnoses:
            if d.sell_point == 1:
                risks['S1_top_reversal'].append(d)
            if d.sell_point == 3:
                risks['S3_breakdown'].append(d)
            if d.confirmed_sell:
                risks['confirmed_sell'].append(d)
            if d.top_divergence:
                risks['top_divergence'].append(d)
            if len(d.ma_breakdowns) >= 2:
                risks['ma_breakdown'].append(d)
            if d.bi_trend_depletion and d.stroke_direction == 1:
                risks['trend_depletion'].append(d)

        for k in risks:
            risks[k].sort(key=lambda x: x.sell_confidence if hasattr(x, 'sell_confidence') else 0, reverse=True)
        return risks

    # ========== Step 3 增强: 涨停板缠论归因分析 ==========

    LIMIT_UP_THRESHOLD = 9.5   # 涨停阈值(%), A股主板10%, 科创/创业20%

    def analyze_limit_up_stocks(
        self, date: str, stock_list: List[str], index_df: pd.DataFrame
    ) -> LimitUpSummary:
        """涨停板缠论归因分析 — 用缠论解释每个涨停的成因，总结共性"""

        summary = LimitUpSummary(date=date, total_limit_up=0)
        limit_up_diagnoses = []

        for sym in stock_list:
            d = self._diagnose_single_limit_up(sym, date, index_df)
            if d is not None:
                limit_up_diagnoses.append(d)

        summary.total_limit_up = len(limit_up_diagnoses)
        summary.limit_up_stocks = limit_up_diagnoses

        if not limit_up_diagnoses:
            return summary

        # 类型分布
        type_dist = defaultdict(int)
        type_confs = defaultdict(list)
        for d in limit_up_diagnoses:
            t = d.limit_up_type or '未分类'
            type_dist[t] += 1
            type_confs[t].append(d.type_confidence)

        summary.type_distribution = dict(type_dist)
        summary.type_avg_confidence = {
            k: np.mean(v) for k, v in type_confs.items()
        }

        # 行业集中度
        ind_counts = defaultdict(int)
        for d in limit_up_diagnoses:
            ind_counts[d.industry] += 1
        summary.industry_concentration = dict(
            sorted(ind_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        )

        # 主导模式
        if type_dist:
            summary.dominant_pattern = max(type_dist, key=type_dist.get)

        # 共性发现
        summary.common_traits = self._extract_limit_up_commonalities(limit_up_diagnoses)

        return summary

    def _diagnose_single_limit_up(
        self, symbol: str, date: str, index_df: pd.DataFrame
    ) -> Optional[LimitUpDiagnosis]:
        """诊断单只涨停股 — 加载数据并做缠论归因"""

        df = self._load_stock_data(symbol, date)
        if df is None or len(df) < 80:
            return None

        n = len(df)
        close = df['close'].values.astype(np.float64)
        high = df['high'].values.astype(np.float64)
        low = df['low'].values.astype(np.float64)
        volume = df['volume'].values.astype(np.float64) if 'volume' in df.columns else np.ones(n)

        # 检查是否涨停 (涨跌幅 >= 9.5%)
        if n < 2:
            return None
        chg = (close[-1] / close[-2] - 1) * 100
        if chg < self.LIMIT_UP_THRESHOLD:
            return None

        # 流动性过滤
        if 'amount' in df.columns:
            avg_to = df['amount'].tail(20).mean()
        elif 'volume' in df.columns:
            avg_to = (df['volume'].tail(20) * df['close'].tail(20)).mean()
        else:
            avg_to = float('inf')
        if avg_to < self.min_volume:
            return None

        # === 计算缠论结构 ===
        ema12 = _calc_ema(close, 12)
        ema26 = _calc_ema(close, 26)
        macd_line = ema12 - ema26
        macd_hist = 2 * (macd_line - _calc_ema(macd_line, 9))
        ema20 = _calc_ema(close, 20)
        ema60 = _calc_ema(close, 60)
        ema120 = _calc_ema(close, 120)

        chan = compute_enhanced_chan_output(
            close, high, low, ema20, ema60, ema120, macd_hist, volume,
        )

        d = LimitUpDiagnosis(symbol=symbol)
        d.name = self._get_stock_name(symbol)
        try:
            d.industry = get_industry_category(d.name)
        except Exception:
            d.industry = '其他'
        d.close = float(close[-1])
        d.change_pct = float(chg)
        d.volume_ratio = float(volume[-1] / np.mean(volume[-6:-1])) if np.mean(volume[-6:-1]) > 0 else 1.0

        # === 涨停前结构 (5日前) ===
        pre_idx = max(0, n - 6)  # 涨停前1天 (涨停日是n-1)
        d.pre_trend_type = int(chan['trend_type'][pre_idx])
        d.pre_pivot_position = int(chan['pivot_position'][pre_idx])
        d.pre_stroke_direction = int(chan['stroke_direction'][pre_idx])
        d.pre_buy_point = int(chan['buy_point'][pre_idx])
        d.pre_buy_confidence = float(chan['buy_confidence'][pre_idx])
        d.pre_bottom_divergence = bool(chan.get('hidden_bottom_divergence', np.zeros(n))[pre_idx])
        d.pre_alignment_score = float(chan['alignment_score'][pre_idx])

        # 涨停前盘整天数（趋势类型=1的连续K线数）
        cons_days = 0
        for i in range(pre_idx, max(0, pre_idx - 30), -1):
            if chan['trend_type'][i] == 1:
                cons_days += 1
            else:
                break
        d.pre_consolidation_days = cons_days

        # 涨停日信号
        d.cur_buy_point = int(chan['buy_point'][-1])
        d.cur_buy_confidence = float(chan['buy_confidence'][-1])
        d.cur_bottom_divergence = bool(chan.get('hidden_bottom_divergence', np.zeros(n))[-1])

        # 中枢突破检测
        zg = float(chan['pivot_zg'][-1])
        zd = float(chan['pivot_zd'][-1])
        if not np.isnan(zg):
            d.pivot_zg = zg
            d.pivot_zd = zd
            d.above_pivot_pct = (close[-1] - zg) / zg * 100 if zg > 0 else 0
            # 突破判断: 涨停日收盘 > ZG 且 前日收盘 <= ZG
            d.pivot_breakout = (close[-1] > zg and close[-2] <= zg * 1.01)
            if not d.pivot_breakout and close[-1] > zg * 1.03:
                d.pivot_breakout = True  # 大幅突破也计入

        if not np.isnan(zd):
            d.below_pivot_pct = (zd - close[-2]) / zd * 100 if zd > 0 else 0

        # 量能
        avg_vol_5 = np.mean(volume[-6:-1]) if len(volume) >= 6 and np.mean(volume[-6:-1]) > 0 else 1
        d.vol_spike = volume[-1] / avg_vol_5

        # 底分型质量
        d.bottom_fx_quality = float(chan.get('bottom_fractal_quality', np.zeros(n))[-1])

        # === 缠论归因分类 ===
        d.limit_up_type, d.type_confidence, d.chan_explanation = self._classify_limit_up_type(d)

        return d

    def _classify_limit_up_type(self, d: LimitUpDiagnosis) -> Tuple[str, float, str]:
        """基于缠论结构对涨停进行归因分类

        六大涨停模式：
        1. 中枢突破板 — 突破中枢上沿ZG，三买B3触发
        2. 底背驰反转板 — 下跌趋势+底背驰+一买B1 → V型反转
        3. 二买确认板 — 一买后回调不破前低，B2确认后加速涨停
        4. 趋势加速板 — 上涨趋势+中枢上方+放量，延续加速
        5. 笔耗尽反转板 — 下跌笔趋势耗尽+底分型 → 反转涨停
        6. 盘整突破板 — 长期盘整后首日突破+放量
        """
        scores = []  # [(type_name, confidence, explanation)]

        # 1. 中枢突破板
        if d.pivot_breakout and d.above_pivot_pct > 0:
            conf = min(0.95, 0.55 + d.above_pivot_pct / 20 + (0.15 if d.vol_spike > 1.5 else 0))
            scores.append(('中枢突破板', conf,
                f'突破中枢上沿ZG({d.pivot_zg:.2f})，收盘高于ZG {d.above_pivot_pct:.1f}%'
                f'{", 放量" if d.vol_spike > 1.5 else ""}'
                f'{", B3三买触发" if d.cur_buy_point == 3 else ""}'))

        # 2. 底背驰反转板
        if d.pre_bottom_divergence and d.pre_trend_type == -2 and d.pre_pivot_position <= 0:
            conf = 0.55 + (0.15 if d.pre_buy_point == 1 else 0) + (0.1 if d.bottom_fx_quality > 0.3 else 0)
            scores.append(('底背驰反转板', min(0.9, conf),
                f'下跌趋势+底背驰，前5日底背驰确认'
                f'{", B1一买触发" if d.pre_buy_point == 1 else ""}'
                f'{", 底分型质量{:.0%}" if d.bottom_fx_quality > 0.3 else ""}'.format(d.bottom_fx_quality)))

        # 3. 二买确认板
        if d.pre_buy_point == 2 and d.pre_trend_type >= 1 and d.pre_pivot_position <= 0:
            conf = 0.60 + (0.15 if d.vol_spike > 1.5 else 0) + (0.1 if d.pre_alignment_score > 0 else 0)
            scores.append(('二买确认板', min(0.9, conf),
                f'B2二买后加速，前日笔方向{d.pre_stroke_direction}'
                f'{", 均线多头排列" if d.pre_alignment_score > 0.3 else ""}'))

        # 4. 趋势加速板
        if d.pre_trend_type == 2 and d.pre_pivot_position == 1 and d.pre_alignment_score > 0.3:
            conf = 0.50 + (0.15 if d.vol_spike > 2 else 0) + (0.1 if d.cur_buy_point >= 3 else 0)
            scores.append(('趋势加速板', min(0.85, conf),
                f'上涨趋势+中枢上方，均线多头排列'
                f'{", 放量加速" if d.vol_spike > 2 else ""}'))

        # 5. 笔耗尽反转板
        if d.pre_stroke_direction == -1 and d.bottom_fx_quality > 0.25:
            # 前日下跌笔 + 底分型质量 → 笔趋势耗尽
            conf = 0.45 + d.bottom_fx_quality * 0.3 + (0.1 if d.vol_spike > 1.5 else 0)
            scores.append(('笔耗尽反转板', min(0.8, conf),
                f'下跌笔趋势耗尽+底分型(质量{d.bottom_fx_quality:.0%})→反转'
                f'{", 量在价先" if d.vol_spike > 2 else ""}'))

        # 6. 盘整突破板
        if d.pre_consolidation_days >= 8 and d.vol_spike > 1.5:
            conf = 0.45 + min(d.pre_consolidation_days / 50, 0.25) + (0.15 if d.vol_spike > 2 else 0)
            scores.append(('盘整突破板', min(0.85, conf),
                f'盘整{d.pre_consolidation_days}天后突破'
                f'{", 倍量启动" if d.vol_spike > 2 else ", 放量启动"}'))

        if scores:
            scores.sort(key=lambda x: x[1], reverse=True)
            return scores[0]

        # 兜底: 根据现有数据定性
        if d.vol_spike > 2:
            return ('放量异动板', 0.35, '无明显缠论买点，但放量明显，关注后续结构演变')
        return ('未分类', 0.2, f'涨{d.change_pct:.1f}%，量比{d.volume_ratio:.1f}，建议观察后续走势结构')

    def _extract_limit_up_commonalities(
        self, diagnoses: List[LimitUpDiagnosis]
    ) -> List[str]:
        """从涨停股中提取共性特征"""
        traits = []
        n = len(diagnoses)

        # 1. 主导模式统计
        type_counts = defaultdict(int)
        for d in diagnoses:
            type_counts[d.limit_up_type] += 1
        top_type, top_count = max(type_counts.items(), key=lambda x: x[1])
        if top_count >= n * 0.3:
            traits.append(f'今日涨停主导模式: {top_type} ({top_count}/{n}, {top_count/n:.0%})，'
                         f'暗示当前市场偏好此类结构')

        # 2. 涨停前结构共性
        pre_up_trend = sum(1 for d in diagnoses if d.pre_trend_type == 2)
        pre_down_trend = sum(1 for d in diagnoses if d.pre_trend_type == -2)
        pre_cons = sum(1 for d in diagnoses if d.pre_trend_type == 1)

        if pre_up_trend >= n * 0.5:
            traits.append(f'涨停前已处上涨趋势: {pre_up_trend}/{n} ({pre_up_trend/n:.0%})，'
                         f'趋势延续型涨停为主，选股应偏向已形成上涨结构的个股')
        elif pre_down_trend >= n * 0.5:
            traits.append(f'涨停前处下跌趋势: {pre_down_trend}/{n} ({pre_down_trend/n:.0%})，'
                         f'超跌反弹型涨停为主，需警惕次日分化')
        elif pre_cons >= n * 0.5:
            traits.append(f'涨停前处盘整: {pre_cons}/{n} ({pre_cons/n:.0%})，'
                         f'盘整突破型，关注突破后的持续性')

        # 3. 中枢位置共性
        pivot_below = sum(1 for d in diagnoses if d.pre_pivot_position == -1)
        pivot_above = sum(1 for d in diagnoses if d.pre_pivot_position == 1)
        if pivot_below >= n * 0.4:
            traits.append(f'{pivot_below}/{n}涨停股前日处中枢下方，中枢突破是主要驱动力')
        if pivot_above >= n * 0.5:
            traits.append(f'{pivot_above}/{n}涨停股前日处中枢上方，强势股延续，追涨风险相对可控')

        # 4. 量能共性
        high_vol = sum(1 for d in diagnoses if d.vol_spike > 2)
        if high_vol >= n * 0.6:
            traits.append(f'{high_vol}/{n}涨停股倍量启动(量比>2x)，量在价先信号明确，'
                         f'但需关注次日量能是否持续')

        # 5. 均线排列共性
        bullish_align = sum(1 for d in diagnoses if d.pre_alignment_score > 0.3)
        if bullish_align >= n * 0.5:
            traits.append(f'{bullish_align}/{n}涨停前均线多头排列，技术共振增强涨停可靠性')

        # 6. 底背驰共性
        bottom_div = sum(1 for d in diagnoses if d.pre_bottom_divergence or d.cur_bottom_divergence)
        if bottom_div >= n * 0.3:
            traits.append(f'{bottom_div}/{n}涨停股呈现底背驰特征，'
                         f'缠论"没有趋势，没有背驰"→背驰后的反转力度较强')

        # 7. 行业集中
        ind_counts = defaultdict(int)
        for d in diagnoses:
            ind_counts[d.industry] += 1
        top_ind = max(ind_counts, key=ind_counts.get)
        top_ind_count = ind_counts[top_ind]
        if top_ind_count >= n * 0.25:
            traits.append(f'涨停集中在[{top_ind}]({top_ind_count}/{n})，'
                         f'板块效应显著，该行业可能处于主升浪')

        # 8. 综合建议
        if n >= 5:
            traits.append(f'操作建议: 今日{n}只涨停股中，'
                         f'建议关注{top_type}标的（缠论结构最清晰）')

        return traits

    # ========== 涨停板报告打印 ==========

    def _print_limit_up_analysis(self, summary: LimitUpSummary, n: int):
        """打印涨停板缠论归因分析"""
        if not summary.limit_up_stocks:
            return

        print(f'\n┌{"─"*76}┐')
        print(f'│  Step 3: 涨停板缠论归因分析{" " * 50}│')
        print(f'├{"─"*76}┤')

        # 概览
        print(f'│  今日涨停: {summary.total_limit_up} 只{" " * 60}│')
        if summary.dominant_pattern:
            print(f'│  主导模式: {summary.dominant_pattern}{" " * (64 - len(summary.dominant_pattern))}│')

        # 类型分布
        type_items = sorted(summary.type_distribution.items(), key=lambda x: x[1], reverse=True)
        dist_str = '  '.join(f'{t}:{c}只' for t, c in type_items[:6])
        print(f'│  类型分布: {dist_str:<64s}│')

        # 行业集中
        top_inds = sorted(summary.industry_concentration.items(), key=lambda x: x[1], reverse=True)[:5]
        if top_inds:
            ind_str = '  '.join(f'{i}:{c}只' for i, c in top_inds)
            print(f'│  涨停行业集中: {ind_str:<58s}│')

        print(f'├{"─"*76}┤')

        # 个股归因明细
        print(f'│  {"代码":<10s} {"名称":<10s} {"涨跌":>6s} {"量比":>5s} '
              f'{"归因类型":<14s} {"置信":>5s} {"涨停前缠论结构":^30s}│')
        print(f'│  {"-"*74}│')

        trend_map = {2: '上涨趋势', 1: '盘整', 0: '--', -2: '下跌趋势'}
        pp_map = {1: '中枢上', 0: '中枢内', -1: '中枢下'}

        for d in summary.limit_up_stocks[:n]:
            pre_structure = (
                f'{trend_map.get(d.pre_trend_type, "?"):>4s} | '
                f'{pp_map.get(d.pre_pivot_position, "?"):>3s} | '
                f'笔:{"↑" if d.pre_stroke_direction == 1 else "↓" if d.pre_stroke_direction == -1 else "—"}'
            )
            print(f'│  {d.symbol:<10s} {d.name:<10s} {d.change_pct:>+5.1f}% '
                  f'{d.vol_spike:>4.1f}x {d.limit_up_type:<14s} {d.type_confidence:>4.0%} '
                  f'{pre_structure:<30s}│')

        print(f'├{"─"*76}┤')

        # 缠论解释
        print(f'│  缠论归因详情:{" " * 62}│')
        for d in summary.limit_up_stocks[:min(n, 8)]:
            # 截短解释以避免溢出
            expl = d.chan_explanation[:72]
            print(f'│  [{d.symbol} {d.name}] {d.limit_up_type} ({d.type_confidence:.0%})')
            print(f'│    → {expl:<70s}│')

        print(f'└{"─"*76}┘')

        # 共性总结
        if summary.common_traits:
            print(f'\n┌{"─"*76}┐')
            print(f'│  涨停板共性总结{" " * 62}│')
            print(f'├{"─"*76}┤')
            for i, trait in enumerate(summary.common_traits):
                print(f'│  {i+1}. {trait:<72s}│')
            print(f'└{"─"*76}┘')

    # ========== Step 6: 作战计划 + 系统进化 ==========

    def generate_portfolio_plan(
        self, diagnoses: List[StockDiagnosis],
        market: MarketVitalSigns, max_positions: int = 8
    ) -> List[StockDiagnosis]:
        """生成明日作战计划 — 行业分散选股"""
        # 根据市场环境定仓位上限
        if market.sentiment_cycle == '冰点':
            max_positions = max(2, max_positions // 4)
        elif market.sentiment_cycle == '高潮':
            max_positions = min(max_positions, 8)
        elif market.sentiment_cycle == '退潮':
            max_positions = max(1, max_positions // 3)
        elif market.bear_risk:
            max_positions = max(1, max_positions // 2)

        candidates = [d for d in diagnoses
                      if d.composite_score >= 0.15 and d.opportunity_rank in ('A', 'B')]
        candidates.sort(key=lambda x: x.composite_score, reverse=True)

        selected = []
        used_industries = set()
        for d in candidates:
            if len(selected) >= max_positions:
                break
            # 行业分散：前2只不限制，后面每个行业最多1只
            if len(selected) < 3 or d.industry not in used_industries:
                selected.append(d)
                used_industries.add(d.industry)

        return selected

    def generate_evolution_patches(
        self, diagnoses: List[StockDiagnosis],
        market: MarketVitalSigns, industries: List[IndustryMainLine]
    ) -> List[EvolutionPatch]:
        """生成系统自进化补丁 — 根据复盘发现，建议参数调整"""

        patches = []

        # 1. 买卖信号浓度 → 调整 buy_threshold
        buy_count = sum(1 for d in diagnoses if d.buy_point > 0)
        buy_rate = buy_count / max(len(diagnoses), 1)
        if buy_rate < 0.05:
            patches.append(EvolutionPatch(
                section='signal', key='buy_threshold',
                current_value=0.06, suggested_value=0.04,
                reason=f'买点信号稀少({buy_rate:.1%})，建议降低买入阈值以捕获更多机会',
                confidence=0.7, urgency='high',
            ))
        elif buy_rate > 0.30:
            patches.append(EvolutionPatch(
                section='signal', key='buy_threshold',
                current_value=0.06, suggested_value=0.08,
                reason=f'买点信号过多({buy_rate:.1%})，建议提高阈值以提升信号质量',
                confidence=0.65, urgency='medium',
            ))

        # 2. 信号类型分布 → 调整缠论买卖点乘数
        b1_count = sum(1 for d in diagnoses if d.buy_point == 1)
        b2_count = sum(1 for d in diagnoses if d.second_buy)
        b3_count = sum(1 for d in diagnoses if d.buy_point == 3)
        s1_count = sum(1 for d in diagnoses if d.sell_point == 1)

        if b2_count > b1_count:
            patches.append(EvolutionPatch(
                section='chan_theory_enhanced.buy_sell_points', key='b2_buy_mult',
                current_value=1.20, suggested_value=1.30,
                reason=f'二买信号({b2_count})比一买({b1_count})更活跃，提高二买权重',
                confidence=0.6, urgency='medium',
            ))

        if s1_count > buy_count * 0.5:
            patches.append(EvolutionPatch(
                section='chan_theory_enhanced.buy_sell_points', key='s1_sell_mult',
                current_value=0.60, suggested_value=0.50,
                reason=f'一卖信号({s1_count})较多，加强卖出削弱力度',
                confidence=0.55, urgency='medium',
            ))

        # 3. 中阴状态过多 → 调整中阴惩罚
        zhongyin_count = sum(1 for d in diagnoses if hasattr(d, 'zhongyin') and getattr(d, 'zhongyin', False))
        if zhongyin_count > len(diagnoses) * 0.3:
            patches.append(EvolutionPatch(
                section='chan_theory_enhanced.signal', key='zhongyin_penalty',
                current_value=0.85, suggested_value=0.75,
                reason=f'中阴股票占比高({zhongyin_count/len(diagnoses):.0%})，加强中阴惩罚',
                confidence=0.6, urgency='low',
            ))

        # 4. 市场状态 → 调整 regime_multiplier
        if market.bear_risk and market.regime == '熊市':
            patches.append(EvolutionPatch(
                section='regime_multiplier', key='bear',
                current_value=0.80, suggested_value=0.70,
                reason=f'熊市风险警报，建议进一步降低仓位',
                confidence=0.7, urgency='high',
            ))

        # 5. 行业主线检测 → 调整行业因子
        top_industries = [ind for ind in industries if ind.hot_rank <= 3 and ind.total_buy_signals >= 3]
        if top_industries:
            for ind in top_industries:
                patches.append(EvolutionPatch(
                    section=f'industry_factors.{ind.industry}', key='enabled',
                    current_value=1, suggested_value=1,
                    reason=f'强主线行业({ind.hot_rank}): {ind.summary}，建议确保该行业因子配置充足',
                    confidence=0.5, urgency='low',
                ))

        return patches

    # ========== 主运行流程 ==========

    def run(self, date: str = None, top_n: int = 15,
            symbols: List[str] = None, evolve: bool = False) -> DailyReviewReport:
        """运行完整复盘 + 可选进化"""
        if date is None:
            date = self._find_latest_date()

        # 获取股票列表
        if symbols is None:
            symbols = self._list_available_stocks()
        print(f'[加载] {len(symbols)} 只股票')

        index_df = self._load_index_data(date)

        # Step 1: 大盘体检
        print('[Step 1/6] 大盘环境深度体检...')
        market = self.examine_market_vitals(date, symbols)

        # Step 2+3+5: 个股全面诊断（缠论+排雷）
        print('[Step 2-5/6] 个股缠论诊断 & 排雷...')
        diagnoses = []
        errors = 0
        for i, sym in enumerate(symbols):
            if (i + 1) % 300 == 0:
                print(f'  进度: {i+1}/{len(symbols)}')
            try:
                d = self.diagnose_stock(sym, date, index_df)
                if d is not None:
                    diagnoses.append(d)
            except Exception as e:
                errors += 1
                if errors <= 3:
                    print(f'  [WARN] {sym}: {e}')

        print(f'  有效诊断: {len(diagnoses)} 只 (错误: {errors})')

        # 行业主线
        print('[Step 2/6] 行业主线分析...')
        industries = self.analyze_industry_main_lines(diagnoses)

        # 机会 & 风险
        opportunities = self.classify_opportunities(diagnoses)
        risks = self.classify_risks(diagnoses)

        # Step 3: 涨停板缠论归因分析
        print('[Step 3/6] 涨停板缠论归因分析...')
        limit_up_summary = self.analyze_limit_up_stocks(date, symbols, index_df)
        print(f'  涨停股: {limit_up_summary.total_limit_up} 只'
              + (f', 主导模式: {limit_up_summary.dominant_pattern}' if limit_up_summary.dominant_pattern else ''))

        # Step 6: 作战计划 + 进化
        print('[Step 6/6] 生成作战计划 + 系统进化建议...')
        portfolio = self.generate_portfolio_plan(diagnoses, market)
        patches = self.generate_evolution_patches(diagnoses, market, industries) if evolve else []

        # 组装报告
        report = DailyReviewReport(
            date=date,
            market=market,
            main_lines=industries,
            opportunities=opportunities,
            risk_alerts=risks,
            portfolio_suggestions=portfolio,
            limit_up_summary=limit_up_summary,
            evolution_patches=patches,
            summary={
                'total_stocks': len(diagnoses),
                'buy_signals': sum(1 for d in diagnoses if d.buy_point > 0),
                'sell_signals': sum(1 for d in diagnoses if d.sell_point > 0),
                'a_rank': sum(1 for d in diagnoses if d.opportunity_rank == 'A'),
                'b_rank': sum(1 for d in diagnoses if d.opportunity_rank == 'B'),
                'up_trend': sum(1 for d in diagnoses if d.trend_type == 2),
                'down_trend': sum(1 for d in diagnoses if d.trend_type == -2),
                'consolidation': sum(1 for d in diagnoses if d.trend_type == 1),
                'confirmed_buy': sum(1 for d in diagnoses if d.confirmed_buy),
                'confirmed_sell': sum(1 for d in diagnoses if d.confirmed_sell),
                'second_buy': sum(1 for d in diagnoses if d.second_buy),
                'high_risk': sum(1 for d in diagnoses if d.risk_level == '高危'),
            },
        )

        # 打印报告
        self._print_report(report, top_n)

        # 导出
        self._export_report(report)

        # 进化
        if evolve and patches:
            self._apply_evolution(patches, date)

        return report

    # ========== 报告打印 ==========

    def _print_report(self, report: DailyReviewReport, n: int):
        """打印完整复盘报告"""
        m = report.market
        summary = report.summary

        # ═══════════ 头部 ═══════════
        print(f'\n{"="*78}')
        print(f'  量化系统每日复盘报告')
        print(f'  日期: {report.date}')
        print(f'{"="*78}')

        # ─── Step 1: 大盘体检 ───
        print(f'\n┌{"─"*76}┐')
        print(f'│  Step 1: 大盘环境深度体检{" " * 49}│')
        print(f'├{"─"*76}┤')
        print(f'│  上证指数: {m.index_close:>10.1f}  ({m.index_change_pct:+.2f}%)  '
              f'{m.index_volume:<4s}  ({m.volume_ratio:.1f}x均量){" " * 5}│')
        print(f'│  市场状态: {m.regime:<4s} (置信度 {m.regime_confidence:.0%})  |  '
              f'动量: {m.momentum_score:+.2f}  |  波动率: {m.volatility_pct:.1%}  │')
        print(f'│  情绪周期: {m.sentiment_cycle:<4s}  |  '
              f'风格: {m.style_regime:<12s}  |  '
              f'{"⚠ 熊市风险" if m.bear_risk else "✓ 无极端风险"}         │')
        print(f'│  赚钱效应: 涨 {m.advancing} / 跌 {m.declining} '
              f'(上涨率 {m.up_pct:.0f}%)  |  '
              f'涨停 {m.limit_up_count} / 跌停 {m.limit_down_count}'
              f'{" (涨停主导)" if m.limit_up_ratio >= 3 else " (跌停潮)" if m.limit_up_ratio <= 0.3 else ""}│')
        if m.index_divergence:
            print(f'│  ⚠ 指数分化: 上证/深成/创业板 走势不同步                     │')
        print(f'│  综合判定: {m.summary:<62s}│')
        print(f'└{"─"*76}┘')

        # ─── Step 2: 行业主线 ───
        print(f'\n┌{"─"*76}┐')
        print(f'│  Step 2: 板块与主线逻辑拆解{" " * 51}│')
        print(f'├{"─"*76}┤')
        print(f'│  {"行业":<16s} {"股票数":>5s} {"涨趋势":>6s} {"跌趋势":>6s} '
              f'{"B1/B2/B3":>10s} {"梯队":>5s} {"综合":>6s} {"评级":^14s}│')
        print(f'│  {"-"*74}│')

        for ind in report.main_lines[:15]:
            formation_bar = '★' * int(ind.formation_score * 3) + '☆' * (3 - int(ind.formation_score * 3))
            print(f'│  {ind.industry:<16s} {ind.stock_count:>5d} '
                  f'{ind.up_trend_pct:>5.0f}% {ind.down_trend_pct:>5.0f}% '
                  f'{ind.b1_count}/{ind.b2_count}/{ind.b3_count}{"":>4s} '
                  f'{formation_bar:<5s} {ind.avg_composite_score:>+5.2f} '
                  f'{ind.summary[:14]:<14s}│')
        print(f'└{"─"*76}┘')

        # ─── Step 3: 涨停板缠论归因 ───
        if report.limit_up_summary:
            self._print_limit_up_analysis(report.limit_up_summary, n)

        # ─── Step 3+5: 缠论机会 ───
        self._print_signal_section(report.opportunities, 'Step 3+5: 缠论结构扫描 — 买点机会', n, is_buy=True)

        # ─── Step 4: 风险预警 ───
        self._print_signal_section(report.risk_alerts, 'Step 4: 风险预警 — 持仓排雷', n, is_buy=False)

        # ─── Step 6: 作战计划 ───
        print(f'\n┌{"─"*76}┐')
        print(f'│  Step 6: 明日作战计划{" " * 56}│')
        print(f'├{"─"*76}┤')

        if report.portfolio_suggestions:
            # 仓位建议
            if report.market.sentiment_cycle in ('冰点', '退潮'):
                pos_advice = '防御仓位 (1-3成)'
            elif report.market.sentiment_cycle == '高潮':
                pos_advice = '积极仓位 (5-7成)'
            elif report.market.bear_risk:
                pos_advice = '轻仓或空仓 (0-2成)'
            else:
                pos_advice = '中性仓位 (3-5成)'

            print(f'│  仓位建议: {pos_advice:<60s}│')
            print(f'│  推荐标的:{" " * 66}│')
            print(f'│  {"#":>2s} {"代码":<10s} {"名称":<10s} {"行业":<14s} '
                  f'{"收盘":>7s} {"买点":>5s} {"止损":>7s} {"综合":>6s} {"等级":>4s}│')
            for i, d in enumerate(report.portfolio_suggestions):
                bp_label = {1: 'B1', 2: 'B2', 3: 'B3'}.get(d.buy_point, '')
                if d.second_buy:
                    bp_label = 'B2*'
                print(f'│  {i+1:>2d} {d.symbol:<10s} {d.name:<10s} {d.industry:<14s} '
                      f'{d.close:>7.2f} {bp_label:>5s} {d.stop_loss_price:>7.2f} '
                      f'{d.composite_score:>+5.2f} {d.opportunity_rank:>4s}│')
        else:
            print(f'│  无符合条件的推荐标的{" " * 54}│')
        print(f'└{"─"*76}┘')

        # ─── 进化补丁 ───
        if report.evolution_patches:
            print(f'\n┌{"─"*76}┐')
            print(f'│  🔧 系统自进化建议{" " * 60}│')
            print(f'├{"─"*76}┤')
            for p in report.evolution_patches:
                urgency_mark = {'urgent': '🔴', 'high': '🟠', 'medium': '🟡', 'low': '🟢'}
                print(f'│  {urgency_mark.get(p.urgency, "⚪")} [{p.section}.{p.key}]')
                print(f'│     当前值: {p.current_value} → 建议值: {p.suggested_value}')
                print(f'│     理由: {p.reason}')
                print(f'│     置信度: {p.confidence:.0%}')
            print(f'└{"─"*76}┘')

        # ─── 摘要 ───
        print(f'\n{"="*78}')
        print(f'  复盘摘要')
        print(f'  {"─"*56}')
        print(f'  总股票数: {summary["total_stocks"]:<6d}  |  '
              f'上涨趋势: {summary["up_trend"]:<4d}  |  '
              f'盘整: {summary["consolidation"]:<4d}  |  '
              f'下跌趋势: {summary["down_trend"]}')
        print(f'  买点信号: {summary["buy_signals"]:<6d}  |  '
              f'卖点信号: {summary["sell_signals"]:<6d}  |  '
              f'A级机会: {summary["a_rank"]:<4d}  |  '
              f'B级机会: {summary["b_rank"]}')
        print(f'  双级别确认买: {summary["confirmed_buy"]:<4d}  |  '
              f'二买确认: {summary["second_buy"]:<4d}  |  '
              f'高危股票: {summary["high_risk"]}')
        print(f'{"="*78}\n')

    def _print_signal_section(
        self, groups: Dict[str, List[StockDiagnosis]],
        title: str, n: int, is_buy: bool
    ):
        """打印信号区域（机会或风险）"""
        section_config = {
            'A_strong_buy': ('🟢 A级 强买信号（多级别确认）', ['A_strong_buy', 'B2_confirmed', 'B1_reversal', 'B3_acceleration', 'double_confirmed']),
            'confirmed_sell': ('🔴 多级别确认卖点', ['confirmed_sell', 'S1_top_reversal', 'S3_breakdown', 'top_divergence', 'ma_breakdown']),
        }

        if is_buy:
            ordered_keys = ['A_strong_buy', 'B2_confirmed', 'B1_reversal', 'B3_acceleration', 'double_confirmed']
            labels = {
                'A_strong_buy': '🟢 A级 强买（多级别确认）',
                'B2_confirmed': '🔵 B2 二买确认（最安全买点）',
                'B1_reversal': '🟡 B1 一买（底部反转）',
                'B3_acceleration': '🟣 B3 三买（趋势加速）',
                'double_confirmed': '⭐ 双级别确认',
            }
        else:
            ordered_keys = ['confirmed_sell', 'S1_top_reversal', 'S3_breakdown', 'top_divergence', 'ma_breakdown']
            labels = {
                'confirmed_sell': '🔴 多级别确认卖点',
                'S1_top_reversal': '🟠 S1 一卖（顶部反转）',
                'S3_breakdown': '🟤 S3 三卖（趋势破位）',
                'top_divergence': '⚠ 顶背驰预警',
                'ma_breakdown': '📉 均线破位（≥2条）',
            }

        print(f'\n┌{"─"*76}┐')
        print(f'│  {title:<70s}│')
        print(f'├{"─"*76}┤')

        any_shown = False
        for key in ordered_keys:
            snaps = groups.get(key, [])
            if not snaps:
                continue
            any_shown = True
            label = labels.get(key, key)
            print(f'│  [{label}] 共 {len(snaps)} 只{" " * (62 - len(label) - len(str(len(snaps))))}│')
            if not is_buy and key == 'ma_breakdown':
                # 均线破位特殊表头
                pass
            else:
                print(f'│  {"代码":<10s} {"名称":<10s} {"行业":<14s} '
                      f'{"收盘":>7s} {"涨跌":>6s} {"走势":>4s} '
                      f'{"中枢位":>5s} {"置信":>5s} {"综合":>6s} │')

            trend_map = {2: '上涨', 1: '盘整', 0: '--', -2: '下跌'}
            pp_map = {1: '上方', 0: '内部', -1: '下方'}

            for s in snaps[:n]:
                conf = s.buy_confidence if is_buy else s.sell_confidence
                print(f'│  {s.symbol:<10s} {s.name:<10s} {s.industry:<14s} '
                      f'{s.close:>7.2f} {s.change_pct:>+5.1f}% '
                      f'{trend_map.get(s.trend_type, "?"):>4s} '
                      f'{pp_map.get(s.pivot_position, "?"):>5s} '
                      f'{conf:>4.0%} '
                      f'{s.composite_score:>+5.2f} │')

        if not any_shown:
            if is_buy:
                print(f'│  今日无显著买点信号{" " * 56}│')
            else:
                print(f'│  今日无显著风险信号{" " * 56}│')

        print(f'└{"─"*76}┘')

    # ========== 导出 ==========

    def _export_report(self, report: DailyReviewReport):
        """导出报告为CSV + JSON"""
        date_str = report.date.replace('-', '')

        # 市场体检 → JSON
        market_path = os.path.join(self.output_dir, f'market_vitals_{date_str}.json')
        with open(market_path, 'w') as f:
            json.dump({
                'date': report.date,
                'index_close': report.market.index_close,
                'index_change_pct': report.market.index_change_pct,
                'index_volume': report.market.index_volume,
                'regime': report.market.regime,
                'sentiment_cycle': report.market.sentiment_cycle,
                'advancing': report.market.advancing,
                'declining': report.market.declining,
                'limit_up': report.market.limit_up_count,
                'limit_down': report.market.limit_down_count,
                'bear_risk': report.market.bear_risk,
                'summary': report.market.summary,
            }, f, ensure_ascii=False, indent=2)

        # 行业主线 → CSV
        ind_rows = []
        for ind in report.main_lines:
            ind_rows.append({
                'industry': ind.industry, 'stock_count': ind.stock_count,
                'up_trend_pct': ind.up_trend_pct, 'down_trend_pct': ind.down_trend_pct,
                'b1': ind.b1_count, 'b2': ind.b2_count, 'b3': ind.b3_count,
                'formation_score': ind.formation_score,
                'avg_composite': ind.avg_composite_score,
                'summary': ind.summary, 'hot_rank': ind.hot_rank,
            })
        if ind_rows:
            pd.DataFrame(ind_rows).to_csv(
                os.path.join(self.output_dir, f'industry_lines_{date_str}.csv'), index=False
            )

        # 进化补丁 → JSON
        if report.evolution_patches:
            patch_path = os.path.join(self.output_dir, f'evolution_patch_{date_str}.json')
            with open(patch_path, 'w') as f:
                json.dump([{
                    'section': p.section, 'key': p.key,
                    'current': p.current_value, 'suggested': p.suggested_value,
                    'reason': p.reason, 'confidence': p.confidence, 'urgency': p.urgency,
                } for p in report.evolution_patches], f, ensure_ascii=False, indent=2)

        # 完整报告文档 → Markdown
        report_path = os.path.join(self.output_dir, f'daily_review_{date_str}.md')
        self._export_document(report, report_path)

        print(f'\n报告已导出至: {self.output_dir}/')
        print(f'  完整报告: daily_review_{date_str}.md')
        print(f'  市场体检: market_vitals_{date_str}.json')
        print(f'  行业主线: industry_lines_{date_str}.csv')
        if report.evolution_patches:
            print(f'  进化补丁: evolution_patch_{date_str}.json')

    def _export_document(self, report: DailyReviewReport, path: str):
        """导出完整复盘报告文档 (Markdown)"""
        m = report.market
        summary = report.summary
        lines = []

        def w(text=''):
            lines.append(text)

        w(f'# 量化系统每日复盘报告')
        w(f'**日期**: {report.date}')
        w()

        # ─── 1. 大盘体检 ───
        w('## 一、大盘环境深度体检')
        w()
        w(f'| 指标 | 数值 | 指标 | 数值 |')
        w(f'|------|------|------|------|')
        w(f'| 上证指数 | {m.index_close:.1f} | 涨跌幅 | {m.index_change_pct:+.2f}% |')
        w(f'| 量能 | {m.index_volume} ({m.volume_ratio:.1f}x均量) | 情绪周期 | {m.sentiment_cycle} |')
        w(f'| 市场状态 | {m.regime} (置信度 {m.regime_confidence:.0%}) | 风格 | {m.style_regime} |')
        w(f'| 动量 | {m.momentum_score:+.2f} | 波动率 | {m.volatility_pct:.1%} |')
        w(f'| 上涨/下跌 | {m.advancing} / {m.declining} ({m.up_pct:.0f}%) | 涨停/跌停 | {m.limit_up_count} / {m.limit_down_count} |')
        w(f'| 综合判定 | {m.summary} | 熊市风险 | {"⚠ 是" if m.bear_risk else "✓ 否"} |')
        w()

        # ─── 2. 行业主线 ───
        w('## 二、行业主线分析')
        w()
        if report.main_lines:
            w(f'| 排名 | 行业 | 股票数 | 上涨趋势 | 下跌趋势 | B1/B2/B3 | 梯队 | 综合评分 | 判定 |')
            w(f'|------|------|--------|----------|----------|----------|------|----------|------|')
            for ind in report.main_lines[:20]:
                formation_bar = '★' * int(ind.formation_score * 3) + '☆' * (3 - int(ind.formation_score * 3))
                w(f'| {ind.hot_rank} | {ind.industry} | {ind.stock_count} | '
                  f'{ind.up_trend_pct:.0f}% | {ind.down_trend_pct:.0f}% | '
                  f'{ind.b1_count}/{ind.b2_count}/{ind.b3_count} | {formation_bar} | '
                  f'{ind.avg_composite_score:+.2f} | {ind.summary} |')
        else:
            w('无数据')
        w()

        # ─── 3. 涨停板缠论归因 ───
        if report.limit_up_summary and report.limit_up_summary.limit_up_stocks:
            lu = report.limit_up_summary
            w('## 三、涨停板缠论归因分析')
            w()
            w(f'- **涨停总数**: {lu.total_limit_up} 只')
            w(f'- **主导模式**: {lu.dominant_pattern}')
            w(f'- **类型分布**: {", ".join(f"{t}:{c}只" for t, c in sorted(lu.type_distribution.items(), key=lambda x: x[1], reverse=True))}')

            # 个股归因表
            w()
            w(f'| 代码 | 名称 | 涨幅 | 量比 | 归因类型 | 置信度 | 涨停前结构 |')
            w(f'|------|------|------|------|----------|--------|------------|')
            trend_map = {2: '上涨趋势', 1: '盘整', 0: '--', -2: '下跌趋势'}
            pp_map = {1: '中枢上', 0: '中枢内', -1: '中枢下'}
            for d in lu.limit_up_stocks[:30]:
                pre = f'{trend_map.get(d.pre_trend_type, "?")}/{pp_map.get(d.pre_pivot_position, "?")}'
                w(f'| {d.symbol} | {d.name} | {d.change_pct:+.1f}% | {d.vol_spike:.1f}x | '
                  f'{d.limit_up_type} | {d.type_confidence:.0%} | {pre} |')

            # 缠论归因详情
            w()
            w('### 缠论归因详情')
            for d in lu.limit_up_stocks[:20]:
                w(f'- **{d.symbol} {d.name}** [{d.limit_up_type}] (置信度 {d.type_confidence:.0%})')
                w(f'  - {d.chan_explanation}')
            w()

            # 共性总结
            if lu.common_traits:
                w('### 涨停板共性总结')
                for i, trait in enumerate(lu.common_traits):
                    w(f'{i+1}. {trait}')
                w()

            # 行业集中
            if lu.industry_concentration:
                w('### 涨停行业集中度')
                w(f'| 行业 | 涨停数 |')
                w(f'|------|--------|')
                for ind, cnt in lu.industry_concentration.items():
                    w(f'| {ind} | {cnt} |')
                w()

        # ─── 4. 买卖点机会 ───
        w('## 四、缠论买卖点机会')
        w()
        op_labels = {
            'A_strong_buy': 'A级 强买（多级别确认）',
            'B2_confirmed': 'B2 二买确认',
            'B1_reversal': 'B1 一买（底部反转）',
            'B3_acceleration': 'B3 三买（趋势加速）',
            'double_confirmed': '双级别确认',
        }
        trend_map = {2: '上涨趋势', 1: '盘整', 0: '--', -2: '下跌趋势'}
        for key, label in op_labels.items():
            snaps = report.opportunities.get(key, [])
            if not snaps:
                continue
            w(f'### {label} ({len(snaps)}只)')
            w(f'| 代码 | 名称 | 行业 | 收盘 | 涨跌 | 走势 | 中枢位 | 置信 | 综合 |')
            w(f'|------|------|------|------|------|------|--------|------|------|')
            pp_map = {1: '上方', 0: '内部', -1: '下方'}
            for s in snaps[:15]:
                w(f'| {s.symbol} | {s.name} | {s.industry} | {s.close:.2f} | '
                  f'{s.change_pct:+.1f}% | {trend_map.get(s.trend_type, "?")} | '
                  f'{pp_map.get(s.pivot_position, "?")} | {s.buy_confidence:.0%} | '
                  f'{s.composite_score:+.2f} |')
            w()

        # ─── 5. 风险预警 ───
        w('## 五、风险预警')
        w()
        risk_labels = {
            'confirmed_sell': '多级别确认卖点',
            'S1_top_reversal': 'S1 一卖（顶部反转）',
            'S3_breakdown': 'S3 三卖（趋势破位）',
            'top_divergence': '顶背驰预警',
            'ma_breakdown': '均线破位（≥2条）',
        }
        for key, label in risk_labels.items():
            snaps = report.risk_alerts.get(key, [])
            if not snaps:
                continue
            w(f'### {label} ({len(snaps)}只)')
            w(f'| 代码 | 名称 | 行业 | 收盘 | 涨跌 | 走势 | 中枢位 | 置信 | 综合 |')
            w(f'|------|------|------|------|------|------|--------|------|------|')
            pp_map = {1: '上方', 0: '内部', -1: '下方'}
            for s in snaps[:15]:
                conf = s.sell_confidence
                w(f'| {s.symbol} | {s.name} | {s.industry} | {s.close:.2f} | '
                  f'{s.change_pct:+.1f}% | {trend_map.get(s.trend_type, "?")} | '
                  f'{pp_map.get(s.pivot_position, "?")} | {conf:.0%} | '
                  f'{s.composite_score:+.2f} |')
            w()

        # ─── 6. 作战计划 ───
        w('## 六、明日作战计划')
        w()
        if report.portfolio_suggestions:
            if m.sentiment_cycle in ('冰点', '退潮'):
                pos_advice = '防御仓位 (1-3成)'
            elif m.sentiment_cycle == '高潮':
                pos_advice = '积极仓位 (5-7成)'
            elif m.bear_risk:
                pos_advice = '轻仓或空仓 (0-2成)'
            else:
                pos_advice = '中性仓位 (3-5成)'
            w(f'**仓位建议**: {pos_advice}')
            w()
            w(f'| # | 代码 | 名称 | 行业 | 收盘 | 买点类型 | 止损 | 综合 | 等级 |')
            w(f'|---|------|------|------|------|----------|------|------|------|')
            for i, d in enumerate(report.portfolio_suggestions):
                bp = {1: 'B1', 2: 'B2', 3: 'B3'}.get(d.buy_point, '')
                if d.second_buy:
                    bp = 'B2*'
                w(f'| {i+1} | {d.symbol} | {d.name} | {d.industry} | {d.close:.2f} | '
                  f'{bp} | {d.stop_loss_price:.2f} | {d.composite_score:+.2f} | {d.opportunity_rank} |')
        else:
            w('无符合条件的推荐标的')
        w()

        # ─── 7. 系统进化 ───
        if report.evolution_patches:
            w('## 七、系统自进化建议')
            w()
            for p in report.evolution_patches:
                urgency_emoji = {'urgent': '🔴', 'high': '🟠', 'medium': '🟡', 'low': '🟢'}
                w(f'- {urgency_emoji.get(p.urgency, "⚪")} **[{p.section}.{p.key}]**')
                w(f'  - 当前值: `{p.current_value}` → 建议值: `{p.suggested_value}`')
                w(f'  - 理由: {p.reason}')
                w(f'  - 置信度: {p.confidence:.0%}')
            w()

        # ─── 摘要 ───
        w('## 复盘摘要')
        w()
        w(f'| 指标 | 数值 | 指标 | 数值 |')
        w(f'|------|------|------|------|')
        w(f'| 总股票数 | {summary["total_stocks"]} | 上涨趋势 | {summary["up_trend"]} |')
        w(f'| 盘整 | {summary["consolidation"]} | 下跌趋势 | {summary["down_trend"]} |')
        w(f'| 买点信号 | {summary["buy_signals"]} | 卖点信号 | {summary["sell_signals"]} |')
        w(f'| A级机会 | {summary["a_rank"]} | B级机会 | {summary["b_rank"]} |')
        w(f'| 双级别确认买 | {summary["confirmed_buy"]} | 二买确认 | {summary["second_buy"]} |')
        w(f'| 高危股票 | {summary["high_risk"]} | 涨停股 | {report.limit_up_summary.total_limit_up if report.limit_up_summary else 0} |')
        w()

        w('---')
        w('*报告由量化系统自动生成 | 缠论复盘引擎*')

        with open(path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

    # ========== 系统自进化 ==========

    def _apply_evolution(self, patches: List[EvolutionPatch], date: str):
        """应用进化补丁 — 更新配置文件和进化历史"""
        # 加载进化历史
        history = []
        if os.path.exists(EVOLUTION_LOG):
            try:
                with open(EVOLUTION_LOG, 'r') as f:
                    history = json.load(f)
            except Exception:
                history = []

        # 记录本次进化
        for p in patches:
            history.append({
                'date': date,
                'section': p.section,
                'key': p.key,
                'from': p.current_value,
                'to': p.suggested_value,
                'reason': p.reason,
                'confidence': p.confidence,
                'urgency': p.urgency,
                'applied': False,  # 需人工确认后才应用
            })

        with open(EVOLUTION_LOG, 'w') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

        # 打印需要手动修改的配置建议
        urgent_patches = [p for p in patches if p.urgency in ('urgent', 'high')]
        if urgent_patches:
            print(f'\n  ╔{"═"*74}╗')
            print(f'  ║  需要立即关注的进化建议 ({len(urgent_patches)} 条){" " * 36}║')
            print(f'  ╠{"═"*74}╣')
            for p in urgent_patches:
                print(f'  ║  文件: strategy/config/factor_config.yaml{" " * 38}║')
                print(f'  ║  路径: {p.section}.{p.key}{" " * (64 - len(p.section) - len(p.key))}║')
                print(f'  ║  {p.current_value} → {p.suggested_value}  (置信度 {p.confidence:.0%}){" " * (46 - len(str(p.current_value)) - len(str(p.suggested_value)))}║')
                print(f'  ║  理由: {p.reason[:62]:<62s}║')
                print(f'  ║{"─"*74}║')
            print(f'  ║  进化历史已保存至: evolution_history.json{" " * 42}║')
            print(f'  ╚{"═"*74}╝')


# ==================== 入口 ====================

def main():
    parser = argparse.ArgumentParser(description='量化系统每日复盘 & 自我进化引擎')
    parser.add_argument('--date', type=str, default=None, help='复盘日期 (YYYY-MM-DD)')
    parser.add_argument('--top', type=int, default=15, help='显示 top N (默认15)')
    parser.add_argument('--symbols', type=str, default=None, help='指定股票，逗号分隔')
    parser.add_argument('--min-volume', type=float, default=None, help='最小成交额过滤')
    parser.add_argument('--evolve', action='store_true', default=False,
                        help='启用系统自进化（生成配置补丁）')
    parser.add_argument('--output', type=str, default=None, help='输出目录')

    args = parser.parse_args()

    symbols = None
    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(',')]

    engine = QuantReviewEngine(min_volume=args.min_volume)
    if args.output:
        engine.output_dir = args.output

    engine.run(date=args.date, top_n=args.top, symbols=symbols, evolve=args.evolve)


if __name__ == '__main__':
    main()
