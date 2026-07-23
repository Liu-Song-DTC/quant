# core/portfolio.py
"""
组合构建器 - 基于因子排名选股

设计原则:
1. 等权top N: 截面rank_pct>0.5的top N只, 等权分配
2. 行业均衡: 每个行业最多5只, 避免过度集中
3. 最小风控: 仅深度回撤和fv_mean降仓
4. 非再平衡日: 执行成本止损+缠论止盈(S1/S2/S3)+均值回归退出，不执行信号止损
5. 再平衡日: 执行完整选股+所有止损机制(含组合止损)
"""
import numpy as np
from collections import deque
from copy import deepcopy
from datetime import date as date_type
from .config_loader import load_config
from .pipeline_logger import plog
from scipy.special import erfinv, erf
import yaml
import os


def _load_industry_ic_weights():
    """加载行业IC权重，用于因子值惩罚（包含分市场IC）"""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base_dir, 'config', 'factor_config.yaml')
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            industry_factors = config.get('industry_factors', {})
            result = {}
            for industry, cfg in industry_factors.items():
                result[industry] = {
                    'neutral': max(cfg.get('ic', 0.05), 0.01),
                    'bull': max(cfg.get('bull_ic', cfg.get('combined_ic', cfg.get('ic', 0.05))), 0.01),
                    'bear': max(cfg.get('bear_ic', cfg.get('bear_combined_ic', cfg.get('ic', 0.05))), 0.01),
                }
            return result
    return {}


def _rank_to_normal(rank_pct, clip=0.001):
    """Quantile→Normal变换：将[0,1]线性排名映射到N(0,1)，拉大尾部差距。

    线性排名: 第1名-第2名 ≈ 第50名-第51名（差距均为 1/N）
    正态变换后: 第1名远高于第2名（尾部被拉伸），中段差距不变

    变换是可逆的: _normal_to_rank(erfinv(2p-1)) = p
    """
    clipped = np.clip(rank_pct, clip, 1.0 - clip)
    return np.sqrt(2.0) * erfinv(2.0 * clipped - 1.0)


def _normal_to_rank(normal_score):
    """逆变换：N(0,1) → [0,1] rank"""
    return (erf(normal_score / np.sqrt(2.0)) + 1.0) / 2.0


class PortfolioConstructor:
    """仓位管理器 - 基于因子排名选股"""

    @staticmethod
    def _nan_safe(val, default=0.0):
        """NaN-safe value extraction: NaN is truthy so `val or default` fails."""
        if val is None:
            return default
        if isinstance(val, float) and np.isnan(val):
            return default
        return val

    def __init__(
        self,
        target_volatility=None,
        entry_speed=None,
        exit_speed=None,
        position_stop_loss=None,
        portfolio_stop_loss=None,
    ):
        config = load_config()
        portfolio_config = config.get_portfolio_config()

        self.position_stop_loss = position_stop_loss if position_stop_loss is not None else portfolio_config.get('position_stop_loss', 0.12)
        self.entry_speed = entry_speed if entry_speed is not None else portfolio_config.get('entry_speed', 0.5)
        self.exit_speed = exit_speed if exit_speed is not None else portfolio_config.get('exit_speed', 0.5)
        self.exit_mode = 'simple'  # 仅成本止损, 让利润奔跑, 保持高资金利用率

        # === 波动率控制 ===
        self.vol_control_enabled = portfolio_config.get('volatility_control_enabled', True)
        self.target_volatility = target_volatility if target_volatility is not None else portfolio_config.get('target_volatility', 0.15)
        self.vol_lookback = portfolio_config.get('volatility_control_lookback', 20)
        self._daily_returns = deque(maxlen=self.vol_lookback * 2)  # store up to 2x for safety

        # === 组合止损 ===
        ps_top = config.get('portfolio_stop_loss', {}) if hasattr(config, 'get') else {}
        self.stop_loss_enabled = ps_top.get('enabled', True)
        self.portfolio_stop_loss = portfolio_stop_loss if portfolio_stop_loss is not None else ps_top.get('trigger_drawdown', 0.10)
        self.base_exposure = portfolio_config.get('base_exposure', 0.85)
        self.emergency_exposure = ps_top.get('emergency_exposure', 0.65)
        self.stop_loss_recovery_days = ps_top.get('recovery_days', 10)
        self.stop_loss_refill = ps_top.get('stop_refill', False)

        # === 单票权重上限 ===
        self.max_single_weight = portfolio_config.get('max_single_weight', 0.12)

        # === 绝对质量门槛 ===
        sel_config = portfolio_config.get('selection', {})
        self.min_rank_pct = sel_config.get('min_rank_pct', 0.5)
        self.min_absolute_score = sel_config.get('min_absolute_score', 0.15)
        self.min_confidence = sel_config.get('min_confidence', 0.80)
        self.exhaustion_max_weight = sel_config.get('exhaustion_max_weight', 0.03)
        self.exhaustion_reduce_mult = sel_config.get('exhaustion_reduce_mult', 0.5)
        self.industry_max_weight = sel_config.get('industry_max_weight', 0.35)
        self.max_per_industry = sel_config.get('max_per_industry', 3)  # ExpB2: 同行业最多N只
        self.max_single_weight_from_cfg = sel_config.get('rank_weight_cap',
                                              sel_config.get('max_single_weight', 0.18))
        self._orig_max_single_weight = self.max_single_weight_from_cfg  # CLB恢复时还原

        # === fv_exposure 可配置参数 ===
        fv_params = portfolio_config.get('fv_exposure_params', {})
        self.fv_low = fv_params.get('fv_low', -0.03)
        self.fv_high = fv_params.get('fv_high', 0.05)
        self.fv_exposure_min = fv_params.get('exposure_min', 0.3)
        self.fv_exposure_max = fv_params.get('exposure_max', 1.0)

        # === 换手惩罚 ===
        self.turnover_bonus = portfolio_config.get('turnover_bonus', 0.04)
        self.max_turnover_ratio = portfolio_config.get('max_turnover_ratio', 0.60)
        self.min_hold_days = portfolio_config.get('min_hold_days', 5)

        # === 累计入选追踪：老朋友优先 ===
        self._selection_history = {}  # {code: count}
        self._selection_history_max_age = 6  # 最多追溯6次调仓
        self._position_entry_dates = {}  # {code: date_str} 记录入场时间

        # === 风险平价 ===
        rp_config = config.get('risk_parity', {}) if hasattr(config, 'get') else portfolio_config.get('risk_parity', {})
        self.risk_parity_enabled = rp_config.get('enabled', False)  # Fix#13: 若启用风险平价，应去掉手动权重上限，让算法自由分配
        self.rp_target_risk = rp_config.get('target_risk_per_stock', 0.10)
        self.rp_max_iterations = rp_config.get('max_iterations', 50)

        # === 增强止损 ===
        es_config = config.get('enhanced_stop_loss', {}) if hasattr(config, 'get') else portfolio_config.get('enhanced_stop_loss', {})
        self.time_stop_days = es_config.get('time_stop_days', 20)
        self.time_stop_min_return = es_config.get('time_stop_min_return', -0.02)
        self.trailing_stop_pct = es_config.get('trailing_stop_pct', 0.15)
        self.trailing_stop_enabled = es_config.get('trailing_stop_enabled', True)
        self.volatility_adaptive_mult = es_config.get('volatility_adaptive_mult', 1.3)
        self.max_adaptive_stop_mult = es_config.get('max_adaptive_stop_mult', 1.5)
        # 分级移动止盈: {profit_pct: trail_pct}
        self.tiered_trailing_stop = es_config.get('tiered_trailing_stop', {
            0.15: 0.10, 0.30: 0.07, 0.50: 0.04
        })
        self.dead_money_exit_days = es_config.get('dead_money_exit_days', 30)
        self.dead_money_return_band = es_config.get('dead_money_return_band', 0.02)
        # 按买点类型分层时间止损
        self.time_stop_by_bp = es_config.get('time_stop_by_buy_point', {
            1: (40, -0.08), 2: (30, -0.05), 3: (20, -0.05)
        })
        # 按买点类型分层均线乖离移动止损
        self.trailing_stop_by_bp = es_config.get('trailing_stop_by_buy_point', {
            1: 0.15, 2: 0.12, 3: 0.10, 'default': 0.08
        })

        # === 均值回归退出 ===
        mr_config = config.get('mean_reversion_exit', {}) if hasattr(config, 'get') else {}
        self.mr_mom60_force = mr_config.get('mom_60d_force_exit', 0.60)
        self.mr_dist60_force = mr_config.get('dist_ma60_force_exit', 0.50)
        self.mr_mom60_reduce = mr_config.get('mom_60d_reduce', 0.40)
        self.mr_dist60_reduce = mr_config.get('dist_ma60_reduce', 0.35)
        self.mr_reduce_pct = mr_config.get('reduce_pct', 0.3)

        # === 组合可调参数（从YAML加载，替代硬编码魔数） ===
        pp = portfolio_config.get('params', {})
        self.max_positions = pp.get('max_positions', 8)
        self.rp_min_weight_ratio = pp.get('risk_parity_min_weight_ratio', 0.5)
        self.rank_decay = pp.get('rank_decay', 0.3)
        self.chan_bonus_sl2 = pp.get('chan_bonus_sl2', 0.04)
        self.chan_bonus_buy_point = pp.get('chan_bonus_buy_point', 0.02)
        self.chan_bonus_trend2 = pp.get('chan_bonus_trend2', 0.01)
        self.mom_60d_fomo_threshold = pp.get('mom_60d_fomo_threshold', 0.30)
        self.mom_60d_fomo_mult = pp.get('mom_60d_fomo_mult', 0.50)  # 底线从0.30→0.50
        self.mom_60d_warn_threshold = pp.get('mom_60d_warn_threshold', 0.20)
        self.mom_60d_warn_mult = pp.get('mom_60d_warn_mult', 0.65)
        self.dist_ma60_extended_threshold = pp.get('dist_ma60_extended_threshold', 0.30)
        self.dist_ma60_extended_mult = pp.get('dist_ma60_extended_mult', 0.40)
        self.isolated_b3_penalty = pp.get('isolated_b3_penalty', -0.08)
        self.b3_vol_severe_penalty = pp.get('b3_vol_severe_penalty', -0.06)
        self.exhaustion_high_threshold = pp.get('exhaustion_high_threshold', 0.30)
        self.exhaustion_moderate_threshold = pp.get('exhaustion_moderate_threshold', 0.15)
        self.sector_momentum_weight = pp.get('sector_momentum_weight', 0.40)
        self.sector_signal_density_weight = pp.get('sector_signal_density_weight', 0.60)
        self.mr_exit_cooldown_days = pp.get('mr_exit_cooldown_days', 14)
        self.ideal_position_max_mult = pp.get('ideal_position_max_mult', 2.0)
        self._min_hold_days = pp.get('min_hold_days', 5)

        # === 缠论止盈配置 ===
        chan_config = config.get('chan_theory', {}) if hasattr(config, 'get') else {}
        tp = chan_config.get('take_profit', {})
        self.chan_tp_enabled = tp.get('enabled', True)
        self.chan_tp_s1_reduce = tp.get('s1_reduce_pct', 1.0)      # S1→全清
        self.chan_tp_s2_reduce = tp.get('s2_reduce_pct', 0.5)      # S2→减半
        self.chan_tp_s3_reduce = tp.get('s3_reduce_pct', 1.0)      # S3→全清
        self.chan_tp_pivot_trailing = tp.get('pivot_trailing_enabled', True)
        self.chan_tp_profit_lock = tp.get('profit_lock_pct', 0.20)  # Fix#11: 从15%提高到20% # 盈利>15%启动移动止盈
        self.chan_tp_bi_sell_exit = tp.get('bi_sell_exit', True)

        # 追踪每个持仓的入场时间和峰值价格
        self._entry_dates: dict = {}  # {code: date}
        self._peak_prices: dict = {}  # {code: peak_price_since_entry}
        self._entry_reasons: dict = {}  # {code: {buy_point, signal_level, trend_type}}
        self._mr_exit_cooldown: dict = {}  # {code: exit_date} 均值回归退出冷却期(Fix#7)
        self._entry_reason_lost_count: dict = {}  # Fix#10: 买入理由消失确认期计数
        self._post_sell_tracking: dict = {}  # {code: {'trigger_price': float, 'reason': str}}

        self.peak_equity = None
        self.last_selection = []  # 每期选股结果
        self.current_ranking = {}
        self.current_n_positions = 0
        self.current_exposure = 1.0
        self._stop_loss_triggered = False
        self._stop_loss_recovery_days = 0
        self.industry_ic = _load_industry_ic_weights()
        self.position_cost = {}
        self.sentiment_multipliers: dict = {}
        self._sector_rotation = None  # 板块轮动分析器

        # === 多策略框架 ===
        from .multi_strategy import MultiStrategyWeights
        self._multi_strategy = MultiStrategyWeights()

        # === 硬最大回撤停止 ===
        hds_config = config.get('hard_drawdown_stop', {}) if hasattr(config, 'get') else {}
        self.hds_enabled = hds_config.get('enabled', True)
        self.hds_trigger = hds_config.get('trigger_drawdown', 0.20)
        self.hds_recovery = hds_config.get('recovery_threshold', 0.10)
        self.hds_close_days = hds_config.get('close_positions_days', 3)
        self._hds_triggered = False
        self._hds_close_day = 0

        # === 连续亏损熔断 ===
        clb_config = config.get('consecutive_loss_breaker', {}) if hasattr(config, 'get') else {}
        self.clb_enabled = clb_config.get('enabled', True)
        self.clb_threshold = clb_config.get('threshold', 3)
        self.clb_loss_floor = clb_config.get('loss_floor', -0.005)
        self.clb_win_floor = clb_config.get('win_floor', 0.005)
        self.clb_exposure_reduction = clb_config.get('exposure_reduction', 0.50)  # 减半, 不归零(防止无法恢复)
        self.clb_reset_after_win = clb_config.get('reset_after_win', 2)
        self._clb_losses = 0
        self._clb_wins = 0
        self._clb_triggered = False
        self._prev_equity = None

        # 另类数据: 懒加载, 用于调整市场暴露度
        self._alt_provider = None

    def _get_smart_money_signal(self, date):
        """北向+融资融券复合信号: 大幅流出/去杠杆→防御, 流入→积极"""
        if self._alt_provider is None:
            try:
                from .alternative_data import AlternativeDataProvider
                self._alt_provider = AlternativeDataProvider()
                self._alt_provider.load_northbound()
                self._alt_provider.load_margin()
            except Exception:
                return 0.0
        nb = self._alt_provider.get_northbound_signal(date)
        mg = self._alt_provider.get_margin_signal(date)
        return nb * 0.6 + mg * 0.4  # 北向主导, 融资辅助

    def set_sector_rotation(self, sr):
        """注入板块轮动分析器（含动量+信号密度）"""
        self._sector_rotation = sr

    def save_tracking_state(self, filepath: str):
        """持久化持仓跟踪状态到 JSON 文件（实盘跨日状态保持）"""
        import json as _json
        from datetime import date as _date

        def _serialize(obj):
            if isinstance(obj, _date):
                return obj.isoformat()
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, np.bool_):
                return bool(obj)
            raise TypeError(f"Unserializable type: {type(obj)}")

        state = {
            "entry_dates": {k: v.isoformat() if hasattr(v, 'isoformat') else str(v)
                          for k, v in self._entry_dates.items()},
            "peak_prices": self._peak_prices,
            "entry_reasons": self._entry_reasons,
            "mr_exit_cooldown": {k: v.isoformat() if hasattr(v, 'isoformat') else str(v)
                               for k, v in self._mr_exit_cooldown.items()},
            "entry_reason_lost_count": self._entry_reason_lost_count,
            "post_sell_tracking": self._post_sell_tracking,
            "current_exposure": self.current_exposure,
            "stop_loss_triggered": self._stop_loss_triggered,
            "stop_loss_recovery_days": self._stop_loss_recovery_days,
            "peak_equity": self.peak_equity,
            # 硬回撤停止
            "hds_triggered": self._hds_triggered,
            "hds_close_day": self._hds_close_day,
            "hds_peak_ref": self._hds_peak_ref,
            # 连续亏损熔断
            "clb_losses": self._clb_losses,
            "clb_wins": self._clb_wins,
            "clb_triggered": self._clb_triggered,
            "prev_equity": self._prev_equity,
            # 波动率控制
            "daily_returns": [x for x in self._daily_returns if not (x != x)],
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            _json.dump(state, f, ensure_ascii=False, indent=2, default=_serialize)

    def restore_tracking_state(self, filepath: str):
        """从 JSON 文件恢复持仓跟踪状态"""
        import json as _json
        from datetime import date as _date

        if not os.path.exists(filepath):
            return

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                state = _json.load(f)
        except Exception:
            return

        for code, ds in state.get("entry_dates", {}).items():
            try:
                self._entry_dates[code] = _date.fromisoformat(ds)
            except (ValueError, TypeError):
                pass
        self._peak_prices = state.get("peak_prices", {})
        self._entry_reasons = state.get("entry_reasons", {})
        for code, ds in state.get("mr_exit_cooldown", {}).items():
            try:
                self._mr_exit_cooldown[code] = _date.fromisoformat(ds)
            except (ValueError, TypeError):
                pass
        self._entry_reason_lost_count = state.get("entry_reason_lost_count", {})
        self._post_sell_tracking = state.get("post_sell_tracking", {})
        self.current_exposure = state.get("current_exposure", 1.0)
        self._stop_loss_triggered = state.get("stop_loss_triggered", False)
        self._stop_loss_recovery_days = state.get("stop_loss_recovery_days", 0)
        self.peak_equity = state.get("peak_equity")
        self._hds_triggered = state.get("hds_triggered", False)
        self._hds_close_day = state.get("hds_close_day", 0)
        self._hds_peak_ref = state.get("hds_peak_ref")
        self._clb_losses = state.get("clb_losses", 0)
        self._clb_wins = state.get("clb_wins", 0)
        self._clb_triggered = state.get("clb_triggered", False)
        self._prev_equity = state.get("prev_equity")
        dr = state.get("daily_returns", [])
        if dr:
            from collections import deque
            self._daily_returns = deque(dr, maxlen=self.vol_lookback * 2)

    def set_sentiment_multipliers(self, multipliers: dict):
        """设置行业情绪乘数（含小说逆向投资调整）

        小说核心："别人恐惧我贪婪，别人贪婪我恐惧"
        - 极端悲观（mult < 0.95）时：逆向加仓，mult 向 1.0 方向反弹
        - 极端乐观（mult > 1.08）时：逆向减仓，mult 向 1.0 方向收敛

        Args:
            multipliers: {industry: multiplier}, 范围 [0.8, 1.2]
        """
        contrarian = {}
        for ind, mult in multipliers.items():
            # NaN guard: NaN comparisons always False, would silently store NaN
            if mult is None or (isinstance(mult, float) and np.isnan(mult)):
                contrarian[ind] = 1.0
                continue
            if mult < 0.95:
                # 情绪极度悲观 → 逆向买入机会（小说："贴吧一片哀嚎时买"）
                # mult 向 1.0 方向拉回一半，悲观时反而略微加仓
                contrarian[ind] = mult + (1.0 - mult) * 0.4
            elif mult > 1.08:
                # 情绪极度乐观 → 逆向减仓（小说："大家都看好时要警惕"）
                # mult 向 1.0 方向收敛
                contrarian[ind] = mult - (mult - 1.0) * 0.3
            else:
                contrarian[ind] = mult
        self.sentiment_multipliers = contrarian

    def update_returns(self, daily_return: float):
        """记录每日收益率（用于波动率控制）

        从 BacktraderExecution.next() 中调用，每次传入当日收益率
        """
        self._daily_returns.append(daily_return)

    def _calc_max_position(self, total_equity: float, prices: dict) -> int:
        """根据资金自动计算最大持仓数。每20000资金支持1个仓位, 范围[3, max_positions]"""
        n = int(total_equity / 20000)
        return max(3, min(n, self.max_positions))

    def _risk_parity_weights(self, selected: list, total_equity: float) -> list:
        """风险平价权重分配 (Naive Risk Parity, Fix#8)

        使用波动率倒数加权 + 迭代优化实现等风险贡献。
        假设: 对角线协方差矩阵 (忽略股票间相关性)。
        对于5-8只跨行业股票的组合，相关性影响有限。

        Args:
            selected: 选中的股票列表，每项含 'risk_vol', 'code'
            total_equity: 总权益

        Returns:
            权重列表，和为1.0
        """
        n = len(selected)
        if n == 0:
            return []
        if n == 1:
            return [1.0]

        vols = np.array([c.get('risk_vol', 0.03) for c in selected])
        vols = np.clip(vols, 0.01, 0.10)

        # 初始权重: 波动率倒数 (Naive Risk Parity)
        inv_vols = 1.0 / vols
        weights = inv_vols / inv_vols.sum()

        # 权重下限: 防止单票权重过低
        min_weight = self.rp_min_weight_ratio / n

        # 迭代优化
        for iteration in range(self.rp_max_iterations):
            risk_contrib = weights * vols
            total_risk = np.sqrt(np.sum(risk_contrib ** 2))
            if total_risk < 1e-10:
                break

            target = total_risk / n
            adj = np.clip(target / (risk_contrib + 1e-10), 0.2, 5.0)  # 限制调整幅度
            weights = weights * (0.5 + 0.5 * adj)
            weights = np.clip(weights, min_weight, 1.0)  # 保底
            weights = weights / weights.sum()

            max_dev = np.max(np.abs(risk_contrib / total_risk - 1.0 / n))
            if max_dev < 0.02:  # 2%容差
                break

        return weights.tolist()

    def _build_desired_value(
        self,
        date,
        universe,
        current_positions,
        signal_store,
        cash,
        prices,
        momentum_score=0.0,
        trend_score=0.0,
        index_volume_ratio=1.0,
        style_score=0.0,
        regime_volatility=0.0,
        market_regime=0,
        bear_risk=False,
        bear_risk_fast=False,
        severe_bear=False,
    ):
        """构建目标持仓 - 等权top N选股"""
        import pandas as pd

        total_equity = cash + sum(current_positions.values())
        n_positions = self._calc_max_position(total_equity, prices)

        # === 熊市仓位: bear_IC为负→因子反指, 空仓保本金 ===
        if bear_risk:
            return {}  # 熊市因子IC为负, 选股=选亏钱
        elif bear_risk_fast:
            n_positions = max(1, n_positions // 5)
            _eff_min_rank = max(self.min_rank_pct, 0.75)
            _eff_min_score = max(self.min_absolute_score, 0.25)
            n_positions = max(2, int(n_positions * 0.5))
            _eff_min_rank = max(self.min_rank_pct, 0.55)
            _eff_min_score = max(self.min_absolute_score, 0.10)
        else:
            _eff_min_rank = self.min_rank_pct
            _eff_min_score = self.min_absolute_score

        # 计算峰值和回撤
        if self.peak_equity is None:
            self.peak_equity = total_equity
        self.peak_equity = max(self.peak_equity, total_equity)
        drawdown = 1 - total_equity / self.peak_equity if self.peak_equity > 0 else 0.0

        # === 收集候选股票 ===
        candidates = []
        # 过滤拒绝原因统计
        _rej = {'no_sig': 0, 'not_buy': 0, 'cooldown': 0, 'bad_factor': 0,
                'no_price': 0, 'too_expensive': 0, 'accepted': 0}
        for code in universe:
            sig = signal_store.get(code, date)
            if sig is None:
                _rej['no_sig'] += 1
                continue

            # 信号层过滤: 必须通过买入信号检查（含MA20趋势过滤等）
            if not getattr(sig, 'buy', False):
                _rej['not_buy'] += 1
                continue

            # Fix#7: 均值回归退出冷却期 — 防止入场-出场循环
            if code in self._mr_exit_cooldown:
                exit_date = self._mr_exit_cooldown[code]
                if isinstance(date, date_type) and isinstance(exit_date, date_type):
                    days_since = (date - exit_date).days
                else:
                    days_since = 999
                if days_since < self.mr_exit_cooldown_days:
                    _rej['cooldown'] += 1
                    continue
                else:
                    del self._mr_exit_cooldown[code]  # 冷却期满,清除记录

            factor_value = getattr(sig, 'factor_value', None)
            if factor_value is None or (isinstance(factor_value, float) and np.isnan(factor_value)):
                _rej['bad_factor'] += 1
                continue

            # 价格约束：跳过无价格数据或价格异常的股票
            price = prices.get(code, 0)
            if price <= 0:
                _rej['no_price'] += 1
                continue
            # 100股整手下，100*price必须不超过理想仓位
            # 允许2倍理想仓位（100股整手会导致超配，但至少能买入）
            ideal_per_stock = total_equity / n_positions
            if price * 100 > ideal_per_stock * self.ideal_position_max_mult:
                _rej['too_expensive'] += 1
                continue

            # === Chan 信号数据 (融合核心) ===
            sl = getattr(sig, 'signal_level', 0)
            chan_buy = getattr(sig, 'chan_buy_point', 0)
            chan_sell = getattr(sig, 'chan_sell_point', 0)
            buy_strength = getattr(sig, 'chan_divergence_strength', 0.0)
            stock_trend = getattr(sig, 'trend_type', 0)
            fn = getattr(sig, 'factor_name', '')
            daily_ret = getattr(sig, 'daily_return', 0.0)

            # Gate 统一过滤：cha_sell/trend/B3/放量下跌 均由 4-Gate 系统综合评分
            # 组合层只设 gate_quality 最低线，不逐项判断

            # 软惩罚：无结构/追高 → 有效得分扣分（不直接拒绝）
            no_chan_penalty = 0.0
            if code not in current_positions:
                div_type = getattr(sig, 'chan_divergence_type', '')
                div_strength = getattr(sig, 'chan_divergence_strength', 0.0)
                has_chan = (
                    chan_buy > 0 or
                    sl >= 1 or
                    (div_type in ('bottom', 'bottom_fx', 'bottom_fx_3x', 'B2') and div_strength > 0.2)
                )
                if not has_chan:
                    no_chan_penalty = -0.20
                # P3修复: 动量不再硬惩罚, 仅极端超买(>60%)提示风险
                mom_60d = self._nan_safe(getattr(sig, 'mom_60d', 0.0))
                dist_ma60 = self._nan_safe(getattr(sig, 'dist_ma60', 0.0))
                if mom_60d > 0.60:
                    no_chan_penalty = min(no_chan_penalty, -0.08)
                if dist_ma60 > 0.50:
                    no_chan_penalty = min(no_chan_penalty, -0.15)

            candidates.append({
                'code': code,
                'factor_value': factor_value,
                'score': getattr(sig, 'score', 0),
                'industry': getattr(sig, 'industry', '') or 'default',
                'risk_vol': getattr(sig, 'risk_vol', 0.03),
                'price': price,
                'sig': sig,
                # Chan 融合字段
                'signal_level': sl,
                'chan_buy_point': chan_buy,
                'chan_sell_point': chan_sell,
                'chan_buy_strength': buy_strength if sl > 0 else 0.0,
                'trend_type': stock_trend,
                'no_chan_penalty': no_chan_penalty,
            })
            _rej['accepted'] += 1

        if not candidates:
            for k, v in _rej.items():
                if k != 'accepted' and v > 0:
                    plog.log_portfolio_candidate(k)
            print(f" [选股] 无候选股票通过初筛 (universe={len(universe)})")
            print(f"   拒绝明细: no_sig={_rej['no_sig']} not_buy={_rej['not_buy']} "
                  f"bad_factor={_rej['bad_factor']} no_price={_rej['no_price']} "
                  f"too_expensive={_rej['too_expensive']} cooldown={_rej['cooldown']}")
            plog.alert(f"portfolio: 0 candidates from {len(universe)} universe at {date}")
            plog.log_selection_context(date, 0, len(universe), 0, 0)
            return {}

        # === 产业链聚焦: 找到主导行业, 限定在产业链内选股 ===
        _chain_concepts = None
        _dom_industry = None
        if len(candidates) >= 3:
            from .industry_chain import get_chain_concepts, _NO_CHAIN_CONCEPTS
            # 统计各行业候选数量和中位数得分
            _ind_stats = {}
            for c in candidates:
                ind = c.get('industry', '')
                if ind not in _ind_stats:
                    _ind_stats[ind] = {'n': 0, 'scores': []}
                _ind_stats[ind]['n'] += 1
                _ind_stats[ind]['scores'].append(c['score'])
            # 找主导行业: 候选数≥2 且中位数分最高
            _best_score = -999
            for ind, s in _ind_stats.items():
                if s['n'] >= 2:
                    _med = np.median(s['scores'])
                    if _med > _best_score:
                        _best_score = _med
                        _dom_industry = ind
            if _dom_industry:
                _chain_concepts = get_chain_concepts(_dom_industry)
                if _chain_concepts:
                    _before = len(candidates)
                    _chained = [c for c in candidates
                                if c.get('industry', '') in _chain_concepts
                                or c.get('industry', '') == _dom_industry]
                    # 保护: 产业链过滤后候选<3只 → 退回不限行业, 避免过度收缩
                    if len(_chained) >= 3:
                        candidates = _chained
                        print(f" [产业链] {_dom_industry} → {len(_chain_concepts)}个关联概念, "
                              f"候选 {_before}→{len(candidates)}")
                    else:
                        print(f" [产业链] {_dom_industry} 过滤后仅{len(_chained)}只(<3) → 退回不限行业")
                else:
                    if _dom_industry not in _NO_CHAIN_CONCEPTS:
                        plog.alert(f"产业链缺失: \"{_dom_industry}\" 未在 INDUSTRY_CHAINS 中定义, 请补充")
                        print(f" [产业链] ⚠ \"{_dom_industry}\" 缺少产业链定义 → 退回不限行业")
            else:
                plog.alert(f"产业链: 候选分散, 无主导行业(所有行业均<2只), 退回不限行业")

        # === 截面排名（全市场 + 行业内混合，正态空间） ===
        # Quantile→Normal变换: 拉大尾部差距，极端值 vs 普通值的区分更显著
        # 使用 score(含gate_quality乘法+基本面+ML)替代纯factor_value, gate信息参与初筛
        factor_values = np.array([c['score'] for c in candidates])
        rank_pct = pd.Series(factor_values).rank(pct=True)
        cross_normal = _rank_to_normal(rank_pct.values)  # → N(0,1)

        # 行业内排名：同行业股票内部比较
        industry_groups: dict = {}
        for i, c in enumerate(candidates):
            ind = c.get('industry', 'default') or 'default'
            if ind not in industry_groups:
                industry_groups[ind] = []
            industry_groups[ind].append((i, c['score']))

        within_normal = np.zeros(len(candidates))  # N(0,1)空间，默认0=中位数
        for ind, members in industry_groups.items():
            if len(members) >= 3:
                indices, vals = zip(*members)
                ind_rank = pd.Series(np.array(vals)).rank(pct=True)
                ind_normal = _rank_to_normal(ind_rank.values)
                for j, idx in enumerate(indices):
                    within_normal[idx] = ind_normal[j]

        # 在正态空间混合: 70%全市场 + 30%行业内，再映射回[0,1]
        blended_normal = 0.7 * cross_normal + 0.3 * within_normal
        blended_rank = _normal_to_rank(blended_normal)

        for i, c in enumerate(candidates):
            c['rank_pct'] = float(blended_rank[i])

        # R6: 关闭行业强势过滤 — 纯score驱动选股，不做行业干预
        force_exit_industries = set()
        self._force_exit_industries = force_exit_industries

        # === 市场仓位: 趋势+熊市双维度 ===
        # 2024实证: trend<0无bear_risk时fwd20=+10.8%(牛市回调反弹,应抄底)
        # 2022实证: trend<0有bear_risk时因子IC为负(必须空仓)
        if trend_score >= 0.5:
            target_exposure = self.base_exposure           # 牛市满仓
        elif trend_score > 0:
            target_exposure = 0.50 * self.base_exposure    # 弱牛半仓
        elif trend_score == 0:
            target_exposure = 0.30 * self.base_exposure    # 震荡轻仓
        else:
            # 趋势为负: 有bear_risk→空仓, 无bear_risk→轻仓抄底回调
            target_exposure = 0.0 if bear_risk else 0.30 * self.base_exposure

        # bear_risk双确认→覆盖, 确保极端情况空仓
        if bear_risk:
            target_exposure = 0.0

        # === 市场缩量降仓: 无量无行情(放松底线) ===
        if index_volume_ratio < 0.5:
            target_exposure = min(target_exposure, 0.35)
        elif index_volume_ratio < 0.7:
            target_exposure = min(target_exposure, 0.50)

        # === Chan强买点熊市豁免: >=2只强买点出现时提高敞口, 但熊市风险下限制豁免力度 ===
        chan_strong_buys = sum(
            1 for c in candidates
            if c.get('signal_level', 0) >= 2 or c.get('chan_buy_point', 0) == 1
        )
        if chan_strong_buys >= 2 and target_exposure < 0.6:
            if bear_risk:
                chan_floor = 0.15  # 熊市保守, 不超bear_cap太多
            elif bear_risk_fast:
                chan_floor = 0.45  # 原逻辑
            else:
                chan_floor = 0.60  # 正常: 不错过底部
            target_exposure = max(target_exposure, chan_floor)

        # === 波动率控制: realized_vol > target → 降仓 ===
        if self.vol_control_enabled and len(self._daily_returns) >= self.vol_lookback:
            recent_rets = list(self._daily_returns)[-self.vol_lookback:]
            realized_vol = float(np.std(recent_rets) * np.sqrt(252))
            if realized_vol > 0.01:
                vol_scale = float(np.clip(self.target_volatility / realized_vol, 0.75, 1.0))
                target_exposure *= vol_scale

        # === 硬止损: 回撤超过阈值时强制降仓（含恢复冷却期）===
        if self.stop_loss_enabled and drawdown > self.portfolio_stop_loss:
            target_exposure = min(target_exposure, self.emergency_exposure)
            if not self._stop_loss_triggered:
                self._stop_loss_triggered = True
                self._stop_loss_recovery_days = self.stop_loss_recovery_days
                # 重置峰值: 全清仓后从当前净值开始新的drawdown计算, 避免立即再次触发
                if self.emergency_exposure < 0.01:
                    self.peak_equity = total_equity
        elif self._stop_loss_triggered and self._stop_loss_recovery_days > 0:
            # 恢复期：逐步提升敞口，防止whipsaw
            target_exposure = min(target_exposure, self.emergency_exposure +
                                  (1.0 - self.emergency_exposure) * (1.0 - self._stop_loss_recovery_days / self.stop_loss_recovery_days))
            self._stop_loss_recovery_days -= 1
            if self._stop_loss_recovery_days <= 0:
                self._stop_loss_triggered = False

        # === 连续亏损熔断：连亏→减半敞口 → 连盈→恢复（每日检查，非仅调仓日）===
        if self.clb_enabled:
            if self._prev_equity is not None:
                period_ret = (total_equity - self._prev_equity) / (self._prev_equity + 1e-10)
                if period_ret < self.clb_loss_floor:
                    self._clb_losses += 1
                    self._clb_wins = 0
                elif period_ret > self.clb_win_floor:
                    self._clb_wins += 1
                    self._clb_losses = 0
                    if self._clb_triggered and self._clb_wins >= self.clb_reset_after_win:
                        self._clb_triggered = False
                        self.max_single_weight_from_cfg = self._orig_max_single_weight
                else:
                    # 中性日（介于loss_floor和win_floor之间）：重置连亏计数
                    self._clb_losses = 0
                if self._clb_losses >= self.clb_threshold:
                    self._clb_triggered = True
                if self._clb_triggered:
                    target_exposure *= self.clb_exposure_reduction
                    self.max_single_weight_from_cfg = self._orig_max_single_weight * self.clb_exposure_reduction
            self._prev_equity = total_equity

        # 滞后平滑: 避免仓位突变（降低平滑系数，更快响应）
        self.current_exposure = 0.3 * self.current_exposure + 0.7 * target_exposure

        # === 选股: 绝对质量门槛 + 截面排名 ===
        # 两步过滤：1) 绝对分数门槛 2) 截面排名门槛
        # 熊市自动收紧(_eff_min_rank/_eff_min_score已在上面根据bear_risk设定)
        qualified = [
            c for c in candidates
            if c['rank_pct'] > _eff_min_rank
            and c.get('score', 0) >= _eff_min_score
        ]

        # 信心过滤: V因子兜底加成
        for c in qualified:
            sig = c['sig']
            confidence = 1.0
            fn = getattr(sig, 'factor_name', '')
            if '_V' in (fn or ''):
                confidence += 0.10
            c['confidence'] = np.clip(confidence, 0.75, 1.3)

        qualified = [c for c in qualified if c['confidence'] >= self.min_confidence]

        # === Gate 统一过滤 (替代分散的 chan/趋势/量能/力竭 逐项硬拒) ===
        # 4-Gate 综合评分 (chan_structure + mtf_alignment + concept_heat + trend_direction)
        # 已覆盖原有的 sl/chan_sell/trend=-2/B3回踩/放量下跌/力竭 等全部维度
        # Gate硬门槛仅排除极端无效信号（gate通过score×gate_quality软性影响排名）
        # _GATE_DEFAULT_MEAN=0.55, 全默认gate_quality≈0.68, 需至少一个Gate有信号才能>0.7
        GATE_FLOOR_NEW = 0.65   # 提高门槛: 64%无结构买入占比过高
        GATE_FLOOR_HOLD = 0.50  # 持仓放宽: gate质量下降不急于清仓
        gate_filtered = []
        for c in qualified:
            sig = c.get('sig')
            if sig is None:
                continue
            code = c['code']
            gate_q = getattr(sig, '_gate_quality', 0.5) if sig else 0.5
            bp = c.get('chan_buy_point', 0)
            tt = getattr(sig, 'trend_type', 0) if sig else 0
            ma_up = getattr(sig, 'ma_trend_up', False) if sig else False

            if code in current_positions:
                if gate_q < GATE_FLOOR_HOLD:
                    continue
            else:
                if gate_q < GATE_FLOOR_NEW:
                    continue

            # 特定保护（Gates 无法完全覆盖的极端情况）
            if bp == 3 and not ma_up:
                continue  # B3需均线多头
            if bp == 2 and tt == -2:
                continue  # B2在下跌趋势不可靠

            # P0: gate对收益预测力IC≈0, 不再参与评分, 仅做二元安全网(通过硬门槛即可)
            gate_filtered.append(c)

        if len(gate_filtered) == 0:
            plog.alert(f"portfolio: {len(qualified)} candidates all failed Gate floor at {date}")
            print(f" [选股] Gate全部淘汰: {len(qualified)}只候选 gate_quality均<门槛, 无法入场")
            return {}

        # === 板块共振: 计算行业集中度, 用于后续排序惩罚 ===
        industry_chan_count = {}
        for c in gate_filtered:
            ind = c.get('industry', '其他')
            industry_chan_count[ind] = industry_chan_count.get(ind, 0) + 1

        qualified = gate_filtered

        current_codes = set(current_positions.keys())

        # 换手控制：现有持仓给予换手惩罚加分，需超过交易成本才能被替换
        hold_threshold = 0.2  # P2: 0.3→0.2, 更宽松的保留条件

        # 计算有效得分: 纯score(已含门控质量) 截面排名 + 时序调整
        # Gate 1-4已在信号层编码为adjusted_score，组合层只做时序/行业/换手微调
        scores = np.array([c['score'] for c in qualified])
        score_rank = pd.Series(scores).rank(pct=True).values
        pool_median_score = float(np.median(scores)) if len(scores) > 0 else 0.3

        for i, c in enumerate(qualified):
            c['is_held'] = c['code'] in current_codes

            rank = float(score_rank[i])
            multiplier = 1.0
            additive = 0.0

            # ── 乘法调整（作用于rank） ──
            sig_ref = c.get('sig')
            mom_60d = self._nan_safe(getattr(sig_ref, 'mom_60d', 0.0))
            dist_ma60 = self._nan_safe(getattr(sig_ref, 'dist_ma60', 0.0))
            vol_regime = self._nan_safe(getattr(sig_ref, 'vol_regime', 1.0))

            # P3修复: 动量不应被惩罚 — 反事实分析表明赢家动量更高(mom_60d +0.043)
            # 仅极端超买(>60%)做温和降权, 不再一刀切砍半
            if mom_60d > 0.60:
                multiplier *= 0.85  # 极端超买轻量降权
            elif mom_60d > 0.45:
                multiplier *= 0.92  # 偏高动量几乎不惩罚

            if dist_ma60 > self.dist_ma60_extended_threshold:
                multiplier *= self.dist_ma60_extended_mult

            bp = c.get('chan_buy_point', 0)
            sl = c.get('signal_level', 0)

            # === 短期趋势质量扣分 (恢复2.04的12条件, 改为扣分制保留候选资格) ===
            vol_ratio_pen = self._nan_safe(getattr(sig_ref, 'volume_ratio', 0.0))
            daily_ret_pen = self._nan_safe(getattr(sig_ref, 'daily_return', 0.0))
            trend_type_pen = int(self._nan_safe(getattr(sig_ref, 'trend_type', 0)))
            stroke_phase_pen = self._nan_safe(getattr(sig_ref, 'stroke_phase', 0.0))
            exhaustion_pen = self._nan_safe(getattr(sig_ref, 'exhaustion_risk', 0.0))
            div_type_pen = str(getattr(sig_ref, 'chan_divergence_type', '') or '')
            div_strength_pen = self._nan_safe(getattr(sig_ref, 'chan_divergence_strength', 0.0))
            gbc_pen = self._nan_safe(getattr(sig_ref, 'gap_breakout_confirm', 0.0))
            profit_d_pen = getattr(sig_ref, 'profit_declining', False)

            # 顶部背离: 2.04数据 top→-6.38%, hidden_top→-4.46%
            if 'top' in div_type_pen.lower():
                additive -= 0.08
            # 利润持续下滑: 基本面恶化
            if profit_d_pen:
                additive -= 0.10
            # 放量下跌: ret<-1.5%且vol_ratio>0.5 → 不是健康回调
            if daily_ret_pen < -0.015 and vol_ratio_pen > 0.5:
                additive -= 0.06
            # 涨停附近不追: 接盘风险
            if daily_ret_pen > 0.09:
                additive -= 0.05
            # 力竭追高: exhaustion>0.5
            if exhaustion_pen > 0.5:
                additive -= 0.10
            # B1需底背离确认
            if bp == 1 and 'bottom' not in div_type_pen.lower():
                additive -= 0.06
            # 下跌趋势中加速下行
            if trend_type_pen == -2 and stroke_phase_pen > 0.2 and not (bp == 3 and sl >= 2):
                additive -= 0.05
            # 非上升趋势+弱势阴跌
            if trend_type_pen != 2 and stroke_phase_pen > 0.2 and daily_ret_pen < 0:
                additive -= 0.04
            # 极端追高: mom>50%且无结构
            if mom_60d > 0.50 and bp == 0:
                additive -= 0.06
            # 严重乖离MA60: dist>45%且无结构
            if dist_ma60 > 0.45 and bp == 0:
                additive -= 0.06

            # B3严重缩量惩罚 (唯一保留的买点特定调整, 有数据支撑)
            if bp == 3:
                vol_ratio = self._nan_safe(getattr(sig_ref, 'volume_ratio', 0.0))
                if -0.30 < vol_ratio <= -0.20:
                    additive += self.b3_vol_severe_penalty

            # 换手保留: 已持仓有交易成本优势, 但弱于候选池中位数时减少保护
            # 用候选池score中位数作基准: 持仓强→保护, 持仓弱→放开让位
            code = c.get('code', '')
            turnover = 0.0
            if c['is_held'] and c['rank_pct'] > hold_threshold:
                held_score = c.get('score', 0.0)
                if held_score > pool_median_score * 1.2:
                    turnover = self.turnover_bonus              # 明显强于中位数 → 保留
                elif held_score > pool_median_score:
                    turnover = self.turnover_bonus * 0.5        # 略强 → 减半保护
                elif held_score > pool_median_score * 0.8:
                    turnover = self.turnover_bonus * 0.15       # 略弱 → 几乎不保护
                else:
                    turnover = 0.0                               # 远弱 → 不保护
            if c.get('chan_sell_point', 0) > 0:
                turnover = 0.0
            if c['is_held'] and c['rank_pct'] <= hold_threshold:
                c['is_held'] = False
                turnover = 0.0

            # B1买点加分: 一买是反转信号, 胜率最高
            if bp == 1 and sl >= 1:
                additive += 0.08
            elif bp == 2 and sl >= 2:
                additive += 0.03  # B2强确认小幅加分

            # BOM质量加分: 高壁垒+高利润个股优先
            bom_score = self._nan_safe(getattr(sig_ref, 'bom_quality_score', 0.3))
            if bom_score > 0.70:
                additive += 0.06
            elif bom_score > 0.55:
                additive += 0.03

            # 动量调整: 赢家动量更高(+4.3pp), 有数据支撑
            mom_adj = (mom_60d - 0.0) * 0.18

            # effective_score: 截面排名 × 乘数 + 数据驱动微调
            c['effective_score'] = rank * multiplier + additive + turnover + mom_adj + c.get('no_chan_penalty', 0.0)

        # 换手约束: 新入场数不超过 max_turnover_ratio × n_positions
        max_new = max(1, int(n_positions * self.max_turnover_ratio))
        # 持仓>=min_hold_days才允许被替换
        # 动态锁定: 新信号显著强于入场信号时可提前解锁
        for c in qualified:
            code = c.get('code', '')
            if c['is_held'] and code in self._position_entry_dates:
                held_days = (date - self._position_entry_dates[code]).days if hasattr(date, 'days') else 999
                lock_bypass = False
                if held_days < self.min_hold_days:
                    # 持仓score显著弱于候选池中位数 → 提前解锁, 允许被替换
                    held_score = c.get('score', 0.0)
                    if held_score < pool_median_score * 0.7:
                        lock_bypass = True   # 远弱于中位数 → 解锁
                c['_locked'] = (held_days < self.min_hold_days) and not lock_bypass
            else:
                c['_locked'] = False

        # 排序：按有效得分降序（已持仓有换手加分, 锁仓排最前）
        qualified.sort(key=lambda x: (-x['_locked'], -x['effective_score']))

        # === 纯score排序选股 ===
        selected = []
        for c in qualified:
            if len(selected) >= n_positions:
                break
            if c in selected:
                continue
            selected.append(c)

        # === 多策略后处理: 按子策略评分调整有效得分 → 可能重排 ===
        if self._multi_strategy.enabled and selected:
            ms_mult = self._multi_strategy.compute_weights(
                selected, signal_store, market_regime, drawdown)
            for c in selected:
                mult = ms_mult.get(c['code'], 1.0)
                if mult <= 0.05:
                    c['effective_score'] -= 10.0  # 强制排到末尾
                else:
                    c['effective_score'] *= mult
            selected.sort(key=lambda x: -x['effective_score'])
            # 移除被强制拒绝的
            selected = [c for c in selected if c['effective_score'] > -5.0]

        # 更新累计入选追踪
        for c in selected:
            code = c.get('code', '')
            self._selection_history[code] = self._selection_history.get(code, 0) + 1
        # 衰减旧记录：每6次调仓清理超过6次调仓的旧记录
        if len(selected) > 0:
            decay_keys = [k for k in self._selection_history if self._selection_history[k] > self._selection_history_max_age]
            for k in decay_keys:
                self._selection_history[k] = max(1, self._selection_history[k] - 1)

        # === 最小持仓: 至少1只，不达标时降级使用 ===
        min_pos = 1
        if len(selected) < min_pos:
            shortage_ratio = max(0.3, len(selected) / max(min_pos, 1))
            target_exposure *= shortage_ratio
            print(f" [选股] 数量不足: 选中{len(selected)}只 < 最低{min_pos}只"
                  f" → 降级使用(敞口×{shortage_ratio:.1f})")

        if not selected:
            return {}

        # 敞口低时集中火力: 减少候选数确保每只股票能买够1手
        if target_exposure < 0.50:
            max_conc = max(2, int(target_exposure * 6))
            if len(selected) > max_conc:
                selected = selected[:max_conc]

        # 记录
        self.current_ranking = {c['code']: i for i, c in enumerate(candidates)}
        self.current_n_positions = len(selected)

        # === 权重分配（rank-weighted + 风险平价）===
        n = len(selected)
        if self.risk_parity_enabled:
            # 协方差风险平价（优先）
            rp_weights = self._risk_parity_weights(selected, total_equity)
            weights = [w * target_exposure for w in rp_weights]
        elif self.industry_ic:
            # 行业IC调整的rank-weighted分配
            # 按effective_score排序后的线性衰减权重: 第一名~1.0, 最后一名~0.5
            raw_weights = []
            for i, c in enumerate(selected):
                ic_dict = self.industry_ic.get(c['industry'], {})
                ic = ic_dict.get('neutral', 0.05)
                # 排名权重: 线性衰减 1.0 → 0.5
                rank_w = 1.0 - self.rank_decay * (i / max(n - 1, 1))
                # 行业IC为权重提供差异化
                ic_mult = np.clip(ic / 0.05, 0.6, 1.5)
                # 信心调整
                conf = c.get('confidence', 1.0)
                # 情绪乘数
                sent_mult = 1.0
                if self.sentiment_multipliers and c['industry'] in self.sentiment_multipliers:
                    sent_mult = self.sentiment_multipliers[c['industry']]
                raw_weights.append(rank_w * ic_mult * conf * sent_mult)
            total_w = sum(raw_weights) + 1e-10
            weights = [w / total_w * target_exposure for w in raw_weights]
        else:
            # 排名加权: 信号强者多配（即使无IC数据也用排名衰减）
            raw_weights = []
            for i in range(n):
                rank_w = 1.0 - self.rank_decay * (i / max(n - 1, 1))
                raw_weights.append(rank_w)
            total_w = sum(raw_weights) + 1e-10
            weights = [w / total_w * target_exposure for w in raw_weights]

        # 单票权重上限: 候选不足时动态放宽以提升资金利用率
        max_single = self.max_single_weight_from_cfg
        if len(selected) < self.max_positions:
            # 候选不足 → 提高单票上限, 集中资金到优质标的
            slack = (self.max_positions - len(selected)) / self.max_positions
            max_single = min(max_single * (1.0 + slack * 1.5), max_single * 1.5)
        # 归一化：仅做比例分配，不做上限（上限在归一化后执行以避免被抵消）
        total_w = sum(weights) + 1e-10
        weights_norm = [w / total_w for w in weights]
        weights = [w * target_exposure for w in weights_norm]

        # === 单票权重上限 — 归一化后执行 ===
        for i in range(len(weights)):
            weights[i] = min(weights[i], max_single)

        # === 力竭/行业上限 — 归一化后执行 ===
        exhaustion_tags = set()
        for i, c in enumerate(selected):
            sig = c.get('sig')
            er = self._nan_safe(getattr(sig, 'exhaustion_risk', 0.0)) if sig else 0.0
            if er > self.exhaustion_high_threshold:
                weights[i] = min(weights[i], self.exhaustion_max_weight * target_exposure)
                exhaustion_tags.add(c['code'])
            elif er > self.exhaustion_moderate_threshold:
                weights[i] *= self.exhaustion_reduce_mult

        # 行业集中度上限
        industry_weights_post = {}
        for i, c in enumerate(selected):
            ind = c.get('industry', '其他')
            industry_weights_post[ind] = industry_weights_post.get(ind, 0.0) + weights[i]
        for ind, total_w in industry_weights_post.items():
            if total_w > self.industry_max_weight * target_exposure:
                scale = (self.industry_max_weight * target_exposure) / total_w
                for i, c in enumerate(selected):
                    if c.get('industry', '其他') == ind:
                        weights[i] *= scale

        # 上限执行后不重新归一化：剩余敞口保留为现金，让上限真正绑定
        # 如果归一化，上限约束会被"剩余资金再分配"瓦解

        # === 最小交易单位检查: 确保每只至少1手 + 过滤买不起的 ===
        desired_value = {}
        valid_selected = []
        valid_weights = []
        # 止损/HDS触发时真正空仓, 其余情况(低敞口/风险压缩/CLB)仍选最优标的
        _force_empty = self._stop_loss_triggered or self._hds_triggered
        for c, w in zip(selected, weights):
            val = w * total_equity
            min_lot = c['price'] * 100
            if val < min_lot:
                if _force_empty:
                    continue  # 止损/熔断中：真正空仓
                # 权重不够买1手: 检查是否值得提高至1手
                if min_lot <= total_equity * max_single * 1.1:
                    val = min_lot  # 提高到1手
                else:
                    # 买1手就超过仓位上限, 跳过这只股票
                    continue
            desired_value[c['code']] = val
            c['weight'] = val / total_equity if total_equity > 0 else 0
            valid_selected.append(c)
            valid_weights.append(c['weight'])

        # 重新归一化: 去掉买不起的股票后重新分配剩余资金
        if valid_selected:
            total_valid_w = sum(valid_weights) + 1e-10
            for c in valid_selected:
                c['weight'] = c['weight'] / total_valid_w * target_exposure
            # 归一化后重新应用单票上限（防止买不起的股票被移除后其余超限）
            for c in valid_selected:
                c['weight'] = min(c['weight'], max_single)
            # 重新归一化上限执行后的权重
            capped_total = sum(c['weight'] for c in valid_selected) + 1e-10
            for c in valid_selected:
                c['weight'] = c['weight'] / capped_total * target_exposure
            # 最终上限保护: 归一化膨胀后再次裁剪（防止重归一化导致单票超限）
            for c in valid_selected:
                c['weight'] = min(c['weight'], max_single)
            for c in valid_selected:
                desired_value[c['code']] = c['weight'] * total_equity

        # 用 valid_selected 替换 selected (用于日志记录)
        selected = valid_selected

        # 记录入场日期(用于最小持仓天数约束)
        for c in selected:
            code = c.get('code', '')
            if code not in current_positions:
                self._position_entry_dates[code] = date

        # 记录选股结果（含缠论+因子详情，用于生成选股理由）
        self.last_selection = [
            {
                'date': date,
                'code': c['code'],
                'score': c['score'],
                'weight': c.get('weight', c['weight'] if 'weight' in c else 0),
                'industry': c.get('industry', ''),
                'rank_pct': c.get('rank_pct', 0),
                # 选股理由所需字段
                'factor_name': c.get('sig') and getattr(c['sig'], 'factor_name', '') or '',
                'factor_value': c.get('factor_value', 0),
                'chan_buy_point': c.get('chan_buy_point', 0),
                'chan_sell_point': c.get('chan_sell_point', 0),
                'chan_buy_strength': c.get('chan_buy_strength', 0.0),
                'trend_type': c.get('trend_type', 0),
                'signal_level': c.get('signal_level', 0),
                'chan_bonus': 0.0,  # 门控系统已处理
                'effective_score': c.get('effective_score', c['score']),
                'is_held': c.get('is_held', False),
                'confidence': c.get('confidence', 1.0),
            }
            for c in selected
        ]

        # 记录选股结果到链路日志
        for c in selected:
            code = c.get('code', '')
            plog.log_portfolio_select(
                code, c.get('score', 0), c.get('weight', c['weight'] if 'weight' in c else 0),
                c.get('industry', ''), c.get('rank_pct', 0),
                is_held=c.get('is_held', False),
            )

        print(f" [选股] date={date} universe={len(universe) if isinstance(universe, list) else '?'} "
              f"candidates={len(candidates)} qualified={len(qualified)} "
              f"gate_passed={len(gate_filtered)} selected={len(selected)}"
              f" exposure={target_exposure:.3f} stopped={int(self._stop_loss_triggered)}"
              f" hds={int(self._hds_triggered)} clb={int(self._clb_triggered)}")

        return desired_value

    def build(
        self,
        date,
        universe,
        current_positions,
        signal_store,
        cash,
        prices,
        market_regime=0,
        momentum_score=0.0,
        bear_risk=False,
        bear_risk_fast=False,
        severe_bear=False,
        trend_score=0.0,
        index_volume_ratio=1.0,
        style_score=0.0,
        regime_volatility=0.0,
        cost=None,
    ):
        """构建目标持仓（外部接口）

        关键设计: 非再平衡日只做个股成本止损, 不做信号止损
        """
        stop_loss_sells = {}
        _exit_tags = {}  # 追踪退出原因 (code -> reason)
        profit_reduce = {}  # 止盈减仓 (code -> target_pct_of_current)
        total_equity = cash + sum(current_positions.values())

        # === 趋势+熊市双确认才强制清仓: 单独trend<0可能是牛市回调===
        # 2024实证: trend<0的51天fwd20=+10.8%(反弹), 需bear_risk过滤
        if trend_score < 0 and bear_risk:
            for code in list(current_positions.keys()):
                stop_loss_sells[code] = 0.0
                _exit_tags[code] = 'trend_bear'
            self._entry_dates.clear()
            self._entry_reasons.clear()
            self._peak_prices.clear()
            self._entry_reason_lost_count.clear()
            self._post_sell_tracking.clear()
            desired_value = {c: 0.0 for c in current_positions}
            return desired_value

        # === 熊市动态止损: 预警期适度收紧, 确认熊市靠敞口控制而非紧止损 ===
        if bear_risk_fast and not bear_risk:
            _eff_stop_loss = 0.07  # 仅预警→7%
        else:
            _eff_stop_loss = self.position_stop_loss  # 正常/确认熊市→10%
        _eff_peak_trail = max(_eff_stop_loss * 0.8, 0.04)

        # === 硬止损优先检查：任何路径都必须先过止损，不可被其他逻辑覆盖 ===
        for code, current_value in current_positions.items():
            if code not in prices or current_value <= 0:
                continue
            if not (cost and code in cost and len(cost[code]) >= 2 and cost[code][0] > 0):
                continue
            avg_cost = cost[code][1]
            pnl_pct = (prices[code] - avg_cost) / avg_cost
            # 绝对止损: 熊市收紧(5-7%), 正常10%
            if pnl_pct <= -_eff_stop_loss:
                stop_loss_sells[code] = 0.0
                _exit_tags[code] = 'cost_stop'
                self._entry_dates.pop(code, None)
                self._entry_reasons.pop(code, None)
                self._peak_prices.pop(code, None)
                self._entry_reason_lost_count.pop(code, None)
                self._post_sell_tracking.pop(code, None)
                continue
            # 最高点回落同步收紧(熊市5%→正常8%)
            if code in self._peak_prices and self._peak_prices[code] > 0:
                dd_from_peak = (self._peak_prices[code] - prices[code]) / self._peak_prices[code]
                if dd_from_peak >= _eff_peak_trail:
                    stop_loss_sells[code] = 0.0
                    _exit_tags[code] = 'peak_trail'
                    self._entry_dates.pop(code, None)
                    self._entry_reasons.pop(code, None)
                    self._peak_prices.pop(code, None)
                    self._entry_reason_lost_count.pop(code, None)
                    self._post_sell_tracking.pop(code, None)
                    continue
            # simple模式: 仅成本止损+峰值回撤, 跳过其他复杂退出
            if self.exit_mode == 'simple':
                continue
            # 时间止损: 持仓>15天且亏损>5%
            if code in self._entry_dates:
                if isinstance(date, date_type):
                    days_held = (date - self._entry_dates[code]).days
                else:
                    days_held = 0
                if days_held > 15 and pnl_pct <= -0.05:
                    stop_loss_sells[code] = 0.0
                    self._entry_dates.pop(code, None)
                    self._entry_reasons.pop(code, None)
                    self._peak_prices.pop(code, None)
                    self._entry_reason_lost_count.pop(code, None)
                    self._post_sell_tracking.pop(code, None)
                    continue
            # 追踪止损: 按入场点分层阈值
            if code in self._peak_prices and self._peak_prices[code] > 0:
                dd_from_peak = (self._peak_prices[code] - prices[code]) / self._peak_prices[code]
                bp = self._entry_reasons.get(code, {}).get('chan_buy_point', 0) if code in self._entry_reasons else 0
                trail_threshold = self.trailing_stop_by_bp.get(bp, 0.10) if hasattr(self, 'trailing_stop_by_bp') else 0.10
                if dd_from_peak >= trail_threshold:
                    stop_loss_sells[code] = 0.0
                    self._entry_dates.pop(code, None)
                    self._entry_reasons.pop(code, None)
                    self._peak_prices.pop(code, None)
                    self._entry_reason_lost_count.pop(code, None)
                    self._post_sell_tracking.pop(code, None)
                    continue
            # 分级移动止盈: 盈利越高止盈线越紧
            if code in self._entry_dates and cost and code in cost and len(cost[code]) >= 2 and cost[code][0] > 0:
                pnl = (prices[code] - cost[code][1]) / cost[code][1]
                for profit_level in sorted(self.tiered_trailing_stop.keys(), reverse=True):
                    if pnl >= profit_level:
                        trail_pct = self.tiered_trailing_stop[profit_level]
                        if code in self._peak_prices and self._peak_prices[code] > 0:
                            dd = (self._peak_prices[code] - prices[code]) / self._peak_prices[code]
                            if dd >= trail_pct:
                                stop_loss_sells[code] = 0.0
                                self._entry_dates.pop(code, None)
                                self._peak_prices.pop(code, None)
                                self._entry_reasons.pop(code, None)
                                self._entry_reason_lost_count.pop(code, None)
                                self._post_sell_tracking.pop(code, None)
                        break
            # 死钱退出: 持仓过久无收益
            if code in self._entry_dates:
                if isinstance(date, date_type):
                    days_held = (date - self._entry_dates[code]).days
                else:
                    days_held = 0
                if days_held > self.dead_money_exit_days and abs(pnl_pct) < self.dead_money_return_band:
                    stop_loss_sells[code] = 0.0
                    self._entry_dates.pop(code, None)
                    self._peak_prices.pop(code, None)
                    self._entry_reasons.pop(code, None)
                    self._entry_reason_lost_count.pop(code, None)
                    self._post_sell_tracking.pop(code, None)

        # === 缠论止盈：分层退出 + 中枢移动止盈 (非调仓日也生效) ===
        chan_force_sells = {}
        for code, current_value in current_positions.items():
            if self.exit_mode == 'simple':
                break  # simple模式跳过所有复杂退出
            if code not in prices or current_value <= 0:
                continue
            sig = signal_store.get(code, date)
            if sig is None:
                continue
            sl = getattr(sig, 'signal_level', 0)
            cs = getattr(sig, 'chan_sell_point', 0)
            bi_sell = getattr(sig, 'chan_divergence_type', '')

            if self.chan_tp_enabled:
                # 双级别确认卖出 (signal_level <= -3) → 全清
                if sl <= -3:
                    chan_force_sells[code] = 0.0
                    self._entry_dates.pop(code, None)
                    self._peak_prices.pop(code, None)
                    self._entry_reasons.pop(code, None)
                # S1 (一卖) → 按配置减仓（默认50%），保留底仓观察趋势
                elif cs == 1:
                    if code not in profit_reduce:
                        profit_reduce[code] = min(self.chan_tp_s1_reduce, 1.0)
                # S2 (二卖) → 按配置减仓，记录触发价用于跟踪止损
                elif cs == 2:
                    if code not in profit_reduce:
                        profit_reduce[code] = min(self.chan_tp_s2_reduce, 1.0)
                    self._post_sell_tracking[code] = {
                        'trigger_price': prices[code],
                        'reason': 'S2'
                    }
                # S3 (三卖) → 强制清仓
                elif cs == 3:
                    chan_force_sells[code] = 0.0
                    self._entry_dates.pop(code, None)
                    self._peak_prices.pop(code, None)
                    self._entry_reasons.pop(code, None)
                # 笔级别卖出 (signal_level == -1) → 按配置
                elif sl == -1 and self.chan_tp_bi_sell_exit:
                    if code not in profit_reduce:
                        profit_reduce[code] = 0.5
            else:
                # 回退到旧逻辑（无缠论止盈配置时）
                if sl <= -2 or cs >= 2:
                    chan_force_sells[code] = 0.0
                    self._entry_dates.pop(code, None)
                    self._peak_prices.pop(code, None)
                    self._entry_reasons.pop(code, None)
                elif sl == -1 or cs == 1:
                    if code not in profit_reduce:
                        profit_reduce[code] = 0.5

        # === 中枢移动止盈：盈利>profit_lock_pct后，跌破中枢下沿即止盈 ===
        if self.exit_mode != 'simple' and self.chan_tp_enabled and self.chan_tp_pivot_trailing:
            for code, current_value in current_positions.items():
                if code in chan_force_sells:
                    continue
                if code not in prices or current_value <= 0:
                    continue
                if cost and code in cost and len(cost[code]) >= 2 and cost[code][0] > 0:
                    avg_cost = cost[code][1]
                    current_price = prices[code]
                    pnl_pct = (current_price - avg_cost) / avg_cost
                    # 盈利超过锁定阈值才启动移动止盈
                    if pnl_pct > self.chan_tp_profit_lock:
                        sig = signal_store.get(code, date)
                        if sig:
                            pivot_zd = getattr(sig, 'chan_pivot_zd', float('nan'))
                            if not np.isnan(pivot_zd) and pivot_zd > 0:
                                # 价格跌破中枢下沿(留0.5%缓冲) → 止盈
                                if current_price < pivot_zd * 0.995:
                                    chan_force_sells[code] = 0.0
                                    self._entry_dates.pop(code, None)
                                    self._entry_reasons.pop(code, None)
                                    self._peak_prices.pop(code, None)
                                    continue
                        # 无中枢但有盈利：启动价格移动止盈（使用分层止损率）
                        if code in self._peak_prices and self._peak_prices[code] > 0:
                            peak = self._peak_prices[code]
                            drawdown = (peak - current_price) / peak
                            entry_bp = self._entry_reasons.get(code, {}).get('buy_point', 0)
                            tiered_pct = self.trailing_stop_by_bp.get(
                                entry_bp, self.trailing_stop_by_bp.get('default', 0.08)
                            )
                            if drawdown > tiered_pct:
                                profit_reduce[code] = 0.5

        # === 趋势衰减卖出: 盈利后缩量+价格回落 → 减仓（不依赖缠论卖点）===
        for code, current_value in current_positions.items():
            if self.exit_mode == 'simple':
                break
            if code in chan_force_sells or code in profit_reduce:
                continue
            if code not in prices or current_value <= 0:
                continue
            if not (cost and code in cost and len(cost[code]) >= 2 and cost[code][0] > 0):
                continue
            avg_cost = cost[code][1]
            profit_pct = (prices[code] - avg_cost) / avg_cost
            # 盈利>30%且从峰值回撤>10% → 全部止盈
            if profit_pct > 0.30 and code in self._peak_prices and self._peak_prices[code] > 0:
                dd = (self._peak_prices[code] - prices[code]) / self._peak_prices[code]
                if dd > 0.10:
                    profit_reduce[code] = 0.0  # 全清
                    continue
            # 盈利>20%且从峰值回撤>8% → 止盈50%
            if profit_pct > 0.20 and code in self._peak_prices and self._peak_prices[code] > 0:
                dd = (self._peak_prices[code] - prices[code]) / self._peak_prices[code]
                if dd > 0.08:
                    profit_reduce[code] = 0.5
                    continue
            # 盈利>15% + 价格低于前高3% → 趋势衰减减仓50%
            if profit_pct > 0.15 and code in self._peak_prices and self._peak_prices[code] > 0:
                below_peak = (self._peak_prices[code] - prices[code]) / self._peak_prices[code]
                if below_peak > 0.03:
                    profit_reduce[code] = 0.5
                    continue

        # === 均值回归退出: 持仓中极端乖离 → 减仓/清仓 (非调仓日也生效) ===
        for code, current_value in current_positions.items():
            if self.exit_mode == 'simple':
                break
            if code in chan_force_sells or code in profit_reduce:
                continue
            if code not in prices or current_value <= 0:
                continue
            sig = signal_store.get(code, date)
            if sig is None:
                continue
            mom_60d = self._nan_safe(getattr(sig, 'mom_60d', 0.0))
            dist_ma60 = self._nan_safe(getattr(sig, 'dist_ma60', 0.0))
            if mom_60d > self.mr_mom60_force:
                chan_force_sells[code] = 0.0
                self._entry_dates.pop(code, None)
                self._entry_reasons.pop(code, None)
                self._peak_prices.pop(code, None)
                self._mr_exit_cooldown[code] = date
            elif dist_ma60 > self.mr_dist60_force:
                chan_force_sells[code] = 0.0
                self._entry_dates.pop(code, None)
                self._entry_reasons.pop(code, None)
                self._peak_prices.pop(code, None)
                self._mr_exit_cooldown[code] = date
            elif mom_60d > self.mr_mom60_reduce:
                if code not in profit_reduce:
                    profit_reduce[code] = self.mr_reduce_pct
            elif dist_ma60 > self.mr_dist60_reduce:
                if code not in profit_reduce:
                    profit_reduce[code] = self.mr_reduce_pct

        # 个股止损检查 - 成本止损 + 时间止损 + 移动止损
        for code, current_value in current_positions.items():
            if code not in prices or current_value <= 0:
                continue

            current_price = prices[code]
            stopped = False
            stop_reason = ""

            sig = signal_store.get(code, date)

            # === 信号层卖出：MA60止损、分数阈值卖出（信号层判断，组合层执行）===
            if self.exit_mode != 'simple' and sig is not None and getattr(sig, 'sell', False):
                stopped = True
                stop_reason = "sig_sell"
                stop_loss_sells[code] = 0.0
                self._entry_dates.pop(code, None)
                self._entry_reasons.pop(code, None)
                self._peak_prices.pop(code, None)
                self._entry_reason_lost_count.pop(code, None)
                self._post_sell_tracking.pop(code, None)
                continue
            chan_div_type = getattr(sig, 'chan_divergence_type', '') if sig else ''
            chan_div_strength = getattr(sig, 'chan_divergence_strength', 0.0) if sig else 0.0
            chan_buy_point = getattr(sig, 'chan_buy_point', 0) if sig else 0
            chan_sell_point = getattr(sig, 'chan_sell_point', 0) if sig else 0
            sl = getattr(sig, 'signal_level', 0) if sig else 0
            # B2/B3：趋势已确认，放宽止损防洗盘
            # B1：抄底信号，反转可能失败，不放宽
            # Fix#6: 取消Chan保护止损放宽 - 结构信号不能替代风险管理
            # 退出路径优先级: 硬止损 > 缠论结构卖出 > 移动止盈 > 时间止损
            # 所有买点类型统一止损线，不再因缠论信号放宽
            chan_protection = False
            chan_protection_mult = 1.0  # 统一为1.0，不放宽

            # 1. 成本止损: 亏损超过position_stop_loss
            avg_cost_check = 0.0  # Fix#7: 供分级移动止损使用
            if cost and code in cost and len(cost[code]) >= 2 and cost[code][0] > 0:
                avg_cost = cost[code][1]
                avg_cost_check = avg_cost  # Fix#7: 保存供后续使用
                pnl_pct = (current_price - avg_cost) / avg_cost

                # 波动率自适应调整：高波动股票放宽止损线
                vol = getattr(sig, 'risk_vol', 0.03) if sig else 0.03
                adaptive_mult = self.volatility_adaptive_mult
                adaptive_stop = self.position_stop_loss * (1 + vol * adaptive_mult * 10)
                adaptive_stop = min(adaptive_stop, self.position_stop_loss * self.max_adaptive_stop_mult)

                # Chan保护：B2/B3放宽1.5x(非2x), B1不放宽
                if chan_protection:
                    adaptive_stop = adaptive_stop * chan_protection_mult

                if pnl_pct < -adaptive_stop:
                    stopped = True
                    stop_reason = "cost_stop"

            # 2. 时间止损: 按买点类型分设天数和亏损阈值
            if self.exit_mode != 'simple' and not stopped and code in self._entry_dates:
                if isinstance(date, date_type):
                    days_held = (date - self._entry_dates[code]).days
                else:
                    days_held = 0
                entry_reason = self._entry_reasons.get(code, {})
                entry_bp = entry_reason.get('buy_point', 0)
                eff_days, eff_min_return = self.time_stop_by_bp.get(
                    entry_bp, (self.time_stop_days, self.time_stop_min_return)
                )
                if days_held > eff_days:
                    if cost and code in cost and len(cost[code]) >= 2 and cost[code][0] > 0:
                        avg_cost = cost[code][1]
                        pnl_pct = (current_price - avg_cost) / avg_cost
                        if pnl_pct < eff_min_return:
                            stopped = True
                            stop_reason = f"time_stop_bp{entry_bp}"

            # 3. 移动止损: 从最高点回撤超过阈值 (与缠论中枢联动)
            # 若缠论中枢存在, 使用 min(价格回撤阈值, 中枢下沿) 作为止损线
            if self.exit_mode != 'simple' and not stopped and self.trailing_stop_enabled and code in self._peak_prices \
               and code not in profit_reduce:
                peak = self._peak_prices[code]
                drawdown_from_peak = (peak - current_price) / peak
                entry_bp_ts = self._entry_reasons.get(code, {}).get('buy_point', 0)

                if avg_cost_check > 0:
                    profit_from_peak = (peak - avg_cost_check) / avg_cost_check
                else:
                    profit_from_peak = 0.10
                # 按买点类型取基础移动止损线（从config读取）
                tiered_trailing_pct = self.trailing_stop_by_bp.get(
                    entry_bp_ts, self.trailing_stop_by_bp.get('default', 0.08)
                )
                # 盈利>15%后收紧止损锁利
                if profit_from_peak > 0.15:
                    tiered_trailing_pct = max(tiered_trailing_pct - 0.03, 0.05)

                # 缠论中枢联动: 若中枢存在, 追踪止损不破中枢下沿
                sig_ts = signal_store.get(code, date)
                if sig_ts:
                    zg = getattr(sig_ts, 'chan_pivot_zg', float('nan'))
                    zd = getattr(sig_ts, 'chan_pivot_zd', float('nan'))
                    if not np.isnan(zg) and not np.isnan(zd) and zg > 0 and zd > 0:
                        dist_from_zd = (current_price - zd) / zd if zd > 0 else 1.0
                        # 价格仍在中枢上方 → 中枢下沿作为硬止损线 (不破结构)
                        if dist_from_zd > 0.02:
                            price_stop_from_chan = zd * 0.995  # 略低于中枢下沿
                            tiered_stop_price = peak * (1 - tiered_trailing_pct)
                            effective_stop = max(tiered_stop_price, price_stop_from_chan)
                            if current_price <= effective_stop:
                                stopped = True
                                stop_reason = "trailing_chan"
                            # 未触发 → 正常检查
                            elif drawdown_from_peak > tiered_trailing_pct:
                                stopped = True
                                stop_reason = "trailing_stop"
                        else:
                            # 已跌破中枢 → 使用纯价格止损
                            if drawdown_from_peak > tiered_trailing_pct:
                                stopped = True
                                stop_reason = "trailing_stop"
                    else:
                        if drawdown_from_peak > tiered_trailing_pct:
                            stopped = True
                            stop_reason = "trailing_stop"
                else:
                    if drawdown_from_peak > tiered_trailing_pct:
                        stopped = True
                        stop_reason = "trailing_stop"

            # === Chan理论退出检查（czsc增强版）：买卖点 + 多级别确认 + 背离 + 趋势耗尽 ===
            if self.exit_mode != 'simple' and not stopped:
                # 获取当前信号
                sig = signal_store.get(code, date)
                chan_div_type = getattr(sig, 'chan_divergence_type', '') if sig else ''
                chan_div_strength = getattr(sig, 'chan_divergence_strength', 0.0) if sig else 0.0
                chan_struct_score = getattr(sig, 'chan_structure_score', 0.0) if sig else 0.0
                chan_sell_point = getattr(sig, 'chan_sell_point', 0) if sig else 0
                sl = getattr(sig, 'signal_level', 0) if sig else 0  # 多级别确认

                # 双级别确认卖出 (signal_level <= -2) → 最强退出，不需要额外阈值
                if sl <= -2:
                    stopped = True
                    stop_reason = f"chan_sl_{sl}"
                # 一卖/二卖/三卖 → 强制退出
                elif chan_sell_point == 1 and chan_div_strength > 0.3:
                    stopped = True
                    stop_reason = f"chan_sell_{chan_sell_point}"
                elif chan_sell_point == 2 and chan_div_strength > 0.25:
                    stopped = True
                    stop_reason = f"chan_sell_{chan_sell_point}"
                elif chan_sell_point == 3 and chan_div_strength > 0.25:
                    stopped = True
                    stop_reason = f"chan_sell_{chan_sell_point}"

                # 顶背离退出：价格高位 + MACD顶背离 → 提前止盈
                if not stopped and chan_div_type == 'top' and chan_div_strength > 0.4:
                    stopped = True
                    stop_reason = "chan_top_divergence"

                # 隐藏顶背离 + 结构走弱 → 趋势可能转弱
                if not stopped and chan_div_type == 'hidden_top' and chan_div_strength > 0.25:
                    if chan_struct_score < 0.2:
                        stopped = True
                        stop_reason = "chan_hidden_divergence"

                # 趋势耗尽：多级别对齐从正转负
                if not stopped and chan_struct_score < -0.4:
                    stopped = True
                    stop_reason = "chan_trend_exhaustion"

                # 一卖信号出现在上涨中且结构完成 → 趋势可能反转
                if not stopped and chan_div_type == 'sell1' and chan_div_strength > 0.35:
                    stopped = True
                    stop_reason = "chan_sell1_reversal"

            # === S2减仓后跟踪止损: 价格继续下跌5%+ → 清仓剩余 ===
            if not stopped and code in self._post_sell_tracking:
                track = self._post_sell_tracking[code]
                if current_price < track['trigger_price'] * 0.95:
                    stopped = True
                    stop_reason = f"{track['reason']}_deterioration"
                elif (isinstance(date, date_type) and code in self._entry_dates and
                      (date - self._entry_dates[code]).days > 15):
                    self._post_sell_tracking.pop(code, None)

            # === 买入理由消失检查：入场时依赖的结构条件不再满足 ===
            if self.exit_mode != 'simple' and not stopped and code in self._entry_reasons:
                reason = self._entry_reasons[code]
                entry_bp = reason.get('buy_point', 0)
                entry_sl = reason.get('signal_level', 0)
                # 当前个股趋势（从最新信号读取，非入场时存储值）
                curr_trend = getattr(sig, 'trend_type', 0) if sig else 0

                # B3入场: 价格跌破中枢下沿 → 三买失败
                if entry_bp == 3:
                    pivot_zd = getattr(sig, 'chan_pivot_zd', float('nan')) if sig else float('nan')
                    if not np.isnan(pivot_zd) and pivot_zd > 0:
                        if current_price < pivot_zd * 0.995:
                            stopped = True
                            stop_reason = "b3_zg_break"
                # B2入场: 趋势转为下跌 → 二买失效
                elif entry_bp == 2:
                    if curr_trend == -2:
                        stopped = True
                        stop_reason = "b2_trend_broken"
                # B1入场: 底背离消失 + 信号级别归零 → 一买失效
                elif entry_bp == 1:
                    curr_bp = getattr(sig, 'chan_buy_point', 0) if sig else 0
                    curr_div = getattr(sig, 'chan_divergence_strength', 0.0) if sig else 0.0
                    if curr_bp == 0 and curr_div < 0.15:
                        stopped = True
                        stop_reason = "b1_divergence_lost"
                # 通用: 买入时有的Chan结构完全消失
                # Fix#10 (修正): 3天确认期 + 结构破位/深度亏损立即退出
                if not stopped:
                    curr_bp = getattr(sig, 'chan_buy_point', 0) if sig else 0
                    curr_sl = getattr(sig, 'signal_level', 0) if sig else 0
                    curr_div = getattr(sig, 'chan_divergence_strength', 0.0) if sig else 0.0
                    if curr_bp == 0 and curr_sl == 0 and curr_div < 0.1:
                        if entry_bp > 0 or entry_sl >= 1:
                            # 结构破位: 价格跌破中枢下沿 → 立即确认退出
                            pivot_zd = getattr(sig, 'chan_pivot_zd', float('nan')) if sig else float('nan')
                            structure_broken = (not np.isnan(pivot_zd) and pivot_zd > 0
                                              and current_price < pivot_zd * 0.995)
                            # 深度亏损: PnL<-12% → 不管结构直接走
                            deep_loss = (avg_cost_check > 0 and current_price < avg_cost_check * 0.88)
                            if structure_broken or deep_loss:
                                stopped = True
                                stop_reason = "entry_lost_structure" if structure_broken else "entry_lost_deep"
                                self._entry_reason_lost_count.pop(code, None)
                            else:
                                # 温和走弱 → 给3天观察期
                                if code not in self._entry_reason_lost_count:
                                    self._entry_reason_lost_count[code] = 0
                                self._entry_reason_lost_count[code] += 1
                                if self._entry_reason_lost_count[code] >= 3:
                                    stopped = True
                                    stop_reason = "entry_reason_lost"
                        else:
                            self._entry_reason_lost_count.pop(code, None)
                    else:
                        self._entry_reason_lost_count.pop(code, None)

            if stopped:
                stop_loss_sells[code] = 0.0
                # 清理追踪状态
                self._entry_dates.pop(code, None)
                self._entry_reasons.pop(code, None)
                self._peak_prices.pop(code, None)
                self._entry_reason_lost_count.pop(code, None)
                self._post_sell_tracking.pop(code, None)

            # 更新峰值价格
            if code in self._entry_dates:
                self._peak_prices[code] = max(self._peak_prices.get(code, 0), current_price)

        desired_value = {}
        # 调仓频率控制: 每10个交易日做一次完整选股, 非调仓日仅止损
        _rebalance_interval = 10
        _should_rebalance = False
        if not hasattr(self, '_last_rebalance_date'):
            self._last_rebalance_date = None
        if self._last_rebalance_date is None:
            _should_rebalance = True
        elif hasattr(date, 'timetuple'):
            # 计算交易日差(近似: 日历日差*0.7≈交易日)
            _calendar_days = (date - self._last_rebalance_date).days
            _should_rebalance = _calendar_days >= _rebalance_interval
        # HDS触发期间禁止新买入
        if _should_rebalance and not self._hds_triggered:
            self._last_rebalance_date = date
            desired_value = self._build_desired_value(
                date=date,
                universe=universe,
                current_positions=current_positions,
                signal_store=signal_store,
                cash=cash,
                prices=prices,
                momentum_score=momentum_score,
                trend_score=trend_score,
                index_volume_ratio=index_volume_ratio,
                style_score=style_score,
                regime_volatility=regime_volatility,
                market_regime=market_regime,
                bear_risk=bear_risk,
                bear_risk_fast=bear_risk_fast,
                severe_bear=severe_bear,
            )

        # 强制卖出 (止损 + Chan卖出 + 弱势板块)
        for code in stop_loss_sells:
            desired_value[code] = 0.0
            self._post_sell_tracking.pop(code, None)
            _reason = _exit_tags.get(code, 'stop_loss')
            _pnl = (prices.get(code, 0) - cost.get(code, [0,0])[1]) / max(cost.get(code, [0,0])[1], 0.01) if code in cost else 0
            plog.log_exit_reason(code, _reason, _pnl)
        for code in chan_force_sells:
            desired_value[code] = 0.0
            self._post_sell_tracking.pop(code, None)
            _pnl = (prices.get(code, 0) - cost.get(code, [0,0])[1]) / max(cost.get(code, [0,0])[1], 0.01) if code in cost else 0
            plog.log_exit_reason(code, 'chan_structure_exit', _pnl)
        # 连续2期弱势板块: 强制清仓该行业现有持仓
        if hasattr(self, '_force_exit_industries') and self._force_exit_industries:
            for code in current_positions:
                sig = signal_store.get(code, date) if signal_store else None
                if sig and getattr(sig, 'industry', '') in self._force_exit_industries:
                    if code not in desired_value:  # 止损优先, 不重复
                        desired_value[code] = 0.0
                        _pnl = (prices.get(code, 0) - cost.get(code, [0,0])[1]) / max(cost.get(code, [0,0])[1], 0.01) if code in cost else 0
                        plog.log_exit_reason(code, 'weak_industry', _pnl)

        # 止盈减仓
        for code, reduce_pct in profit_reduce.items():
            if code in desired_value:
                desired_value[code] *= (1 - reduce_pct)
            elif code in current_positions:
                desired_value[code] = current_positions[code] * (1 - reduce_pct)

        # 事件驱动: 始终从现有持仓开始, 止损卖出覆盖 + 新买入补票
        adjusted = deepcopy(current_positions)

        for code, target in desired_value.items():
            current = current_positions.get(code, 0.0)
            diff = target - current

            if code in stop_loss_sells:
                adjusted[code] = 0.0
                continue

            # 忽略微小调整
            if abs(diff) < total_equity * 0.01:
                if code in current_positions:
                    adjusted[code] = current
                continue

            if diff > 0:
                adjusted[code] = current + self.entry_speed * diff
            else:
                # 止损卖出
                adjusted[code] = current + diff

        # 更新入场追踪
        for code, target_val in adjusted.items():
            if target_val > 0 and code not in self._entry_dates:
                self._entry_dates[code] = date
                self._peak_prices[code] = prices.get(code, 0)

        return adjusted
