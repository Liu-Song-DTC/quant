# core/portfolio.py
"""
组合构建器 - 基于因子排名选股

设计原则:
1. 等权top N: 截面rank_pct>0.5的top N只, 等权分配
2. 行业均衡: 每个行业最多5只, 避免过度集中
3. 最小风控: 仅深度回撤和fv_mean降仓
4. 止损仅限再平衡日: 避免非再平衡日过早卖出
"""
import numpy as np
from collections import deque
from copy import deepcopy
from datetime import date as date_type
from .config_loader import load_config
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


class PortfolioConstructor:
    """仓位管理器 - 基于因子排名选股"""

    MAX_POSITIONS = 12

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

        # === 波动率控制 ===
        self.vol_control_enabled = portfolio_config.get('volatility_control_enabled', True)
        self.target_volatility = target_volatility if target_volatility is not None else portfolio_config.get('target_volatility', 0.15)
        self.vol_lookback = portfolio_config.get('volatility_control_lookback', 20)
        self._daily_returns = deque(maxlen=self.vol_lookback * 2)  # store up to 2x for safety

        # === 组合止损 ===
        self.stop_loss_enabled = portfolio_config.get('portfolio_stop_loss_enabled', True)
        self.portfolio_stop_loss = portfolio_stop_loss if portfolio_stop_loss is not None else portfolio_config.get('portfolio_stop_loss', 0.08)
        self.emergency_exposure = portfolio_config.get('emergency_exposure', 0.3)

        # === 动态最小持仓（小说：熊市允许空仓）===
        dyn_min_pos = portfolio_config.get('dynamic_min_positions', {})
        self._min_positions_map = {
            1: dyn_min_pos.get('bull', 2),      # 牛市至少2只
            0: dyn_min_pos.get('neutral', 1),   # 中性至少1只
            -1: dyn_min_pos.get('bear', 0),     # 熊市允许空仓
        }

        # === 单票权重上限 ===
        self.max_single_weight = portfolio_config.get('max_single_weight', 0.12)

        # === 绝对质量门槛（小说：龙头洁癖）===
        sel_config = portfolio_config.get('selection', {})
        self.min_rank_pct = sel_config.get('min_rank_pct', 0.5)
        self.min_absolute_score = sel_config.get('min_absolute_score', 0.15)
        self.min_confidence = sel_config.get('min_confidence', 0.80)
        self.exhaustion_max_weight = sel_config.get('exhaustion_max_weight', 0.03)
        self.exhaustion_reduce_mult = sel_config.get('exhaustion_reduce_mult', 0.5)
        self.industry_max_weight = sel_config.get('industry_max_weight', 0.20)
        self.max_single_weight_from_cfg = sel_config.get('max_single_weight', self.max_single_weight)

        # === fv_exposure 可配置参数 ===
        fv_params = portfolio_config.get('fv_exposure_params', {})
        self.fv_low = fv_params.get('fv_low', -0.03)
        self.fv_high = fv_params.get('fv_high', 0.05)
        self.fv_exposure_min = fv_params.get('exposure_min', 0.3)
        self.fv_exposure_max = fv_params.get('exposure_max', 1.0)

        # === 换手惩罚 ===
        self.turnover_bonus = portfolio_config.get('turnover_bonus', 0.02)

        # === 风险平价 ===
        rp_config = config.get('risk_parity', {}) if hasattr(config, 'get') else portfolio_config.get('risk_parity', {})
        self.risk_parity_enabled = rp_config.get('enabled', False)
        self.rp_target_risk = rp_config.get('target_risk_per_stock', 0.10)
        self.rp_max_iterations = rp_config.get('max_iterations', 50)

        # === 增强止损 ===
        es_config = config.get('enhanced_stop_loss', {}) if hasattr(config, 'get') else portfolio_config.get('enhanced_stop_loss', {})
        self.time_stop_days = es_config.get('time_stop_days', 60)
        self.time_stop_min_return = es_config.get('time_stop_min_return', -0.02)
        self.trailing_stop_pct = es_config.get('trailing_stop_pct', 0.15)
        self.trailing_stop_enabled = es_config.get('trailing_stop_enabled', True)
        self.volatility_adaptive_mult = es_config.get('volatility_adaptive_mult', 1.5)

        # === 缠论止盈配置 ===
        chan_config = config.get('chan_theory', {}) if hasattr(config, 'get') else {}
        tp = chan_config.get('take_profit', {})
        self.chan_tp_enabled = tp.get('enabled', True)
        self.chan_tp_s1_reduce = tp.get('s1_reduce_pct', 1.0)      # S1→全清
        self.chan_tp_s2_reduce = tp.get('s2_reduce_pct', 0.5)      # S2→减半
        self.chan_tp_s3_reduce = tp.get('s3_reduce_pct', 1.0)      # S3→全清
        self.chan_tp_pivot_trailing = tp.get('pivot_trailing_enabled', True)
        self.chan_tp_profit_lock = tp.get('profit_lock_pct', 0.15) # 盈利>15%启动移动止盈
        self.chan_tp_bi_sell_exit = tp.get('bi_sell_exit', True)

        # 追踪每个持仓的入场时间和峰值价格
        self._entry_dates: dict = {}  # {code: date}
        self._peak_prices: dict = {}  # {code: peak_price_since_entry}
        self._entry_reasons: dict = {}  # {code: {buy_point, signal_level, trend_type}}

        self.peak_equity = None
        self.industry_ic = _load_industry_ic_weights()
        self.position_cost = {}
        self.sentiment_multipliers: dict = {}  # 情绪乘数，由外部注入
        self.last_selection = []
        self.current_ranking = {}
        self.current_n_positions = 0
        self.current_exposure = 1.0
        self._stop_loss_triggered = False
        self._stop_loss_recovery_days = 0
        # 小说优化：延长最短持仓至7天，防止被洗盘震出
        # 核心逻辑："主力洗盘时缩量回调，不要被吓出去"
        self._min_hold_days = 5  # 最短持仓天数（降低以允许快速止损）

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

    @staticmethod
    def _calc_max_position(total_equity: float, prices: dict) -> int:
        """根据资金自动计算最大持仓数

        每10000资金支持1个仓位, 范围[5, 12]
        """
        n = int(total_equity / 10000)
        return max(5, min(n, PortfolioConstructor.MAX_POSITIONS))

    def _get_min_positions(self, market_regime: int) -> int:
        """动态最小持仓数（小说：熊市允许空仓）

        Args:
            market_regime: 1=bull, 0=neutral, -1=bear

        Returns:
            当前市场状态下允许的最小持仓数
        """
        return self._min_positions_map.get(market_regime, 0)

    def _risk_parity_weights(self, selected: list, total_equity: float) -> list:
        """风险平价权重分配：每只股票贡献相等的风险预算

        Args:
            selected: 选中的股票列表，每项含 'risk_vol', 'code'
            total_equity: 总权益

        Returns:
            权重列表，和为 target_exposure
        """
        n = len(selected)
        if n == 0:
            return []

        # 用波动率作为风险的代理指标
        vols = np.array([c.get('risk_vol', 0.03) for c in selected])
        vols = np.clip(vols, 0.01, 0.10)  # 限制极端值

        # 初始等权
        weights = np.ones(n) / n
        target_risk_contrib = self.rp_target_risk / n

        # 简单实现：与波动率倒数成正比（Naive Risk Parity）
        inv_vols = 1.0 / vols
        weights = inv_vols / inv_vols.sum()

        # 迭代优化风险贡献（最多N次迭代）
        for _ in range(self.rp_max_iterations):
            # 风险贡献：w_i * sigma_i
            risk_contrib = weights * vols
            # 总风险
            total_risk = np.sqrt(np.sum((weights * vols) ** 2))
            if total_risk < 1e-10:
                break
            # 调整权重使风险贡献更均衡
            target_risk_contrib = total_risk / n
            adj = target_risk_contrib / (risk_contrib + 1e-10)
            weights = weights * (0.5 + 0.5 * adj)  # 平滑调整
            weights = weights / weights.sum()
            # 收敛检查
            max_dev = np.max(np.abs(risk_contrib / total_risk - 1.0 / n))
            if max_dev < 0.01:
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
        market_regime=0,
        momentum_score=0.0,
        bear_risk=False,
        trend_score=0.0,
    ):
        """构建目标持仓 - 等权top N选股"""
        import pandas as pd

        total_equity = cash + sum(current_positions.values())
        n_positions = self._calc_max_position(total_equity, prices)

        # 计算峰值和回撤
        if self.peak_equity is None:
            self.peak_equity = total_equity
        self.peak_equity = max(self.peak_equity, total_equity)
        drawdown = 1 - total_equity / self.peak_equity if self.peak_equity > 0 else 0.0

        # === 收集候选股票 ===
        candidates = []
        for code in universe:
            sig = signal_store.get(code, date)
            if sig is None:
                continue

            factor_value = getattr(sig, 'factor_value', None)
            if factor_value is None or (isinstance(factor_value, float) and np.isnan(factor_value)):
                continue

            # 价格约束：跳过无价格数据或价格异常的股票
            price = prices.get(code, 0)
            if price <= 0:
                continue
            # 100股整手下，100*price必须不超过理想仓位
            # 允许2倍理想仓位（100股整手会导致超配，但至少能买入）
            ideal_per_stock = total_equity / n_positions
            if price * 100 > ideal_per_stock * 2:
                continue

            # === Chan 信号数据 (融合核心) ===
            sl = getattr(sig, 'signal_level', 0)
            chan_buy = getattr(sig, 'chan_buy_point', 0)
            chan_sell = getattr(sig, 'chan_sell_point', 0)
            buy_strength = getattr(sig, 'chan_divergence_strength', 0.0)
            stock_trend = getattr(sig, 'trend_type', 0)   # 个股走势类型
            fn = getattr(sig, 'factor_name', '')

            # Chan 卖出信号: 禁止新入场
            if sl < 0 or chan_sell > 0:
                if code not in current_positions:
                    continue
            # 个股下跌趋势且无Chan买点 → 禁止新入场
            if stock_trend == -2 and chan_buy == 0 and sl <= 0:
                if code not in current_positions:
                    continue
            # 当日跌幅检查: B3当天下跌需区分"健康回调"vs"破位"
            daily_ret = getattr(sig, 'daily_return', 0.0)
            if chan_buy == 3 and daily_ret < -0.03:
                zg = getattr(sig, 'chan_pivot_zg', float('nan'))
                if not np.isnan(zg) and zg > 0:
                    dist_from_zg = (price - zg) / zg
                    if dist_from_zg < -0.01:
                        if code not in current_positions:
                            continue
                    elif dist_from_zg > 0.05:
                        if code not in current_positions:
                            continue
                else:
                    if daily_ret < -0.04 and code not in current_positions:
                        continue
            # 放量下跌检查: 跌>1.5%且量>1.5x均量 → 不是健康回调, 是出货
            if daily_ret < -0.015 and chan_buy > 0:
                vol_ratio = getattr(sig, 'volume_ratio', 1.0) if hasattr(sig, 'volume_ratio') else 1.0
                if vol_ratio > 0.5:  # volume_ratio是压缩值, >0.5 ≈ 放量
                    if code not in current_positions:
                        continue

            # 硬门控: 新入场必须有Chan结构确认 (买点/背离/底分型)
            # 纯因子驱动不参与选股，已持仓的不受影响
            if code not in current_positions:
                div_type = getattr(sig, 'chan_divergence_type', '')
                div_strength = getattr(sig, 'chan_divergence_strength', 0.0)
                has_chan = (
                    chan_buy > 0 or
                    sl >= 1 or
                    (div_type in ('bottom', 'bottom_fx', 'bottom_fx_3x', 'B2') and div_strength > 0.2)
                )
                if not has_chan:
                    continue
                # 均值回归时机过滤: 有缠论买点但严重追高也不入场
                # Chan确认结构，均值回归确认时机
                mom_60d = getattr(sig, 'mom_60d', 0.0) or 0.0
                dist_ma60 = getattr(sig, 'dist_ma60', 0.0) or 0.0
                if mom_60d > 0.30:
                    continue
                if dist_ma60 > 0.30:
                    continue

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
            })

        if not candidates:
            return {}

        # === 截面排名（全市场 + 行业内混合） ===
        factor_values = np.array([c['factor_value'] for c in candidates])
        cross_rank = pd.Series(factor_values).rank(pct=True)

        # 行业内排名：同行业股票内部比较，选出"行业最佳"
        industry_groups: dict = {}
        for i, c in enumerate(candidates):
            ind = c.get('industry', 'default') or 'default'
            if ind not in industry_groups:
                industry_groups[ind] = []
            industry_groups[ind].append((i, c['factor_value']))

        within_industry_rank = np.full(len(candidates), 0.5)
        for ind, members in industry_groups.items():
            if len(members) >= 3:
                indices, vals = zip(*members)
                ind_rank = pd.Series(np.array(vals)).rank(pct=True)
                for j, idx in enumerate(indices):
                    within_industry_rank[idx] = ind_rank.iloc[j]

        # 混合排名: 70%全市场 + 30%行业内
        blended_rank = 0.7 * cross_rank.values + 0.3 * within_industry_rank

        for i, c in enumerate(candidates):
            c['rank_pct'] = float(blended_rank[i])

        # === 市场仓位调整（v8 趋势主导）===
        # 平滑插值: trend_score ∈ [-0.5, 0.5] 线性映射到 [0.3, 1.0]
        # 消除之前在 trend_score=0 和 -0.5 处的仓位悬崖
        if trend_score > 0.5:
            market_signal = 1.0
            floor, ceiling = 0.8, 1.0
        elif trend_score > -0.5:
            t = (trend_score + 0.5)  # [-0.5, 0.5] → [0, 1.0]
            market_signal = float(0.3 + t * 0.7)  # [0.3, 1.0]
            floor = float(max(0.2, market_signal - 0.2))
            ceiling = float(min(1.0, market_signal + 0.2))
        else:
            market_signal = 0.3
            floor, ceiling = 0.2, 0.5

        # 叠加熊市风险（适度防御）
        if bear_risk and drawdown > 0.10:
            market_signal = min(market_signal, 0.6)

        target_exposure = float(np.clip(market_signal, floor, ceiling))

        # === Chan结构信号豁免：B1/强买点出现时不因熊市过度降仓 ===
        # 缠论B1恰好在下跌末端触发，此时市场regime判定为"熊市",
        # 若按regime降仓则系统性地错失最佳买入时机
        chan_strong_buys = sum(
            1 for c in candidates
            if c.get('signal_level', 0) >= 2 or c.get('chan_buy_point', 0) == 1
        )
        if chan_strong_buys >= 2 and target_exposure < 0.6:
            # >=2只股票出现强Chan买点 → 结构信号优先，最低敞口0.6
            target_exposure = max(target_exposure, 0.6)
            floor = max(floor, 0.5)

        # === 卖点聚合预警: 大量卖点出现 → 市场可能转向 ===
        chan_sell_count = sum(1 for c in candidates if c.get('chan_sell_point', 0) > 0)
        sell_ratio = chan_sell_count / max(len(candidates), 1)
        if sell_ratio > 0.30:
            target_exposure *= 0.65   # 超30%候选出现Chan卖点 → 大盘可能转向
        elif sell_ratio > 0.20:
            target_exposure *= 0.80   # 超20% → 温和降仓

        # === 市场广度门控: 候选股平均当日收益为负 → 市场偏弱 ===
        candidate_rets = []
        for c in candidates:
            sig = c.get('sig')
            if sig:
                dr = getattr(sig, 'daily_return', 0)
                candidate_rets.append(dr)
        if candidate_rets:
            avg_cand_ret = np.mean(candidate_rets)
            up_ratio = sum(1 for r in candidate_rets if r > 0) / len(candidate_rets)
            if up_ratio < 0.35:                    # 不到35%候选上涨 → 弱势市场
                target_exposure *= 0.60
            elif up_ratio < 0.45:                  # 不到45% → 偏弱
                target_exposure *= 0.80
            elif up_ratio > 0.65:                  # 超65%上涨 → 强势市场, 可以更积极
                target_exposure = min(1.0, target_exposure * 1.10)

        # === 波动率控制: 实现波动率超过目标时降仓 ===
        if self.vol_control_enabled and len(self._daily_returns) >= self.vol_lookback:
            recent_rets = list(self._daily_returns)[-self.vol_lookback:]
            realized_vol = float(np.std(recent_rets) * np.sqrt(252))  # 年化
            if realized_vol > 0.01:
                vol_scale = self.target_volatility / realized_vol
                target_exposure *= float(np.clip(vol_scale, 0.65, 1.0))

        # === 组合止损: 回撤超过阈值时强制降仓至紧急敞口（含恢复冷却期） ===
        if self.stop_loss_enabled and drawdown > self.portfolio_stop_loss:
            target_exposure = min(target_exposure, self.emergency_exposure)
            self._stop_loss_triggered = True
            self._stop_loss_recovery_days = 10
        elif self._stop_loss_triggered and self._stop_loss_recovery_days > 0:
            # 恢复期：逐步提升敞口，防止whipsaw
            target_exposure = min(target_exposure, self.emergency_exposure +
                                  (1.0 - self.emergency_exposure) * (1.0 - self._stop_loss_recovery_days / 10))
            self._stop_loss_recovery_days -= 1
            if self._stop_loss_recovery_days <= 0:
                self._stop_loss_triggered = False

        # 滞后平滑: 避免仓位突变（降低平滑系数，更快响应）
        self.current_exposure = 0.3 * self.current_exposure + 0.7 * target_exposure

        # === 选股: 绝对质量门槛 + 截面排名 ===
        # 小说"龙头洁癖"：不仅要排名靠前，还要自身够格
        # 两步过滤：1) 绝对分数门槛 2) 截面排名门槛
        if bear_risk or market_regime == -1:
            min_rank = 0.50
        elif market_regime == 1:
            min_rank = max(self.min_rank_pct, 0.55)
        else:
            min_rank = self.min_rank_pct

        # 第一轮：绝对质量门槛（龙头洁癖）
        # 信号太弱的股票，截面排名再高也不选
        qualified = [
            c for c in candidates
            if c['rank_pct'] > min_rank
            and c.get('score', 0) >= self.min_absolute_score
        ]

        # 第二轮：信心过滤
        # 注意: factor_quality>0 的信号历史收益更差(0.73% vs 1.07%), DYN因子胜率最低(9.3%),
        # 因此不再对这两项给予信心加成。仅保留V因子兜底的微弱加成。
        for c in qualified:
            sig = c['sig']
            confidence = 1.0
            fn = getattr(sig, 'factor_name', '')
            if '_V' in (fn or ''):
                confidence += 0.10  # V因子兜底,小幅加成
            c['confidence'] = np.clip(confidence, 0.7, 1.3)

        # 信心不足的股票淘汰（熊市提高门槛）
        eff_min_conf = self.min_confidence
        if bear_risk or market_regime == -1:
            eff_min_conf = min(self.min_confidence + 0.10, 0.95)
        elif market_regime == 0:
            eff_min_conf = self.min_confidence
        qualified = [c for c in qualified if c['confidence'] >= eff_min_conf]

        # === 短期趋势过滤: 防止买入已走弱的股票 ===
        # 大龙地产教训: B3回踩确认 → 缩量阴跌 → 回踩变破位
        # 五个拒绝条件 (任一触发即过滤):
        #   1) 量能枯竭: vol_ratio < -0.30 (≈原始0.50x均量)
        #   2) 下跌加速: 下跌趋势 + 下跌笔早期
        #   3) 弱势震荡: 非上升趋势 + 阴跌
        #   4) B3回踩失败: B3买点 + 缩量阴跌(stroke>0.3+vol<0) → 回踩变破位
        #   5) 力竭追高: exhaustion>0.5 → 极高力竭直接拒绝(不只是降权)
        short_term_rejects = []
        for c in qualified:
            sig = c.get('sig')
            if sig is None:
                continue
            vol_ratio = getattr(sig, 'volume_ratio', 0.0) or 0.0
            trend_type = getattr(sig, 'trend_type', 0) or 0
            stroke_phase = getattr(sig, 'stroke_phase', 0) or 0
            daily_ret = getattr(sig, 'daily_return', 0) or 0
            buy_point = getattr(sig, 'chan_buy_point', 0) or 0
            exhaustion = getattr(sig, 'exhaustion_risk', 0) or 0

            # 条件1: 量能枯竭
            if vol_ratio < -0.30:
                short_term_rejects.append(c)
                continue
            # 条件2: 下跌趋势中加速下行
            if trend_type == -2 and stroke_phase > 0.2:
                short_term_rejects.append(c)
                continue
            # 条件3: 非上升趋势 + 弱势阴跌
            if trend_type != 2 and stroke_phase > 0.2 and daily_ret < 0:
                short_term_rejects.append(c)
                continue
            # 条件4: B3回踩失败 — 缩量+阴跌=回踩变破位
            if buy_point == 3 and stroke_phase > 0.3 and vol_ratio < 0 and daily_ret < 0:
                short_term_rejects.append(c)
                continue
            # 条件5: 极高度力竭 → 直接拒绝, 不给降权机会
            if exhaustion > 0.5:
                short_term_rejects.append(c)
                continue
            # 条件6: 当日涨幅>9% → 涨停板附近不追, 接盘风险太高
            if daily_ret > 9.0:
                short_term_rejects.append(c)
                continue
            # 条件7: B1买点需底背离确认 → 没有背离的B1是在接飞刀
            if buy_point == 1:
                div_type = getattr(sig, 'chan_divergence_type', '') or ''
                div_strength = getattr(sig, 'chan_divergence_strength', 0) or 0
                if 'bottom' not in str(div_type).lower() or div_strength < 0.2:
                    short_term_rejects.append(c)
                    continue
            # 条件8: 高开低走 → 开盘诱多, 收盘翻绿(跌>1.5%+振幅>4%)
            amp = getattr(sig, 'daily_return', 0) or 0
            if daily_ret < -1.5:
                gbc = getattr(sig, 'gap_breakout_confirm', 0) or 0
                if gbc > 0.2:
                    short_term_rejects.append(c)
                    continue
            # 条件9: 利润持续下滑 → 基本面恶化, 技术面再好也不买
            profit_declining = getattr(sig, 'profit_declining', False) if sig else False
            if profit_declining:
                short_term_rejects.append(c)
                continue
            # 条件10: 顶部/隐藏顶部背离 → 历史数据中top背离未来收益-6.38%, hidden_top -4.46%
            div_type = getattr(sig, 'chan_divergence_type', '') or ''
            div_type_lower = str(div_type).lower()
            if 'top' in div_type_lower:
                short_term_rejects.append(c)
                continue

        current_codes = set(current_positions.keys())
        # 已持仓的短期走弱股票不做强制卖出(交给止损模块)
        held_rejects = [c for c in short_term_rejects if c['code'] in current_codes]
        new_rejects = [c for c in short_term_rejects if c['code'] not in current_codes]
        if new_rejects:
            qualified = [c for c in qualified if c not in new_rejects]

        # === Chan买点门控: 结构确认是入场前提 ===
        # 均值回归调整在评分阶段生效，不在门控阶段过滤
        chan_passed = []
        for c in qualified:
            bp = c.get('chan_buy_point', 0)
            sl = c.get('signal_level', 0)
            if sl < 1 and bp <= 0:
                continue
            # 均线趋势检查: EMA20 > EMA60 是缠论买点的方向前提
            sig = c.get('sig')
            ma_up = getattr(sig, 'ma_trend_up', False) if sig else False
            tt = getattr(sig, 'trend_type', 0) if sig else 0
            if bp == 3 and not ma_up:
                continue  # B3是趋势跟随买点, 必须均线多头
            if bp == 2 and tt == -2:
                continue  # B2在下跌趋势中不可靠
            chan_passed.append(c)
        if len(chan_passed) == 0:
            return {}

        # === 板块共振: 计算行业集中度, 用于后续排序惩罚 ===
        industry_chan_count = {}
        for c in chan_passed:
            ind = c.get('industry', '其他')
            industry_chan_count[ind] = industry_chan_count.get(ind, 0) + 1

        qualified = chan_passed

        # 换手控制：现有持仓给予换手惩罚加分，需超过交易成本才能被替换
        hold_threshold = 0.4  # 已持仓股票，rank_pct>0.4即可保留

        # 计算有效得分: 截面排名(score) + 均值回归调整
        # score已包含因子+Chan boost+P0/P1/P2, 无需再算一次Chan加成
        scores = np.array([c['score'] for c in qualified])
        score_rank = pd.Series(scores).rank(pct=True).values  # score截面排名

        for i, c in enumerate(qualified):
            c['is_held'] = c['code'] in current_codes

            sl = c.get('signal_level', 0)
            cb = c.get('chan_buy_point', 0)
            stock_trend = c.get('trend_type', 0)

            # Chan结构作为轻微加分（score已包含主要Chan信息）
            if sl >= 2:
                chan_bonus = 0.04
            elif cb > 0:
                chan_bonus = 0.02
            else:
                chan_bonus = 0.0

            if stock_trend == 2:
                chan_bonus += 0.01

            c['chan_quality'] = 0.5 + chan_bonus
            c['effective_score'] = float(score_rank[i]) + chan_bonus

            # 板块共振惩罚: 孤立B3(行业唯一)扣0.08分
            if industry_chan_count.get(c.get('industry', '其他'), 0) < 2:
                c['effective_score'] -= 0.08

            # 弱势行业惩罚: 历史胜率比<0.85的行业系统性降权
            ind = c.get('industry', '')
            if '医药' in str(ind):
                c['effective_score'] -= 0.10  # W/L=0.79, 最弱行业
            elif '化工' in str(ind):
                c['effective_score'] -= 0.06  # W/L=0.80
            elif '传媒' in str(ind):
                c['effective_score'] -= 0.06  # W/L=0.83

            # DYN因子惩罚: 动态因子avg_score为负, 系统性降权
            fn = getattr(sig, 'factor_name', '') if sig else ''
            if fn and fn.startswith('DYN_'):
                c['effective_score'] -= 0.08

            # ── 均值回归调整（基于历史赢家/输家特征分析） ──
            # 分析发现: 高60日动量的股票未来收益显著更差 (IC=-0.060)
            # 买入"已经涨了很多"的股票是亏损的主要来源
            sig = c.get('sig')
            mom_60d = getattr(sig, 'mom_60d', 0.0) or 0.0
            dist_ma60 = getattr(sig, 'dist_ma60', 0.0) or 0.0
            max_dd_20d = getattr(sig, 'max_dd_20d', 0.0) or 0.0
            vol_regime = getattr(sig, 'vol_regime', 1.0) or 1.0

            # 1. 高动量惩罚: 60日涨幅>20%→未来均值回归风险大
            if mom_60d > 0.25:
                c['effective_score'] -= 0.12   # 严重追高
            elif mom_60d > 0.15:
                c['effective_score'] -= 0.06   # 温和追高
            elif -0.05 <= mom_60d <= 0.05:
                c['effective_score'] += 0.02   # 横盘整理后突破,最佳买点

            # 2. 远离MA60惩罚: 价格远超60日均线→回归压力
            if dist_ma60 > 0.25:
                c['effective_score'] -= 0.10
            elif dist_ma60 > 0.12:
                c['effective_score'] -= 0.05
            elif dist_ma60 < -0.05:
                c['effective_score'] += 0.03   # 低于MA60,超跌反弹机会

            # 3. 波动率扩张奖励: 低波动转高波动→趋势启动信号
            if vol_regime > 1.3:
                c['effective_score'] += 0.03
            elif vol_regime < 0.7:
                c['effective_score'] -= 0.03   # 波动率收缩,缺乏动能

            # 4. 回调买入奖励: 20日最大回撤>10%→超跌反弹
            if max_dd_20d < -0.12:
                c['effective_score'] += 0.04   # 深度回调,反弹空间大
            elif max_dd_20d < -0.06:
                c['effective_score'] += 0.02   # 温和回调

            # 换手加分 (已持仓不易被替换)
            turnover = self.turnover_bonus if (c['is_held'] and c['rank_pct'] > hold_threshold) else 0.0
            if c.get('chan_sell_point', 0) > 0:
                turnover = 0.0
            if c['is_held'] and c['rank_pct'] <= hold_threshold:
                c['is_held'] = False
                turnover = 0.0
            c['effective_score'] += turnover

        # 排序：按有效得分降序（已持仓有换手加分）
        qualified.sort(key=lambda x: -x['effective_score'])

        # === 纯score排序选股 ===
        # 取消行业配额制: 让最好的信号自然集中, 不做摊大饼
        # 行业集中度由权重分配层的 industry_max_weight 控制
        selected = []
        for c in qualified:
            if len(selected) >= n_positions:
                break
            if c not in selected:
                selected.append(c)

        # === 动态最小持仓（小说：熊市允许空仓）===
        min_pos = self._get_min_positions(market_regime)
        if len(selected) < min_pos:
            # 候选不足 → 宁愿空仓，不降低标准
            # "不会空仓的人永远长不大"
            return {}

        if not selected:
            return {}

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
            regime_key = {1: 'bull', -1: 'bear', 0: 'neutral'}.get(market_regime, 'neutral')
            raw_weights = []
            for i, c in enumerate(selected):
                ic_dict = self.industry_ic.get(c['industry'], {})
                ic = ic_dict.get(regime_key, ic_dict.get('neutral', 0.05))
                # 排名权重: 线性衰减 1.0 → 0.5
                rank_w = 1.0 - 0.3 * (i / max(n - 1, 1))
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
            weights = [target_exposure / n] * n

        # 单票权重上限: 候选不足时动态放宽以提升资金利用率
        max_single = self.max_single_weight_from_cfg
        if len(selected) < self.MAX_POSITIONS:
            # 候选不足 → 提高单票上限, 集中资金到优质标的
            slack = (self.MAX_POSITIONS - len(selected)) / self.MAX_POSITIONS
            max_single = min(max_single * (1.0 + slack * 1.5), 0.18)  # 最多放宽到18%
        capped_weights = [min(w, max_single) for w in weights]

        # === 力竭风险过滤: 高位+背离的股票降权 ===
        exhaustion_tags = set()
        for i, c in enumerate(selected):
            sig = c.get('sig')
            er = getattr(sig, 'exhaustion_risk', 0) if sig else 0
            if er > 0.3:
                # 力竭风险高: 权重降至 exhaustion_max_weight 以内
                capped_weights[i] = min(capped_weights[i], self.exhaustion_max_weight)
                exhaustion_tags.add(c['code'])
            elif er > 0.15:
                # 中度力竭: 权重打折
                capped_weights[i] *= self.exhaustion_reduce_mult

        # === 行业集中度上限: 单行业总权重不超过 industry_max_weight ===
        industry_weights = {}
        for i, c in enumerate(selected):
            ind = c.get('industry', '其他')
            if ind not in industry_weights:
                industry_weights[ind] = 0.0
            industry_weights[ind] += capped_weights[i]

        for ind, total_w in industry_weights.items():
            if total_w > self.industry_max_weight:
                scale = self.industry_max_weight / total_w
                for i, c in enumerate(selected):
                    if c.get('industry', '其他') == ind:
                        capped_weights[i] *= scale

        total_capped = sum(capped_weights) + 1e-10
        weights = [w / total_capped * target_exposure for w in capped_weights]

        # === 最小交易单位检查: 确保每只至少1手 + 过滤买不起的 ===
        desired_value = {}
        valid_selected = []
        valid_weights = []
        for c, w in zip(selected, weights):
            val = w * total_equity
            min_lot = c['price'] * 100
            if val < min_lot:
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
                desired_value[c['code']] = c['weight'] * total_equity

        # 用 valid_selected 替换 selected (用于日志记录)
        selected = valid_selected

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
                'chan_bonus': c.get('chan_bonus', 0.0),
                'effective_score': c.get('effective_score', c['score']),
                'is_held': c.get('is_held', False),
                'confidence': c.get('confidence', 1.0),
            }
            for c in selected
        ]

        return desired_value

    def build(
        self,
        date,
        universe,
        current_positions,
        signal_store,
        cash,
        prices,
        market_regime,
        momentum_score=0.0,
        bear_risk=False,
        trend_score=0.0,
        cost=None,
        rebalance=False,
    ):
        """构建目标持仓（外部接口）

        关键设计: 非再平衡日只做个股成本止损, 不做信号止损
        """
        stop_loss_sells = {}
        profit_reduce = {}  # 止盈减仓 (code -> target_pct_of_current)
        total_equity = cash + sum(current_positions.values())

        # === 缠论止盈：分层退出 + 中枢移动止盈 (非调仓日也生效) ===
        chan_force_sells = {}
        for code, current_value in current_positions.items():
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
                # S1 (一卖) → 全清或按配置减仓
                elif cs == 1:
                    if self.chan_tp_s1_reduce >= 1.0:
                        chan_force_sells[code] = 0.0
                        self._entry_dates.pop(code, None)
                        self._entry_reasons.pop(code, None)
                        self._peak_prices.pop(code, None)
                    else:
                        if code not in profit_reduce:
                            profit_reduce[code] = self.chan_tp_s1_reduce
                # S2 (二卖) → 按配置减仓
                elif cs == 2:
                    if self.chan_tp_s2_reduce >= 1.0:
                        chan_force_sells[code] = 0.0
                        self._entry_dates.pop(code, None)
                        self._entry_reasons.pop(code, None)
                        self._peak_prices.pop(code, None)
                    elif code not in profit_reduce:
                        profit_reduce[code] = self.chan_tp_s2_reduce
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
        if self.chan_tp_enabled and self.chan_tp_pivot_trailing:
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
                        # 无中枢但有盈利：启动价格移动止盈
                        if code in self._peak_prices and self._peak_prices[code] > 0:
                            peak = self._peak_prices[code]
                            drawdown = (peak - current_price) / peak
                            if drawdown > self.trailing_stop_pct:
                                profit_reduce[code] = 0.5

        # === 均值回归退出: 持仓中极端乖离 → 减仓/清仓 (非调仓日也生效) ===
        for code, current_value in current_positions.items():
            if code in chan_force_sells or code in profit_reduce:
                continue
            if code not in prices or current_value <= 0:
                continue
            sig = signal_store.get(code, date)
            if sig is None:
                continue
            mom_60d = getattr(sig, 'mom_60d', 0.0) or 0.0
            dist_ma60 = getattr(sig, 'dist_ma60', 0.0) or 0.0
            # 60日涨幅>50% → 均值回归压力极大 → 清仓锁定利润
            if mom_60d > 0.50:
                chan_force_sells[code] = 0.0
                self._entry_dates.pop(code, None)
                self._entry_reasons.pop(code, None)
                self._peak_prices.pop(code, None)
            # 距MA60>40% → 严重乖离 → 清仓
            elif dist_ma60 > 0.40:
                chan_force_sells[code] = 0.0
                self._entry_dates.pop(code, None)
                self._entry_reasons.pop(code, None)
                self._peak_prices.pop(code, None)
            # 60日涨幅>30% → 温和超买 → 减仓50%
            elif mom_60d > 0.30:
                if code not in profit_reduce:
                    profit_reduce[code] = 0.5
            # 距MA60>25% → 乖离偏大 → 减仓30%
            elif dist_ma60 > 0.25:
                if code not in profit_reduce:
                    profit_reduce[code] = 0.3

        # 个股止损检查 - 成本止损 + 时间止损 + 移动止损
        for code, current_value in current_positions.items():
            if code not in prices or current_value <= 0:
                continue

            current_price = prices[code]
            stopped = False
            stop_reason = ""

            # === Chan理论保护：买入点区域放宽止损（防洗盘） ===
            # B1(抄底)反转失败率高 → 不放宽；B2/B3(趋势确认后) → 可放宽
            sig = signal_store.get(code, date)
            chan_div_type = getattr(sig, 'chan_divergence_type', '') if sig else ''
            chan_div_strength = getattr(sig, 'chan_divergence_strength', 0.0) if sig else 0.0
            chan_buy_point = getattr(sig, 'chan_buy_point', 0) if sig else 0
            chan_sell_point = getattr(sig, 'chan_sell_point', 0) if sig else 0
            sl = getattr(sig, 'signal_level', 0) if sig else 0
            # B2/B3：趋势已确认，放宽止损防洗盘
            # B1：抄底信号，反转可能失败，不放宽
            if chan_buy_point in (2, 3) and chan_div_strength > 0.2:
                chan_protection = True
                chan_protection_mult = 1.5  # B2/B3适度放宽
            elif sl >= 2 and chan_div_strength > 0.2:
                chan_protection = True
                chan_protection_mult = 1.5  # 线段级买入适度放宽
            elif chan_div_type in ('bottom', 'hidden_bottom') and chan_div_strength > 0.3:
                chan_protection = True
                chan_protection_mult = 1.3  # 底背离温和放宽
            else:
                chan_protection = False
                chan_protection_mult = 1.0

            # 1. 成本止损: 亏损超过position_stop_loss
            if cost and code in cost and len(cost[code]) >= 2 and cost[code][0] > 0:
                avg_cost = cost[code][1]
                pnl_pct = (current_price - avg_cost) / avg_cost

                # 波动率自适应调整：高波动股票放宽止损线
                vol = getattr(sig, 'risk_vol', 0.03) if sig else 0.03
                adaptive_mult = self.volatility_adaptive_mult
                adaptive_stop = self.position_stop_loss * (1 + vol * adaptive_mult * 10)
                adaptive_stop = min(adaptive_stop, self.position_stop_loss * 1.5)

                # Chan保护：B2/B3放宽1.5x(非2x), B1不放宽
                if chan_protection:
                    adaptive_stop = adaptive_stop * chan_protection_mult

                if pnl_pct < -adaptive_stop:
                    stopped = True
                    stop_reason = "cost_stop"

            # 2. 时间止损: 持仓超N天且微亏/微赚
            if not stopped and code in self._entry_dates:
                if isinstance(date, date_type):
                    days_held = (date - self._entry_dates[code]).days
                else:
                    days_held = 0
                if days_held > self.time_stop_days:
                    if cost and code in cost and len(cost[code]) >= 2 and cost[code][0] > 0:
                        avg_cost = cost[code][1]
                        pnl_pct = (current_price - avg_cost) / avg_cost
                        if pnl_pct < self.time_stop_min_return:
                            stopped = True
                            stop_reason = "time_stop"

            # 3. 移动止损: 从最高点回撤超过阈值
            # 注意: 若已被中枢移动止盈标记为 profit_reduce (减半), 则不再全平,
            # 避免同一阈值下全平覆盖减半逻辑
            if not stopped and self.trailing_stop_enabled and code in self._peak_prices \
               and code not in profit_reduce:
                peak = self._peak_prices[code]
                drawdown_from_peak = (peak - current_price) / peak
                if drawdown_from_peak > self.trailing_stop_pct:
                    stopped = True
                    stop_reason = "trailing_stop"

            # === Chan理论退出检查（czsc增强版）：买卖点 + 多级别确认 + 背离 + 趋势耗尽 ===
            if not stopped:
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

            # === 买入理由消失检查：入场时依赖的结构条件不再满足 ===
            if not stopped and code in self._entry_reasons:
                reason = self._entry_reasons[code]
                entry_bp = reason.get('buy_point', 0)
                entry_sl = reason.get('signal_level', 0)
                entry_trend = reason.get('trend_type', 0)

                # B3入场: 价格跌破中枢下沿 → 三买失败
                if entry_bp == 3:
                    pivot_zd = getattr(sig, 'chan_pivot_zd', float('nan')) if sig else float('nan')
                    if not np.isnan(pivot_zd) and pivot_zd > 0:
                        if current_price < pivot_zd * 0.995:
                            stopped = True
                            stop_reason = "b3_zg_break"
                # B2入场: 趋势转为下跌 → 二买失效
                elif entry_bp == 2:
                    if trend_type == -2:
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
                if not stopped:
                    curr_bp = getattr(sig, 'chan_buy_point', 0) if sig else 0
                    curr_sl = getattr(sig, 'signal_level', 0) if sig else 0
                    curr_div = getattr(sig, 'chan_divergence_strength', 0.0) if sig else 0.0
                    if curr_bp == 0 and curr_sl == 0 and curr_div < 0.1:
                        if entry_bp > 0 or entry_sl >= 1:
                            # 入场理由完全消失 → 不管盈亏退出
                            stopped = True
                            stop_reason = "entry_reason_lost"

            if stopped:
                stop_loss_sells[code] = 0.0
                # 清理追踪状态
                self._entry_dates.pop(code, None)
                self._entry_reasons.pop(code, None)
                self._peak_prices.pop(code, None)

            # 更新峰值价格
            if code in self._entry_dates:
                self._peak_prices[code] = max(self._peak_prices.get(code, 0), current_price)

        desired_value = {}
        if rebalance:
            desired_value = self._build_desired_value(
                date=date,
                universe=universe,
                current_positions=current_positions,
                signal_store=signal_store,
                cash=cash,
                prices=prices,
                market_regime=market_regime,
                momentum_score=momentum_score,
                bear_risk=bear_risk,
                trend_score=trend_score,
            )

        # 强制卖出 (止损 + Chan卖出)
        for code in stop_loss_sells:
            desired_value[code] = 0.0
        for code in chan_force_sells:
            desired_value[code] = 0.0

        # 止盈减仓
        for code, reduce_pct in profit_reduce.items():
            if code in desired_value:
                desired_value[code] *= (1 - reduce_pct)
            elif code in current_positions:
                desired_value[code] = current_positions[code] * (1 - reduce_pct)

        adjusted = {}

        # 非调仓日保持现有持仓(除非成本止损或Chan卖出信号)
        if not rebalance and not stop_loss_sells and not chan_force_sells:
            adjusted = deepcopy(current_positions)

        # 执行目标持仓
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

            # 渐进进出
            if diff > 0:
                move = self.entry_speed * diff
            else:
                move = self.exit_speed * diff

            adjusted[code] = current + move

        # 再平衡日：清仓不在目标中的旧持仓，并更新入场追踪
        if rebalance:
            for code, current in list(current_positions.items()):
                if code in adjusted:
                    continue
                if code in desired_value:
                    continue
                # 最短持仓保护：新买入的持仓未满min_hold_days不卖出
                if code in self._entry_dates:
                    if isinstance(date, date_type):
                        days_held = (date - self._entry_dates[code]).days
                    else:
                        days_held = 0
                    if days_held < self._min_hold_days:
                        # 保留持仓，不卖出
                        adjusted[code] = current
                        continue
                adjusted[code] = 0.0
                # 清理追踪
                self._entry_dates.pop(code, None)
                self._entry_reasons.pop(code, None)
                self._peak_prices.pop(code, None)

            # 记录新入场持仓
            for code, target_val in adjusted.items():
                if target_val > 0 and code not in self._entry_dates:
                    self._entry_dates[code] = date
                    self._peak_prices[code] = prices.get(code, 0)
                    # 记录入场理由（买入理由消失时用于退出判断）
                    sig = signal_store.get(code, date)
                    if sig:
                        self._entry_reasons[code] = {
                            'buy_point': getattr(sig, 'chan_buy_point', 0),
                            'signal_level': getattr(sig, 'signal_level', 0),
                            'trend_type': getattr(sig, 'trend_type', 0),
                        }

        return adjusted
