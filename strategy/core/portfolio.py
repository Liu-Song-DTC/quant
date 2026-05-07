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

    MAX_POSITIONS = 8

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

        # === 绝对质量门槛（小说：龙头洁癖）===
        sel_config = portfolio_config.get('selection', {})
        self.min_rank_pct = sel_config.get('min_rank_pct', 0.5)
        self.min_absolute_score = sel_config.get('min_absolute_score', 0.15)
        self.min_confidence = sel_config.get('min_confidence', 0.80)

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

        # 追踪每个持仓的入场时间和峰值价格
        self._entry_dates: dict = {}  # {code: date}
        self._peak_prices: dict = {}  # {code: peak_price_since_entry}

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
        self._min_hold_days = 7  # 最短持仓天数，防止洗盘whipsaw

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

        每2.5万资金支持1个仓位, 范围[3, 8]
        """
        n = int(total_equity / 25000)
        return max(3, min(n, PortfolioConstructor.MAX_POSITIONS))

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

            # 价格约束：100股整手下，100*price必须不超过理想仓位
            # 允许2倍理想仓位（100股整手会导致超配，但至少能买入）
            price = prices.get(code, 0)
            ideal_per_stock = total_equity / n_positions
            if price > 0 and price * 100 > ideal_per_stock * 2:
                continue

            candidates.append({
                'code': code,
                'factor_value': factor_value,
                'score': getattr(sig, 'score', 0),
                'industry': getattr(sig, 'industry', '') or 'default',
                'risk_vol': getattr(sig, 'risk_vol', 0.03),
                'price': price,
                'sig': sig,
            })

        if not candidates:
            return {}

        # === 截面排名 ===
        factor_values = np.array([c['factor_value'] for c in candidates])
        rank_pct = pd.Series(factor_values).rank(pct=True)

        for i, c in enumerate(candidates):
            c['rank_pct'] = rank_pct.iloc[i]

        # === 市场仓位调整（简化版 v4）===
        # 核心风控：渐进式熊市乘数 + 指数动量辅助
        # 移除 fv_exposure（截面因子均值噪声大）和 trend_adj（与 bear_risk 高度相关）
        market_signal = 1.0

        # 指数动量辅助：强负动量时降仓（独立于 bear_risk 的数据源）
        if momentum_score < -0.5:
            mom_adj = 0.6 + 0.4 * (momentum_score + 1.0) / 0.5  # [-1,-0.5] → [0.6, 1.0]
            market_signal = min(market_signal, max(mom_adj, 0.6))

        # 渐进式熊市乘数：根据回撤深度分档
        if bear_risk:
            if drawdown > 0.20:          # 深熊：回撤>20%
                floor, ceiling = 0.50, 0.65
            elif drawdown > 0.10:        # 浅熊：回撤10-20%
                floor, ceiling = 0.60, 0.75
            else:                         # 风险预警但未深跌
                floor, ceiling = 0.70, 0.85
        elif market_regime == 1:  # bull
            floor, ceiling = 0.80, 1.0
        else:  # neutral
            floor, ceiling = 0.70, 1.0
        target_exposure = float(np.clip(market_signal, floor, ceiling))

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
            min_rank = 0.40
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
        for c in qualified:
            sig = c['sig']
            confidence = 1.0
            fq = getattr(sig, 'factor_quality', 0)
            if fq > 0.05:
                confidence += fq * 0.5
            fn = getattr(sig, 'factor_name', '')
            if '_V' in (fn or ''):
                confidence += 0.15
            if fn and fn.startswith('DYN_'):
                confidence += 0.05
            c['confidence'] = np.clip(confidence, 0.7, 1.3)

        # 信心不足的股票淘汰（熊市提高门槛）
        eff_min_conf = self.min_confidence
        if bear_risk or market_regime == -1:
            eff_min_conf = min(self.min_confidence + 0.10, 0.95)
        elif market_regime == 0:
            eff_min_conf = self.min_confidence
        qualified = [c for c in qualified if c['confidence'] >= eff_min_conf]

        # 换手控制：现有持仓给予换手惩罚加分，需超过交易成本才能被替换
        current_codes = set(current_positions.keys())
        hold_threshold = 0.4  # 已持仓股票，rank_pct>0.4即可保留

        # 计算有效得分：已持仓加分(turnover_bonus)，需超越此加分才能替换
        for c in qualified:
            c['is_held'] = c['code'] in current_codes
            if c['is_held'] and c['rank_pct'] > hold_threshold:
                c['effective_score'] = c['factor_value'] + self.turnover_bonus
            elif c['is_held'] and c['rank_pct'] <= hold_threshold:
                c['is_held'] = False
                c['effective_score'] = c['factor_value']
            else:
                c['effective_score'] = c['factor_value']

        # 排序：按有效得分降序（已持仓有换手加分）
        qualified.sort(key=lambda x: -x['effective_score'])

        # === 多轮行业分散选股 ===
        # Round 1: 每行业只选最强冠军（保证多样性）
        # Round 2: 若仓位未满，从最强行业中补充亚军
        MAX_PER_INDUSTRY = 1
        MIN_INDUSTRIES = 4

        # Round 1: 行业冠军
        industry_champions = {}
        for c in qualified:
            ind = c['industry']
            if ind not in industry_champions:
                industry_champions[ind] = c
            elif c['effective_score'] > industry_champions[ind]['effective_score']:
                industry_champions[ind] = c

        # 按冠军质量排序行业
        ind_by_quality = sorted(
            industry_champions.items(),
            key=lambda x: x[1]['effective_score'],
            reverse=True
        )

        # Round 1 选择：每个行业首位冠军
        selected = []
        industry_count = {}
        for ind, champion in ind_by_quality:
            if len(selected) >= n_positions:
                break
            selected.append(champion)
            industry_count[ind] = 1

        # Round 2: 若仓位未满且行业覆盖足够，从最强行业补充亚军
        if len(selected) < n_positions and len(industry_count) >= MIN_INDUSTRIES:
            for c in qualified:
                ind = c['industry']
                if industry_count.get(ind, 0) >= MAX_PER_INDUSTRY + 1:
                    continue
                if c in selected:
                    continue
                selected.append(c)
                industry_count[ind] = industry_count.get(ind, 0) + 1
                if len(selected) >= n_positions:
                    break

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

        # === 权重分配（Kelly最优 + 风险平价）===
        n = len(selected)
        if self.risk_parity_enabled:
            # 协方差风险平价（优先）
            rp_weights = self._risk_parity_weights(selected, total_equity)
            weights = [w * target_exposure for w in rp_weights]
        elif self.industry_ic:
            # Kelly最优：仓位 = edge / variance（半凯利上限0.5）
            regime_key = {1: 'bull', -1: 'bear', 0: 'neutral'}.get(market_regime, 'neutral')
            raw_weights = []
            for c in selected:
                ic_dict = self.industry_ic.get(c['industry'], {})
                ic = ic_dict.get(regime_key, ic_dict.get('neutral', 0.05))
                # Edge: IC × |signal_strength|
                signal = np.clip(abs(c.get('factor_value', 0)), 0, 1)
                edge = max(ic * signal, 0.001)
                # Variance: volatility²
                vol = c.get('risk_vol', 0.03)
                variance = max(vol ** 2, 0.0001)
                # Half-Kelly fraction
                kelly_fraction = min(edge / variance * 0.01, 0.5)
                # 情绪乘数
                if self.sentiment_multipliers and c['industry'] in self.sentiment_multipliers:
                    kelly_fraction *= self.sentiment_multipliers[c['industry']]
                # 信心加权
                kelly_fraction *= c.get('confidence', 1.0)
                raw_weights.append(max(kelly_fraction, 0.01))
            total_w = sum(raw_weights) + 1e-10
            weights = [w / total_w * target_exposure for w in raw_weights]
        else:
            weights = [target_exposure / n] * n

        desired_value = {}
        for c, w in zip(selected, weights):
            val = w * total_equity
            # 确保至少1手（100股），避免整手取整后变为0
            min_lot = c['price'] * 100
            if val < min_lot:
                val = min_lot
            desired_value[c['code']] = val

        # 记录选股结果
        self.last_selection = [
            {
                'date': date,
                'code': c['code'],
                'score': c['score'],
                'weight': w,
                'industry': c.get('industry', ''),
                'rank_pct': c.get('rank_pct', 0),
            }
            for c, w in zip(selected, weights)
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
        total_equity = cash + sum(current_positions.values())

        # 个股止损检查 - 成本止损 + 时间止损 + 移动止损
        for code, current_value in current_positions.items():
            if code not in prices or current_value <= 0:
                continue

            current_price = prices[code]
            stopped = False
            stop_reason = ""

            # === Chan理论保护：买入点区域放宽止损（防洗盘） ===
            chan_protection = False
            sig = signal_store.get(code, date)
            chan_div_type = getattr(sig, 'chan_divergence_type', '') if sig else ''
            chan_div_strength = getattr(sig, 'chan_divergence_strength', 0.0) if sig else 0.0
            if chan_div_type in ('bottom', 'hidden_bottom') and chan_div_strength > 0.3:
                chan_protection = True

            # 1. 成本止损: 亏损超过position_stop_loss
            if cost and code in cost and len(cost[code]) >= 2 and cost[code][0] > 0:
                avg_cost = cost[code][1]
                pnl_pct = (current_price - avg_cost) / avg_cost

                # 波动率自适应调整：高波动股票放宽止损线
                vol = getattr(sig, 'risk_vol', 0.03) if sig else 0.03
                adaptive_mult = self.volatility_adaptive_mult
                adaptive_stop = self.position_stop_loss * (1 + vol * adaptive_mult * 10)
                adaptive_stop = min(adaptive_stop, self.position_stop_loss * 1.5)

                # Chan理论保护：底背离买入点区域放宽止损2倍（防洗盘）
                if chan_protection:
                    adaptive_stop = adaptive_stop * 2.0

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
            if not stopped and self.trailing_stop_enabled and code in self._peak_prices:
                peak = self._peak_prices[code]
                drawdown_from_peak = (peak - current_price) / peak
                if drawdown_from_peak > self.trailing_stop_pct:
                    stopped = True
                    stop_reason = "trailing_stop"

            # === Chan理论退出检查：背离 + 趋势耗尽 ===
            if not stopped:
                # 获取当前信号
                sig = signal_store.get(code, date)
                chan_div_type = getattr(sig, 'chan_divergence_type', '') if sig else ''
                chan_div_strength = getattr(sig, 'chan_divergence_strength', 0.0) if sig else 0.0
                chan_struct_score = getattr(sig, 'chan_structure_score', 0.0) if sig else 0.0

                # 顶背离退出：价格高位 + MACD顶背离 → 提前止盈
                if chan_div_type == 'top' and chan_div_strength > 0.4:
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

            if stopped:
                stop_loss_sells[code] = 0.0
                # 清理追踪状态
                self._entry_dates.pop(code, None)
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

        # 强制卖出
        for code in stop_loss_sells:
            desired_value[code] = 0.0

        adjusted = {}

        # 非调仓日保持现有持仓(除非成本止损)
        if not rebalance and not stop_loss_sells:
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
                self._peak_prices.pop(code, None)

            # 记录新入场持仓
            for code, target_val in adjusted.items():
                if target_val > 0 and code not in self._entry_dates:
                    self._entry_dates[code] = date
                    self._peak_prices[code] = prices.get(code, 0)

        return adjusted
