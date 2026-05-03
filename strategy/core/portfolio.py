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

    MIN_POSITIONS = 2
    MAX_POSITIONS = 5

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
        self.entry_speed = entry_speed if entry_speed is not None else portfolio_config.get('entry_speed', 1.0)
        self.exit_speed = exit_speed if exit_speed is not None else portfolio_config.get('exit_speed', 1.0)

        # === 波动率控制 ===
        self.vol_control_enabled = portfolio_config.get('volatility_control_enabled', True)
        self.target_volatility = target_volatility if target_volatility is not None else portfolio_config.get('target_volatility', 0.15)
        self.vol_lookback = portfolio_config.get('volatility_control_lookback', 20)
        self._daily_returns = deque(maxlen=self.vol_lookback * 2)  # store up to 2x for safety

        # === 组合止损 ===
        self.stop_loss_enabled = portfolio_config.get('portfolio_stop_loss_enabled', True)
        self.portfolio_stop_loss = portfolio_stop_loss if portfolio_stop_loss is not None else portfolio_config.get('portfolio_stop_loss', 0.08)
        self.emergency_exposure = portfolio_config.get('emergency_exposure', 0.3)

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
        self._min_hold_days = 5  # 最短持仓天数，防止whipsaw

    def set_sentiment_multipliers(self, multipliers: dict):
        """设置行业情绪乘数，用于调整行业 IC 权重

        Args:
            multipliers: {industry: multiplier}, 范围 [0.8, 1.2]
        """
        self.sentiment_multipliers = multipliers

    def update_returns(self, daily_return: float):
        """记录每日收益率（用于波动率控制）

        从 BacktraderExecution.next() 中调用，每次传入当日收益率
        """
        self._daily_returns.append(daily_return)

    @staticmethod
    def _calc_max_position(total_equity: float, prices: dict) -> int:
        """根据资金自动计算最大持仓数 - 超级集中持仓

        每4万资金支持1个仓位, 范围[2, 5]
        100k→2-3只(极致集中)
        """
        n = int(total_equity / 40000)
        return max(PortfolioConstructor.MIN_POSITIONS, min(n, PortfolioConstructor.MAX_POSITIONS))

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

        # === 市场仓位调整 ===
        # 多信号融合择时: fv_mean(截面) + momentum(指数动量) + trend(均线排列)
        fv_mean = float(np.mean(factor_values))
        # fv_mean连续映射到[exposure_min, exposure_max]，使用可配置参数
        fv_range = self.fv_high - self.fv_low
        fv_exposure = self.fv_exposure_min + (fv_mean - self.fv_low) / fv_range * (self.fv_exposure_max - self.fv_exposure_min)
        fv_exposure = float(np.clip(fv_exposure, self.fv_exposure_min, self.fv_exposure_max))

        # momentum_score融合: 强负动量时额外降仓（提高阈值，减少不必要的降仓）
        # momentum ∈ [-1, 1], 当momentum<-0.5时开始降仓
        mom_adj = 1.0
        if momentum_score < -0.5:
            mom_adj = 0.6 + 0.4 * (momentum_score + 1.0) / 0.5  # [-1,-0.5] → [0.6, 1.0]
            mom_adj = max(mom_adj, 0.6)

        # trend_score融合: 空头排列时额外降仓（提高下限）
        trend_adj = 1.0
        if trend_score < 0:
            trend_adj = 0.8 + 0.2 * (1.0 + trend_score)  # [-1, 0] → [0.8, 1.0]

        target_exposure = fv_exposure * mom_adj * trend_adj
        # 渐进式熊市乘数：根据回撤深度分档，而非熊市/中性/牛市一刀切
        # 解决熊市一刀切 0.15-0.30 在 V 型反弹中滞后的问题
        if bear_risk:
            if drawdown > 0.20:          # 深熊：回撤>20%
                floor, ceiling = 0.25, 0.40
            elif drawdown > 0.10:        # 浅熊：回撤10-20%
                floor, ceiling = 0.45, 0.65
            else:                         # 风险预警但未深跌
                floor, ceiling = 0.55, 0.75
        elif market_regime == 1:  # bull
            floor = 0.8
            ceiling = 1.0
        else:  # neutral
            floor = 0.65
            ceiling = 1.0
        target_exposure = float(np.clip(target_exposure, floor, ceiling))

        # === 波动率控制: 实现波动率超过目标时降仓 ===
        if self.vol_control_enabled and len(self._daily_returns) >= self.vol_lookback:
            recent_rets = list(self._daily_returns)[-self.vol_lookback:]
            realized_vol = float(np.std(recent_rets) * np.sqrt(252))  # 年化
            if realized_vol > 0.01:
                vol_scale = self.target_volatility / realized_vol
                target_exposure *= float(np.clip(vol_scale, 0.5, 1.0))

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

        # === 选股: 固定门槛 → top N → 行业均衡 ===
        # 取消三层 fallback，弱信号时期宁愿少选股也不填弱仓
        # 熊市自动降低 rank 门槛以释放更多候选，牛市保持高门槛
        if bear_risk or market_regime == -1:
            min_rank = 0.40  # 熊市放宽：更多候选可供选择
        elif market_regime == 1:
            min_rank = 0.55  # 牛市收紧：只选最强
        else:
            min_rank = 0.50  # 中性

        qualified = [c for c in candidates if c['rank_pct'] > min_rank]

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

        # 行业均衡选股：行业上限随持仓数动态调整
        industry_count = {}
        industry_cap = max(2, int(np.ceil(n_positions / 3)))

        selected = []
        for c in qualified:
            ind = c['industry']
            if industry_count.get(ind, 0) >= industry_cap:
                continue
            selected.append(c)
            industry_count[ind] = industry_count.get(ind, 0) + 1
            if len(selected) >= n_positions:
                break

        if not selected:
            return {}

        # 记录
        self.current_ranking = {c['code']: i for i, c in enumerate(candidates)}
        self.current_n_positions = len(selected)

        # === 权重分配（IC加权 or 风险平价）===
        n = len(selected)
        if self.risk_parity_enabled:
            rp_weights = self._risk_parity_weights(selected, total_equity)
            weights = [w * target_exposure for w in rp_weights]
        elif self.industry_ic:
            regime_key = {1: 'bull', -1: 'bear', 0: 'neutral'}.get(market_regime, 'neutral')
            max_ic = 0.05
            for c in selected:
                ic_dict = self.industry_ic.get(c['industry'], {})
                ic_val = ic_dict.get(regime_key, ic_dict.get('neutral', 0.05))
                max_ic = max(max_ic, ic_val)

            raw_weights = []
            for c in selected:
                ic_dict = self.industry_ic.get(c['industry'], {})
                ic = ic_dict.get(regime_key, ic_dict.get('neutral', 0.05))
                ic_w = (ic / max_ic) ** 1.5
                # 情绪乘数调整
                if self.sentiment_multipliers and c['industry'] in self.sentiment_multipliers:
                    ic_w *= self.sentiment_multipliers[c['industry']]
                raw_weights.append(ic_w)
            total_w = sum(raw_weights)
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

            # 1. 成本止损: 亏损超过position_stop_loss
            if cost and code in cost and len(cost[code]) >= 2 and cost[code][0] > 0:
                avg_cost = cost[code][1]
                pnl_pct = (current_price - avg_cost) / avg_cost

                # 波动率自适应调整：高波动股票放宽止损线
                sig = signal_store.get(code, date)
                vol = getattr(sig, 'risk_vol', 0.03) if sig else 0.03
                adaptive_mult = self.volatility_adaptive_mult
                adaptive_stop = self.position_stop_loss * (1 + vol * adaptive_mult * 10)
                adaptive_stop = min(adaptive_stop, self.position_stop_loss * 1.5)

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
