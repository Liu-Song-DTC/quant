# core/portfolio.py
import numpy as np
from copy import deepcopy

class PortfolioConstructor:
    """
    Signal -> Risk Budget -> Target Position
    """

    def __init__(
        self,
        max_position=10,

        # 风险控制 - 控制波动
        target_volatility=0.10,   # 保持适度波动

        # 动态调仓
        entry_speed=0.6,          # 提高入场速度
        exit_speed=1.0,           # 出场更果断

        # 止损参数 - 放宽止损以减少误杀
        position_stop_loss=0.15,  # 放宽单股止损线
        portfolio_stop_loss=0.12, # 放宽组合止损线
    ):
        self.max_position = max_position
        self.target_volatility = target_volatility

        self.entry_speed = entry_speed
        self.exit_speed = exit_speed

        self.position_stop_loss = position_stop_loss
        self.portfolio_stop_loss = portfolio_stop_loss

        self.defensive_mode = False
        self.peak_equity = None
        self.position_cost = {}  # 记录持仓成本

    # =====================================================
    # 主构建函数
    # =====================================================

    def _build_desired_value(
        self,
        date,
        universe,
        current_positions,
        signal_store,
        cash,
        prices,
        market_regime,
    ):
        # 根据市场状态调整敞口 - 保守一些，追求稳健收益
        if market_regime == 1:
            max_gross_exposure = 0.75  # 牛市适度
        elif market_regime == 0:
            max_gross_exposure = 0.50  # 震荡市谨慎
        else:
            max_gross_exposure = 0.15  # 熊市极度保守

        total_equity = cash + sum(current_positions.values())
        if self.peak_equity is None:
            self.peak_equity = total_equity
        self.peak_equity = max(self.peak_equity, total_equity)

        drawdown = 1 - total_equity / self.peak_equity

        # 回撤控制 - 更平滑的递减
        if drawdown > 0.08:
            max_gross_exposure *= 0.3
        elif drawdown > 0.05:
            max_gross_exposure *= 0.55
        elif drawdown > 0.03:
            max_gross_exposure *= 0.75

        # =====================================================
        #  选股（只选 buy 且 score>0，提高阈值）
        # =====================================================

        candidates = []
        for code in universe:
            sig = signal_store.get(code, date)
            # 买入门槛根据市场状态调整 - 保持较高门槛
            min_score = 0.25 if market_regime >= 0 else 0.35
            if sig and sig.buy and sig.score > min_score:
                candidates.append((sig.score, code))

        candidates.sort(reverse=True)
        # 熊市减少持仓数量
        effective_max = self.max_position if market_regime >= 0 else max(2, self.max_position // 2)
        selected = [c for _, c in candidates[: effective_max]]

        if not selected:
            return {}

        # =====================================================
        #  风险权重（score / vol）- 波动率倒数加权
        # =====================================================

        raw_weight = {}
        for code in selected:
            sig = signal_store.get(code, date)
            if sig.risk_vol > 0:
                # 使用 score / vol 作为权重，低波动股票获得更高权重
                raw_weight[code] = sig.score / sig.risk_vol

        if not raw_weight:
            return {}

        total_raw = sum(raw_weight.values())

        # 归一化
        weights = {c: w / total_raw for c, w in raw_weight.items()}

        # =====================================================
        #  目标波动控制（真正使用 target_volatility）
        # =====================================================

        # 估算组合当前平均波动
        portfolio_vol = np.sqrt(
            sum(
                (weights[c] * signal_store.get(c, date).risk_vol) ** 2
                for c in weights
            )
        )

        if portfolio_vol > self.target_volatility:
            scale = self.target_volatility / portfolio_vol
        else:
            scale = 1.0

        for c in weights:
            weights[c] *= scale

        # =====================================================
        #  总仓位限制
        # =====================================================

        gross = sum(abs(w) for w in weights.values())
        if gross > max_gross_exposure:
            scale = max_gross_exposure / gross
            for c in weights:
                weights[c] *= scale

        # =====================================================
        #  转换为目标市值
        # =====================================================

        desired_value = {
            c: weights[c] * total_equity
            for c in weights
        }
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
        cost,
        rebalance,
    ):
        # =====================================================
        #  止损检查（每天执行，不依赖 rebalance）
        # =====================================================
        stop_loss_sells = {}
        total_equity = cash + sum(current_positions.values())

        for code, current_value in current_positions.items():
            if code not in prices or current_value <= 0:
                continue

            # 检查单股止损
            if code in cost and len(cost[code]) >= 2 and cost[code][0] > 0:
                avg_cost = cost[code][1]
                current_price = prices[code]
                pnl_pct = (current_price - avg_cost) / avg_cost

                if pnl_pct < -self.position_stop_loss:
                    # 触发止损，清空该仓位
                    stop_loss_sells[code] = 0.0

            # 检查信号是否转为卖出
            sig = signal_store.get(code, date)
            if sig and sig.sell and sig.score < -0.3:  # 明确卖出信号
                stop_loss_sells[code] = 0.0

        # =====================================================
        #  构建目标仓位
        # =====================================================
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
            )

        # 合并止损卖出
        for code in stop_loss_sells:
            desired_value[code] = 0.0

        # =====================================================
        #  动态调仓（平滑靠近目标）
        # =====================================================

        adjusted = {}

        if not rebalance and not stop_loss_sells:
            adjusted = deepcopy(current_positions)

        for code, target in desired_value.items():
            current = current_positions.get(code, 0.0)
            diff = target - current

            # 止损单立即执行，不做平滑
            if code in stop_loss_sells:
                adjusted[code] = 0.0
                continue

            if diff > 0:
                # 加仓
                move = self.entry_speed * diff
            else:
                # 减仓（更快）
                move = self.exit_speed * diff

            adjusted[code] = current + move

        # 未被选中 → 立即退出（不再逐步）
        for code, current in current_positions.items():
            if rebalance and code not in desired_value:
                # 直接清仓，不再逐步退出
                adjusted[code] = 0.0

        return adjusted

