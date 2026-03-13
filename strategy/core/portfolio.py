# core/portfolio.py
import numpy as np
from copy import deepcopy

class PortfolioConstructor:
    """最优版本的仓位管理"""

    def __init__(
        self,
        max_position=8,
        target_volatility=0.15,
        entry_speed=0.8,
        exit_speed=1.0,
        position_stop_loss=0.12,
        portfolio_stop_loss=0.10,
    ):
        self.max_position = max_position
        self.target_volatility = target_volatility
        self.entry_speed = entry_speed
        self.exit_speed = exit_speed
        self.position_stop_loss = position_stop_loss
        self.portfolio_stop_loss = portfolio_stop_loss
        self.peak_equity = None
        self.position_cost = {}

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
        if market_regime == 1:
            max_gross_exposure = 1.0
        elif market_regime == 0:
            max_gross_exposure = 0.6
        else:
            max_gross_exposure = 0.2

        total_equity = cash + sum(current_positions.values())
        if self.peak_equity is None:
            self.peak_equity = total_equity
        self.peak_equity = max(self.peak_equity, total_equity)

        drawdown = 1 - total_equity / self.peak_equity

        if drawdown > 0.10:
            max_gross_exposure *= 0.15
        elif drawdown > 0.06:
            max_gross_exposure *= 0.3
        elif drawdown > 0.03:
            max_gross_exposure *= 0.5

        candidates = []
        for code in universe:
            sig = signal_store.get(code, date)
            min_score = 0.30 if market_regime >= 0 else 0.40
            if sig and sig.buy and sig.score > min_score:
                candidates.append((sig.score, code, sig.risk_vol))

        candidates.sort(reverse=True)
        effective_max = self.max_position if market_regime >= 0 else max(2, self.max_position // 2)
        selected = [c for _, c, _ in candidates[: effective_max]]

        if not selected:
            return {}

        raw_weight = {}
        for code in selected:
            sig = signal_store.get(code, date)
            if sig.risk_vol > 0:
                raw_weight[code] = sig.score / sig.risk_vol

        if not raw_weight:
            return {}

        total_raw = sum(raw_weight.values())
        weights = {c: w / total_raw for c, w in raw_weight.items()}

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

        gross = sum(abs(w) for w in weights.values())
        if gross > max_gross_exposure:
            scale = max_gross_exposure / gross
            for c in weights:
                weights[c] *= scale

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
        stop_loss_sells = {}
        total_equity = cash + sum(current_positions.values())

        for code, current_value in current_positions.items():
            if code not in prices or current_value <= 0:
                continue

            if code in cost and len(cost) >= 2 and cost[code][0] > 0:
                avg_cost = cost[code][1]
                current_price = prices[code]
                pnl_pct = (current_price - avg_cost) / avg_cost

                if pnl_pct < -self.position_stop_loss:
                    stop_loss_sells[code] = 0.0

            sig = signal_store.get(code, date)
            if sig and sig.sell and sig.score < -0.20:
                stop_loss_sells[code] = 0.0

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

        for code in stop_loss_sells:
            desired_value[code] = 0.0

        adjusted = {}

        if not rebalance and not stop_loss_sells:
            adjusted = deepcopy(current_positions)

        for code, target in desired_value.items():
            current = current_positions.get(code, 0.0)
            diff = target - current

            if code in stop_loss_sells:
                adjusted[code] = 0.0
                continue

            if diff > 0:
                move = self.entry_speed * diff
            else:
                move = self.exit_speed * diff

            adjusted[code] = current + move

        for code, current in current_positions.items():
            if rebalance and code not in desired_value:
                adjusted[code] = 0.0

        return adjusted
