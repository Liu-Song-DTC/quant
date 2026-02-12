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

        # 风险控制
        target_volatility=0.15,   # 组合目标波动

        # 动态调仓
        entry_speed=0.5,
        exit_speed=0.7,
    ):
        self.max_position = max_position
        self.target_volatility = target_volatility

        self.entry_speed = entry_speed
        self.exit_speed = exit_speed

        self.defensive_mode = False
        self.peak_equity = None

    # =====================================================
    # 主构建函数
    # =====================================================

    def build(
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
            max_gross_exposure= 1.0
        elif market_regime == 0:
            max_gross_exposure = 0.6
        else:
            max_gross_exposure = 0.3

        total_equity = cash + sum(current_positions.values())
        # 初始化 peak equity
        if self.peak_equity is None:
            self.peak_equity = total_equity
        self.peak_equity = max(self.peak_equity, total_equity)

        drawdown = 1 - total_equity / self.peak_equity

        if drawdown > 0.2:
            self.defensive_mode = True

        if total_equity >= self.peak_equity:
            self.defensive_mode = False

        if self.defensive_mode:
            max_gross_exposure *= 0.5

        # =====================================================
        #  选股（只选 buy 且 score>0）
        # =====================================================

        candidates = []
        for code in universe:
            sig = signal_store.get(code, date)
            if sig and sig.buy and sig.score > 0:
                candidates.append((sig.score, code))

        candidates.sort(reverse=True)
        selected = [c for _, c in candidates[: self.max_position]]

        if not selected:
            return self._gradual_exit(current_positions)

        # =====================================================
        #  风险权重（score / vol）
        # =====================================================

        raw_weight = {}
        for code in selected:
            sig = signal_store.get(code, date)
            if sig.risk_vol > 0:
                raw_weight[code] = sig.score

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

        # =====================================================
        #  动态调仓（平滑靠近目标）
        # =====================================================

        adjusted = {}

        for code, target in desired_value.items():
            current = current_positions.get(code, 0.0)
            diff = target - current

            if diff > 0:
                # 加仓
                move = self.entry_speed * diff
            else:
                # 减仓
                move = self.exit_speed * diff

            adjusted[code] = current + move

        # 未被选中 → 逐步退出
        for code, current in current_positions.items():
            if code not in desired_value:
                move = -self.exit_speed * current
                remain = current + move
                if remain > 0:
                    adjusted[code] = remain

        return adjusted

    # =====================================================
    # 辅助函数：无信号时逐步退出
    # =====================================================

    def _gradual_exit(self, current_positions):
        adjusted = {}
        for code, value in current_positions.items():
            remain = value * 0.5
            if remain > 0:
                adjusted[code] = remain
        return adjusted

