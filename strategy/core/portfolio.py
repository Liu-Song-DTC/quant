# core/portfolio.py
from copy import deepcopy

class PortfolioConstructor:
    """
    把 signal + 约束 → 目标仓位
    """
    def __init__(
        self,
        max_position=10,

        # 风险预算
        risk_ratio=1.0,   # 抽象单位，总风险预算
        target_volatility=0.2,       # 组合目标波动（用于 scaling）

        # 路径控制
        entry_step=0.25,
        exit_step=0.5,

        # 单票约束
        max_weight=1.2,

        # 组合约束
        max_gross_exposure=1.0,

        # 回撤控制
        drawdown_rules=None,  # [(0.1, 0.8), (0.2, 0.6)]
    ):
        self.max_position = max_position
        self.risk_ratio = risk_ratio
        self.target_volatility = target_volatility

        self.entry_step = entry_step
        self.exit_step = exit_step

        self.max_weight = 1 / max_position * max_weight
        self.max_gross_exposure = max_gross_exposure

        self.drawdown_rules = drawdown_rules or []

    def build(
        self,
        date,
        universe,
        current_positions,    # code -> 当前市值
        signal_store,
        cash,
        prices,
        initial_cash,
    ):
        """
        return: Dict[code, target_value]
        """

        forced_exit = set()
        for code in universe:
            sig = signal_store.get(code, date)
            if sig and sig.sell:
                forced_exit.add(code)

        # =====================================================
        # 1. Signal → 选股
        # =====================================================
        candidates = []
        for code in universe:
            sig = signal_store.get(code, date)
            if sig and sig.buy:
                candidates.append((sig.score, code))

        candidates.sort(reverse=True)
        selected = [c for _, c in candidates[: self.max_position]]

        if not selected:
            return {}

        # =====================================================
        # 2. 风险定价：1 / vol
        # =====================================================
        inv_risk_vol = {}
        for code in selected:
            sig = signal_store.get(code, date)
            risk_vol = sig.risk_vol
            score = sig.score
            if score and score > 0 and risk_vol and risk_vol > 0:
                inv_risk_vol[code] = score / risk_vol

        total_inv_vol = sum(inv_risk_vol.values())
        if total_inv_vol == 0:
            return {}

        total_equity = cash + sum(current_positions.values())
        portfolio_risk_budget = total_equity * self.risk_ratio
        desired = {}
        for code, iv in inv_risk_vol.items():
            weight = iv / total_inv_vol
            # 风险 → 市值
            desired[code] = weight * portfolio_risk_budget

        # =====================================================
        # 3. 动态建仓 / 减仓
        # =====================================================
        adjusted = {}

        for code, target_value in desired.items():
            current = current_positions.get(code, 0.0)

            if target_value > current:
                step = self.entry_step * target_value
                adjusted[code] = current + min(step, target_value - current)
            else:
                step = self.exit_step * current
                adjusted[code] = max(target_value, current - step)

        for code, current in current_positions.items():
            if code not in desired:
                step = self.exit_step * current
                remain = max(0.0, current - step)
                if remain > 0:
                    adjusted[code] = remain

        # =====================================================
        # 4. 单票最大权重限制
        # =====================================================
        total = sum(adjusted.values())
        for code in adjusted:
            adjusted[code] = min(
                adjusted[code],
                total * self.max_weight
            )

        # =====================================================
        # 5. 组合级风险上限
        # =====================================================
        if total > 0:
            limit = self.max_gross_exposure * total
            if total > limit:
                scale = limit / total
                for code in adjusted:
                    adjusted[code] *= scale

        # =====================================================
        # 6. 组合回撤控制（Equity-based scaling）
        # =====================================================
        drawdown = 1.0 - total_equity / initial_cash
        for dd, scale in sorted(self.drawdown_rules):
            if drawdown >= dd:
                for code in adjusted:
                    adjusted[code] *= scale

        for code in forced_exit:
            if code in current_positions:
                adjusted[code] = 0.0

        return adjusted
