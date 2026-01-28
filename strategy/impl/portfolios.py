# impl/portfolios.py
from core.portfolio import PortfolioConstructor

class ConstrainedTopKPortfolio(PortfolioConstructor):
    def __init__(self, max_position):
        self.max_position = max_position

    def build(
        self,
        date,
        universe,
        current_positions,
        signal_store,
        cash,
        prices,
    ):
        candidates = []

        for code in universe:
            sig = signal_store.get(code, date)
            if sig and sig.buy:
                candidates.append((sig.score, code))

        candidates.sort(reverse=True)

        target_codes = [
            code for _, code in candidates[: self.max_position]
        ]

        total_equity = cash + sum(current_positions.values())
        if not target_codes:
            return {}

        weight = 1.0 / len(target_codes)
        target = {}

        for code in target_codes:
            target[code] = total_equity * weight

        return target

