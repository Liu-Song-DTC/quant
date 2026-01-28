# core/portfolio.py
from abc import ABC, abstractmethod

class PortfolioConstructor(ABC):
    """
    把 signal + 约束 → 目标仓位
    """

    @abstractmethod
    def build(
        self,
        date,
        universe,
        current_positions,
        signal_store,
        cash,
        prices,
    ) -> dict:
        """
        return: dict[code -> target_value]
        """
        pass

