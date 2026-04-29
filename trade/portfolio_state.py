import json
import os
from datetime import date
from typing import Dict, List, Optional


class PortfolioState:
    """组合状态管理 — JSON持久化持仓/现金/成本"""

    def __init__(self, state_file: str):
        self.state_file = state_file
        self.data = self._default()

    def _default(self) -> dict:
        return {
            "last_update": None,
            "cash": 0.0,
            "positions": {},
            "trade_history": [],
        }

    @classmethod
    def load(cls, state_file: str) -> "PortfolioState":
        ps = cls(state_file)
        if os.path.exists(state_file) and os.path.getsize(state_file) > 0:
            with open(state_file, "r", encoding="utf-8") as f:
                try:
                    ps.data = json.load(f)
                except json.JSONDecodeError:
                    ps.data = ps._default()
        else:
            ps.data = ps._default()
        return ps

    def save(self):
        os.makedirs(os.path.dirname(self.state_file) or ".", exist_ok=True)
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)

    @property
    def cash(self) -> float:
        return self.data.get("cash", 0.0)

    @cash.setter
    def cash(self, value: float):
        self.data["cash"] = value

    @property
    def last_update(self) -> Optional[str]:
        return self.data.get("last_update")

    def init_cash(self, amount: float):
        if self.data.get("last_update") is None:
            self.data["cash"] = amount
            self.data["last_update"] = str(date.today())
            self.save()

    def get_current_positions(self, prices: Dict[str, float]) -> Dict[str, float]:
        """返回 {code: market_value} 格式供策略使用"""
        result = {}
        for code, pos in self.data.get("positions", {}).items():
            shares = pos["shares"]
            price = prices.get(code, pos.get("cost_price", 0))
            if shares > 0 and price > 0:
                result[code] = shares * price
        return result

    def get_cost_basis(self) -> Dict[str, list]:
        """返回 {code: [shares, avg_cost]} 供策略使用"""
        result = {}
        for code, pos in self.data.get("positions", {}).items():
            if pos["shares"] > 0:
                result[code] = [pos["shares"], pos["cost_price"]]
        return result

    def update_after_trade(self, trades: List[dict]):
        """用户确认执行交易后更新持仓

        trades: [{"action": "buy"/"sell", "code": "600519", "shares": 100, "price": 1680.0}, ...]
        """
        positions = self.data.setdefault("positions", {})

        for trade in trades:
            action = trade["action"]
            code = trade["code"]
            shares = trade["shares"]
            price = trade["price"]

            if action == "buy":
                self.data["cash"] -= shares * price
                if code in positions:
                    pos = positions[code]
                    total_shares = pos["shares"] + shares
                    total_cost = pos["cost_price"] * pos["shares"] + price * shares
                    pos["shares"] = total_shares
                    pos["cost_price"] = total_cost / total_shares if total_shares > 0 else 0
                    pos["cost_total"] = pos["shares"] * pos["cost_price"]
                else:
                    positions[code] = {
                        "shares": shares,
                        "cost_price": price,
                        "cost_total": shares * price,
                    }
            elif action == "sell":
                self.data["cash"] += shares * price
                if code in positions:
                    pos = positions[code]
                    pos["shares"] -= shares
                    pos["cost_total"] = pos["shares"] * pos["cost_price"]
                    if pos["shares"] <= 0:
                        del positions[code]

            # 记录交易历史
            self.data.setdefault("trade_history", []).append({
                "date": str(date.today()),
                "action": action,
                "code": code,
                "shares": shares,
                "price": price,
            })

        self.data["last_update"] = str(date.today())
        self.save()

    def summary(self, prices: Dict[str, float] = None) -> dict:
        """组合摘要"""
        total_market_value = 0.0
        pos_details = []
        for code, pos in self.data.get("positions", {}).items():
            if pos["shares"] <= 0:
                continue
            price = (prices or {}).get(code, pos["cost_price"])
            market_value = pos["shares"] * price
            total_market_value += market_value
            pnl = (price - pos["cost_price"]) / pos["cost_price"] if pos["cost_price"] > 0 else 0
            pos_details.append({
                "code": code,
                "shares": pos["shares"],
                "cost_price": pos["cost_price"],
                "current_price": price,
                "market_value": market_value,
                "pnl_pct": pnl,
            })

        total_value = self.cash + total_market_value
        return {
            "cash": self.cash,
            "market_value": total_market_value,
            "total_value": total_value,
            "positions": pos_details,
            "last_update": self.last_update,
        }
