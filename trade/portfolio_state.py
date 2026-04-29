"""组合状态 — JSON持久化，可直接手动编辑

文件格式 (portfolio_state.json):
{
  "last_update": "2026-04-29",
  "cash": 45000.0,
  "positions": {
    "600519": {"shares": 100, "cost_price": 1680.0},
    "000858": {"shares": 200, "cost_price": 165.0}
  },
  "trade_history": [
    {"date": "2026-04-29", "action": "buy", "code": "600519", "shares": 100, "price": 1680.0}
  ]
}

编辑后运行 `python main.py status` 验证。
"""
import json
import os
from datetime import date
from typing import Dict, List, Optional


class PortfolioState:
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
        """返回 {code: market_value} 供策略使用"""
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
        """确认交易后更新持仓

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
                else:
                    positions[code] = {"shares": shares, "cost_price": price}
            elif action == "sell":
                self.data["cash"] += shares * price
                if code in positions:
                    pos = positions[code]
                    pos["shares"] -= shares
                    if pos["shares"] <= 0:
                        del positions[code]

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
        total_mv = 0.0
        details = []
        for code, pos in self.data.get("positions", {}).items():
            if pos["shares"] <= 0:
                continue
            price = (prices or {}).get(code, pos["cost_price"])
            mv = pos["shares"] * price
            total_mv += mv
            pnl = (price - pos["cost_price"]) / pos["cost_price"] if pos["cost_price"] > 0 else 0
            details.append({
                "code": code, "shares": pos["shares"],
                "cost_price": pos["cost_price"], "current_price": price,
                "market_value": mv, "pnl_pct": pnl,
            })
        return {
            "cash": self.cash, "market_value": total_mv,
            "total_value": self.cash + total_mv,
            "positions": details, "last_update": self.last_update,
        }
