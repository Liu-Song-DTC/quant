"""组合状态 — 极简JSON, 可直接手动编辑

trade/portfolio_state.json:
{
  "cash": 45000.0,
  "positions": {
    "600519": {"shares": 100, "cost_price": 1680.0},
    "000858": {"shares": 200, "cost_price": 165.0}
  }
}

手动编辑后运行 `python main.py status` 验证。
"""
import json
import os
from typing import Dict, Optional


class PortfolioState:
    def __init__(self, state_file: str):
        self.state_file = state_file
        self.data = self._default()

    def _default(self) -> dict:
        return {"cash": 0.0, "positions": {}}

    @property
    def _internal_file(self) -> str:
        """内部状态文件: exposure/peak_equity 持久化, 用户无需关心"""
        return self.state_file.replace(".json", ".internal.json")

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

    @property
    def positions(self) -> dict:
        return self.data.get("positions", {})

    def load_internal(self) -> dict:
        """加载内部状态, 不存在则返回默认值"""
        path = self._internal_file
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        return {"exposure": 1.0, "peak_equity": 0.0}

    def save_internal(self, exposure: float, peak_equity: float):
        """保存内部状态"""
        path = self._internal_file
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"exposure": exposure, "peak_equity": peak_equity}, f)

    def get_current_positions(self, prices: Dict[str, float]) -> Dict[str, float]:
        """返回 {code: market_value} 供策略使用"""
        result = {}
        for code, pos in self.positions.items():
            shares = pos["shares"]
            price = prices.get(code, pos.get("cost_price", 0))
            if shares > 0 and price > 0:
                result[code] = shares * price
        return result

    def check_stop_loss(self, prices: Dict[str, float], stop_loss: float = 0.12) -> list:
        """检查持仓止损, 返回触发止损的股票列表 [{code, pnl_pct, cost_price, current_price}]"""
        triggered = []
        for code, pos in self.positions.items():
            if pos["shares"] <= 0:
                continue
            price = prices.get(code, 0)
            if price <= 0:
                continue
            pnl = (price - pos["cost_price"]) / pos["cost_price"]
            if pnl < -stop_loss:
                triggered.append({
                    "code": code, "shares": pos["shares"],
                    "cost_price": pos["cost_price"], "current_price": price,
                    "pnl_pct": pnl,
                })
        return triggered

    def get_cost_basis(self) -> Dict[str, list]:
        """返回 {code: [shares, avg_cost]} 供策略使用"""
        result = {}
        for code, pos in self.positions.items():
            if pos["shares"] > 0:
                result[code] = [pos["shares"], pos["cost_price"]]
        return result

    def update_after_trade(self, trades: list):
        """确认交易后更新持仓"""
        positions = self.data.setdefault("positions", {})
        for trade in trades:
            action, code, shares, price = trade["action"], trade["code"], trade["shares"], trade["price"]

            if action == "buy":
                self.data["cash"] -= shares * price
                if code in positions:
                    pos = positions[code]
                    total = pos["shares"] + shares
                    pos["cost_price"] = (pos["cost_price"] * pos["shares"] + price * shares) / total
                    pos["shares"] = total
                else:
                    positions[code] = {"shares": shares, "cost_price": price}
            elif action == "sell":
                self.data["cash"] += shares * price
                if code in positions:
                    positions[code]["shares"] -= shares
                    if positions[code]["shares"] <= 0:
                        del positions[code]

        self.save()

    def summary(self, prices: Dict[str, float] = None) -> dict:
        total_mv = 0.0
        details = []
        for code, pos in self.positions.items():
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
            "positions": details,
        }
