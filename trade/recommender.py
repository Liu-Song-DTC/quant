"""交易建议生成 — 目标持仓 vs 当前持仓 → 买卖清单"""
import math
from typing import Dict, List, Optional


class Recommender:
    """对比目标持仓和当前持仓，生成具体交易建议"""

    MIN_TRADE_VALUE = 500  # 最低交易金额（元），低于此忽略

    def generate(
        self,
        target_positions: Dict[str, float],
        current_positions: Dict[str, float],
        prices: Dict[str, float],
        cash: float,
        cost: dict = None,
        market_regime: dict = None,
        selections: list = None,
    ) -> dict:
        """生成交易建议

        Returns:
            {
                "date": str,
                "market_regime": dict,
                "buys": [{"code", "shares", "price", "estimated_cost"}],
                "sells": [{"code", "shares", "price", "estimated_revenue", "reason"}],
                "holds": [{"code", "shares", "current_value", "target_value", "weight"}],
                "trades": [{"action", "code", "shares", "price"}],  # 供confirm使用
                "total_buy": float,
                "total_sell": float,
                "cash_sufficient": bool,
            }
        """
        cost = cost or {}
        market_regime = market_regime or {}
        selections = selections or []
        selection_map = {s["code"]: s for s in selections}

        # 计算当前每只股票的持仓数量
        current_shares = {}
        for code, value in current_positions.items():
            price = prices.get(code, 0)
            if price > 0:
                current_shares[code] = int(value / price / 100) * 100  # 取整手

        # 计算目标数量
        target_shares = {}
        for code, value in target_positions.items():
            price = prices.get(code, 0)
            if price > 0:
                target_shares[code] = int(value / price / 100) * 100

        buys = []
        sells = []
        holds = []
        trades = []

        all_codes = set(target_positions.keys()) | set(current_positions.keys())

        for code in all_codes:
            price = prices.get(code, 0)
            if price <= 0:
                continue

            cur = current_shares.get(code, 0)
            tgt = target_shares.get(code, 0)
            diff = tgt - cur

            sel = selection_map.get(code, {})

            if diff > 0:
                # 买入
                shares = (diff // 100) * 100  # 整手
                if shares <= 0:
                    continue
                est_cost = shares * price
                if est_cost < self.MIN_TRADE_VALUE:
                    continue
                buys.append({
                    "code": code,
                    "shares": shares,
                    "price": price,
                    "estimated_cost": est_cost,
                    "score": sel.get("score", 0),
                    "weight": sel.get("weight", 0),
                    "industry": sel.get("industry", ""),
                })
                trades.append({"action": "buy", "code": code, "shares": shares, "price": price})

            elif diff < 0:
                # 卖出
                shares = ((-diff) // 100) * 100
                if shares <= 0:
                    continue
                shares = min(shares, cur)  # 不超过持仓
                est_revenue = shares * price
                if est_revenue < self.MIN_TRADE_VALUE and cur > shares:
                    continue

                # 卖出原因
                reason = ""
                if code not in target_positions:
                    reason = "不在目标池"
                elif tgt < cur * 0.5:
                    reason = "减仓"
                else:
                    reason = "小幅调整"
                # 检查止损
                if code in cost and len(cost[code]) >= 2 and cost[code][0] > 0:
                    avg_cost = cost[code][1]
                    pnl = (price - avg_cost) / avg_cost
                    if pnl < -0.12:
                        reason = f"止损(亏{pnl:.1%})"

                sells.append({
                    "code": code,
                    "shares": shares,
                    "price": price,
                    "estimated_revenue": est_revenue,
                    "reason": reason,
                })
                trades.append({"action": "sell", "code": code, "shares": shares, "price": price})

            else:
                # 持有
                if cur > 0:
                    cur_value = cur * price
                    tgt_value = target_positions.get(code, 0)
                    total_equity = cash + sum(current_positions.values())
                    weight = cur_value / total_equity if total_equity > 0 else 0
                    holds.append({
                        "code": code,
                        "shares": cur,
                        "current_value": cur_value,
                        "target_value": tgt_value,
                        "weight": weight,
                        "industry": sel.get("industry", ""),
                    })

        total_buy = sum(b["estimated_cost"] for b in buys)
        total_sell = sum(s["estimated_revenue"] for s in sells)
        cash_after = cash - total_buy + total_sell

        return {
            "market_regime": market_regime,
            "buys": buys,
            "sells": sells,
            "holds": holds,
            "trades": trades,
            "total_buy": total_buy,
            "total_sell": total_sell,
            "cash_before": cash,
            "cash_after": cash_after,
            "cash_sufficient": cash_after >= 0,
        }
