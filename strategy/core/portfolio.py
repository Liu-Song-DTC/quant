# core/portfolio.py
import numpy as np
from copy import deepcopy


class PortfolioConstructor:
    """仓位管理器 - 基于信号质量和风险信息分配仓位"""

    def __init__(
        self,
        max_position=5,
        target_volatility=0.20,
        entry_speed=1.0,
        exit_speed=1.0,
        position_stop_loss=0.10,
        portfolio_stop_loss=0.08,
    ):
        self.max_position = max_position
        self.target_volatility = target_volatility
        self.entry_speed = entry_speed
        self.exit_speed = exit_speed
        self.position_stop_loss = position_stop_loss
        self.portfolio_stop_loss = portfolio_stop_loss
        self.peak_equity = None
        self.position_cost = {}
        self.consecutive_losses = 0

    def _calculate_position_limit(self, drawdown, risk_extreme_exists):
        """根据回撤和极端状态计算总仓位上限"""
        # 基础仓位上限
        if drawdown > 0.15:
            max_gross_exposure = 0.30
        elif drawdown > 0.10:
            max_gross_exposure = 0.50
        elif drawdown > 0.05:
            max_gross_exposure = 0.80
        else:
            max_gross_exposure = 1.0

        # 极端波动状态额外降仓
        if risk_extreme_exists:
            max_gross_exposure = max_gross_exposure * 0.7

        return max_gross_exposure

    def _calculate_stock_position(self, sig):
        """
        计算单个股票的仓位

        公式:
        1. 基础仓位 = score * (1 / risk_vol)
        2. 极端降仓 = 基础仓位 * 0.7 if risk_extreme else 1.0
        """
        if not sig or not sig.buy or sig.score <= 0:
            return 0.0

        # 限制 score 范围
        score = max(0.0, min(1.0, sig.score))

        # 限制 risk_vol 范围，避免除零和过大/过小
        risk_vol = max(0.01, min(1.0, sig.risk_vol))

        # 基础仓位 = score * (1 / risk_vol)
        # 高分低波给高仓位
        base_position = score * (1.0 / risk_vol)

        # 极端状态降仓
        if sig.risk_extreme:
            base_position *= 0.7

        return base_position

    def _build_desired_value(
        self,
        date,
        universe,
        current_positions,
        signal_store,
        cash,
        prices,
    ):
        """构建目标持仓"""
        total_equity = cash + sum(current_positions.values())

        # 计算峰值和回撤
        if self.peak_equity is None:
            self.peak_equity = total_equity
        self.peak_equity = max(self.peak_equity, total_equity)

        drawdown = 1 - total_equity / self.peak_equity if self.peak_equity > 0 else 0.0

        # 检查是否存在极端状态
        risk_extreme_exists = False
        candidates = []

        for code in universe:
            sig = signal_store.get(code, date)
            if sig and sig.buy and sig.score > 0.10:
                # 检查极端状态
                if sig.risk_extreme:
                    risk_extreme_exists = True

                # 计算基础仓位
                position = self._calculate_stock_position(sig)

                if position > 0:
                    candidates.append({
                        'code': code,
                        'position': position,
                        'score': sig.score,
                        'risk_vol': sig.risk_vol,
                        'sig': sig,
                    })

        if not candidates:
            return {}

        # 按基础仓位排序，选出前N个
        candidates.sort(key=lambda x: x['position'], reverse=True)
        selected = candidates[:self.max_position]

        if not selected:
            return {}

        # 计算总仓位
        total_position = sum(c['position'] for c in selected)

        # 计算总仓位上限
        max_gross_exposure = self._calculate_position_limit(drawdown, risk_extreme_exists)

        # 归一化并应用仓位上限
        raw_weights = {}
        for c in selected:
            # 归一化仓位
            normalized_position = (c['position'] / total_position) * max_gross_exposure if total_position > 0 else 0
            raw_weights[c['code']] = normalized_position

        # 构建目标市值
        desired_value = {
            c: raw_weights[c] * total_equity
            for c in raw_weights
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
        market_regime,  # 保留参数但不使用
        cost,
        rebalance,
    ):
        """构建目标持仓（外部接口）"""
        stop_loss_sells = {}
        total_equity = cash + sum(current_positions.values())

        # 个股止损检查
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
            )

        # 强制卖出
        for code in stop_loss_sells:
            desired_value[code] = 0.0

        adjusted = {}

        # 非调仓日保持现有持仓
        if not rebalance and not stop_loss_sells:
            adjusted = deepcopy(current_positions)

        # 调仓逻辑
        for code, target in desired_value.items():
            current = current_positions.get(code, 0.0)
            diff = target - current

            if code in stop_loss_sells:
                adjusted[code] = 0.0
                continue

            # 渐进进出
            if diff > 0:
                move = self.entry_speed * diff
            else:
                move = self.exit_speed * diff

            adjusted[code] = current + move

        # 调仓日清仓不在候选列表中的持仓
        for code, current in current_positions.items():
            if rebalance and code not in desired_value:
                adjusted[code] = 0.0

        return adjusted
