# core/portfolio.py
import numpy as np
from copy import deepcopy
from .config_loader import load_config


class PortfolioConstructor:
    """仓位管理器 - 基于信号质量和风险信息分配仓位"""

    def __init__(
        self,
        max_position=None,
        target_volatility=None,
        entry_speed=None,
        exit_speed=None,
        position_stop_loss=None,
        portfolio_stop_loss=None,
    ):
        # 从配置文件加载默认值
        config = load_config()
        portfolio_config = config.get_portfolio_config()

        self.max_position = max_position if max_position is not None else portfolio_config.get('max_position', 10)
        self.target_volatility = target_volatility if target_volatility is not None else portfolio_config.get('target_volatility', 0.20)
        self.entry_speed = entry_speed if entry_speed is not None else portfolio_config.get('entry_speed', 1.0)
        self.exit_speed = exit_speed if exit_speed is not None else portfolio_config.get('exit_speed', 1.0)
        self.position_stop_loss = position_stop_loss if position_stop_loss is not None else portfolio_config.get('position_stop_loss', 0.10)
        self.portfolio_stop_loss = portfolio_stop_loss if portfolio_stop_loss is not None else portfolio_config.get('portfolio_stop_loss', 0.08)

        # 波动率控制和组合止损默认关闭
        self.volatility_control_enabled = portfolio_config.get('volatility_control_enabled', False)
        self.portfolio_stop_loss_enabled = portfolio_config.get('portfolio_stop_loss_enabled', False)
        self.emergency_exposure = portfolio_config.get('emergency_exposure', 0.30)

        self.peak_equity = None
        self.position_cost = {}
        self.consecutive_losses = 0
        # 波动率控制相关
        self.equity_history = []  # 历史净值
        self.volatility_lookback = config.get('volatility_control.lookback_period', 20)
        self.current_volatility = 0.0
        # 组合止损相关
        self.portfolio_stop_loss_triggered = False

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

        # 组合止损触发后强制降仓
        if self.portfolio_stop_loss_triggered:
            max_gross_exposure = min(max_gross_exposure, self.emergency_exposure)

        return max_gross_exposure

    def _calculate_volatility(self):
        """计算组合波动率"""
        if len(self.equity_history) < 2:
            return 0.0

        # 只看最近 N 天的收益
        lookback = min(self.volatility_lookback, len(self.equity_history) - 1)
        recent_equity = self.equity_history[-lookback-1:]

        if len(recent_equity) < 2:
            return 0.0

        # 计算日收益率
        returns = []
        for i in range(1, len(recent_equity)):
            if recent_equity[i-1] > 0:
                ret = (recent_equity[i] - recent_equity[i-1]) / recent_equity[i-1]
                returns.append(ret)

        if not returns:
            return 0.0

        # 年化波动率 (假设252交易日)
        vol = np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0.0
        return vol

    def _apply_volatility_control(self, max_gross_exposure):
        """根据目标波动率调整仓位上限"""
        if not self.volatility_control_enabled:
            return max_gross_exposure

        # 计算当前波动率
        self.current_volatility = self._calculate_volatility()

        if self.current_volatility <= 0:
            return max_gross_exposure

        # 如果当前波动率高于目标，降低仓位
        if self.current_volatility > self.target_volatility:
            # 波动率越高，仓位越低 - 使用更温和的调整系数
            vol_ratio = self.target_volatility / self.current_volatility
            # 限制调整幅度，使用更温和的系数 (0.5 instead of 0.3)
            vol_ratio = max(0.5, min(1.0, vol_ratio))
            adjusted_exposure = max_gross_exposure * (0.5 + 0.5 * vol_ratio)  # 更温和的调整
            return adjusted_exposure

        # 如果当前波动率低于目标，可适当提高仓位（但不高于基础上限）
        elif self.current_volatility < self.target_volatility * 0.7:
            vol_ratio = self.target_volatility / max(self.current_volatility, 0.01)
            vol_ratio = min(1.1, vol_ratio)
            adjusted_exposure = min(max_gross_exposure * vol_ratio, max_gross_exposure)
            return adjusted_exposure

        return max_gross_exposure

    def _check_portfolio_stop_loss(self, total_equity):
        """检查并触发组合止损"""
        if not self.portfolio_stop_loss_enabled:
            return False

        # 记录当前净值
        self.equity_history.append(total_equity)

        # 计算峰值和回撤
        if self.peak_equity is None:
            self.peak_equity = total_equity
        self.peak_equity = max(self.peak_equity, total_equity)

        drawdown = 1 - total_equity / self.peak_equity if self.peak_equity > 0 else 0.0

        # 检查是否触发止损
        if drawdown >= self.portfolio_stop_loss and not self.portfolio_stop_loss_triggered:
            self.portfolio_stop_loss_triggered = True
            print(f"[组合止损] 触发! 回撤: {drawdown:.2%}, 峰值: {self.peak_equity:.2f}, 当前: {total_equity:.2f}")
            return True

        # 如果已经触发，需要等待净值恢复后才能解除
        # 当净值回到峰值以上时解除
        if self.portfolio_stop_loss_triggered and total_equity >= self.peak_equity * 0.98:
            self.portfolio_stop_loss_triggered = False
            print(f"[组合止损] 解除! 净值已恢复")

        return self.portfolio_stop_loss_triggered

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

        # 检查并更新组合止损状态
        self._check_portfolio_stop_loss(total_equity)

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

        # 按仓位排序，选出前N个
        candidates.sort(key=lambda x: x['position'], reverse=True)
        selected = candidates[:self.max_position]

        if not selected:
            return {}

        # 计算总仓位
        total_position = sum(c['position'] for c in selected)

        # 计算总仓位上限
        max_gross_exposure = self._calculate_position_limit(drawdown, risk_extreme_exists)

        # 应用波动率控制
        max_gross_exposure = self._apply_volatility_control(max_gross_exposure)

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

        # 每天记录净值，用于波动率计算
        self.equity_history.append(total_equity)

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
