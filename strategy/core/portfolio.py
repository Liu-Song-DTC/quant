# core/portfolio.py
"""
组合构建器 - 基于因子排名选股

设计原则:
1. 等权top N: 截面rank_pct>0.5的top N只, 等权分配
2. 行业均衡: 每个行业最多5只, 避免过度集中
3. 最小风控: 仅深度回撤和fv_mean降仓
4. 止损仅限再平衡日: 避免非再平衡日过早卖出
"""
import numpy as np
from copy import deepcopy
from .config_loader import load_config


class PortfolioConstructor:
    """仓位管理器 - 基于因子排名选股"""

    def __init__(
        self,
        max_position=None,
        target_volatility=None,
        entry_speed=None,
        exit_speed=None,
        position_stop_loss=None,
        portfolio_stop_loss=None,
    ):
        config = load_config()
        portfolio_config = config.get_portfolio_config()

        self.max_position = max_position if max_position is not None else portfolio_config.get('max_position', 10)
        self.position_stop_loss = position_stop_loss if position_stop_loss is not None else portfolio_config.get('position_stop_loss', 0.12)
        self.entry_speed = entry_speed if entry_speed is not None else portfolio_config.get('entry_speed', 1.0)
        self.exit_speed = exit_speed if exit_speed is not None else portfolio_config.get('exit_speed', 1.0)

        self.peak_equity = None
        self.position_cost = {}
        self.last_selection = []
        self.current_ranking = {}
        self.current_n_positions = 0

    def _build_desired_value(
        self,
        date,
        universe,
        current_positions,
        signal_store,
        cash,
        prices,
        market_regime=0,
        momentum_score=0.0,
        bear_risk=False,
    ):
        """构建目标持仓 - 等权top N选股"""
        import pandas as pd

        total_equity = cash + sum(current_positions.values())

        # 计算峰值和回撤
        if self.peak_equity is None:
            self.peak_equity = total_equity
        self.peak_equity = max(self.peak_equity, total_equity)
        drawdown = 1 - total_equity / self.peak_equity if self.peak_equity > 0 else 0.0

        # === 收集候选股票 ===
        candidates = []
        for code in universe:
            sig = signal_store.get(code, date)
            if sig is None:
                continue

            factor_value = getattr(sig, 'factor_value', None)
            if factor_value is None or (isinstance(factor_value, float) and np.isnan(factor_value)):
                continue

            candidates.append({
                'code': code,
                'factor_value': factor_value,
                'score': getattr(sig, 'score', 0),
                'industry': getattr(sig, 'industry', '') or 'default',
                'sig': sig,
            })

        if not candidates:
            return {}

        # === 截面排名 ===
        factor_values = np.array([c['factor_value'] for c in candidates])
        rank_pct = pd.Series(factor_values).rank(pct=True)

        for i, c in enumerate(candidates):
            c['rank_pct'] = rank_pct.iloc[i]

        # === 市场仓位调整 ===
        # 全仓: 截面排名已经选出了最好的股票
        # 仅极端回撤时降仓（尾部风险保护）
        target_exposure = 1.0
        if drawdown > 0.20:
            target_exposure = 0.5

        # === 选股: rank_pct > 0.5 → top N → 行业均衡 ===
        qualified = [c for c in candidates if c['rank_pct'] > 0.5]
        if not qualified:
            qualified = [c for c in candidates if c['rank_pct'] > 0.3]

        # 按factor_value降序
        qualified.sort(key=lambda x: -x['factor_value'])

        # 行业均衡选股
        industry_count = {}
        industry_cap = 5
        n_positions = self.max_position

        selected = []
        for c in qualified:
            ind = c['industry']
            if industry_count.get(ind, 0) >= industry_cap:
                continue
            selected.append(c)
            industry_count[ind] = industry_count.get(ind, 0) + 1
            if len(selected) >= n_positions:
                break

        if not selected:
            return {}

        # 记录
        self.current_ranking = {c['code']: i for i, c in enumerate(candidates)}
        self.current_n_positions = len(selected)

        # === 等权分配 ===
        n = len(selected)
        weight_per_stock = target_exposure / n

        desired_value = {}
        for c in selected:
            desired_value[c['code']] = weight_per_stock * total_equity

        # 记录选股结果
        self.last_selection = [
            {
                'date': date,
                'code': c['code'],
                'score': c['score'],
                'weight': weight_per_stock,
                'industry': c.get('industry', ''),
                'rank_pct': c.get('rank_pct', 0),
            }
            for c in selected
        ]

        return desired_value

    def build(
        self,
        date,
        universe,
        current_positions,
        signal_store,
        cash,
        prices,
        market_regime,
        momentum_score=0.0,
        bear_risk=False,
        cost=None,
        rebalance=False,
    ):
        """构建目标持仓（外部接口）

        关键设计: 非再平衡日只做个股成本止损, 不做信号止损
        """
        stop_loss_sells = {}
        total_equity = cash + sum(current_positions.values())

        # 个股止损检查 - 仅成本止损
        for code, current_value in current_positions.items():
            if code not in prices or current_value <= 0:
                continue

            # 成本止损: 亏损超过position_stop_loss
            if cost and code in cost and len(cost[code]) >= 2 and cost[code][0] > 0:
                avg_cost = cost[code][1]
                current_price = prices[code]
                pnl_pct = (current_price - avg_cost) / avg_cost
                if pnl_pct < -self.position_stop_loss:
                    stop_loss_sells[code] = 0.0

            # 不使用信号卖出: 组合层用截面rank_pct排序选股
            # factor_value的绝对值不决定卖出，排名才决定

        desired_value = {}
        if rebalance:
            desired_value = self._build_desired_value(
                date=date,
                universe=universe,
                current_positions=current_positions,
                signal_store=signal_store,
                cash=cash,
                prices=prices,
                market_regime=market_regime,
                momentum_score=momentum_score,
                bear_risk=bear_risk,
            )

        # 强制卖出
        for code in stop_loss_sells:
            desired_value[code] = 0.0

        adjusted = {}

        # 非调仓日保持现有持仓(除非成本止损)
        if not rebalance and not stop_loss_sells:
            adjusted = deepcopy(current_positions)

        # 执行目标持仓
        for code, target in desired_value.items():
            current = current_positions.get(code, 0.0)
            diff = target - current

            if code in stop_loss_sells:
                adjusted[code] = 0.0
                continue

            # 忽略微小调整
            if abs(diff) < total_equity * 0.01:
                if code in current_positions:
                    adjusted[code] = current
                continue

            # 渐进进出
            if diff > 0:
                move = self.entry_speed * diff
            else:
                move = self.exit_speed * diff

            adjusted[code] = current + move

        # 再平衡日：清仓不在目标中的旧持仓
        if rebalance:
            for code, current in current_positions.items():
                if code in adjusted:
                    continue
                if code in desired_value:
                    continue
                adjusted[code] = 0.0

        return adjusted
