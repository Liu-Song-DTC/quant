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
import yaml
import os


def _load_industry_ic_weights():
    """加载行业IC权重，用于因子值惩罚（包含分市场IC）"""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base_dir, 'config', 'factor_config.yaml')
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            industry_factors = config.get('industry_factors', {})
            result = {}
            for industry, cfg in industry_factors.items():
                result[industry] = {
                    'neutral': max(cfg.get('ic', 0.05), 0.01),
                    'bull': max(cfg.get('bull_ic', cfg.get('combined_ic', cfg.get('ic', 0.05))), 0.01),
                    'bear': max(cfg.get('bear_ic', cfg.get('bear_combined_ic', cfg.get('ic', 0.05))), 0.01),
                }
            return result
    return {}


class PortfolioConstructor:
    """仓位管理器 - 基于因子排名选股"""

    MIN_POSITIONS = 3
    MAX_POSITIONS = 15

    def __init__(
        self,
        target_volatility=None,
        entry_speed=None,
        exit_speed=None,
        position_stop_loss=None,
        portfolio_stop_loss=None,
    ):
        config = load_config()
        portfolio_config = config.get_portfolio_config()

        self.position_stop_loss = position_stop_loss if position_stop_loss is not None else portfolio_config.get('position_stop_loss', 0.12)
        self.entry_speed = entry_speed if entry_speed is not None else portfolio_config.get('entry_speed', 1.0)
        self.exit_speed = exit_speed if exit_speed is not None else portfolio_config.get('exit_speed', 1.0)

        self.peak_equity = None
        self.industry_ic = _load_industry_ic_weights()
        self.position_cost = {}
        self.last_selection = []
        self.current_ranking = {}
        self.current_n_positions = 0
        self.current_exposure = 1.0

    @staticmethod
    def _calc_max_position(total_equity: float, prices: dict) -> int:
        """根据资金自动计算最大持仓数

        每1万资金支持1个仓位, 范围[3, 15]
        100k→10只(与基线对齐)
        """
        n = int(total_equity / 10000)
        return max(PortfolioConstructor.MIN_POSITIONS, min(n, PortfolioConstructor.MAX_POSITIONS))  # fv_mean滞后仓位状态

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
        trend_score=0.0,
    ):
        """构建目标持仓 - 等权top N选股"""
        import pandas as pd

        total_equity = cash + sum(current_positions.values())
        n_positions = self._calc_max_position(total_equity, prices)

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

            # 价格约束：100股整手下，100*price必须接近理想单只仓位
            # 否则该股在bt_execution中因floor(raw)=0而无法买入，成为死权重
            # 允许2倍理想仓位（100股整手会导致超配，但至少能买入）
            price = prices.get(code, 0)
            ideal_per_stock = total_equity / n_positions
            if price > 0 and price * 100 > ideal_per_stock * 2:
                continue

            candidates.append({
                'code': code,
                'factor_value': factor_value,
                'score': getattr(sig, 'score', 0),
                'industry': getattr(sig, 'industry', '') or 'default',
                'risk_vol': getattr(sig, 'risk_vol', 0.03),
                'price': price,
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
        # 多信号融合择时: fv_mean(截面) + momentum(指数动量) + trend(均线排列)
        fv_mean = float(np.mean(factor_values))
        # fv_mean连续映射到[0.3, 1.0]
        fv_exposure = 0.3 + (fv_mean + 0.03) / (0.05 + 0.03) * 0.7
        fv_exposure = float(np.clip(fv_exposure, 0.3, 1.0))

        # momentum_score融合: 强负动量时额外降仓
        # momentum ∈ [-1, 1], 当momentum<-0.3时开始降仓
        mom_adj = 1.0
        if momentum_score < -0.3:
            mom_adj = 0.5 + 0.5 * (momentum_score + 1.0) / 0.7  # [-1,-0.3] → [0.5, 1.0]
            mom_adj = max(mom_adj, 0.5)

        # trend_score融合: 空头排列时额外降仓
        trend_adj = 1.0
        if trend_score < 0:
            trend_adj = 0.7 + 0.3 * (1.0 + trend_score)  # [-1, 0] → [0.7, 1.0]

        target_exposure = fv_exposure * mom_adj * trend_adj
        floor = 0.15 if bear_risk else 0.25
        target_exposure = float(np.clip(target_exposure, floor, 1.0))
        # 滞后平滑: 避免仓位突变
        self.current_exposure = 0.5 * self.current_exposure + 0.5 * target_exposure

        # === 选股: rank_pct > 0.5 → top N → 行业均衡 ===
        qualified = [c for c in candidates if c['rank_pct'] > 0.5]
        if not qualified:
            qualified = [c for c in candidates if c['rank_pct'] > 0.3]

        # 换手控制：现有持仓的rank_pct仍>hold_threshold时优先保留
        current_codes = set(current_positions.keys())
        hold_threshold = 0.5  # 已持仓股票，rank_pct>0.5即可保留

        # 按factor_value降序，但已持仓优先
        for c in qualified:
            c['is_held'] = c['code'] in current_codes
            if c['is_held'] and c['rank_pct'] > hold_threshold:
                pass  # 已持仓且仍合格，保留
            elif c['is_held'] and c['rank_pct'] <= hold_threshold:
                c['is_held'] = False  # 信号太弱，标记为替换

        # 排序：已持仓优先（稳定持仓），再按factor_value排序
        qualified.sort(key=lambda x: (-x['is_held'], -x['factor_value']))

        # 行业均衡选股
        industry_count = {}
        industry_cap = 5

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

        # === IC加权行业权重分配（分市场IC）===
        n = len(selected)
        if self.industry_ic:
            regime_key = {1: 'bull', -1: 'bear', 0: 'neutral'}.get(market_regime, 'neutral')
            max_ic = 0.05
            for c in selected:
                ic_dict = self.industry_ic.get(c['industry'], {})
                ic_val = ic_dict.get(regime_key, ic_dict.get('neutral', 0.05))
                max_ic = max(max_ic, ic_val)

            raw_weights = []
            for c in selected:
                ic_dict = self.industry_ic.get(c['industry'], {})
                ic = ic_dict.get(regime_key, ic_dict.get('neutral', 0.05))
                ic_w = (ic / max_ic) ** 1.5
                raw_weights.append(ic_w)
            total_w = sum(raw_weights)
            weights = [w / total_w * target_exposure for w in raw_weights]
        else:
            weights = [target_exposure / n] * n

        desired_value = {}
        for c, w in zip(selected, weights):
            desired_value[c['code']] = w * total_equity

        # 记录选股结果
        self.last_selection = [
            {
                'date': date,
                'code': c['code'],
                'score': c['score'],
                'weight': w,
                'industry': c.get('industry', ''),
                'rank_pct': c.get('rank_pct', 0),
            }
            for c, w in zip(selected, weights)
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
        trend_score=0.0,
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
                trend_score=trend_score,
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
