# core/portfolio.py
import numpy as np
from copy import deepcopy
from .config_loader import load_config

# 弱势行业减配系数（基于滚动验证IR结果）
# IR < 0.1: 减配50%
# IR < 0.15: 减配25%
INDUSTRY_DISCOUNT = {
    '交运': 0.5,           # IR=0.05
    '通信/计算机': 0.5,    # IR=-0.04
    '半导体/光伏': 0.75,   # IR=0.12
}

# 换仓门槛（降低换手率）
REBALANCE_THRESHOLD = 0.15  # 持仓变化超过15%才调仓


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
        self.volatility_control_enabled = config.get('volatility_control.enabled', False)
        self.portfolio_stop_loss_enabled = config.get('portfolio_stop_loss.enabled', False)
        self.emergency_exposure = config.get('portfolio_stop_loss.emergency_exposure', 0.30)
        # 行业加权选股
        self.enable_industry_weighting = portfolio_config.get('enable_industry_weighting', True)

        self.peak_equity = None
        self.position_cost = {}
        self.consecutive_losses = 0
        # 波动率控制相关
        self.equity_history = []  # 历史净值
        self.volatility_lookback = config.get('volatility_control.lookback_period', 20)
        self.current_volatility = 0.0
        # 组合止损相关
        self.portfolio_stop_loss_triggered = False

    def _calculate_position_limit(self, drawdown, risk_extreme_exists, market_regime=0):
        """根据回撤和极端状态计算总仓位上限

        A股优化：收紧阈值，控制回撤
        - 5%/10%/15% → 0.50/0.70/0.90

        熊市保护：market_regime=-1时降至30%
        """
        # 基础仓位上限（收紧阈值，降低回撤风险）
        if drawdown > 0.15:
            max_gross_exposure = 0.50
        elif drawdown > 0.10:
            max_gross_exposure = 0.70
        elif drawdown > 0.05:
            max_gross_exposure = 0.90
        else:
            max_gross_exposure = 1.0

        # 熊市仓位保护 - 降至60%（30%太激进会错过反弹）
        if market_regime == -1:
            max_gross_exposure = min(max_gross_exposure, 0.60)

        # 极端波动状态额外降仓
        if risk_extreme_exists:
            max_gross_exposure = max_gross_exposure * 0.6

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
            # 限制调整幅度：最低降到70%（0.7），避免过度减仓
            vol_ratio = max(0.7, min(1.0, vol_ratio))
            adjusted_exposure = max_gross_exposure * vol_ratio
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
        3. 行业减配 = 基础仓位 * industry_discount（弱势行业）
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

        # 行业减配（弱势行业）
        if sig.industry and sig.industry in INDUSTRY_DISCOUNT:
            base_position *= INDUSTRY_DISCOUNT[sig.industry]

        return base_position

    def _build_desired_value(
        self,
        date,
        universe,
        current_positions,
        signal_store,
        cash,
        prices,
        market_regime=0,
    ):
        """构建目标持仓

        统一选股逻辑（与验证一致）:
        1. 收集所有有信号的股票（不限制buy信号）
        2. 按分数排序，选Top-N
        3. 计算仓位权重
        """
        import pandas as pd

        total_equity = cash + sum(current_positions.values())

        # 检查并更新组合止损状态
        self._check_portfolio_stop_loss(total_equity)

        # 计算峰值和回撤
        if self.peak_equity is None:
            self.peak_equity = total_equity
        self.peak_equity = max(self.peak_equity, total_equity)

        drawdown = 1 - total_equity / self.peak_equity if self.peak_equity > 0 else 0.0

        # 收集所有候选股票（不限制buy信号，与验证一致）
        risk_extreme_exists = False
        candidates = []

        for code in universe:
            sig = signal_store.get(code, date)
            # 只要有信号且分数不是NaN就可以作为候选
            if sig and sig.score is not None and not (isinstance(sig.score, float) and sig.score != sig.score):
                # 检查极端状态
                if sig.risk_extreme:
                    risk_extreme_exists = True

                candidates.append({
                    'code': code,
                    'score': sig.score,
                    'risk_vol': sig.risk_vol,
                    'industry': sig.industry or '',
                    'sig': sig,
                })

        if not candidates:
            return {}

        # 按分数排序（与验证一致）
        candidates.sort(key=lambda x: x['score'], reverse=True)

        # 直接选Top-N（简化逻辑，与验证一致）
        selected = candidates[:self.max_position]

        if not selected:
            return {}

        # 计算仓位权重（简化：等权）
        total_position = 0
        for c in selected:
            # 简化权重计算：等权 + 波动率调整
            base_weight = 1.0

            # 波动率调整
            risk_vol = max(0.01, min(1.0, c['risk_vol']))
            vol_factor = min(1.0 / risk_vol, 2.0)  # 限制波动率影响

            # 极端状态降仓
            extreme_factor = 0.8 if c['sig'].risk_extreme else 1.0

            # 最终仓位权重
            c['position'] = base_weight * vol_factor * extreme_factor
            total_position += c['position']

        # 计算总仓位上限
        max_gross_exposure = self._calculate_position_limit(drawdown, risk_extreme_exists, market_regime)
        max_gross_exposure = self._apply_volatility_control(max_gross_exposure)

        # 归一化并应用仓位上限
        raw_weights = {}
        for c in selected:
            normalized_position = (c['position'] / total_position) * max_gross_exposure if total_position > 0 else 0
            raw_weights[c['code']] = normalized_position

        # 构建目标市值
        desired_value = {
            c: raw_weights[c] * total_equity
            for c in raw_weights
        }

        return desired_value

    # 负IR行业（应该排除或减配）
    EXCLUDE_INDUSTRIES = {'军工', '通信/计算机', '新能源车/风电'}

    @staticmethod
    def select_stocks(signals_df: 'pd.DataFrame',
                     top_n: int = 10,
                     use_percentile: bool = True,
                     percentile_range: tuple = (0.7, 0.9),
                     industry_weights: dict = None,
                     exclude_industries: set = None) -> 'pd.DataFrame':
        """统一选股方法

        使用排名百分位选股，避免极端反转效应。
        此方法被验证和回测共同使用，保证逻辑一致。

        Args:
            signals_df: 包含 code, score, industry 列的 DataFrame
            top_n: 选股数量
            use_percentile: 是否使用分位选股（默认True，选70-90%分位）
            percentile_range: 分位范围，默认(0.7, 0.9)即Q4
            industry_weights: 行业权重 dict，如 {'自动化/制造': 0.2, ...}
            exclude_industries: 排除的行业集合（默认排除负IR行业）

        Returns:
            选中的股票 DataFrame
        """
        import pandas as pd
        import numpy as np

        if signals_df is None or len(signals_df) == 0:
            return pd.DataFrame()

        df = signals_df.copy()

        # 默认不排除行业，可通过参数启用
        # 注意：排除行业会影响行业权重计算，需谨慎使用

        # 计算排名百分位
        df['rank_pct'] = df.groupby('date')['score'].rank(pct=True)

        if industry_weights is None:
            # 等权选股：使用分位选股
            selected = []
            for date, group in df.groupby('date'):
                if use_percentile:
                    # 选70-90%分位(Q4)
                    top = group[(group['rank_pct'] > percentile_range[0]) &
                               (group['rank_pct'] < percentile_range[1])]
                    if len(top) < 3:
                        top = group.nlargest(top_n, 'rank_pct')
                else:
                    top = group.nlargest(top_n, 'rank_pct')
                selected.append(top)

            if selected:
                return pd.concat(selected, ignore_index=True)
            return pd.DataFrame()
        else:
            # 行业加权选股 - 使用加权平均而非等权平均
            results = []
            for date, group in df.groupby('date'):
                weighted_return = 0
                total_weight = 0

                for industry, weight in industry_weights.items():
                    ind_group = group[group['industry'] == industry]
                    if len(ind_group) == 0:
                        continue

                    # 行业内的股票也按分位选
                    if use_percentile:
                        top_ind = ind_group[(ind_group['rank_pct'] > percentile_range[0]) &
                                           (ind_group['rank_pct'] < percentile_range[1])]
                        if len(top_ind) < 2:
                            n_select = max(1, int(top_n * weight * 3))
                            top_ind = ind_group.nlargest(n_select, 'rank_pct')
                    else:
                        n_select = max(1, int(top_n * weight * 3))
                        top_ind = ind_group.nlargest(n_select, 'rank_pct')

                    if len(top_ind) > 0:
                        # 加权平均：行业收益 * 行业权重
                        if 'return' in top_ind.columns:
                            ind_ret = top_ind['return'].mean()
                        elif 'future_ret' in top_ind.columns:
                            ind_ret = top_ind['future_ret'].mean()
                        else:
                            ind_ret = 0
                        weighted_return += ind_ret * weight
                        total_weight += weight

                if total_weight > 0:
                    weighted_return /= total_weight
                    results.append({'date': date, 'return': weighted_return})

            if results:
                return pd.DataFrame(results)
            return pd.DataFrame()

    def build(
        self,
        date,
        universe,
        current_positions,
        signal_store,
        cash,
        prices,
        market_regime,  # 市场状态用于熊市保护
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
                market_regime=market_regime,
            )

        # 强制卖出
        for code in stop_loss_sells:
            desired_value[code] = 0.0

        adjusted = {}

        # 非调仓日保持现有持仓
        if not rebalance and not stop_loss_sells:
            adjusted = deepcopy(current_positions)

        # 调仓逻辑（添加换仓门槛降低换手）
        for code, target in desired_value.items():
            current = current_positions.get(code, 0.0)
            diff = target - current

            if code in stop_loss_sells:
                adjusted[code] = 0.0
                continue

            # 换仓门槛：变化小于阈值时不调整（降低换手率）
            if abs(diff) < total_equity * REBALANCE_THRESHOLD * 0.1:  # 变化小于总资产1.5%不调
                adjusted[code] = current
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
