# core/portfolio.py
import numpy as np
from copy import deepcopy
from .config_loader import load_config

# === 阶段3优化：基于买入准确率的行业权重调整 ===
# 数据来源：validation_results.csv 的买入信号准确率分析
# 准确率>52%: 增配（权重1.2）
# 准确率50-52%: 正常（权重1.0）
# 准确率<50%: 减配（权重0.5-0.75）
INDUSTRY_DISCOUNT = {
    # 高准确率行业 - 增配
    '通信/计算机': 1.2,     # 准确率54.55%，增配20%
    '化工': 1.1,           # 准确率52.03%，增配10%
    # 中等准确率 - 正常
    '电子': 1.0,           # 准确率51.00%
    '电力设备': 1.0,       # 准确率50.94%
    # 低准确率行业 - 减配
    '互联网/软件': 0.6,    # 准确率47.83%，减配40%
    '金融': 0.65,          # 准确率48.85%，减配35%
    '基建/地产/石油石化': 0.6,  # 准确率48.09%，减配40%
    '交运': 0.6,           # 准确率48.32%，减配40%
    '半导体/光伏': 0.8,    # 准确率49.57%，减配20%
}

# === 模块1优化: 降低换手率 ===
# 核心思路:
# 1. 提高换仓门槛，减少小幅波动导致的频繁交易
# 2. 引入"保留区域"概念，排名下降不多时不立即卖出
# 3. 渐进调整仓位而非一次性全部调整

# 换仓门槛（提高以降低换手率）
# 实际阈值 = total_equity * REBALANCE_THRESHOLD * 0.1
# 0.25 → 2.5% of equity 变化才触发调仓
REBALANCE_THRESHOLD = 0.25  # 持仓变化超过25%才调仓（从15%提高）

# 保留区域参数
KEEP_RANK_BUFFER = 2  # 排名在此范围内下降仍保留


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
        self.max_single_weight = portfolio_config.get('max_single_weight', 0.15)  # 单只股票最大权重

        # 波动率控制和组合止损默认关闭
        self.volatility_control_enabled = config.get('volatility_control.enabled', False)
        self.portfolio_stop_loss_enabled = config.get('portfolio_stop_loss.enabled', False)
        self.emergency_exposure = config.get('portfolio_stop_loss.emergency_exposure', 0.30)
        # 行业加权选股
        self.enable_industry_weighting = portfolio_config.get('enable_industry_weighting', True)

        self.peak_equity = None
        self.position_cost = {}
        self.last_selection = []  # 记录最近一次选股结果 (date, code, score, weight, industry)
        self.consecutive_losses = 0
        # 波动率控制相关
        self.equity_history = []  # 历史净值
        self.volatility_lookback = config.get('volatility_control.lookback_period', 20)
        self.current_volatility = 0.0
        # 组合止损相关
        self.portfolio_stop_loss_triggered = False
        # === 模块1优化: 智能换仓 ===
        self.current_ranking = {}  # code -> rank (0-indexed) for smart rebalancing
        self.current_n_positions = 0  # current max positions for keep zone calculation

    def _get_risk_multiplier(self, regime, risk_extreme, momentum_score=0.0):
        """计算风险乘数

        注意：momentum_score是滞后指标，不应过度依赖
        """
        if risk_extreme:  # 极端波动
            return 0.7
        return 1.0  # 正常仓位

    def _get_position_count(self, market_regime, risk_extreme_exists):
        """计算动态持仓数量

        熊市减少持仓数量但更集中（IC更高，命中率更高）
        """
        base_count = self.max_position

        if market_regime == -1:  # 熊市
            # 熊市减少持仓数量，更集中
            return max(3, int(base_count * 0.5))
        elif risk_extreme_exists:  # 极端波动
            # 极端波动也减少持仓
            return max(4, int(base_count * 0.6))
        else:
            return base_count

    def _calculate_position_limit(self, drawdown, risk_extreme_exists, market_regime=0, momentum_score=0.0):
        """根据回撤和极端状态计算总仓位上限

        A股优化：收紧阈值，控制回撤
        - 5%/10%/15% → 0.50/0.70/0.90
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

        # 应用风险乘数
        risk_multiplier = self._get_risk_multiplier(market_regime, risk_extreme_exists, momentum_score)
        max_gross_exposure = max_gross_exposure * risk_multiplier

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
        momentum_score=0.0,
    ):
        """构建目标持仓 - 基于因子排名选股

        核心改进：
        1. 使用 factor_value 进行排名（而非 score）
        2. 截面标准化（排名百分位）
        3. 直接利用 IC 的排序能力
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

        # === 排名选股：收集所有股票的因子值 ===
        risk_extreme_exists = False
        candidates = []

        for code in universe:
            sig = signal_store.get(code, date)
            if sig is None:
                continue

            # 使用 factor_value 作为排名依据（核心改变）
            factor_value = getattr(sig, 'factor_value', None)
            if factor_value is None or (isinstance(factor_value, float) and np.isnan(factor_value)):
                continue

            # 检查极端状态
            if getattr(sig, 'risk_extreme', False):
                risk_extreme_exists = True

            candidates.append({
                'code': code,
                'factor_value': factor_value,
                'score': getattr(sig, 'score', 0),
                'risk_vol': getattr(sig, 'risk_vol', 0.03),
                'industry': getattr(sig, 'industry', '') or 'default',
                'sig': sig,
            })

        if not candidates:
            return {}

        # === 截面标准化：使用排名百分位 ===
        factor_values = np.array([c['factor_value'] for c in candidates])
        rank_pct = pd.Series(factor_values).rank(pct=True)  # 0-1 之间的排名

        for i, c in enumerate(candidates):
            c['rank_pct'] = rank_pct.iloc[i]

        # === 按排名选择股票 ===
        # 行业分散限制
        industry_count = {}
        industry_cap = 2  # 单个行业最多选2只

        # 按排名百分位排序
        candidates.sort(key=lambda x: x['rank_pct'], reverse=True)

        selected = []
        for c in candidates:
            ind = c['industry']
            if industry_count.get(ind, 0) >= industry_cap:
                continue

            # 排除排名太低的股票（后50%）
            if c['rank_pct'] < 0.5:
                continue

            selected.append(c)
            industry_count[ind] = industry_count.get(ind, 0) + 1

            # 动态持仓数量
            n_positions = self._get_position_count(market_regime, risk_extreme_exists)
            if len(selected) >= n_positions:
                break

        if not selected:
            return {}

        # 记录排名信息
        self.current_ranking = {c['code']: i for i, c in enumerate(candidates)}

        # === 计算权重 ===
        total_position = 0
        for i, c in enumerate(selected):
            # 指数衰减权重
            score_weight = np.exp(-0.15 * i)

            # 波动率调整
            risk_vol = max(0.01, min(1.0, c['risk_vol']))
            vol_factor = min(1.0 / risk_vol, 2.0)

            # 极端状态降仓
            extreme_factor = 0.7 if c['sig'].risk_extreme else 1.0

            # === 因子质量加权 ===
            # 高质量因子给的股票更高权重
            factor_quality = getattr(c['sig'], 'factor_quality', 0.0)
            # 质量因子：质量越高权重越大，范围[0.5, 1.5]
            quality_factor = 0.5 + factor_quality * 10  # 假设quality在0-0.1范围
            quality_factor = max(0.5, min(1.5, quality_factor))

            c['position'] = score_weight * vol_factor * extreme_factor * quality_factor
            total_position += c['position']

        # 总仓位上限
        max_gross_exposure = self._calculate_position_limit(drawdown, risk_extreme_exists, market_regime, momentum_score)
        max_gross_exposure = self._apply_volatility_control(max_gross_exposure)

        # 归一化
        raw_weights = {}
        for c in selected:
            raw_weights[c['code']] = (c['position'] / total_position) * max_gross_exposure if total_position > 0 else 0

        # === 风险控制：限制单只股票权重 ===
        # 防止过度集中，提高组合分散度
        for code in raw_weights:
            if raw_weights[code] > self.max_single_weight:
                raw_weights[code] = self.max_single_weight

        # 重新归一化（权重被限制后需要重新调整）
        total_weight = sum(raw_weights.values())
        if total_weight > 0:
            raw_weights = {c: w / total_weight * max_gross_exposure for c, w in raw_weights.items()}

        # 构建目标市值
        desired_value = {c: raw_weights[c] * total_equity for c in raw_weights}

        # 记录选股结果
        self.last_selection = [
            {
                'date': date,
                'code': c['code'],
                'score': c['score'],
                'weight': raw_weights.get(c['code'], 0),
                'industry': c.get('industry', ''),
                'rank_pct': c.get('rank_pct', 0),
            }
            for c in selected
        ]

        self.current_n_positions = len(selected)

        return desired_value

    def build(
        self,
        date,
        universe,
        current_positions,
        signal_store,
        cash,
        prices,
        market_regime,  # 市场状态用于熊市保护
        momentum_score=0.0,  # 市场动量分数，用于动态仓位调整
        cost=None,
        rebalance=False,
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
            # 使用 sig.sell 作为卖出条件（factor_value < -0.15）
            # 不再叠加 score < -0.20，避免过于严格的卖出条件
            if sig and sig.sell:
                stop_loss_sells[code] = 0.0

            # 止盈/降级卖出：如果分数转为负值，卖出持仓
            # 这确保了不再被看好的股票被及时卖出
            if sig and sig.score < 0:
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
                momentum_score=momentum_score,
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

        # === 模块1优化: 智能换仓 ===
        # 不再简单地清仓不在Top-N的股票，而是:
        # 1. 检查旧持仓是否仍在新排名的"保留区域"内
        # 2. 如果在保留区域内，保留该持仓并渐进调整权重
        # 3. 只有真正出局（排名>KEEP_RANK_BUFFER）的才卖出

        if rebalance:
            keep_zone = self.current_n_positions + KEEP_RANK_BUFFER  # 保留区域大小

            for code, current in current_positions.items():
                if code in adjusted:
                    # 已经在上面的循环中被处理（要么在desired_value中，要么被止损卖出）
                    continue

                # 检查该持仓是否在保留区域内
                if hasattr(self, 'current_ranking') and code in self.current_ranking:
                    rank = self.current_ranking[code]
                    if rank < keep_zone:
                        # 在保留区域内，降低权重但不卖出
                        # 权重逐渐降低到原来的一半
                        kept_weight = current * 0.5
                        if kept_weight > total_equity * REBALANCE_THRESHOLD * 0.05:
                            adjusted[code] = kept_weight
                        else:
                            adjusted[code] = 0.0
                    else:
                        # 真正出局，卖出
                        adjusted[code] = 0.0
                else:
                    # 不在候选列表中（没有信号或被排除），卖出
                    adjusted[code] = 0.0

        return adjusted
