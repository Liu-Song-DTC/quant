# core/pipeline_logger.py
"""
选股链路追踪日志系统

全链路追踪: 因子选择 → 信号生成 → Gate过滤 → 组合构建 → 执行成交
每个阶段记录: 输入量、过滤量、输出量、质量分布、异常标记

用法:
    from core.pipeline_logger import plog

    plog.stage('signal_gen').count('buy_signals', 42)
    plog.flush_stage('signal_gen')  # 输出该阶段汇总
    plog.report()                   # 输出全链路报告
"""

import time
import sys
import math
import numpy as np
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple
from .decision_journal import DecisionJournal


class StageTracker:
    """单阶段追踪器 — 计数器 + 采样值 + 分布统计"""

    def __init__(self, name: str):
        self.name = name
        self.counters: Dict[str, int] = defaultdict(int)
        self.samples: Dict[str, list] = defaultdict(list)  # 最多保留 5000 个样本
        self._max_samples = 5000
        self._start_time = time.time()
        self._last_flush = time.time()
        self.population: Dict[str, dict] = {}

    def count(self, key: str, n: int = 1):
        self.counters[key] += n

    def sample(self, key: str, value: float):
        if value is None or not math.isfinite(value):
            self.counters[f'nonfinite_{key}'] += 1
            return
        value = float(value)
        p = self.population.setdefault(key, {'n': 0, 'mean': 0.0, 'm2': 0.0,
                                              'min': value, 'max': value, 'below_07': 0})
        p['n'] += 1
        delta = value - p['mean']
        p['mean'] += delta / p['n']
        p['m2'] += delta * (value - p['mean'])
        p['min'] = min(p['min'], value)
        p['max'] = max(p['max'], value)
        p['below_07'] += int(value < 0.7)
        if len(self.samples[key]) < self._max_samples:
            self.samples[key].append(value)

    def incr(self, key: str):
        """递增计数器 (语义别名)"""
        self.count(key, 1)

    def add(self, key: str, value: float):
        """累加值到计数器"""
        self.counters[key] = self.counters.get(key, 0) + value

    def snapshot(self) -> dict:
        """返回当前阶段的快照统计"""
        s = {'stage': self.name, 'counters': dict(self.counters),
             'elapsed_s': round(time.time() - self._start_time, 1)}
        for k, vals in self.samples.items():
            if vals:
                arr = np.array(vals)
                p = self.population[k]
                s[f'{k}_dist'] = {
                    'n': p['n'], 'mean': round(p['mean'], 5),
                    'std': round(float(np.sqrt(max(0., p['m2'] / p['n']))), 5),
                    'min': round(p['min'], 5), 'max': round(p['max'], 5),
                    'below_07': p['below_07'],
                    'quantile_sample_n': len(arr), 'quantile_scope': 'first_bounded_observations',
                    'p50': round(float(np.percentile(arr, 50)), 5),
                    'p90': round(float(np.percentile(arr, 90)), 5),
                }
        return s

    def reset(self):
        self.counters.clear()
        self.samples.clear()
        self.population.clear()
        self._last_flush = time.time()


class PipelineLogger:
    """全链路日志系统

    阶段顺序:
      factor_ic    — 因子IC预计算
      factor_sel   — 因子选择 (DYN vs fixed, 质量分布)
      signal_gen   — 信号生成 (buy/sell count, score分布, gate分布)
      gate_filter  — 双层门控: 4系统Gate(G1-G4) + 9买点Gate(B1-B5,BP6-8,BP0)
      portfolio    — 组合构建 (候选过滤链, 选股结果, 权重分配)
      execution    — 执行成交 (fill rate, 冲击成本, 换手)
    """

    def __init__(self):
        self.stages: Dict[str, StageTracker] = {}
        self._alerts: List[str] = []
        self._seq = 0  # 序号递增
        self._verbosity = 2  # 0=quiet, 1=summary, 2=detail
        self.journal = DecisionJournal()
        self._exit_trigger_evidence = {}

    def detail(self, event, date=None, code=None, **payload):
        if 'source_line' in payload and self.journal.enabled:
            # Literal line numbers drift when the caller is edited. Retain the
            # old annotation, but make the recorded location navigable.
            payload['declared_source_line'] = payload['source_line']
            try:
                frame = sys._getframe(1)
                try:
                    payload['source_line'] = frame.f_lineno
                    payload['source_file'] = frame.f_code.co_filename
                    payload['source_line_basis'] = 'runtime_call'
                finally:
                    del frame
            except (AttributeError, ValueError):
                payload['source_line_basis'] = 'unverified_annotation'
        # Observational only: keep concrete triggers alongside the historical
        # generic "stop_loss" caller label. Several requests may coexist; the
        # latest event is not asserted to be the sole cause of a later fill.
        trigger = payload.get('reason') if event == 'position_exit_trigger' else payload.get('trigger_reason') if event == 'position_exit_rule' else None
        if trigger and code is not None:
            records = self._exit_trigger_evidence.setdefault(code, deque(maxlen=8))
            records.append(dict(date=str(date if date is not None else self.journal.decision_date)[:10],
                                reason=trigger, event=event, source_line=payload.get('source_line'),
                                source_file=payload.get('source_file'),
                                source_line_basis=payload.get('source_line_basis')))
        self.journal.event(event, date=date, code=code, **payload)

    def bom_source(self, date, code, **payload):
        """Emit provenance transitions, keeping only the current stock's signature."""
        signature=(code,repr(payload))
        if getattr(self,'_bom_trace_signature',None)!=signature:
            self.detail('bom_source_transition',date=date,code=code,**payload)
            self._bom_trace_signature=signature

    def decision_day(self, date, **payload):
        self._exit_trigger_evidence.clear()
        self.journal.decision_date = date
        self.detail('decision_day', date=date, **payload)

    def candidate(self, date, code, stage, passed, reason='', sig=None, **payload):
        values = {}
        if sig is not None:
            for key in ('buy', 'sell', 'score', 'factor_value', 'factor_name', 'industry',
                        'chan_buy_point', 'bi_buy_point', 'chan_sell_point', 'signal_level',
                        'chan_divergence_type', 'chan_divergence_strength', 'trend_type',
                        'ma_trend_up', 'ml_score', 'pre_discount_score', 'adjusted_score',
                        'risk_vol', 'gate_quality', 'bom_quality_score'):
                values[key] = getattr(sig, key, None)
            values['gate_quality'] = getattr(sig, '_gate_quality', values.get('gate_quality'))
        self.detail('candidate', date=date, code=code, stage=stage, passed=bool(passed),
                    reason=reason, signal=values, **payload)

    @staticmethod
    def factor_input_observation(values, index):
        """Describe the untouched indicator input; never supplies a score value."""
        if values is None:
            return {'available': False, 'reason': 'column_missing_or_none', 'raw_value': None}
        try:
            value = float(values[index])
        except (TypeError, IndexError, KeyError, ValueError, OverflowError):
            return {'available': False, 'reason': 'unreadable_or_out_of_bounds', 'raw_value': None}
        if not math.isfinite(value):
            return {'available': False, 'reason': 'nonfinite', 'raw_value': None}
        return {'available': True, 'reason': 'observed', 'raw_value': value}

    @staticmethod
    def concept_input_observation(calculator, code, heat):
        """Read the existing snapshot only; observation failures never affect heat."""
        values = {'heat': heat, 'membership_count': None, 'quote_count': None,
                  'finite_quote_count': None, 'mean_known_ema_return': None,
                  'max_chain_prior': None, 'map_version': None, 'coverage': 'unknown'}
        try:
            mapping = getattr(calculator, '_stock_concepts', None)
            scores = getattr(calculator, '_concept_scores', None)
            values['map_version'] = getattr(calculator, '_cmap_cur', None)
            if not isinstance(mapping, dict) or not isinstance(scores, dict):
                return values
            concepts = mapping.get(code, [])
            quotes = [scores[c] for c in concepts if c in scores]
            finite = [float(v) for v in quotes
                      if isinstance(v, (int, float, np.number)) and math.isfinite(float(v))]
            values.update(membership_count=len(concepts), quote_count=len(quotes),
                          finite_quote_count=len(finite),
                          mean_known_ema_return=float(np.mean(finite)) if finite else None,
                          coverage=('no_membership' if not concepts else
                                    'no_market_quotes' if not scores else
                                    'members_but_no_quotes' if not quotes else 'has_quotes'))
            chain = getattr(calculator, '_chain_signals', None)
            if isinstance(chain, dict):
                values['max_chain_prior'] = max((chain.get(c, 0.0) for c in concepts), default=0.0)
        except Exception as error:
            values['observation_error'] = type(error).__name__
        return values

    def stage(self, name: str) -> StageTracker:
        if name not in self.stages:
            self.stages[name] = StageTracker(name)
        return self.stages[name]

    def alert(self, msg: str):
        """记录需要关注的异常"""
        self._alerts.append(msg)
        if self._verbosity >= 1:
            print(f"  ⚠ {msg}")

    # ── 便捷方法 ──

    def log_factor_sel(self, factor_name: str, is_dynamic: bool,
                       industry: str, quality: float, n_factors: int):
        """记录因子选择结果 (per stock per bar)"""
        s = self.stage('factor_sel')
        tag = 'DYN' if is_dynamic else 'FIXED'
        s.incr(f'{tag}_hits')
        s.incr(f'total_calls')
        s.sample(f'{tag}_quality', quality)
        s.sample(f'{tag}_n_factors', n_factors)
        if is_dynamic:
            s.incr(f'DYN_{industry}' if industry else 'DYN_unknown')

    def log_signal(self, buy: bool, sell: bool, score: float,
                   gate_quality: float, buy_point: int = 0,
                   industry: str = '', factor_name: str = '', valid: bool = True):
        """一次调用对应一个最终Signal；中性及无效行也计数，不重复计买卖同现行。"""
        s = self.stage('signal_gen')
        s.incr('total_bars')
        if not valid:
            s.incr('invalid_bars')
            return
        s.incr('valid_bars')
        if not buy and not sell:
            s.incr('neutral_bars')
        if buy and sell:
            s.incr('buy_sell_overlap_bars')
        if buy:
            s.incr('buy_signals')
            s.sample('buy_score', score)
            s.sample('buy_gate_quality', gate_quality)
            s.incr(f'buy_bp{buy_point}')
            if industry:
                s.incr(f'buy_ind_{industry}')
            # 因子归因: 按因子名前8字符分组 (去后缀标签)
            fn_short = factor_name[:8] if factor_name else 'unknown'
            s.incr(f'factor_{fn_short}')
        if sell:
            s.incr('sell_signals')
            s.sample('sell_score', score)
        s.sample('score', score)
        s.sample('gate_quality', gate_quality)

    def log_gate_dimensions(self, g1: float, g2: float, g3: float, g4: float,
                            gate_quality: float, passed: bool):
        """每次股票装配仅采最新bar；完整逐bar值在signal_decision事件。"""
        s = self.stage('gate_filter')
        for i, (name, val) in enumerate([('G1', g1), ('G2', g2), ('G3', g3), ('G4', g4)], 1):
            s.sample(f'{name}_grade', val)
        s.sample('composite_gate_quality', gate_quality)
        if not passed:
            s.incr('hard_reject_count')

    def signal_store_summary(self, frame, source):
        """主进程只读汇总实际消费行，避免把worker局部统计当成全局统计。"""
        buy=frame['buy'].fillna(False).astype(bool)
        sell=frame['sell'].fillna(False).astype(bool)
        record=dict(source=source,rows=len(frame),buy=int(buy.sum()),sell=int(sell.sum()),
                    overlap=int((buy&sell).sum()),neutral=int((~buy&~sell).sum()),
                    scope='Final store rows after membership masking; not the legacy pre-filter diagnostic counter.',
                    generation_details_available=source=='fresh_generation')
        self.detail('signal_store_summary',**record)
        print(f"[SIGNAL_FINAL] source={source} rows={len(frame)} buy={record['buy']} "
              f"sell={record['sell']} overlap={record['overlap']} neutral={record['neutral']}")

    def log_exit_reason(self, code: str, reason: str, pnl_pct: float = 0.0, days_held: int = None):
        """记录持仓退出原因 (止损/止盈/信号卖出/换仓)"""
        s = self.stage('portfolio')
        s.incr(f'exit_{reason}')
        s.sample(f'exit_pnl_{reason}', pnl_pct)
        if days_held is not None:
            s.sample('exit_days_held', days_held)
        current_day = str(self.journal.decision_date)[:10]
        trigger_evidence = [dict(row) for row in self._exit_trigger_evidence.get(code, ())
                            if row['date'] == current_day]
        self.detail('exit_intent', code=code, reason=reason, reported_pnl_pct=pnl_pct,
                    days_held=days_held, pnl_basis='caller_estimate_not_realized_cashflow',
                    trigger_evidence=trigger_evidence, reason_scope='legacy_caller_label',
                    trigger_evidence_scope='Last8 recorded same-day triggers for this code; full events remain in journal',
                    executed=False, execution_scope='Request only; fill and cash PnL require execution ledger reconciliation')

    def log_regime_execution(self, regime: int, daily_ret: float, nav: float):
        """记录分市场状态的执行表现"""
        s = self.stage('execution')
        regime_name = {1: 'bull', 0: 'neutral', -1: 'bear'}.get(regime, 'unknown')
        s.sample(f'ret_{regime_name}', daily_ret)
        s.incr(f'days_{regime_name}')

    def log_buy_point_gate(self, bp_name: str, rejected: bool, *, date=None, code=None):
        """记录买点门控通过/拒绝 (9门控: B1/B2/B3/B4/B5/BP6/BP7/BP8/BP0龙虎榜)"""
        s = self.stage('gate_filter')
        s.incr(f'bp_{bp_name}_checked')
        if rejected:
            s.incr(f'bp_{bp_name}_rejected')
        self.detail('signal_buy_point_gate', date=date, code=code,
                    buy_point=bp_name, rejected=rejected,
                    scope='executed branch result; later rules may override; see final signal')

    def log_data_quality(self, code: str, issue: str):
        """记录数据质量问题 (缺失/异常值)"""
        s = self.stage('data_quality')
        s.incr(f'issue_{issue}')
        # 保留最近20个问题样本用于诊断
        if s.counters.get(f'sample_{issue}', 0) < 20:
            s.incr(f'sample_{issue}')
            self.alert(f"data: {code} {issue}")

    def log_selection_context(self, date, market_regime: int, universe_size: int,
                              signal_count: int, gate_pass_count: int):
        """当候选不足时记录上下文 (诊断为什么选不出)"""
        s = self.stage('portfolio')
        tag = f'empty_context_r{market_regime}'
        s.incr(tag)
        # 只保留最近5次的详细信息
        if s.counters.get(tag, 0) <= 5:
            regime_name = {1: '牛', 0: '震', -1: '熊'}.get(market_regime, '?')
            self.alert(f"选股不足 [{date}] 市场={regime_name} universe={universe_size} "
                       f"signals={signal_count} gate_pass={gate_pass_count}")

    def log_gate_reject(self, gate_quality: float, reason: str):
        """记录Gate过滤拒绝"""
        s = self.stage('gate_filter')
        s.incr(f'reject_{reason}')
        s.sample('reject_gate_quality', gate_quality)

    def log_portfolio_candidate(self, reason: str):
        """记录组合层候选过滤"""
        s = self.stage('portfolio')
        s.incr(f'reject_{reason}')

    def log_portfolio_select(self, code: str, score: float, weight: float,
                             industry: str, rank_pct: float,
                             is_held: bool = False):
        """记录组合层选中"""
        s = self.stage('portfolio')
        s.incr('selected')
        s.sample('selected_score', score)
        s.sample('selected_weight', weight)
        s.sample('selected_rank', rank_pct)
        if is_held:
            s.incr('held_retained')
        s.incr(f'ind_{industry}')
        self.detail('portfolio_selected', code=code, score=score, weight=weight,
                    industry=industry, rank_pct=rank_pct, is_held=is_held)

    def log_execution(self, buy_filled: int = 0, sell_filled: int = 0,
                      buy_attempted: int = 0, sell_attempted: int = 0,
                      buy_limit_skip: int = 0, sell_limit_skip: int = 0,
                      cash_insufficient: int = 0, tplus1_blocked: int = 0,
                      buy_bear_blocked: int = 0, impact_cost: float = 0.0):
        """记录执行成交"""
        s = self.stage('execution')
        s.add('buy_filled', buy_filled)
        s.add('sell_filled', sell_filled)
        s.add('buy_attempted', buy_attempted)
        s.add('sell_attempted', sell_attempted)
        s.add('buy_limit_skip', buy_limit_skip)
        s.add('sell_limit_skip', sell_limit_skip)
        s.add('cash_insufficient', cash_insufficient)
        s.add('tplus1_blocked', tplus1_blocked)
        s.add('buy_bear_blocked', buy_bear_blocked)
        s.add('impact_cost_total', impact_cost)

    def log_backtest_daily(self, date, nav: float, drawdown: float,
                           n_positions: int, exposure: float,
                           daily_ret: float = 0.0):
        """记录每个交易日的实际账户状态，与输出账表逐日核对。"""
        s = self.stage('execution')
        s.sample('daily_nav', nav)
        s.sample('daily_drawdown', drawdown)
        s.sample('daily_n_positions', n_positions)
        s.sample('daily_exposure', exposure)
        s.sample('daily_return', daily_ret)
        self.detail('account_day', date=date, nav=nav, drawdown=drawdown,
                    n_positions=n_positions, exposure=exposure, daily_return=daily_ret)

    # ── 输出 ──

    def flush_stage(self, name: str) -> dict:
        """输出单个阶段的汇总并重置"""
        if name not in self.stages:
            return {}
        s = self.stages[name]
        snap = s.snapshot()
        if self._verbosity >= 1:
            self._print_stage(snap)
        s.reset()
        return snap

    def _print_stage(self, snap: dict):
        """格式化输出阶段快照"""
        name = snap['stage']
        counters = snap.get('counters', {})
        elapsed = snap.get('elapsed_s', 0)

        print(f"\n{'─' * 55}")
        print(f" [{name}] 耗时 {elapsed:.1f}s")
        print(f"{'─' * 55}")

        # 计数器按key分组输出
        if name == 'factor_sel':
            total = counters.get('total_calls', 1)
            dyn_hits = counters.get('DYN_hits', 0)
            fixed_hits = counters.get('FIXED_hits', 0)
            print(f"  因子选择: DYN={dyn_hits} ({100*dyn_hits/max(total,1):.1f}%)  "
                  f"FIXED={fixed_hits} ({100*fixed_hits/max(total,1):.1f}%)  "
                  f"总计={total}")
            # DYN按行业分布 (top 10)
            ind_items = sorted(
                [(k[4:], v) for k, v in counters.items() if k.startswith('DYN_')],
                key=lambda x: -x[1]
            )[:10]
            if ind_items:
                print(f"  DYN行业分布 (top10):")
                for ind, cnt in ind_items:
                    print(f"    {ind}: {cnt}")

        elif name == 'signal_gen':
            total = counters.get('total_bars', 1)
            buy = counters.get('buy_signals', 0)
            sell = counters.get('sell_signals', 0)
            print(f"  最终信号行: total_bars={total} valid={counters.get('valid_bars', 0)} "
                  f"invalid={counters.get('invalid_bars', 0)} neutral={counters.get('neutral_bars', 0)} "
                  f"buy_sell_overlap={counters.get('buy_sell_overlap_bars', 0)} "
                  f"buy={buy} ({100*buy/max(total,1):.2f}%)  "
                  f"sell={sell} ({100*sell/max(total,1):.2f}%)")
            if 'buy_score_dist' in snap:
                d = snap['buy_score_dist']
                print(f"  buy_score: n={d['n']} mean={d['mean']:.4f} std={d['std']:.4f} "
                      f"[{d['min']:.4f}, {d['max']:.4f}] p50={d['p50']:.4f}")
            if 'gate_quality_dist' in snap:
                d = snap['gate_quality_dist']
                print(f"  gate_quality: n={d['n']} mean={d['mean']:.3f} p50={d['p50']:.3f} p90={d['p90']:.3f}")
            # 买点分布
            bp_items = sorted(
                [(k[7:], v) for k, v in counters.items() if k.startswith('buy_bp')],
                key=lambda x: -x[1]
            )
            if bp_items:
                print(f"  买点分布: {', '.join(f'BP{bp}={cnt}' for bp, cnt in bp_items[:8])}")
            # 因子归因 (top 12)
            factor_items = sorted(
                [(k[7:], v) for k, v in counters.items() if k.startswith('factor_')],
                key=lambda x: -x[1]
            )[:12]
            if factor_items:
                print(f"  买入信号因子归因 (top12):")
                for fn, cnt in factor_items:
                    pct = 100 * cnt / max(buy, 1)
                    print(f"    {fn}: {cnt} ({pct:.1f}%)")
            # 行业分布 (top 10)
            ind_items = sorted(
                [(k[8:], v) for k, v in counters.items() if k.startswith('buy_ind_')],
                key=lambda x: -x[1]
            )[:10]
            if ind_items:
                print(f"  买入行业 (top10):")
                for ind, cnt in ind_items:
                    print(f"    {ind}: {cnt}")

        elif name == 'gate_filter':
            print('  Gate维度摘要：每只股票装配最新bar样本；逐bar明细以decision trace为准。')
            # Gate分项质量分布 (4系统门控)
            for g_name in ['G1', 'G2', 'G3', 'G4']:
                key = f'{g_name}_grade_dist'
                if key in snap:
                    d = snap[key]
                    print(f"  {g_name}: mean={d['mean']:.3f} p50={d['p50']:.3f} p90={d['p90']:.3f} "
                          f"[{d['min']:.3f}, {d['max']:.3f}]")
            if 'composite_gate_quality_dist' in snap:
                d = snap['composite_gate_quality_dist']
                print(f"  综合gate_quality: mean={d['mean']:.3f} p50={d['p50']:.3f} "
                      f"<0.7={100*d['below_07']/max(d['n'],1):.1f}%")
            hard_rejects = counters.get('hard_reject_count', 0)
            print(f"  Hard拒绝: {hard_rejects}")

            # 买点门控通过率 (9门控)
            bp_gates = ['B1', 'B2', 'B3', 'B4', 'B5', 'BP6', 'BP7', 'BP8', 'BP0']
            bp_lines = []
            for bp in bp_gates:
                checked = counters.get(f'bp_{bp}_checked', 0)
                rejected = counters.get(f'bp_{bp}_rejected', 0)
                if checked > 0:
                    pct = 100 * rejected / checked
                    bp_lines.append(f'{bp}: {rejected}/{checked}={pct:.0f}%')
            if bp_lines:
                print(f"  买点门控拒绝率: {' | '.join(bp_lines)}")

        elif name == 'portfolio':
            selected = counters.get('selected', 0)
            held = counters.get('held_retained', 0)
            reject_total = sum(v for k, v in counters.items() if k.startswith('reject_'))
            print(f"  组合构建: selected={selected} held_retained={held}  reject_total={reject_total}")
            for k, v in sorted(counters.items()):
                if k.startswith('reject_'):
                    print(f"    reject {k[7:]}: {v}")
            # 退出原因分布
            exit_items = sorted(
                [(k[5:], v) for k, v in counters.items() if k.startswith('exit_') and not k.startswith('exit_pnl') and not k.startswith('exit_days')],
                key=lambda x: -x[1]
            )
            if exit_items:
                print(f"  持仓退出原因:")
                for reason, cnt in exit_items:
                    pnl_key = f'exit_pnl_{reason}_dist'
                    pnl_str = ''
                    if pnl_key in snap:
                        pnl_str = f" (avg PnL={snap[pnl_key]['mean']*100:.1f}%)"
                    print(f"    {reason}: {cnt}{pnl_str}")
            if 'selected_score_dist' in snap:
                d = snap['selected_score_dist']
                print(f"  选中score: mean={d['mean']:.4f} std={d['std']:.4f} "
                      f"[{d['min']:.4f}, {d['max']:.4f}]")
            if 'selected_weight_dist' in snap:
                d = snap['selected_weight_dist']
                print(f"  选中weight: mean={d['mean']:.3f} max={d['max']:.3f}")
            ind_items = sorted(
                [(k[4:], v) for k, v in counters.items() if k.startswith('ind_')],
                key=lambda x: -x[1]
            )[:8]
            if ind_items:
                print(f"  选中行业分布: {', '.join(f'{ind}={cnt}' for ind, cnt in ind_items)}")
            # 空候选上下文
            empty_keys = sorted([k for k in counters if k.startswith('empty_context_')])
            if empty_keys:
                print(f"  选股不足上下文:")
                for k in empty_keys:
                    print(f"    {k}: {counters[k]}次")

        elif name == 'execution':
            print(f"  执行成交:")
            ba = max(counters.get('buy_attempted', 0), 1)
            sa = max(counters.get('sell_attempted', 0), 1)
            bf = counters.get('buy_filled', 0)
            sf = counters.get('sell_filled', 0)
            print(f"    买入: {bf}/{counters.get('buy_attempted',0)} = {100*bf/ba:.1f}%  "
                  f"(涨停跳过={counters.get('buy_limit_skip',0)}, 现金不足={counters.get('cash_insufficient',0)})")
            print(f"    卖出: {sf}/{counters.get('sell_attempted',0)} = {100*sf/sa:.1f}%  "
                  f"(跌停跳过={counters.get('sell_limit_skip',0)}, T+1拦截={counters.get('tplus1_blocked',0)})")
            # 分市场状态收益
            regime_lines = []
            for regime_name in ['bull', 'neutral', 'bear']:
                ret_key = f'ret_{regime_name}_dist'
                days_key = f'days_{regime_name}'
                if ret_key in snap:
                    d = snap[ret_key]
                    days = counters.get(days_key, 0)
                    regime_lines.append(
                        f"{regime_name}: {days}天 ret_mean={d['mean']*100:.2f}% ret_std={d['std']*100:.2f}%"
                    )
            if regime_lines:
                print(f"    分市场: {' | '.join(regime_lines)}")
            if 'daily_nav_dist' in snap:
                d = snap['daily_nav_dist']
                print(f"  每日状态: n={d['n']} NAV range=[{d['min']:.0f}, {d['max']:.0f}]")
            if 'daily_drawdown_dist' in snap:
                d = snap['daily_drawdown_dist']
                print(f"  回撤分布: mean={d['mean']*100:.1f}% 最深={d['min']*100:.1f}% "
                      f"p90={d['p90']*100:.1f}% (分位数基于{d['quantile_sample_n']}条样本)")

        elif name == 'data_quality':
            issues = sorted(
                [(k[6:], v) for k, v in counters.items() if k.startswith('issue_')],
                key=lambda x: -x[1]
            )
            if issues:
                print(f"  数据质量问题:")
                for issue, cnt in issues:
                    print(f"    {issue}: {cnt}")

    def report(self, print_all: bool = True) -> dict:
        """输出全链路汇总报告"""
        report_data = {
            'stages': {name: s.snapshot() for name, s in self.stages.items()},
            'alerts': list(self._alerts),
        }

        if print_all and self._verbosity >= 1:
            print(f"\n{'=' * 55}")
            print(f" 全链路追踪报告")
            print(f"{'=' * 55}")
            for name in ['factor_sel', 'signal_gen', 'gate_filter', 'portfolio', 'execution', 'data_quality']:
                if name in report_data['stages']:
                    self._print_stage(report_data['stages'][name])
            if self._alerts:
                print(f"\n 异常标记 ({len(self._alerts)}):")
                for a in self._alerts:
                    print(f"  - {a}")
            print(f"{'=' * 55}\n")

        return report_data

    def reset_all(self):
        for s in self.stages.values():
            s.reset()
        self._alerts.clear()
        self._exit_trigger_evidence.clear()


# ── 全局单例 ──
plog = PipelineLogger()
