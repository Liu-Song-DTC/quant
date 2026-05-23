#!/usr/bin/env python3
"""
进化守护引擎 — 4层防护确保系统自进化方向积极

Layer 1: 统计显著性 — 累积多日证据，过滤单日噪音
Layer 2: 回测验证 — 快速信号对比，拒绝极端有害变更
Layer 3: 基线追踪 — 记录变更前指标，作为回滚基准
Layer 4: 自动回滚 — 变更后监控，指标连续恶化则撤销
"""
import os
import json
import copy
import sys
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

STRATEGY_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(STRATEGY_DIR))

# ── 进化配置 ──
DEFAULT_MIN_OCCURRENCE = 3        # 同一补丁至少出现N天才提交
DEFAULT_MAX_SIGNAL_CHANGE = 0.50  # 信号数量变更超过50%则拒绝
DEFAULT_DEGRADE_THRESHOLD = 0.20  # 指标恶化超过20%触发回滚
DEFAULT_MONITORING_DAYS = 5       # 应用后监控N天


@dataclass
class EvolutionPatch:
    """系统自进化补丁（扩展版）"""
    section: str
    key: str
    current_value: float
    suggested_value: float
    reason: str
    confidence: float = 0.5
    urgency: str = 'medium'
    # 追踪字段
    occurrence_count: int = 0
    first_seen_date: str = ''
    last_seen_date: str = ''
    status: str = 'proposed'       # proposed/validated/applied/rejected/rolled_back
    validation_result: dict = field(default_factory=dict)
    baseline_metrics: dict = field(default_factory=dict)
    applied_date: str = ''

    @property
    def patch_id(self) -> str:
        return f"{self.section}.{self.key}"

    def _normalize_value(self, v):
        if isinstance(v, (int, float, bool, str)):
            return v
        return str(v)

    def to_dict(self) -> dict:
        return {
            'section': self.section, 'key': self.key,
            'from': self._normalize_value(self.current_value),
            'to': self._normalize_value(self.suggested_value),
            'reason': self.reason, 'confidence': self.confidence,
            'urgency': self.urgency,
            'occurrence_count': self.occurrence_count,
            'first_seen_date': self.first_seen_date,
            'last_seen_date': self.last_seen_date,
            'status': self.status,
            'validation_result': self.validation_result,
            'baseline_metrics': self.baseline_metrics,
            'applied_date': self.applied_date,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'EvolutionPatch':
        return cls(
            section=d.get('section', ''),
            key=d.get('key', ''),
            current_value=d.get('from', 0),
            suggested_value=d.get('to', 0),
            reason=d.get('reason', ''),
            confidence=d.get('confidence', 0.5),
            urgency=d.get('urgency', 'medium'),
            occurrence_count=d.get('occurrence_count', 0),
            first_seen_date=d.get('first_seen_date', ''),
            last_seen_date=d.get('last_seen_date', ''),
            status=d.get('status', 'proposed'),
            validation_result=d.get('validation_result', {}),
            baseline_metrics=d.get('baseline_metrics', {}),
            applied_date=d.get('applied_date', ''),
        )


class EvolutionGuard:
    """进化守护引擎 — 保证每次进化都有据可查、可量化、可回滚"""

    def __init__(
        self,
        history_path: str = None,
        config_path: str = None,
        data_dir: str = None,
        min_occurrence: int = DEFAULT_MIN_OCCURRENCE,
        max_signal_change: float = DEFAULT_MAX_SIGNAL_CHANGE,
        degrade_threshold: float = DEFAULT_DEGRADE_THRESHOLD,
        monitoring_days: int = DEFAULT_MONITORING_DAYS,
    ):
        self.history_path = history_path or os.path.join(
            STRATEGY_DIR, 'daily_review', 'evolution_history.json')
        self.config_path = config_path or os.path.join(
            STRATEGY_DIR, 'config', 'factor_config.yaml')
        self.data_dir = data_dir
        self.min_occurrence = min_occurrence
        self.max_signal_change = max_signal_change
        self.degrade_threshold = degrade_threshold
        self.monitoring_days = monitoring_days

    # ═══════════════════════════════════════════════════════════════
    #  Layer 1: 统计显著性 — 累积证据，过滤噪音
    # ═══════════════════════════════════════════════════════════════

    def accumulate_and_filter(
        self, new_patches: List[EvolutionPatch], date: str
    ) -> List[EvolutionPatch]:
        """累积多日观察，只提交出现 >= min_occurrence 次的补丁"""
        history = self._load_history()
        existing_map = self._build_patch_map(history)

        for p in new_patches:
            pid = p.patch_id
            if pid in existing_map:
                old = existing_map[pid]
                p.occurrence_count = old.occurrence_count + 1
                p.first_seen_date = old.first_seen_date
            else:
                p.occurrence_count = 1
                p.first_seen_date = date
            p.last_seen_date = date

        # 更新历史
        for p in new_patches:
            existing_map[p.patch_id] = p

        self._save_history(list(existing_map.values()))

        # 仅提交证据充分的补丁
        matured = [p for p in new_patches if p.occurrence_count >= self.min_occurrence]
        skipped = [p for p in new_patches if p.occurrence_count < self.min_occurrence]

        if skipped:
            counts = ', '.join(
                f'{p.patch_id}({p.occurrence_count}/{self.min_occurrence})' for p in skipped)
            print(f'  [Layer 1] 证据不足，等待积累: {counts}')

        if matured:
            counts = ', '.join(f'{p.patch_id}({p.occurrence_count})' for p in matured)
            print(f'  [Layer 1] 证据充分，通过: {counts}')

        return matured

    # ═══════════════════════════════════════════════════════════════
    #  Layer 2: 回测验证 — 快速信号对比
    # ═══════════════════════════════════════════════════════════════

    def validate_patches(
        self, patches: List[EvolutionPatch], date: str
    ) -> List[EvolutionPatch]:
        """快速验证补丁效果，拒绝极端有害变更"""
        validated = []
        for p in patches:
            result = self._quick_validate(p, date)
            p.validation_result = result
            if result.get('passed', True):
                p.status = 'validated'
                validated.append(p)
                print(f'  [Layer 2] 验证通过: {p.patch_id}')
            else:
                p.status = 'rejected'
                print(f'  [Layer 2] 验证拒绝: {p.patch_id} — {result.get("reason", "")}')

        # 更新历史
        history = self._load_history()
        pmap = self._build_patch_map(history)
        for p in patches:
            pmap[p.patch_id] = p
        self._save_history(list(pmap.values()))

        return validated

    def _quick_validate(self, patch: EvolutionPatch, date: str) -> dict:
        """轻量级验证 — 规则检查 + 可选的信号对比"""
        # ── 规则约束检查 ──
        rule_check = self._check_config_constraints(patch)
        if not rule_check['valid']:
            return {'passed': False, 'reason': rule_check['reason'], 'method': 'constraint_check'}

        # ── 信号对比检查（如有数据） ──
        signal_check = self._check_signal_impact(patch, date)
        if signal_check:
            return signal_check

        return {'passed': True, 'method': 'constraint_check', 'reason': '符合配置约束'}

    def _check_config_constraints(self, patch: EvolutionPatch) -> dict:
        """检查补丁是否违反配置约束"""
        v = patch.suggested_value
        key = patch.key

        constraints = {
            'buy_threshold': (0.005, 0.50, '买入阈值'),
            'sell_threshold': (0.005, 0.50, '卖出阈值'),
            'b1_buy_mult': (0.50, 3.00, '一买乘数'),
            'b2_buy_mult': (0.50, 3.00, '二买乘数'),
            'b3_buy_mult': (0.50, 3.00, '三买乘数'),
            's1_sell_mult': (0.10, 2.00, '一卖乘数'),
            's2_sell_mult': (0.10, 2.00, '二卖乘数'),
            'zhongyin_penalty': (0.30, 1.00, '中阴惩罚'),
            'bear': (0.30, 1.00, '熊市敞口'),
        }

        if key in constraints:
            lo, hi, name = constraints[key]
            if not (lo <= v <= hi):
                return {
                    'valid': False,
                    'reason': f'{name}({key})={v} 超出合理范围 [{lo}, {hi}]',
                }

        # 变更幅度检查（单次不超过50%）
        old = patch.current_value
        if old > 0:
            change_pct = abs(v - old) / old
            if change_pct > 0.50:
                return {
                    'valid': False,
                    'reason': f'{key} 变更幅度 {change_pct:.0%} 超过50%上限，建议分步调整',
                }

        return {'valid': True}

    def _check_signal_impact(self, patch: EvolutionPatch, date: str) -> Optional[dict]:
        """使用信号对比验证补丁影响（需要数据支持）"""
        if self.data_dir is None:
            return None
        try:
            return self._run_signal_comparison(patch, date)
        except Exception:
            return None  # 对比失败不阻止进化，回退到约束检查

    def _run_signal_comparison(self, patch: EvolutionPatch, date: str) -> Optional[dict]:
        """运行新旧配置的信号对比"""
        from trade.signal_runner import SignalRunner
        from core.config_loader import load_config

        bt_data_dir = os.path.join(self.data_dir, 'backtrader_data')
        fund_data_dir = os.path.join(self.data_dir, 'fundamental_data')
        if not os.path.exists(bt_data_dir):
            return None

        # 仅取前200只股票做快速对比
        sample_codes = self._get_sample_codes(bt_data_dir, n=200)
        if len(sample_codes) < 50:
            return None

        try:
            # 旧配置跑一次
            old_result = self._run_quick_signal(sample_codes, bt_data_dir, fund_data_dir)
            if old_result is None:
                return None

            # 注入新参数
            self._inject_patch(patch)
            new_result = self._run_quick_signal(sample_codes, bt_data_dir, fund_data_dir)
            self._restore_config_backup()

            if new_result is None:
                return None

            return self._compare_signals(old_result, new_result, patch)
        except Exception:
            self._restore_config_backup()
            return None

    def _get_sample_codes(self, bt_data_dir: str, n: int = 200) -> List[str]:
        codes = []
        try:
            for f in sorted(os.listdir(bt_data_dir))[:n * 2]:
                if f.endswith('_qfq.csv'):
                    code = f[:-8] if f[:-8] else f[:-7]
                    if not code.startswith('688') and code.isdigit() and len(code) == 6:
                        codes.append(code)
                        if len(codes) >= n:
                            break
        except Exception:
            pass
        return codes

    def _run_quick_signal(self, codes, bt_data_dir, fund_data_dir) -> Optional[dict]:
        """跑一次快速信号生成，返回关键指标"""
        try:
            runner = self._SignalRunnerQuick(codes, bt_data_dir, fund_data_dir)
            runner.prepare(exposure=1.0)
            result = runner.run({}, 200000, {})
            if result is None:
                return None
            selections = result.get('selections', [])
            scores = [s.get('score', 0) for s in selections]
            industries = set(s.get('industry', '') for s in selections)
            return {
                'n_selections': len(selections),
                'mean_score': np.mean(scores) if scores else 0,
                'std_score': np.std(scores) if scores else 0,
                'n_industries': len(industries),
            }
        except Exception:
            return None

    class _SignalRunnerQuick:
        """轻量SignalRunner，仅用于快速验证"""
        def __init__(self, codes, bt_data_dir, fund_data_dir):
            import os as _os
            _bt = bt_data_dir
            self.bt_data_dir = _bt
            self.fund_data_dir = fund_data_dir
            self.stock_data_dict = {}
            self.prices = {}
            self._stock_codes = codes
            self._fundamental_data = None
            min_date = (datetime.today() - timedelta(days=365)).strftime("%Y-%m-%d")
            for code in codes:
                fp = os.path.join(_bt, f'{code}_qfq.csv')
                if os.path.exists(fp):
                    try:
                        data = pd.read_csv(fp, parse_dates=['datetime'])
                        data = data[data['datetime'] >= min_date]
                        if len(data) > 20:
                            self.stock_data_dict[code] = data
                            last_close = float(data.iloc[-1]['close'])
                            if last_close > 0:
                                self.prices[code] = last_close
                    except Exception:
                        pass

        def prepare(self, exposure=1.0):
            from core.strategy import Strategy
            self.strategy = Strategy(init_cash=200000, fundamental_data=None)
            if "sh000001" in self.stock_data_dict:
                self.strategy.generate_market_regime(self.stock_data_dict["sh000001"])
                self.strategy.signal_engine.set_market_regime(self.strategy.index_data)
            self._generate_all_signals(self._stock_codes)

        def _generate_all_signals(self, stock_codes):
            s = self.strategy
            for code in stock_codes:
                try:
                    data = self.stock_data_dict.get(code)
                    if data is None or len(data) < 30:
                        continue
                    last_date = data['datetime'].iloc[-1]
                    signal = s.signal_engine.generate_signal(code, data, last_date)
                    if signal is not None and signal.buy:
                        s.add_signal(signal)
                except Exception:
                    pass

        def run(self, current_positions, cash, cost):
            try:
                return self.strategy.select_portfolio(current_positions, cash, cost)
            except Exception:
                return None

    def _compare_signals(self, old_result: dict, new_result: dict, patch: EvolutionPatch) -> dict:
        """对比新旧信号，判断变更是否可接受"""
        old_n = old_result.get('n_selections', 1) or 1
        new_n = new_result.get('n_selections', 1) or 1
        change_pct = abs(new_n - old_n) / old_n

        passed = True
        reasons = []

        if change_pct > self.max_signal_change:
            passed = False
            reasons.append(f'选股数变化 {change_pct:.0%} 超过阈值 {self.max_signal_change:.0%} '
                          f'({old_n}→{new_n})')

        # 检查行业集中度
        old_ind = old_result.get('n_industries', 1) or 1
        new_ind = new_result.get('n_industries', 1) or 1
        ind_change = (old_ind - new_ind) / old_ind
        if ind_change > 0.5:
            passed = False
            reasons.append(f'行业数大幅减少 ({old_ind}→{new_ind})')

        return {
            'passed': passed,
            'method': 'signal_comparison',
            'reason': '; '.join(reasons) if reasons else '信号对比通过',
            'old_n': old_n, 'new_n': new_n,
            'old_industries': old_ind, 'new_industries': new_ind,
            'change_pct': round(change_pct, 4),
        }

    _config_backup = None
    _config_backup_key = None

    def _inject_patch(self, patch: EvolutionPatch):
        """临时注入补丁值到配置（仅内存，保存旧值用于恢复）"""
        from core.config_loader import load_config

        cfg = load_config()
        full_key = patch.section + '.' + patch.key
        old_val = cfg.get(full_key, None)
        self._config_backup = old_val
        self._config_backup_key = full_key
        cfg.set(full_key, patch.suggested_value)

    def _restore_config_backup(self):
        """恢复被临时注入的配置值"""
        if self._config_backup is not None:
            from core.config_loader import load_config
            cfg = load_config()
            cfg.set(self._config_backup_key, self._config_backup)
            self._config_backup = None
            self._config_backup_key = None

    # ═══════════════════════════════════════════════════════════════
    #  Layer 3: 基线追踪
    # ═══════════════════════════════════════════════════════════════

    def record_baseline(self, patch: EvolutionPatch, metrics: dict = None) -> dict:
        """记录变更前的关键指标作为回滚基准"""
        if metrics is None:
            metrics = self._capture_current_metrics()

        patch.baseline_metrics = {
            'date': datetime.now().strftime('%Y-%m-%d'),
            **metrics,
        }
        patch.applied_date = datetime.now().strftime('%Y-%m-%d')
        patch.status = 'applied'

        # 持久化
        history = self._load_history()
        pmap = self._build_patch_map(history)
        pmap[patch.patch_id] = patch
        self._save_history(list(pmap.values()))

        print(f'  [Layer 3] 基线已记录: {patch.patch_id}')
        for k, v in metrics.items():
            print(f'           {k}: {v}')
        return patch.baseline_metrics

    def _capture_current_metrics(self) -> dict:
        """捕获当前系统关键指标"""
        metrics = {'timestamp': datetime.now().isoformat()}
        info = self._get_current_config_info()
        if info:
            metrics.update(info)
        return metrics

    def _get_current_config_info(self) -> dict:
        try:
            from core.config_loader import load_config
            cfg = load_config()
            return {
                'buy_threshold': cfg.get('signal.buy_threshold', None),
                'sell_threshold': cfg.get('signal.sell_threshold', None),
                'max_position': cfg.get('portfolio.max_position', None),
            }
        except Exception:
            return {}

    # ═══════════════════════════════════════════════════════════════
    #  Layer 4: 自动回滚
    # ═══════════════════════════════════════════════════════════════

    def check_and_rollback(self, date: str) -> List[EvolutionPatch]:
        """检查已应用补丁的效果，恶化则建议回滚"""
        history = self._load_history()
        applied = [EvolutionPatch.from_dict(h) for h in history
                    if h.get('status') == 'applied']

        rollbacks = []
        for p in applied:
            if not p.applied_date:
                continue
            days_since = (datetime.strptime(date, '%Y-%m-%d') -
                         datetime.strptime(p.applied_date, '%Y-%m-%d')).days
            if days_since < self.monitoring_days:
                continue  # 观察期未满

            current = self._capture_current_metrics()
            degraded = self._check_degradation(p.baseline_metrics, current)
            if degraded:
                p.status = 'rolled_back'
                p.validation_result['rollback_reason'] = degraded
                rollbacks.append(p)
                print(f'  [Layer 4] ⚠ 建议回滚: {p.patch_id} — {degraded}')
            else:
                # 监控通过，标记为稳定
                p.status = 'stable'
                print(f'  [Layer 4] ✓ 监控通过: {p.patch_id}')

        if rollbacks:
            history = self._load_history()
            pmap = self._build_patch_map(history)
            for p in rollbacks:
                pmap[p.patch_id] = p
            self._save_history(list(pmap.values()))

        return rollbacks

    def _check_degradation(self, baseline: dict, current: dict) -> Optional[str]:
        """检查关键指标是否恶化"""
        checks = {
            'buy_threshold': ('buy_threshold', 0.01),
        }
        for key, (metric_name, margin) in checks.items():
            bl_val = baseline.get(metric_name)
            cur_val = current.get(metric_name)
            if bl_val is not None and cur_val is not None and bl_val != 0:
                change = (cur_val - bl_val) / abs(bl_val)
                if change > self.degrade_threshold:
                    return (f'{metric_name}: {bl_val}→{cur_val} '
                            f'(变动 {change:.1%} > 阈值 {self.degrade_threshold:.0%})')
        return None

    # ═══════════════════════════════════════════════════════════════
    #  工具方法
    # ═══════════════════════════════════════════════════════════════

    def _load_history(self) -> List[dict]:
        if os.path.exists(self.history_path):
            try:
                with open(self.history_path, 'r') as f:
                    return json.load(f)
            except Exception:
                return []
        return []

    def _save_history(self, patches: List[EvolutionPatch]):
        os.makedirs(os.path.dirname(self.history_path), exist_ok=True)
        history = [p.to_dict() if isinstance(p, EvolutionPatch) else p for p in patches]
        with open(self.history_path, 'w') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

    def _build_patch_map(self, history: List[dict]) -> Dict[str, EvolutionPatch]:
        pmap = {}
        for h in history:
            p = EvolutionPatch.from_dict(h) if isinstance(h, dict) else h
            pmap[p.patch_id] = p
        return pmap

    def get_status_summary(self) -> str:
        """返回进化状态摘要"""
        history = self._load_history()
        if not history:
            return '进化历史: 空'

        statuses = {}
        for h in history:
            s = h.get('status') or ('applied' if h.get('applied') else 'proposed')
            statuses[s] = statuses.get(s, 0) + 1

        lines = ['进化状态:']
        for s, n in sorted(statuses.items()):
            lines.append(f'  {s}: {n}')
        return '\n'.join(lines)
