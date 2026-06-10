# analysis/live_vs_backtest_monitor.py
"""
实盘vs回测持仓持续监控。

读取回测参考CSV (bt_execution输出) → 对比实盘SignalRunner持仓 → 报告偏离度。
用于检测策略漂移、数据变化或配置不一致。
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import date as date_type
from pathlib import Path


class LiveVsBacktestMonitor:
    """每日对比实盘持仓与回测参考，检测策略偏离。"""

    def __init__(self, reference_path: str = None, divergence_threshold: float = 0.30):
        if reference_path is None:
            strategy_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            reference_path = os.path.join(strategy_dir, 'rolling_validation_results', 'backtest_reference.csv')
        self.ref_path = Path(reference_path)
        self.threshold = divergence_threshold
        self.ref = None
        self._divergence_days = 0

    def load_reference(self):
        """加载回测参考数据"""
        if self.ref_path.exists():
            self.ref = pd.read_csv(self.ref_path, parse_dates=['date'])
            print(f"[Monitor] 加载回测参考: {len(self.ref)} 天 -> {self.ref_path}")
        else:
            print(f"[Monitor] 回测参考文件不存在: {self.ref_path}")

    def compare(self, live_positions: dict, live_date) -> dict:
        """对比实盘持仓与回测参考。

        Args:
            live_positions: {code: weight} 当前实盘持仓
            live_date: 对比日期

        Returns:
            dict with overlap metrics
        """
        if self.ref is None:
            self.load_reference()
        if self.ref is None:
            return {'status': 'no_reference', 'date': str(live_date)}

        date_ts = pd.Timestamp(live_date)
        row = self.ref[self.ref['date'] == date_ts]
        if row.empty:
            return {'status': 'no_reference', 'date': str(live_date)}

        try:
            bt_weights = json.loads(row.iloc[0]['code_weights'])
        except (json.JSONDecodeError, KeyError):
            return {'status': 'parse_error', 'date': str(live_date)}

        bt_codes = set(bt_weights.keys())
        live_codes = set(live_positions.keys())

        overlap = bt_codes & live_codes
        only_bt = bt_codes - live_codes
        only_live = live_codes - bt_codes

        total_bt_w = sum(bt_weights.values()) + 1e-10
        overlap_w = sum(bt_weights.get(c, 0) for c in overlap)
        overlap_pct = overlap_w / total_bt_w

        status = 'ok' if overlap_pct > (1 - self.threshold) else 'diverged'
        if status == 'diverged':
            self._divergence_days += 1
        else:
            self._divergence_days = 0

        return {
            'date': str(live_date),
            'overlap_codes': len(overlap),
            'total_bt_codes': len(bt_codes),
            'total_live_codes': len(live_codes),
            'overlap_weight_pct': round(overlap_pct, 3),
            'only_bt': list(only_bt)[:5],
            'only_live': list(only_live)[:5],
            'divergence_days': self._divergence_days,
            'status': status,
        }

    def print_report(self, result: dict):
        """格式化输出对比结果"""
        if result.get('status') == 'no_reference':
            return
        pct = result['overlap_weight_pct'] * 100
        icon = 'OK' if result['status'] == 'ok' else 'DIVERGED'
        print(f"[Monitor] {result['date']}: {icon} 持仓重合度={pct:.0f}% "
              f"({result['overlap_codes']}/{result['total_bt_codes']}只重叠)")
        if result['status'] == 'diverged':
            print(f"  回测独有: {', '.join(result['only_bt'])}")
            print(f"  实盘独有: {', '.join(result['only_live'])}")
            if result['divergence_days'] >= 3:
                print(f"  ⚠ 已连续偏离 {result['divergence_days']} 天，建议检查!")
