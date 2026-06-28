# analysis/backtest_diagnostics.py
"""
回测诊断模块 — 在每个改动点记录关键指标，回测结束时输出对比报告。

记录内容:
1. 因子选择分布 (DYN/IND/默认/V41)
2. 买入信号按买点类型(B1-B5)和因子家族分布
3. ML预测覆盖率和信号强度
4. 因子择时连续权重分布
5. 另类数据命中率
6. 门控质量分布
7. 风险防御触发记录(硬回撤/熔断)

用法: 回测结束后自动调用 print_diagnostics()
"""

import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
from collections import Counter, defaultdict


class BacktestDiagnostics:
    """回测诊断收集器 — 单例，在回测各阶段记录指标"""

    def __init__(self):
        self.reset()

    def reset(self):
        """每次回测前重置"""
        self.metrics = {
            # 因子选择
            'factor_selection': Counter(),       # {path: count}
            'factor_family': Counter(),           # {family: count}
            # 买入信号
            'buy_signals': 0,
            'buy_by_buy_point': Counter(),        # {bp: count}
            'buy_by_factor_family': Counter(),    # {family: count}
            # ML
            'ml_predictions_total': 0,
            'ml_predictions_active': 0,
            'ml_validation_ic': None,
            # 因子择时
            'trend_score_samples': [],
            'blend_weight_samples': [],            # [(bull_w, neutral_w, bear_w)]
            # 另类数据
            'alt_northbound_hits': 0,
            'alt_margin_hits': 0,
            'alt_dragon_tiger_hits': 0,
            # 门控
            'gate_quality_samples': [],
            'hard_rejects': 0,
            # 风险防御
            'hds_triggered': False,
            'clb_triggered': False,
            'clb_max_losses': 0,
            # 成交
            'fill_stats': {},
        }
        self.warnings = []

    # ---- 记录接口 ----

    def record_factor_selection(self, factor_name: str, is_industry: bool, is_dynamic: bool):
        """记录每次因子选择"""
        if is_dynamic:
            self.metrics['factor_selection']['DYN'] += 1
            self.metrics['factor_family']['DYN'] += 1
        elif is_industry:
            self.metrics['factor_selection']['IND'] += 1
            self.metrics['factor_family']['IND'] += 1
        elif factor_name in ('MOM', 'REV', 'SHARPE'):
            self.metrics['factor_selection']['DEFAULT'] += 1
            self.metrics['factor_family'][factor_name] += 1
        elif factor_name == 'V41':
            self.metrics['factor_selection']['V41'] += 1
            self.metrics['factor_family']['V41'] += 1
        else:
            self.metrics['factor_selection']['OTHER'] += 1

    def record_buy_signal(self, buy_point: int, factor_name: str):
        """记录每个买入信号的买点和因子"""
        self.metrics['buy_signals'] += 1
        self.metrics['buy_by_buy_point'][buy_point] += 1
        fam = 'DYN' if str(factor_name).startswith('DYN_') else \
              'IND' if str(factor_name).startswith('IND_') else \
              'SHARPE' if str(factor_name).startswith('SHARPE') else \
              'MOM' if str(factor_name).startswith('MOM') else \
              'REV' if str(factor_name).startswith('REV') else 'OTHER'
        self.metrics['buy_by_factor_family'][fam] += 1

    def record_ml(self, total: int, active: int, val_ic: float = None):
        self.metrics['ml_predictions_total'] = total
        self.metrics['ml_predictions_active'] = active
        if val_ic is not None:
            self.metrics['ml_validation_ic'] = val_ic

    def record_factor_timing(self, bull_w: float, neutral_w: float, bear_w: float):
        self.metrics['blend_weight_samples'].append((bull_w, neutral_w, bear_w))

    def record_alt_data(self, northbound: bool = False, margin: bool = False, dragon_tiger: bool = False):
        if northbound: self.metrics['alt_northbound_hits'] += 1
        if margin: self.metrics['alt_margin_hits'] += 1
        if dragon_tiger: self.metrics['alt_dragon_tiger_hits'] += 1

    def record_gate(self, gate_quality: float, hard_reject: bool = False):
        self.metrics['gate_quality_samples'].append(gate_quality)
        if hard_reject:
            self.metrics['hard_rejects'] += 1

    def record_risk_event(self, hds: bool = False, clb: bool = False, clb_losses: int = 0):
        if hds: self.metrics['hds_triggered'] = True
        if clb: self.metrics['clb_triggered'] = True
        self.metrics['clb_max_losses'] = max(self.metrics['clb_max_losses'], clb_losses)

    def record_fill_stats(self, stats: dict):
        self.metrics['fill_stats'] = stats

    # ---- 诊断报告 ----

    def print_report(self):
        """回测结束时打印诊断报告"""
        m = self.metrics
        print("\n" + "=" * 70)
        print("回测诊断报告 (Backtest Diagnostics)")
        print("=" * 70)

        # 1. 因子选择分布
        total_sel = sum(m['factor_selection'].values()) or 1
        print(f"\n--- 因子选择分布 (总计 {total_sel:,}) ---")
        for path in ['DYN', 'IND', 'DEFAULT', 'V41', 'OTHER']:
            cnt = m['factor_selection'].get(path, 0)
            pct = cnt / total_sel * 100
            bar = '█' * int(pct / 2)
            print(f"  {path:<10}: {cnt:>8,} ({pct:5.1f}%) {bar}")
        if m['factor_selection'].get('DYN', 0) / total_sel < 0.01:
            self.warnings.append("DYN因子命中率<1%，动态因子选择几乎无效")

        # 2. 买入信号分布
        total_buys = m['buy_signals'] or 1
        print(f"\n--- 买入信号分布 (总计 {total_buys:,}) ---")
        bp_names = {0: '无结构', 1: 'B1一买', 2: 'B2二买', 3: 'B3三买', 4: 'B4回调', 5: 'B5启动'}
        for bp in sorted(m['buy_by_buy_point'].keys()):
            cnt = m['buy_by_buy_point'][bp]
            print(f"  {bp_names.get(bp, f'BP{bp}'):<10}: {cnt:>6,} ({cnt/total_buys*100:5.1f}%)")
        if m['buy_by_buy_point'].get(0, 0) / total_buys > 0.35:
            self.warnings.append(f"无结构买入占比{m['buy_by_buy_point'][0]/total_buys*100:.0f}% (>35%), B4/B5未生效或覆盖不足")

        # 因子家族买入分布
        print(f"\n  按因子家族:")
        for fam in ['DYN', 'IND', 'SHARPE', 'MOM', 'REV', 'OTHER']:
            cnt = m['buy_by_factor_family'].get(fam, 0)
            if cnt > 0:
                print(f"    {fam:<10}: {cnt:>6,} ({cnt/total_buys*100:5.1f}%)")

        # 3. ML
        print(f"\n--- ML预测 ---")
        print(f"  总预测: {m['ml_predictions_total']:,}")
        print(f"  有效预测(|pred|>0.01): {m['ml_predictions_active']:,} "
              f"({m['ml_predictions_active']/max(m['ml_predictions_total'],1)*100:.1f}%)")
        if m['ml_validation_ic'] is not None:
            print(f"  验证IC: {m['ml_validation_ic']:.4f}")
            if abs(m['ml_validation_ic']) < 0.02:
                self.warnings.append(f"ML验证IC={m['ml_validation_ic']:.4f} (<0.02), ML近乎无效")

        # 4. 因子择时
        if m['blend_weight_samples']:
            bw = np.array(m['blend_weight_samples'])
            print(f"\n--- 因子择时权重 (N={len(bw):,}) ---")
            print(f"  bull_w:   mean={bw[:,0].mean():.3f} std={bw[:,0].std():.3f}")
            print(f"  neutral_w: mean={bw[:,1].mean():.3f} std={bw[:,1].std():.3f}")
            print(f"  bear_w:   mean={bw[:,2].mean():.3f} std={bw[:,2].std():.3f}")
            if bw[:,0].std() < 0.02:
                self.warnings.append("因子择时bull_w std<0.02，权重几乎无变化，连续插值未生效")

        # 5. 另类数据
        print(f"\n--- 另类数据 ---")
        print(f"  北向资金命中: {m['alt_northbound_hits']:,}")
        print(f"  融资融券命中: {m['alt_margin_hits']:,}")
        print(f"  龙虎榜命中:   {m['alt_dragon_tiger_hits']:,}")
        if m['alt_northbound_hits'] == 0:
            self.warnings.append("北向资金数据不可用（下载失败或缓存缺失）")

        # 6. 门控
        if m['gate_quality_samples']:
            gq = np.array(m['gate_quality_samples'])
            print(f"\n--- 门控质量 (N={len(gq):,}) ---")
            print(f"  mean={gq.mean():.3f} median={np.median(gq):.3f}")
            print(f"  P10={np.percentile(gq,10):.3f} P90={np.percentile(gq,90):.3f}")
            print(f"  硬拒绝: {m['hard_rejects']:,}")

        # 7. 风险防御
        print(f"\n--- 风险防御 ---")
        print(f"  硬回撤停止触发: {m['hds_triggered']}")
        print(f"  连亏熔断触发:   {m['clb_triggered']} (最大连亏={m['clb_max_losses']})")

        # 8. 成交
        fs = m.get('fill_stats', {})
        if fs:
            ba = max(fs.get('buy_attempted', 1), 1)
            sa = max(fs.get('sell_attempted', 1), 1)
            print(f"\n--- 成交率 ---")
            print(f"  买入: {fs.get('buy_filled',0)}/{fs.get('buy_attempted',0)} = "
                  f"{fs.get('buy_filled',0)/ba*100:.1f}%")
            print(f"  卖出: {fs.get('sell_filled',0)}/{fs.get('sell_attempted',0)} = "
                  f"{fs.get('sell_filled',0)/sa*100:.1f}%")
            print(f"  涨停跳过: {fs.get('buy_limit_up_skip',0)}  T+1拦截: {fs.get('sell_tplus1_blocked',0)}")

        # 警告汇总
        if self.warnings:
            print(f"\n--- ⚠ 警告 ({len(self.warnings)}条) ---")
            for w in self.warnings:
                print(f"  ⚠ {w}")
        else:
            print(f"\n--- ✓ 无警告 ---")

        print("=" * 70)

    def save(self, output_dir: str = None):
        """保存诊断数据到JSON"""
        if output_dir is None:
            strategy_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            output_dir = os.path.join(strategy_dir, 'rolling_validation_results')
        path = os.path.join(output_dir, 'backtest_diagnostics.json')

        data = {
            'factor_selection': dict(self.metrics['factor_selection']),
            'factor_family': dict(self.metrics['factor_family']),
            'buy_signals': self.metrics['buy_signals'],
            'buy_by_buy_point': dict(self.metrics['buy_by_buy_point']),
            'buy_by_factor_family': dict(self.metrics['buy_by_factor_family']),
            'ml_predictions_total': self.metrics['ml_predictions_total'],
            'ml_predictions_active': self.metrics['ml_predictions_active'],
            'ml_validation_ic': self.metrics['ml_validation_ic'],
            'alt_hits': {
                'northbound': self.metrics['alt_northbound_hits'],
                'margin': self.metrics['alt_margin_hits'],
                'dragon_tiger': self.metrics['alt_dragon_tiger_hits'],
            },
            'fill_stats': self.metrics.get('fill_stats', {}),
            'risk_events': {
                'hds_triggered': self.metrics['hds_triggered'],
                'clb_triggered': self.metrics['clb_triggered'],
                'clb_max_losses': self.metrics['clb_max_losses'],
            },
            'warnings': self.warnings,
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        print(f"诊断数据已保存: {path}")


# 全局单例
_diagnostics: BacktestDiagnostics = None

def get_diagnostics() -> BacktestDiagnostics:
    global _diagnostics
    if _diagnostics is None:
        _diagnostics = BacktestDiagnostics()
    return _diagnostics
