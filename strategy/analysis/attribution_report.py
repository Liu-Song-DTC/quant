# analysis/attribution_report.py
"""
每笔交易归因报告。

从 backtest_signals.csv + portfolio_selections.csv 重构每笔交易的完整生命周期：
入场买点/因子/行业 → 出场原因 → 持仓天数 → 归因分解。

用法: python analysis/attribution_report.py
"""

import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
from collections import defaultdict


class AttributionReport:
    """重构交易并输出结构化归因报告"""

    def __init__(self, data_dir: str = None):
        if data_dir is None:
            strategy_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            data_dir = os.path.join(strategy_dir, 'rolling_validation_results')
        self.data_dir = Path(data_dir)
        self.signals_path = self.data_dir / 'backtest_signals.csv'
        self.selections_path = self.data_dir / 'portfolio_selections.csv'
        self.trades = None

    def reconstruct_trades(self) -> pd.DataFrame:
        """从选股结果重构每笔交易"""
        if not self.selections_path.exists():
            print(f"选股文件不存在: {self.selections_path}")
            return pd.DataFrame()

        sel = pd.read_csv(self.selections_path, parse_dates=['date'])
        if sel.empty:
            return pd.DataFrame()

        sig = pd.read_csv(self.signals_path, parse_dates=['date'])

        trades = []
        for code, group in sel[sel['weight'] > 0].groupby('code'):
            group = group.sort_values('date')
            entry = None
            for _, row in group.iterrows():
                if entry is None:
                    entry = row
                else:
                    # 获取入场时信号详情
                    entry_sig = sig[(sig['code'] == code) & (sig['date'] == entry['date'])]
                    exit_sig = sig[(sig['code'] == code) & (sig['date'] == row['date'])]

                    trade = {
                        'code': code,
                        'entry_date': str(entry['date'].date()) if hasattr(entry['date'], 'date') else str(entry['date']),
                        'exit_date': str(row['date'].date()) if hasattr(row['date'], 'date') else str(row['date']),
                        'entry_weight': entry['weight'],
                        'entry_score': entry['score'],
                        'entry_industry': entry.get('industry', ''),
                        'entry_buy_point': int(entry_sig['chan_buy_point'].values[0]) if len(entry_sig) > 0 else 0,
                        'entry_signal_level': int(entry_sig['signal_level'].values[0]) if len(entry_sig) > 0 else 0,
                        'entry_factor': str(entry_sig['factor_name'].values[0]) if len(entry_sig) > 0 else '',
                        'entry_trend_type': int(entry_sig['trend_type'].values[0]) if len(entry_sig) > 0 else 0,
                        'exit_sell_point': int(exit_sig['chan_sell_point'].values[0]) if len(exit_sig) > 0 else 0,
                        'exit_divergence': str(exit_sig['chan_divergence_type'].values[0]) if len(exit_sig) > 0 else '',
                        'exit_signal_level': int(exit_sig['signal_level'].values[0]) if len(exit_sig) > 0 else 0,
                    }
                    trades.append(trade)
                    entry = row

        self.trades = pd.DataFrame(trades)
        return self.trades

    def generate_summary(self) -> dict:
        """生成归因摘要"""
        if self.trades is None:
            self.reconstruct_trades()
        if self.trades is None or len(self.trades) == 0:
            return {'total_trades': 0}

        df = self.trades

        # 出境原因分类
        def classify_exit(row):
            if row['exit_sell_point'] >= 3:
                return 'S3_强制清仓'
            elif row['exit_sell_point'] == 2:
                return 'S2_减仓'
            elif row['exit_sell_point'] == 1:
                return 'S1_减仓'
            elif row['exit_signal_level'] <= -2:
                return '双级别卖出'
            elif 'top' in str(row['exit_divergence']).lower():
                return '顶背离退出'
            else:
                return '调仓轮出'

        df['exit_category'] = df.apply(classify_exit, axis=1)

        # 按买点聚合
        by_buy_point = {}
        for bp, grp in df.groupby('entry_buy_point'):
            by_buy_point[int(bp)] = {
                'count': len(grp),
                'avg_weight': round(grp['entry_weight'].mean(), 4),
                'avg_score': round(grp['entry_score'].mean(), 4),
            }

        # 按出场原因聚合
        by_exit = df['exit_category'].value_counts().to_dict()

        # 按入场因子聚合 (Top 10)
        by_factor = df['entry_factor'].value_counts().head(10).to_dict()

        # 按行业聚合
        by_industry = df['entry_industry'].value_counts().head(10).to_dict()

        return {
            'total_trades': len(df),
            'unique_codes': df['code'].nunique(),
            'by_buy_point': by_buy_point,
            'by_exit_reason': by_exit,
            'by_factor': by_factor,
            'by_industry': by_industry,
        }

    def save(self, output_dir: str = None):
        """保存交易CSV和归因摘要JSON"""
        if output_dir is None:
            output_dir = self.data_dir

        if self.trades is None:
            self.reconstruct_trades()

        # 保存每笔交易CSV
        csv_path = os.path.join(output_dir, 'trade_attribution.csv')
        self.trades.to_csv(csv_path, index=False)
        print(f"交易归因: {len(self.trades)} 笔 -> {csv_path}")

        # 保存摘要JSON
        summary = self.generate_summary()
        json_path = os.path.join(output_dir, 'attribution_summary.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
        print(f"归因摘要: {json_path}")
        self._print_summary(summary)

    def _print_summary(self, s: dict):
        """打印归因摘要到控制台"""
        print("\n" + "=" * 50)
        print(f"交易归因报告: {s.get('total_trades', 0)} 笔交易")
        print("=" * 50)

        by_bp = s.get('by_buy_point', {})
        if by_bp:
            print("\n按买点类型:")
            for bp in sorted(by_bp.keys()):
                info = by_bp[bp]
                bp_name = {0: '无结构', 1: 'B1一买', 2: 'B2二买', 3: 'B3三买', 4: 'B4回调', 5: 'B5启动'}.get(bp, f'BP{bp}')
                print(f"  {bp_name}: {info['count']}笔  avg_score={info['avg_score']:.3f}")

        by_exit = s.get('by_exit_reason', {})
        if by_exit:
            print("\n按出场原因:")
            for reason, count in sorted(by_exit.items(), key=lambda x: -x[1]):
                print(f"  {reason}: {count}笔")

        by_factor = s.get('by_factor', {})
        if by_factor:
            print("\n按入场因子 (Top 5):")
            for fn, count in list(by_factor.items())[:5]:
                print(f"  {fn}: {count}笔")


if __name__ == '__main__':
    report = AttributionReport()
    report.reconstruct_trades()
    report.save()
