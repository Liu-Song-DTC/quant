"""特征可用时点审计 (2026-10-28, 交接计划 §3 任务 3)。

回答: ML 实际用了哪些特征? 每个特征的**真实可用时点**是什么? 历史上缺失多少?
缺失如何被表示? 配置里声明但管线从不产出的特征有哪些?

只读: 读封存因子缓存 + 已提交源码 + 配置; 不写生产产物。

用法:
  python strategy/analysis/probe_feature_availability_1028.py \
      --factor-parquet <factor_df_*.parquet> [--out docs/feature_availability_audit_1028.csv]
"""
import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path

import pandas as pd
import yaml

PROJECT = Path(__file__).resolve().parents[2]
CORE = PROJECT / 'strategy' / 'core'

# 特征 → (数据类别, 真实可用时点, 缺失表示)
# 数据类别: 量价 / 基本面 / 量价推断 / 缺列
# 可用时点写的是"该特征在 T 日的取值何时才真实可知" (决定 T 日收盘后能否用于 T+1 开盘)
#
# ⚠ 分类只对**实际进入模型**的特征有意义; 配置声明但管线不产出的 (缺列) 一律归"缺列"。
FUNDAMENTAL_FEATURES = {
    'fund_roe': '净资产收益率(报告期) ⇒ 数据可用日 = 披露截止日',
    'fund_gross_margin': '销售毛利率(报告期) ⇒ 数据可用日 = 披露截止日',
    'fund_cf_to_profit': '经营现金流/净利润(报告期) ⇒ 数据可用日 = 披露截止日',
    'tech_fund_combo': '技术×基本面组合(基本面部分同上报期口径)',
}
# 量价**推断** (不是真实观测): 名字像"资金/盈利", 实为量价公式
INFERRED_PRICE_FEATURES = {
    'smart_money_flow': '**量价推断**(capital_flow.py: 收益×波动率推断主力意图), 非真实资金流观测',
    'earnings_quality': '**量价代理**(factor_calculator.py: inv_turnover × 1/(1+|vol20|·10)), 非财报盈利质量',
}
# 开盘即可知 (其余量价都是收盘后可知)
OPEN_KNOWN = {'gap_ratio', 'intraday_ret'}

# 线性/组合评分消费的因子 (compute_composite_factors 产出)
LINEAR_FACTORS = {
    'trend_mom_v41', 'trend_mom_v24', 'trend_mom_v46', 'mom_x_lowvol_20_20',
    'mom_x_lowvol_20_10', 'mom_x_lowvol_10_10', 'mom_x_lowvol_10_20',
    'mom_diff_5_20', 'mom_diff_10_20', 'rsi_factor', 'volatility', 'volume_ratio',
    'fund_score', 'fund_roe', 'fund_profit_growth', 'fund_revenue_growth',
    'fund_gross_margin', 'fund_cf_to_profit', 'ln_cap', 'concept_heat',
}


def classify(name, available):
    """→ (数据类别, 真实可用时点, 缺失表示)"""
    if not available:
        return ('缺列', '—',
                '管线不产出 ⇒ 训练时剔除 / 预测时零填充(ml_predictor.py:505-510)')
    if name in FUNDAMENTAL_FEATURES:
        return ('基本面', FUNDAMENTAL_FEATURES[name],
                '缺失 ⇒ 因子层 NaN ⇒ 截面填充')
    if name in INFERRED_PRICE_FEATURES:
        return ('量价推断', INFERRED_PRICE_FEATURES[name],
                '无底层数据缺失 ⇒ 恒有值')
    if name in OPEN_KNOWN:
        return ('量价', 'T 日开盘/日内 ⇒ T+1 开盘前已知', '窗口不足 ⇒ NaN ⇒ 零填充')
    return ('量价', 'T 日收盘 ⇒ T+1 开盘可用', '窗口不足 ⇒ NaN ⇒ 零填充')


def find_definition(name):
    """在已提交源码里找该特征的定义位置 (core/*.py)。"""
    pat = re.compile(rf"['\"]{re.escape(name)}['\"]")
    hits = []
    for f in sorted(CORE.glob('*.py')):
        try:
            for i, line in enumerate(f.read_text(encoding='utf-8', errors='ignore').splitlines(), 1):
                if pat.search(line):
                    hits.append(f'{f.relative_to(PROJECT)}:{i}')
        except OSError:
            continue
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--factor-parquet', default='')
    ap.add_argument('--out', default=str(PROJECT / 'docs' / 'feature_availability_audit_1028.csv'))
    args = ap.parse_args()

    cfg = yaml.safe_load((PROJECT / 'strategy/config/factor_config.yaml').read_text(encoding='utf-8'))
    whitelist = cfg.get('ml', {}).get('feature_whitelist', []) or []

    pq = args.factor_parquet
    if not pq:
        cands = sorted(glob.glob(str(PROJECT / '.review_runs/*/strategy/cache/factor_df_*.parquet')))
        pq = cands[-1] if cands else ''
    df = pd.read_parquet(pq) if pq else None
    cols = set(df.columns) if df is not None else set()

    rows = []
    for name in whitelist:
        # 因子表里的原始列, 或中性化后的 _neu 列
        raw = name if name in cols else None
        neu = f'{name}_neu' if f'{name}_neu' in cols else None
        base = raw or neu
        miss = ''
        miss_by_year = ''
        if df is not None and base:
            s = df[base]
            miss = f'{float(s.isna().mean()):.4f}'
            yr = df.assign(_m=s.isna()).groupby(pd.to_datetime(df['date']).dt.year)['_m'].mean()
            miss_by_year = ';'.join(f'{y}:{v:.3f}' for y, v in yr.items())
        cat, when, policy = classify(name, bool(base))
        defs = find_definition(name)
        rows.append({
            'feature': name,
            'status': '进入ML' if base else '缺列(零填充/剔除)',
            'factor_column': base or '',
            'data_class': cat,
            'availability_timepoint': when,
            'missing_policy': policy,
            'missing_ratio_overall': miss,
            'missing_ratio_by_year': miss_by_year,
            'in_linear_score': 'Y' if name in LINEAR_FACTORS else 'N',
            'definition_sites': ';'.join(defs[:4]),
        })

    out = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False, encoding='utf-8')

    n_in = int((out['status'] == '进入ML').sum())
    n_absent = len(out) - n_in
    print(f'因子表: {pq or "(未找到)"}')
    print(f'配置白名单 {len(out)} 项 ⇒ 进入ML {n_in} 项, 缺列 {n_absent} 项')
    print(f'清单: {args.out}')
    if df is not None:
        for cat in out['data_class'].unique():
            sub = out[out['data_class'] == cat]
            if (sub['status'] == '进入ML').any():
                vals = sub.loc[sub['status'] == '进入ML', 'missing_ratio_overall'].astype(float)
                print(f'  {cat}: {len(sub)} 项(进入 {int((sub["status"]=="进入ML").sum())}), '
                      f'缺失率中位 {vals.median():.4f}, 最大 {vals.max():.4f}')
    print(f'线性/组合评分另消费: {int((out["in_linear_score"]=="Y").sum())} 项')
    return 0


if __name__ == '__main__':
    sys.exit(main())
