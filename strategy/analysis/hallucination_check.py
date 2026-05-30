#!/usr/bin/env python3
"""
幻觉与噪音检测 — 针对 v8 趋势捕捉规则变更
输出: strategy/analysis_results/hallucination_check_YYYYMMDD.md

检测维度:
  1. MTF折扣前后IC对比 (pre_discount_score vs score)
  2. 新增字段单因子IC扫描
  3. 分层准确率: MTF对齐/周线多头/缠论信号
  4. Regime切换频率与分层准确率
  5. 熊市风险信号分层
"""

import pandas as pd
import numpy as np
import os, sys, warnings
from datetime import datetime

warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT_DIR = os.path.join(BASE, 'rolling_validation_results')
OUT_DIR = os.path.join(BASE, 'analysis_results')
os.makedirs(OUT_DIR, exist_ok=True)

TODAY = datetime.now().strftime('%Y%m%d')
OUT_FILE = os.path.join(OUT_DIR, f'hallucination_check_{TODAY}.md')

# ── Load & Merge ─────────────────────────────────────────
print("加载数据...")
sig = pd.read_csv(os.path.join(RESULT_DIR, 'backtest_signals.csv'),
                  low_memory=False)
val = pd.read_csv(os.path.join(RESULT_DIR, 'validation_results.csv'),
                  low_memory=False)

sig['date'] = pd.to_datetime(sig['date'])
val['date'] = pd.to_datetime(val['date'])

# Ensure code is consistent string type
sig['code'] = sig['code'].astype(str).str.zfill(6)
val['code'] = val['code'].astype(str).str.zfill(6)

print(f"backtest_signals: {len(sig):,} rows")
print(f"validation_results: {len(val):,} rows")

# Merge future_ret from val into sig
merge_cols = ['date', 'code']
df = sig.merge(val[merge_cols + ['future_ret']], on=merge_cols, how='inner')
print(f"merged (with future_ret): {len(df):,} rows")
print(f"日期范围: {df['date'].min()} ~ {df['date'].max()}")

# Helper
def spearman_ic(a, b):
    """Safe Spearman IC"""
    mask = a.notna() & b.notna()
    if mask.sum() < 30:
        return np.nan, np.nan
    from scipy.stats import spearmanr
    r, p = spearmanr(a[mask], b[mask])
    return r, p

def stars(ic, p):
    if np.isnan(ic):
        return '?'
    if abs(ic) >= 0.05 and p < 0.01:
        return '★★★ 强有效'
    if abs(ic) >= 0.03 and p < 0.01:
        return '★★ 有效'
    if abs(ic) >= 0.01 and p < 0.05:
        return '★ 弱有效'
    if abs(ic) < 0.01 and p > 0.05:
        return '✗ 幻觉'
    if abs(ic) < 0.01:
        return '⚠ 噪音(IC≈0)'
    return '⚠ 边缘'

def w(s=''):
    with open(OUT_FILE, 'a') as f:
        f.write(s + '\n')

# Clear output file
with open(OUT_FILE, 'w') as f:
    f.write(f'# 幻觉与噪音检测报告 — {TODAY}\n\n')
    f.write(f'数据: {len(df):,} 条, {df["date"].min().date()} ~ {df["date"].max().date()}\n\n')

# ════════════════════════════════════════════════════════
# DIMENSION 1: pre_discount_score vs score IC
# ════════════════════════════════════════════════════════
print('\n=== 维度1: MTF折扣前后IC对比 ===')
w('## 一、MTF折扣前后IC对比\n')

if 'pre_discount_score' in df.columns and 'score' in df.columns:
    ic_pre, p_pre = spearman_ic(df['pre_discount_score'], df['future_ret'])
    ic_post, p_post = spearman_ic(df['score'], df['future_ret'])
    delta = ic_post - ic_pre

    w(f'| 指标 | IC | p-value | 判定 |')
    w(f'|------|----|---------|------|')
    w(f'| pre_discount_score (折扣前) | {ic_pre:.4f} ({ic_pre*100:.2f}%) | {p_pre:.4f} | {stars(ic_pre, p_pre)} |')
    w(f'| score (折扣后) | {ic_post:.4f} ({ic_post*100:.2f}%) | {p_post:.4f} | {stars(ic_post, p_post)} |')
    w(f'| **MTF贡献 Δ** | **{delta:+.4f}** | — | {"✓ MTF有效增强" if delta > 0.003 else ("⚠ MTF无显著影响" if abs(delta) < 0.003 else "✗ MTF在破坏信号!")} |')

    print(f'  IC(pre)={ic_pre:.4f}  →  IC(post)={ic_post:.4f}  Δ={delta:+.4f}')

    # By industry
    w('\n### 分行业IC对比\n')
    w('| 行业 | IC(折扣前) | IC(折扣后) | Δ | 判定 |')
    w('|------|-----------|-----------|----|------|')
    for ind in sorted(df['industry'].dropna().unique()):
        sub = df[df['industry'] == ind]
        ic1, _ = spearman_ic(sub['pre_discount_score'], sub['future_ret'])
        ic2, _ = spearman_ic(sub['score'], sub['future_ret'])
        d = ic2 - ic1
        flag = '✓' if d > 0.003 else ('—' if abs(d) < 0.003 else '✗')
        if not np.isnan(ic1) and not np.isnan(ic2):
            w(f'| {ind} | {ic1:.4f} | {ic2:.4f} | {d:+.4f} | {flag} |')
            if flag == '✗':
                print(f'  ✗ {ind}: IC {ic1:.4f} → {ic2:.4f} (MTF搞砸了)')
else:
    w('⚠ pre_discount_score 或 score 字段缺失，跳过\n')
    print('  ⚠ 缺少关键字段')

# ════════════════════════════════════════════════════════
# DIMENSION 2: New field IC scan
# ════════════════════════════════════════════════════════
print('\n=== 维度2: 新增字段IC扫描 ===')
w('\n## 二、新增字段单因子IC扫描\n')

new_fields = [c for c in df.columns if c not in {
    'date', 'code', 'buy', 'sell', 'industry', 'factor_name',
    'factor_value', 'weight', 'future_ret', 'score'
}]

w(f'共 {len(new_fields)} 个候选新字段\n')
w('| 字段 | IC | p-value | |IC| | 判定 | n |')
w('|------|----|---------|-----|------|---|')

hallucination_fields = []
weak_fields = []
good_fields = []

for f in sorted(new_fields):
    if df[f].dtype == 'object':
        continue
    ic, p = spearman_ic(df[f], df['future_ret'])
    if np.isnan(ic):
        continue
    n_valid = df[f].notna().sum()
    rating = stars(ic, p)
    w(f'| {f} | {ic:.4f} | {p:.4f} | {abs(ic)*100:.1f}% | {rating} | {n_valid:,} |')

    if '幻觉' in rating:
        hallucination_fields.append((f, ic, p))
    elif '弱' in rating or '噪音' in rating:
        weak_fields.append((f, ic, p))
    else:
        good_fields.append((f, ic, p))

w(f'\n**总结:** 幻觉{len(hallucination_fields)}个 / 弱信号{len(weak_fields)}个 / 有效{len(good_fields)}个')

if hallucination_fields:
    w('\n### 🚨 幻觉字段 (IC≈0 且 p>0.05, 建议移除或降低权重)\n')
    for f, ic, p in hallucination_fields:
        w(f'- `{f}`: IC={ic:.4f}, 参与信号计算但无预测力 → **建议从score公式中移除**')
        print(f'  🚨 {f}: IC={ic:.4f} p={p:.3f}')

if weak_fields:
    w('\n### ⚠ 弱信号字段 (边际贡献, 考虑收紧使用条件)\n')
    for f, ic, p in weak_fields:
        w(f'- `{f}`: IC={ic:.4f}')

if good_fields:
    w('\n### ✓ 有效字段\n')
    for f, ic, p in good_fields:
        w(f'- `{f}`: IC={ic:.4f}')

# ════════════════════════════════════════════════════════
# DIMENSION 3: Stratified accuracy
# ════════════════════════════════════════════════════════
print('\n=== 维度3: 分层准确率 ===')
w('\n## 三、分层准确率分析\n')

buy_df = df[df['buy'] == True].dropna(subset=['future_ret'])
print(f'买入信号总数: {len(buy_df):,}')

# 3.1 MTF aligned vs not
w('\n### 3.1 MTF对齐分层 (mtf_alignment_score)\n')
if 'mtf_alignment_score' in df.columns:
    w('| 分组 | n | 买入准确率 | 平均future_ret |')
    w('|------|---|-----------|---------------|')
    for label, cond in [('对齐(mtf>0)', buy_df['mtf_alignment_score'] > 0),
                          ('不对齐(mtf≤0)', buy_df['mtf_alignment_score'] <= 0),
                          ('强对齐(mtf>0.3)', buy_df['mtf_alignment_score'] > 0.3),
                          ('强不对齐(mtf<-0.3)', buy_df['mtf_alignment_score'] < -0.3)]:
        sub = buy_df[cond].dropna(subset=['future_ret'])
        if len(sub) > 30:
            acc = (sub['future_ret'] > 0).mean()
            avg = sub['future_ret'].mean()
            w(f'| {label} | {len(sub):,} | {acc:.2%} | {avg*100:.2f}% |')
            print(f'  {label}: acc={acc:.2%} avg={avg*100:.2f}% n={len(sub):,}')

# 3.2 weekly_trend_up
w('\n### 3.2 周线多头分层\n')
if 'avg_trend_strength' in df.columns:
    w('| 分组 | n | 买入准确率 | 平均future_ret |')
    w('|------|---|-----------|---------------|')
    for label, cond in [('趋势强(>0.3)', buy_df['avg_trend_strength'] > 0.3),
                          ('趋势中(0~0.3)', (buy_df['avg_trend_strength'] > 0) & (buy_df['avg_trend_strength'] <= 0.3)),
                          ('趋势弱(≤0)', buy_df['avg_trend_strength'] <= 0)]:
        sub = buy_df[cond].dropna(subset=['future_ret'])
        if len(sub) > 30:
            acc = (sub['future_ret'] > 0).mean()
            avg = sub['future_ret'].mean()
            w(f'| {label} | {len(sub):,} | {acc:.2%} | {avg*100:.2f}% |')
            print(f'  {label}: acc={acc:.2%} avg={avg*100:.2f}% n={len(sub):,}')

# 3.3 chan_buy_point
w('\n### 3.3 缠论买点分层\n')
if 'chan_buy_point' in df.columns:
    w('| 分组 | n | 买入准确率 | 平均future_ret |')
    w('|------|---|-----------|---------------|')
    for label, cond in [('缠论买点触发', buy_df['chan_buy_point'] == True),
                          ('无缠论买点', buy_df['chan_buy_point'] != True)]:
        sub = buy_df[cond].dropna(subset=['future_ret'])
        if len(sub) > 30:
            acc = (sub['future_ret'] > 0).mean()
            avg = sub['future_ret'].mean()
            w(f'| {label} | {len(sub):,} | {acc:.2%} | {avg*100:.2f}% |')
            print(f'  {label}: acc={acc:.2%} avg={avg*100:.2f}% n={len(sub):,}')

# 3.4 chan_divergence_type
if 'chan_divergence_type' in df.columns:
    w('\n### 3.4 缠论背离类型分层\n')
    w('| 背离类型 | n | 买入准确率 | 平均future_ret |')
    w('|---------|---|-----------|---------------|')
    for dtype in sorted(buy_df['chan_divergence_type'].dropna().unique()):
        sub = buy_df[buy_df['chan_divergence_type'] == dtype].dropna(subset=['future_ret'])
        if len(sub) > 20:
            acc = (sub['future_ret'] > 0).mean()
            avg = sub['future_ret'].mean()
            w(f'| {dtype} | {len(sub):,} | {acc:.2%} | {avg*100:.2f}% |')
            print(f'  {dtype}: acc={acc:.2%} avg={avg*100:.2f}% n={len(sub):,}')

# 3.5 MTF discount factor bins
w('\n### 3.5 MTF折扣因子分档\n')
if 'mtf_discount_factor' in df.columns:
    w('| 折扣因子区间 | n | 买入准确率 | 平均future_ret |')
    w('|-------------|---|-----------|---------------|')
    bins = [(0, 0.5, '重折扣[0~0.5)'), (0.5, 0.7, '中折扣[0.5~0.7)'),
            (0.7, 0.9, '轻折扣[0.7~0.9)'), (0.9, 1.0, '无折扣[0.9~1.0]'),
            (1.0, 1.2, '溢价[1.0~1.2]')]
    for lo, hi, label in bins:
        sub = buy_df[(buy_df['mtf_discount_factor'] >= lo) & (buy_df['mtf_discount_factor'] < hi)]
        sub = sub.dropna(subset=['future_ret'])
        if len(sub) > 30:
            acc = (sub['future_ret'] > 0).mean()
            avg = sub['future_ret'].mean()
            w(f'| {label} | {len(sub):,} | {acc:.2%} | {avg*100:.2f}% |')
            print(f'  {label}: acc={acc:.2%} avg={avg*100:.2f}% n={len(sub):,}')

# ════════════════════════════════════════════════════════
# DIMENSION 4: Time decay of IC
# ════════════════════════════════════════════════════════
print('\n=== 维度4: 新字段IC时序稳定性 ===')
w('\n## 四、新字段IC时序稳定性\n')

df['year_month'] = df['date'].dt.to_period('M')

# pre_discount vs score IC monthly
if 'pre_discount_score' in df.columns:
    w('\n### 4.1 折扣前后月度IC对比\n')
    w('| 月份 | IC(折扣前) | IC(折扣后) | Δ | 判定 |')
    w('|------|-----------|-----------|----|------|')
    better_months = 0
    worse_months = 0
    for month in sorted(df['year_month'].unique())[-12:]:
        sub = df[df['year_month'] == month]
        ic1, _ = spearman_ic(sub['pre_discount_score'], sub['future_ret'])
        ic2, _ = spearman_ic(sub['score'], sub['future_ret'])
        if not np.isnan(ic1) and not np.isnan(ic2):
            d = ic2 - ic1
            flag = '✓' if d > 0 else '✗'
            if d > 0:
                better_months += 1
            else:
                worse_months += 1
            w(f'| {month} | {ic1:.4f} | {ic2:.4f} | {d:+.4f} | {flag} |')

    w(f'\nMTF折扣后改善的月份: {better_months}/{better_months+worse_months}')
    if worse_months > better_months:
        w('\n⚠ MTF折扣在多数月份降低了IC → 趋势规则整体为噪音')
        print('  ⚠ MTF在多数月份降低了IC!')

# ════════════════════════════════════════════════════════
# DIMENSION 5: Signal density / turnover impact
# ════════════════════════════════════════════════════════
print('\n=== 维度5: 信号密度与换手 ===')
w('\n## 五、信号密度与换手影响\n')

# Buy signal rate per day
daily_buy_rate = df.groupby('date')['buy'].mean()
w(f'- 日均买入信号比例: {daily_buy_rate.mean()*100:.1f}%')
w(f'- 买入比例标准差: {daily_buy_rate.std()*100:.1f}%')
w(f'- 最大单日买入比例: {daily_buy_rate.max()*100:.1f}%')
w(f'- 最小单日买入比例: {daily_buy_rate.min()*100:.1f}%')

# Score distribution change
if 'pre_discount_score' in df.columns:
    score_std_pre = df['pre_discount_score'].std()
    score_std_post = df['score'].std()
    w(f'\n- 折扣前score标准差: {score_std_pre:.4f}')
    w(f'- 折扣后score标准差: {score_std_post:.4f}')
    if score_std_post < score_std_pre * 0.9:
        w('  ⚠ MTF折扣大幅压缩了分数方差 → 信号区分度下降')

# ════════════════════════════════════════════════════════
# SUMMARY
# ════════════════════════════════════════════════════════
print('\n=== 总结 ===')
w('\n## 六、结论与建议\n')

issues = []

if 'pre_discount_score' in df.columns and 'score' in df.columns:
    ic_pre, _ = spearman_ic(df['pre_discount_score'], df['future_ret'])
    ic_post, _ = spearman_ic(df['score'], df['future_ret'])
    if not np.isnan(ic_pre) and not np.isnan(ic_post):
        if ic_post < ic_pre - 0.003:
            issues.append(f'🔴 MTF折扣降低了整体IC ({ic_pre*100:.2f}%→{ic_post*100:.2f}%), 趋势规则存在幻觉, 建议回滚或大幅收紧')
        elif abs(ic_post - ic_pre) < 0.003:
            issues.append(f'🟡 MTF折扣未显著改变IC ({ic_pre*100:.2f}%→{ic_post*100:.2f}%), 趋势规则为噪音, 建议简化')
        else:
            issues.append(f'🟢 MTF折扣提升了整体IC ({ic_pre*100:.2f}%→{ic_post*100:.2f}%), 趋势规则有效')

if hallucination_fields:
    names = ', '.join(f'`{f}`' for f, _, _ in hallucination_fields)
    issues.append(f'🔴 {len(hallucination_fields)}个幻觉字段需移除或降权: {names}')

if weak_fields:
    names = ', '.join(f'`{f}`' for f, _, _ in weak_fields[:3])
    issues.append(f'🟡 {len(weak_fields)}个弱信号字段建议收紧: {names}...')

for issue in issues:
    w(f'- {issue}')
    print(f'  {issue}')

if not issues:
    w('✅ 未检测到明显的幻觉或噪音问题')

w(f'\n---\n*报告生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*')

print(f'\n报告已保存: {OUT_FILE}')
