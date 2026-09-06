#!/usr/bin/env python
"""V-验证体系: 回测结果的统计验证 (honest Sharpe 的地基)

输入: rolling_validation_results/equity_curve.csv (由bt_execution.py每次回测落盘)
输出:
  1. 年度分解: 逐年收益/夏普/最大回撤 (稳定性)
  2. 设计期 vs 持有期分半报告: 2021-2024(设计窗口) vs 2025-2026(未用于决策的持有期)
  3. 滚动12月夏普: 最小/最大/为负窗口数 (收益是否均匀)
  4. MC置信区间: 日收益块bootstrap + 年块bootstrap, 给出夏普95%CI (路径不确定性)
  5. 回撤画像: top10回撤区间(峰/谷/修复日/深度/历时)
  6. 汇总JSON: rolling_validation_results/sharpe_validation.json

判读纪律: 持有期夏普的下界(年块bootstrap CI下限) > 1.0 才算"诚实达标"。
"""
import json
import os
import sys

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EQ_PATH = os.path.join(BASE, 'rolling_validation_results', 'equity_curve.csv')
OUT_PATH = os.path.join(BASE, 'rolling_validation_results', 'sharpe_validation.json')

ANNUAL_FACTOR = 252
DESIGN_END = '2025-01-01'   # 设计期: 2021-2024 (所有手工决策基于的窗口)
BOOT_BLOCK = 5              # 日收益块bootstrap块长(交易日)
N_BOOT = 2000


def load_equity(path=EQ_PATH):
    if not os.path.exists(path):
        print(f"[错误] 未找到 {path} — 需先跑一次带净值落盘的bt_execution.py (>= 9/5晚版本)")
        sys.exit(1)
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    df['ret'] = df['daily_ret'].astype(float)
    return df


def sharpe_of(rets):
    r = np.asarray(rets, dtype=float)
    r = r[~np.isnan(r)]
    if len(r) < 20 or np.std(r) == 0:
        return np.nan
    return float(np.mean(r) / np.std(r) * np.sqrt(ANNUAL_FACTOR))


def max_dd(nav):
    peak = np.maximum.accumulate(nav)
    dd = (nav - peak) / peak
    return float(dd.min())


def annual_table(df):
    df = df.copy()
    df['y'] = df['date'].dt.year
    rows = []
    for y, g in df.groupby('y'):
        rets = g['ret'].dropna()
        nav = g['nav']
        rows.append({
            'year': int(y),
            'return': float(nav.iloc[-1] / nav.iloc[0] - 1) if len(nav) > 1 else np.nan,
            'sharpe': sharpe_of(rets),
            'max_dd': max_dd(nav.values),
            'n_days': int(len(rets)),
        })
    return pd.DataFrame(rows)


def rolling_sharpe(df, window=252):
    s = df['ret'].dropna().rolling(window, min_periods=60).apply(
        lambda x: float(np.mean(x) / np.std(x) * np.sqrt(ANNUAL_FACTOR)) if np.std(x) > 0 else np.nan)
    s = s.dropna()
    min_date = None
    if len(s) > 0:
        min_pos = int(np.argmin(s.values))
        min_date = str(df['date'].iloc[s.index[min_pos]].date())
    return {'min': float(s.min()), 'max': float(s.max()), 'mean': float(s.mean()),
            'neg_windows': int((s < 0).sum()), 'total_windows': int(len(s)),
            'min_date': min_date}


def block_bootstrap_daily(rets, block=BOOT_BLOCK, n=N_BOOT, seed=42):
    """日收益圆形块bootstrap → 夏普分布 (保持日内自相关)"""
    rng = np.random.default_rng(seed)
    r = np.asarray(rets, dtype=float)
    r = r[~np.isnan(r)]
    n_blocks = int(np.ceil(len(r) / block))
    sharpe_dist = np.empty(n)
    for i in range(n):
        starts = rng.integers(0, len(r), size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block) % len(r) for s in starts])
        s = sharpe_of(r[idx])
        sharpe_dist[i] = s if not np.isnan(s) else -99
    sharpe_dist = sharpe_dist[sharpe_dist > -90]
    return {'ci95_low': float(np.percentile(sharpe_dist, 2.5)),
            'ci95_high': float(np.percentile(sharpe_dist, 97.5)),
            'mean': float(np.mean(sharpe_dist))}


def block_bootstrap_years(df, n=N_BOOT, seed=42):
    """按年块bootstrap整条路径 → 夏普分布 (含年度异质性, 策略级不确定性)"""
    rng = np.random.default_rng(seed)
    df = df.copy()
    df['y'] = df['date'].dt.year
    years = sorted(df['y'].unique())
    year_rets = {y: g['ret'].dropna().values for y, g in df.groupby('y')}
    sharpe_dist = np.empty(n)
    for i in range(n):
        sampled = rng.choice(years, size=len(years), replace=True)
        concat = np.concatenate([year_rets[y] for y in sampled])
        s = sharpe_of(concat)
        sharpe_dist[i] = s if not np.isnan(s) else -99
    sharpe_dist = sharpe_dist[sharpe_dist > -90]
    return {'ci95_low': float(np.percentile(sharpe_dist, 2.5)),
            'ci95_high': float(np.percentile(sharpe_dist, 97.5)),
            'mean': float(np.mean(sharpe_dist))}


def dd_episodes(df):
    """top10回撤区间: 峰/谷/修复日/深度/历时"""
    nav = df['nav'].values
    dates = df['date'].values
    peak_idx, episodes = 0, []
    for i in range(1, len(nav)):
        if np.isnan(nav[i]):
            continue
        if nav[i] >= nav[peak_idx]:
            peak_idx = i
            continue
        dd = (nav[i] - nav[peak_idx]) / nav[peak_idx]
        # 谷底: 记录峰→当前最低点, 直到创新高才闭合
        trough_idx = i + np.argmin(nav[i:])
        recovery_idx = None
        for j in range(trough_idx, len(nav)):
            if not np.isnan(nav[j]) and nav[j] >= nav[peak_idx]:
                recovery_idx = j
                break
        episodes.append({
            'peak': str(pd.Timestamp(dates[peak_idx]).date()),
            'trough': str(pd.Timestamp(dates[trough_idx]).date()),
            'recovery': str(pd.Timestamp(dates[recovery_idx]).date()) if recovery_idx else None,
            'depth': float(nav[trough_idx] / nav[peak_idx] - 1),
            'days_peak_to_trough': int(trough_idx - peak_idx),
            'days_to_recovery': int(recovery_idx - peak_idx) if recovery_idx else None,
        })
    # 去重取最深top10
    seen, out = set(), []
    for e in sorted(episodes, key=lambda x: x['depth']):
        key = (e['peak'], e['trough'])
        if key in seen:
            continue
        seen.add(key)
        out.append(e)
    return out[:10]


def main():
    df = load_equity()
    rets_all = df['ret'].dropna()
    nav_all = df['nav'].values
    total_ret = float(nav_all[-1] / nav_all[0] - 1)
    years = (df['date'].iloc[-1] - df['date'].iloc[0]).days / 365.25
    cagr = float((nav_all[-1] / nav_all[0]) ** (1 / years) - 1)

    report = {
        'period': [str(df['date'].iloc[0].date()), str(df['date'].iloc[-1].date())],
        'total_return': total_ret, 'cagr': cagr,
        'sharpe_full': sharpe_of(rets_all), 'max_dd_full': max_dd(nav_all),
        'calmar': float(cagr / abs(max_dd(nav_all))) if max_dd(nav_all) != 0 else np.nan,
    }

    print("=" * 60)
    print("V-验证体系: 夏普统计验证报告")
    print("=" * 60)
    print(f"区间: {report['period'][0]} → {report['period'][1]}  ({len(rets_all)}个交易日)")
    print(f"总收益: {total_ret*100:.2f}% | CAGR: {cagr*100:.2f}%")
    print(f"全期夏普: {report['sharpe_full']:.4f} | 最大回撤: {report['max_dd_full']*100:.2f}% | "
          f"Calmar: {report['calmar']:.3f}")

    # 1. 年度分解
    print("\n--- 年度分解 ---")
    at = annual_table(df)
    report['annual'] = at.to_dict('records')
    print(at.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # 2. 设计期 vs 持有期
    print("\n--- 设计期(2021-2024) vs 持有期(2025+) ---")
    split = {'design': {}, 'holdout': {}}
    for key, sub in [('design', df[df['date'] < DESIGN_END]),
                     ('holdout', df[df['date'] >= DESIGN_END])]:
        r = sub['ret'].dropna()
        nav = sub['nav'].values
        ret = float(nav[-1] / nav[0] - 1)
        split[key] = {'return': ret, 'sharpe': sharpe_of(r), 'max_dd': max_dd(nav)}
        print(f"  [{key}] 收益={ret*100:.2f}% 夏普={split[key]['sharpe']:.4f} "
              f"最大回撤={split[key]['max_dd']*100:.2f}%")
    report['split'] = split

    # 3. 滚动12月夏普
    print("\n--- 滚动12月夏普 ---")
    rs = rolling_sharpe(df)
    report['rolling_sharpe'] = rs
    print(f"  窗口={rs['total_windows']}个 | 最小={rs['min']:.2f} (于{rs['min_date']}) | "
          f"最大={rs['max']:.2f} | 均值={rs['mean']:.2f} | 为负={rs['neg_windows']}个")

    # 4. MC置信区间
    print(f"\n--- MC bootstrap 夏普95%CI (N={N_BOOT}) ---")
    bd = block_bootstrap_daily(rets_all.values)
    by = block_bootstrap_years(df)
    report['bootstrap_daily'] = bd
    report['bootstrap_yearly'] = by
    print(f"  日收益块bootstrap: 95%CI=[{bd['ci95_low']:.3f}, {bd['ci95_high']:.3f}] "
          f"(均值{bd['mean']:.3f})")
    print(f"  年块bootstrap:     95%CI=[{by['ci95_low']:.3f}, {by['ci95_high']:.3f}] "
          f"(均值{by['mean']:.3f})")
    print(f"  → 诚实达标判定: 持有期夏普={split['holdout']['sharpe']:.3f}, "
          f"年块CI下限={by['ci95_low']:.3f} "
          f"{'✅ 双>1.0' if split['holdout']['sharpe'] > 1.0 and by['ci95_low'] > 1.0 else '⚠️ 未达诚实标准'}")

    # 5. 回撤画像
    print("\n--- top10回撤区间 ---")
    eps = dd_episodes(df)
    report['dd_episodes'] = eps
    for e in eps:
        rec = e['recovery'] if e['recovery'] else '未修复'
        print(f"  峰{e['peak']}→谷{e['trough']} 深度{e['depth']*100:.1f}% "
              f"历时{e['days_peak_to_trough']}日→修复{rec}(+{e['days_to_recovery']}日)")

    with open(OUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n汇总已保存: {OUT_PATH}")


if __name__ == '__main__':
    main()
