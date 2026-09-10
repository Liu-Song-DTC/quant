#!/usr/bin/env python3
"""2026-09-10 阶段4扫查: factor_df中ML看不见的非白名单原始列 → 月度截面IC
白名单60个名字(factor_config.yaml ml.feature_whitelist); factor_df 131列 =
meta(3) + future_ret + 64原始 + 62_rank + ln_cap/concept_heat。
rank列与z-score截面等价, 跳过; 目标 = 原始列中不在白名单的部分。
测度: 逐月(面板每月首个日期)截面Spearman vs future_ret(fwd10), 2021-2026。
"""
import yaml
import numpy as np
import pandas as pd

PKL = '/mnt/d/quant/strategy/cache/factor_df_2757s_808d_3851bd96.parquet'
YAML = '/mnt/d/quant/strategy/config/factor_config.yaml'


def main():
    cfg = yaml.safe_load(open(YAML))
    wl = set(cfg['ml']['feature_whitelist'])
    print(f'白名单声明: {len(wl)} 个')
    df = pd.read_parquet(PKL)
    print(f'面板: {len(df)} 行 x {df.shape[1]} 列')
    df['date'] = pd.to_datetime(df['date'])
    print(f'日期范围: {df.date.min()} ~ {df.date.max()}, 唯一日期 {df.date.nunique()}, '
          f'股票 {df.code.nunique()} 只, 每日期均n={len(df)/df.date.nunique():.0f}')
    df['month'] = df['date'].dt.to_period('M')
    first_date = df.groupby('month')['date'].min()
    sub = df[df['date'].isin(first_date.values)]
    print(f'月度截面: {len(sub)} 行 x {sub.month.nunique()} 月')

    meta = {'code', 'date', 'industry', 'future_ret', 'month'}
    raw = [c for c in df.columns if c not in meta and not c.endswith('_rank')]
    present = sorted(set(raw) & wl)
    invisible = sorted(set(raw) - wl)
    print(f'原始列: {len(raw)} = 白名单可见 {len(present)} + ML看不见 {len(invisible)}')
    print(f'白名单声明但parquet无: {sorted(wl - set(raw))}')
    print(f'\nML看不见的原始列({len(invisible)}): {invisible}')

    res = {}
    for month, g in sub.groupby('month'):
        g = g[['future_ret'] + raw].dropna(subset=['future_ret'])
        if len(g) < 50:
            continue
        ic = g[raw].corrwith(g['future_ret'], method='spearman')
        for c, v in ic.items():
            if pd.notna(v):
                res.setdefault(c, []).append((month, v))
    print('\n[IC扫查] 按|均值IC|排序 (月度截面 vs fwd10):')
    rows = []
    for c, lst in res.items():
        s = pd.Series([v for _, v in lst], index=[m for m, _ in lst])
        if len(s) < 6:
            continue
        ir = s.mean() / s.std() if s.std() > 0 else 0
        rows.append((c, s.mean(), ir, len(s), 100 * (s.abs() > 0.05).mean()))
    rows.sort(key=lambda x: -abs(x[1]))
    for c, m, ir, n, hit in rows:
        star = ' <<<不可见' if c in invisible else ''
        print(f'  {c:28s} ic={m:+.4f} IR={ir:+.2f} |IC|>5%={hit:3.0f}% 月数={n}{star}')
    # 不可见列前5的逐年
    inv_top = [c for c, m, ir, n, hit in rows if c in invisible][:5]
    for c in inv_top:
        s = pd.Series([v for _, v in res[c]], index=[m for m, _ in res[c]])
        yr = s.groupby(s.index.year).mean()
        print(f'  [{c}] 逐年: ' + '  '.join(f'{i}:{v:+.3f}' for i, v in yr.items()))


if __name__ == '__main__':
    main()
