"""季度权重覆盖探针 (2026-09-22, 只读, 零生产写入)

背景: yaml×census对账发现固定模式评分路径P0在 中性/牛市 regime 读的是
**全局** industry_factors (factor_config.yaml, 2026-03-20后未实质更新, 377概念),
而季度标定文件 (401概念, 每季滚动重标定) 只经两条路进入评分:
  (a) 熊/弱趋势分支 (bear_factors)
  (b) P1兜底 (仅当全局未命中: 36个仅季度概念+新概念)
即: 365个交集概念的 中性/牛市 评分跑在6个月前旧权重上。
验证CSV初步信号(混杂regime效应, 非因果): 仅全局路径buys mean_fwd 0.0061
vs 仅季度 0.0129, 2026年 −1.14% vs +4.18%。

本探针: 在factor_df缓存(生产评分同源rank值)上离线重算 — 对每个
(date, 概念) 截面计算 score_g(全局权重)/score_q(季度权重)/score_qb(季度熊),
逐日Spearman IC vs future_ret, 分年汇总+配对差。
裁决: score_q系统性>score_g → "P0中性/牛市改读季度权重"正期望; 反之关闭。
概念解析复刻生产: STOCK_CONCEPT_MAP(风格过滤) + CONCEPT_INCEPTION(PIT gate)
+ 季度配置成员资格。
"""
import os
import pickle
import sys

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr

sys.path.insert(0, '/mnt/d/quant/strategy')

CACHE_F = '/mnt/d/quant/strategy/cache/factor_df_5190s_811d_85f064b9.parquet'
QD = '/mnt/d/quant/strategy/config/quarterly_factors'
GLOBAL_Y = '/mnt/d/quant/strategy/config/factor_config.yaml'
MAP_P = '/mnt/d/quant/data/stock_concept_map.pkl'
INCEP_P = '/mnt/d/quant/data/concept_inception.pkl'

STYLE_KW = ['融资融券', '深股通', '沪股通', '富时罗素', '标准普尔', 'MSCI',
            '创业板综', '机构重仓', 'QFII', '破增发', '破发股', '昨日高',
            '中证500', '深成500', '中盘股', '小盘股', '央国企改革',
            '西部大开发', '年报预增', '专精特新', '上证380', 'HS300',
            '微盘股', '百元股', '大盘股', '小盘成长', '小盘价值',
            '转债标的', '长江三角', '深圳特区', '破净股', '创投']


def qid_of(ds):
    d = pd.Timestamp(ds)
    return f'{d.year}Q{(d.month - 1) // 3 + 1}'


def load_maps():
    with open(MAP_P, 'rb') as f:
        raw = pickle.load(f)
    cmap = {}
    for code, concepts in raw.items():
        filtered = [c for c in concepts if not any(kw in c for kw in STYLE_KW)]
        if filtered:
            cmap[code] = filtered
    if os.path.exists(INCEP_P):
        with open(INCEP_P, 'rb') as f:
            raw_i = pickle.load(f)
        incep = {k: pd.Timestamp(v) for k, v in raw_i.items()}
    else:
        incep = {}
    return cmap, incep


def main():
    g = yaml.safe_load(open(GLOBAL_Y))['industry_factors']
    idx = yaml.safe_load(open(os.path.join(QD, 'index.yaml')))['quarters']
    qs = {}
    for qid, info in idx.items():
        qs[qid] = yaml.safe_load(open(os.path.join(QD, info['file'])))['industry_factors']
    cmap, incep = load_maps()

    df = pd.read_parquet(CACHE_F)
    df['date'] = pd.to_datetime(df['date'])
    df['qid'] = df['date'].apply(qid_of)
    print(f'factor_df: {len(df)} 行, {df["code"].nunique()} 股, {df["date"].nunique()} 日')

    # 概念解析: 复刻 _get_specific_industry P0 (per quarter memo)
    # 返回 key 或 None; 只取 全局∩该季季度 的交集概念 (可切换集)
    def resolve(row):
        code = row['code']
        qcfg = qs.get(row['qid'], {})
        concepts = cmap.get(code)
        if not concepts:
            return None
        ts = row['date']
        for c in concepts:
            _inc = incep.get(c)
            if _inc is not None and ts < _inc:
                continue
            if c in qcfg:
                return c if c in g else None  # 只保留交集(可切换)概念
        return None

    df['key'] = df.apply(resolve, axis=1)
    n_switch = df['key'].notna().sum()
    print(f'可切换集(全局∩季度交集概念): {n_switch} 行 '
          f'({100 * n_switch / len(df):.1f}%)')
    df = df[df['key'].notna()].copy()

    # score计算 (apply逐行, 已缩减样本)
    def score_row(r, which):
        key = r['key']
        if which == 'g':
            cfg, fk, wk = g.get(key, {}), 'factors', 'weights'
        elif which == 'q':
            cfg, fk, wk = qs.get(r['qid'], {}).get(key, {}), 'factors', 'weights'
        else:
            cfg, fk, wk = qs.get(r['qid'], {}).get(key, {}), 'bear_factors', 'bear_weights'
        if not cfg:
            return np.nan
        fl, wl = cfg.get(fk, []), cfg.get(wk, [])
        if not fl or len(fl) != len(wl):
            return np.nan
        s = 0.0
        for f, w in zip(fl, wl):
            if f in df.columns:
                v = r.get(f, np.nan)
                if pd.notna(v):
                    s += float(v) * w
        return s

    for col, which in [('score_g', 'g'), ('score_q', 'q'), ('score_qb', 'qb')]:
        df[col] = df.apply(lambda r: score_row(r, which), axis=1)
    df = df.dropna(subset=['score_g', 'score_q'])
    df = df.dropna(subset=['future_ret'])
    print(f'双score可算: {len(df)} 行')

    def ic_by_date(d, col):
        def _ic(sub):
            if len(sub) < 20 or sub[col].nunique() < 5:
                return np.nan
            rho = spearmanr(sub[col], sub['future_ret']).correlation
            return rho if np.isfinite(rho) else np.nan
        return d.groupby('date').apply(_ic)

    df['year'] = df['date'].dt.year
    ic_g = ic_by_date(df, 'score_g')
    ic_q = ic_by_date(df, 'score_q')
    ic_qb = ic_by_date(df, 'score_qb')
    for label, ics in [('全局', ic_g), ('季度', ic_q), ('季度熊', ic_qb)]:
        n = ics.notna().sum()
        print(f'\n[{label}权重] 逐日IC: n={n}, mean={ics.mean():+.5f}, '
              f't={ics.mean()/ics.std()*np.sqrt(n):.2f}, 正比例={100*(ics>0).mean():.1f}%')
        by = ics.groupby(ics.index.year).agg(ic=('mean'), n=('count'))
        print(by.to_string())

    d = (ic_q - ic_g).dropna()
    print(f'\n[配对差 季度-全局] n={len(d)}, mean={d.mean():+.5f}, '
          f't={d.mean()/d.std()*np.sqrt(len(d)):.2f}, 正比例={100*(d>0).mean():.1f}%')
    print(d.groupby(d.index.year).agg(ic=('mean'), n=('count')).to_string())

    d2 = (ic_qb - ic_g).dropna()
    print(f'\n[配对差 季度熊-全局] n={len(d2)}, mean={d2.mean():+.5f}, '
          f't={d2.mean()/d2.std()*np.sqrt(len(d2)):.2f}, 正比例={100*(d2>0).mean():.1f}%')

    pd.DataFrame({'ic_g': ic_g, 'ic_q': ic_q, 'ic_qb': ic_qb}).to_csv(
        '/tmp/probe_qweight_ic_20260922.csv')
    print('\nIC序列 → /tmp/probe_qweight_ic_20260922.csv (零生产写入)')


if __name__ == '__main__':
    main()
