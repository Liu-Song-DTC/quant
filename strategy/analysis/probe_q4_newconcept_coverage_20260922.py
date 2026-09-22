"""Q4标定新概念覆盖预检 (2026-09-22, 只读, 零生产写入)

背景: yaml_census_audit §5 — 季度标定的真实增益=新概念覆盖(仅季度36概念季度
权重IC +0.2423 vs 默认+0.0089) + 熊分支 + gating。9/30 Q4标定(calib_2026Q4_tail)
窗口=2021-10-01~2026-09-30, min_codes=20股以下概念跳过标定。

本预检给焦点纪律提供弹药: 36个仅季度概念(2026Q3-全局)在标定窗口内的
估计样本量 — 两种口径:
  A. 标定实际口径 (offline_calibration.py:143-144: 每概念用ALL所属股票,
     无inception gate) — 标定脚本真正喂给IC估计的样本;
  B. 生产消费口径 (inception PIT gate) — 生产signal_engine只消费
     inception之后的概念归属行。
两者差异大的概念 = 标定估计样本被前视归属行稀释, 权重可靠性打折扣。

数据源: validation_results.csv (9/22 fresh, 全池行) × stock_concept_map.pkl
(风格过滤) × concept_inception.pkl。零写入: /tmp/probe_q4_cov_20260922.csv
"""
import os
import pickle
import sys

import pandas as pd
import yaml

sys.path.insert(0, '/mnt/d/quant/strategy')

VAL_CSV = '/mnt/d/quant/strategy/rolling_validation_results/validation_results.csv'
QD = '/mnt/d/quant/strategy/config/quarterly_factors'
GLOBAL_Y = '/mnt/d/quant/strategy/config/factor_config.yaml'
MAP_P = '/mnt/d/quant/data/stock_concept_map.pkl'
INCEP_P = '/mnt/d/quant/data/concept_inception.pkl'
CALIB_START = pd.Timestamp('2021-10-01')

STYLE_KW = ['融资融券', '深股通', '沪股通', '富时罗素', '标准普尔', 'MSCI',
            '创业板综', '机构重仓', 'QFII', '破增发', '破发股', '昨日高',
            '中证500', '深成500', '中盘股', '小盘股', '央国企改革',
            '西部大开发', '年报预增', '专精特新', '上证380', 'HS300',
            '微盘股', '百元股', '大盘股', '小盘成长', '小盘价值',
            '转债标的', '长江三角', '深圳特区', '破净股', '创投']


def main():
    # ---- 配置 ----
    g = yaml.safe_load(open(GLOBAL_Y))['industry_factors']
    idx_cfg = yaml.safe_load(open(os.path.join(QD, 'index.yaml')))['quarters']
    latest_qid = list(idx_cfg)[-1]
    q_last = yaml.safe_load(open(os.path.join(QD, idx_cfg[latest_qid]['file'])))['industry_factors']
    qonly = sorted(set(q_last) - set(g))
    print(f'仅季度概念({latest_qid}-全局): {len(qonly)}')
    print('清单:', qonly)

    with open(MAP_P, 'rb') as f:
        raw = pickle.load(f)
    cmap = {}
    for code, concepts in raw.items():
        filtered = [c for c in concepts if not any(kw in c for kw in STYLE_KW)]
        if filtered:
            cmap[code] = filtered
    with open(INCEP_P, 'rb') as f:
        raw_i = pickle.load(f)
    incep = {k: pd.Timestamp(v) for k, v in raw_i.items()}
    incep_dates = pd.Series(incep)

    # ---- 全池行 (date, code) ----
    v = pd.read_csv(VAL_CSV, usecols=['date', 'code'],
                    low_memory=False, dtype={'code': str})
    v = v.drop_duplicates()
    v['date'] = pd.to_datetime(v['date'])
    v = v[v['date'] >= CALIB_START].copy()
    print(f'窗口行数(2021-10-01+): {len(v)}, 天数: {v["date"].nunique()}, '
          f'股票: {v["code"].nunique()}')

    # ---- 无gate口径 (标定实际): 股票全部概念 ----
    v['concepts'] = v['code'].map(cmap)
    va = v.dropna(subset=['concepts']).explode('concepts')
    va = va[va['concepts'].isin(qonly)]
    a_grp = va.groupby('concepts').agg(
        stock_days=('code', 'size'),
        n_stocks=('code', 'nunique'),
        n_days=('date', 'nunique'))

    # ---- gate口径 (生产消费): inception过滤 ----
    vg = v.dropna(subset=['concepts']).explode('concepts')
    vg = vg[vg['concepts'].isin(qonly)]
    # 概念inception: 若concept无inception记录 → 视为始终有效 (与生产一致)
    vg['inc'] = vg['concepts'].map(incep_dates)
    vg = vg[(vg['inc'].isna()) | (vg['date'] >= vg['inc'])]
    g_grp = vg.groupby('concepts').agg(
        stock_days_gated=('code', 'size'),
        n_stocks_gated=('code', 'nunique'),
        n_days_gated=('date', 'nunique'))

    res = a_grp.join(g_grp, how='outer').fillna(0).astype(int)
    res['gate_share'] = (res['stock_days_gated'] / res['stock_days']).round(3)
    res = res.sort_values('stock_days')
    pd.set_option('display.width', 200)
    print('\n=== 36仅季度概念 标定窗口样本量 (A=标定实际无gate, B=生产消费gate) ===')
    print(res.to_string())
    print('\nmin_codes=20股以下被标定跳过:', res[res['n_stocks'] < 20].index.tolist())
    print('gate_share<0.5 (估计样本被前视行稀释>半):',
          res[res['gate_share'] < 0.5].index.tolist())

    # 对照: 全部概念的分布参考 (标定实际口径)
    va_all = v.dropna(subset=['concepts']).explode('concepts')
    all_grp = va_all.groupby('concepts').size()
    print(f'\n对照: 全部概念标定口径stock_days分布 (n={len(all_grp)}):')
    print(all_grp.describe().round(0).to_string())
    qonly_sd = res['stock_days']
    print(f'\n36仅季度概念 stock_days分位: min={qonly_sd.min()}, '
          f'25%={qonly_sd.quantile(0.25):.0f}, 中位={qonly_sd.median():.0f}, '
          f'75%={qonly_sd.quantile(0.75):.0f}, max={qonly_sd.max()}')

    res.to_csv('/tmp/probe_q4_cov_20260922.csv')
    print('\n细节 → /tmp/probe_q4_cov_20260922.csv (零生产写入)')


if __name__ == '__main__':
    main()
