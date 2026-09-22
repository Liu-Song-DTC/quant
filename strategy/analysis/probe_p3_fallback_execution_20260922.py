"""P3兜底信号是否进入组合层探针 (2026-09-22, 只读, 零生产写入)

背景: yaml_census_audit_20260922.md §8 — 真P3兜底(概念无配置→硬编码四因子评分)
≈3.1% buys, 2026 mean future_ret −6.9%显著为负。本探针回答: P3信号是否实际被
执行(portfolio_selections)。若执行且权重不小 → "P3禁买/加严"成为候选臂
(code-patch arm, 冷跑四指标裁决); 若从不执行 → 方向廉价关闭。

方法: validation_results.csv (9/22 fresh) buy=1行按生产消费链复刻分类P3
(同 probe_factor_name_crosscheck: STYLE_KW过滤+inception gate+季度成员资格+
首命中; tokens剥除尾缀flag)。与 portfolio_selections.csv (570行执行记录)
在 (date,code) 精确join + 陈信号匹配(sel_date前15日内最近buy行)兜底。
另报: P3 buy行在各执行日的score竞争地位(当日buy分数中的百分位)。

零写入: 只写 /tmp/probe_p3_exec_20260922.csv
"""
import os
import pickle
import re
import sys

import pandas as pd
import yaml

sys.path.insert(0, '/mnt/d/quant/strategy')

VAL_CSV = '/mnt/d/quant/strategy/rolling_validation_results/validation_results.csv'
SEL_CSV = '/mnt/d/quant/strategy/rolling_validation_results/portfolio_selections.csv'
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
P3_NAMES = {'trend_lowvol', 'relative_strength', 'low_downside', 'momentum_reversal'}


def qid_of(ds):
    d = pd.Timestamp(ds)
    return f'{d.year}Q{(d.month - 1) // 3 + 1}'


def tokens_of(name):
    if not isinstance(name, str):
        return None
    name2 = re.sub(r'_[A-Z0-9]+$', '', name)
    parts = name2.split('+')
    return [p for p in parts if p]


def main():
    g = yaml.safe_load(open(GLOBAL_Y))['industry_factors']
    idx_cfg = yaml.safe_load(open(os.path.join(QD, 'index.yaml')))['quarters']
    qs = {qid: yaml.safe_load(open(os.path.join(QD, info['file'])))['industry_factors']
          for qid, info in idx_cfg.items()}
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

    v = pd.read_csv(VAL_CSV, usecols=['date', 'code', 'buy', 'score', 'factor_name',
                                      'future_ret'],
                    low_memory=False, dtype={'code': str})
    b = v[v['buy'] == 1].copy()
    del v
    b['date'] = pd.to_datetime(b['date'])
    b['qid'] = b['date'].apply(qid_of)
    print(f'buy行: {len(b)}')

    def resolve(code, date, qid):
        qcfg = qs.get(qid, {})
        concepts = cmap.get(code)
        if not concepts:
            return None
        ts = date
        for c in concepts:
            _inc = incep.get(c)
            if _inc is not None and ts < _inc:
                continue
            if c in qcfg:
                return c
        return None

    b['key'] = b.apply(lambda r: resolve(r['code'], r['date'], r['qid']), axis=1)

    def cfg_sets_of(key, qid):
        sets = {}
        if key:
            gcfg = g.get(key, {})
            qcfg = qs.get(qid, {}).get(key, {})
            if gcfg:
                sets['g'] = set(gcfg.get('factors', []))
                sets['gbull'] = set(gcfg.get('bull_factors', []))
                sets['gbear'] = set(gcfg.get('bear_factors', []))
            if qcfg:
                sets['q'] = set(qcfg.get('factors', []))
                sets['qb'] = set(qcfg.get('bear_factors', []))
        return sets

    def is_p3(row):
        tk = tokens_of(row['factor_name'])
        if not tk:
            return False
        if not set(tk) <= P3_NAMES:
            return False
        sets = cfg_sets_of(row['key'], row['qid'])
        for s in sets.values():
            if len(tk) <= len(s) and all(t in s for t in tk):
                return False  # 配置命中, 非P3兜底
        return True

    b['p3'] = b.apply(is_p3, axis=1)
    n_p3 = int(b['p3'].sum())
    print(f'P3兜底buy行: {n_p3} ({100 * n_p3 / len(b):.2f}%)')
    print('P3行分年:')
    print(b[b['p3']].groupby(b['date'].dt.year).size().to_string())
    print('\nP3行 score分布 vs 全buy:')
    print(b[['p3', 'score']].groupby('p3')['score'].describe().to_string())
    print('\nP3行 future_ret均值:',
          f'{b.loc[b["p3"], "future_ret"].mean():+.5f}',
          '(全buy', f'{b["future_ret"].mean():+.5f})')

    # ---- 执行join ----
    sel = pd.read_csv(SEL_CSV, dtype={'code': str})
    sel['date'] = pd.to_datetime(sel['date'])
    print(f'\n执行记录: {len(sel)}行, {sel["date"].nunique()}日, '
          f'{sel["code"].nunique()}股')

    # 精确join: (date, code) buy行
    m = sel.merge(b[b['p3']][['date', 'code', 'score', 'future_ret', 'factor_name']],
                  on=['date', 'code'], how='left', indicator=True,
                  suffixes=('_sel', '_val'))
    p3_exact = m[m['_merge'] == 'both']
    print(f'\n=== 精确join (选择日当天即P3 buy) ===')
    print(f'P3被执行行: {len(p3_exact)} / {len(sel)} ({100 * len(p3_exact) / len(sel):.1f}%)')
    if len(p3_exact):
        print(p3_exact[['date', 'code', 'weight', 'score_sel', 'score_val',
                        'future_ret', 'factor_name']].to_string())
        wd = (p3_exact['weight'] * p3_exact['future_ret']).sum()
        print(f'weight×future_ret合计: {wd:+.5f} (权重和 {p3_exact["weight"].sum():.3f})')
        print('分年:')
        print(p3_exact.groupby(p3_exact['date'].dt.year).agg(
            n=('code', 'size'), wsum=('weight', 'sum'),
            wfr=('future_ret', lambda x: (p3_exact.loc[x.index, 'weight'] * x).sum())).to_string())

    # 陈信号匹配: 未命中的选择行 → sel_date前15日内最近buy行是否P3
    unmatched = m[m['_merge'] == 'left_only'].copy()
    unmatched = unmatched.drop(columns=['score_val', 'future_ret', 'factor_name'])
    print(f'\n=== 陈信号匹配 ({len(unmatched)}未命中选择行, 前15日最近buy) ===')
    rows = []
    for _, r in unmatched.iterrows():
        sub = b[(b['code'] == r['code']) & (b['date'] <= r['date'])
                & (b['date'] >= r['date'] - pd.Timedelta(days=15))]
        if len(sub):
            last = sub.iloc[-1]
            rows.append((r['date'], r['code'], r['weight'], last['date'],
                         bool(last['p3']), last['score'], last['future_ret']))
        else:
            rows.append((r['date'], r['code'], r['weight'], None, None, None, None))
    um = pd.DataFrame(rows, columns=['sel_date', 'code', 'weight', 'last_buy_date',
                                     'last_is_p3', 'last_score', 'last_fr'])
    stale_p3 = um[um['last_is_p3'] == True]  # noqa: E712
    print(f'最近buy为P3的选择行: {len(stale_p3)} / {len(um)}')
    if len(stale_p3):
        print(stale_p3.to_string())
        print(f'weight×future_ret合计: '
              f'{(stale_p3["weight"] * stale_p3["last_fr"]).sum():+.5f}')

    # 未命中且15日内无buy行的选择 (信号更老/其他通道)
    no_buy = um[um['last_is_p3'].isna()]
    print(f'15日内无buy行的选择: {len(no_buy)} (其他通道如REV60/bp2)')

    # P3 buy行的当日score竞争地位 (与当日全部buy分数的分位)
    print('\n=== P3 buy行score当日竞争地位 ===')
    b['rank'] = b.groupby('date')['score'].rank(pct=True)
    print('P3行score当日百分位:')
    print(b[b['p3']]['rank'].describe().to_string())
    sel_scores = sel.merge(b[['date', 'code', 'score', 'rank', 'p3']],
                           on=['date', 'code'], how='left')
    print('被执行行的score当日百分位 (全被执行):')
    print(sel_scores['rank'].describe().to_string())

    b[['date', 'code', 'p3', 'score', 'future_ret', 'factor_name', 'key']].to_csv(
        '/tmp/probe_p3_exec_20260922.csv', index=False)
    print('\n细节 → /tmp/probe_p3_exec_20260922.csv (零生产写入)')


if __name__ == '__main__':
    main()
