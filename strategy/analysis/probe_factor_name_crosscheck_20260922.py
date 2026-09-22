"""生产ground-truth交叉验证 (2026-09-22, 只读, 零生产写入)

前序探针(probe_qweight_coverage/blend_bear)的IC裁决依赖离线复刻的概念解析
(STOCK_CONCEPT_MAP风格过滤 + inception gate + 季度成员资格 + 首个命中)。
本脚本用生产验证CSV的 factor_name 列做ground truth:
  若离线复刻与生产消费链一致, 则 switchable 概念行的 factor_name 应与
  分支预测的配置因子前缀吻合:
    - 熊日: 季度bear_factors[:k]
    - 牛日: 全局bull_factors[:k] (无bull则全局factors)
    - 中性日: 全局factors[:k]
    - 概念不在全局但在季度: P1 → 季度factors[:k] (中性/牛日)
匹配规则: factor_name 末token去'_F'后缀 → tokens 与 预测因子列表的有序前缀比对。
特殊名(REV60/REV/MOM/V41/NONE/单因子名等)单列计数, 不参与匹配率分母。
只读: 只写 /tmp/probe_factor_name_xcheck_20260922.csv。
"""
import os
import re
import pickle
import sys

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, '/mnt/d/quant/strategy')

CACHE_F = '/mnt/d/quant/strategy/cache/factor_df_5190s_811d_85f064b9.parquet'
QD = '/mnt/d/quant/strategy/config/quarterly_factors'
GLOBAL_Y = '/mnt/d/quant/strategy/config/factor_config.yaml'
MAP_P = '/mnt/d/quant/data/stock_concept_map.pkl'
INCEP_P = '/mnt/d/quant/data/concept_inception.pkl'
BTD = '/mnt/d/quant/data/stock_data/backtrader_data'
VAL_CSV = '/mnt/d/quant/strategy/rolling_validation_results/validation_results.csv'
FROMDATE = pd.Timestamp('2021-01-01')
TODATE = pd.Timestamp('2026-09-17')

STYLE_KW = ['融资融券', '深股通', '沪股通', '富时罗素', '标准普尔', 'MSCI',
            '创业板综', '机构重仓', 'QFII', '破增发', '破发股', '昨日高',
            '中证500', '深成500', '中盘股', '小盘股', '央国企改革',
            '西部大开发', '年报预增', '专精特新', '上证380', 'HS300',
            '微盘股', '百元股', '大盘股', '小盘成长', '小盘价值',
            '转债标的', '长江三角', '深圳特区', '破净股', '创投']


def qid_of(ds):
    d = pd.Timestamp(ds)
    return f'{d.year}Q{(d.month - 1) // 3 + 1}'


def main():
    # ---- regime (probe_qweight_blend_bear 同款) ----
    from core.market_regime_detector import MarketRegimeDetector
    idx = pd.read_csv(os.path.join(BTD, 'sh000001_qfq.csv'), parse_dates=['datetime'])
    idx = idx[(idx['datetime'] >= FROMDATE) & (idx['datetime'] <= TODATE)]
    small = pd.read_csv(os.path.join(BTD, 'sh000852_qfq.csv'), parse_dates=['datetime'])
    growth = pd.read_csv(os.path.join(BTD, '399006_qfq.csv'), parse_dates=['datetime'])
    det = MarketRegimeDetector()
    det.generate(idx, small_cap_df=small, growth_df=growth)
    reg = det.index_data.set_index('datetime')
    bear_dates = set(reg.index[(reg['regime'] == -1) | (reg['trend_score'] < -0.05)])
    bull_dates = set(reg.index[reg['regime'] == 1])

    # ---- 配置 + 概念映射 ----
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

    # ---- 验证CSV buys ----
    v = pd.read_csv(VAL_CSV, usecols=['date', 'code', 'buy', 'factor_name', 'industry', 'future_ret'],
                    low_memory=False, dtype={'code': str})
    b = v[v['buy'] == 1].copy()
    b['date'] = pd.to_datetime(b['date'])
    b['qid'] = b['date'].apply(qid_of)
    print(f'buys: {len(b)}')

    # ---- 概念解析复刻 (P0, 无切换过滤) ----
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
    n_key = int(b['key'].notna().sum())
    print(f'概念解析命中: {n_key} ({100 * n_key / len(b):.1f}%)')

    # ---- 分支预测 ----
    def predict(row):
        """返回 (来源, 预测因子列表) 或 None"""
        key = row['key']
        d = row['date']
        qcfg = qs.get(row['qid'], {}).get(key) if key else None
        gcfg = g.get(key) if key else None
        if d in bear_dates:
            if qcfg and qcfg.get('bear_factors'):
                return ('qb', qcfg['bear_factors'])
            if gcfg and gcfg.get('bear_factors'):
                return ('gb', gcfg['bear_factors'])
            if gcfg:
                return ('g', gcfg.get('factors', []))
            if qcfg:
                return ('q_p1', qcfg.get('factors', []))
            return None
        if d in bull_dates and gcfg and gcfg.get('bull_factors'):
            return ('gbull', gcfg['bull_factors'])
        if gcfg:
            return ('g', gcfg.get('factors', []))
        if qcfg:
            return ('q_p1', qcfg.get('factors', []))
        return None

    b['pred'] = b.apply(predict, axis=1)
    b_pred = b[b['pred'].notna()].copy()
    b_pred[['src', 'predf']] = pd.DataFrame(b_pred['pred'].tolist(), index=b_pred.index)
    b_pred = b_pred[b_pred['predf'].str.len() > 0]
    print(f'可预测行(有配置因子): {len(b_pred)}')

    # ---- 匹配 (末token去尾部flag字母: F/FV/FBA/T/LA/... 为加成通道标记非因子名) ----
    def tokens_of(name):
        if not isinstance(name, str):
            return None
        # 末token尾缀flag含数字 (R2=bp2通道, E-K1), 用[A-Z0-9]+完整剥除;
        # 因子名全小写无数字, 特殊类名(REV60/V41)无下划线前缀, 故安全
        name2 = re.sub(r'_[A-Z0-9]+$', '', name)
        parts = name2.split('+')
        return [p for p in parts if p]

    def all_factors_of(key, qid):
        """同概念所有可能配置因子集 (用于分支错位诊断)"""
        sets = {}
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

    SPECIAL = {'REV60', 'REV', 'MOM', 'V41', 'NONE', 'trend_lowvol',
               'momentum_reversal', 'relative_strength', 'low_downside'}

    def match(row):
        tk = tokens_of(row['factor_name'])
        if tk is None:
            return 0
        pred = row['predf']
        k = len(tk)
        if k <= len(pred) and tk == pred[:k]:
            return 1  # 有序前缀吻合
        if set(tk) <= SPECIAL or (len(tk) == 1 and tk[0] in SPECIAL):
            return 3  # 类加成/特殊通道名 (E-K1 bp2类等)
        avail = all_factors_of(row['key'], row['qid'])
        if any(set(tk) <= s for s in avail.values()):
            return 4  # 同概念其他分支/其他来源的因子 (分支错位)
        return 2  # 真不吻合

    b_pred['m'] = b_pred.apply(match, axis=1)
    print('\n=== 分来源匹配率 (1=吻合 3=特殊通道 4=同概念他分支 2=真不吻合) ===')
    for src, sub in b_pred.groupby('src'):
        ok = (sub['m'] == 1).sum()
        sp = (sub['m'] == 3).sum()
        ob = (sub['m'] == 4).sum()
        bad = (sub['m'] == 2).sum()
        nk = (sub['m'] == 0).sum()
        print(f'  {src:>6}: n={len(sub)}, 吻合={ok} ({100 * ok / max(len(sub), 1):.1f}%), '
              f'特殊={sp}, 他分支={ob}, 真不吻合={bad}, 不可判={nk}')

    # ---- ground-truth路径census: 生产factor_name实际匹配的配置 (审计§2新鲜版) ----
    P3_NAMES = {'trend_lowvol', 'relative_strength', 'low_downside', 'momentum_reversal'}

    def path_of(row):
        tk = tokens_of(row['factor_name'])
        if not tk:
            return 'unparseable'
        avail = all_factors_of(row['key'], row['qid'])
        # 配置集合优先: P3四元组同时是许多配置的公共前三因子,
        # 必须先排除配置命中再判P3兜底 (否则P3-default被高估)
        for k, s in avail.items():
            if len(tk) <= len(s) and all(t in s for t in tk):
                return k
        if set(tk) <= P3_NAMES:
            return 'P3-default'
        if len(tk) == 1 and tk[0] in SPECIAL:
            return 'special'
        return 'unknown'

    b_pred['path'] = b_pred.apply(path_of, axis=1)
    print('\n=== ground-truth路径census (factor_name实际匹配, 新鲜版) ===')
    pc = b_pred['path'].value_counts()
    tot = len(b_pred)
    for p, n in pc.items():
        print(f'  {p:>8}: {n} ({100 * n / tot:.1f}%)')
    print('  (全buy口径分母=包括未预测组)')
    # future_ret per path (验证CSV fresh future_ret)
    if 'future_ret' in b_pred.columns:
        print('\n  mean future_ret by path:')
        grp = b_pred.groupby('path')['future_ret'].agg(['count', 'mean'])
        print(grp.to_string())
        print('\n  2026年 mean future_ret by path:')
        grp26 = b_pred[b_pred['date'] >= '2026-01-01'].groupby('path')['future_ret'].agg(['count', 'mean'])
        print(grp26.to_string())

    # 熊日子组
    bd = b_pred[b_pred['date'].isin(bear_dates)]
    if len(bd):
        print('\n=== 熊日子组 ===')
        for src, sub in bd.groupby('src'):
            ok = (sub['m'] == 1).sum()
            sp = (sub['m'] == 3).sum()
            ob = (sub['m'] == 4).sum()
            bad = (sub['m'] == 2).sum()
            print(f'  {src:>6}: n={len(sub)}, 吻合={ok} ({100 * ok / max(len(sub), 1):.1f}%), '
                  f'特殊={sp}, 他分支={ob}, 真不吻合={bad}')

    # 他分支(4)诊断: 生产实际用了哪个配置
    ob_rows = b_pred[b_pred['m'] == 4]
    if len(ob_rows):
        print(f'\n他分支行诊断 (n={len(ob_rows)}) — 生产名匹配的同概念配置分布:')
        cnt = {}
        for _, row in ob_rows.head(20000).iterrows():
            tk = tokens_of(row['factor_name'])
            avail = all_factors_of(row['key'], row['qid'])
            hits = [k for k, s in avail.items() if set(tk) <= s]
            for h in hits:
                cnt[h] = cnt.get(h, 0) + 1
        print('  ', cnt)

    # 真不吻合样例
    bad = b_pred[b_pred['m'] == 2]
    if len(bad):
        union_hits = 0
        for _, row in bad.iterrows():
            tk = tokens_of(row['factor_name'])
            if not tk:
                continue
            avail = all_factors_of(row['key'], row['qid'])
            if not avail:
                continue
            if all(t in set().union(*avail.values()) for t in tk):
                union_hits += 1
        print(f'\n真不吻合中 全体token落在同概念配置并集内 (跨集组合/加成注入): '
              f'{union_hits} ({100 * union_hits / len(bad):.1f}%)')
    print(f'\n真不吻合样例 (n={len(bad)}):')
    print(bad[['date', 'code', 'key', 'src', 'factor_name']].head(10).to_string())
    if len(bad):
        print('\n真不吻合 factor_name top:')
        print(bad['factor_name'].value_counts().head(8).to_string())

    # 特殊名/未解析组
    spec = b[b['pred'].isna()]
    print(f'\n未预测组(无解析或无配置): {len(spec)}')
    print(spec['factor_name'].value_counts().head(10).to_string())

    b_pred.to_csv('/tmp/probe_factor_name_xcheck_20260922.csv', index=False)
    print('\n细节 → /tmp/probe_factor_name_xcheck_20260922.csv (零生产写入)')


if __name__ == '__main__':
    main()
