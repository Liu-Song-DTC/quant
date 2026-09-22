"""被执行仓位的评分路径归因 (2026-09-22, 只读, 零生产写入)

评分链审计§8给了全buy层的路径census (qb 43.1%/gbull 32.5%/g 15.3%/...),
但组合层只执行570行 — 执行集与全buy集的路径结构是否一致? 哪个路径的
执行贡献了真金白银? 若某路径执行集中且 future_ret 显著为负/正,
对应分支的加严/放宽成为候选方向 (评分链已审计: 权重切换/λ-blend关闭,
但"执行集内路径质量"尚未看过)。

方法: portfolio_selections.csv (570行, 生产执行) × /tmp/probe_factor_name_xcheck
CSV (全buy行路径分类, 9/22 fresh验证CSV): 同日buy行精确join, 未命中者取
sel_date前15日内最近buy行。future_ret=20日forward (FORWARD_PERIOD), 是
持仓前20日的粗略代理 (非实现P&L — 实现在trade_realized.csv, 本探针只做
路径比较)。零写入: 只写 /tmp/probe_exec_path_20260922.csv
"""
import pandas as pd

XC = '/tmp/probe_factor_name_xcheck_20260922.csv'
SEL = '/mnt/d/quant/strategy/rolling_validation_results/portfolio_selections.csv'


def main():
    x = pd.read_csv(XC, low_memory=False, dtype={'code': str})
    x['date'] = pd.to_datetime(x['date'])
    print(f'xcheck buy行: {len(x)}, 路径: {sorted(x["path"].dropna().unique())}')

    sel = pd.read_csv(SEL, dtype={'code': str})
    sel['date'] = pd.to_datetime(sel['date'])

    # 精确join
    m = sel.merge(x[['date', 'code', 'path', 'future_ret', 'factor_name']],
                  on=['date', 'code'], how='left')
    matched = m[m['path'].notna()].copy()
    print(f'执行行同日join命中: {len(matched)} / {len(sel)}')

    # 未命中 → 15日内最近buy
    miss = m[m['path'].isna()].copy()
    miss = miss.drop(columns=['path', 'future_ret', 'factor_name'])
    rows = []
    for _, r in miss.iterrows():
        sub = x[(x['code'] == r['code']) & (x['date'] <= r['date'])
                & (x['date'] >= r['date'] - pd.Timedelta(days=15))]
        if len(sub):
            last = sub.iloc[-1]
            rows.append({'date': r['date'], 'code': r['code'], 'weight': r['weight'],
                         'path': last['path'], 'future_ret': last['future_ret'],
                         'factor_name': last['factor_name'],
                         'buy_date': last['date']})
        else:
            rows.append({'date': r['date'], 'code': r['code'], 'weight': r['weight'],
                         'path': 'no_buy_15d', 'future_ret': None,
                         'factor_name': None, 'buy_date': None})
    stale = pd.DataFrame(rows)
    print(f'陈信号兜底: {len(stale[stale["path"] != "no_buy_15d"])} 命中, '
          f'{len(stale[stale["path"] == "no_buy_15d"])} 无buy')

    ex = pd.concat([matched[['date', 'code', 'weight', 'path', 'future_ret',
                             'factor_name']], stale], ignore_index=True)

    print('\n=== 执行集路径census (570行) ===')
    pc = ex['path'].value_counts()
    for p, n in pc.items():
        print(f'  {p:>10}: {n} ({100 * n / len(ex):.1f}%)')

    print('\n全buy层路径census (对照, 从xcheck):')
    xc = x['path'].value_counts()
    for p, n in xc.items():
        print(f'  {p:>10}: {n} ({100 * n / len(x):.1f}%)')

    print('\n=== 执行集 per-path 贡献 (weight×future_ret = 前20日近似) ===')
    ex['wfr'] = ex['weight'] * ex['future_ret']
    grp = ex.groupby('path').agg(n=('code', 'size'), wsum=('weight', 'sum'),
                                 wfr=('wfr', 'sum'), mean_fr=('future_ret', 'mean'),
                                 wfr_share=('wfr', lambda s: 100 * s.sum() / ex['wfr'].sum()))
    print(grp.to_string())
    print(f'\n全执行集: Σw={ex["weight"].sum():.2f}, Σw×fr={ex["wfr"].sum():+.4f}, '
          f'mean_fr={ex["future_ret"].mean():+.4f}')

    print('\n=== 分年 per-path wfr ===')
    ex['year'] = ex['date'].dt.year
    piv = ex.pivot_table(index='year', columns='path', values='wfr',
                         aggfunc='sum', fill_value=0.0)
    print(piv.round(4).to_string())
    print('\n分年 Σw×fr:')
    print(ex.groupby('year')['wfr'].sum().round(4).to_string())

    ex.to_csv('/tmp/probe_exec_path_20260922.csv', index=False)
    print('\n细节 → /tmp/probe_exec_path_20260922.csv (零生产写入)')


if __name__ == '__main__':
    main()
