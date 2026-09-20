#!/usr/bin/env python3
"""
Regime条件化ML blend前置探针 (2026-09-20, 只读)

动机: C1家族未测变体 = "ML blend按regime调节(bull高/bear低)"。此前置探针
用现成CSV零重生成回答: ML分数在bull/neutral/bear下的截面IC是否regime依赖?
若ML在bear日IC崩塌 → 条件化有因果动机, 才值得付全链重生成的臂成本;
若三regime无差异 → 机制先验不存在, 方向关闭 (省3h+/臂)。

口径: buy=1行(选股决策人群), Spearman(ml_score, future_ret)按日分regime桶;
因子侧分数由score反解: factor_side = (score - 0.4*ml_score)/0.6 (blend=0.4,
C1审计证实score=0.6*factor+0.4*ml)。分段: 全史(2021-06+)/2025春季暴跌段
(3/15~4/15, MDD 17.92%所在)/2026Q3(7/1~9/3, OOS段)。

执行: cd strategy && /mnt/d/quant/.venv/bin/python analysis/probe_ml_regime_20260920.py
"""
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scipy.stats import spearmanr

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RVR = os.path.join(BASE_DIR, 'rolling_validation_results')

BLEND = 0.4  # C1审计: score = (1-BLEND)*factor_side + BLEND*ml_score


def load():
    sig = pd.read_csv(os.path.join(RVR, 'backtest_signals.csv'),
                      usecols=['date', 'code', 'buy', 'score', 'ml_score'],
                      dtype={'code': str}, low_memory=False)
    sig['date'] = pd.to_datetime(sig['date'])
    val = pd.read_csv(os.path.join(RVR, 'validation_results.csv'),
                      usecols=['date', 'code', 'buy', 'future_ret'],
                      dtype={'code': str}, low_memory=False)
    val['date'] = pd.to_datetime(val['date'])
    m = sig.merge(val, on=['date', 'code', 'buy'], how='inner')
    print(f"合并后行数: {len(m)} (信号 {len(sig)} / 验证 {len(val)})")
    # regime per date from cached factor_df
    fd = pd.read_pickle('/tmp/v2_factor_df_0917.pkl')
    reg = fd.groupby('date')['regime'].first()
    del fd
    m['regime'] = m['date'].map(reg)
    m = m.dropna(subset=['regime', 'ml_score', 'future_ret'])
    m['factor_side'] = (m['score'] - BLEND * m['ml_score']) / (1 - BLEND)
    print(f"有效行(regime+future_ret+ml非空): {len(m)}, 日期 {m['date'].min().date()}~{m['date'].max().date()}")
    return m


def ic_bucket(df, tag):
    """按日截面IC, 再按regime桶平均"""
    out = []
    for d, g in df.groupby('date'):
        if len(g) < 10:
            continue
        for col in ('ml_score', 'factor_side'):
            r, _ = spearmanr(g[col], g['future_ret'])
            if r == r:  # 非NaN
                out.append((d, g['regime'].iloc[0], col, r))
    r = pd.DataFrame(out, columns=['date', 'regime', 'col', 'ic'])
    if r.empty:
        return None
    piv = r.pivot_table(index='regime', columns='col', values='ic', aggfunc='mean')
    n = r.pivot_table(index='regime', columns='col', values='ic', aggfunc='count')
    print(f"\n  [{tag}] 截面IC均值 (日数):")
    for regime in (1, 0, -1):
        if regime in piv.index:
            line = f"    regime={int(regime):+d}: "
            for col in ('ml_score', 'factor_side'):
                if col in piv.columns:
                    line += f"{col} {piv.at[regime, col]:+.4f} ({int(n.at[regime, col])}日)  "
            print(line)
    return piv


def main():
    print("Regime条件化ML blend前置探针 (buy=1行, 截面IC按regime桶)")
    m = load()
    b = m[m['buy'] == 1]
    print(f"buy=1行: {len(b)}")
    ic_bucket(b, '全史 buy=1')
    ic_bucket(b[(b['date'] >= '2025-03-15') & (b['date'] <= '2025-04-15')],
              '2025春季暴跌段 (3/15~4/15)')
    ic_bucket(b[b['date'] >= '2026-07-01'], '2026Q3 OOS段 (7/1~)')
    # 日数分布
    print("\n  regime日数分布 (buy=1有IC日):")
    print("    " + str(b.groupby('date')['regime'].first().value_counts().to_dict()))


if __name__ == '__main__':
    main()
