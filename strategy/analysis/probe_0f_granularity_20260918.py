# -*- coding: utf-8 -*-
"""
probe_0f_granularity_20260918: 0f季度日历池的粒度成本探针 (0f-v2 daily决策前置).

季度池(现状0f)的两个失效模式量化:
  A. 死股续命: 季度池把已失流动性的码保留最多3个月 (daily池会在当天剔除) —
     0f已成交买单中, 买日不属于daily池的比例 + 其前瞻收益 vs daily合法买单。
  B. 锁定热度: daily入池早于季度入池的码, 其"被锁窗口"(daily入池→季度入池)
     的涨幅 = 季度池错过的一段行情 (2024-25流动性爆发期的核心问题)。

只读探针: 不写任何生产文件, 不动yaml/指纹。
运行: /mnt/d/quant/.venv/bin/python strategy/analysis/probe_0f_granularity_20260918.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import pandas as pd
import numpy as np

from core.stock_pool import get_pool_membership_map, _quarter_boundaries

_BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SEL_PATH = os.path.join(_BASE, 'strategy', 'rolling_validation_results', 'portfolio_selections.csv')
DATA_DIR = os.path.join(_BASE, 'data', 'stock_data', 'backtrader_data')
IDX = {'sh000001', 'sh000852', '000001', '399006'}

t0 = pd.Timestamp.now()
print(f'[{t0:%H:%M:%S}] 探针启动', flush=True)

# ── 0f selections (刚完成的季度0f冷跑产物) ──
if not os.path.exists(SEL_PATH):
    print(f'FATAL: {SEL_PATH} 不存在'); sys.exit(1)
sel = pd.read_csv(SEL_PATH)
print(f'selections 列: {sel.columns.tolist()}', flush=True)
print(f'selections 行数: {len(sel)}', flush=True)
code_col = 'code' if 'code' in sel.columns else sel.columns[0]
date_col = 'date' if 'date' in sel.columns else sel.columns[1]
sel[date_col] = pd.to_datetime(sel[date_col])
sel['year'] = sel[date_col].dt.year
print(f'年度分布:\n{sel.groupby("year").size()}', flush=True)

# ── 季度membership (复用冷跑缓存, 键=3374606d8186, 边界须与冷跑一致) ──
qb = _quarter_boundaries(pd.Timestamp('2019-01-01'), pd.Timestamp('2026-09-17'))
m_q = get_pool_membership_map(qb, data_dir=DATA_DIR, cache_key='3374606d8186')
print(f'季度map: {len(m_q)} 边界 (缓存键3374606d8186)', flush=True)
qb_sorted = sorted(m_q.keys())

# ── daily membership (无缓存单遍扫描, 2020-12-31..2026-09-17) ──
db = list(pd.date_range('2020-12-31', '2026-09-17', freq='D'))
t1 = pd.Timestamp.now()
m_d = get_pool_membership_map(db, data_dir=DATA_DIR, cache_key=None)
print(f'daily map: {len(m_d)} 边界, 扫描耗时 {(pd.Timestamp.now()-t1).total_seconds():.0f}s', flush=True)
db_sorted = sorted(m_d.keys())

def prev_boundary(sorted_bounds, t):
    i = np.searchsorted(sorted_bounds, t, side='right') - 1
    return sorted_bounds[i] if i >= 0 else None

# ── A. 死股续命: 0f买单中买日∉daily池的比例 + 前瞻收益 ──
codes_needed = set(sel[code_col].astype(str))
print(f'\n=== A. 死股续命 === 需加载 {len(codes_needed)} 只K线', flush=True)
close_map = {}
for i, c in enumerate(codes_needed):
    p = os.path.join(DATA_DIR, f'{c}_qfq.csv')
    if os.path.exists(p):
        try:
            df = pd.read_csv(p, usecols=['datetime', 'close'])
            df['datetime'] = pd.to_datetime(df['datetime'])
            close_map[c] = df.set_index('datetime')['close']
        except Exception:
            pass
print(f'K线加载完成 {len(close_map)} 只', flush=True)

def fwd_ret(code, t, n):
    s = close_map.get(code)
    if s is None:
        return np.nan
    idx = s.index.searchsorted(t, side='left')
    j = min(idx + n, len(s) - 1)
    if j <= idx or idx >= len(s):
        return np.nan
    return s.iloc[j] / s.iloc[idx] - 1.0

rows = []
for _, r in sel.iterrows():
    c, t = str(r[code_col]), r[date_col]
    b = prev_boundary(db_sorted, t)
    if b is None:
        continue
    rows.append({
        'code': c, 'date': t, 'year': r['year'],
        'in_daily': c in m_d[b] or c in IDX,
        'fwd5': fwd_ret(c, t, 5), 'fwd20': fwd_ret(c, t, 20),
    })
dfa = pd.DataFrame(rows)
print(f'买单总数 {len(dfa)}; daily合法 {dfa["in_daily"].sum()} '
      f'({dfa["in_daily"].mean()*100:.1f}%); 死股续命 {(~dfa["in_daily"]).sum()} '
      f'({(~dfa["in_daily"]).mean()*100:.1f}%)', flush=True)
for y, g in dfa.groupby('year'):
    dead = g[~g['in_daily']]
    ok = g[g['in_daily']]
    print(f'  {y}: 总买{len(g)} 死股{len(dead)} ({len(dead)/max(len(g),1)*100:.0f}%) | '
          f'fwd5 死股{dead["fwd5"].mean()*100:+.2f}% vs 合法{ok["fwd5"].mean()*100:+.2f}% | '
          f'fwd20 死股{dead["fwd20"].mean()*100:+.2f}% vs 合法{ok["fwd20"].mean()*100:+.2f}%', flush=True)

# ── B. 锁定热度: daily入池早于季度入池的码, 被锁窗口涨幅 ──
print(f'\n=== B. 锁定热度 (2024-01..2026-09) ===', flush=True)
b_start = pd.Timestamp('2024-01-01')
codes_daily = set().union(*[m_d[b] for b in m_d if b >= b_start])
codes_all = codes_daily | set(sel[code_col].astype(str))
daily_admit, q_admit = {}, {}
for c in codes_all:
    da = min((b for b in db_sorted if c in m_d[b]), default=None)
    qa = min((b for b in qb_sorted if c in m_q[b]), default=None)
    if da is not None:
        daily_admit[c] = da
    if qa is not None:
        q_admit[c] = qa
print(f'daily曾入池(2024后) {len(daily_admit)}; 有季度入池记录 {len(q_admit)}', flush=True)
locked = []
for c, da in daily_admit.items():
    qa = q_admit.get(c)
    if qa is None or da >= qa:
        continue
    s = close_map.get(c)
    if s is None:
        p = os.path.join(DATA_DIR, f'{c}_qfq.csv')
        if os.path.exists(p):
            try:
                df = pd.read_csv(p, usecols=['datetime', 'close'])
                df['datetime'] = pd.to_datetime(df['datetime'])
                s = df.set_index('datetime')['close']
                close_map[c] = s
            except Exception:
                continue
        else:
            continue
    i0, i1 = s.index.searchsorted(da, side='left'), s.index.searchsorted(qa, side='left')
    if i1 > i0 and i0 < len(s):
        ret = s.iloc[min(i1, len(s)-1)] / s.iloc[i0] - 1.0
        locked.append({'code': c, 'daily_admit': da, 'q_admit': qa,
                       'lag_days': (qa - da).days, 'lag_ret': ret,
                       'year': da.year})
dfl = pd.DataFrame(locked)
print(f'被锁码总数 {len(dfl)} (2024-01后daily先入池, 季度池等待)', flush=True)
for y, g in dfl.groupby('year'):
    print(f'  {y}: {len(g)}只 | 中位锁窗 {g["lag_days"].median():.0f}天 | '
          f'锁窗涨幅 mean {g["lag_ret"].mean()*100:+.1f}% median {g["lag_ret"].median()*100:+.1f}% | '
          f'锁窗≥20%: {(g["lag_ret"]>=0.20).sum()}只', flush=True)
top = dfl.nlargest(20, 'lag_ret')[['code', 'daily_admit', 'q_admit', 'lag_ret']]
print('锁窗涨幅top20:', flush=True)
print(top.to_string(index=False), flush=True)
print(f'[{pd.Timestamp.now():%H:%M:%S}] 探针完成, 总耗时 {(pd.Timestamp.now()-t0).total_seconds():.0f}s', flush=True)
