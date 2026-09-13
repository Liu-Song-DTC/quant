#!/usr/bin/env python3
"""alpha_factory.py — 缺口1: 老数据新表达式挖掘 (alpha工厂, 2026-09-13)

背景: 阶段3/4死的是"新数据源", 从没试过"老数据新表达式"。本脚本在
  63个生产原始因子 + 18个新OHLCV基础列(影线/VWAP偏离/效率比/波动期限结构/
  量价能量/换手加速度等, 全部避开现有63列) 上挖掘算子树表达式。

方法:
  - 面板: factor_df (code × 804个稠密日期) 逐列unstack; fwd20标签由qfq CSV
    收盘矩阵独立计算 (生产future_ret只做一致性identity检查, 不直接当标签)。
  - 新基础列: 在完整日频序列上计算(与生产因子同语义的交易日窗口),
    再采样到grid (避免grid 2日间隔扭曲窗口)。
  - 生成: 手工canonical (~164, 模板族+交叉对+三操作符) + 随机算子树 (~180,
    确定种子, depth 2-3, 纯向量化ts操作符; rank/decay/corr需rolling.apply
    约100s/表达式, 排除)。
  - 评估: 月度锚点(每月首个grid日, 2021-01~2026-08), 截面Spearman IC vs
    市场调整fwd20 (与probe_ml_blindspots_0910同约定), 逐锚点rank向量化。

预置闸 (跑前定, 只读, 不写任何生产文件):
  D1 覆盖率: 有限值占比≥50% 且 有效锚点≥40个(月, n≥50)
  D2 非常数: nanstd > 1e-8
  D3 去重: 与已收表达式在子采样列上 |corr| < 0.95
  D4 池IC_f20 ≥ 0.030        (噪声天花板≈0.009)
  D5 IC正比例 ≥ 60%
  D6 ≥5/6年块IC均值非负
  E  幸存者top-6配对(tail2同法, vs 生产score=adjusted_score):
     均值差≥+0.10pp 且 正比例≥55% 且 ≥4/6年非负
  F  CSCV家族诊断: 全部过D1-D3的表达式 × 6年块 IC矩阵,
     cscv_pbo(全枚举C(6,3)=20) vs 同(N,S=6)零分布带(K=40)。
     家族PBO低于零带5分位=选优稳健; 带内=空间整体似噪声(幸存者可疑);
     高于95分位=病态。诊断性 — 硬否决权在D/E闸。
  仅当E全过且家族PBO<5分位的幸存者才进入接线讨论(重训+冷跑四指标铁律)。

执行: cd strategy && /mnt/d/quant/.venv/bin/python analysis/alpha_factory.py \
      > logs/alpha_factory_0913.log 2>&1
"""
import os
import sys
import time
import resource
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cscv  # 同目录, 已入库52bb044

ROOT = '/mnt/d/quant'
CACHE = os.path.join(ROOT, 'strategy', 'cache')
PARQUET = os.path.join(CACHE, 'factor_df_2718s_809d_d814a206.parquet')
BT = os.path.join(ROOT, 'data', 'stock_data', 'backtrader_data')
SIG = os.path.join(ROOT, 'strategy', 'rolling_validation_results',
                   'backtest_signals.csv')
IDX = os.path.join(BT, 'sh000001_qfq.csv')
OUT_CSV = os.path.join(ROOT, 'strategy', 'rolling_validation_results',
                       'alpha_factory_survivors.csv')
OUT_MAT = os.path.join(ROOT, 'strategy', 'rolling_validation_results',
                       'alpha_factory_cscv_matrix.csv')

START, END = '2021-01-01', '2026-08-01'   # IC锚点区间
FWD = 20
END_OK = '2026-08-13'                     # 信号f20截止 (tail2同)
RNG_SEED = 20260913

# ==== 预置闸 ====
GATE_IC = 0.030
GATE_POS = 0.60
GATE_YEARS = 5            # D6: 6年中≥5年非负
GATE_ANCHORS = 40         # D1: 有效锚点数下限 (~67个月)
GATE_COVER = 0.50         # D1: 有限值占比下限
GATE_DEDUP = 0.95         # D3: 去重corr阈值
GATE_DIFF = 0.0010        # E: +0.10pp
GATE_PAIR_POS = 0.55
GATE_PAIR_YEARS = 4
N_TOP = 6
CSCV_NULL_K = 40


def log(msg):
    print(msg, flush=True)


# ==================== 1. 面板加载 ====================
def load_panel():
    t0 = time.time()
    fdf = pd.read_parquet(PARQUET)
    skip = {'code', 'date', 'industry', 'future_ret'}
    base_cols = [c for c in fdf.columns
                 if c not in skip and not c.endswith('_rank')]
    fdf = fdf[['code', 'date'] + base_cols + ['future_ret']].copy()
    bad = fdf['code'].astype(str).str.startswith(('8', '43', '92', '399'))
    if bad.any():
        fdf = fdf[~bad]
    piv = fdf.set_index(['code', 'date']).sort_index()
    assert piv.index.is_unique, 'factor_df存在重复(code,date)'
    panel_codes = list(piv.index.get_level_values('code').unique())
    dates = piv.index.get_level_values('date').unique()
    log(f'[1] 面板: {len(panel_codes)}只 × {len(dates)}日期 '
        f'({dates.min():%Y-%m-%d}~{dates.max():%Y-%m-%d}), '
        f'{len(base_cols)}生产基础列, {time.time()-t0:.0f}s')
    panels = {}
    for c in base_cols + ['future_ret']:
        panels[c] = piv[c].unstack('date').values.astype(np.float64)
    log(f'[1] unstack完成 {time.time()-t0:.0f}s, '
        f'rss={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6:.1f}GB')
    return base_cols, dates, panels, panel_codes


# ==================== 2. 收盘矩阵与f20标签 ====================
def build_close_and_f20(dates):
    """tail2同法: 指数日历D, close矩阵T×C, 每股票20日前瞻收益"""
    t0 = time.time()
    idx = pd.read_csv(IDX, usecols=['datetime'], parse_dates=['datetime'])
    # 日历须覆盖全grid + fwd20前瞻 (grid从2020-01-10起, 尾部到2026-09-10+20日)
    idx = idx[(idx.datetime >= '2019-12-01') & (idx.datetime <= '2026-10-15')]
    D = idx['datetime'].values.astype('datetime64[ns]')
    T = len(D)
    codes = []
    for fn in sorted(os.listdir(BT)):
        if not fn.endswith('_qfq.csv'):
            continue
        c = fn[:-len('_qfq.csv')]
        if c.startswith(('sh', 'sz')) or c.startswith(('4', '8', '92', '399')):
            continue
        codes.append(c)
    colmap = {c: i for i, c in enumerate(codes)}
    close = np.full((T, len(codes)), np.nan, dtype=np.float32)
    for i, c in enumerate(codes):
        try:
            df = pd.read_csv(os.path.join(BT, f'{c}_qfq.csv'),
                             usecols=['datetime', 'close'])
        except Exception:
            continue
        dt = df['datetime'].values.astype('datetime64[ns]')
        pos = np.searchsorted(D, dt)
        pos = pos[pos < T]
        if len(pos) == 0:
            continue
        close[pos, i] = df['close'].values[:len(pos)].astype(np.float32)
    close = pd.DataFrame(close).ffill(axis=0).values
    tmap = {d: i for i, d in enumerate(D)}
    gpos = np.array([tmap[d] for d in dates if d in tmap])
    gdates = np.array([d for d in dates if d in tmap])
    assert len(gpos) == len(dates), f'{len(dates)-len(gpos)}个grid日不在指数日历'
    labels = {}
    for h in (10, 20):
        mat = np.full((len(gdates), len(codes)), np.nan, dtype=np.float64)
        ok = gpos + h < T
        mat[ok] = close[gpos[ok] + h] / close[gpos[ok]] - 1.0
        labels[h] = mat
    log(f'[2] 收盘矩阵 {T}天×{len(codes)}只, f10/f20标签完成 '
        f'{time.time()-t0:.0f}s')
    return D, codes, colmap, labels


# ==================== 3. 新OHLCV基础列 ====================
# 全部在完整日频序列上计算(与生产因子同语义), 再采样到grid
NEW_SPECS = [
    ('cp5',   lambda df: (df.c - df.l) / (df.h - df.l + 1e-9) - 0.5),
    ('ush5',  lambda df: (df.h - df[['o', 'c']].max(axis=1))
     / (df.h - df.l + 1e-9)),
    ('lsh5',  lambda df: (df[['o', 'c']].min(axis=1) - df.l)
     / (df.h - df.l + 1e-9)),
    ('body5', lambda df: (df.c - df.o).abs() / (df.h - df.l + 1e-9)),
    ('vwdev', lambda df: df.c * df.v / (df.a + 1e-9) - 1),
    ('tail',  lambda df: (df.c * df.v / (df.a + 1e-9) - 1)
     * np.sign(df.c - df.o)),
    ('effr',  lambda df: (df.c - df.c.shift(20)).abs()
     / (df.c.diff().abs().rolling(20, min_periods=10).sum() + 1e-9)),
    ('upv',   lambda df: df.r.clip(lower=0).rolling(20, min_periods=10).std()
     / (df.r.rolling(20, min_periods=10).std() + 1e-9)),
    ('dnv',   lambda df: df.r.clip(upper=0).rolling(20, min_periods=10).std()
     / (df.r.rolling(20, min_periods=10).std() + 1e-9)),
    ('vcrv',  lambda df: df.r.rolling(5, min_periods=3).std()
     / (df.r.rolling(20, min_periods=10).std() + 1e-9) - 1),
    ('gap5',  lambda df: (df.o / df.c.shift(1) - 1).fillna(0)
     .rolling(5, min_periods=3).sum()),
    ('marea', lambda df: df.r.rolling(5, min_periods=3).mean()
     - df.r.rolling(20, min_periods=10).mean()),
    ('rasym', lambda df: df.r.rolling(20, min_periods=10).mean()
     / (df.r.rolling(20, min_periods=10).std() + 1e-9)),
    ('ampsh', lambda df: df.amp.rolling(5, min_periods=3).mean()
     / (df.amp.rolling(20, min_periods=10).mean() + 1e-9) - 1),
    ('hilo20', lambda df: (df.c - df.l.rolling(20, min_periods=10).min())
     / (df.h.rolling(20, min_periods=10).max()
        - df.l.rolling(20, min_periods=10).min() + 1e-9)),
    ('vz',    lambda df: (df.v - df.v.rolling(20, min_periods=10).mean())
     / (df.v.rolling(20, min_periods=10).std() + 1e-9)),
    ('pvv20', lambda df: df.pvv.rolling(5, min_periods=3).mean()
     / (df.pvv.rolling(20, min_periods=10).mean() + 1e-9) - 1),
    ('tacc',  lambda df: df.t.diff(5) / (df.t.abs().clip(0.1) * 5)),
]
SMOOTH5 = {'cp5', 'ush5', 'lsh5', 'body5', 'vwdev', 'tail'}


def build_new_bases(dates, code_list):
    """单遍qfq CSV → 18个新基础列面板 (code × dates)"""
    t0 = time.time()
    arrs = {name: np.full((len(code_list), len(dates)), np.nan)
            for name, _ in NEW_SPECS}
    D64 = dates.values.astype('datetime64[ns]')
    for i, code in enumerate(code_list):
        path = os.path.join(BT, f'{code}_qfq.csv')
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_csv(path, usecols=['datetime', 'open', 'high', 'low',
                                            'close', 'volume', 'amount',
                                            'turnover_rate'])
        except Exception:
            continue
        dt = df['datetime'].values.astype('datetime64[ns]')
        pos = np.searchsorted(dt, D64)
        pos_c = np.clip(pos, 0, len(dt) - 1)
        m = (pos < len(dt)) & (dt[pos_c] == D64)
        if m.sum() < 60:
            continue
        pi = pos[m]
        df = df.rename(columns={'open': 'o', 'high': 'h', 'low': 'l',
                                'close': 'c', 'volume': 'v',
                                'amount': 'a', 'turnover_rate': 't'})
        for c in ('o', 'h', 'l', 'c', 'v', 'a', 't'):
            df[c] = pd.to_numeric(df[c], errors='coerce')
        df['r'] = df.c.pct_change()
        df['amp'] = (df.h - df.l) / df.c.shift(1)
        df['pvv'] = df.v * df.r.abs()
        for name, fn in NEW_SPECS:
            try:
                raw = fn(df)
            except Exception:
                continue
            if name in SMOOTH5:
                raw = raw.rolling(5, min_periods=3).mean()
            arrs[name][i, m] = raw.values[pi]
    log(f'[3] 新基础列 {len(NEW_SPECS)}个完成 {time.time()-t0:.0f}s '
        f'(有限占比: ' +
        ', '.join(f'{k}={100*np.isfinite(v).mean():.0f}%'
                  for k, v in arrs.items()) + ')')
    return arrs


# ==================== 4. 操作符 ====================
def _shift(x, d):
    out = np.full_like(x, np.nan)
    if d > 0:
        out[:, d:] = x[:, :-d]
    else:
        out[:, :d] = x[:, -d:]
    return out


def _roll(x, w, fn_name):
    """沿日期轴(axis=1)滚动: 用转置实现 (本venv的rolling不支持axis参数)"""
    df = pd.DataFrame(x)
    r = df.T.rolling(w, min_periods=max(2, w // 2))
    return getattr(r, fn_name)().T.values


OPS_PARAM = {
    'delay': lambda x, d: _shift(x, d),
    'delta': lambda x, d: x - _shift(x, d),
    'roc': lambda x, d: x / (_shift(x, d) + 1e-9) - 1,
    'mean': lambda x, w: _roll(x, w, 'mean'),
    'std': lambda x, w: _roll(x, w, 'std'),
    'max': lambda x, w: _roll(x, w, 'max'),
    'min': lambda x, w: _roll(x, w, 'min'),
    'sum': lambda x, w: _roll(x, w, 'sum'),
    'skew': lambda x, w: _roll(x, w, 'skew'),
    'kurt': lambda x, w: _roll(x, w, 'kurt'),
    'zscore': lambda x, w: (x - _roll(x, w, 'mean'))
    / (_roll(x, w, 'std') + 1e-9),
    'mratio': lambda x, w: x / (_roll(x, w, 'mean') + 1e-9) - 1,
}
OPS_UNARY = {
    'sign': np.sign, 'tanh': np.tanh, 'abs': np.abs, 'neg': np.negative,
}
OPS_BIN = {
    'add': lambda a, b: a + b,
    'sub': lambda a, b: a - b,
    'mul': lambda a, b: np.clip(a * b, -30, 30),
    'div': lambda a, b: np.clip(a / (b + 1e-9), -30, 30),
    'max2': np.maximum, 'min2': np.minimum,
}
W_PARAM = {'delay': [1, 2, 3, 5, 10, 20], 'roc': [1, 2, 3, 5, 10, 20],
           'delta': [1, 2, 3, 5, 10, 20]}
W_ROLL = {'mean': [3, 5, 10, 20, 60], 'std': [5, 10, 20, 60],
          'max': [5, 10, 20], 'min': [5, 10, 20], 'sum': [5, 10, 20],
          'skew': [10, 20, 60], 'kurt': [20, 60], 'zscore': [5, 10, 20, 60],
          'mratio': [3, 5, 10, 20]}


def evl(tree, panels):
    if isinstance(tree, str):
        return panels[tree]
    op = tree[0]
    if op in OPS_PARAM:
        return OPS_PARAM[op](evl(tree[1], panels), tree[2])
    if op in OPS_UNARY:
        return OPS_UNARY[op](evl(tree[1], panels))
    return OPS_BIN[op](evl(tree[1], panels), evl(tree[2], panels))


def tname(tree):
    if isinstance(tree, str):
        return tree
    parts = [str(tree[0])] + [tname(x) if not isinstance(x, (int, float))
                              else str(x) for x in tree[1:]]
    return '(' + '_'.join(parts) + ')'


# ==================== 5. 表达式生成 ====================
def canonical_exprs():
    ts = {}
    nb = ['cp5', 'ush5', 'lsh5', 'body5', 'vwdev', 'tail', 'effr', 'upv',
          'dnv', 'vcrv', 'gap5', 'marea', 'rasym', 'ampsh', 'hilo20', 'vz',
          'pvv20', 'tacc']
    for b in nb:
        ts[f'd5_{b}'] = ('delta', b, 5)
        ts[f'd10_{b}'] = ('delta', b, 10)
        ts[f'z10_{b}'] = ('zscore', b, 10)
        ts[f'z20_{b}'] = ('zscore', b, 20)
        ts[f'mr20_{b}'] = ('mratio', b, 20)
    pairs = [('smart_money_flow', 'cp5'), ('smart_money_flow', 'tail'),
             ('wash_sale_score', 'pvv20'), ('low_downside', 'rasym'),
             ('volatility', 'effr'), ('volume_ratio', 'ampsh'),
             ('turnover_shrink', 'vz'), ('limit_pullback_score', 'gap5'),
             ('gap_breakout_confirm', 'hilo20'),
             ('vol_opening_confirm', 'tail'), ('residual_momentum', 'cp5'),
             ('short_reversal', 'body5'), ('exhaustion_risk', 'ush5'),
             ('stroke_phase', 'lsh5'), ('consolidation_breakout', 'ampsh'),
             ('relative_strength', 'vcrv'), ('ema20_slope', 'effr'),
             ('trend_lowvol', 'pvv20'), ('momentum_reversal', 'upv'),
             ('inv_turnover', 'tacc')]
    for a, b in pairs:
        ts[f'mul_{a}x{b}'] = ('mul', a, b)
        ts[f'div_{a}/{b}'] = ('div', a, b)
        ts[f'sub_{a}-{b}'] = ('sub', a, b)
    cross = [('sub', 'tail', 'ush5'), ('mul', 'cp5', 'vz'),
             ('mul', 'hilo20', 'effr'), ('sub', 'vwdev', 'lsh5'),
             ('mul', 'cp5', 'pvv20'), ('mul', 'tail', 'pvv20'),
             ('sub', 'gap5', 'vcrv'), ('mul', 'marea', 'upv'),
             ('div', 'rasym', 'vcrv'), ('mul', 'hilo20', 'vz'),
             ('mul', 'tacc', 'ampsh'), ('sub', 'upv', 'dnv')]
    for op, a, b in cross:
        ts[f'{op}_{a}x{b}'] = (op, a, b)
    tri = [
        ('zscore', ('mul', 'smart_money_flow', 'cp5'), 10),
        ('mratio', ('mul', 'smart_money_flow', 'tail'), 20),
        ('delta', ('mul', 'wash_sale_score', 'pvv20'), 5),
        ('zscore', ('sub', 'tail', 'vcrv'), 20),
        ('mean', ('div', 'limit_pullback_score', 'ampsh'), 10),
        ('mratio', ('mul', 'low_downside', 'rasym'), 10),
        ('delta', ('mul', 'volatility', 'effr'), 5),
        ('zscore', ('mul', 'gap_breakout_confirm', 'hilo20'), 10),
        ('mratio', ('mul', 'turnover_shrink', 'vz'), 20),
        ('delta', ('mul', 'consolidation_breakout', 'ampsh'), 10),
        ('zscore', ('mul', 'cp5', 'vz'), 10),
        ('mean', ('sub', 'tail', 'ush5'), 5),
        ('mratio', ('mul', 'hilo20', 'effr'), 20),
        ('delta', ('mul', 'tacc', 'ampsh'), 10),
        ('zscore', ('mul', 'marea', 'upv'), 20),
        ('mratio', ('div', 'rasym', 'vcrv'), 10),
    ]
    for i, t in enumerate(tri):
        ts[f'tri{i:02d}'] = t
    return ts


def rand_exprs(n, bases, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    out = {}
    attempts = 0
    while len(out) < n and attempts < n * 40:
        attempts += 1
        depth = int(rng.integers(2, 4))

        def gen(d):
            if d <= 1 or rng.random() < 0.4:
                return rng.choice(bases)
            u = rng.random()
            if u < 0.55:
                op = rng.choice(list(OPS_PARAM))
                ws = W_ROLL[op] if op in W_ROLL else W_PARAM[op]
                return (op, gen(d - 1), int(rng.choice(ws)))
            if u < 0.85:
                return (rng.choice(list(OPS_BIN)), gen(d - 1), gen(d - 1))
            return (rng.choice(list(OPS_UNARY)), gen(d - 1))

        t = gen(depth)
        nm = f'r{len(out):03d}_{tname(t)[:60]}'
        if nm not in out:
            out[nm] = t
    return out


# ==================== 6. IC评估 ====================
def prep_anchors(dates, f20_label):
    """锚点=每月首个grid日; 预计算每锚点市场调整f20 (raw, 排名留给eval_ic)"""
    months = pd.date_range(START, END, freq='MS')
    gmap = {pd.Timestamp(d): j for j, d in enumerate(dates)}
    anchors = []
    for m in months:
        mm = pd.Timestamp(m)
        if mm in gmap:
            anchors.append((mm, gmap[mm]))
        else:
            j = np.searchsorted(dates, m)
            if j < len(dates) and dates[j] < m + pd.DateOffset(months=1):
                anchors.append((dates[j], j))
    ys = []
    for _, j in anchors:
        y = f20_label[:, j].copy()
        m = np.isfinite(y)
        if m.sum() < 50:
            ys.append(None)
            continue
        y[m] -= np.nanmedian(y)      # 市场调整
        ys.append(y)
    log(f'[6] 锚点 {len(anchors)}个, 有效(≥50只) '
        f'{sum(1 for y in ys if y is not None)}个')
    return anchors, ys


def eval_ic(arr, anchors, ys):
    """arr: code×dates → 每锚点Spearman IC
    x与y都在共同有效子集上排名(与blindspot探针spearmanr逐点等价, 1e-9级)"""
    ics = []
    for (m, j), y in zip(anchors, ys):
        if y is None:
            continue
        x = arr[:, j]
        mm = np.isfinite(x) & np.isfinite(y)
        if mm.sum() < 50:
            continue
        rx = rankdata(x[mm])
        ry = rankdata(y[mm])
        rx = rx - rx.mean()
        ry = ry - ry.mean()
        den = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
        if den == 0:
            continue
        ics.append((m, float((rx * ry).sum() / den)))
    if not ics:
        return None
    df = pd.DataFrame(ics, columns=['month', 'ic'])
    yr = df.groupby(df.month.dt.year)['ic'].mean()
    return {
        'n': len(df), 'mean_ic': df.ic.mean(),
        'ir': df.ic.mean() / df.ic.std(),
        'pos_share': (df.ic > 0).mean(), 'yearly': yr,
    }


def identity_rank_ic():
    """identity烟测: eval_ic管线 vs scipy.spearmanr 单锚点一致性
    (共同有效子集上双方各自排名, 须<1e-9级一致)"""
    rng = np.random.default_rng(1)
    x = rng.standard_normal((300, 2))
    x[::7, 1] = np.nan
    y = x[:, 1].copy()
    m = np.isfinite(y)
    y[m] -= np.median(y[m])
    mine = eval_ic(x, [(pd.Timestamp('2021-01-04'), 0)], [y])
    mm = np.isfinite(x[:, 0]) & m
    ref = spearmanr(x[mm, 0], x[mm, 1])[0]
    assert abs(mine['mean_ic'] - ref) < 1e-9, \
        f'rank-IC实现偏差: {mine["mean_ic"]} vs {ref}'
    log(f'[id] eval_ic vs spearmanr一致 (Δ<1e-9, '
        f'mine={mine["mean_ic"]:.6f} ref={ref:.6f})')


# ==================== 7. CSCV家族 ====================
def cscv_family(rows, years=range(2021, 2027)):
    """rows: [{name, yearly(Series)}...] → M矩阵 → PBO vs 零带"""
    M, names = [], []
    for r in rows:
        yr = r['yearly']
        if len(yr) < 6:
            continue
        names.append(r['name'])
        M.append([yr.get(y, np.nan) for y in years])
    M = np.array(M)
    keep = np.isfinite(M).all(axis=1)
    M, names = M[keep], [n for n, k in zip(names, keep) if k]
    if len(M) < 10:
        log(f'[8] CSCV家族行数不足({len(M)}), 跳过')
        return None
    res = cscv.cscv_pbo(M, n_combos=None, seed=42)
    null = cscv.cscv_null(N=len(M), S=6, K=CSCV_NULL_K, n_combos=20, seed=0)
    band = (np.percentile(null, 5), np.percentile(null, 95))
    log(f'[8] CSCV家族: N={len(M)}×S=6, PBO={res["pbo"]*100:.1f}% vs '
        f'零带5-95分位[{band[0]*100:.1f},{band[1]*100:.1f}]% '
        f'(均值{null.mean()*100:.1f}%)')
    where = ('低于5分位(选优稳健)' if res['pbo'] < band[0]
             else '高于95分位(病态)' if res['pbo'] > band[1]
             else '带内(空间似噪声)')
    log(f'[8] 判读: {where}')
    return M, names, res, band


# ==================== main ====================
def main():
    t0 = time.time()
    log('=' * 70)
    log('alpha_factory 2026-09-13 — 缺口1: 老数据新表达式挖掘')
    log(f'预置闸: D4 池IC≥{GATE_IC:.3f} D5 正比例≥{GATE_POS:.0%} '
        f'D6 ≥{GATE_YEARS}/6年 E top{N_TOP}配对≥+{GATE_DIFF*100:.2f}pp/'
        f'{GATE_PAIR_POS:.0%}/{GATE_PAIR_YEARS}/6年')
    identity_rank_ic()

    base_cols, dates, panels, panel_codes = load_panel()
    D, codes, colmap, labels = build_close_and_f20(dates)
    f10_mat, f20_mat = labels[10], labels[20]
    # 标签按panel行序(code_list)重排
    n_codes = len(panel_codes)
    panels = {k: v[:n_codes] for k, v in panels.items()}
    f20_label = np.full((n_codes, len(dates)), np.nan)
    f10_label = np.full((n_codes, len(dates)), np.nan)
    for i, c in enumerate(panel_codes):
        if c in colmap:
            f20_label[i] = f20_mat[:, colmap[c]]
            f10_label[i] = f10_mat[:, colmap[c]]
    log(f'[2] f20标签面板 {f20_label.shape}, '
        f'有限占比{100*np.isfinite(f20_label).mean():.1f}%')
    # identity: 生产future_ret(config dynamic_factor.forward_period=10 → f10)
    # vs 独立CSV计算的f10/f20 — f10须corr≈1(同一标签口径), f20按预期低
    fut = panels['future_ret']
    m10 = np.isfinite(fut) & np.isfinite(f10_label)
    m20 = np.isfinite(fut) & np.isfinite(f20_label)
    if m10.sum() > 1000:
        c10 = np.corrcoef(fut[m10], f10_label[m10])[0, 1]
        d10 = np.abs(fut[m10] - f10_label[m10])
        c20 = np.corrcoef(fut[m20], f20_label[m20])[0, 1]
        log(f'[id] future_ret vs CSV标签: f10 corr={c10:.4f} '
            f'中位差{np.median(d10):.4f}; f20 corr={c20:.4f}')
        assert c10 > 0.99, f'f10 identity失败 corr={c10}'
        log('[id] PASS: 生产future_ret=f10(口径一致), 本厂标签=f20(探针约定)')
    panels['future_ret'] = f20_label  # 以独立计算标签为准
    newb = build_new_bases(dates, panel_codes)
    panels.update(newb)
    all_bases = base_cols + [n for n, _ in NEW_SPECS]
    log(f'[4] 基础列合计 {len(all_bases)} '
        f'(生产{len(base_cols)} + 新{len(NEW_SPECS)})')

    canon = canonical_exprs()
    exprs = dict(canon)
    exprs.update(rand_exprs(180, all_bases))
    log(f'[5] 表达式: canonical {len(canon)} + 随机 '
        f'{len(exprs)-len(canon)} = {len(exprs)}')

    anchors, ys = prep_anchors(dates, f20_label)
    sub = np.arange(0, len(dates), 20)   # 去重参考子采样列
    accepted_arrays = []
    rows, survivors = [], {}
    t_evl = time.time()
    for k, (name, tree) in enumerate(exprs.items()):
        try:
            arr = evl(tree, panels)
        except Exception as e:
            log(f'  [eval跳过] {name}: {e}')
            continue
        if np.isfinite(arr).mean() < GATE_COVER:
            continue
        if np.nanstd(arr) <= 1e-8:
            continue
        xs = arr[:, sub]
        dup = False
        for acc in accepted_arrays:
            dd = pd.DataFrame(np.column_stack([xs.ravel(),
                                               acc.ravel()])).dropna()
            if len(dd) > 30 and abs(dd[0].corr(dd[1])) > GATE_DEDUP:
                dup = True
                break
        if dup:
            continue
        accepted_arrays.append(xs)
        r = eval_ic(arr, anchors, ys)
        if r is None or r['n'] < GATE_ANCHORS:
            continue
        rows.append({'name': name, **r})
        ok_d = (r['mean_ic'] >= GATE_IC and r['pos_share'] >= GATE_POS
                and (r['yearly'] >= 0).sum() >= GATE_YEARS)
        if ok_d:
            survivors[name] = arr
            log(f'  [幸存] {name[:60]:60s} IC={r["mean_ic"]:+.4f} '
                f'IR={r["ir"]:+.2f} 正{r["pos_share"]:.0%} n={r["n"]}')
        if (k + 1) % 50 == 0:
            log(f'  [进度] {k+1}/{len(exprs)} '
                f'({time.time()-t_evl:.0f}s, 收{len(rows)} '
                f'幸存{len(survivors)})')
    log(f'[6] 评估完成: {len(rows)}过D1-D3, 幸存(D4-D6全过) '
        f'{len(survivors)}个')
    if rows:
        df_rows = pd.DataFrame(rows).sort_values('mean_ic', ascending=False)
        log('\n[6] IC前10:')
        for _, r in df_rows.head(10).iterrows():
            yr = '  '.join(f'{y}:{r.yearly.get(y, np.nan):+.3f}'
                           for y in range(2021, 2027))
            log(f'  {r["name"][:55]:55s} IC={r["mean_ic"]:+.4f} '
                f'IR={r["ir"]:+.2f} 正{r["pos_share"]:.0%}\n      {yr}')
    else:
        df_rows = pd.DataFrame(columns=['name'])

    # ==== E 配对阶段 (幸存者 only) ====
    pair_ok = {}
    if survivors:
        t_p = time.time()
        sig = pd.read_csv(SIG, usecols=['code', 'date', 'buy', 'score'],
                          dtype={'code': str})
        sig = sig[sig.buy == True].copy()
        sig['code'] = sig['code'].str.zfill(6)
        sig = sig[~sig['code'].str.startswith(('4', '8', '92', '399'))]
        sig['date'] = pd.to_datetime(sig['date'])
        sig = sig[(sig.date >= '2021-04-01') & (sig.date <= END_OK)]
        tmap = {d: i for i, d in enumerate(D)}
        sig = sig[sig.date.isin(tmap)]
        gd = dates.values.astype('datetime64[ns]')
        sd = sig['date'].values.astype('datetime64[ns]')
        gj = np.searchsorted(gd, sd, side='right') - 1
        age = (sd - gd[np.clip(gj, 0, len(gd) - 1)]) / np.timedelta64(1, 'D')
        sig['gj'] = gj
        sig['age'] = age
        sig = sig[(sig.gj >= 0) & (sig.age <= 10)]
        cmap = {c: i for i, c in enumerate(panel_codes)}
        sig = sig[sig.code.isin(cmap)]
        ci0 = sig['code'].map(cmap).values
        sig['f20'] = f20_label[ci0, sig['gj'].values]
        sig = sig[np.isfinite(sig.f20)]
        ci = sig['code'].map(cmap).values        # 过滤后重算行索引
        gj_i = sig['gj'].values.astype(int)
        log(f'[7] 信号池 {len(sig)}行 {sig.date.nunique()}日, '
            f'{time.time()-t_p:.0f}s')
        for name, arr in survivors.items():
            xv = arr[ci, gj_i]
            mm = np.isfinite(xv) & np.isfinite(sig.f20)
            if mm.sum() < 1000:
                log(f'  [E跳过] {name[:50]}: 有效行{mm.sum()}')
                continue
            s = sig[mm].copy()
            s['x'] = xv[mm]
            diffs = []
            for d, g in s.groupby('date'):
                if len(g) < max(10, N_TOP):
                    continue
                tx = g.nlargest(N_TOP, 'x').f20.mean()
                ts = g.nlargest(N_TOP, 'score').f20.mean()
                if np.isfinite(tx) and np.isfinite(ts):
                    diffs.append((d, tx - ts))
            if len(diffs) < 60:
                continue
            df_d = pd.DataFrame(diffs, columns=['date', 'diff'])
            yr = df_d.groupby(df_d.date.dt.year)['diff'].mean()
            md = df_d['diff'].mean()
            pos = (df_d['diff'] > 0).mean()
            yok = (yr >= 0).sum()
            ok_e = (md >= GATE_DIFF and pos >= GATE_PAIR_POS
                    and yok >= GATE_PAIR_YEARS)
            ystr = '  '.join(f'{y}:{yr.get(y, np.nan)*100:+.2f}pp'
                             for y in range(2021, 2027))
            log(f'  [E{"过" if ok_e else "否"}] {name[:45]:45s} '
                f'{md*100:+.2f}pp 正{pos:.0%} {yok}/6年\n      {ystr}')
            if ok_e:
                pair_ok[name] = arr
    else:
        log('[7] 无幸存者, 跳过配对')

    # ==== F CSCV家族 ====
    cscv_out = cscv_family(rows)
    if cscv_out is not None:
        M, names, res, band = cscv_out
        pd.DataFrame(M, index=names,
                     columns=[str(y) for y in range(2021, 2027)]
                     ).to_csv(OUT_MAT)
        log(f'[8] 矩阵已存 {OUT_MAT}')

    # ==== 输出 ====
    df_rows.to_csv(OUT_CSV, index=False)
    log(f'[8] IC明细已存 {OUT_CSV}')
    log(f'\n=== 终局: D4-D6幸存 {len(survivors)}个, '
        f'E配对全过 {len(pair_ok)}个 ===')
    for k in pair_ok:
        log(f'  全过: {k}')
    log(f'总耗时 {time.time()-t0:.0f}s, '
        f'rss={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6:.1f}GB')
    log('(接线动作: 仅当E全过且家族PBO<5分位才进入讨论, 本脚本只读)')


if __name__ == '__main__':
    main()
