#!/usr/bin/env python
"""表驱动复权重建 (2026-10-16) — 替代 `rebuild_mult_qfq.py` 的"从价格反推事件"

**为什么**: `rebuild_mult_qfq.py` 从 `Δqfq = a·Δraw` 反推公司行动幅度, 但东财 qfq 序列
  比 raw 小很多(002706: qfq≈4.76 vs raw≈52.47, 小 11 倍) ⇒ 2 位小数舍入误差被放大 11 倍
  ⇒ 送转比例估错(10转10 估成 2.30 而非 2.00)。
  实测(`/tmp/probe_arbiter.py`, 官方分红表裁判):
    送转日  n=178   我们 1.232%  vs  东财加性 0.092%  vs  新浪 0.030%
    非事件日       我们 0.0007%  vs  东财加性 0.0124%
  ⇒ 分红修好了, **送转搞砸了**。

**怎么做**: 不再反推。直接读 `data/alternative_data/corporate_actions.pkl`
  (由 `fetch_actions_table.py` 从 `ak.stock_fhps_detail_em` 冻结), 前向构造财富指数:
    W(0) = raw(0)
    W(t) = W(t−1) · ((1+s_t)·raw(t) + D_t) / raw(t−1)      t 是除权日
         = W(t−1) · raw(t) / raw(t−1)                      否则
    P(t) = W(t) · k                                         k = 锚因子

**锚因子 k 与铁律(2026-10-16 用户裁定: 更新今天不得改写历史)**:
  收益率 P(t)/P(t−1) 与 k 无关 ⇒ k 只决定**绝对价位**, 不影响信号。
  但价位影响 **A 股一手=100 股的手数取整粒度** ⇒ k 必须让价格≈真实价。
  (⚠ 不能改成交价用 raw: 分红/送转的收益正是体现在复权序列里, 按 raw 成交会**丢掉分红收益**。
   全程用复权价才是正确的总收益记账。)

  三种锚:
    --anchor frozen (生产默认): k = raw(T0)/W(T0), **T0=首次构建日, 持久化后永不变**
        ✓ 满足铁律 f(t;T)≡f(t;t): 值永久冻结, 与样本终点 T 无关
        ✓ 切换日价格 ≡ 真实价 ⇒ 手数取整忠实; 之后按事后除权缓慢漂移(可量化, 有界)
        ⚠ 2026-10-17 修: 旧实现在"锚漂移"时 `k = k_cur` **采用新锚**并写回磁盘,
          把 frozen 降级成 latest。触发条件不是"数据修正", 而是**新交易日恰为除权日**
          (那天分子分母不同乘) ⇒ 10转10 直接把整条历史砍半。详见 build_one 内注释。
    --anchor first  : k=1 ⇒ P ≡ W。值冻结但高价股价格被放大数倍(实测 001309 达 3.6×)
        ⇒ 手数取整严重失真(仓位被腰斩), **不用于生产**, 仅作"值冻结"的对照
    --anchor latest : k = raw(T)/W(T) 每次构建重算 ⇒ **违反铁律**(全部历史价位随 T 缩放), 仅对照

  k 落盘: `data/stock_data/qfq_anchor.json`。**该文件一旦生成即为合约, 不得重算**
  (重算 = 改写历史 = 违反铁律)。新增股票自动补录。
  s_t = 每股送转股数(表: 每10股送转Y ⇒ s=Y/10), D_t = 每股现金分红(税前, 与交易所除权参考价口径一致)。

优点: 送转/分红幅度精确 / 可复现(表已冻结) / **样本前分红自动免疫**(前向从样本首日起算)。

用法:
  python rebuild_table_qfq.py                    # 全市场
  python rebuild_table_qfq.py --only 002706,600519
  python rebuild_table_qfq.py --out /path/dir
"""
import os
import sys
import glob
import json
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from market_codes import is_stock_code, INDEX_REAL        # noqa: E402

PROJ = '/mnt/d/quant'
RAW = f'{PROJ}/data/stock_data/raw_data'
BT = f'{PROJ}/data/stock_data/backtrader_data'
CA = f'{PROJ}/data/alternative_data/corporate_actions.pkl'
OUT = f'{PROJ}/data/stock_data/backtrader_data_tbl'
ANCHOR_FILE = f'{PROJ}/data/stock_data/qfq_anchor.json'   # frozen 锚的冻结因子(一旦生成即合约)

COLS = ['datetime', 'open', 'high', 'low', 'close', 'volume', 'openinterest',
        'amount', 'amplitude', 'change_percent', 'change_amount', 'turnover_rate']
# ⚠ 指数 bt 文件是**另一套列**: 只有前 8 列, 没有 amplitude/change_percent/
#   change_amount/turnover_rate (指数无涨跌幅/换手率语义)。生产旧文件就是这个格式,
#   沿用分支一直原样保着它 ⇒ 显式重建必须**逐列对齐**, 否则换名后指数文件变形,
#   而 bt_execution.py:242 直接读 sh000001_qfq.csv 做 regime 检测。
IDX_COLS = COLS[:8]


def _settle_guard():
    """铁律守卫: raw 的**最后一行**是否仍是"今天未收盘的临时值"。

    实测(2026-10-16, 300只/61.5万共同格, 09-04快照 vs 今天): 294 只有变化, 其中 **293 只
    的变化格全部是旧快照的最后一行** (001309 2026-09-04 405.06→391.30 / 600519
    1312.96→1330.00 / 603929 155.27→150.26); 仅 1 只(603088)有已结算行变, 幅度 0.0039。
    根因 = 当天未收盘的临时价, 下一交易日下载即修正。
    用它建 W ⇒ 次日必被改写 ⇒ f(T;T) ≠ f(T;T+1) ⇒ **违反铁律**。
    生产下载在 19:53~20:36(收盘后) ⇒ 正常流安全; 此处只在"疑似盘中快照"时拒绝构建。
    (拒绝而非修正: 修正=改写历史, 那是另一种违反。)
    """
    p = sorted(glob.glob(f'{RAW}/*/none.csv'))
    if not p:
        return
    # 抽样 200 只取**最大**末行日期 —— 单只可能停牌(末行停在很久以前), 会漏判。
    # 抽样而非全量: 全量读 5000+ 文件约 1 分钟, 而本函数只需要一个"今天是否已有行"的信号。
    lt = max(pd.to_datetime(pd.read_csv(f, encoding='utf-8-sig', usecols=['日期'])['日期']).max()
             for f in p[::max(1, len(p) // 200)])
    now = pd.Timestamp.now()
    if lt.normalize() == now.normalize() and (now.hour, now.minute) < (15, 5):
        raise SystemExit(
            f"[铁律守卫] raw 最后一行 {lt.date()} 是今天且现在 {now:%H:%M} 未收盘 "
            f"⇒ 该行是临时值(实测次日必被修正), 拒绝构建。收盘后重跑。")


def build_one(sym, actions, raw_df, bt_df, anchor='frozen', kfix=None, drift=None):
    """返回 (P_close, nev)。P 的锚因子见模块 docstring 的锚点说明。

    actions: 可迭代的 (ex_date, s, D[, mult]) —— 4 元组的第 4 项是**直接乘数**
      (新浪因子通道, 见 2026-10-17 融合说明): `mult` 非 None 时用它替换
      `((1+s)·raw(t)+D)/raw(t−1)` 的分子口径, 即 `r = mult · raw(t)/raw(t−1)`。
      用于填补东财分红送配表**结构性不覆盖**的事件(配股 / 重整转增 / 特别股息)。

    kfix: frozen 锚的冻结因子 {sym: k}。缺失时由本函数算出并**回写调用方**
          (调用方负责持久化 ⇒ 下次重建逐位复用)。
    drift: 可选 list; 锚因子与重算值不符时追加 (sym, k_old, k_new)。
           **不符 = 事件集变了(数据修正)** ⇒ 采用新锚并上报。稳定态下永不触发。
    """
    dt = pd.to_datetime(raw_df['日期']).values
    raw = raw_df['收盘'].values.astype(float)
    n = len(raw)
    # ⚠ 2026-10-17: 原为 `n < 60` ⇒ 26 只次新股**整只**走"沿用生产"分支, 即**旧加性文件**,
    #   在口径上留了一道 5,224(乘性) + 26(加性) 的混合口子。
    #   代价实测: 001232(n=48, 唯一有事件的次新股, D=0.5) 的 P 全段 = raw − 0.50,
    #   非事件日比值恒等式**逐日违反**(全市场 26 个违规全是它一只)。
    #   其余 25 只 C=0 ⇒ 加性≡raw, 无失真(但口径仍是旧的)。
    #   `n<60` 对**价格文件**没有理由 —— 短序列的兜底在因子层(V41, idx<60), 不在价格层。
    #   降到 n<2 只拒退化文件; 无事件的股重建结果 ≡ 沿用结果(P=W·k=raw, k=1), 零风险。
    if n < 2:
        return None
    # 事件聚合到"第一个 >= 除权日的交易日"(除权日可能落在非交易日/停牌日)
    idx = pd.Series(np.arange(n), index=pd.DatetimeIndex(dt))
    ev = {}
    for act in actions:
        ex, s, D = act[0], act[1], act[2]
        mm = act[3] if len(act) > 3 else None
        pos = idx.index.searchsorted(pd.Timestamp(ex), side='left')
        if pos >= n:
            continue                      # 样本之后 ⇒ 忽略
        key = dt[pos]
        s0, D0, m0 = ev.get(key, (0.0, 0.0, None))
        ev[key] = ((1 + s0) * (1 + s) - 1, D0 + D, mm if mm is not None else m0)
    W = np.empty(n); W[0] = raw[0]
    nev = 0
    for t in range(1, n):
        r = raw[t] / raw[t - 1]
        e = ev.get(dt[t])
        if e is not None and raw[t - 1] > 0:
            s, D, mm = e
            r = (mm * raw[t] / raw[t - 1]) if mm is not None \
                else (((1 + s) * raw[t] + D) / raw[t - 1])
            nev += 1
        W[t] = W[t - 1] * r
    if not np.isfinite(W[-1]) or W[-1] <= 0:
        return None
    # 锚因子 k (三种口径, 详见模块 docstring):
    #   frozen (生产): k 冻结在首次构建日 ⇒ 值永久不变(铁律) 且价格≈真实价(手数取整忠实)
    #   first        : k=1 ⇒ P≡W, 值冻结但高价股价格被放大数倍 ⇒ 手数取整失真, 仅对照
    #   latest       : k=raw(T)/W(T) 每次构建重算 ⇒ 违反铁律, 仅对照
    if anchor == 'first':
        P = W
    elif anchor == 'latest':
        P = W * (raw[-1] / W[-1])
    else:                                   # frozen
        k_cur = float(raw[-1] / W[-1])
        k = kfix.get(sym)
        if k is None:
            k = k_cur
            kfix[sym] = k
        elif abs(k - k_cur) > 1e-9 * max(abs(k), 1e-12):
            # 锚漂移 ⇒ **保留冻结 k, 只报告, 绝不采用 k_cur**(2026-10-17 修)。
            #
            # 旧实现在这里 `k = k_cur; kfix[sym] = k`, 理由写在注释里:
            #   "锚漂移只可能来自数据修正(增删事件)。稳定态下 raw(末)/W(末) 对
            #    '新增一个交易日'不变(分子分母同乘), 故永不触发。"
            # **这个推理漏了"新交易日恰是除权日"**: 那天 W 乘的是
            # ((1+s)·raw(t)+D)/raw(t−1) 而非 raw(t)/raw(t−1), 分子分母不再同乘,
            # ⇒ k_cur 按 `raw(t)/((1+s)·raw(t)+D)` 变。10转10 时该因子 = 0.5
            # ⇒ **整条历史被砍半**。于是 `frozen` 被悄悄降级成 `latest`
            # —— 而 `latest` 正是本文件 docstring 自己标注"违反铁律, 仅对照"的那档。
            #
            # 且 line 311 `old.update(kfix)` 会把漂移值**写回磁盘**, 上面那句
            # "已有键绝不覆盖" 并不成立 ⇒ 冻结合约被销毁, 不可恢复。
            #
            # 保留冻结 k 的代价(诚实记账): P(末) 从此偏离 raw(末), 偏离量 = 锚点日
            # 以来该股的累计事件因子; 影响**回测手数取整**(P 越大 100股/手越粗)。
            # 出单层已改读裸价(generate_trade_orders.py 的 price_scale) ⇒ 实盘不受影响。
            if drift is not None:
                drift.append((sym, k, k_cur))
        P = W * k
    if (P <= 0).any():
        return None
    return P, nev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=None)
    ap.add_argument('--only', default='')
    ap.add_argument('--anchor', choices=['frozen', 'first', 'latest'], default='frozen',
                    help='frozen=锚冻结在首建日(生产默认,满足铁律); first=P≡W 仅对照; latest=每次重锚 仅对照')
    ap.add_argument('--anchor-file', default=ANCHOR_FILE)
    ap.add_argument('--events', default=CA,
                    help='事件表 pickle。默认东财 corporate_actions.pkl (code,ex_date,s,D); '
                         '可传融合表 (…,mult,src) —— mult 非空时按**直接乘数**消费(新浪通道填补)')
    a = ap.parse_args()
    _settle_guard()                     # 铁律守卫: 拒绝用"今天未收盘的临时 raw"构建
    if a.out is None:
        a.out = OUT if a.anchor == 'frozen' else f'{OUT}_{a.anchor}anchor'
    os.makedirs(a.out, exist_ok=True)

    # frozen 锚因子: 读既有(合约, 不重算) → 缺失的股票由 build_one 补算 → 落盘
    kfix = {}
    if a.anchor == 'frozen':
        if os.path.exists(a.anchor_file):
            with open(a.anchor_file) as fh:
                kfix = json.load(fh)
            print(f"锚因子: 载入 {len(kfix)} 只 (冻结于 {a.anchor_file}, 不重算)")
        else:
            print(f"锚因子: 首建, 将生成 {a.anchor_file} (此后为合约, 不得重算)")
    k_new = 0

    ca = pd.read_pickle(a.events)
    ca['code'] = ca['code'].astype(str).str.zfill(6)
    if 'mult' in ca.columns:      # 融合表: (ex_date, s, D, mult) —— 同日多事件时 mult 由首条携带
        ca_by = {}
        for c, g in ca.groupby('code'):
            rows = []
            for r in g.itertuples():
                mm = r.mult if (isinstance(r.mult, float) and np.isfinite(r.mult)) else None
                rows.append((r.ex_date, 0.0 if not np.isfinite(r.s) else r.s,
                             0.0 if not np.isfinite(r.D) else r.D, mm))
            ca_by[c] = rows
        print(f"事件表: {a.events}  {len(ca)} 条 / {ca.code.nunique()} 只"
              f"  (含直接乘数 {int(ca['mult'].notna().sum() - ca['s'].notna().sum())} 条)")
    else:
        ca_by = {c: g[['ex_date', 's', 'D']].values.tolist() for c, g in ca.groupby('code')}
        print(f"事件表: {a.events}  {len(ca)} 条 / {ca.code.nunique()} 只 (s,D 口径)")

    if a.only:
        syms = [s.strip().zfill(6) for s in a.only.split(',')]
    else:
        # 主集: raw_data 有原始价(none.csv)的股票
        syms = sorted(os.path.basename(os.path.dirname(p))
                      for p in glob.glob(f'{RAW}/*/none.csv'))
        syms = [s for s in syms if s[0] in '036']       # 北交所排除(铁律)
        # 指数伪码排除(见 market_codes.py): 它们的 6 位代码与深市股票相同, 新浪会抓到
        # **同代码深市股票**的事件 ⇒ 造出假除权(实测 000851 被塞 1.40×, 历史改写 4044 点)。
        # 排除后落入下方"补集沿用生产文件"分支 ⇒ 与基线逐位相同。
        syms = [s for s in syms if is_stock_code(s)]
        # 补集: **重建没产出的股票一律沿用生产文件**, 保证股票池与基线逐位相同
        #   (a) 118 只退市股 —— raw_data 只有 qfq.csv 无原始价, 且官方分红表查不到
        #       (ak.stock_fhps_detail_em 对退市码返回 None) ⇒ 无法反解 raw
        #   (b) 26 只次新股 —— 样本 <60 天, build_one 主动跳过
        # 目的: 复权口径臂保持**单变量对照**, ΔNAV 只归因于复权方法而非股票池。
        prod_syms = sorted(os.path.basename(p)[:-8] for p in glob.glob(f'{BT}/*_qfq.csv'))

    report = {}
    stats = {}
    drift = []
    for i, sym in enumerate(syms):
        bp = f'{BT}/{sym}_qfq.csv'
        rp = f'{RAW}/{sym}/none.csv'
        if not (os.path.exists(bp) and os.path.exists(rp)):
            stats['missing'] = stats.get('missing', 0) + 1
            continue
        try:
            b = pd.read_csv(bp)
            r = pd.read_csv(rp, encoding='utf-8-sig')
        except Exception as e:
            stats['read_fail'] = stats.get('read_fail', 0) + 1
            continue
        r['日期'] = pd.to_datetime(r['日期'])
        r = r.sort_values('日期').reset_index(drop=True)
        acts = ca_by.get(sym, [])
        got = build_one(sym, acts, r, b, anchor=a.anchor, kfix=kfix, drift=drift)
        if got is None:
            stats['build_fail'] = stats.get('build_fail', 0) + 1
            continue
        P, nev = got
        m = r.rename(columns={'日期': 'datetime'})
        f = P / m['收盘'].values.astype(float)
        out = m[['datetime']].copy()
        for src, dst in [('开盘', 'open'), ('最高', 'high'), ('最低', 'low')]:
            out[dst] = m[src].values.astype(float) * f
        out['close'] = P
        # ⚠ 涨跌幅/涨跌额/振幅 必须按**复权价**重算, 不能照抄裸价的列(2026-10-16 修):
        #   裸价的涨跌幅在除权日含事件跳变(603298 2024-05-31: 裸价 −30.40% vs 复权 −0.83%),
        #   而 bt_execution.py:861-876 用 `change_percent >= 9.5` 判涨停 ⇒ 照抄裸价会**漏判
        #   除权日涨停**(除权日涨停在裸价口径下只显示 1.10·P_ref/raw_prev − 1 ≈ +6.3%)。
        #   生产 qfq 文件本身就是重算的(实测 603298 2024-05-31 change_percent=−0.89 = 自身 close 比值)。
        #   换手率与价格口径无关, 沿用裸价列。
        prev = np.concatenate([[np.nan], P[:-1]])
        with np.errstate(invalid='ignore', divide='ignore'):
            out['change_percent'] = np.round((P / prev - 1) * 100, 2)
            out['change_amount'] = np.round(P - prev, 2)
            out['amplitude'] = np.round((out['high'].values - out['low'].values) / prev * 100, 2)
        out['turnover_rate'] = m['换手率'].values
        # ⚠ 2026-10-17 修: **旧 bt 文件里没有的行(= 本次刷新新增的交易日)必须从 raw 补量额**。
        #   旧实现 `merge(旧bt,[volume,amount])` + `fillna(0)`: 新行匹配不上 ⇒ volume=0 /
        #   amount=NaN, 而下一轮重建又把这份 0 当源 ⇒ **永久固化**。
        #   实测 10-17 刷新后 5,216 只的 2026-10-16 行 volume=0。后果不是"少个统计量":
        #     (a) 因子层: volume_ratio / volume_surge / price_volume_corr_20 /
        #         Amihud |ret|/(close·vol) / volume_contraction 全被 0 污染;
        #         `amount` NaN ⇒ factor_preparer.py:209 的 ln_cap 直接落 0.0;
        #     (b) 执行层: `tradable` 硬闸 `volume_m > 100` ⇒ 该日**整池不可交易**,
        #         而 bt_execution.py:2356 `universe = [c for c in stock_codes if ok[...]]`
        #         ⇒ 当日 universe 为空 ⇒ generate_positions 返回 {} ⇒ `_pending={}`
        #         ⇒ **调仓日被静默跳过**, 且 `_last_rebalance_date` 已被推后 ⇒ 相位错位。
        #   根因是**路径依赖**(产物内容取决于上一轮产物) ⇒ 违反铁律 f(t;T)≡f(t;t)。
        #   ⚠ 修法刻意**不做**"量额一律取自 raw": 旧 bt 文件的 volume 是**派生浮点**
        #     (600519 2016-01-04: 旧 17349.69 vs raw 17350), 一律换源会改动全市场
        #     约 1,300 万格的**既有**值(<1% 但非零) ⇒ 那才是改写历史。
        #   故只补**丢失签名**: 旧 bt 里没有该行, 或旧值 volume==0 / amount 缺失 而 raw 有值。
        #   真实停牌日 raw 也是 0 ⇒ 不会误补。其余既有值逐位保留。
        #   实测爆炸半径(400 只抽样): 变化格 19,314, 其中 1996–2015 占 18,927(退市股旧链路
        #   整段缺量), 2026 占 384(= 10-16 新行), **回测窗口 2021-01-01~2026-10-15 内 0 格**。
        b2 = b.copy()
        b2['datetime'] = pd.to_datetime(b2['datetime'])
        out = out.merge(b2[['datetime', 'volume', 'openinterest', 'amount']],
                        on='datetime', how='left')
        _dtm = pd.to_datetime(out['datetime'])
        _rv = (pd.to_numeric(m['成交量'], errors='coerce').values.astype(float)
               if '成交量' in m.columns else np.full(len(m), np.nan))
        _ra = (pd.to_numeric(m['成交额'], errors='coerce').values.astype(float)
               if '成交额' in m.columns else np.full(len(m), np.nan))
        _rv = pd.Series(_rv, index=pd.DatetimeIndex(pd.to_datetime(m['datetime'])))
        _ra = pd.Series(_ra, index=pd.DatetimeIndex(pd.to_datetime(m['datetime'])))
        _rv = _rv[~_rv.index.duplicated()]
        _ra = _ra[~_ra.index.duplicated()]
        _rv_out = _dtm.map(_rv)
        _ra_out = _dtm.map(_ra)
        _lost_v = (out['volume'].isna() | (out['volume'] == 0)) & (_rv_out > 0)
        _lost_a = out['amount'].isna() & _ra_out.notna()
        if _lost_v.any():
            out.loc[_lost_v, 'volume'] = _rv_out[_lost_v].values
        if _lost_a.any():
            out.loc[_lost_a, 'amount'] = _ra_out[_lost_a].values
        out['volume'] = out['volume'].fillna(0)
        # ⚠ 2026-10-24 修 (navlaw −27% 事故): volume 的 int/float **文本格式**必须跟随
        #   旧 bt 文件 — merge 引入 NaN(新交易日行)会把整列升成 float64 ⇒ to_csv 全列
        #   写 '17521.0' ⇒ **数值零变化但逐字节历史改写** ⇒ ML chunk 字节级指纹全失效
        #   ⇒ 全量重训 (实测 2026-10-23: 002731/002388 格式噪声 × 000016 事件撤回
        #   ⇒ 24选股日翻转 ⇒ NAV 1,321,718→965,332)。
        #   规则: 旧 bt volume 是整型 dtype 且本列全整值 ⇒ 回写 int64 (旧文件本就是
        #   float 格式的股如 002673 不动, 保持其既有格式)。
        if pd.api.types.is_integer_dtype(b2['volume']) \
                and out['volume'].notna().all() \
                and float(np.abs(out['volume'].values
                                 - np.round(out['volume'].values)).max(initial=0.0)) == 0.0:
            out['volume'] = out['volume'].astype('int64')
        out['openinterest'] = out['openinterest'].fillna(0.0)
        out['amount'] = out['amount'].fillna(np.nan)
        if _lost_v.any() or _lost_a.any():
            print(f'  [量额补丢失] {sym}: 量 {int(_lost_v.sum())} 行 / 额 {int(_lost_a.sum())} 行 '
                  f'取自 raw (旧 bt 缺失或为0)', flush=True)
        out['datetime'] = pd.to_datetime(out['datetime']).dt.strftime('%Y-%m-%d')
        for c in ['open', 'high', 'low', 'close']:
            out[c] = out[c].round(2)
        out = out[COLS]
        out.to_csv(os.path.join(a.out, f'{sym}_qfq.csv'), index=False)
        report[sym] = dict(nev=nev, nact=len(acts),
                           amp=round(float(np.std(np.diff(P) / P[:-1]) /
                                           max(np.std(np.diff(m['收盘'].values.astype(float)) /
                                                      m['收盘'].values.astype(float)[:-1]), 1e-12)), 4))
        if i % 500 == 0:
            print(f'  ...{i}/{len(syms)}', flush=True)

    n_idx = ncarry = 0
    if not a.only:
        # ⚠ 原地重建守卫必须**先**跑: 下面两个分支都往 a.out 写文件, 守卫若排在写之后
        #   就成了"先破坏再报错"。--out == BT 时沿用分支 `open(dst,'wb')` 会先把源
        #   截断成 0 字节再读 ⇒ 文件被清空。必须走"建到临时目录 + 原子换名"。
        _inplace = [s for s in prod_syms
                    if os.path.abspath(f'{BT}/{s}_qfq.csv') ==
                    os.path.abspath(os.path.join(a.out, f'{s}_qfq.csv'))]
        if _inplace:
            raise SystemExit(
                f"[接线守卫] --out 与生产目录 {BT} 相同 ⇒ 沿用生产文件会把源文件截断成 0 字节"
                f" (涉及 {len(_inplace)} 只, 如 {_inplace[:3]})。请建到临时目录后原子换名。")

        # ── 真指数: 必须**显式重建**, 否则落入下方"沿用生产"分支被永久冻结 ──────
        # 指数无公司行动 ⇒ P ≡ raw(不做任何复权)。但 raw 目录里是 `qfq.csv` 而**不是**
        # `none.csv` ⇒ 不进 `syms`; 且 `is_stock_code()` 排除 INDEX_REAL ⇒ 主循环也不碰。
        # 不管的话它们就落进"沿用生产"被**逐字节冻结在切换前那一天**。
        # (2026-10-17 实测: raw 已到 10-16 而 bt 停在 10-15 ⇒ 新鲜度门禁硬失败, 链按
        #  fail-closed 中止 —— 三个指数都是回测输入: 上证(regime) + 创业板指 + 中证1000。)
        # 列序也不同: raw = date,open,close,high,low,volume,amount
        #             bt  = datetime,open,high,low,close,volume,openinterest,amount
        for idx in INDEX_REAL:
            rp = f'{RAW}/{idx}/qfq.csv'
            if not os.path.exists(rp):
                stats[f'idx_missing_{idx}'] = stats.get(f'idx_missing_{idx}', 0) + 1
                continue
            try:
                d = pd.read_csv(rp, encoding='utf-8-sig')
                d.columns = [str(c).strip().lstrip('﻿') for c in d.columns]
                o = pd.DataFrame({
                    'datetime': pd.to_datetime(d['date']).dt.strftime('%Y-%m-%d'),
                    'open': d['open'].astype(float), 'high': d['high'].astype(float),
                    'low': d['low'].astype(float), 'close': d['close'].astype(float),
                    'volume': d['volume'].fillna(0), 'openinterest': 0.0,
                    'amount': d['amount'].astype(float),
                })
                o = o[IDX_COLS]           # 指数用 8 列格式(见 IDX_COLS 注释)
                o.to_csv(os.path.join(a.out, f'{idx}_qfq.csv'), index=False)
            except Exception as e:
                print(f'  !!! 指数 {idx} 重建失败({type(e).__name__}: {e})')
                stats[f'idx_fail_{idx}'] = stats.get(f'idx_fail_{idx}', 0) + 1
                continue
            report[idx] = dict(nev=0, nact=0, amp=1.0, index=True)
            n_idx += 1

        # 补集逐位沿用生产文件(见上方注释; 原地重建守卫已提前到本分支开头)
        for s in prod_syms:
            if s in report:
                continue
            src, dst = f'{BT}/{s}_qfq.csv', os.path.join(a.out, f'{s}_qfq.csv')
            with open(src, 'rb') as fi, open(dst, 'wb') as fo:
                fo.write(fi.read())
            ncarry += 1

    # frozen 锚因子落盘(原子). 已有键**绝不覆盖** ⇒ 历史价位永久冻结(铁律)
    if a.anchor == 'frozen':
        old = {}
        if os.path.exists(a.anchor_file):
            with open(a.anchor_file) as fh:
                old = json.load(fh)
        k_new = sum(1 for s in kfix if s not in old)
        old.update(kfix)
        tmp = a.anchor_file + '.tmp'
        with open(tmp, 'w') as fh:
            json.dump(old, fh, sort_keys=True)
        os.replace(tmp, a.anchor_file)
        print(f"  锚因子: {len(old)} 只 (新增 {k_new}) → {a.anchor_file}")

    with open(os.path.join(a.out, '_rebuild_table_report.json'), 'w') as fh:
        json.dump(report, fh)
    print(f"\n=== 落盘 {a.out} ===")
    print(f"  表驱动重建 {len(report) - n_idx}   指数 {n_idx}   沿用生产 {ncarry}   异常 {stats}")
    if drift:
        print(f"  ⚠ 锚漂移 {len(drift)} 只 (**已保留冻结锚**, 未采用新锚 — 铁律优先):")
        print(f"     ⇒ 这些股的 P(末) 自此偏离 raw(末) ×(k/k_cur), 历史价位不变。")
        print(f"     ⇒ 若漂移来自**回溯新增事件**(见 build_events_table.py 的守卫), 那才是真问题;")
        print(f"        若来自新交易日恰为除权日, 属正常。")
        for s, ko, kn in drift[:10]:
            print(f"      {s}  k {ko:.8f} (冻结)  vs k_cur {kn:.8f}  ⇒ P/raw 末端 ×{ko / kn:.6f}")
    else:
        print("  锚漂移 0 只 (事件集未变 ⇒ 铁律成立)")
    _stk = {k: v for k, v in report.items() if not v.get('index')}
    if _stk:
        amp = np.array([v['amp'] for v in _stk.values()])
        print(f"  amp(事件日跳变幅度比) 中位 {np.median(amp):.4f}  >1.10 占比 {(amp > 1.10).mean():.1%}"
              f"  max {amp.max():.2f}   (统计 {len(amp)} 只个股, 不含指数)")
        print(f"  事件总数 {sum(v['nev'] for v in _stk.values())}")


if __name__ == '__main__':
    main()
