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

PROJ = '/mnt/d/quant'
RAW = f'{PROJ}/data/stock_data/raw_data'
BT = f'{PROJ}/data/stock_data/backtrader_data'
CA = f'{PROJ}/data/alternative_data/corporate_actions.pkl'
OUT = f'{PROJ}/data/stock_data/backtrader_data_tbl'
ANCHOR_FILE = f'{PROJ}/data/stock_data/qfq_anchor.json'   # frozen 锚的冻结因子(一旦生成即合约)

COLS = ['datetime', 'open', 'high', 'low', 'close', 'volume', 'openinterest',
        'amount', 'amplitude', 'change_percent', 'change_amount', 'turnover_rate']


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
    lt = pd.to_datetime(pd.read_csv(p[0], encoding='utf-8-sig', usecols=['日期'])['日期']).max()
    now = pd.Timestamp.now()
    if lt.normalize() == now.normalize() and (now.hour, now.minute) < (15, 5):
        raise SystemExit(
            f"[铁律守卫] raw 最后一行 {lt.date()} 是今天且现在 {now:%H:%M} 未收盘 "
            f"⇒ 该行是临时值(实测次日必被修正), 拒绝构建。收盘后重跑。")


def build_one(sym, actions, raw_df, bt_df, anchor='frozen', kfix=None):
    """返回 (P_close, nev)。P 的锚因子见模块 docstring 的锚点说明。

    kfix: frozen 锚的冻结因子 {sym: k}。缺失时由本函数算出并**回写调用方**
          (调用方负责持久化 ⇒ 下次重建逐位复用)。
    """
    dt = pd.to_datetime(raw_df['日期']).values
    raw = raw_df['收盘'].values.astype(float)
    n = len(raw)
    if n < 60:
        return None
    # 事件聚合到"第一个 >= 除权日的交易日"(除权日可能落在非交易日/停牌日)
    idx = pd.Series(np.arange(n), index=pd.DatetimeIndex(dt))
    ev = {}
    for ex, s, D in actions:
        pos = idx.index.searchsorted(pd.Timestamp(ex), side='left')
        if pos >= n:
            continue                      # 样本之后 ⇒ 忽略
        key = dt[pos]
        s0, D0 = ev.get(key, (0.0, 0.0))
        ev[key] = ((1 + s0) * (1 + s) - 1, D0 + D)   # 同日多事件合并
    W = np.empty(n); W[0] = raw[0]
    nev = 0
    for t in range(1, n):
        r = raw[t] / raw[t - 1]
        e = ev.get(dt[t])
        if e is not None and raw[t - 1] > 0:
            s, D = e
            r = ((1 + s) * raw[t] + D) / raw[t - 1]
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
        k = kfix.get(sym)
        if k is None:
            k = float(raw[-1] / W[-1])
            kfix[sym] = k
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

    ca = pd.read_pickle(CA)
    ca_by = {c: g[['ex_date', 's', 'D']].values.tolist() for c, g in ca.groupby('code')}

    if a.only:
        syms = [s.strip().zfill(6) for s in a.only.split(',')]
    else:
        # 主集: raw_data 有原始价(none.csv)的股票
        syms = sorted(os.path.basename(os.path.dirname(p))
                      for p in glob.glob(f'{RAW}/*/none.csv'))
        syms = [s for s in syms if s[0] in '036']       # 北交所排除(铁律)
        # 补集: **重建没产出的股票一律沿用生产文件**, 保证股票池与基线逐位相同
        #   (a) 118 只退市股 —— raw_data 只有 qfq.csv 无原始价, 且官方分红表查不到
        #       (ak.stock_fhps_detail_em 对退市码返回 None) ⇒ 无法反解 raw
        #   (b) 26 只次新股 —— 样本 <60 天, build_one 主动跳过
        # 目的: 复权口径臂保持**单变量对照**, ΔNAV 只归因于复权方法而非股票池。
        prod_syms = sorted(os.path.basename(p)[:-8] for p in glob.glob(f'{BT}/*_qfq.csv'))

    report = {}
    stats = {}
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
        got = build_one(sym, acts, r, b, anchor=a.anchor, kfix=kfix)
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
        b2 = b.copy()
        b2['datetime'] = pd.to_datetime(b2['datetime'])
        out = out.merge(b2[['datetime', 'volume', 'openinterest', 'amount']],
                        on='datetime', how='left')
        out['volume'] = out['volume'].fillna(0)
        out['openinterest'] = out['openinterest'].fillna(0.0)
        out['amount'] = out['amount'].fillna(np.nan)
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

    # 补集逐位沿用生产文件(见上方注释)
    ncarry = 0
    if not a.only:
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
    print(f"  表驱动重建 {len(report)}   沿用生产 {ncarry}   异常 {stats}")
    if report:
        amp = np.array([v['amp'] for v in report.values()])
        print(f"  amp(事件日跳变幅度比) 中位 {np.median(amp):.4f}  >1.10 占比 {(amp > 1.10).mean():.1%}"
              f"  max {amp.max():.2f}")
        print(f"  事件总数 {sum(v['nev'] for v in report.values())}")


if __name__ == '__main__':
    main()
