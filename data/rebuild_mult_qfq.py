#!/usr/bin/env python
"""乘性复权重建 (2026-10-16) — 把东财**加性**前复权改写成**乘性**总收益序列。

动机(用户: "我玩的是稳定的回测曲线"):
  东财 qfq(t) = (raw(t) − C(t))/Π(t), C(t)=t **之后**累计现金分红
  ⇒ ① 失真: 段内 ret_q/ret_n = raw·Π/qfq = 1/(1−C/(raw·Π)) — 生产数据 13.9% 股票
       收益率失真 >10%, 尾部 7.2× (002601 3.56×, 600066 3.17×)
     ② **重述**: 新分红 ⇒ C 变 ⇒ **全部历史 qfq 被改写** ⇒ 回测曲线不可复现
        (记忆 anchor-refresh-drift: "fq_reload 全史重写" 是 −38.3% 漂移源之一)

重建后 Q:
  段内 Q(t) ∝ raw(t)  — **完全不含 qfq/C** ⇒ 新分红不影响历史
  除权日 Q(d)/Q(d−1) = (raw(d) + D_d)/raw(d−1), D_d = 当日分红
  ⇒ 数学上: J = Δqfq − a·Δraw = a·D_d (C 项在段内恒定, 相减抵消)
  ⇒ **对新的未来分红完全免疫** — 从根上消除重锚重述

⚠ 数据是**混合模型**: backtrader_data 里 ~2.3% 是新浪**乘性**(600519/000858/000333/
  300750 等, 9/3 repair 修过) ⇒ 必须先做模型分派, 乘性直接直通。

**事件日检测(两遍法, 2026-10-16 定稿)** — 全流程最脆的一环:
  a = 1/Π 是**分段常数**, 只在送转/配股日跳变 ⇒ 找错事件日 = a 错 = Q 全错。
  · v5 判据 `(gr>2%) & (gq<0.6·gr)`: 漏判**反号事件** —— 000012/2017-07-19
    raw −8.03% 而 qfq **+8.06%**(行情上涨恰好抵消拆价) ⇒ gr≈gq ⇒ 失效。
  · v6 判据(符号/量级失配, 见 split_candidates_v6): 补上反号, 但假阳性造出假分段。
  · **两遍法**: 先 v5, 过守卫则直接用(已 OK 的股逐位零扰动); 失败再试 v5|v6。
    实测 5378 只: 回归 **0**, fallback 90→31, amp_Q>1.10 由 30→15。
  · 守卫 = `cviol>0.05`(段内 C=raw−qfq/a 单调性违反) 或 `amp_Q ∉ [0.60,1.10]`。
    旧守卫 `nexd>60` 已废: J 门限 0.012 **低于 2 位小数舍入噪声底**(σ_J≈0.0058),
    低价股 4% 的日子被误计成除权日 ⇒ 90 只假失败。

用法:
  python rebuild_mult_qfq.py [--limit N] [--out DIR] [--workers W]
输出: backtrader_data_mul/ (与 backtrader_data 同格式), _rebuild_report.json
**只写新目录, 不碰生产文件。**
"""
import os
import sys
import glob
import json
import time
import argparse
import numpy as np
import pandas as pd

RAW = '/mnt/d/quant/data/stock_data/raw_data'
BT = '/mnt/d/quant/data/stock_data/backtrader_data'
OUT = '/mnt/d/quant/data/stock_data/backtrader_data_mul'

TOL = 0.006                          # J 舍入门限系数
GAP_MIN = 0.02                       # raw 跳空门槛
GAP_QFQ_FRAC = 0.6                   # 且 |Δqfq|/qfq < 该比例 × |Δraw|/raw (送转: qfq 连续)
HUGE_GAP = 0.18                      # raw 跳空超出涨跌停 ⇒ 必是公司行动
HUGE_QFQ = 0.12                      # 且 qfq 仍平稳(排除 20% 板跌停)
MERGE_TOL = 0.02                     # 相邻段 a 相对差 < 2% ⇒ 合并(分红假阳性)
# ⚠ 旧守卫 `nexd > 60` 已废: J 的门限 TOL·(1+a)=0.012 **低于 2 位小数的舍入噪声底**
#   (σ_J = sqrt(2+2a²)·0.0029 ≈ 0.0058), 低价股 4% 的日子被误计成"除权日"
#   ⇒ 90 只假失败。实测该计数在真失败群与正常群上完全无分离(中位 6 vs 4)。
# ⚠ 模型分派已于 2026-10-16 **取消**(旧 `mult_resid < aff_resid` 判据是掷硬币, 见 rebuild_one)。
#   现在一律跑仿射重建 —— 它对加性/纯乘性/总收益乘性**三类都正确**, 是超集。
MAX_CVIOL = 0.05                     # 段内 C=raw−qfq/a 单调性违反 (真守卫)
AMP_LO, AMP_HI = 0.60, 1.10          # amp_Q=σ(ret_Q)/σ(ret_raw) 合理带(真守卫)
COLS = ['datetime', 'open', 'high', 'low', 'close', 'volume', 'openinterest',
        'amount', 'amplitude', 'change_percent', 'change_amount', 'turnover_rate']


def _est_a(seg, mask, dq, dr, nseg, prev=None):
    """逐段 median(Δqfq/Δraw); 样本不足 ⇒ 用 prev/邻居。"""
    a = np.full(nseg, np.nan) if prev is None else prev.copy()
    for k in range(nseg):
        m = mask & (seg[1:] == k)
        if m.sum() >= 30:
            a[k] = float(np.median(dq[m] / dr[m]))
    for k in range(nseg):
        if not np.isfinite(a[k]):
            nb = [a[j] for j in (k - 1, k + 1) if 0 <= j < nseg and np.isfinite(a[j])]
            a[k] = nb[0] if nb else 1.0
    return a


def _reseg(seg, a_seg):
    """相邻段 a 相近(<MERGE_TOL) ⇒ 合并。返回新 seg。"""
    n = len(seg)
    out = np.zeros(n, dtype=np.int32)
    cur = 0
    for t in range(1, n):
        if seg[t] != seg[t - 1]:
            a0, a1 = a_seg[seg[t - 1]], a_seg[seg[t]]
            if abs(a0 - a1) / max(abs(a0), abs(a1), 1e-12) >= MERGE_TOL:
                cur += 1
        out[t] = cur
    return out


def split_candidates_v6(raw, qfq):
    """v6 事件日候选: **符号/量级失配** 判据。

    送转日模型: Δqfq = a_new·Δraw + (a_new−a_old)·raw(t−1)·(…) ⇒ 与 Δraw 失配。
    普通日: Δqfq = a·Δraw (同号、比值≈a)。

    ⚠ v5 的 (gr>GAP_MIN)&(gq<0.6·gr) 判据会漏判 —— 000012/2017-07-19
      raw −8.03% 而 qfq **+8.06%**(行情上涨抵消拆价) ⇒ gr≈gq ⇒ 判据失效。
    ⚠ 也不可用 rr/rq —— 普通日 rr/rq = qfq·Π/raw = 1−C/raw = **失真因子本身**
      (002601 因此报 92 个假候选, 000983 报 280 个)。
    ⇒ 用稳健运行中位 r_med 作参照: 事件日 = 大跳空 且 (反号 | 比值越界)。
      假阳性无害: 若两侧 a 相同, 送转日公式退化为普通日公式(见 Q 构造)。
    """
    n = len(raw)
    dr = np.diff(raw); dq = np.diff(qfq)
    ok = np.abs(dr) > 0.005 * raw[:-1]
    r = np.full(n - 1, np.nan)
    r[ok] = dq[ok] / dr[ok]
    r_med = pd.Series(r).rolling(121, center=True, min_periods=20).median().values
    big = np.abs(dr) > 0.015 * raw[:-1]
    sign_bad = np.sign(dq) != np.sign(dr)
    mag_bad = (np.abs(dq) > 3.0 * r_med * np.abs(dr)) | \
              (np.abs(dq) < 0.25 * r_med * np.abs(dr))
    return np.nan_to_num(big & (sign_bad | mag_bad) & np.isfinite(r_med),
                         nan=False).astype(bool)


def split_candidates_v5(raw, qfq):
    """v5 旧判据(保留用于 A/B): raw 跳空且 qfq 相对平稳。会漏判反号事件。"""
    rr = raw[1:] / raw[:-1]; rq = qfq[1:] / qfq[:-1]
    gr = np.abs(rr - 1); gq = np.abs(rq - 1)
    return np.nan_to_num(
        ((gr > GAP_MIN) & (gq < GAP_QFQ_FRAC * gr)) |
        ((gr > HUGE_GAP) & (gq < HUGE_QFQ)), nan=False).astype(bool)


def split_candidates(raw, qfq):
    """**并集** — v5 与 v6 各覆盖一半盲区 (A/B 实测):

      v5 漏: 反号事件 (000012/2017-07-19 raw −8.03% 而 qfq +8.06%)
      v6 漏: 小跳空但比值越界的事件
      union 在 90 只 v5-fallback 上 90/90 可重建, 300 只已 OK 股 0 回归。
    假阳性无害: 两侧 a 相同时送转日公式退化为普通日公式 (见 Q 构造)。
    """
    return split_candidates_v5(raw, qfq) | split_candidates_v6(raw, qfq)


def _amp(x, raw):
    """收益率波动率比 σ(ret_x)/σ(ret_raw)。真乘性总收益序列应 ≈1。"""
    rn = np.diff(raw) / raw[:-1]
    if rn.std() <= 0:
        return 1.0
    return float(np.std(np.diff(x) / x[:-1]) / rn.std())


def affine_recon(raw, qfq, cand_fn=split_candidates):
    n = len(raw)
    dr = np.diff(raw); dq = np.diff(qfq)
    rr = raw[1:] / raw[:-1]; rq = qfq[1:] / qfq[:-1]
    gr = np.abs(rr - 1); gq = np.abs(rq - 1)
    # 送转候选: raw 跳空但 qfq 相对平稳。
    split = cand_fn(raw, qfq)
    seg = np.zeros(n, dtype=np.int32)
    for t in range(1, n):
        seg[t] = seg[t - 1] + (1 if split[t - 1] else 0)
    ok = (np.abs(dr) > 0.005 * raw[:-1]) & ~split

    a_seg = _est_a(seg, ok, dq, dr, int(seg[-1]) + 1)
    for _ in range(10):
        for _ in range(3):                       # 剔除分红日重估
            at = a_seg[seg[1:]]
            bnd = np.abs(dq - at * dr) > TOL * (1 + at)
            a_seg = _est_a(seg, ok & ~bnd, dq, dr, len(a_seg), prev=a_seg)
        new = _reseg(seg, a_seg)                 # 合并 a 相近的相邻段(去分红假阳性)
        if new[-1] == seg[-1]:
            break
        seg = new
        a_seg = _est_a(seg, ok, dq, dr, int(seg[-1]) + 1)
    split = np.zeros(n - 1, dtype=bool)          # 最终段边界
    split[seg[1:] != seg[:-1]] = True
    Q = np.empty(n); Q[0] = raw[0]
    d_eff = np.zeros(n)
    for t in range(1, n):
        k = seg[t]; a = a_seg[k]
        if split[t - 1]:
            a0 = a_seg[seg[t - 1]]
            D = max((raw[t - 1] - qfq[t - 1] / a0) - (raw[t] - qfq[t] / a), 0.0)
            Q[t] = Q[t - 1] * (raw[t] * (a / a0) + D) / raw[t - 1]
        else:
            J = dq[t - 1] - a * dr[t - 1]
            if abs(J) < TOL * (1 + a):
                J = 0.0
            d_eff[t] = J / a
            Q[t] = Q[t - 1] * (raw[t] + J / a) / raw[t - 1]
    if Q[-1] > 0:
        Q *= raw[-1] / Q[-1]
    return Q, d_eff, a_seg, seg


def c_viol(raw, qfq, a_seg, seg):
    """段内 C=raw−qfq/a 的单调性违反(分红只应让 C 下降)。"""
    C = raw - qfq / a_seg[seg]
    worst = 0.0
    for k in range(len(a_seg)):
        c = C[seg == k]
        if len(c) >= 30 and raw[seg == k].mean() > 0:
            worst = max(worst, float((c - np.minimum.accumulate(c)).max() / raw[seg == k].mean()))
    return worst


def rebuild_one(sym):
    """返回 (状态, 统计 dict)。"""
    bt_path = os.path.join(BT, f'{sym}_qfq.csv')
    raw_path = os.path.join(RAW, sym, 'none.csv')
    try:
        b = pd.read_csv(bt_path)
        r = pd.read_csv(raw_path, encoding='utf-8-sig')
    except Exception as e:
        return 'read_fail', {'err': str(e)[:80]}
    if len(b) < 60 or len(r) < 60:
        return 'short', {}
    b['datetime'] = pd.to_datetime(b['datetime'])
    r['日期'] = pd.to_datetime(r['日期'])
    m = b[['datetime', 'open', 'high', 'low', 'close']].merge(
        r[['日期', '开盘', '最高', '最低', '收盘']].rename(
            columns={'开盘': 'r_open', '最高': 'r_high', '最低': 'r_low', '收盘': 'r_close'}),
        left_on='datetime', right_on='日期', how='inner').sort_values('datetime')
    if len(m) < 60:
        return 'no_overlap', {}
    raw = m['r_close'].values.astype(float)
    qfq = m['close'].values.astype(float)
    if (raw <= 0).any() or (qfq <= 0).any():
        return 'nonpositive', {}

    dr = np.diff(raw); dq = np.diff(qfq)
    rr = raw[1:] / raw[:-1]; rq = qfq[1:] / qfq[:-1]
    mult_resid = float(np.median(np.abs(rq / rr - 1)))
    ok = np.abs(dr) > 0.005 * raw[:-1]
    a0 = float(np.median(dq[ok] / dr[ok])) if ok.sum() >= 100 else 1.0
    aff_resid = float(np.median(np.abs(dq - a0 * dr)) / raw.mean())
    # ── 模型分派(2026-10-16 **取消**): 一律跑仿射重建, 不再有 mult 直通 ──
    # 旧判据 `mult_resid < aff_resid` 是**掷硬币**:
    #   ① 两个统计量**都在量化噪声底**(实测 603187: 0.00121 vs 0.00139) ⇒ 边距 0.0002 无意义;
    #   ② `a0` 是**全局标量**而 a=1/Π **每段都变** ⇒ 拆股多的股(603187 有 5 次, Π 跨度 5.1×)
    #      a0=0.716 拟合不到任何一段, aff_resid 被"全局 a 失配"污染而非"是不是仿射"。
    #   ⇒ 实测后果: 592 只走 mult 直通(Q≡qfq, **重建零作用**), 其中 364 只其实是加性失真股
    #      —— 恰是 C/raw 最大的一群(s_med 中位 0.669) ⇒ 加性失真最重的股被放过。
    # 中间试过的替代判据(绝对漂移 d60 vs 量化地板)也**不行**: 000750 的 s 从 0.53 漂到 1.00
    #   (失真巨大) 但低价股 2 位小数的分辨率让 d60=0.006 只到地板的 2.4 倍 ⇒ 仍误判。
    # ⇒ 正解是**不判**: 仿射重建对**三类模型都正确**, 是它们的超集 —
    #   · 纯乘性 `qfq=raw/Π`: 段内 a=1/Π 恒定 ⇒ J=dq−a·dr=0 ⇒ Q∝raw, 与 qfq 段内收益逐位相同;
    #   · 总收益乘性(分红已乘性入 Π): 除权日 J=a·D≠0 ⇒ 加回 ⇒ 与 qfq 收益相同;
    #   · 加性 `qfq=(raw−C)/Π`: 正是要修的模型。
    #   唯一风险是**检测错**(假分段 ⇒ a 估错 / 假除权), 而那是 cviol + amp 守卫的职责, 不是分派的。
    _s = qfq / raw
    _K = min(60, len(_s) // 4)
    d60 = float(np.median(np.abs(_s[_K:] / _s[:-_K] - 1))) if _K >= 10 else 0.0
    drift_floor = 0.01 / max(float(np.median(qfq)), 1e-9)   # 仅供审计, 不参与分派

    amp_qfq = _amp(qfq, raw)
    # **两遍法**: 先 v5, 通过守卫则直接用(对已 OK 的股逐位零扰动);
    # 否则再试 union。单用 union 会引入 13/5378 回归 —— v6 的假阳性候选
    # 造出假分段 ⇒ a 估错 (002601 从 amp 0.90 退回 3.56)。
    Q = None; det = 'fallback'
    for fn, tag in ((split_candidates_v5, 'v5'), (split_candidates, 'union')):
        Q2, d_eff, a_seg, seg = affine_recon(raw, qfq, fn)
        cv = c_viol(raw, qfq, a_seg, seg)
        aq = _amp(Q2, raw)
        if cv <= MAX_CVIOL and AMP_LO <= aq <= AMP_HI:
            Q, cviol, amp_Q, det = Q2, cv, aq, tag
            nseg, nsplit = len(a_seg), int(seg[-1])
            nexd = int((np.abs(d_eff) > 1e-9).sum())
            break
    model = 'affine' if det != 'fallback' else 'fallback'
    if Q is None:
        Q = qfq.copy(); amp_Q = amp_qfq
        nseg = nsplit = nexd = 0; cviol = 0.0

    # OHLC: Q ∝ raw 段内 ⇒ 按 f(t)=Q_close/raw_close 缩放
    f = Q / raw
    out = m[['datetime']].copy()
    for src, dst in [('r_open', 'open'), ('r_high', 'high'), ('r_low', 'low')]:
        out[dst] = m[src].values.astype(float) * f
    out['close'] = Q
    out = out.merge(b, on='datetime', how='left', suffixes=('', '_o'))
    out['volume'] = out['volume'].fillna(0)
    out['openinterest'] = out.get('openinterest', 0.0)
    # raw 口径真值列(none.csv 本就正确, 不被复权改写)
    rr2 = r.rename(columns={'日期': 'datetime'})
    out = out.merge(rr2[['datetime', '振幅', '涨跌幅', '涨跌额', '换手率']],
                    on='datetime', how='left')
    out['amplitude'] = out['振幅'].fillna(out.get('amplitude', np.nan))
    out['change_percent'] = out['涨跌幅'].fillna(out.get('change_percent', np.nan))
    out['change_amount'] = out['涨跌额'].fillna(out.get('change_amount', np.nan))
    out['turnover_rate'] = out['换手率'].fillna(out.get('turnover_rate', np.nan))
    out['amount'] = out.get('amount', np.nan)
    out['datetime'] = out['datetime'].dt.strftime('%Y-%m-%d')
    for c in ['open', 'high', 'low', 'close']:
        out[c] = out[c].round(2)
    out = out[COLS]
    return 'ok', {'model': model, 'det': det, 'nseg': nseg, 'nsplit': nsplit,
                  'nexd': nexd, 'cviol': round(cviol, 5),
                  'amp_qfq': round(amp_qfq, 4), 'amp_Q': round(amp_Q, 4),
                  # 分派审计字段(2026-10-16): d60/floor 是新判据的两侧, mult_resid/aff_resid 是旧判据
                  'd60': round(d60, 6), 'drift_floor': round(drift_floor, 6),
                  'mult_resid': round(mult_resid, 6), 'aff_resid': round(aff_resid, 6)}, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--out', default=OUT)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    files = sorted(glob.glob(f'{BT}/*_qfq.csv'))
    syms = [os.path.basename(f).replace('_qfq.csv', '') for f in files]
    if args.limit:
        syms = syms[:args.limit]

    stats = {'ok': 0, 'mult': 0, 'affine': 0, 'fallback': 0,
             'read_fail': 0, 'short': 0, 'no_overlap': 0, 'nonpositive': 0,
             'amp_qfq_gt110': 0, 'amp_Q_gt110': 0, 'sym': {}}
    t0 = time.time()
    for i, sym in enumerate(syms):
        if sym.startswith(('sh', 'sz', 'bj')):      # 指数: 无复权, 直接复制
            try:
                pd.read_csv(f'{BT}/{sym}_qfq.csv').to_csv(
                    os.path.join(args.out, f'{sym}_qfq.csv'), index=False)
                stats['sym'][sym] = {'model': 'index'}
            except Exception:
                pass
            continue
        st, info, *rest = rebuild_one(sym)
        if st == 'ok' and rest and rest[0] is not None:
            rest[0].to_csv(os.path.join(args.out, f'{sym}_qfq.csv'), index=False)
            stats[info['model']] = stats.get(info['model'], 0) + 1
            if info.get('amp_qfq', 0) > 1.10:
                stats['amp_qfq_gt110'] += 1
            if info.get('amp_Q', 0) > 1.10:
                stats['amp_Q_gt110'] += 1
        else:
            stats[st] = stats.get(st, 0) + 1
            try:                                     # 失败 ⇒ 原样复制(不劣化)
                pd.read_csv(f'{BT}/{sym}_qfq.csv').to_csv(
                    os.path.join(args.out, f'{sym}_qfq.csv'), index=False)
            except Exception:
                pass
        stats['sym'][sym] = info
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(syms)}  {time.time()-t0:.0f}s", flush=True)

    stats['sym'] = {k: v for k, v in stats['sym'].items() if v.get('model') != 'index'}
    with open(os.path.join(args.out, '_rebuild_report.json'), 'w') as fh:
        json.dump(stats, fh, ensure_ascii=False, indent=1)
    print(f"\n完成 {len(syms)} 只, {time.time()-t0:.0f}s → {args.out}")
    for k in ['mult', 'affine', 'fallback', 'read_fail', 'short', 'no_overlap', 'nonpositive']:
        print(f"  {k:<12} {stats.get(k, 0)}")
    print(f"  amp>1.10: qfq={stats['amp_qfq_gt110']} → Q={stats['amp_Q_gt110']}")


if __name__ == '__main__':
    main()
