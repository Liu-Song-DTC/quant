#!/usr/bin/env python
"""探针(2026-10-16): **重锚不变性** — 加性 qfq vs 乘性 Q, 在"未来分红"发生后的历史漂移。

背景(用户核心问题: "为什么数据历史要根据新更新的数据变化"):
  生产 backtrader_data/*_qfq.csv 是东财**加性**前复权 qfq(t)=(raw(t)−C(t))/Π(t),
  C(t) = t **之后**的累计分红。**每一次新的分红 D 都会改写整段历史**:
      t < d:  C(t) += D  ⇒  qfq_new(t) = qfq_old(t) − D/Π(t)
  段内 Π 恒定 ⇒ 这是**仿射平移** ⇒ 收益率序列**变**(闭式 ret_q/ret_n = 1/(1−C/(raw·Π)))。
  ⇒ 每天刷新一次数据, 高分红股/老日期的历史收益率就被改一次 ⇒ 曲线不可复现。

  乘性重建 Q 的判据: J = Δqfq − a·Δraw 在**段内**对常数平移不变
  (Δqfq 不变 ⇒ J 不变 ⇒ Q 不变) ⇒ 历史收益率**与锚解耦**。

本探针 = 把上述机制**合成**出来并逐日测量漂移:
  ① 对每只股票取决策点 d (序列 60% 处), 施加"未来分红" D = 1% × raw(d)
     —— 按**段**施加平移 D·a_k(忠实还原 C += D ⇒ qfq −= D/Π), 而非全局常数。
  ② 旧管线: 比较 ret_qfq' vs ret_qfq 在 t<d 的漂移
     新管线: 比较 ret_Q'  vs ret_Q  在 t<d 的漂移
  ③ 判据: 漂移日占比 / 最大绝对漂移 / 中位绝对漂移。

⚠ 不声称"重建让曲线更好" —— 那要跑臂(.arms/run_arms10.sh)。本探针只证
  **不变性这一条数学性质**, 与收益高低无关。只读, 不写生产文件。
"""
import os
import sys
import glob
import numpy as np
import pandas as pd

sys.path.insert(0, '/mnt/d/quant/data')
from rebuild_mult_qfq import (affine_recon, split_candidates, split_candidates_v5,
                              _amp, MAX_CVIOL, AMP_LO, AMP_HI)  # noqa: E402

RAW = '/mnt/d/quant/data/stock_data/raw_data'
BT = '/mnt/d/quant/data/stock_data/backtrader_data'
N = int(os.environ.get('N_STOCK', 300))
DIV_FRAC = 0.01          # 合成分红 = raw(d) 的 1% (A股典型股息率量级)


def load(sym):
    b = pd.read_csv(f'{BT}/{sym}_qfq.csv', usecols=['datetime', 'close'])
    b['datetime'] = pd.to_datetime(b['datetime'])
    q = b.rename(columns={'datetime': '日期', 'close': '收盘'})
    r = pd.read_csv(f'{RAW}/{sym}/none.csv', encoding='utf-8-sig')[['日期', '收盘']]
    r['日期'] = pd.to_datetime(r['日期'])
    return r.merge(q, on='日期', suffixes=('_n', '_q')).sort_values('日期')


def recon(raw, qfq):
    """与生产工具同款两遍分派 (v5 优先, 失败退 union) —— 逐位对齐 rebuild_one"""
    mr = float(np.median(np.abs((qfq[1:] / qfq[:-1]) / (raw[1:] / raw[:-1]) - 1)))
    dr = np.diff(raw); dq = np.diff(qfq)
    ok = np.abs(dr) > 0.005 * raw[:-1]
    a0 = float(np.median(dq[ok] / dr[ok])) if ok.sum() >= 100 else 1.0
    ar = float(np.median(np.abs(dq - a0 * dr)) / raw.mean())
    if mr < ar:
        return qfq.copy(), None
    for fn in (split_candidates_v5, split_candidates):
        Q, d_eff, a_seg, seg = affine_recon(raw, qfq, fn)
        # 简化守卫: 只用 amp 带 (c_viol 需额外调用, 对不变性无影响)
        aq = _amp(Q, raw)
        if AMP_LO <= aq <= AMP_HI:
            return Q, (a_seg, seg)
    return qfq.copy(), None


def drift(ret_a, ret_b, hi):
    """t < hi 段的漂移统计 (hi = d 的索引)"""
    da, db = ret_a[:hi - 1], ret_b[:hi - 1]
    m = np.isfinite(da) & np.isfinite(db)
    if m.sum() == 0:
        return 0.0, 0.0, 0.0
    diff = np.abs(da[m] - db[m])
    return float((diff > 1e-9).mean()), float(diff.max()), float(np.median(diff))


def main():
    files = sorted(glob.glob(f'{BT}/*_qfq.csv'))
    codes = [os.path.basename(f).replace('_qfq.csv', '') for f in files]
    rng = np.random.default_rng(20261016)
    syms = [codes[i] for i in rng.choice(len(files), N, replace=False)]
    rows = []
    for sym in syms:
        if sym.startswith(('sh', 'sz', 'bj')):
            continue
        try:
            m = load(sym)
        except Exception:
            continue
        if len(m) < 1200:
            continue
        raw = m['收盘_n'].values.astype(float)
        qfq = m['收盘_q'].values.astype(float)
        if (raw <= 0).any() or (qfq <= 0).any():
            continue
        Q, meta = recon(raw, qfq)
        if meta is None:                      # 乘性直通: 平移不影响它, 跳过
            continue
        a_seg, seg = meta
        n = len(raw)
        d = int(n * 0.60)
        D = DIV_FRAC * raw[d]
        # 忠实合成: C += D ⇒ qfq_new(t) = qfq(t) − D/Π = qfq(t) − D·a_seg[seg(t)]
        qfq2 = qfq.copy()
        shift = D * a_seg[seg[:d]]
        qfq2[:d] = qfq2[:d] - shift
        if (qfq2 <= 0).any():
            continue                          # 平移后负价 = 现实中会被负价修复救回, 此处剔除
        Q2, _ = recon(raw, qfq2)

        rq, rq2 = qfq[1:] / qfq[:-1] - 1, qfq2[1:] / qfq2[:-1] - 1
        rQ, rQ2 = Q[1:] / Q[:-1] - 1, Q2[1:] / Q2[:-1] - 1
        fq, mq, mq2 = drift(rq, rq2, d)
        fQ, mQ, mQ2 = drift(rQ, rQ2, d)
        # 累计效应: d 之前整段的累计收益在重锚前后差多少 (曲线形状漂移的标量)
        cq = float(abs(np.nansum(rq[:d - 1]) - np.nansum(rq2[:d - 1]))) * 100
        cQ = float(abs(np.nansum(rQ[:d - 1]) - np.nansum(rQ2[:d - 1]))) * 100
        rows.append((sym, n, d, fq, mq * 100, fQ, mQ * 100, mq2 * 100, mQ2 * 100, cq, cQ))

    df = pd.DataFrame(rows, columns=['sym', 'n', 'd', 'qfq_frac', 'qfq_max%',
                                     'Q_frac', 'Q_max%', 'qfq_med%', 'Q_med%',
                                     'qfq_cum%', 'Q_cum%'])
    print(f"样本 {len(df)} 只 (排除乘性直通/短序列/平移后负价)")
    if not len(df):
        return
    print(f"\n  === 施加一次'未来分红'(1%×当日价, 按段忠实平移)后, d 之前历史的收益率漂移 ===")
    print(f"  {'':<16}{'漂移日占比':>12}{'最大漂移%':>12}{'中位漂移%':>12}")
    for c1, c2, c3, lbl in [('qfq_frac', 'qfq_max%', 'qfq_med%', '旧管线 加性qfq'),
                            ('Q_frac', 'Q_max%', 'Q_med%', '新管线 乘性Q')]:
        print(f"  {lbl:<16}{df[c1].mean()*100:>11.2f}%{df[c2].max():>12.3f}{df[c3].median():>12.5f}")
    print(f"\n  === 逐股: 旧管线漂移 > 1bp 的股票数 ===")
    print(f"  加性qfq: {(df['qfq_max%'] > 0.01).sum()}/{len(df)}   "
          f"乘性Q: {(df['Q_max%'] > 0.01).sum()}/{len(df)}")
    print(f"  加性qfq: 最大漂移 > 1% 的: {(df['qfq_max%'] > 1).sum()}/{len(df)}   "
          f"乘性Q: {(df['Q_max%'] > 1).sum()}/{len(df)}")
    print(f"\n  === 整段累计收益漂移 (d 之前全部历史, 百分点) ===")
    print(f"  加性qfq: 中位={df['qfq_cum%'].median():.3f}pp  90分位={df['qfq_cum%'].quantile(.9):.3f}pp  最大={df['qfq_cum%'].max():.3f}pp")
    print(f"  乘性Q  : 中位={df['Q_cum%'].median():.3f}pp  90分位={df['Q_cum%'].quantile(.9):.3f}pp  最大={df['Q_cum%'].max():.3f}pp")
    print(f"\n  === 漂移最大的 10 只 (旧管线) ===")
    print(df.nlargest(10, 'qfq_max%').to_string(index=False))


if __name__ == '__main__':
    main()
