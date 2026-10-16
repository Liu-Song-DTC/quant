"""诊断: 全市场 11 只 Q 仍有漂移的股票 — 是不是分段检测器被平移改变了输出?"""
import os, sys, glob
import numpy as np, pandas as pd
sys.path.insert(0, '/mnt/d/quant/data')
sys.path.insert(0, '/tmp')
from probe_reanchor_invariance import load, recon, RAW, BT, DIV_FRAC

files = sorted(glob.glob(f'{BT}/*_qfq.csv'))
codes = [os.path.basename(f).replace('_qfq.csv','') for f in files]
bad = []
for sym in codes:
    if sym.startswith(('sh','sz','bj')): continue
    try: m = load(sym)
    except Exception: continue
    if len(m) < 1200: continue
    raw = m['收盘_n'].values.astype(float); qfq = m['收盘_q'].values.astype(float)
    if (raw<=0).any() or (qfq<=0).any(): continue
    Q, meta = recon(raw, qfq)
    if meta is None: continue
    a_seg, seg = meta
    n = len(raw); d = int(n*0.60); D = DIV_FRAC*raw[d]
    qfq2 = qfq.copy(); qfq2[:d] = qfq2[:d] - D*a_seg[seg[:d]]
    if (qfq2<=0).any(): continue
    Q2, meta2 = recon(raw, qfq2)
    rQ, rQ2 = Q[1:]/Q[:-1]-1, Q2[1:]/Q2[:-1]-1
    diff = np.abs(rQ[:d-1]-rQ2[:d-1]); diff = diff[np.isfinite(diff)]
    if len(diff) and diff.max() > 0.01:
        nseg1 = len(a_seg); nseg2 = len(meta2[0]) if meta2 else -1
        nd = int((diff > 1e-9).sum())
        bad.append((sym, n, d, nseg1, nseg2, nd, diff.max()*100,
                    float(np.median(a_seg)), float(np.median(meta2[0])) if meta2 else np.nan))
bad.sort(key=lambda x: -x[6])
print(f"{'sym':<9}{'n':>6}{'d':>6}{'nseg旧':>7}{'nseg新':>7}{'漂移日':>7}{'最大%':>10}{'a中位旧':>9}{'a中位新':>9}")
for r in bad[:15]:
    print(f"{r[0]:<9}{r[1]:>6}{r[2]:>6}{r[3]:>7}{r[4]:>7}{r[5]:>7}{r[6]:>10.3f}{r[7]:>9.4f}{r[8]:>9.4f}")
print(f"合计 {len(bad)} 只 (判据: d 前最大日漂移 > 1%)")
