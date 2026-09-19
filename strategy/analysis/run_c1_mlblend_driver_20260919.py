#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""C1 ML blend权重重标定驱动 v3 (2026-09-19晚) — 修复v1/v2的两个致命缺陷:
1. **列目标错误**: 回测消费的是CSV的 `score` 列(生成时 score=gate_score=adjusted_score,
   signal_engine.py:701-702, 逐行验证 max|score-adj|=0.0), v2只改adjusted_score列
   → 注入惰性(臂1四指标逐位=基线732,689)。v3同时改写 score 与 adjusted_score 两列。
2. **bash跑者静默死亡**: v2跑者(运行器bash+set -e)在臂2开头无痕迹消失 → v3改为
   单进程python驱动(快照→注入→回测→解析→复原), 全部输出flush直写。
3. A项(alt_market, date纯函数)逐日重算并精确剥离(v2机制保留) — mask行与non-mask
   行的A均原样保留 → 纯ML权重变更。
identity烟测(w1=0.4): max|diff|<1e-12, 否则中止, 不跑任何臂。
**诚实局限(裁决时必读)**: buy/sell标志为w0=0.4生成时冻结, 本注入只测"排名层"
(w1在w0买入集内的选股重排), 不含门槛重门控。方向性4-0胜才值得付ml节regen
(~2.5-3h/臂, 非豁免)做忠实臂。
用法: python analysis/run_c1_mlblend_driver_20260919.py 30 50 55
"""
import os, sys, json, shutil, subprocess, re
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, '/mnt/d/quant/strategy')
from core.alternative_data import get_provider

BASE = '/mnt/d/quant/strategy'
RVD = os.path.join(BASE, 'rolling_validation_results')
ARMS_DIR = os.path.join(BASE, 'arms_20260919')
PY = '/mnt/d/quant/.venv/bin/python'
W0 = 0.4

PROD_FILES = ['portfolio_selections.csv', 'trade_realized.csv', 'equity_curve.csv',
              'regime_state.csv', 'yaogu_watchlist.csv', '.signal_code_fp']

METRIC_RE = {
    'nav': re.compile(r'最终净值:\s*([\d,]+)\s*\(总收益\s*([\d.]+)%'),
    'sharpe': re.compile(r'Sharpe:\s*([\d.]+)'),
    'mdd': re.compile(r'最大回撤:\s*([\d.]+)%'),
    'years': re.compile(r'^\s*(\d{4}):\s*([-\d.]+)%\s*\(最大回撤\s*([\d.]+)%', re.M),
}


def read(path):
    with open(path, encoding='utf-8') as f:
        return f.read()


def inject(w1, out_path):
    """v3注入: 改写 score+adjusted_score 两列. 返回 stats."""
    print(f"加载生产signals (w1={w1})...", flush=True)
    sig = pd.read_csv(f'{RVD}/backtest_signals.csv', dtype={'code': str})
    n = len(sig)
    print(f"{n} 行加载完成", flush=True)

    provider = get_provider()
    dates = pd.to_datetime(sig['date'])
    uniq_dates = sorted(set(dates))
    A_map = {}
    for d in uniq_dates:
        prev = (d - pd.Timedelta(days=1)).date()
        nb = provider.get_northbound_signal(prev)
        mg = provider.get_margin_signal(prev)
        A_map[d] = (nb * 0.6 + mg * 0.4) * 0.15
    A = np.array([A_map[d] for d in dates])
    print(f"A项: {len(uniq_dates)} 日期, |A|max={np.abs(A).max():.4f}, "
          f"非零日期比例={(np.abs(np.array([A_map[d] for d in uniq_dates])) > 1e-12).mean()*100:.1f}%",
          flush=True)

    mask = np.abs(sig['ml_score'].to_numpy()) > 0.01
    z = np.tanh(sig['ml_score'].to_numpy() * 3)
    # score==adjusted_score(已验证), 用score做源, 两列同写
    adj = sig['score'].to_numpy().astype(np.float64)
    s_clean = (adj - W0 * z - A) / (1 - W0)
    adj_new = adj.copy()
    adj_new[mask] = (1 - w1) * s_clean[mask] + A[mask] + w1 * z[mask]
    print(f"注入行: {mask.sum()} ({mask.mean()*100:.1f}%), w0={W0}→w1={w1}", flush=True)

    if abs(w1 - W0) < 1e-9:
        dmax = np.abs(adj_new - adj).max()
        print(f"[identity烟测] w1==w0, max|adj_new-adj| = {dmax:.3e}", flush=True)
        assert dmax < 1e-12, "identity烟测失败! 剥离/重blend机制与生产不一致, 中止"

    sig['score'] = adj_new
    sig['adjusted_score'] = adj_new
    sig.to_csv(out_path, index=False)
    print(f"写入 {out_path}", flush=True)
    # 释放大对象再回测(本机15.8GB, driver残留+bt子进程同驻易OOM)
    import gc
    del sig, adj, adj_new, s_clean, z, dates, A
    gc.collect()


def parse_metrics(log_path):
    m = {'nav': None, 'ret': None, 'sharpe': None, 'mdd': None, 'years': {}}
    txt = read(log_path) if os.path.exists(log_path) else ''
    g = METRIC_RE['nav'].search(txt)
    if g:
        m['nav'] = int(g.group(1).replace(',', ''))
        m['ret'] = float(g.group(2))
    g = METRIC_RE['sharpe'].search(txt)
    if g:
        m['sharpe'] = float(g.group(1))
    g = METRIC_RE['mdd'].search(txt)
    if g:
        m['mdd'] = float(g.group(1))
    for g in METRIC_RE['years'].finditer(txt):
        m['years'][int(g.group(1))] = (float(g.group(2)), float(g.group(3)))
    return m


def snapshot(tag_dir):
    os.makedirs(tag_dir, exist_ok=True)
    for f in PROD_FILES:
        p = os.path.join(RVD, f)
        if os.path.exists(p):
            shutil.copy2(p, os.path.join(tag_dir, 'pre_' + f))


def restore(tag_dir):
    for f in PROD_FILES:
        p = os.path.join(RVD, f)
        pre = os.path.join(tag_dir, 'pre_' + f)
        if os.path.exists(pre):
            shutil.copy2(pre, p)


def main():
    pcts = [int(x) for x in sys.argv[1:]] if len(sys.argv) > 1 else [30, 50, 55]
    os.makedirs(ARMS_DIR, exist_ok=True)
    # 生产信号原件: 一次性归档到sig_base, 每臂后从这里复原(免每臂3GB备份)
    sig_base = os.path.join(ARMS_DIR, '_baseline_sig')
    os.makedirs(sig_base, exist_ok=True)
    prod_sig = os.path.join(RVD, 'backtest_signals.csv')
    base_sig = os.path.join(sig_base, 'backtest_signals.csv')
    if not os.path.exists(base_sig):
        print("归档生产信号原件...", flush=True)
        shutil.copy2(prod_sig, base_sig)
    fp0 = read(os.path.join(RVD, '.signal_code_fp'))

    # identity烟测: w=0.4 必须逐位复原
    smoke = os.path.join(RVD, 'backtest_signals.mlw40smoke.csv')
    inject(W0, smoke)
    os.remove(smoke)
    print("=== identity烟测通过, 开始臂序列 ===\n", flush=True)

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    summary = []
    for i, pct in enumerate(pcts, 1):
        w1 = pct / 100.0
        tag = os.path.join(ARMS_DIR, f'C1_mlblend_{pct:03d}')
        print(f"===== [{i}/{len(pcts)}] C1 w={pct}% =====", flush=True)
        snapshot(tag)
        try:
            inj = os.path.join(RVD, f'backtest_signals.mlw{pct}.csv')
            inject(w1, inj)
            print("替换生产信号CSV...", flush=True)
            shutil.copy2(inj, prod_sig)
            log = os.path.join(tag, f'run_{ts}.log')
            print("跑回测...", flush=True)
            with open(log, 'w', encoding='utf-8') as lf:
                rc = subprocess.run([PY, 'bt_execution.py'], cwd=BASE,
                                    stdout=lf, stderr=subprocess.STDOUT).returncode
            m = parse_metrics(log)
            m['rc'] = rc
            with open(os.path.join(tag, 'metrics.json'), 'w') as f:
                json.dump(m, f, indent=1, ensure_ascii=False)
            for f in PROD_FILES:
                p = os.path.join(RVD, f)
                if os.path.exists(p):
                    shutil.copy2(p, os.path.join(tag, 'post_' + f))
            line = (f"C1_{pct:03d}: rc={rc} NAV={m['nav']} ret={m['ret']}% "
                    f"Sharpe={m['sharpe']} MDD={m['mdd']}%")
            print(line, flush=True)
            summary.append(line)
        finally:
            restore(tag)
            print("复原生产信号+产物...", flush=True)
            shutil.copy2(base_sig, prod_sig)
            assert read(os.path.join(RVD, '.signal_code_fp')) == fp0, "fp sidecar漂移!"
            if os.path.exists(inj):
                os.remove(inj)
    print("\n===== 汇总 (基线=732,689/193.08%/1.1961/17.92%) =====")
    for s in summary:
        print(s)
    with open(os.path.join(ARMS_DIR, f'c1_summary_{ts}.txt'), 'w') as f:
        f.write('\n'.join(summary))


if __name__ == '__main__':
    main()
