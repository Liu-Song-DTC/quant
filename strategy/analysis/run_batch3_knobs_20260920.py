#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""批次3旋钮臂驱动 (2026-09-20): C2峰值定位 + C14 + C2×C5c交互对.
协议同batch2(快照→yaml编辑→纯回测→四指标→复原→归档, 信号fp不变→复用~20min/臂);
新增: 多编辑臂(交互臂同时改两个豁免键)。缺省跑前4臂(峰值定位+C14);
交互对 BUF0_ref / C2_055_BUF0 由C2裁决后按名选择:
  BUF0_ref    = 0.45+缓冲0.0   (前生产态参照)
  C2_055_BUF0 = 0.55+缓冲0.0   (C2在无C5c态的效果)
交互分解: Δ(0.55|buf0) = C2_055_BUF0 − BUF0_ref; 与 Δ(0.55|buf0.05)=C2_055−基线
比较即C2×C5c交互。用法: python analysis/run_batch3_knobs_20260920.py [臂名...]
"""
import os, sys, json, shutil, subprocess, re
from datetime import datetime

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RVD = os.path.join(BASE, 'rolling_validation_results')
YAML = os.path.join(BASE, 'config', 'factor_config.yaml')
ARMS_DIR = os.path.join(BASE, 'arms_20260920')
PY = '/mnt/d/quant/.venv/bin/python'

PROD_FILES = ['portfolio_selections.csv', 'trade_realized.csv', 'equity_curve.csv',
              'regime_state.csv', 'yaogu_watchlist.csv', '.signal_code_fp']
CODE_FILES = [YAML]

# 臂 = [(old, new), ...] 列表式, 支持多编辑
ARMS = {
    'C2_050':      [('bp2_score_boost: 0.45', 'bp2_score_boost: 0.50')],
    'C2_060':      [('bp2_score_boost: 0.45', 'bp2_score_boost: 0.60')],
    'C2_065':      [('bp2_score_boost: 0.45', 'bp2_score_boost: 0.65')],
    'C14_080':     [('base_exposure: 0.85', 'base_exposure: 0.80')],
    # 带\n: 'replacement_buffer: 0.0'是'0.05'的前缀子串, 裸串会断言误伤
    'BUF0_ref':    [('replacement_buffer: 0.05\n', 'replacement_buffer: 0.0\n')],
    'C2_055_BUF0': [('bp2_score_boost: 0.45', 'bp2_score_boost: 0.55'),
                    ('replacement_buffer: 0.05\n', 'replacement_buffer: 0.0\n')],
}

METRIC_RE = {
    'nav': re.compile(r'最终净值:\s*([\d,]+)\s*\(总收益\s*([\d.]+)%'),
    'sharpe': re.compile(r'Sharpe:\s*([\d.]+)'),
    'mdd': re.compile(r'最大回撤:\s*([\d.]+)%'),
    'years': re.compile(r'^\s*(\d{4}):\s*([-\d.]+)%\s*\(最大回撤\s*([\d.]+)%', re.M),
}


def read(path):
    with open(path, encoding='utf-8') as f:
        return f.read()


def write(path, txt):
    with open(path, 'w', encoding='utf-8') as f:
        f.write(txt)


def snapshot(tag_dir):
    os.makedirs(tag_dir, exist_ok=True)
    for f in PROD_FILES:
        p = os.path.join(RVD, f)
        if os.path.exists(p):
            shutil.copy2(p, os.path.join(tag_dir, 'pre_' + f))
    for p in CODE_FILES:
        shutil.copy2(p, os.path.join(tag_dir, 'pre_' + os.path.basename(p)))


def restore(tag_dir):
    for f in PROD_FILES:
        p = os.path.join(RVD, f)
        pre = os.path.join(tag_dir, 'pre_' + f)
        if os.path.exists(pre):
            shutil.copy2(pre, p)
    for p in CODE_FILES:
        shutil.copy2(os.path.join(tag_dir, 'pre_' + os.path.basename(p)), p)


def run_backtest(log_path):
    with open(log_path, 'w', encoding='utf-8') as lf:
        r = subprocess.run([PY, 'bt_execution.py'], cwd=BASE,
                           stdout=lf, stderr=subprocess.STDOUT)
    return r.returncode


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


def main():
    sel = sys.argv[1:] or ['C2_050', 'C2_060', 'C2_065', 'C14_080']
    for s in sel:
        assert s in ARMS, f"未知臂: {s} (可选 {list(ARMS.keys())})"
    os.makedirs(ARMS_DIR, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    summary = []
    for i, name in enumerate(sel, 1):
        edits = ARMS[name]
        tag = os.path.join(ARMS_DIR, name)
        desc = ', '.join(f'{o} → {n}' for o, n in edits)
        print(f"\n===== [{i}/{len(sel)}] {name}: {desc} =====", flush=True)
        snapshot(tag)
        try:
            txt = read(YAML)
            for old, new in edits:
                assert txt.count(old) == 1, f"yaml旧值匹配数异常: {txt.count(old)} ({old})"
                assert txt.count(new) == 0, f"yaml已含新值: {new}"
                txt = txt.replace(old, new)
            write(YAML, txt)
            log = os.path.join(tag, f'run_{ts}.log')
            rc = run_backtest(log)
            m = parse_metrics(log)
            m['rc'] = rc
            with open(os.path.join(tag, 'metrics.json'), 'w') as f:
                json.dump(m, f, indent=1, ensure_ascii=False)
            for f in PROD_FILES:
                p = os.path.join(RVD, f)
                if os.path.exists(p):
                    shutil.copy2(p, os.path.join(tag, 'post_' + f))
            line = (f"{name}: rc={rc} NAV={m['nav']} ret={m['ret']}% "
                    f"Sharpe={m['sharpe']} MDD={m['mdd']}%")
            print(line, flush=True)
            summary.append(line)
        finally:
            restore(tag)
            txt = read(YAML)
            for old, new in edits:
                assert txt.count(old) == 1, f"yaml复原失败: {old}"
                assert txt.count(new) == 0, f"yaml残留新值: {new}"
    print("\n===== 汇总 =====")
    for s in summary:
        print(s)
    with open(os.path.join(ARMS_DIR, f'batch3_summary_{ts}.txt'), 'w') as f:
        f.write('\n'.join(summary))


if __name__ == '__main__':
    main()
