#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Alpha 2.0 冷跑臂驱动 (2026-09-17): 串行跑配置臂(豁免节, 无信号重生成)。
协议: 快照生产产物 → 应用yaml编辑(逐串替换, 记录原文) → 跑bt_execution →
抓四指标 → 恢复yaml+产物 → 归档臂产物。生产零残留。
用法: python analysis/opt_arm_driver_20260917.py <arms_spec.json>
arms_spec.json: [{"name":"0d_impact","edits":[["impact_cost_enabled: false","impact_cost_enabled: true"]]}, ...]
"""
import os, sys, json, shutil, subprocess, re, glob
from datetime import datetime

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RVD = os.path.join(BASE, 'rolling_validation_results')
YAML = os.path.join(BASE, 'config', 'factor_config.yaml')
ARMS_DIR = os.path.join(BASE, 'arms_20260917')
PY = '/mnt/d/quant/.venv/bin/python'

PROD_FILES = ['portfolio_selections.csv', 'trade_realized.csv', 'equity_curve.csv',
              'regime_state.csv', 'yaogu_watchlist.csv', 'backtest_signals.csv', '.signal_code_fp']
METRIC_RE = {
    'nav': re.compile(r'最终净值:\s*([\d,]+)\s*\(总收益\s*([\d.]+)%'),
    'sharpe': re.compile(r'Sharpe:\s*([\d.]+)'),
    'mdd': re.compile(r'最大回撤:\s*([\d.]+)%'),
    'years': re.compile(r'^\s*(\d{4}):\s*([-\d.]+)%\s*\(最大回撤\s*([\d.]+)%'),
}


def snapshot_files(tag_dir):
    os.makedirs(tag_dir, exist_ok=True)
    for f in PROD_FILES:
        p = os.path.join(RVD, f)
        if os.path.exists(p):
            shutil.copy2(p, os.path.join(tag_dir, 'pre_' + f))


def restore_files(tag_dir):
    for f in PROD_FILES:
        p = os.path.join(RVD, f)
        pre = os.path.join(tag_dir, 'pre_' + f)
        if os.path.exists(pre):
            shutil.copy2(pre, p)


def run_backtest(log_path):
    with open(log_path, 'w', encoding='utf-8') as lf:
        r = subprocess.run([PY, 'bt_execution.py'], cwd=BASE,
                           stdout=lf, stderr=subprocess.STDOUT)
    return r.returncode


def parse_metrics(log_path):
    m = {'nav': None, 'ret': None, 'sharpe': None, 'mdd': None, 'years': {}}
    with open(log_path, encoding='utf-8', errors='replace') as f:
        txt = f.read()
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


def apply_yaml_edits(edits):
    """edits: [(old_substr, new_substr), ...] — 必须唯一匹配。返回原文记录。"""
    with open(YAML, encoding='utf-8') as f:
        txt = f.read()
    originals = []
    for old, new in edits:
        n = txt.count(old)
        if n != 1:
            raise ValueError(f"yaml edit not unique({n}): {old!r}")
        originals.append((old, new))
        txt = txt.replace(old, new)
    with open(YAML, 'w', encoding='utf-8') as f:
        f.write(txt)
    return originals


def restore_yaml(originals):
    with open(YAML, encoding='utf-8') as f:
        txt = f.read()
    for old, new in reversed(originals):
        assert txt.count(new) == 1, f"yaml restore failed for {new!r}"
        txt = txt.replace(new, old)
    with open(YAML, 'w', encoding='utf-8') as f:
        f.write(txt)


def main():
    spec_path = sys.argv[1]
    with open(spec_path, encoding='utf-8') as f:
        arms = json.load(f)
    os.makedirs(ARMS_DIR, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    summary = []
    for i, arm in enumerate(arms, 1):
        name = arm['name']
        tag = os.path.join(ARMS_DIR, name)
        os.makedirs(tag, exist_ok=True)
        print(f"\n===== [{i}/{len(arms)}] {name} =====", flush=True)
        snapshot_files(tag)
        originals = apply_yaml_edits(arm['edits'])
        try:
            log = os.path.join(tag, f'run_{ts}.log')
            rc = run_backtest(log)
            m = parse_metrics(log)
            m['rc'] = rc
            with open(os.path.join(tag, 'metrics.json'), 'w') as f:
                json.dump(m, f, indent=1, ensure_ascii=False)
            # 臂产物快照
            for f in PROD_FILES:
                p = os.path.join(RVD, f)
                if os.path.exists(p):
                    shutil.copy2(p, os.path.join(tag, 'post_' + f))
            line = (f"{name}: rc={rc} NAV={m['nav']} ret={m['ret']}% "
                    f"Sharpe={m['sharpe']} MDD={m['mdd']}%")
            print(line, flush=True)
            summary.append(line)
            # A臂复现烟测: 现态基线868,611(9/15刷新后9/14区间, =935,444-66,833漂移,
            # run-1 equity_curve终值868,610.96逐位证实), 偏差>1000中止(环境状态可疑)
            if name.startswith('A_repro'):
                exp = 868611
                if m['nav'] is None or abs(m['nav'] - exp) > 1000:
                    print(f"!!! A臂复现失败: {m['nav']} vs {exp} — 中止队列, 环境状态可疑", flush=True)
                    sys.exit(2)
                print(f"A臂复现通过: {m['nav']} vs {exp} ✓", flush=True)
        finally:
            restore_yaml(originals)
            restore_files(tag)
        # 删除臂快照中的pre/post大CSV(1.66G×2×N), 只留metrics+log+selections
        for f in ['pre_backtest_signals.csv', 'post_backtest_signals.csv']:
            p = os.path.join(tag, f)
            if os.path.exists(p):
                os.remove(p)
    print("\n===== 汇总 =====")
    for s in summary:
        print(s)
    with open(os.path.join(ARMS_DIR, f'summary_{ts}.txt'), 'w') as f:
        f.write('\n'.join(summary))


if __name__ == '__main__':
    main()
