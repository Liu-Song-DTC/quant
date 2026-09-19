#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""批次2旋钮臂驱动 (2026-09-19): 纯yaml豁免节编辑 → 纯回测 → 四指标 → 复原.
协议同opt_c5_driver(快照→应用→跑→抓四指标→复原→归档); 差异: 不touch信号文件
(信号fp不变→每臂复用已有信号, ~20min/臂), 不patch代码, years正则已修(re.M)。
用法: python analysis/run_batch2_knobs_20260919.py [臂名1 臂名2 ...]  (缺省=全部9臂)
臂清单(2026-09-19 probe-verified live, 全fp豁免节):
  C3_off   entry_chan_gate bearhard→off       (E-N5采纳bearhard; hard已败不跑)
  C4_025   fast_min_score 0.30→0.25           (E-O2已否决0.35/0.40, 只跑反方向)
  C6_010/C6_020  rank_decay 0.15→0.10|0.20    (0.15=旧网格峰值, 峰周细化)
  C9_015/C9_025  hold_threshold 0.2→0.15|0.25 (P2已0.3→0.2)
  C11_lb60 volatility_control lookback 20→60
死旋钮已剔除: C13 dynamic_rebalance全库零生产reader; C11 blend_weight/long_lookback
零reader(config_loader只映射enabled+lookback_period); **C10 clb=实操零触发**
(154/154选股行clb=0: loss_floor -1.5%日损失P≈5%→4连≈0.01事件/全史, 且E-H6熔断
已覆盖回撤降仓语义) → threshold臂=no-op废弃。
"""
import os, sys, json, shutil, subprocess, re
from datetime import datetime

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RVD = os.path.join(BASE, 'rolling_validation_results')
YAML = os.path.join(BASE, 'config', 'factor_config.yaml')
ARMS_DIR = os.path.join(BASE, 'arms_20260919')
PY = '/mnt/d/quant/.venv/bin/python'

# 信号文件不入快照: 纯yaml豁免节编辑不改fp, 信号必然复用(逐臂断言sidecar不变)
PROD_FILES = ['portfolio_selections.csv', 'trade_realized.csv', 'equity_curve.csv',
              'regime_state.csv', 'yaogu_watchlist.csv', '.signal_code_fp']
CODE_FILES = [YAML]

ARMS = {
    'C2_035':   ('bp2_score_boost: 0.45', 'bp2_score_boost: 0.35'),
    'C2_055':   ('bp2_score_boost: 0.45', 'bp2_score_boost: 0.55'),
    'C8_005':   ('turnover_bonus: 0.1', 'turnover_bonus: 0.05'),
    'C8_015':   ('turnover_bonus: 0.1', 'turnover_bonus: 0.15'),
    'C3_off':   ('entry_chan_gate: bearhard', 'entry_chan_gate: off'),
    'C4_025':   ('fast_min_score: 0.30', 'fast_min_score: 0.25'),
    'C6_010':   ('rank_decay: 0.15', 'rank_decay: 0.10'),
    'C6_020':   ('rank_decay: 0.15', 'rank_decay: 0.20'),
    'C9_015':   ('hold_threshold: 0.2', 'hold_threshold: 0.15'),
    'C9_025':   ('hold_threshold: 0.2', 'hold_threshold: 0.25'),
    'C11_lb60': ('lookback_period: 20', 'lookback_period: 60'),
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
    sel = sys.argv[1:] or list(ARMS.keys())
    for s in sel:
        assert s in ARMS, f"未知臂: {s} (可选 {list(ARMS.keys())})"
    os.makedirs(ARMS_DIR, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    summary = []
    for i, name in enumerate(sel, 1):
        old, new = ARMS[name]
        tag = os.path.join(ARMS_DIR, name)
        print(f"\n===== [{i}/{len(sel)}] {name}: {old} → {new} =====", flush=True)
        snapshot(tag)
        applied = False
        try:
            txt = read(YAML)
            assert txt.count(old) == 1, f"yaml旧值匹配数异常: {txt.count(old)}"
            assert txt.count(new) == 0, f"yaml已含新值: {new}"
            write(YAML, txt.replace(old, new))
            applied = True
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
            assert txt.count(old) == 1, "yaml复原失败!"
            assert txt.count(new) == 0, "yaml残留新值!"
    print("\n===== 汇总 =====")
    for s in summary:
        print(s)
    with open(os.path.join(ARMS_DIR, f'batch2_summary_{ts}.txt'), 'w') as f:
        f.write('\n'.join(summary))


if __name__ == '__main__':
    main()
