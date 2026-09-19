#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""C1 忠实臂驱动 (2026-09-19): yaml ml.blend_weight 0.4→w1, 全链信号regen+回测.
与v3注入臂(排名层偏导, 门槛冻结)不同: 本臂重新生成信号 — 动态阈值/买入标志
/排名全部在w1下重算, 是诚实的端到端检验。成本~2.5-3h(regen)+~25min(bt)。
ml节非豁免 → fp变化 → regen自动触发; raw因子缓存键也含ml节 → raw重算(保守)。
协议: 快照(yaml+产物+fp) → 编辑yaml → 回测(regen) → 抓四指标 → 复原
(yaml/产物/fp侧车/信号CSV从_baseline_sig) → 断言最终态md5==基线。
用法: python analysis/run_c1_faithful_20260919.py 0.55
"""
import os, sys, json, shutil, subprocess, re, hashlib
from datetime import datetime

BASE = '/mnt/d/quant/strategy'
RVD = os.path.join(BASE, 'rolling_validation_results')
YAML = os.path.join(BASE, 'config', 'factor_config.yaml')
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


def md5(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 24), b''):
            h.update(chunk)
    return h.hexdigest()


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
    w1 = float(sys.argv[1]) if len(sys.argv) > 1 else 0.55
    old, new = f'blend_weight: {W0}', f'blend_weight: {w1}'
    tag = os.path.join(ARMS_DIR, f'C1_faithful_{str(w1).replace(".", "_")}')
    os.makedirs(tag, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')

    # 前置断言: 生产态=基线
    prod_sig = os.path.join(RVD, 'backtest_signals.csv')
    base_sig = os.path.join(ARMS_DIR, '_baseline_sig', 'backtest_signals.csv')
    assert os.path.exists(base_sig), "无基线信号归档(先跑v3驱动)"
    prod_md5, base_md5 = md5(prod_sig), md5(base_sig)
    assert prod_md5 == base_md5, f"生产信号≠基线归档, 拒跑: {prod_md5} vs {base_md5}"
    txt = read(YAML)
    assert txt.count(old) == 1, f"yaml旧值匹配数异常: {txt.count(old)}"
    assert txt.count(new) == 0, f"yaml已含新值: {new}"
    fp0 = read(os.path.join(RVD, '.signal_code_fp'))
    print(f"前置断言全过: 生产=基线({base_md5[:8]}), fp={fp0.strip()}", flush=True)

    # 快照
    shutil.copy2(YAML, os.path.join(tag, 'pre_factor_config.yaml'))
    for f in PROD_FILES:
        p = os.path.join(RVD, f)
        if os.path.exists(p):
            shutil.copy2(p, os.path.join(tag, 'pre_' + f))

    try:
        with open(YAML, 'w', encoding='utf-8') as f:
            f.write(txt.replace(old, new))
        assert read(YAML).count(new) == 1
        print(f"yaml已编辑: {old} → {new} (fp将变化, regen自动触发)", flush=True)
        log = os.path.join(tag, f'run_{ts}.log')
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
        # 忠实臂生成的信号CSV归档(备裁决层取证)
        shutil.copy2(prod_sig, os.path.join(tag, 'post_backtest_signals.csv'))
        print(f"C1_faithful_{w1}: rc={rc} NAV={m['nav']} ret={m['ret']}% "
              f"Sharpe={m['sharpe']} MDD={m['mdd']}%", flush=True)
    finally:
        # 复原: yaml/产物/fp侧车/信号CSV
        shutil.copy2(os.path.join(tag, 'pre_factor_config.yaml'), YAML)
        for f in PROD_FILES:
            pre = os.path.join(tag, 'pre_' + f)
            if os.path.exists(pre):
                shutil.copy2(pre, os.path.join(RVD, f))
        shutil.copy2(base_sig, prod_sig)
        assert read(YAML).count(old) == 1 and read(YAML).count(new) == 0, "yaml复原失败!"
        assert read(os.path.join(RVD, '.signal_code_fp')) == fp0, "fp侧车复原失败!"
        assert md5(prod_sig) == base_md5, "信号CSV复原失败!"
        print("复原完成并断言通过: yaml/fp/信号/产物全回基线", flush=True)


if __name__ == '__main__':
    main()
