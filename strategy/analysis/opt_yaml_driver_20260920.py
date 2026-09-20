#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""yaml/代码旋钮臂驱动 v2 (2026-09-20晚): 免指纹节旋钮bracket, 信号复用~17min/臂。
v1只支持yaml单替换; v2支持任意file的多对(旧,新)字符串替换 — portfolio.py硬编码系数臂
(fp仅覆盖signal_engine/ml_predictor/bt_execution/因子文件/季度权重, portfolio.py不在内,
 见bt_execution.py:152 _signal_code_fingerprint → 代码臂同17min/臂)。
协议: 快照→替换→跑→抓四指标→复原→断言。每对(旧,新)须唯一匹配(count==1)且旧=生产现值。
用法: python analysis/opt_yaml_driver_20260920.py <spec.json>
spec: [{"name":"C15_mh3", "edits":[["    min_hold_days: 5","    min_hold_days: 3"]]},
       {"name":"C16_bonus0", "file":"core/portfolio.py",
        "edits":[["if bp == 1 and sl >= 1:\n                additive += 0.08",
                  "if bp == 1 and sl >= 1:\n                additive += 0.0"], ...]}, ...]
file缺省=config/factor_config.yaml。
"""
import os, sys, json, shutil, subprocess, re
from datetime import datetime

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RVD = os.path.join(BASE, 'rolling_validation_results')
YAML = 'config/factor_config.yaml'
ARMS_DIR = os.path.join(BASE, 'arms_20260920')
PY = '/mnt/d/quant/.venv/bin/python'
ENV = dict(os.environ, QUANT_ALT_NO_AUTOREFRESH='1')  # 数据态冻结(臂运行零alt写入)

PROD_FILES = ['portfolio_selections.csv', 'trade_realized.csv', 'equity_curve.csv',
              'regime_state.csv', 'yaogu_watchlist.csv', 'backtest_signals.csv', '.signal_code_fp']
CODE_FILES = ['config/factor_config.yaml', 'core/portfolio.py']

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
    for rel in CODE_FILES:
        p = os.path.join(BASE, rel)
        shutil.copy2(p, os.path.join(tag_dir, 'pre_' + os.path.basename(p)))


def restore(tag_dir):
    for f in PROD_FILES:
        p = os.path.join(RVD, f)
        pre = os.path.join(tag_dir, 'pre_' + f)
        if os.path.exists(pre):
            shutil.copy2(pre, p)
    for rel in CODE_FILES:
        p = os.path.join(BASE, rel)
        shutil.copy2(os.path.join(tag_dir, 'pre_' + os.path.basename(p)), p)


def run_backtest(log_path):
    with open(log_path, 'w', encoding='utf-8') as lf:
        r = subprocess.run([PY, 'bt_execution.py'], cwd=BASE,
                           stdout=lf, stderr=subprocess.STDOUT, env=ENV)
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
    with open(sys.argv[1], encoding='utf-8') as f:
        arms = json.load(f)
    os.makedirs(ARMS_DIR, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    summary = []
    sig_base = os.path.join(ARMS_DIR, '_baseline_sig')
    os.makedirs(sig_base, exist_ok=True)
    for i, arm in enumerate(arms, 1):
        name = arm['name']
        tgt = os.path.join(BASE, arm.get('file', YAML))
        pairs = arm.get('edits') or [arm['edit']]
        tag = os.path.join(ARMS_DIR, name)
        os.makedirs(tag, exist_ok=True)
        print(f"\n===== [{i}/{len(arms)}] {name} ({arm.get('file', YAML)}, {len(pairs)}对替换) =====", flush=True)
        for f in ['backtest_signals.csv', '.signal_code_fp']:
            src = os.path.join(sig_base, f)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(RVD, f))
        snapshot(tag)
        try:
            txt = read(tgt)
            for old, new in pairs:
                assert txt.count(old) == 1, f"{name}: {old[:60]!r} 匹配数异常: {txt.count(old)}"
                txt = txt.replace(old, new)
            write(tgt, txt)
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
            pre_fp = os.path.join(tag, 'pre_.signal_code_fp')
            post_fp = os.path.join(RVD, '.signal_code_fp')
            if os.path.exists(pre_fp) and os.path.exists(post_fp) and \
                    read(pre_fp) != read(post_fp):
                shutil.copy2(post_fp, os.path.join(sig_base, '.signal_code_fp'))
                shutil.copy2(os.path.join(RVD, 'backtest_signals.csv'),
                             os.path.join(sig_base, 'backtest_signals.csv'))
                print(f"[滚动基线] {name}重生成过信号 → 信号基线已前移", flush=True)
            line = (f"{name}: rc={rc} NAV={m['nav']} ret={m['ret']}% "
                    f"Sharpe={m['sharpe']} MDD={m['mdd']}%")
            print(line, flush=True)
            summary.append(line)
        finally:
            restore(tag)
            rtxt = read(tgt)
            for old, new in pairs:
                assert rtxt.count(old) == 1, f"{name}: 复原失败! {old[:60]!r} count={rtxt.count(old)}"
        for f in ['pre_backtest_signals.csv', 'post_backtest_signals.csv']:
            p = os.path.join(tag, f)
            if os.path.exists(p):
                os.remove(p)
    print("\n===== 汇总 =====")
    for s in summary:
        print(s)
    with open(os.path.join(ARMS_DIR, f'yaml_summary_{ts}.txt'), 'w') as f:
        f.write('\n'.join(summary))


if __name__ == '__main__':
    main()
