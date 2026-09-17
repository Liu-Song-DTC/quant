#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""C5不对称缓冲臂驱动 (2026-09-17): yaml buffer=0.05 + portfolio.py机制patch变体。
协议同opt_arm_driver(快照→应用→跑→抓四指标→复原→归档), 增加代码patch支持。
用法: python analysis/opt_c5_driver_20260917.py <spec.json>
spec: [{"name":"C5b_dd05","patch":"C5b","dd":0.05}, {"name":"A_repro","patch":null}, ...]
  patch值: "C5b"|"C5d"|"C5c"|null(纯yaml/无操作臂)
所有臂yaml统一 replacement_buffer 0.0→0.05 (A_repro除外: 不编辑yaml)。
"""
import os, sys, json, shutil, subprocess, re
from datetime import datetime

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RVD = os.path.join(BASE, 'rolling_validation_results')
YAML = os.path.join(BASE, 'config', 'factor_config.yaml')
PORT = os.path.join(BASE, 'core', 'portfolio.py')
ARMS_DIR = os.path.join(BASE, 'arms_20260917')
PY = '/mnt/d/quant/.venv/bin/python'

PROD_FILES = ['portfolio_selections.csv', 'trade_realized.csv', 'equity_curve.csv',
              'regime_state.csv', 'yaogu_watchlist.csv', 'backtest_signals.csv', '.signal_code_fp']
CODE_FILES = [PORT, YAML]

OLD_BLOCK = (
    "            # C实验: 换仓缓冲 — 已持仓δ保护(有卖点不保护), 0=关闭\n"
    "            repl_buffer = self.replacement_buffer if (c['is_held'] and c.get('chan_sell_point', 0) == 0) else 0.0")

PATCHES = {
    'C5b': lambda dd: OLD_BLOCK + (
        "\n            if drawdown > {dd}:  # C5b: 组合回撤期解除缓冲, 滞跌名可被顶替"
        "\n                repl_buffer = 0.0").format(dd=dd),
    'C5d': OLD_BLOCK + (
        "\n            if bear_risk_fast:  # C5d: FAST期解除缓冲(槽位少, 保护加剧集中)"
        "\n                repl_buffer = 0.0"),
    'C5c': OLD_BLOCK + (
        "\n            _rc = getattr(self, '_a1_raw_cost', {}).get(code)"
        "\n            if not (_rc and len(_rc) >= 2 and _rc[0] > 0 and prices.get(code, 0) > _rc[1]):"
        "\n                repl_buffer = 0.0  # C5c: 仅盈利持仓受保护(亏损名让位)"),
}

METRIC_RE = {
    'nav': re.compile(r'最终净值:\s*([\d,]+)\s*\(总收益\s*([\d.]+)%'),
    'sharpe': re.compile(r'Sharpe:\s*([\d.]+)'),
    'mdd': re.compile(r'最大回撤:\s*([\d.]+)%'),
    'years': re.compile(r'^\s*(\d{4}):\s*([-\d.]+)%\s*\(最大回撤\s*([\d.]+)%'),
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
    with open(sys.argv[1], encoding='utf-8') as f:
        arms = json.load(f)
    os.makedirs(ARMS_DIR, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    summary = []
    # 信号基线滚动目录 (2026-09-17修复): 信号代码指纹用mtime而非内容, copy2复原会
    # 改变mtime→每臂都触发~1.2h信号重生成。修复: 若某臂的运行实际重生成过信号
    # (post侧car指纹≠pre侧), 把重生成的信号CSV+sidecar滚动为本批基线, 后续臂复用。
    sig_base = os.path.join(ARMS_DIR, '_baseline_sig')
    os.makedirs(sig_base, exist_ok=True)
    for i, arm in enumerate(arms, 1):
        name, patch, dd = arm['name'], arm.get('patch'), arm.get('dd')
        tag = os.path.join(ARMS_DIR, name)
        os.makedirs(tag, exist_ok=True)
        print(f"\n===== [{i}/{len(arms)}] {name} =====", flush=True)
        # 滚动基线: 若存在, 先同步到生产 (保证每臂起点=最新一致信号态)
        for f in ['backtest_signals.csv', '.signal_code_fp']:
            src = os.path.join(sig_base, f)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(RVD, f))
        snapshot(tag)
        applied_yaml = False
        try:
            # yaml: A_repro不编辑; 其余臂统一 buffer 0.05
            if patch is not None or name.startswith('C5_ref'):
                txt = read(YAML)
                assert txt.count('replacement_buffer: 0.0') == 1
                write(YAML, txt.replace('replacement_buffer: 0.0', 'replacement_buffer: 0.05'))
                applied_yaml = True
            # code patch
            if patch:
                ptxt = read(PORT)
                assert ptxt.count(OLD_BLOCK) == 1, f"OLD_BLOCK匹配数异常: {ptxt.count(OLD_BLOCK)}"
                new = PATCHES[patch](dd) if patch == 'C5b' else PATCHES[patch]
                write(PORT, ptxt.replace(OLD_BLOCK, new))
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
            # 信号重生成检测 → 滚动基线
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
            assert read(PORT).count(OLD_BLOCK) == 1, "portfolio.py复原失败!"
            assert read(YAML).count('replacement_buffer: 0.0') == 1, "yaml复原失败!"
        for f in ['pre_backtest_signals.csv', 'post_backtest_signals.csv']:
            p = os.path.join(tag, f)
            if os.path.exists(p):
                os.remove(p)
    print("\n===== 汇总 =====")
    for s in summary:
        print(s)
    with open(os.path.join(ARMS_DIR, f'c5_summary_{ts}.txt'), 'w') as f:
        f.write('\n'.join(summary))


if __name__ == '__main__':
    main()
