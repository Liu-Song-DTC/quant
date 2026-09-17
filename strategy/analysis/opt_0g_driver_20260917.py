#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""0g dragon_tiger OFF臂驱动 (2026-09-17): 关闭龙虎榜个股加成, 量化其总贡献
(诚实部分+下载日累计前视artifact混合) 并给出PIT诚实基线。
协议同C5驱动: 快照→patch signal_engine.py→跑(信号fp变化自动强制重生成~1.2h)
→抓四指标→字节复原→归档 arms_20260917/0g_dt_off/。
用法: python analysis/opt_0g_driver_20260917.py
"""
import os, json, shutil, subprocess, re
from datetime import datetime

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RVD = os.path.join(BASE, 'rolling_validation_results')
SIGE = os.path.join(BASE, 'core', 'signal_engine.py')
ARMS_DIR = os.path.join(BASE, 'arms_20260917', '0g_dt_off')
PY = '/mnt/d/quant/.venv/bin/python'

PROD_FILES = ['backtest_signals.csv', '.signal_code_fp', 'equity_curve.csv',
              'portfolio_selections.csv', 'trade_realized.csv', 'regime_state.csv',
              'yaogu_watchlist.csv']

OLD_BLOCK = (
    "                # 个股级: 龙虎榜独立买点信号 — 机构大买不经过因子筛选, 直接强化\n"
    "                dt_signal = np.zeros(n)\n"
    "                if code:\n"
    "                    for i in range(60, n):\n"
    "                        dt_sig = self._alt_data.get_dragon_tiger_signal(code, (pd.to_datetime(dates[i]) - pd.Timedelta(days=1)).date())\n"
    "                        if abs(dt_sig) > 0.01:\n"
    "                            dt_signal[i] = dt_sig\n"
    "                            adjusted_score[i] += dt_sig * 0.30\n"
    "                            if self._diag is not None and i == n - 1:\n"
    "                                self._diag.record_alt_data(dragon_tiger=True)")

NEW_BLOCK = (
    "                # 个股级: 龙虎榜独立买点信号 — 0g臂: 关闭(PIT化验证)\n"
    "                # 原机制用下载日全窗口累计值(含未来事件, 前视artifact已量化\n"
    "                # +44,240/5.1%尾部漂移), 此臂量化总贡献并给出PIT诚实基线。\n"
    "                dt_signal = np.zeros(n)")

METRIC_RE = {
    'nav': re.compile(r'最终净值:\s*([\d,]+)\s*\(总收益\s*([\d.]+)%'),
    'sharpe': re.compile(r'Sharpe:\s*([\d.]+)'),
    'mdd': re.compile(r'最大回撤:\s*([\d.]+)%'),
    'years': re.compile(r'^\s*(\d{4}):\s*([-\d.]+)%\s*\(最大回撤\s*([\d.]+)%'),
}


def read(p):
    with open(p, encoding='utf-8') as f:
        return f.read()


def write(p, txt):
    with open(p, 'w', encoding='utf-8') as f:
        f.write(txt)


def snapshot(tag_dir):
    os.makedirs(tag_dir, exist_ok=True)
    for f in PROD_FILES:
        p = os.path.join(RVD, f)
        if os.path.exists(p):
            shutil.copy2(p, os.path.join(tag_dir, 'pre_' + f))
    shutil.copy2(SIGE, os.path.join(tag_dir, 'pre_signal_engine.py'))


def restore(tag_dir):
    for f in PROD_FILES:
        p = os.path.join(RVD, f)
        pre = os.path.join(tag_dir, 'pre_' + f)
        if os.path.exists(pre):
            shutil.copy2(pre, p)
    shutil.copy2(os.path.join(tag_dir, 'pre_signal_engine.py'), SIGE)


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
    os.makedirs(ARMS_DIR, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    snapshot(ARMS_DIR)
    try:
        ptxt = read(SIGE)
        assert ptxt.count(OLD_BLOCK) == 1, f"OLD_BLOCK匹配数异常: {ptxt.count(OLD_BLOCK)}"
        write(SIGE, ptxt.replace(OLD_BLOCK, NEW_BLOCK))
        log = os.path.join(ARMS_DIR, f'run_{ts}.log')
        with open(log, 'w', encoding='utf-8') as lf:
            rc = subprocess.run([PY, 'bt_execution.py'], cwd=BASE,
                                stdout=lf, stderr=subprocess.STDOUT).returncode
        m = parse_metrics(log)
        m['rc'] = rc
        with open(os.path.join(ARMS_DIR, 'metrics.json'), 'w') as f:
            json.dump(m, f, indent=1, ensure_ascii=False)
        for f in PROD_FILES:
            p = os.path.join(RVD, f)
            if os.path.exists(p):
                shutil.copy2(p, os.path.join(ARMS_DIR, 'post_' + f))
        print(f"0g_dt_off: rc={rc} NAV={m['nav']} ret={m['ret']}% "
              f"Sharpe={m['sharpe']} MDD={m['mdd']}%")
        print(f"年份: {m['years']}")
    finally:
        restore(ARMS_DIR)
        assert read(SIGE).count(OLD_BLOCK) == 1, "signal_engine.py复原失败!"
        print("生产状态已字节复原")


if __name__ == '__main__':
    main()
