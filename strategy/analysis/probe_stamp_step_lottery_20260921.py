"""印花税阶跃时点彩票探针 (2026-09-21, 只读, 不修改任何生产文件/fp)

背景: 扁平税率bracket(probe_cost_chaos_bracket_20260921)已证 — 统一税率变化下NAV响应
单调(0.0001/0.0003/0.0005/0.0007/0.0010 → 749,756/744,196/732,689/709,452/689,108),
且生产万5处斜率不对称: 下行侧算术级(每0.0002步+0.8~1.6%), 上行侧放大~2.5×
(每0.0002步−3.2%). 而时间变化率反事实(万10→2023-08-28→万5, probe_stamp_tax_20260921)
测得−4.41% = 算术1.4%的3.1× — 与扁平bracket的行为矛盾, 假说: **时点局部化的税率
扰动(阶跃)分叉轨迹, 统一扰动不敏感**。本探针给阶跃假说n=5: 同型阶跃(0.001→D→0.0005)
在不同时点D的NAV响应 + 各臂measured/arithmetic放大比散点。

设计:
  - 臂1 cert_fresh: 扁平0.0005@新fd → 恒等检查(harness忠实性)
  - 臂2 cert_reuse: 扁平0.0005@复用fd → 恒等检查(fd复用无污染自证; 若失败,
    中间全部作废 — 静态分析已证_vectorized_backtest对fd仅get_st_timeline只读)
  - 臂3-7 step: 0.001→D→0.0005, D ∈ {2022-01-01, 2023-01-01, 2023-08-28(真实),
    2024-01-01, 2025-01-01} — 真实阶跃臂跨进程复现700,382位点核验
  - 各臂: measured ΔNAV vs 生产732,689, arithmetic ΔNAV(用生产轨迹审计CSV的
    卖出毛额×0.0005×复利因子), 放大比=measured/arithmetic

方法: 同前两探针exec补丁法(STAMP_TAX→_STAMP_ARR[i]), fd复用+首双臂恒等自证。
零写入: to_csv全局no-op + 前后md5比对。输出 /tmp/probe_step_lottery_20260921.csv
"""
import os
import sys
import gc
import hashlib
import inspect

import numpy as np
import pandas as pd

sys.path.insert(0, '/mnt/d/quant/strategy')
import bt_execution as M  # noqa: E402

_STRAT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RVR = os.path.join(_STRAT_DIR, 'rolling_validation_results')
_THIS = os.path.dirname(os.path.abspath(__file__))

PROD = {'final_value': 732689.0, 'ret': 193.08, 'sharpe': 1.1961, 'mdd': 17.92,
        'years': {2021: 9.74, 2022: -4.65, 2023: 8.99, 2024: 29.50, 2025: 50.58, 2026: 33.27}}

# 年复利因子: 年中NAV(几何均值)到终值732,689 — 由生产年收益计算
CF_MIDYEAR = {2021: 2.83, 2022: 2.77, 2023: 2.71, 2024: 2.28, 2025: 1.64, 2026: 1.15}

STEP_DATES = [np.datetime64('2022-01-01'), np.datetime64('2023-01-01'),
              np.datetime64('2023-08-28'), np.datetime64('2024-01-01'),
              np.datetime64('2025-01-01')]


def _md5_snapshot():
    out = {}
    if os.path.isdir(_RVR):
        for f in sorted(os.listdir(_RVR)):
            p = os.path.join(_RVR, f)
            if os.path.isfile(p):
                with open(p, 'rb') as fh:
                    out[f] = hashlib.md5(fh.read()).hexdigest()
    return out


def _assert_zero_writes(snap_before):
    snap_after = _md5_snapshot()
    diffs = [k for k in set(snap_before) | set(snap_after) if snap_before.get(k) != snap_after.get(k)]
    assert not diffs, f"探针写入生产产物! 差异文件: {diffs}"
    print(f"[零写入核验] rolling_validation_results {len(snap_after)}文件 md5 全部不变 ✓")


def _build_strategy(fundamental_data):
    strategy = M.Strategy(
        init_cash=M.CASH,
        fundamental_data=fundamental_data,
        sentiment_orchestrator=None,
    )
    _signals_csv = os.path.join(_RVR, 'backtest_signals.csv')
    assert os.path.exists(_signals_csv), "信号CSV缺失"
    assert not M._signals_stale(_signals_csv), "信号CSV stale — 探针拒绝触发信号重生成, 中止"
    strategy.signal_store.finalize(_signals_csv, membership_map=_membership_map)
    _idx_path = os.path.join(M.DATA_PATH, 'sh000001_qfq.csv')
    _idx_df = pd.read_csv(_idx_path, parse_dates=['datetime'])
    if M.FROMDATE:
        _idx_df = _idx_df[_idx_df['datetime'] >= M.FROMDATE]
    if M.TODATE:
        _idx_df = _idx_df[_idx_df['datetime'] <= M.TODATE]
    strategy.generate_market_regime(_idx_df)
    return strategy


def _check_identity(tag, r):
    fv = float(r['final_value'])
    print(f"[{tag}] final={fv:,.1f} sharpe={r['sharpe']:.4f} mdd={abs(r['max_drawdown'])*100:.2f}%")
    ok = True
    if abs(fv - PROD['final_value']) > 0.5:
        print(f"[{tag}] ✗ NAV偏差 {fv - PROD['final_value']:,.2f}"); ok = False
    if abs(r['sharpe'] - PROD['sharpe']) > 2e-4:
        print(f"[{tag}] ✗ Sharpe偏差 {r['sharpe'] - PROD['sharpe']:+.5f}"); ok = False
    if abs(abs(r['max_drawdown']) * 100 - PROD['mdd']) > 0.02:
        print(f"[{tag}] ✗ MDD偏差"); ok = False
    for yr, rr in sorted(r['annual_returns'].items()):
        exp = PROD['years'].get(int(yr))
        if exp is not None and abs(rr * 100 - exp) > 0.03:
            print(f"[{tag}] ✗ {yr} 年收益 {rr*100:.2f}% vs 生产 {exp}%"); ok = False
    _prod_curve = pd.read_csv(os.path.join(_RVR, 'equity_curve.csv'))
    _pnav = _prod_curve['nav'].values.astype(np.float64)
    _maxdiff = float(np.nanmax(np.abs(r['nav'] - _pnav)))
    print(f"[{tag}] 净值曲线逐日最大偏差 vs RVD: {_maxdiff:.6f}")
    if _maxdiff > 0.01:
        print(f"[{tag}] ✗ 净值曲线不一致"); ok = False
    if ok:
        print(f"[{tag}] ✓ 逐位复现生产锚点 732,689")
    return ok


if __name__ == '__main__':
    print(f"环境: {sys.executable}")
    print(f"fp核验: {M._signal_code_fingerprint()} (预期 6f1cb6b1)")
    assert M._signal_code_fingerprint() == '6f1cb6b1', "fp变化 — 生产代码已改, 探针中止"
    _orig_to_csv = pd.DataFrame.to_csv
    pd.DataFrame.to_csv = lambda self, *a, **k: None
    snap0 = _md5_snapshot()

    # 股票池过滤 (与前两探针一致)
    stock_pool_enabled = M.config.get('stock_pool.enabled', True)
    stock_codes = []
    for f in os.listdir(M.DATA_PATH):
        if f.startswith('._'):
            continue
        if f.endswith('_qfq.csv') and f != 'sh000001_qfq.csv':
            stock_codes.append(f.replace('_qfq.csv', ''))
        elif f.endswith('_hfq.csv') and f != 'sh000001_hfq.csv':
            stock_codes.append(f.replace('_hfq.csv', ''))
    _membership_map = M._load_pool_membership()
    if stock_pool_enabled and _membership_map is not None:
        _union = set()
        for _codes in _membership_map.values():
            _union |= _codes
        stock_codes = [c for c in stock_codes if c in _union]
        print(f"基本面数据加载(日历union池): {len(stock_codes)} 只")

    # 日历 + 生产轨迹卖出审计(算术基准)
    _idx_fp = os.path.join(M.DATA_PATH, 'sh000001_qfq.csv')
    _idx_df = pd.read_csv(_idx_fp, parse_dates=['datetime'])
    _idx_df = _idx_df[(_idx_df['datetime'] >= pd.Timestamp(M.FROMDATE)) &
                      (_idx_df['datetime'] <= pd.Timestamp(M.TODATE))]
    calendar = pd.DatetimeIndex(sorted(_idx_df['datetime'].unique()))
    n_dates = len(calendar)
    print(f"日历: {n_dates} 交易日 ({calendar[0].date()} ~ {calendar[-1].date()})")
    aud = pd.read_csv(os.path.join(_THIS, 'probe_stamp_audit_20260921.csv'), parse_dates=['date'])
    aud['year'] = aud['date'].dt.year

    # exec补丁: STAMP_TAX → _STAMP_ARR[i]
    src = inspect.getsource(M._vectorized_backtest)
    assert src.count('STAMP_TAX') == 2, "STAMP_TAX出现次数≠2 — 代码结构变了, 人工复核"
    src2 = src.replace('STAMP_TAX', '_STAMP_ARR[i]')
    exec(compile(src2, 'probe_step_patched', 'exec'), M.__dict__)
    print("exec补丁完成: 卖单2处税率数组化")

    print("构造FundamentalData(只读复用)...")
    fd = M.FundamentalData(M.FUNDAMENTAL_PATH, stock_codes)
    print("FundamentalData构造完成")

    # ---- 臂1+2: 恒等自证 (新fd / 复用fd) ----
    for tag in ['cert_fresh', 'cert_reuse']:
        print(f"\n===== {tag} flat 0.0005 =====")
        M._STAMP_ARR = np.full(n_dates, 0.0005)
        gc.collect()
        strat = _build_strategy(fd)
        r = M._vectorized_backtest(strat, fd, M.FROMDATE, M.TODATE, M.CASH)
        ok = _check_identity(tag, r)
        if not ok:
            print("✗ 恒等自证失败 — 中止"); sys.exit(1)
        if tag == 'cert_fresh':
            print("[自证] 新fd恒等过 — harness忠实")
        else:
            print("[自证] 复用fd恒等过 — fd复用零污染, 后续臂有效")
        del strat, r
        gc.collect()

    # ---- 臂3-7: 阶跃时点D ----
    rows = []
    for D in STEP_DATES:
        label = str(D)[:10]
        print(f"\n===== step {label} (0.001→D→0.0005) =====")
        M._STAMP_ARR = np.where(calendar.values < D, 0.001, 0.0005).astype(np.float64)
        gc.collect()
        strat = _build_strategy(fd)
        r = M._vectorized_backtest(strat, fd, M.FROMDATE, M.TODATE, M.CASH)
        fv = float(r['final_value'])
        # 算术基准: 生产轨迹上D之前卖单补税0.0005, 年中复利
        pre = aud[aud['date'].values < D]
        extra_first = 0.0005 * pre['proceeds'].sum()
        extra_comp = sum(0.0005 * g['proceeds'].sum() * CF_MIDYEAR.get(int(yr), 1.0)
                         for yr, g in pre.groupby('year'))
        meas = fv - PROD['final_value']
        ratio = meas / -extra_comp if extra_comp > 0 else float('nan')
        rows.append({'step': label, 'final': fv, 'meas_dnav': meas,
                     'arith_first': extra_first, 'arith_comp': extra_comp,
                     'amp_ratio': ratio, 'sharpe': r['sharpe'],
                     'mdd': abs(r['max_drawdown']) * 100})
        print(f"[step {label}] final={fv:,.0f} (Δ{meas:+,.0f} = {meas/PROD['final_value']*100:+.2f}%) "
              f"sharpe={r['sharpe']:.4f} mdd={abs(r['max_drawdown'])*100:.2f}%")
        print(f"[step {label}] 算术: 一阶{extra_first:,.0f}元 / 复利{extra_comp:,.0f}元 "
              f"→ 放大比 measured/arithmetic = {ratio:.2f}")
        for yr in sorted(r['annual_returns']):
            print(f"  {yr}: {r['annual_returns'][yr]*100:+.2f}%", end='')
        print()
        del strat, r
        gc.collect()

    res = pd.DataFrame(rows)
    print("\n===== 阶跃时点汇总 =====")
    print(res.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print(f"\n[彩票度量] 放大比散点: mean={res['amp_ratio'].mean():.2f} "
          f"std={res['amp_ratio'].std():.2f} min={res['amp_ratio'].min():.2f} max={res['amp_ratio'].max():.2f}")
    _orig_to_csv(res, '/tmp/probe_step_lottery_20260921.csv', index=False)

    _assert_zero_writes(snap0)
    pd.DataFrame.to_csv = _orig_to_csv
    print("\n探针完成 (零生产写入, fp未动)")
