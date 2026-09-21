"""滑点扁平bracket: 成本栈最后一块未量化 (2026-09-22, 只读, 不修改任何生产文件/fp)

背景: 成本模型审计(9/21-22)已闭合印花税+佣金两大项 — 净对账≈诚实。但成本栈
合计≈23% NAV中最大单项是滑点: 生产slip_rate=万10双边(0.001), impact_cost_enabled
=false → 滑点是价差+冲击+执行摩擦的唯一代理。一阶: 0.001×双边成交额≈13% NAV。
本探针量化滑点假设的NAV敏感度, 并给出诚实画像项: 实盘(50万账户/开盘限价单,
价差万2-5) vs 模型(万10)的保守缓冲带。

设计: 同印花税扁平bracket的exec补丁法 — 替换滑点乘法三处
  (1.0 - slip_rate) ×2 → (1.0 - _SLIP_ARR[i])  [Step A全卖+Step B减仓]
  (1.0 + slip_rate) ×1 → (1.0 + _SLIP_ARR[i])  [Step C买入]
臂: [0.001(恒等), 0.0005, 0.0015, 0.0020, 0.001(终检恒等)] — 终检臂=fd复用零污染自证。
输出: /tmp/probe_slippage_results_20260922.csv + curves。
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

ARMS = [0.0010, 0.0005, 0.0015, 0.0020, 0.0010]  # 恒等 → 低→高, 终检0.0010在最后
PROD_RATE = 0.0010

# 年复利因子: 年中NAV到终值 — 由生产年收益计算(同step探针)
CF_MIDYEAR = {2021: 2.83, 2022: 2.77, 2023: 2.71, 2024: 2.28, 2025: 1.64, 2026: 1.15}


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

    # 股票池过滤 (与前三探针一致)
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

    # 日历
    _idx_fp = os.path.join(M.DATA_PATH, 'sh000001_qfq.csv')
    _idx_df = pd.read_csv(_idx_fp, parse_dates=['datetime'])
    _idx_df = _idx_df[(_idx_df['datetime'] >= pd.Timestamp(M.FROMDATE)) &
                      (_idx_df['datetime'] <= pd.Timestamp(M.TODATE))]
    calendar = pd.DatetimeIndex(sorted(_idx_df['datetime'].unique()))
    n_dates = len(calendar)
    print(f"日历: {n_dates} 交易日 ({calendar[0].date()} ~ {calendar[-1].date()})")

    # exec补丁: 滑点乘法三处 → _SLIP_ARR[i]
    src = inspect.getsource(M._vectorized_backtest)
    n_sell = src.count('(1.0 - slip_rate)')
    n_buy = src.count('(1.0 + slip_rate)')
    assert n_sell == 2 and n_buy == 1, \
        f"滑点乘法出现次数异常 (sell={n_sell}, buy={n_buy}, 预期 2/1) — 代码结构变了, 人工复核"
    src2 = src.replace('(1.0 - slip_rate)', '(1.0 - _SLIP_ARR[i])')
    src2 = src2.replace('(1.0 + slip_rate)', '(1.0 + _SLIP_ARR[i])')
    exec(compile(src2, 'probe_slippage_patched', 'exec'), M.__dict__)
    print("exec补丁完成: 滑点乘法3处数组化 (卖2买1)")

    print("构造FundamentalData(只读复用)...")
    fd = M.FundamentalData(M.FUNDAMENTAL_PATH, stock_codes)
    print("FundamentalData构造完成")

    # 算术基准: 双边成交额≈2×卖单毛额(稳态: Σ买≈Σ卖+初始现金-终现金≈Σ卖)
    aud = pd.read_csv(os.path.join(_THIS, 'probe_stamp_audit_20260921.csv'), parse_dates=['date'])
    aud['year'] = aud['date'].dt.year

    results = []
    curves = {'date': [d.strftime('%Y-%m-%d') for d in calendar]}
    first_identity_ok = None
    for arm_idx, rate in enumerate(ARMS):
        tag = f"arm{arm_idx+1} slip {rate:.4f}"
        print(f"\n===== {tag} =====")
        M._SLIP_ARR = np.full(n_dates, rate)
        gc.collect()
        strat = _build_strategy(fd)
        r = M._vectorized_backtest(strat, fd, M.FROMDATE, M.TODATE, M.CASH)
        fv = float(r['final_value'])
        # 算术Δ: Δrate × 2 × 各年卖单毛额 × 年中复利因子
        drate = rate - PROD_RATE
        arith = sum(drate * 2.0 * g['proceeds'].sum() * CF_MIDYEAR.get(int(yr), 1.0)
                    for yr, g in aud.groupby('year'))
        meas = fv - PROD['final_value']
        amp = meas / arith if abs(arith) > 1.0 else float('nan')
        results.append({'arm': arm_idx + 1, 'rate': rate,
                        'final_value': fv, 'ret_pct': (fv / M.CASH - 1) * 100,
                        'sharpe': r['sharpe'], 'mdd_pct': abs(r['max_drawdown']) * 100,
                        'meas_dnav': meas, 'arith_comp': arith, 'amp_ratio': amp})
        if rate == PROD_RATE and arm_idx > 0:
            curves['nav_terminal_identity'] = r['nav']
        else:
            curves[f'nav_{rate:.4f}'] = r['nav']
        if rate == PROD_RATE:
            ident_ok = _check_identity(tag, r)
            if arm_idx == 0:
                first_identity_ok = ident_ok
                if not first_identity_ok:
                    print("✗ 首臂恒等失败 — exec副本不忠实, 中止")
                    sys.exit(1)
            else:
                print(f"[终检恒等] fd复用无污染自证: {'通过 ✓ — 中间臂有效' if ident_ok else '失败 ✗ — 中间臂作废'}")
        else:
            print(f"[{tag}] final={fv:,.1f} (Δ{(fv/PROD['final_value']-1)*100:+.2f}% vs 生产) "
                  f"ret={results[-1]['ret_pct']:+.2f}% sharpe={r['sharpe']:.4f} "
                  f"mdd={results[-1]['mdd_pct']:.2f}%")
            print(f"[{tag}] 算术Δ {arith:+,.0f}元 → 放大比 measured/arithmetic = {amp:.2f}")
            for yr in sorted(r['annual_returns']):
                print(f"  {yr}: {r['annual_returns'][yr]*100:+.2f}%", end='')
            print()
        del strat, r
        gc.collect()

    res = pd.DataFrame(results)
    print("\n===== 滑点响应面汇总 =====")
    print(res.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    uniq = res.drop_duplicates(subset='rate', keep='first').sort_values('rate')
    navs = uniq['final_value'].values
    rates_u = uniq['rate'].values
    deltas = np.diff(navs)
    sign_flips = sum(1 for d in deltas if d < 0)
    print(f"\n[响应面] rate升序 {['%.4f' % r_ for r_ in rates_u]} NAV Δ序列: "
          f"{[f'{d:+,.0f}' for d in deltas]}")
    print(f"[响应面] Δ序列全负(NAV随滑点单调递减): {sign_flips == len(deltas)} — "
          f"{'单调成本函数(确定性, 非彩票带)' if sign_flips == len(deltas) else '存在符号翻转=彩票带'}")
    if len(deltas) >= 3:
        down_pp = [f'{d/PROD["final_value"]*100:+.2f}pp' for d in deltas[:1]]
        up_pp = [f'{d/PROD["final_value"]*100:+.2f}pp' for d in deltas[1:3]]
        print(f"[响应面] 斜率不对称(生产万10两侧): 下行每0.0005步 {down_pp} / "
              f"上行每0.0005步 {up_pp}")
    if 'nav_terminal_identity' in curves:
        d_id = float(np.nanmax(np.abs(np.asarray(curves['nav_terminal_identity']) -
                                      np.asarray(curves['nav_0.0010']))))
        print(f"[终检] 首臂vs终检恒等曲线逐日最大偏差: {d_id:.6f} "
              f"{'(fd复用零污染, 中间臂有效)' if d_id < 0.01 else '(fd复用污染! 中间臂作废)'}")
    _orig_to_csv(res, os.path.join('/tmp', 'probe_slippage_results_20260922.csv'), index=False)
    _orig_to_csv(pd.DataFrame(curves), os.path.join('/tmp', 'probe_slippage_curves_20260922.csv'), index=False)

    _assert_zero_writes(snap0)
    pd.DataFrame.to_csv = _orig_to_csv
    print("\n探针完成 (零生产写入, fp未动)")
