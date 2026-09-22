"""滑点双边不对称分解探针 (2026-09-22, 只读, 不修改任何生产文件/fp)

背景: 扁平bracket(同日)已闭合滑点成本栈 — 万5/万10/万15/万20 →
777,549/732,689/670,298/615,121, 严格单调确定性成本函数, 无尖锐膝点,
斜率不对称温和~1.3×。但生产是**双边同率**(yaml单键 slippage: 0.001):
买入侧滑点自带对冲(target_shares=目标值÷buy_px → 买价升→股数减→现金需求
近似不变), 卖出侧无此对冲(股数固定, 卖现缩水直接砍下一单=印花税式现金悬崖)。
→ 双边弹性应不同: 卖侧每bp敏感度 > 买侧每bp敏感度。

本探针: 同exec补丁法, 把滑点乘法三处拆成卖/买两个数组:
  (1.0 - slip_rate) ×2 → (1.0 - _SELL_SLIP_ARR[i])  [Step A全卖+Step B减仓]
  (1.0 + slip_rate) ×1 → (1.0 + _BUY_SLIP_ARR[i])   [Step C买入]
臂(串行): [(万10,万10)恒等 → (万5,万10)买侧减半 → (万10,万5)卖侧减半 →
           (万10,万10)终检恒等(fd复用零污染自证)]。
裁决问题:
  1) B5S10 vs B10S5 谁大 = 哪侧弹性主导 (预测: 卖侧, 印花税机制同构);
  2) 与扁平臂对账: B5S10+B10S5 ≈ 2×755,119(两半臂之和≈2×扁平万5/万10中点);
  3) 若卖侧主导 → fill_cost_feedback校准优先卖侧, 且yaml需拆 sell_slippage
     键(自然重锚时的建模选择, 不改代码)。
输出: /tmp/probe_slip_asym_results_20260922.csv + curves。
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

# (buy_rate, sell_rate): 恒等 → 买侧减半 → 卖侧减半 → 终检恒等
ARMS = [(0.0010, 0.0010), (0.0005, 0.0010), (0.0010, 0.0005), (0.0010, 0.0010)]

# 年复利因子: 年中NAV到终值 — 由生产年收益计算(同step/flat探针)
CF_MIDYEAR = {2021: 2.83, 2022: 2.77, 2023: 2.71, 2024: 2.28, 2025: 1.64, 2026: 1.15}

# 扁平bracket参照 (同日probe_slippage_bracket_20260922, 逐位复现过)
FLAT_5BP_NAV = 777549.0   # 万5双边
FLAT_10BP_NAV = 732689.0  # 万10双边=生产


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

    # 股票池过滤 (与前四探针一致)
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

    # exec补丁: 滑点乘法三处拆卖/买两数组
    src = inspect.getsource(M._vectorized_backtest)
    n_sell = src.count('(1.0 - slip_rate)')
    n_buy = src.count('(1.0 + slip_rate)')
    assert n_sell == 2 and n_buy == 1, \
        f"滑点乘法出现次数异常 (sell={n_sell}, buy={n_buy}, 预期 2/1) — 代码结构变了, 人工复核"
    src2 = src.replace('(1.0 - slip_rate)', '(1.0 - _SELL_SLIP_ARR[i])')
    src2 = src2.replace('(1.0 + slip_rate)', '(1.0 + _BUY_SLIP_ARR[i])')
    exec(compile(src2, 'probe_slip_asym_patched', 'exec'), M.__dict__)
    print("exec补丁完成: 滑点乘法3处拆卖/买 (卖2买1)")

    print("构造FundamentalData(只读复用)...")
    fd = M.FundamentalData(M.FUNDAMENTAL_PATH, stock_codes)
    print("FundamentalData构造完成")

    # 算术基准: 稳态Σ买≈Σ卖≈Σ卖单毛额(同flat探针的proceeds口径)
    aud = pd.read_csv(os.path.join(_THIS, 'probe_stamp_audit_20260921.csv'), parse_dates=['date'])
    aud['year'] = aud['date'].dt.year
    proceeds_by_year = {int(yr): float(g['proceeds'].sum()) for yr, g in aud.groupby('year')}

    results = []
    curves = {'date': [d.strftime('%Y-%m-%d') for d in calendar]}
    first_identity_ok = None
    for arm_idx, (br, sr) in enumerate(ARMS):
        tag = f"arm{arm_idx+1} buy万{br*10000:.0f} sell万{sr*10000:.0f}"
        print(f"\n===== {tag} =====")
        M._BUY_SLIP_ARR = np.full(n_dates, br)
        M._SELL_SLIP_ARR = np.full(n_dates, sr)
        gc.collect()
        strat = _build_strategy(fd)
        r = M._vectorized_backtest(strat, fd, M.FROMDATE, M.TODATE, M.CASH)
        fv = float(r['final_value'])
        # 算术Δ: 买侧=dbuy×Σ买额×CF, 卖侧=dsell×Σ卖额×CF (稳态Σ买≈Σ卖)
        d_buy, d_sell = br - 0.0010, sr - 0.0010
        arith_buy = sum(d_buy * proceeds_by_year.get(yr, 0.0) * CF_MIDYEAR.get(yr, 1.0)
                        for yr in proceeds_by_year)
        arith_sell = sum(d_sell * proceeds_by_year.get(yr, 0.0) * CF_MIDYEAR.get(yr, 1.0)
                         for yr in proceeds_by_year)
        arith = arith_buy + arith_sell
        meas = fv - PROD['final_value']
        amp = meas / arith if abs(arith) > 1.0 else float('nan')
        results.append({'arm': arm_idx + 1, 'buy_rate': br, 'sell_rate': sr,
                        'final_value': fv, 'ret_pct': (fv / M.CASH - 1) * 100,
                        'sharpe': r['sharpe'], 'mdd_pct': abs(r['max_drawdown']) * 100,
                        'meas_dnav': meas, 'arith_buy': arith_buy, 'arith_sell': arith_sell,
                        'amp_ratio': amp})
        curves[f'nav_b{br:.4f}_s{sr:.4f}_arm{arm_idx+1}'] = r['nav']
        if br == 0.0010 and sr == 0.0010:
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
            print(f"[{tag}] 算术Δ 买侧{arith_buy:+,.0f} + 卖侧{arith_sell:+,.0f} = {arith:+,.0f}元 "
                  f"→ 放大比 measured/arithmetic = {amp:.2f}")
            for yr in sorted(r['annual_returns']):
                print(f"  {yr}: {r['annual_returns'][yr]*100:+.2f}%", end='')
            print()
        del strat, r
        gc.collect()

    # 弹性分解 + 扁平对账
    print("\n===== 双边弹性分解 =====")
    b5s10 = results[1]['final_value']
    b10s5 = results[2]['final_value']
    print(f"买侧减半(B5S10): {b5s10:,.1f} (Δ{b5s10-PROD['final_value']:+,.1f})  "
          f"amp={results[1]['amp_ratio']:.2f}")
    print(f"卖侧减半(B10S5): {b10s5:,.1f} (Δ{b10s5-PROD['final_value']:+,.1f})  "
          f"amp={results[2]['amp_ratio']:.2f}")
    mid = (FLAT_5BP_NAV + FLAT_10BP_NAV) / 2
    print(f"弹性比 卖侧/买侧 = {(b10s5-PROD['final_value'])/(b5s10-PROD['final_value']):.2f} "
          f"(预测>1: 卖侧无股数对冲, 印花税机制同构)")
    print(f"扁平对账: 两半臂之和 {b5s10+b10s5:,.1f} vs 2×中点 {2*mid:,.1f} "
          f"(偏差 {(b5s10+b10s5-2*mid)/(2*mid)*100:+.2f}%) — 弹性线性可加性检验")
    print(f"扁平万5(双边) Δ={FLAT_5BP_NAV-PROD['final_value']:+,.1f} vs "
          f"两半臂Δ和={(b5s10-PROD['final_value'])+(b10s5-PROD['final_value']):+,.1f}")

    _orig_to_csv(pd.DataFrame(results),
                 os.path.join('/tmp', 'probe_slip_asym_results_20260922.csv'), index=False)
    _orig_to_csv(pd.DataFrame(curves),
                 os.path.join('/tmp', 'probe_slip_asym_curves_20260922.csv'), index=False)
    _assert_zero_writes(snap0)
    pd.DataFrame.to_csv = _orig_to_csv
    print("\n探针完成 (零生产写入, fp未动)")
