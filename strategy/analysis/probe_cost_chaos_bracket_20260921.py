"""成本混沌bracket: 扁平印花税率响应面探针 (2026-09-21, 只读, 不修改任何生产文件/fp)

背景: 印花税时间变化率审计(probe_stamp_tax_20260921)发现反事实点估计−4.41%被路径混沌
污染 — 万5期(税率与生产完全相同)的2024-26年分解也各掉−0.5~1.6pp≠税效应, 每笔均价
¥0.87的持续微扰(2021-23)经100股取整+现金不足比例缩放+C5c缓冲状态链分叉持仓轨迹。
结论"成本臂裁决只能走一阶会计口径"目前n=1(仅时间变化率一个反事实点)。

本探针: 扁平税率bracket {0.0001, 0.0003, 0.0005, 0.0007, 0.0010} — 问响应面形状:
NAV对税率是单调成本函数, 还是彩票带? 若单调→时间变化率的分叉来自时点异质性;
若非单调→"点估计=轨迹彩票"获n=5实证, 且给出重锚彩票带宽度(未来自然重锚修税时
新锚点携带的±带)。同时: 0.0010臂=全程万10反事实(政策上界对照), 0.0005臂=恒等。

方法: 同stamp探针的exec补丁法 — inspect.getsource(_vectorized_backtest) → 卖单两行
STAMP_TAX替换为_STAMP_ARR[i] → exec回模块命名空间(不落盘, fp不动)。
效率设计: FundamentalData只构造一次(已核验_vectorized_backtest对fd仅get_st_timeline
只读访问, 惰性缓存幂等) — 终检恒等臂(第6跑, 复用fd后再跑0.0005)作为fd复用无污染
的自证: 若终检逐位复现生产, 则fd复用行为惰性, 中间臂有效; 否则中间臂作废。

零写入保证: pd.DataFrame.to_csv全局no-op + 运行前后 rolling_validation_results 全文件md5比对。
输出: /tmp/probe_cost_chaos_results_20260921.csv (臂级四指标+年分解),
      /tmp/probe_cost_chaos_curves_20260921.csv (全臂逐日净值)。
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

# ---- 生产锚点 (恒等烟测判据) ----
PROD = {'final_value': 732689.0, 'ret': 193.08, 'sharpe': 1.1961, 'mdd': 17.92,
        'years': {2021: 9.74, 2022: -4.65, 2023: 8.99, 2024: 29.50, 2025: 50.58, 2026: 33.27}}

ARMS = [0.0005, 0.0001, 0.0003, 0.0007, 0.0010]  # 恒等 → 低→高, 终检0.0005在最后


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


def _check_identity(tag, r, strict_curve=True):
    """对标生产锚点: 标量+年分解(+逐日净值曲线, strict_curve时)。返回bool。"""
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
    if strict_curve:
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
    # 1. 零写入保护
    _orig_to_csv = pd.DataFrame.to_csv
    pd.DataFrame.to_csv = lambda self, *a, **k: None
    snap0 = _md5_snapshot()

    # 2. 镜像main装配: 股票池过滤 (与stamp探针一致)
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
    if stock_pool_enabled:
        if _membership_map is not None:
            _union = set()
            for _codes in _membership_map.values():
                _union |= _codes
            stock_codes = [c for c in stock_codes if c in _union]
            print(f"基本面数据加载(日历union池): {len(stock_codes)} 只")

    # 3. 日历 (与函数内日历对齐)
    _idx_fp = os.path.join(M.DATA_PATH, 'sh000001_qfq.csv')
    _idx_df = pd.read_csv(_idx_fp, parse_dates=['datetime'])
    _idx_df = _idx_df[(_idx_df['datetime'] >= pd.Timestamp(M.FROMDATE)) &
                      (_idx_df['datetime'] <= pd.Timestamp(M.TODATE))]
    calendar = pd.DatetimeIndex(sorted(_idx_df['datetime'].unique()))
    n_dates = len(calendar)
    print(f"日历: {n_dates} 交易日 ({calendar[0].date()} ~ {calendar[-1].date()})")

    # 4. exec补丁: STAMP_TAX → _STAMP_ARR[i] (仅税率, 无审计注入 — 最小改动面)
    src = inspect.getsource(M._vectorized_backtest)
    n_stamp = src.count('STAMP_TAX')
    assert n_stamp == 2, f"STAMP_TAX出现{n_stamp}次, 预期2 — 代码结构变了, 人工复核"
    src2 = src.replace('STAMP_TAX', '_STAMP_ARR[i]')
    exec(compile(src2, 'probe_chaos_patched', 'exec'), M.__dict__)
    print("exec补丁完成: 卖单2处税率数组化")

    # 5. FundamentalData只构造一次 (fd复用, 终检臂自证无污染)
    print("构造FundamentalData(5189只, 只读复用)...")
    fd = M.FundamentalData(M.FUNDAMENTAL_PATH, stock_codes)
    print("FundamentalData构造完成")

    results = []
    curves = {'date': [d.strftime('%Y-%m-%d') for d in calendar]}
    first_identity_ok = None
    for arm_idx, rate in enumerate(ARMS):
        tag = f"arm{arm_idx+1} flat {rate:.4f}"
        print(f"\n===== {tag} =====")
        M._STAMP_ARR = np.full(n_dates, rate)
        gc.collect()
        strat = _build_strategy(fd)
        r = M._vectorized_backtest(strat, fd, M.FROMDATE, M.TODATE, M.CASH)
        fv = float(r['final_value'])
        results.append({'arm': arm_idx + 1, 'rate': rate,
                        'final_value': fv, 'ret_pct': (fv / M.CASH - 1) * 100,
                        'sharpe': r['sharpe'], 'mdd_pct': abs(r['max_drawdown']) * 100})
        if rate == 0.0005 and arm_idx > 0:
            curves['nav_terminal_identity'] = r['nav']
        else:
            curves[f'nav_{rate:.4f}'] = r['nav']
        if rate == 0.0005:
            # 恒等检查: 首臂严格(曲线), 终检臂同严格
            ident_ok = _check_identity(tag, r, strict_curve=True)
            if arm_idx == 0:
                first_identity_ok = ident_ok
                if not first_identity_ok:
                    print("✗ 首臂恒等失败 — exec副本不忠实或fd复用设计有误, 中止")
                    sys.exit(1)
            else:
                print(f"[终检恒等] fd复用无污染自证: {'通过 ✓ — 中间臂有效' if ident_ok else '失败 ✗ — 中间臂作废'}")
        else:
            print(f"[{tag}] final={fv:,.1f} (Δ{(fv/PROD['final_value']-1)*100:+.2f}% vs 生产) "
                  f"ret={results[-1]['ret_pct']:+.2f}% sharpe={r['sharpe']:.4f} "
                  f"mdd={results[-1]['mdd_pct']:.2f}%")
            for yr in sorted(r['annual_returns']):
                print(f"  {yr}: {r['annual_returns'][yr]*100:+.2f}%", end='')
            print()
        del strat, r
        gc.collect()

    # 6. 汇总表
    res = pd.DataFrame(results)
    print("\n===== 响应面汇总 =====")
    print(res.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    # 单调性速检: 5个唯一rate升序, NAV对rate的Δ符号 (终检恒等臂不参与)
    uniq = res.drop_duplicates(subset='rate', keep='first').sort_values('rate')
    navs = uniq['final_value'].values
    rates_u = uniq['rate'].values
    deltas = np.diff(navs)
    sign_flips = sum(1 for d in deltas if d < 0)
    print(f"\n[响应面] rate升序 {['%.4f' % r_ for r_ in rates_u]} NAV Δ序列: "
          f"{[f'{d:+,.0f}' for d in deltas]}")
    print(f"[响应面] Δ序列全负(NAV随税率单调递减): {sign_flips == len(deltas)} — "
          f"{'单调成本函数(确定性, 非彩票带)' if sign_flips == len(deltas) else '存在符号翻转=彩票带'}")
    print(f"[响应面] 斜率不对称(生产万5两侧): 下行每0.0002步 "
          f"{[f'{d/PROD[\"final_value\"]*100:+.2f}pp' for d in deltas[:2]]} / "
          f"上行每0.0002步 {[f'{d/PROD[\"final_value\"]*100:+.2f}pp' for d in deltas[2:]]}")
    if 'nav_terminal_identity' in curves:
        d_id = float(np.nanmax(np.abs(np.asarray(curves['nav_terminal_identity']) -
                                      np.asarray(curves['nav_0.0005']))))
        print(f"[终检] 首臂vs终检恒等曲线逐日最大偏差: {d_id:.6f} "
              f"{'(fd复用零污染, 中间臂有效)' if d_id < 0.01 else '(fd复用污染! 中间臂作废)'}")
    _orig_to_csv(res, os.path.join('/tmp', 'probe_cost_chaos_results_20260921.csv'), index=False)
    _orig_to_csv(pd.DataFrame(curves), os.path.join('/tmp', 'probe_cost_chaos_curves_20260921.csv'), index=False)

    # 7. 零写入核验
    _assert_zero_writes(snap0)
    pd.DataFrame.to_csv = _orig_to_csv
    print("\n探针完成 (零生产写入, fp未动)")
