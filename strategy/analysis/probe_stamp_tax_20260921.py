"""印花税时间变化率审计探针 (2026-09-21, 只读, 不修改任何生产文件/fp)

背景: A股印花税卖出单边 — 2008-09-19起万10(0.001), 2023-08-28起万5(0.0005)。
生产 bt_execution.py:44 STAMP_TAX=0.0005 扁平 → 2021-01~2023-08-27所有卖单少收万5
(佣金万5已比真实万1-2.5保守, 无乐观偏差; 过户费万0.1双边≈可忽略, 不在此审计)。

方法: inspect.getsource(_vectorized_backtest) → 卖单两行 STAMP_TAX 替换为按日数组
_STAMP_ARR[i], exec回模块命名空间 (不落盘, fp不受影响)。两段:
  run1 恒等烟测: _STAMP_ARR=0.0005扁平 → 必须逐位复现生产锚点 732,689/193.08%/1.1961/17.92%
       (跨实现身份核验: exec副本+no-op to_csv+审计注入 与生产逐位一致 → 才可信任run2)
  run2 反事实: _STAMP_ARR=万10(2023-08-28前)/万5 → 正确印花税下的诚实锚点 + 逐笔少收审计

零写入保证: pd.DataFrame.to_csv全局no-op + 运行前后 rolling_validation_results 全文件md5比对。
输出: /tmp/probe_stamp_flat_curve.csv, /tmp/probe_stamp_counter_curve.csv,
      analysis/probe_stamp_audit_20260921.csv (逐笔卖出: date/proceeds/rate_used/extra_tax)。
"""
import os
import sys
import gc
import hashlib
import inspect

import numpy as np
import pandas as pd

sys.path.insert(0, '/mnt/d/quant/strategy')
import bt_execution as M  # noqa: E402  (模块级常量在导入时已从yaml载入)

_STRAT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RVR = os.path.join(_STRAT_DIR, 'rolling_validation_results')
_CUTOFF = np.datetime64('2023-08-28')  # 印花税万10→万5生效日

# ---- 生产锚点 (恒等烟测判据) ----
PROD = {'final_value': 732689.0, 'ret': 193.08, 'sharpe': 1.1961, 'mdd': 17.92,
        'years': {2021: 9.74, 2022: -4.65, 2023: 8.99, 2024: 29.50, 2025: 50.58, 2026: 33.27}}


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


if __name__ == '__main__':
    print(f"环境: {sys.executable}")
    # 1. 零写入保护: 全局no-op to_csv + 生产产物md5快照
    _orig_to_csv = pd.DataFrame.to_csv
    pd.DataFrame.to_csv = lambda self, *a, **k: None
    snap0 = _md5_snapshot()

    # 2. 镜像main装配: 股票池过滤 + 基本面 (不跑 __main__)
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
        else:
            stock_pool = M.get_stock_pool(todate=M._pool_todate(),
                                          bse_exclude=M.config.get('stock_pool.bse_exclude', True))
            stock_codes = [c for c in stock_codes if c in stock_pool]
            print(f"基本面数据加载(股票池): {len(stock_codes)} 只")
    else:
        print(f"基本面数据加载(全市场): {len(stock_codes)} 只")

    # 3. 印花税率按日数组: 与函数内日历(指数交易日)对齐
    _idx_fp = os.path.join(M.DATA_PATH, 'sh000001_qfq.csv')
    _idx_df = pd.read_csv(_idx_fp, parse_dates=['datetime'])
    _idx_df = _idx_df[(_idx_df['datetime'] >= pd.Timestamp(M.FROMDATE)) &
                      (_idx_df['datetime'] <= pd.Timestamp(M.TODATE))]
    calendar = pd.DatetimeIndex(sorted(_idx_df['datetime'].unique()))
    n_dates = len(calendar)
    print(f"日历: {n_dates} 交易日 ({calendar[0].date()} ~ {calendar[-1].date()})")

    # 4. exec补丁: STAMP_TAX → _STAMP_ARR[i] + 逐笔卖出审计注入
    src = inspect.getsource(M._vectorized_backtest)
    n_stamp = src.count('STAMP_TAX')
    assert n_stamp == 2, f"STAMP_TAX出现{n_stamp}次, 预期2 — 代码结构变了, 人工复核"
    src2 = src.replace('STAMP_TAX', '_STAMP_ARR[i]')
    inj = {
        "cash += float(positions[j]) * sell_px * (1.0 - COMMISSION - _STAMP_ARR[i] - impact)":
            "float(positions[j]) * sell_px",
        "cash += abs(diff) * sell_px * (1.0 - COMMISSION - _STAMP_ARR[i] - impact)":
            "abs(diff) * sell_px",
    }
    for key, proceeds in inj.items():
        assert src2.count(key) == 1, f"卖单行不唯一/缺失: {key[:60]}"
        idx = src2.index(key)
        indent = src2[src2.rfind('\n', 0, idx) + 1:idx]
        src2 = src2.replace(key, f"_stamp_audit.append((date, {proceeds}))\n{indent}{key}")
    exec(compile(src2, 'probe_stamp_patched', 'exec'), M.__dict__)
    print("exec补丁完成: 卖单2处税率数组化 + 2处审计注入")

    # ---- run1: 恒等烟测 (扁平万5, 必须逐位复现生产) ----
    M._STAMP_ARR = np.full(n_dates, 0.0005)
    M._stamp_audit = []
    fd1 = M.FundamentalData(M.FUNDAMENTAL_PATH, stock_codes)
    strat1 = _build_strategy(fd1)
    print("\n===== run1 恒等烟测 (flat 0.0005) =====")
    r1 = M._vectorized_backtest(strat1, fd1, M.FROMDATE, M.TODATE, M.CASH)
    fv1 = float(r1['final_value'])
    print(f"[identity] final={fv1:,.1f} sharpe={r1['sharpe']:.4f} mdd={abs(r1['max_drawdown'])*100:.2f}%")
    ok = True
    if abs(fv1 - PROD['final_value']) > 0.5:
        print(f"[identity] ✗ NAV偏差 {fv1 - PROD['final_value']:,.2f}"); ok = False
    if abs(r1['sharpe'] - PROD['sharpe']) > 2e-4:
        print(f"[identity] ✗ Sharpe偏差 {r1['sharpe'] - PROD['sharpe']:+.5f}"); ok = False
    if abs(abs(r1['max_drawdown']) * 100 - PROD['mdd']) > 0.02:
        print(f"[identity] ✗ MDD偏差"); ok = False
    for yr, rr in sorted(r1['annual_returns'].items()):
        exp = PROD['years'].get(int(yr))
        if exp is not None and abs(rr * 100 - exp) > 0.03:
            print(f"[identity] ✗ {yr} 年收益 {rr*100:.2f}% vs 生产 {exp}%"); ok = False
    # 净值曲线逐日比对 (比标量更严格的恒等证据): vs RVD现行生产曲线。
    # 注意: C26臂目录的 pre_equity_curve.csv 是批次中间态旧曲线(终点721,744.68),
    # 不是生产 — 勿用。生产曲线=RVD当前(已核验与C26 post逐位全等, 终点732,688.93)。
    _prod_curve = pd.read_csv(os.path.join(_RVR, 'equity_curve.csv'))
    _pnav = _prod_curve['nav'].values.astype(np.float64)
    print(f"[identity] 参照: RVD equity_curve.csv 终点 {_pnav[-1]:,.2f} ({len(_pnav)}行)")
    _maxdiff = float(np.nanmax(np.abs(r1['nav'] - _pnav)))
    print(f"[identity] 净值曲线逐日最大偏差: {_maxdiff:.6f}")
    if _maxdiff > 0.01:
        print("[identity] ✗ 净值曲线不一致"); ok = False
    assert ok, "恒等烟测失败 — exec副本不忠实, 反事实不可信, 中止"
    print("[identity] ✓ 四指标+年分解+逐日净值曲线复现生产锚点 732,689 — exec副本忠实")

    # ---- run2: 时间变化率反事实 (万10 → 2023-08-28 → 万5) ----
    M._STAMP_ARR = np.where(calendar.values < _CUTOFF, 0.001, 0.0005).astype(np.float64)
    n_pre = int((calendar.values < _CUTOFF).sum())
    n_post = n_dates - n_pre
    print(f"\n===== run2 反事实 (万10→{_CUTOFF}→万5; {n_pre}日万10 / {n_post}日万5) =====")
    M._stamp_audit = []
    gc.collect()
    fd2 = M.FundamentalData(M.FUNDAMENTAL_PATH, stock_codes)
    strat2 = _build_strategy(fd2)
    r2 = M._vectorized_backtest(strat2, fd2, M.FROMDATE, M.TODATE, M.CASH)
    fv2 = float(r2['final_value'])

    # ---- 逐笔审计汇总 ----
    aud = pd.DataFrame(M._stamp_audit, columns=['date', 'proceeds'])
    aud['date'] = pd.to_datetime(aud['date'])
    aud['pre_cutoff'] = aud['date'].values < _CUTOFF
    aud['rate_used'] = np.where(aud['pre_cutoff'], 0.001, 0.0005)
    aud['extra_tax'] = aud['proceeds'] * (aud['rate_used'] - 0.0005)  # 相对生产的补收
    aud['year'] = aud['date'].dt.year
    _orig_to_csv(aud, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   'probe_stamp_audit_20260921.csv'), index=False)
    tot_extra = aud['extra_tax'].sum()
    print(f"\n[审计] 卖出 {len(aud)} 笔 (万10期 {int(aud['pre_cutoff'].sum())} / 万5期 {int((~aud['pre_cutoff']).sum())})")
    print(f"[审计] 补收印花税一阶合计: {tot_extra:,.0f} 元 = 最终净值 {fv1:,.0f} 的 {tot_extra/fv1*100:.2f}%")
    for yr, g in aud.groupby('year'):
        print(f"  {yr}: {len(g):4d}笔, 补收 {g['extra_tax'].sum():,.0f} 元")
    print(f"\n[反事实] 诚实锚点: {fv2:,.0f}  (生产 {fv1:,.0f}, Δ {fv2-fv1:,.0f} = {(fv2/fv1-1)*100:+.2f}%)")
    print(f"[反事实] Sharpe {r2['sharpe']:.4f} (Δ{r2['sharpe']-r1['sharpe']:+.4f})  "
          f"MDD {abs(r2['max_drawdown'])*100:.2f}% (Δ{abs(r2['max_drawdown'])*100-abs(r1['max_drawdown'])*100:+.2f}pp)")
    print(f"[反事实] 总收益 {(fv2/M.CASH-1)*100:.2f}% (Δ{(fv2-fv1)/M.CASH*100:+.2f}pp)")
    for yr in sorted(r1['annual_returns']):
        d = r2['annual_returns'][yr] - r1['annual_returns'][yr]
        if abs(d) > 0.001:
            print(f"  年分解 {yr}: {r1['annual_returns'][yr]*100:+.2f}% → {r2['annual_returns'][yr]*100:+.2f}% (Δ{d*100:+.2f}pp)")

    # ---- 净值曲线落盘(/tmp) ----
    _orig_to_csv(pd.DataFrame({'date': [d.strftime('%Y-%m-%d') for d in calendar],
                               'nav': r1['nav']}), '/tmp/probe_stamp_flat_curve.csv', index=False)
    _orig_to_csv(pd.DataFrame({'date': [d.strftime('%Y-%m-%d') for d in calendar],
                               'nav': r2['nav']}), '/tmp/probe_stamp_counter_curve.csv', index=False)

    # ---- 零写入核验 ----
    _assert_zero_writes(snap0)
    pd.DataFrame.to_csv = _orig_to_csv
    print("\n探针完成 (零生产写入, fp未动)")
