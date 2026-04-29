"""
验证实盘与回测一致性: 用同一天数据, 对比 signal_runner vs backtest 的输出
"""
import sys
import os
import multiprocessing

import pandas as pd
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'strategy'))

from datetime import datetime, timedelta, date


def main():
    # 使用 backtest 的最后一天做对比
    TEST_DATE = '2025-12-31'
    BT_DATA_DIR = os.path.join(ROOT, 'data', 'stock_data', 'backtrader_data')
    FUND_DATA_DIR = os.path.join(ROOT, 'data', 'stock_data', 'fundamental_data')

    print(f"验证日期: {TEST_DATE}")
    print(f"数据目录: {BT_DATA_DIR}")

    # ============================================================
    # Part 1: 模拟 SignalRunner 生成目标持仓
    # ============================================================
    print("\n" + "=" * 60)
    print("Part 1: SignalRunner 实盘模式")
    print("=" * 60)

    from trade.signal_runner import SignalRunner

    runner = SignalRunner(
        bt_data_dir=BT_DATA_DIR,
        fund_data_dir=FUND_DATA_DIR,
        max_position=10,
    )

    # 限制数据到 TEST_DATE 之前（模拟回测窗口）
    min_date_str = '2023-01-01'
    for code in list(runner.stock_data_dict.keys()):
        d = runner.stock_data_dict[code]
        d = d[(d['datetime'] >= min_date_str) & (d['datetime'] <= TEST_DATE)]
        if len(d) > 20:
            runner.stock_data_dict[code] = d
            last_row = d.iloc[-1]
            if last_row['close'] > 0:
                runner.prices[code] = float(last_row['close'])
        else:
            del runner.stock_data_dict[code]
            runner.prices.pop(code, None)

    print(f"数据日期范围: {min_date_str} → {TEST_DATE}")

    # 准备策略
    runner.prepare(max_position=10)

    # 模拟空持仓（等同回测首次调仓）
    test_date = date(2025, 12, 31)

    # 获取该日可交易股票池
    tradable = []
    for code, price in runner.prices.items():
        if code == "sh000001":
            continue
        if price < runner.MIN_PRICE:
            continue
        if code in runner.stock_data_dict:
            last_vol = runner.stock_data_dict[code][runner.stock_data_dict[code]['datetime'] <= str(test_date)].iloc[-1].get('volume', 0)
            if last_vol < runner.MIN_VOLUME:
                continue
        tradable.append(code)

    print(f"可交易股票池: {len(tradable)} 只")

    # 生成目标持仓
    target = runner.strategy.generate_positions(
        date=test_date,
        universe=tradable,
        current_positions={},
        cash=100000,
        prices=runner.prices,
        cost={},
        rebalance=True,
    )

    trade_positions = set(target.keys())
    print(f"\nSignalRunner 目标持仓 ({len(trade_positions)} 只):")
    for code, value in sorted(target.items(), key=lambda x: -x[1]):
        sel = [s for s in runner.strategy.portfolio.last_selection if s['code'] == code]
        info = sel[0] if sel else {}
        print(f"  {code}: ¥{value:,.0f}  weight={info.get('weight', 0):.3f}  industry={info.get('industry', '')}")

    # 获取选股列表
    trade_selections = {}
    if hasattr(runner.strategy.portfolio, 'last_selection'):
        for s in runner.strategy.portfolio.last_selection:
            trade_selections[s['code']] = s

    # ============================================================
    # Part 2: 读取 backtest 的 portfolio_selections.csv
    # ============================================================
    print("\n" + "=" * 60)
    print("Part 2: 回测 portfolio_selections 记录")
    print("=" * 60)

    bt_sel_path = os.path.join(ROOT, 'strategy', 'rolling_validation_results', 'portfolio_selections.csv')
    bt_sel = pd.read_csv(bt_sel_path)
    bt_sel['date'] = bt_sel['date'].astype(str)
    bt_date_sel = bt_sel[bt_sel['date'] == TEST_DATE]

    if len(bt_date_sel) == 0:
        print(f"⚠ 回测 portfolio_selections 中没有 {TEST_DATE} 的数据!")
        print(f"回测日期范围: {bt_sel['date'].min()} → {bt_sel['date'].max()}")
        nearest = bt_sel.iloc[(bt_sel['date'].apply(lambda x: abs(pd.to_datetime(x) - pd.to_datetime(TEST_DATE)))).argsort()[:1]]
        print(f"最近日期: {nearest['date'].values[0]}")
        return

    bt_positions = set(bt_date_sel['code'].tolist())
    print(f"回测目标持仓 ({len(bt_positions)} 只):")
    for _, row in bt_date_sel.iterrows():
        print(f"  {row['code']}: weight={row['weight']:.3f}  industry={row['industry']}")

    # ============================================================
    # Part 3: 对比
    # ============================================================
    print("\n" + "=" * 60)
    print("Part 3: 对比分析")
    print("=" * 60)

    common = trade_positions & bt_positions
    only_trade = trade_positions - bt_positions
    only_bt = bt_positions - trade_positions

    print(f"\n共同持仓:  {len(common)} 只")
    print(f"仅实盘有:  {len(only_trade)} 只")
    print(f"仅回测有:  {len(only_bt)} 只")
    overlap = len(common) / max(len(bt_positions), 1) * 100
    print(f"重叠率:    {overlap:.0f}%")

    if only_trade:
        print(f"\n仅实盘持仓:")
        for c in only_trade:
            s = trade_selections.get(c, {})
            print(f"  {c}: weight={s.get('weight', 0):.3f} {s.get('industry', '')}")
    if only_bt:
        print(f"\n仅回测持仓:")
        for _, row in bt_date_sel[bt_date_sel['code'].isin(only_bt)].iterrows():
            print(f"  {row['code']}: weight={row['weight']:.3f} {row['industry']}")

    # 检查市场状态
    print(f"\n=== 市场状态对比 ===")
    if "sh000001" in runner.stock_data_dict:
        idx_data = runner.stock_data_dict["sh000001"]
        idx_row = idx_data[idx_data['datetime'].dt.date == test_date]
        if not idx_row.empty:
            regime = int(idx_row["regime"].values[0]) if "regime" in idx_row.columns else 'N/A'
            momentum = float(idx_row["momentum_score"].values[0]) if "momentum_score" in idx_row.columns else 'N/A'
            bear = bool(idx_row["bear_risk"].values[0]) if "bear_risk" in idx_row.columns else 'N/A'
            print(f"market_regime: regime={regime}, momentum={momentum:.3f}, bear_risk={bear}")

    # 因子模式
    from strategy.core.config_loader import load_config
    config = load_config()
    print(f"factor_mode: {config.get('factor_mode', 'N/A')}")
    stats = runner.strategy.signal_engine._stats
    print(f"signal_stats: dyn_success={stats['dynamic_success']}, fixed_ind={stats['fixed_industry']}, "
          f"fixed_def={stats['fixed_default']}, dyn_fallback={stats['dynamic_fallback_fixed']}")

    # 结论
    print(f"\n{'=' * 60}")
    if overlap >= 70:
        print(f"✓ 实盘与回测持仓高度重叠 ({overlap:.0f}%)，系统对齐良好")
    elif overlap >= 50:
        print(f"△ 实盘与回测持仓部分重叠 ({overlap:.0f}%)，有差异但可接受")
    else:
        print(f"✗ 实盘与回测持仓差异较大 ({overlap:.0f}%)，需要排查!")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    multiprocessing.set_start_method('fork', force=True)
    main()
