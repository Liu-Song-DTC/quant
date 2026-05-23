"""
持仓诊断器 — 用最新数据重新跑策略，判断是否还符合选股标准

用法: python strategy/analysis/selection_diagnosis.py output/选股数据_20260519.xlsx

输出每只股票: 持有 / 卖出 + 具体原因
"""
import sys, os, re, glob
import pandas as pd, numpy as np
from pathlib import Path
from datetime import datetime, timedelta

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "strategy"))
from core.factor_calculator import calculate_indicators
from core.signal_engine import SignalEngine


def load_selection(excel_path: str):
    raw = pd.read_excel(excel_path, sheet_name='本次选股', header=None)
    header_row = next((i for i, row in raw.iterrows() if row.astype(str).str.contains('序号').any()), 3)
    df = pd.read_excel(excel_path, sheet_name='本次选股', header=header_row)
    df = df.dropna(subset=['序号'])
    df = df[pd.to_numeric(df['序号'], errors='coerce').notna()].copy()
    df['股票代码'] = df['股票代码'].astype(int).astype(str).str.zfill(6)
    return df


def evaluate_stock(code: str, name: str, industry: str, selection_date: str):
    """用最新K线数据重新跑策略，判断是否仍符合"""
    bt_path = ROOT / f"data/stock_data/backtrader_data/{code}_qfq.csv"
    if not bt_path.exists():
        return {"verdict": "数据缺失", "action": "—", "reasons": ["无行情数据"]}

    d = pd.read_csv(str(bt_path))
    date_col = [c for c in d.columns if 'date' in c.lower()][0]
    close_col = [c for c in d.columns if 'close' in c.lower()][0]
    high_col = [c for c in d.columns if 'high' in c.lower()][0]
    low_col = [c for c in d.columns if 'low' in c.lower()][0]
    vol_col = [c for c in d.columns if 'volume' in c.lower()][0]
    d[date_col] = d[date_col].astype(str)

    # 找最新一根K线（今天）
    today_idx = len(d) - 1
    if today_idx < 60:
        return {"verdict": "数据不足", "action": "—", "reasons": ["K线<60根"]}

    # 用全部历史数据计算指标
    n = today_idx + 1
    close = d[close_col].values[:n].astype(float)
    high = d[high_col].values[:n].astype(float)
    low = d[low_col].values[:n].astype(float)
    volume = d[vol_col].values[:n].astype(float)

    buy_price = close[today_idx - 1] if today_idx > 0 else close[today_idx]
    today_price = close[today_idx]
    today_ret = (today_price - buy_price) / buy_price * 100

    ind = calculate_indicators(close, high, low, volume)
    i = today_idx

    # ── 重新跑策略的所有检查 ──
    se = SignalEngine()
    market_info = {'regime': 1}  # conservative: assume bull

    buy_point = int(ind.get('buy_point', np.zeros(n))[i])
    chan_sell_score = float(ind.get('chan_sell_score', np.zeros(n))[i])
    chan_buy = float(ind.get('chan_buy_score', np.zeros(n))[i])
    trend_type = int(ind.get('trend_type', np.zeros(n))[i])
    bottom_div = float(ind.get('bottom_divergence', np.zeros(n))[i])
    bottom_fx_q = float(ind.get('bottom_fractal_quality', np.zeros(n))[i])
    signal_level = int(ind.get('signal_level', np.zeros(n))[i])
    bv = float(ind.get('b3_breakout_vol_ratio', np.zeros(n))[i])
    pv = float(ind.get('b3_pullback_vol_ratio', np.zeros(n))[i])
    ps = float(ind.get('b3_pullback_shallowness', np.zeros(n))[i])
    tc = bool(ind.get('b3_trend_confirmed', np.zeros(n, dtype=bool))[i])
    zg = float(ind.get('chan_pivot_zg', np.full(n, np.nan))[i])
    zd = float(ind.get('chan_pivot_zd', np.full(n, np.nan))[i])
    ema60_v = float(ind.get('ema60', np.zeros(n))[i])
    ema120_v = float(ind.get('ema120', np.zeros(n))[i])
    ma20_v = float(ind.get('ma20', np.zeros(n))[i])
    daily_ret = float(ind.get('ret', np.zeros(n))[i])
    vol_ratio = float(ind.get('volume_ratio', np.zeros(n))[i])
    ema20 = float(ind.get('ema20', np.zeros(n))[i])
    ema60 = float(ind.get('ema60', np.zeros(n))[i])

    sell_reasons = []

    # 1. ZG硬止损: 跌破中枢上沿
    if buy_point == 3 and not np.isnan(zg) and zg > 0:
        if today_price < zg * 0.99:
            sell_reasons.append(f"ZG破位(收盘{today_price:.2f}<ZG×0.99={zg*0.99:.2f})")

    # 2. 趋势逆转: 上涨→下跌 (B2在下跌趋势中买入是正常的, 不单独构成卖出)
    if trend_type == -2:
        if buy_point != 2:  # B2本身就是做下跌末端的, 趋势-2正常
            sell_reasons.append("趋势转为下跌")

    # 3. Chan卖点出现 (这是最强的卖出信号)
    if chan_sell_score > 0.5:
        sell_reasons.append(f"Chan卖点信号({chan_sell_score:.2f})")

    # 4. 买点消失: 只有同时满足以下条件才构成卖出
    if buy_point == 0 and chan_buy < 0.3 and signal_level == 0:
        if today_ret < -1.0 or trend_type <= 0 or chan_sell_score > 0.3:
            sell_reasons.append("买点信号消失(结构走坏)")

    # 5. 周线转空 (涨了的股票不算, 持有观察)
    if ema60_v > 0 and ema120_v > 0 and ema60_v <= ema120_v:
        if today_ret < -0.5:  # 只有跌了才构成卖出理由
            sell_reasons.append("周线转空(EMA60≤EMA120)")

    # 6. 均线破位: 跌破EMA20且EMA20向下
    if today_price < ema20 * 0.98:
        ema20_prev = float(ind.get('ema20', np.zeros(n))[i-1]) if i > 0 else ema20
        if ema20 < ema20_prev:
            sell_reasons.append(f"跌破EMA20(收盘{today_price:.2f}<EMA20={ema20:.2f})")

    # 7. 放量下跌
    if daily_ret < -0.02 and vol_ratio > 0.5:
        sell_reasons.append(f"放量下跌(跌{daily_ret*100:.1f}%+量比{vol_ratio:.1f})")

    # 8. 距MA20严重乖离
    if ma20_v > 0:
        dist_ma20 = (today_price - ma20_v) / ma20_v
        if dist_ma20 > 0.15:
            sell_reasons.append(f"距MA20过远({dist_ma20*100:.0f}%),止盈")
        elif dist_ma20 < -0.10:
            sell_reasons.append(f"深度破位MA20({dist_ma20*100:.0f}%)")

    # ── 决定 ──
    if sell_reasons:
        action = "卖出"
        reason_str = "; ".join(sell_reasons)
    else:
        # 仍符合策略: 买点还在, 结构完好
        bp_label = {3: 'B3', 2: 'B2', 1: 'B1', 0: '因子'}.get(buy_point, '—')
        action = "持有"
        reason_str = f"买点{bp_label}仍有效, 结构完好"

    return {
        "code": code,
        "name": name,
        "industry": industry,
        "buy_price": buy_price,
        "today_price": today_price,
        "today_ret": today_ret,
        "buy_point": buy_point,
        "trend_type": trend_type,
        "zg": zg if not np.isnan(zg) else None,
        "action": action,
        "reason": reason_str,
    }


def main():
    if len(sys.argv) < 2:
        print("用法: python strategy/analysis/selection_diagnosis.py output/选股数据_20260519.xlsx")
        sys.exit(1)

    excel_path = sys.argv[1]
    df = load_selection(excel_path)
    date_match = re.search(r'(\d{8})', str(excel_path))
    if not date_match:
        print("无法从文件名推断选股日期")
        sys.exit(1)
    selection_date = f"{date_match.group(1)[:4]}-{date_match.group(1)[4:6]}-{date_match.group(1)[6:8]}"

    print("=" * 70)
    print(f"  持仓诊断: {selection_date} 选股 → 最新数据")
    print("=" * 70)

    results = []
    for _, row in df.iterrows():
        code = str(row.get('股票代码', '')).zfill(6)
        name = str(row.get('股票名称', ''))
        industry = str(row.get('行业', ''))
        if not code or code == 'nan': continue

        r = evaluate_stock(code, name, industry, selection_date)
        results.append(r)
        ret_str = f"{r['today_ret']:+.2f}%" if r['today_ret'] else "—"
        bp_str = {3: 'B3', 2: 'B2', 1: 'B1', 0: '—'}.get(r['buy_point'], '?')
        zg_str = f"ZG={r['zg']:.2f}" if r['zg'] else "—"
        print(f"  [{r['action']}] {r['code']} {r['name']:<6s} "
              f"¥{r['buy_price']:.2f}→¥{r['today_price']:.2f} ({ret_str}) "
              f"| 买点:{bp_str} 趋势:{r['trend_type']} {zg_str}")
        print(f"        → {r['reason']}")

    # ── 汇总 ──
    holds = [r for r in results if r['action'] == '持有']
    sells = [r for r in results if r['action'] == '卖出']

    print(f"\n{'=' * 70}")
    print(f"  总结: 持有 {len(holds)}只 | 卖出 {len(sells)}只")
    print(f"{'=' * 70}")

    if sells:
        print(f"\n  🔴 卖出:")
        for r in sells:
            print(f"     {r['code']} {r['name']} — {r['reason']}")

    if holds:
        print(f"\n  🟢 持有:")
        for r in holds:
            ret_str = f"{r['today_ret']:+.2f}%" if r['today_ret'] else "—"
            print(f"     {r['code']} {r['name']} ({ret_str}) — {r['reason']}")


if __name__ == "__main__":
    main()
