#!/usr/bin/env python3
"""
生成 trade_orders.json 给 miniqmt 做盘中高抛低吸。

用法:
    python strategy/generate_trade_orders.py [--date 2026-07-16] [--cash 250000]

前置条件:
    需要 backtest_signals.csv 覆盖目标日期。若信号过期:
      cd strategy && python bt_execution.py  # 重新生成信号

流程:
    1. 加载现有信号 CSV (backtest_signals.csv)
    2. 若目标日期不在 CSV 中 → 报错退出
    3. 运行组合构建 → 得到目标持仓
    4. 对比当前持仓 → 生成 open/adjust/close
    5. 计算止盈止损价 (stop=现价*0.94, tp=现价*1.15)
    6. 输出 trade_orders.json + 更新 current_positions.json
"""

import os, sys, json, argparse, shutil, subprocess
import pandas as pd
import numpy as np
from datetime import date as date_type, datetime
from typing import Dict, List, Tuple

# 确保 strategy 目录在 path 中
_STRATEGY_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_STRATEGY_DIR)
if _STRATEGY_DIR not in sys.path:
    sys.path.insert(0, _STRATEGY_DIR)

from core.config_loader import load_config
from core.signal_store import SignalStore
from core.portfolio import PortfolioConstructor
from core.market_regime_detector import MarketRegimeDetector
from core.repo_sweep import compute_repo_sweep, sweep_guidance
from core.cb_sleeve import (compute_cb_sleeve, load_snapshot as cb_load_snapshot,
                            load_state as cb_load_state, save_state as cb_save_state)
from core.live_account import (load_account, save_account, mark_to_market,
                               last_nav, record_nav, apply_orders_to_cash)

config = load_config()

# ── 路径 ────────────────────────────────────────────────────────
DATA_PATH = config.get('paths.data', os.path.join(_PROJECT_DIR, 'data/stock_data/backtrader_data/'))
# ── 真实价层 (2026-10-17) ─────────────────────────────────────────────────────
# `DATA_PATH` 里是**复权价**: 信号/中枢/回测都在复权空间算, 收益与分红才正确。
# 但**下单价格必须是真实成交价** —— 复权价在除权日与真实价分叉(10转10 之后复权价
# 是真实价的 2 倍), 拿它出单会按错价位、错股数下单。
# 旧链路侥幸没事: 东财 qfq **每次除权整条重锚** ⇒ 末行 ≡ 真实价。表驱动复权改用
# **冻结锚**(满足铁律"更新今天不得改写历史") ⇒ 该巧合消失, 必须显式读裸价。
# 实测(raw_data/{code}/none.csv 是唯一 PIT 安全的价格输入)。
RAW_DATA_DIR = os.path.join(_PROJECT_DIR, 'data/stock_data/raw_data')
FUNDAMENTAL_DIR = os.path.join(_PROJECT_DIR, 'data/stock_data/fundamental_data')
SIGNALS_CSV = os.path.join(_STRATEGY_DIR, 'rolling_validation_results', 'backtest_signals.csv')
POSITIONS_FILE = os.path.join(_PROJECT_DIR, 'current_positions.json')
ORDERS_FILE = os.path.join(_PROJECT_DIR, 'trade_orders.json')
REBUILD_STATE_FILE = os.path.join(_PROJECT_DIR, 'live_rebuild_state.json')
# 跨日组合状态(平滑/波动率降仓/组合止损/连亏熔断/入场日期) + 账户台账(现金/净值史)
PORTFOLIO_STATE_FILE = os.path.join(_PROJECT_DIR, 'live_portfolio_state.json')

# 实盘调仓周期 = 回测同一旋钮(单一真源, 禁止两处各写一个数).
# 2026-10-14: 实盘此前硬编码 rebalance_interval=1(每日全重建), 而回测生产=10日历日.
# A轴实证(rb1臂, todate=10-13): interval=1 = NAV 818,735(−188,181), Sharpe 1.4988→1.1521,
# 笔数505→2,226, 均ret +3.79%→+1.00% = 广谱churn磨损(非彩票承载). 对齐后实盘与锚点同周期.
REBALANCE_INTERVAL = int(config.get('backtest.rebalance_days', 10))
OUTPUT_DIR = os.path.join(_PROJECT_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)
INDEX_DATA_PATH = os.path.join(DATA_PATH, 'sh000001_qfq.csv')

# ── 默认参数 ─────────────────────────────────────────────────────
# 2026-09-09: 对齐真实账户总资金(~50万, 用户确认). 旧值30万导致9/8起
# 持仓市值46.9万>1.5×30万=45万, 每日触发脏数据重置, 系统丢失真实持仓.
# 真实入金/出金后需同步此值. 仅影响实盘出单排仓, 不影响回测(yaml backtest.cash).
DEFAULT_CASH = 500000.0


def build_stock_file_map() -> Dict[str, str]:
    """扫描 DATA_PATH, 构建 {code: filepath} 映射."""
    file_map = {}
    for f in os.listdir(DATA_PATH):
        if f.startswith('._'):
            continue
        path = os.path.join(DATA_PATH, f)
        if f.endswith('_qfq.csv'):
            code = f[:-8]
        elif f.endswith('_hfq.csv'):
            code = f[:-8]
        else:
            continue
        file_map[code] = path
    return file_map


def build_st_filter(codes: List[str], target_date) -> Tuple[set, set]:
    """ST排除 + 基本面新鲜度检查 (2026-08-31 ST漏判修复; 2026-10-18 铁律修复).

    出单路径的独立ST校验, 不依赖回测信号CSV的过滤:
      - ST判定**逐字复用回测的消费逻辑**(bt_execution.py:2122-2133):
        `get_st_timeline(code)` → `searchsorted(avail, target_date, 'right') - 1` → `stf[idx]`
      - 新鲜度: 最新数据可用日期早于 target_date-120天 → 数据陈旧, ST判定可能失效 → 告警
    返回 (st_codes, stale_codes)。

    **2026-10-18 铁律修复**: 原实现自己**重推了一遍** ST 判定 —— 直读财务 CSV 的
    `股票简称` **可变列**。而回测侧 a7ea977 已把该列换成**冻结文件**
    `data/alternative_data/st_timeline_frozen.pkl`(理由见 core/st_timeline.py)。
    两边从此**判据不同**: 下一次基本面刷新改写简称(新戴帽/摘帽)时, 实盘立刻换答案
    而回测不变 ⇒ `f(t;T) ≠ f(t;t)`, 违反铁律。
    (实测 2026-10-16 目标日两边 0 分叉 —— 冻结是按"逐位保持行为"铸的, 所以这是
     **潜伏**分叉, 只在刷新后才现形。)

    改为调用**同一个** `get_st_timeline` ⇒ 两侧按构造恒等, 不再靠"重推一遍"对齐。
    ⚠ 冻结文件是**数据态**不是代码: 它不知道 2026-10-17 之后的新戴帽。要更新必须
    显式**重新冻结 + 登记数据重锚**, 不能靠这里偷偷读可变列。
    """
    from core.fundamental import FundamentalData
    fd = FundamentalData(FUNDAMENTAL_DIR)
    st_codes, stale_codes = set(), set()
    td = pd.Timestamp(target_date)
    cutoff = td - pd.Timedelta(days=120)
    key = td.strftime('%Y%m%d')
    for code in codes:
        df = fd._get_available_data(code, td)
        if df is None or df.empty:
            stale_codes.add(code)  # 无任何可用数据, 视为无法验证
            continue
        # ST判定: 与 bt_execution.py:2122-2133 逐字同构(同一数据源 + 同一查表)
        try:
            tl = fd.get_st_timeline(code)
            if tl is not None:
                _av, _stf = tl
                _idx = int(np.searchsorted(_av, key, side='right')) - 1
                if _idx >= 0 and bool(_stf[_idx]):
                    st_codes.add(code)
                    continue
        except Exception:
            pass  # 与回测一致: 时间线取不到 ⇒ 视为非ST, 不阻断出单
        try:
            latest_avail = pd.Timestamp(df['数据可用日期'].astype(str).max())
            if latest_avail < cutoff:
                stale_codes.add(code)
        except Exception:
            stale_codes.add(code)
    return st_codes, stale_codes


def build_raw_file_map() -> Dict[str, str]:
    """扫描 RAW_DATA_DIR, 构建 {code: none.csv 路径} —— **真实价**来源."""
    file_map = {}
    if not os.path.isdir(RAW_DATA_DIR):
        return file_map
    for code in os.listdir(RAW_DATA_DIR):
        p = os.path.join(RAW_DATA_DIR, code, 'none.csv')
        if os.path.exists(p):
            file_map[code] = p
    return file_map


_RAW_COLS = {'日期': 'datetime', '开盘': 'open', '最高': 'high', '最低': 'low', '收盘': 'close'}


def read_raw_prices(code: str, raw_map: Dict[str, str], cols=('datetime', 'close')):
    """读裸价(none.csv) 的指定列, 返回英文列名的 DataFrame; 失败返回 None."""
    filepath = raw_map.get(code)
    if not filepath:
        return None
    try:
        want = {k: v for k, v in _RAW_COLS.items() if v in cols}
        df = pd.read_csv(filepath, encoding='utf-8-sig', usecols=list(want.keys()))
        df = df.rename(columns=want)
        df['datetime'] = pd.to_datetime(df['datetime'])
        return df.sort_values('datetime').reset_index(drop=True)
    except Exception:
        return None


def price_scale(code: str, date, raw_map: Dict[str, str], stock_file_map: Dict[str, str]) -> float:
    """复权价 → 真实价 换算因子 `S(t) = raw(t)/P(t)`。

    信号里的缠论中枢是在**复权序列**上算的, 而 60 日高低/现价是**真实价**;
    两者要取 max/min 就必须同口径 ⇒ 用本因子把中枢搬到真实价空间。
    S 在两次除权之间是常数(非事件日 P 与 raw 同比), 除权日才跳变。
    任一日期取不到价 ⇒ 返回 1.0 并**告警**(静默返回默认值是本仓库已知的 bug 温床)。
    """
    dt = pd.Timestamp(date)
    r = read_raw_prices(code, raw_map, ('datetime', 'close'))
    if r is None:
        print(f"  [真实价][降级] {code} 无裸价文件, 复权↔真实 换算因子取 1.0")
        return 1.0
    rv = r.loc[r['datetime'] == dt, 'close']
    bf = stock_file_map.get(code)
    if bf is None or rv.empty or float(rv.iloc[0]) <= 0:
        print(f"  [真实价][降级] {code} {dt.date()} 裸价缺失, 换算因子取 1.0")
        return 1.0
    try:
        b = pd.read_csv(bf, parse_dates=['datetime'], usecols=['datetime', 'close'])
        bv = b.loc[b['datetime'] == dt, 'close']
        if bv.empty or float(bv.iloc[0]) <= 0:
            raise ValueError('复权价缺失')
        return float(rv.iloc[0]) / float(bv.iloc[0])
    except Exception as e:
        print(f"  [真实价][降级] {code} {dt.date()} 复权价读取失败({type(e).__name__}), 换算因子取 1.0")
        return 1.0


def load_prices_for_date(raw_map: Dict[str, str], target_date) -> Dict[str, float]:
    """读取所有股票在 target_date 的**真实收盘价**(裸价, 非复权价 —— 见 RAW_DATA_DIR 注释)."""
    prices = {}
    target_dt = pd.Timestamp(target_date)
    for code in raw_map:
        try:
            df = read_raw_prices(code, raw_map, ('datetime', 'close'))
            if df is None:
                continue
            row = df.loc[df['datetime'] == target_dt, 'close']
            if not row.empty:
                prices[code] = float(row.iloc[0])
        except Exception:
            continue
    print(f"价格数据: {len(prices)} 只股票有 {target_date} 的真实价")
    return prices


def get_latest_price_date():
    """最新K线交易日(指数文件末行) — 出单定价基准.

    2026-10-13修复(用户实盘反馈): 信号回退日的收盘价已过时(10-12信号配10-12收盘,
    而10-13收盘已可用), 股数/止损/监控市值须按最新收盘价算. 信号日与价格日分离.
    """
    try:
        df = pd.read_csv(INDEX_DATA_PATH, usecols=['datetime'])
        return pd.Timestamp(df['datetime'].iloc[-1]) if len(df) else None
    except Exception:
        return None


def load_last_known_close(code: str, raw_map: Dict[str, str],
                          stock_file_map: Dict[str, str] = None) -> float:
    """某股最后已知**真实**收盘价 — 当日停牌时价格日无行情行的兜底.

    118 只**退市股** raw_data 里只有 `qfq.csv` 没有 `none.csv` ⇒ 裸价层拿不到价。
    这类码不在股票池、不可交易, 但若残留在持仓文件里, 返回 0 会让监控市值静默归零。
    ⇒ 回退到复权文件末行并**告警**(口径不完全等价, 必须可见)。
    """
    df = read_raw_prices(code, raw_map, ('datetime', 'close'))
    if df is not None and len(df):
        return float(df['close'].iloc[-1])
    if stock_file_map and stock_file_map.get(code):
        try:
            b = pd.read_csv(stock_file_map[code], usecols=['close'])
            if len(b):
                print(f"  [真实价][降级] {code} 无裸价(退市股?), 用复权文件末行 "
                      f"{float(b['close'].iloc[-1]):.2f} 代真实价")
                return float(b['close'].iloc[-1])
        except Exception:
            pass
    return 0.0


def get_index_regime(target_date) -> Tuple[int, float, float, bool, bool]:
    """获取目标日期的市场状态."""
    if not os.path.exists(INDEX_DATA_PATH):
        return 0, 0.0, 0.0, False, False

    df = pd.read_csv(INDEX_DATA_PATH, parse_dates=['datetime'])
    detector = MarketRegimeDetector()
    regime_df = detector.generate(df)
    del df

    # 修复(2026-10-14): dt.date列是datetime.date对象, 与pd.Timestamp比较恒False
    # → row.empty恒真 → 本函数自2026-07-17(197bc321)起一直静默返回默认中性值,
    # 实盘出单层从未看到bear_risk/trend_score(回测侧用date对象比较, 一直正常).
    # 10/13实盘买6只 vs 回测BEAR空仓的分歧根因. 修复=转date对象比较.
    target_dt = pd.Timestamp(target_date).date()
    row = regime_df[regime_df['datetime'].dt.date == target_dt]
    if row.empty:
        return 0, 0.0, 0.0, False, False

    regime = int(row['regime'].values[0]) if 'regime' in row.columns else 0
    momentum = float(row['momentum_score'].values[0]) if 'momentum_score' in row.columns else 0.0
    trend = float(row['trend_score'].values[0]) if 'trend_score' in row.columns else 0.0
    bear_risk = bool(row['bear_risk'].values[0]) if 'bear_risk' in row.columns else False
    bear_risk_fast = bool(row['bear_risk_fast'].values[0]) if 'bear_risk_fast' in row.columns else False
    return regime, momentum, trend, bear_risk, bear_risk_fast


def compute_support_resistance(
    code: str,
    target_date,
    price: float,
    signal_store,
    raw_map: Dict[str, str],
    stock_file_map: Dict[str, str],
    kline_date=None,
) -> Tuple[float, float, dict]:
    """基于缠论结构计算支撑位(止损)和压力位(止盈).

    target_date=信号日(中枢数据来源), kline_date=K线回看锚定日(默认同信号日);
    出单用最新收盘价时kline_date=价格日, 60日高低含最新bar.

    止损 (支撑):
      1. 最近60日最低点 × 0.995
      2. 中枢下沿 ZD × 0.995
      3. 兜底: 当前价 × 0.93
      → 取三者最大值 (最紧的支撑)

    止盈 (压力):
      1. 最近60日最高点 × 1.005
      2. 中枢上沿 ZG × 1.005
      3. 上限: 当前价 × 1.25
      → 取三者最小值 (最早的压力)

    Returns:
        (support_price, resistance_price, detail_dict)
    """
    detail = {'code': code, 'price': price}

    # ── 1. 从信号中取缠论中枢数据 ──────────────────────────
    # ⚠ 中枢是**复权价**口径(回测在复权序列上算的), 而下面 60 日高低与现价是**真实价**。
    #   两者要取 max/min 就必须同口径 ⇒ 用 S = raw(t)/P(t) 把中枢搬到真实价空间。
    #   不同口径直接混用会让中枢一侧**恒输**(除权后复权价可比真实价小几倍)⇒ 中枢通道
    #   被静默忽略, 而输出上看不出差别 —— 正是本文件反复标记的"查表失败返回默认值"温床。
    sig = signal_store.get(code, target_date) if signal_store else None
    _S = price_scale(code, target_date, raw_map, stock_file_map)
    pivot_zg = (float(getattr(sig, 'chan_pivot_zg', np.nan)) * _S) if sig else np.nan
    pivot_zd = (float(getattr(sig, 'chan_pivot_zd', np.nan)) * _S) if sig else np.nan
    detail['pivot_zg'] = pivot_zg
    detail['pivot_zd'] = pivot_zd
    detail['price_scale'] = _S

    # ── 2. 从市场数据取60日高低点 (**真实价**, 止损止盈要挂在真实价位上) ──
    recent_high = price
    recent_low = price
    df = read_raw_prices(code, raw_map, ('datetime', 'high', 'low'))
    if df is not None:
        try:
            target_dt = pd.Timestamp(kline_date or target_date)
            df = df[df['datetime'] <= target_dt]
            if len(df) >= 20:
                lookback = min(60, len(df))
                recent_high = float(df['high'].iloc[-lookback:].max())
                recent_low = float(df['low'].iloc[-lookback:].min())
            else:
                # 静默降级修复(2026-10-14): 行数不足时 recent_high/low 保持 = price,
                # 止损/止盈退化为 93%/125% 兜底 — 必须可见, 否则用户按假位挂单.
                print(f"  [支撑压力][降级] {code} K线仅{len(df)}行(<20), "
                      f"止损/止盈退化为兜底 {price*0.93:.2f}/{price*1.25:.2f}")
            del df
        except Exception as _e:
            # 静默降级修复(2026-10-14): 原 `except Exception: pass` 会把读失败吞成
            # 兜底位(93%/125%), 与"读到真60日高低"在输出上无法区分 — 同
            # live-regime-blind-bug 一类"查表失败返回默认值"的温床.
            print(f"  [支撑压力][降级] {code} K线读取失败({type(_e).__name__}: {_e}), "
                  f"止损/止盈退化为兜底 {price*0.93:.2f}/{price*1.25:.2f}")
    else:
        print(f"  [支撑压力][降级] {code} 无裸价K线文件, "
              f"止损/止盈退化为兜底 {price*0.93:.2f}/{price*1.25:.2f}")
    detail['high_60d'] = recent_high
    detail['low_60d'] = recent_low

    # ── 3. 计算支撑位 (止损) ────────────────────────────────
    supports = [('93%兜底', price * 0.93)]
    if recent_low > 0:
        supports.append(('60日低点', recent_low * 0.995))
    if not np.isnan(pivot_zd) and pivot_zd > 0:
        supports.append(('中枢下沿ZD', pivot_zd * 0.995))

    best_support = max(supports, key=lambda x: x[1])
    support_price = round(best_support[1], 2)
    detail['support_reason'] = best_support[0]
    detail['support_candidates'] = [(r, round(v, 2)) for r, v in supports]

    # ── 4. 计算压力位 (止盈) ────────────────────────────────
    resistances = [('125%上限', price * 1.25)]
    if recent_high > 0:
        resistances.append(('60日高点', recent_high * 1.005))
    if not np.isnan(pivot_zg) and pivot_zg > 0:
        resistances.append(('中枢上沿ZG', pivot_zg * 1.005))

    best_resistance = min(resistances, key=lambda x: x[1])
    resistance_price = round(best_resistance[1], 2)
    detail['resistance_reason'] = best_resistance[0]
    detail['resistance_candidates'] = [(r, round(v, 2)) for r, v in resistances]

    return support_price, resistance_price, detail


def load_current_positions() -> Dict[str, dict]:
    """加载当前持仓."""
    if os.path.exists(POSITIONS_FILE):
        with open(POSITIONS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def save_current_positions(positions: Dict[str, dict]):
    """保存当前持仓(覆写前自动备份, 防止误运行不可逆)."""
    if os.path.exists(POSITIONS_FILE):
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        bak = f"{POSITIONS_FILE}.bak_{ts}"
        shutil.copy2(POSITIONS_FILE, bak)
        print(f"旧持仓已备份: {bak}")
    with open(POSITIONS_FILE, 'w', encoding='utf-8') as f:
        json.dump(positions, f, ensure_ascii=False, indent=2)
    print(f"持仓已保存: {POSITIONS_FILE}")


def load_rebuild_state():
    """读取上次完整重建日期(实盘跨进程调仓周期状态).

    缺失/损坏 → None = 视为未重建(build()首日强制重建), 与回测中
    _last_rebalance_date=None 的初值语义一致.
    """
    try:
        with open(REBUILD_STATE_FILE, 'r', encoding='utf-8') as f:
            _d = json.load(f).get('last_rebuild_date')
        return pd.Timestamp(_d).date() if _d else None
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"  [WARN] 调仓状态文件读取失败({e}), 按未重建处理")
        return None


def save_rebuild_state(last_rebuild_date):
    """写入上次完整重建日期. 仅在真实出单(非dry-run)路径调用."""
    try:
        _d = last_rebuild_date if hasattr(last_rebuild_date, 'strftime') else pd.Timestamp(last_rebuild_date).date()
        with open(REBUILD_STATE_FILE, 'w', encoding='utf-8') as f:
            json.dump({'last_rebuild_date': _d.strftime('%Y-%m-%d'),
                       'updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')},
                      f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"  [WARN] 调仓状态文件写入失败: {e}")
        return False


def build_orders(
    target_date: str,
    adjusted_positions: Dict[str, float],
    prices: Dict[str, float],
    prev_positions: Dict[str, dict],
    total_equity: float,
    signal_store,
    raw_map: Dict[str, str],
    stock_file_map: Dict[str, str],
    selection_map: Dict[str, dict] = None,
    price_date=None,
) -> Tuple[List[dict], Dict[str, dict]]:
    """
    对比当前持仓与目标持仓, 生成订单列表.
    止盈止损使用缠论结构的支撑/压力位.
    订单附带现价/股数/评分/权重 (2026-10-13用户要求: 现价,买多少,评分,权重比).

    Returns:
        (orders, new_positions)
    """
    orders = []
    new_positions = {}
    order_details = []  # 收集计算明细用于打印
    selection_map = selection_map or {}

    target_codes = set(adjusted_positions.keys())
    prev_codes = set(prev_positions.keys())

    # ── close: 之前持有但目标中已无 ──
    for code in sorted(prev_codes - target_codes):
        o = {'stock_code': _normalize_code(code), 'action': 'close'}
        _p = prices.get(code, 0)
        if _p > 0:
            o['price'] = round(_p, 2)
        _sh = prev_positions.get(code, {}).get('shares')
        if _sh:
            o['shares'] = _sh
        orders.append(o)

    # ── open / adjust / keep ──
    for code in sorted(target_codes):
        target_amount = adjusted_positions[code]
        if target_amount <= 0:
            if code in prev_codes:
                o = {'stock_code': _normalize_code(code), 'action': 'close'}
                _p = prices.get(code, 0)
                if _p > 0:
                    o['price'] = round(_p, 2)
                _sh = prev_positions.get(code, {}).get('shares')
                if _sh:
                    o['shares'] = int(_sh)
                orders.append(o)
            continue

        price = prices.get(code, 0)
        if price <= 0:
            continue

        prev = prev_positions.get(code)

        if prev is None:
            # 新建仓: 整手取整(科创板200股最小+1股递增), 不足最小申报则跳过
            shares = _lot_shares(code, target_amount, price)
            if shares < 1:
                print(f"  [SKIP] {code} 资金不足最小申报 (target={target_amount:.0f}, price={price:.2f})")
                continue
            actual_amount = shares * price
            sl, tp, detail = compute_support_resistance(
                code, target_date, price, signal_store, raw_map, stock_file_map,
                kline_date=price_date
            )
            _sel = selection_map.get(code) or {}
            orders.append({
                'stock_code': _normalize_code(code),
                'action': 'open',
                'amount': int(actual_amount),
                'stop_loss_price': sl,
                'take_profit_price': tp,
                'price': round(price, 2),
                'shares': int(shares),
                'score': round(float(_sel['score']), 4) if _sel.get('score') is not None else None,
                'weight': round(float(_sel['weight']), 4) if _sel.get('weight') is not None else None,
                'weight_pct': round(actual_amount / total_equity, 4) if total_equity else None,
            })
            new_positions[code] = {
                'entry_price': price,
                'amount': int(actual_amount),
                'shares': shares,
                'entry_date': str(price_date or target_date)[:10],
            }
            order_details.append((_normalize_code(code), 'open', detail))
        else:
            entry_price = prev.get('entry_price', price)
            prev_amount = prev.get('amount', 0)

            if abs(target_amount - prev_amount) > total_equity * 0.01:
                shares = _lot_shares(code, target_amount, price)
                if shares < 1:
                    # 调整后不足最小申报→清仓
                    orders.append({'stock_code': _normalize_code(code), 'action': 'close'})
                    print(f"  [CLOSE] {code} 调整后不足最小申报 (target={target_amount:.0f})")
                    new_positions.pop(code, None)
                    continue
                actual_amount = shares * price
                sl, tp, detail = compute_support_resistance(
                    code, target_date, price, signal_store, raw_map, stock_file_map,
                    kline_date=price_date
                )
                _sel = selection_map.get(code) or {}
                orders.append({
                    'stock_code': _normalize_code(code),
                    'action': 'adjust',
                    'amount': int(actual_amount),
                    'stop_loss_price': sl,
                    'take_profit_price': tp,
                    'price': round(price, 2),
                    'shares': int(shares),
                    'prev_shares': int(prev.get('shares', 0) or 0),
                    'score': round(float(_sel['score']), 4) if _sel.get('score') is not None else None,
                    'weight': round(float(_sel['weight']), 4) if _sel.get('weight') is not None else None,
                    'weight_pct': round(actual_amount / total_equity, 4) if total_equity else None,
                })
                new_positions[code] = {
                    'entry_price': entry_price,
                    'amount': int(actual_amount),
                    'shares': shares,
                    'entry_date': prev.get('entry_date', target_date),
                }
                order_details.append((_normalize_code(code), 'adjust', detail))
            else:
                new_positions[code] = prev

    return orders, new_positions, order_details


def build_monitor_list(
    target_date: str,
    positions: Dict[str, dict],
    prices: Dict[str, float],
    signal_store,
    raw_map: Dict[str, str],
    stock_file_map: Dict[str, str],
    price_date=None,
) -> List[dict]:
    """持仓监控清单(2026-08-27): 全部持仓的最新支撑/压力位+浮盈亏+过滤警示.

    订单只给 open/adjust 的票带止损止盈, 老持仓缺盘中参考位 ->
    每晚为全部持仓重算, 写入 trade_orders.json 的 'monitor' 字段.
    过滤警示复用三套过滤集(减持计划期/已执行减持/解禁), fail-open.
    """
    monitor = []
    if not positions:
        return monitor

    warn_map = {}
    _asof = str(price_date or target_date)[:10]
    try:
        from core.alternative_data import AlternativeDataProvider
        p = AlternativeDataProvider()
        plan_codes = p.get_reduction_plan_codes(_asof)
        exec_codes = p.get_reduction_codes(_asof, 30)
        unlock_codes = p.get_unlock_codes(_asof, 30, 0.05)
        for c in positions:
            w = []
            if c in plan_codes:
                w.append('减持计划期')
            if c in exec_codes:
                w.append('减持期')
            if c in unlock_codes:
                w.append('解禁临近')
            if w:
                warn_map[c] = w
    except Exception as e:
        print(f"  [monitor] 过滤集加载失败(跳过警示): {e}")

    for code in sorted(positions):
        pos = positions[code]
        price = prices.get(code, 0) or pos.get('entry_price', 0)
        try:
            sl, tp, _ = compute_support_resistance(
                code, target_date, price, signal_store, raw_map, stock_file_map,
                kline_date=price_date)
        except Exception:
            sl, tp = 0.0, 0.0
        entry = pos.get('entry_price', 0)
        pnl_pct = (price / entry - 1) * 100 if entry else 0.0
        days_held = ''
        try:
            days_held = (date_type.fromisoformat(_asof)
                         - date_type.fromisoformat(str(pos.get('entry_date', target_date))[:10])).days
        except Exception:
            pass
        monitor.append({
            'stock_code': _normalize_code(code),
            'shares': pos.get('shares'),
            'latest_price': round(price, 2),
            'market_value': round(price * pos.get('shares', 0)),
            'pnl_pct': round(pnl_pct, 1),
            'days_held': days_held,
            'stop_loss_price': round(sl, 2) if sl else None,
            'take_profit_price': round(tp, 2) if tp else None,
            'warnings': warn_map.get(code, []),
        })
    return monitor


def print_monitor_list(monitor: List[dict]):
    print(f"\n持仓监控清单 ({len(monitor)}只, 次日执行参考):")
    print(f"  {'代码':>12s} {'现价':>8s} {'盈亏%':>7s} {'持有天':>5s} {'止损(支撑)':>10s} {'止盈(压力)':>10s} {'警示'}")
    for m in monitor:
        warn = ' '.join(m['warnings'])
        print(f"  {m['stock_code']:>12s} {m['latest_price']:>8.2f} {m['pnl_pct']:>+7.1f} "
              f"{str(m['days_held']):>5s} {m['stop_loss_price'] or 0:>10.2f} "
              f"{m['take_profit_price'] or 0:>10.2f} {warn}")


def _normalize_code(code: str) -> str:
    """归一化股票代码: 6位数字 → 交易所后缀格式."""
    code = str(code).zfill(6)
    if code.startswith(('60', '68')):
        return f"{code}.SH"
    else:
        return f"{code}.SZ"


def _lot_shares(code: str, amount: float, price: float) -> int:
    """整手规则: 科创板(688/689)最小200股+1股递增, 其余100股整手. 返回0=不足最小申报."""
    if price <= 0:
        return 0
    if code.startswith(('688', '689')):
        return int(amount / price) if amount / price >= 200 else 0
    s = int(amount / price / 100) * 100
    return s if s >= 100 else 0


def main():
    parser = argparse.ArgumentParser(description='生成 trade_orders.json 给 miniqmt')
    parser.add_argument('--date', type=str, default=None,
                        help='目标日期 (YYYY-MM-DD), 默认最新交易日')
    parser.add_argument('--cash', type=float, default=DEFAULT_CASH,
                        help=f'总资金 (默认: {DEFAULT_CASH})')
    parser.add_argument('--dry-run', action='store_true',
                        help='仅打印结果, 不写入文件')
    parser.add_argument('--skip-freshness', action='store_true',
                        help='跳过数据新鲜度门禁 (仅调试用, 实盘勿用)')
    parser.add_argument('--force-reset', action='store_true',
                        help='确认持仓全清重选 (仅当核对后确认账户无真实持仓时使用)')
    parser.add_argument('--force-rerun', action='store_true',
                        help='同日重跑: 允许覆盖当日台账记录 (默认拒绝, 防止状态/现金二次记账)')
    args = parser.parse_args()

    # ── 数据新鲜度门禁 (fail-closed) ──────────────────────────
    if not args.skip_freshness:
        gate = os.path.join(_STRATEGY_DIR, 'check_data_freshness.py')
        rc = subprocess.call([sys.executable, gate])
        if rc != 0:
            print("\n数据新鲜度门禁未通过, 拒绝出单。"
                  "刷新数据后重试, 或调试用 --skip-freshness 绕过。")
            sys.exit(1)

    # ── 确定目标日期 ──────────────────────────────────────────
    if args.date:
        target_date = pd.Timestamp(args.date).date()
        date_specified = True
    else:
        target_date = date_type.today()
        date_specified = False
    print(f"目标日期: {target_date} | 总资金: {args.cash:,.0f}")

    # ── 1. 加载信号 ──────────────────────────────────────────
    stock_file_map = build_stock_file_map()
    print(f"股票文件: {len(stock_file_map)} 只")
    # 真实价层(见 RAW_DATA_DIR 注释): 下单/止损止盈用裸价, 信号/中枢/回测用复权价。
    raw_map = build_raw_file_map()
    print(f"裸价文件: {len(raw_map)} 只 (下单定价来源)")
    if not raw_map:
        print(f"[拒绝出单] 裸价目录为空: {RAW_DATA_DIR} —— 无法定价")
        sys.exit(1)

    if not os.path.exists(SIGNALS_CSV):
        print(f"\n错误: 信号文件不存在 ({SIGNALS_CSV})")
        print("请先运行: cd strategy && python bt_execution.py")
        sys.exit(1)

    signal_store = SignalStore()
    signal_store.finalize(SIGNALS_CSV)

    # 检查目标日期是否有信号
    # 修复(2026-10-14): 原为 `test_code = next(c for c in stock_file_map ...)` 抽**任意首只**
    # 股票试探 — build_stock_file_map 用 os.listdir, 顺序随文件系统, 该股票恰好无信号
    # (新股/停牌/不在信号universe)时误判为"当日无信号" → 静默回退到旧日期出单.
    # 改用 SignalStore 的日期索引判定, 与抽样无关.
    if not signal_store.has_date(target_date):
        if date_specified:
            print(f"\n错误: 目标日期 {target_date} 无信号数据")
            print(f"请更新信号CSV: cd strategy && python bt_execution.py")
            sys.exit(1)
        else:
            # 自动回退到信号CSV中的最新日期
            sig_df = pd.read_csv(SIGNALS_CSV, usecols=['date'])
            latest_sig_date = pd.Timestamp(sig_df['date'].max()).date()
            del sig_df
            print(f"  (今日 {target_date} 无信号, 回退到最新交易日期 {latest_sig_date})")
            target_date = latest_sig_date

    # ── 2. 获取价格 ───────────────────────────────────────────
    # 信号日(target_date)与价格日(price_date)分离: 信号取信号CSV最新日, 但订单定价
    # (现价/股数/止损/监控市值)用最新K线收盘 — 显式--date历史语义仍按该日定价
    if date_specified:
        price_date = pd.Timestamp(target_date)
    else:
        price_date = get_latest_price_date() or pd.Timestamp(target_date)
    if price_date > pd.Timestamp(target_date):
        print(f"  [价格日] 信号日 {str(target_date)[:10]} 收盘已过时, "
              f"订单定价改用最新K线收盘日 {str(price_date)[:10]}")

    # ── 信号新鲜度硬闸 (fail-closed, 2026-10-14) ──────────────
    # 教训(10/13实盘): 锚链实验晚于每日流未推进todate → 信号止于10-12, 旧行为仅
    # 打印"(今日X无信号, 回退到最新交易日期Y)"便继续出单 = 旧信号配新价格。
    # 规则: 信号CSV最后日期必须 ≥ 价格日(订单定价所用K线日), 否则拒绝出单。
    _sig_max = signal_store.max_date()
    _px_date = str(pd.Timestamp(price_date).date())
    if _sig_max and _sig_max < _px_date and not args.skip_freshness:
        print(f"\n[拒绝出单] 信号止于 {_sig_max}, 而价格日已到 {_px_date} — 信号落后于数据。")
        print(f"  请先重生成信号: cd strategy && python bt_execution.py")
        sys.exit(1)
    prices = load_prices_for_date(raw_map, price_date)

    # ── 3. 构建候选 universe ─────────────────────────────────
    # 基础过滤: 有价格, 非指数, 非低价股
    # ST 过滤: 使用 stock_pool (与回测统一), 不在池内则跳过
    stock_pool_enabled = config.get('stock_pool.enabled', True)
    allowed_codes = None
    if stock_pool_enabled:
        from core.stock_pool import get_stock_pool, _quarter_boundary_prev, _month_boundary_prev, get_pool_membership_map
        pool_calendar = config.get('stock_pool.pool_calendar', 'off')
        if pool_calendar in ('quarterly', 'monthly', 'daily'):
            # 0f日历池: quarterly=target_date最近季度末 / monthly=最近月末 /
            # daily=target_date当日 的as-of成员 ∩ 最新流动性池(实盘安全滤镜)。
            # 与回测信号membership闸同构。
            if pool_calendar == 'daily':
                _b = pd.Timestamp(target_date)
            elif pool_calendar == 'monthly':
                _b = _month_boundary_prev(pd.Timestamp(target_date))
            else:
                _b = _quarter_boundary_prev(pd.Timestamp(target_date))
            # 0f-v3松弛参数: 日历membership与安全滤镜用同一relax规则 (回测同构)
            _rf = config.get('stock_pool.pool_relax_floor', 1.0)
            _rm = config.get('stock_pool.pool_relax_momentum', None)
            _m = get_pool_membership_map([_b], relax_floor=_rf, relax_momentum=_rm)
            _cal = _m[_b]
            _today = get_stock_pool(relax_floor=_rf, relax_momentum=_rm)
            _dropped = _cal - _today
            allowed_codes = _cal & _today
            if _dropped:
                print(f"  [安全滤镜] {len(_dropped)} 只日历成员近期失流动性, "
                      f"排除买单: {sorted(_dropped)[:15]}")
            print(f"股票池(日历{_b.date()}+安全滤镜): {len(_cal)} ∩ {len(_today)} "
                  f"= {len(allowed_codes)} 只")
        else:
            allowed_codes = get_stock_pool()
            print(f"股票池: {len(allowed_codes)} 只 (含指数)")

    universe = []
    for code in stock_file_map:
        if code == 'sh000001':
            continue
        if code not in prices or prices[code] <= 0:
            continue
        if prices[code] < 2.0:
            continue
        if allowed_codes is not None and code not in allowed_codes:
            continue
        universe.append(code)

    # ST过滤 + 基本面新鲜度检查 (2026-08-31修复): 出单路径独立校验, 不依赖回测信号CSV
    st_codes, stale_codes = build_st_filter(universe, target_date)
    if st_codes:
        print(f"  [ST排除] {len(st_codes)} 只: {sorted(st_codes)}")
        universe = [c for c in universe if c not in st_codes]
    if stale_codes:
        print(f"  [WARN] {len(stale_codes)} 只基本面数据陈旧>120天, ST判定可能失效")

    # 科创板权限(2026-10-13用户确认已有): 9/14的688/689出单层剔除撤销,
    # 回测口径(含科创板)与实盘可交易集重合. 整手差异由 _lot_shares 处理.

    print(f"可交易股票: {len(universe)} 只 (过滤后)")

    # ── 4. 获取市场状态 ──────────────────────────────────────
    market_regime, momentum_score, trend_score, bear_risk, bear_risk_fast = get_index_regime(
        target_date
    )
    regime_names = {1: '牛市', 0: '中性', -1: '熊市'}
    print(f"市场状态: {regime_names.get(market_regime, '未知')} (regime={market_regime})")

    # ── 5. 运行组合构建 ──────────────────────────────────────
    # 调仓周期对齐回测(2026-10-14): 回测生产=10日历日, 实盘曾=1(每日全重建, 已证伪).
    # 跨进程状态由 live_rebuild_state.json 承载 — 进程每次重启, 必须显式注入上次重建日,
    # 否则 _last_rebalance_date=None 会让 build() 每天都走完整重建(即 interval=1 行为).
    _rebuild_state = load_rebuild_state()
    portfolio = PortfolioConstructor(rebalance_interval=REBALANCE_INTERVAL)
    # 跨日组合状态 (2026-10-22 架构修复): 旧路径每次进程只注入 _last_rebalance_date,
    # 其余状态全部归零 ⇒ 实盘丢失 ①敞口平滑 current_exposure ②波动率降仓(20日收益窗)
    # ③组合止损(peak/drawdown) ④连亏熔断 ⑤入场日期/min_hold. 回测实例跨全史存活,
    # 状态连续. 这里用类里已有但此前无人调用的 save/restore_tracking_state 落地.
    _state_restored = False
    if os.path.exists(PORTFOLIO_STATE_FILE):
        portfolio.restore_tracking_state(PORTFOLIO_STATE_FILE)
        _state_restored = True
        print(f"组合状态: 已恢复 (E={portfolio.current_exposure:.3f}, "
              f"peak={portfolio.peak_equity}, stop={int(portfolio._stop_loss_triggered)}, "
              f"clb={int(portfolio._clb_triggered)}, "
              f"收益窗={len(portfolio._daily_returns)}/{portfolio.vol_lookback})")
    else:
        print("组合状态: 无记录 → 初始状态起步 (与回测首日语义一致)")
    if _rebuild_state is not None:
        portfolio._last_rebalance_date = _rebuild_state
        print(f"调仓状态: 上次完整重建 {_rebuild_state} (间隔={REBALANCE_INTERVAL}日历日)")
    else:
        print(f"调仓状态: 无记录 → 本次完整重建 (间隔={REBALANCE_INTERVAL}日历日)")

    # 读取当前持仓
    prev_positions = load_current_positions()

    # 价格日无行情的持仓(当日停牌): 补最后已知收盘价, 订单/监控显示不落成本价
    for _code in prev_positions:
        if _code not in prices:
            _lp = load_last_known_close(_code, raw_map, stock_file_map)
            if _lp > 0:
                prices[_code] = _lp
                print(f"  [停牌补价] {_code} 价格日无行情, 用最后收盘 {_lp:.2f}")

    # (科创板权限已确认, 9/14的幻影持仓清理撤销 — 688/689持仓按真实持仓对待)

    # ── 账户台账 (2026-10-22 架构修复): nav = 现金 + Σshares×收盘, 实盘复利 ──
    # 旧口径 total_equity ≡ 静态 --cash, 持仓 amount 是下单时市值且从不更新 ⇒
    # 浮盈浮亏不进净值、已实现盈亏永久丢失、账户增长后系统性低配. 详见 core/live_account.py.
    # 历史回放写保护 (2026-10-22): 台账/持仓是真实账户状态, 只有"目标日=信号最新日"
    # 才是实盘决策. 历史 --date 写文件会把历史订单记进今天的账户, 必须显式 --dry-run.
    _sig_latest = str(signal_store.max_date() or '')[:10]
    if not args.dry_run and _sig_latest != str(target_date)[:10]:
        print(f"\n[拒绝写入] 目标日 {target_date} ≠ 信号最新日 {_sig_latest}")
        print(f"  历史回放会污染 live_account.json / current_positions.json; 请加 --dry-run.")
        sys.exit(1)

    account_capital = args.cash
    account = load_account(account_capital, prev_positions)

    # amount 统一为"最新收盘市值" (旧值=下单时市值): build_orders 的 1% 调整阈值、
    # 卖出回款估算、监控盈亏都读 amount — 陈旧值会让"该调不调/不该调乱调".
    for _code, _info in prev_positions.items():
        _sh = float(_info.get('shares', 0) or 0)
        if _sh <= 0:
            _ep = float(_info.get('entry_price', 0) or 0)
            if _ep > 0:
                _sh = float(_info.get('amount', 0) or 0) / _ep
                _info['shares'] = _sh
        _px = float(prices.get(_code, 0) or _info.get('entry_price', 0) or 0)
        if _sh > 0 and _px > 0:
            _info['amount'] = int(round(_sh * _px))

    _pos_value, _n_missing = mark_to_market(prev_positions, prices)
    _nav = float(account['cash']) + _pos_value
    _last_nav = last_nav(account)
    _last_date = account.get('last_date')
    _same_day = bool(_last_date) and str(_last_date)[:10] == str(target_date)[:10]
    print(f"账户台账: 现金{account['cash']:,.0f} + 持仓市值{_pos_value:,.0f} = NAV {_nav:,.0f}"
          + (f" | 上期NAV {_last_nav:,.0f} ({_last_date})" if _last_nav else " | 首日"))
    if _n_missing:
        print(f"  [WARN] {_n_missing} 只持仓无当日行情, 按 entry_price 估值")
    if _same_day and not args.dry_run and not args.force_rerun:
        print(f"\n[拒绝出单] 台账已有 {target_date} 记录: 同日重跑会二次记账(收益窗/现金/状态).")
        print(f"  若首次运行在中途失败, 请先核对/还原 live_account.json、"
              f"{os.path.basename(PORTFOLIO_STATE_FILE)}、current_positions.json; "
              f"确认无误后加 --force-rerun (仅重生成订单, 不再改台账/状态).")
        sys.exit(1)

    # 脏数据守卫(2026-08-30 P0 + 2026-10-22 台账口径): 持仓市值 > 1.5×上期NAV
    # 只可能是持仓文件漂移/大额入金. 自动全清是危险动作(0h 2026-09-17), 必须显式确认.
    _ref_nav = _last_nav if _last_nav else float(account_capital)
    if _pos_value > _ref_nav * 1.5:
        if not args.force_reset:
            print(f"\n[拒绝出单] 持仓市值{_pos_value:,.0f} > 1.5×上期NAV{_ref_nav:,.0f},"
                  f" 持仓文件与真实账户可能已漂移, 自动全清重选会与真实持仓叠加重复买入。")
            print(f"  请核对真实持仓后二选一: 手动修正 {POSITIONS_FILE}; "
                  f"或确认账户无真实持仓/已入金后加 --force-reset 全清重选。")
            sys.exit(1)
        print(f"  [WARN] 持仓市值{_pos_value:,.0f} > 1.5×上期NAV{_ref_nav:,.0f}, "
              f"--force-reset 已确认: 清空持仓, 现金重置为初始资金{account_capital:,.0f}")
        prev_positions = {}
        _pos_value = 0.0
        _nav = float(account_capital)
        account['cash'] = float(account_capital)
        account['initial_capital'] = float(account_capital)

    # 日收益喂给波动率降仓窗口 (回测 bt_execution:2413 每日 update_returns 的同构).
    if _last_nav and _last_nav > 0 and not _same_day:
        _dret = _nav / _last_nav - 1.0
        portfolio.update_returns(_dret)
        print(f"日收益: {_dret:+.4%} | 波动率窗口 {len(portfolio._daily_returns)}/"
              f"{portfolio.vol_lookback}")
    elif _same_day:
        print(f"  [WARN] 同日重跑({target_date}): 不重复计入日收益; "
              f"台账/状态写入将被跳过")

    current_positions = {
        code: float(info['amount']) for code, info in prev_positions.items()
    }

    # 构建成本信息
    cost = {}
    for code, info in prev_positions.items():
        cost[code] = [info['entry_price'], info['entry_price']]  # [avg_cost, current_cost]

    # 排仓尺度=账户NAV(与回测同构); 可用现金=台账现金
    total_equity = _nav
    cash = max(float(account['cash']), 0.0)

    adjusted = portfolio.build(
        date=target_date,
        universe=universe,
        current_positions=current_positions,
        signal_store=signal_store,
        cash=cash,
        prices=prices,
        market_regime=market_regime,
        momentum_score=momentum_score,
        trend_score=trend_score,
        bear_risk=bear_risk,
        bear_risk_fast=bear_risk_fast,
        cost=cost,
    )

    # 调仓状态落盘: build()通过闸门时会把 _last_rebalance_date 置为本次日期;
    # trend_bear双确认清仓路径在闸门之前早退(portfolio.py:2243) → 不推进, 与回测一致.
    _new_rebuild_date = getattr(portfolio, '_last_rebalance_date', None)
    if _new_rebuild_date is not None and _new_rebuild_date != _rebuild_state:
        if args.dry_run:
            print(f"  [调仓状态][dry-run] 完整重建日将更新: {_rebuild_state} → {_new_rebuild_date} (不落盘)")
        elif save_rebuild_state(_new_rebuild_date):
            print(f"  [调仓状态] 完整重建日更新: {_rebuild_state} → {_new_rebuild_date}")
    else:
        print(f"  [调仓状态] 本次未做完整重建(不出新选股, 仅出场/清仓)")

    # 持仓中的ST强制清仓 (2026-08-31修复): 从目标中移除 → build_orders生成close
    held_st = [c for c in current_positions if c in st_codes]
    if held_st:
        print(f"  [ST强制卖出] 持仓含ST: {held_st}")
        for c in held_st:
            adjusted.pop(c, None)
    held_stale = [c for c in current_positions if c in stale_codes]
    if held_stale:
        print(f"  [WARN] 持仓基本面数据陈旧, ST状态无法验证: {held_stale}")

    # ── 6. 生成订单 ──────────────────────────────────────────
    # 0.98 目标敞口闸 (2026-10-22, 与回测 bt_execution.py:2459-2469 同构):
    # 回测把**目标持仓总市值**压到 98%×(现金+持仓市值), 实盘旧路径只在买单层
    # 用 100% 可用现金兜底 ⇒ 满仓时实盘比回测多 2pp 敞口且无视目标总市值.
    _target_total = float(sum(v for v in adjusted.values() if v > 0))
    if _target_total > _nav * 0.98:
        _tscale = (_nav * 0.98) / max(_target_total, 1.0)
        adjusted = {c: v * _tscale for c, v in adjusted.items()}
        print(f"  [0.98闸] 目标市值{_target_total:,.0f} > 98%×NAV{_nav*0.98:,.0f}"
              f" → 等比缩放 {_tscale:.4f}")
    selection_map = {}
    if hasattr(portfolio, 'last_selection') and portfolio.last_selection:
        for sel in portfolio.last_selection:
            selection_map[str(sel.get('code', '')).zfill(6)] = sel

    orders, new_positions, order_details = build_orders(
        str(target_date), adjusted, prices, prev_positions, total_equity,
        signal_store, raw_map, stock_file_map, selection_map=selection_map,
        price_date=price_date,
    )

    # ── 0h 现金闸 (2026-09-17): 买单总额 ≤ 可用现金 + 同日卖出回款 ──
    # 实锤: 9/17晨 pos_sum=673,705>50万时 cash=0 仍出320k买单 — build()内部
    # total_equity=cash+pos_sum 随shadow book漂移放大(槽位=equity/20000), 排仓
    # 尺度失真 → 6只新开仓无融资写入持仓文件 → 膨胀至993,544。此闸只限实盘
    # 出单路径, 回测(yaml backtest.cash)不动。
    _sell_proceeds = 0.0
    _buy_total = 0.0
    for o in orders:
        _code = o['stock_code'].split('.')[0].zfill(6)
        _prev_amt = prev_positions.get(_code, {}).get('amount', 0)
        if o['action'] == 'close':
            _sell_proceeds += _prev_amt
        elif o['action'] == 'open':
            _buy_total += o['amount']
        elif o['action'] == 'adjust':
            _delta = o['amount'] - _prev_amt
            if _delta > 0:
                _buy_total += _delta
            else:
                _sell_proceeds += -_delta
    _avail = max(float(account['cash']), 0.0) + _sell_proceeds
    if _buy_total > _avail + 1:
        _scale = (_avail / _buy_total) if _buy_total > 0 else 0.0
        print(f"\n[现金闸] 买单总额{_buy_total:,.0f} > 可用资金{_avail:,.0f} "
              f"(台账现金{max(float(account['cash']), 0):,.0f} + 卖出回款{_sell_proceeds:,.0f})")
        _drop = []
        for o in orders:
            if o['action'] not in ('open', 'adjust'):
                continue
            _code = o['stock_code'].split('.')[0].zfill(6)
            _prev_amt = prev_positions.get(_code, {}).get('amount', 0)
            if o['action'] == 'adjust' and o['amount'] <= _prev_amt:
                continue  # 净卖出单不受现金闸约束
            _price = prices.get(_code, 0)
            if _price <= 0:
                continue
            _orig = o['amount']
            if o['action'] == 'open':
                _new_amt = o['amount'] * _scale
                _prev_amt = 0
            else:
                _new_amt = _prev_amt + (o['amount'] - _prev_amt) * _scale
            # 修复(2026-10-14): 原为硬编码100股整手 — 科创板(688/689)最小申报200股,
            # 缩放后 _new_amt/_price ∈ [100,200) 时会产出 100 股的无效单(_shares>=100 通过检查).
            # 改用与建单层同一函数 _lot_shares(科创板=200股起+1股递增, 其余=100股整手).
            # 非688/689 逐位等价: _lot_shares 的 "<100 返回0" 对应此处 "<1 取消".
            _shares = _lot_shares(_code, _new_amt, _price)
            if _shares < 1 or _new_amt <= _prev_amt:
                print(f"  [现金闸] {o['stock_code']} 缩减后不足1手/无净买入, 取消该单")
                _drop.append(o)
                if o['action'] == 'open':
                    new_positions.pop(_code, None)
                else:
                    new_positions[_code] = prev_positions[_code]
                continue
            _amt = int(_shares * _price)
            o['amount'] = _amt
            o['shares'] = _shares  # 2026-10-13: 现金闸缩放后同步shares/weight_pct(否则打印与金额失同步)
            if total_equity:
                o['weight_pct'] = round(_amt / total_equity, 4)
            new_positions[_code]['amount'] = _amt
            new_positions[_code]['shares'] = _shares
            print(f"  [现金闸] {o['stock_code']} {_amt:>8,} ({_shares}股×{_price:.2f}, 原{_orig:,})")
        for o in _drop:
            orders.remove(o)

    # ── 逆回购sweep估算 (2026-09-21, 实盘路径预备, yaml默认关闭) ──
    # 订单后闲置现金 = 现金 + 卖出回款 − 最终买单总额(现金闸缩放后重算),
    # 与0h现金闸同口径。纯估算建议块: 以QMT侧执行时实际可用余额为准。
    _buy_final = 0.0
    for o in orders:
        if o['action'] not in ('open', 'adjust'):
            continue
        _code = o['stock_code'].split('.')[0].zfill(6)
        _prev_amt = prev_positions.get(_code, {}).get('amount', 0)
        if o['action'] == 'open':
            _buy_final += o['amount']
        elif o['amount'] > _prev_amt:
            _buy_final += o['amount'] - _prev_amt
    _post_order_cash = max(float(account['cash']), 0.0) + _sell_proceeds - _buy_final

    # ── 转债hybrid10 sleeve建议块 (2026-10-01, 实盘路径预备, yaml默认关闭) ──
    # 月频全换仓差分; 出资来自闲置现金(与repo_sweep共享池: 转债先, sweep扫余量)。
    # 研究依据 strategy/analysis/cov_research_20260926.md v3: hybrid10
    # (rank(债价)+w×rank(纯债溢价率), w=1/n=10/cap=130) 月Sharpe 1.02, 危机月12/12。
    # cb块任何失败都只跳过本块, 绝不阻塞股票订单(双保险: 内层catch+外层try)。
    _cb_cfg = config.get('live_monitoring.cb_sleeve', {}) or {}
    _cb_result = None
    if (_cb_cfg or {}).get('enabled', False):
        try:
            _cb_cache_dir = (os.environ.get('QUANT_CB_CACHE')
                             or os.path.join(_PROJECT_DIR, 'data', 'alternative_data'))
            _cb_result = compute_cb_sleeve(
                _post_order_cash, _cb_cfg, cb_load_state(),
                cb_load_snapshot(_cb_cache_dir, pd.Timestamp(target_date)),
                pd.Timestamp(target_date))
        except Exception as e:
            print(f"  [cb_sleeve] 生成失败(不影响股票订单): {e}")
    _cb_spend = (_cb_result['spend_est']
                 if _cb_result and _cb_result.get('rebalanced') else 0.0)

    _sweep_cfg = config.get('live_monitoring.repo_sweep', {}) or {}
    _sweep = compute_repo_sweep(max(_post_order_cash - _cb_spend, 0.0), _sweep_cfg)

    output = {
        'date': str(target_date),
        'price_date': str(price_date)[:10],
        'orders': orders,
    }
    if _cb_result and _cb_result.get('rebalanced'):
        output['cb_sleeve'] = {
            'month': _cb_result['month'],
            'picks': _cb_result['picks'],
            'alloc_total': _cb_result['alloc_total'],
            'spend_est': _cb_result['spend_est'],
            'reason': _cb_result['reason'],
            'orders': _cb_result['orders'],
            'notes': _cb_result['notes'],
        }
        print(f"\n[转债sleeve] {_cb_result['reason']}")
        for o in _cb_result['orders']:
            _act = '买入' if o['action'] == 'open' else '卖出'
            print(f"  {_act:4s} {o['code']} {o['qty_zhang']:>4}张 "
                  f"≈{o['est_amount']:>8,.0f}元 (估价{o['est_price']:.2f}/张)")
    if _sweep['enabled']:
        output['repo_sweep'] = {
            'code': _sweep['code'],
            'tenor': _sweep['tenor'],
            'amount': _sweep['amount'],
            'reserve': _sweep['reserve'],
            'estimated_annual_yield': _sweep['estimated_annual_yield'],
            'reason': _sweep['reason'],
            'notes': sweep_guidance(_sweep),
        }
        print(f"\n[逆回购sweep] {_sweep['reason']}")
        if _sweep['amount'] >= 1000:
            print(f"  建议单: 卖出 {_sweep['code']} {_sweep['amount']:,}元 "
                  f"(年化估算≈{_sweep['estimated_annual_yield']:,.0f}元, 股票单成交后最后下)")




    print(f"\n{'='*60}")
    print(f"订单生成: {len(orders)} 条")
    for o in orders:
        if o['action'] == 'close':
            _extra = ''
            if o.get('price') and o.get('shares'):
                _extra = f" 现价={o['price']:.2f} 全卖{o['shares']:,}股≈{o['price']*o['shares']:,.0f}元"
            print(f"  卖出清仓 {o['stock_code']}{_extra}")
        else:
            _sc = o.get('score')
            _sc_s = f"{_sc:.4f}" if _sc is not None else "  —   "
            _wp = o.get('weight_pct')
            _wp_s = f"{_wp*100:.1f}%" if _wp is not None else "—"
            if o['action'] == 'open':
                _act = f"买入 {o.get('shares', 0):>5,}股"
            else:
                _prev_sh = prev_positions.get(
                    o['stock_code'].split('.')[0].zfill(6), {}).get('shares', 0) or 0
                _d = o.get('shares', 0) - _prev_sh
                _act = (f"调仓 {o.get('shares', 0):,}股(净{'+' if _d >= 0 else ''}{_d:,}股)"
                        if _d else f"持有 {o.get('shares', 0):,}股")
            print(f"  {o['action']:6s} {o['stock_code']} "
                  f"现价={o.get('price', 0):>7.2f} "
                  f"{_act} "
                  f"金额={o['amount']:>8,} "
                  f"评分={_sc_s} "
                  f"权重={_wp_s:>5s} "
                  f"止损={o.get('stop_loss_price', 'N/A'):>7} "
                  f"止盈={o.get('take_profit_price', 'N/A'):>7}")

    # ── 打印支撑/压力计算明细 ─────────────────────────────────
    if order_details:
        print(f"\n止盈止损计算明细:")
        print(f"  {'代码':>12s} {'现价':>8s} {'止损(支撑)':>10s} {'来源':>12s} {'止盈(压力)':>10s} {'来源':>12s}")
        print(f"  {'-'*70}")
        for code, action, d in order_details:
            sl = d.get('support_reason', '')
            tp = d.get('resistance_reason', '')
            # 从 orders 中找到对应的 sl/tp
            sl_price = next((o['stop_loss_price'] for o in orders if o.get('stock_code') == code and 'stop_loss_price' in o), 0)
            tp_price = next((o['take_profit_price'] for o in orders if o.get('stock_code') == code and 'take_profit_price' in o), 0)
            print(f"  {code:>12s} {d['price']:>8.2f} {sl_price:>10.2f} {sl:>12s} {tp_price:>10.2f} {tp:>12s}")

    # ── 持仓监控清单: 全持仓最新支撑/压力 + 过滤警示 ──────────
    try:
        monitor = build_monitor_list(str(target_date), new_positions, prices,
                                     signal_store, raw_map, stock_file_map,
                                     price_date=price_date)
        output['monitor'] = monitor
        print_monitor_list(monitor)
    except Exception as e:
        print(f"  [monitor] 生成失败(不影响订单): {e}")

    if args.dry_run:
        print("\n[Dry-run] 不写入文件")
        print(json.dumps(output, ensure_ascii=False, indent=2))
        return

    # ── 7. 写入文件 ──────────────────────────────────────────
    with open(ORDERS_FILE, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n订单已写入: {ORDERS_FILE}")

    # ── 台账 + 组合状态落盘 (2026-10-22 架构修复) ──────────────
    # 顺序: 订单 → 持仓 → 台账 → 组合状态. 台账 last_date 是"今日已记账"提交标记;
    # 同日重跑(--force-rerun)只重写订单, 绝不二次记账/二次推进状态.
    if _same_day:
        print("  [台账] 同日重跑(--force-rerun): 跳过持仓/净值/现金/状态写入 "
              "(请人工核对持仓与现金)")
    else:
        save_current_positions(new_positions)
        _pre_cash = float(account['cash'])
        record_nav(account, str(target_date), _nav, _pre_cash, _pos_value)
        _sm = apply_orders_to_cash(account, orders, prev_positions, prices)
        save_account(account)
        print(f"  [台账] 成交估算: 买{_sm['buy_amt']:,.0f} 卖{_sm['sell_amt']:,.0f} "
              f"费用{_sm['fees']:,.0f} → 现金{account['cash']:,.0f}")
        if _sm['skipped']:
            print(f"  [WARN] 无价跳过记账: {_sm['skipped']}")
        portfolio.save_tracking_state(PORTFOLIO_STATE_FILE)
        print(f"组合状态已保存: {PORTFOLIO_STATE_FILE}")

    # 转债sleeve状态落盘 (月度节奏跟踪, 独立于股票持仓文件)
    if _cb_result and _cb_result.get('rebalanced'):
        cb_save_state(_cb_result['new_state'])
        print(f"[转债sleeve] 状态已更新: {_cb_result['new_state']['month']} "
              f"{len(_cb_result['new_state']['positions'])}只")

    # ── 8. 打印选股详情 + 保存CSV ─────────────────────────────
    if hasattr(portfolio, 'last_selection') and portfolio.last_selection:
        print(f"\n选股详情:")
        rows = []
        for sel in portfolio.last_selection:
            print(f"  {sel['code']:>8s}  score={sel['score']:.4f}  "
                  f"weight={sel.get('weight', 0):.3f}  "
                  f"rank_pct={sel.get('rank_pct', 0):.3f}  "
                  f"industry={sel.get('industry', '')}  "
                  f"factor={sel.get('factor_name', '')}")
            rows.append(sel)
        if rows:
            sel_df = pd.DataFrame(rows)
            csv_path = os.path.join(OUTPUT_DIR, f'选股数据_{target_date}.csv')
            sel_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            print(f"\n选股数据已保存: {csv_path}")


if __name__ == '__main__':
    main()
