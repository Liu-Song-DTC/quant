"""
股票池模块 - 按流动性/质量筛选，控制回测和选股规模

筛选逻辑 (质量优先):
1. 次新股过滤: 排除上市不足1年的股票（K线数据不足，缠论结构不可靠）
2. 价格过滤: 排除仙股(<3元)和过高价股(>200元)
3. 僵尸股过滤: 排除日均换手率<0.5%的股票
4. 流动性地板: 排除日均成交额<5000万的股票
5. 复合排名: 60%流动性 + 25%价格稳定 + 15%交易活跃度
6. 排雷: ST + 科创板 + 异常数据
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path


def _get_data_dir():
    """获取backtrader数据目录"""
    base = Path(__file__).parent.parent.parent
    return str(base / 'data' / 'stock_data' / 'backtrader_data')


def _quarter_boundary_prev(t: pd.Timestamp) -> pd.Timestamp:
    """t之前(含)最近的季度末 (3/31, 6/30, 9/30, 12/31)."""
    q = (t.month - 1) // 3  # 0..3
    q_end_month = q * 3 + 3  # 3, 6, 9, 12 (恒有效)
    end = pd.Timestamp(t.year, q_end_month, 1) + pd.offsets.MonthEnd(0)
    if end > t:
        if q == 0:
            end = pd.Timestamp(t.year - 1, 12, 31)
        else:
            end = pd.Timestamp(t.year, q * 3, 1) + pd.offsets.MonthEnd(0)
    return end


def _quarter_boundaries(earliest_date, todate):
    """季度边界列表(升序): 从prev(earliest_date)到prev(todate)的全部季度末.

    单调性保证: 任何date ∈ [earliest_date, todate], 其最近季度末边界b≤date
    必在列表中 (b≥prev(earliest_date)=首边界)。首个边界可能早于earliest_date —
    那些更早的因子日期也归入首边界, 无空洞。
    """
    first = _quarter_boundary_prev(earliest_date)
    last = _quarter_boundary_prev(todate)
    bounds = []
    b = first
    while b <= last:
        bounds.append(b)
        # 季度末+3月=下一季度末; DateOffset(months=3)保留日号(9/30→12/30),
        # 故追加MonthEnd(0)锚回月末 (12/30→12/31, 已是月末则不变)
        b = pd.Timestamp(b) + pd.DateOffset(months=3) + pd.offsets.MonthEnd(0)
    return bounds


def _daily_boundaries(earliest_date, todate):
    """每日边界列表(升序): [earliest_date .. todate]的全部日历日.

    daily模式(0f-v2, 2026-09-18): 实盘同构 — 每晚用当日可得数据重算池,
    池成员资格as-of边界日(含当日K线), 入池零滞后。周末/节假日无K线行,
    cut=searchsorted仍落在最近交易日, 与get_stock_pool(todate=b)语义一致。
    """
    return list(pd.date_range(pd.Timestamp(earliest_date), pd.Timestamp(todate), freq='D'))


def get_pool_membership_map(boundaries, data_dir=None, min_price: float = 2.0,
                            bse_exclude: bool = True, cache_key: str = None) -> dict:
    """季度日历池成员映射: {boundary_timestamp: set(codes)} — as-of每个边界的池成员.

    0f池口径修复(2026-09-17): 与get_stock_pool同准则(60bar最小长度+价格阈值+
    近20日成交额流动性+北交所排除), 但按季度边界求as-of成员资格 — 单遍扫描
    (每文件读datetime/close/amount/volume一次, 逐边界向量化评估), 避免
    len(boundaries)×全文件扫读。缓存: strategy/cache/pool_membership_{cache_key}.parquet
    (cache_key=数据指纹+参数+本文件hash, 调用方计算)。

    Args:
        boundaries: 升序的季度末Timestamp列表
        cache_key: 缓存键后缀 (None=不缓存)

    Returns:
        {boundary_str: set(codes)}, 指数(sh000001/sh000852/000001/399006)恒成员
    """
    if data_dir is None:
        data_dir = _get_data_dir()
    if not os.path.exists(data_dir) or not boundaries:
        return {}

    _IDX = {'sh000001', 'sh000852', '000001', '399006'}
    boundaries = sorted(pd.Timestamp(b) for b in boundaries)

    # ── 缓存 ──
    cache_path = None
    if cache_key:
        cache_dir = str(Path(__file__).parent.parent / 'cache')
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f'pool_membership_{cache_key}.parquet')
        if os.path.exists(cache_path):
            try:
                df = pd.read_parquet(cache_path, columns=['boundary', 'code'])
                df['boundary'] = pd.to_datetime(df['boundary'])
                m = {b: set() for b in boundaries}
                for b, codes in df.groupby('boundary')['code']:
                    if b in m:
                        m[b] = set(codes)
                for b in boundaries:
                    m[b] |= _IDX
                print(f"股票池成员映射: 缓存命中 ({len(boundaries)} 边界, {len(df)} 条)")
                return m
            except Exception as e:
                print(f"[WARN] 股票池成员映射缓存读取失败, 重算: {e}")

    # ── 单遍扫描 ──
    exclusion_set = get_exclusion_set()
    rows = []
    valid_files = []
    for f in os.listdir(data_dir):
        if f.startswith('._') or f.startswith(('sh000001', 'sh000852')):
            continue
        if f.endswith('_qfq.csv'):
            code = f[:-8]
        elif f.endswith('_hfq.csv'):
            code = f[:-8]
            if os.path.exists(os.path.join(data_dir, f'{code}_qfq.csv')):
                continue
        else:
            continue
        if bse_exclude and is_bse_code(code):
            continue
        valid_files.append((f, code))

    for f, code in valid_files:
        if code in exclusion_set:
            continue
        filepath = os.path.join(data_dir, f)
        try:
            df = pd.read_csv(filepath, usecols=['datetime', 'close', 'amount', 'volume'])
            has_amount = True
        except ValueError:
            # 无amount列 → volume回退 (与get_stock_pool的elif分支一致)
            try:
                df = pd.read_csv(filepath, usecols=['datetime', 'close', 'volume'])
                has_amount = False
            except Exception:
                continue
        except Exception:
            continue
        dt = pd.to_datetime(df['datetime'])
        if code.startswith('688'):
            min_amount = 40_000_000
            min_vol = 300_000
        elif code.startswith('300'):
            min_amount = 80_000_000
            min_vol = 1_000_000
        else:
            min_amount = 160_000_000
            min_vol = 1_000_000
        # 逐边界as-of评估: 截断长度≥60 / 最新收盘价≥min_price / 近20日流动性
        for b in boundaries:
            # searchsorted定位截断点 (dt升序)
            cut = int(np.searchsorted(dt.values, np.datetime64(b), side='right'))
            if cut < 60:
                continue
            last_close = df['close'].iloc[cut - 1]
            if pd.isna(last_close) or last_close < min_price:
                continue
            if has_amount:
                seg = df['amount'].iloc[max(0, cut - 20):cut]
                if seg.mean() < min_amount:
                    continue
            else:
                seg_vol = df['volume'].iloc[max(0, cut - 20):cut]
                if seg_vol.mean() < min_vol:
                    continue
            rows.append((b, code))
        del df
    m = {b: set(_IDX) for b in boundaries}
    for b, code in rows:
        m[b].add(code)

    if cache_path:
        try:
            pd.DataFrame([(b, c) for b in boundaries for c in m[b]],
                         columns=['boundary', 'code']).to_parquet(cache_path, index=False)
        except Exception as e:
            print(f"[WARN] 股票池成员映射缓存写入失败: {e}")

    counts = {str(b.date()): len(m[b]) for b in boundaries}
    print(f"股票池成员映射: {len(valid_files)} 文件单遍扫描, {len(rows)} 条成员记录 "
          f"({len(boundaries)} 边界: {counts})")
    return m


# 北交所代码段 (2026-09-03 用户指令: 北交所股票全部不要).
# 与沪深主/创/科(000/001/002/003/300/301/302/600/601/603/605/688/689)无交集, 无碰撞风险.
_BSE_PREFIXES = ('43', '82', '83', '87', '88', '92')


def is_bse_code(code: str) -> bool:
    """北交所股票判定 (新段920xxx / 旧段82/83/87/88xxxx / 新三板43xxxx)"""
    return code.startswith(_BSE_PREFIXES)


def _get_metadata_dir():
    """获取stock_metadata目录"""
    base = Path(__file__).parent.parent.parent
    return str(base / 'data' / 'stock_data' / 'stock_metadata')


def _load_market_cap_whitelist():
    """加载市值白名单（>=500亿总市值），用于加速回测"""
    import os as _os
    base = Path(__file__).parent.parent
    whitelist_path = str(base / 'config' / 'large_cap_whitelist.txt')
    if not _os.path.exists(whitelist_path):
        return None  # 无白名单文件 → 不过滤
    with open(whitelist_path, 'r') as f:
        codes = {line.strip() for line in f if line.strip()}
    return codes if codes else None


def get_stock_pool(min_price: float = 2.0,
                   data_dir: str = None,
                   todate: str = None,
                   bse_exclude: bool = True) -> set:
    """获取股票池 — 全市场除科创板外全部纳入

    Args:
        min_price: 最低价格（排除仙股，复权后价格）
        data_dir: 数据目录路径
        todate: 截止日期(YYYY-MM-DD), 流动性只看此日之前数据
        bse_exclude: 排除北交所 (2026-09-04 实验开关; 实盘路径默认True=用户指令恒排除)

    Returns:
        set of stock codes
    """
    if data_dir is None:
        data_dir = _get_data_dir()

    if not os.path.exists(data_dir):
        return set()

    exclusion_set = get_exclusion_set()

    valid_files = []
    bse_skipped = 0
    for f in os.listdir(data_dir):
        # 指数文件不参与候选筛选 (Fix#43: sh000852中证1000同sh000001)
        if f.startswith('._') or f.startswith(('sh000001', 'sh000852')):
            continue
        if f.endswith('_qfq.csv'):
            code = f[:-8]
        elif f.endswith('_hfq.csv'):
            code = f[:-8]
            # 避免 qfq/hfq 重复
            qfq_path = os.path.join(data_dir, f'{code}_qfq.csv')
            if os.path.exists(qfq_path):
                continue
        else:
            continue
        if bse_exclude and is_bse_code(code):
            # 2026-09-03: 北交所股票全部排除 (用户指令); bse_exclude=False 仅供回测归因实验
            bse_skipped += 1
            continue
        valid_files.append((f, code))
    if bse_skipped:
        print(f"股票池: 排除北交所 {bse_skipped} 只 (用户指令, 2026-09-03)")

    selected = set()
    data_errors = 0
    liquidity_filtered = 0

    for item, code in valid_files:
        if code in exclusion_set:
            continue

        filepath = os.path.join(data_dir, item)
        try:
            df = pd.read_csv(filepath)
            # 截断到todate, 防止未来数据污染流动性判断
            if todate is not None and 'datetime' in df.columns:
                df['_dt'] = pd.to_datetime(df['datetime'])
                df = df[df['_dt'] <= pd.Timestamp(todate)]
            if len(df) < 60:
                data_errors += 1
                continue

            last_price = df['close'].iloc[-1]
            if last_price <= 0 or np.isnan(last_price) or last_price < min_price:
                continue

            # 流动性过滤: 近20日日均成交额
            # 主板: 1.6亿 | 创业板(300): 8000万 | 科创板(688): 4000万
            if code.startswith('688'):
                min_amount = 40_000_000
            elif code.startswith('300'):
                min_amount = 80_000_000
            else:
                min_amount = 160_000_000
            min_vol = 300_000 if code.startswith('688') else 1_000_000
            if 'amount' in df.columns and len(df) >= 20:
                avg_amount = df['amount'].iloc[-20:].mean()
                if avg_amount < min_amount:
                    liquidity_filtered += 1
                    continue
            # 流动性补充: 无amount列时, 近20日日均成交量
            elif 'volume' in df.columns and len(df) >= 20:
                avg_vol = df['volume'].iloc[-20:].mean()
                if avg_vol < min_vol:
                    liquidity_filtered += 1
                    continue

            selected.add(code)
        except Exception:
            data_errors += 1
            continue

    selected.add('sh000001')
    selected.add('sh000852')  # Fix#43: 中证1000(小盘风格输入), 回测端stock_codes已排除
    print(f"股票池: {len(valid_files)} 总文件 -> 科创板{len(exclusion_set)} | 异常{data_errors} | 流动性{liquidity_filtered} -> {len(selected)} 只 (含sh000001/sh000852)")
    return selected


def load_st_stocks() -> set:
    """从stock_list_full.csv加载ST股票代码（含*ST和ST前缀的股票）

    Returns:
        set of ST stock codes (不含前缀, 如 '000001')
    """
    metadata_dir = _get_metadata_dir()
    # stock_list_full.csv 包含ST标记, stock_list.csv 不包含
    stock_list_path = os.path.join(metadata_dir, 'stock_list_full.csv')

    if not os.path.exists(stock_list_path):
        # 回退到 stock_list.csv
        stock_list_path = os.path.join(metadata_dir, 'stock_list.csv')
        if not os.path.exists(stock_list_path):
            print("警告: 未找到 stock_list，无法过滤ST股票")
            return set()

    df = pd.read_csv(stock_list_path, dtype={'symbol': str})
    st_codes = set()
    for _, row in df.iterrows():
        name = str(row.get('name', ''))
        symbol = str(row.get('symbol', ''))
        # ST命名: '*ST香雪', 'ST逸飞' 等
        if ('ST' in name or '*ST' in name) and symbol:
            st_codes.add(symbol)

    # 同时过滤退市股票（名称含"退市"）
    for _, row in df.iterrows():
        name = str(row.get('name', ''))
        symbol = str(row.get('symbol', ''))
        if '退市' in name and symbol:
            st_codes.add(symbol)

    print(f"ST股票过滤: 识别 {len(st_codes)} 只ST/退市股票")
    return st_codes


def is_star_board(code: str) -> bool:
    """判断是否为科创板股票 (688xxx)"""
    return code.startswith('688')


def get_exclusion_set() -> set:
    """获取需要排除的股票集合.

    ST不再静态排除 — ST状态随时间变化(每年有新增/摘帽)，静态快照会错误排除非ST期的股票。
    改为在回测中由 fundamental_data.is_st() 逐日判断（bt_execution 向量化路径已实现）。

    Returns:
        set of stock codes to exclude
    """
    # 科创板(688xxx) 不再排除 — 2025-2026年妖股主要集中在科创板
    star_codes = set()

    excluded = star_codes
    print(f"股票排除: 科创板 {len(star_codes)} = 共排除 {len(excluded)} 只 (ST改为逐日判断)")
    return excluded
