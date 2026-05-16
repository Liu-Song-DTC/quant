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


def _get_metadata_dir():
    """获取stock_metadata目录"""
    base = Path(__file__).parent.parent.parent
    return str(base / 'data' / 'stock_data' / 'stock_metadata')


def get_stock_pool(min_price: float = 3.0,
                   max_price: float = 1000.0,
                   min_turnover_rate: float = 0.003,
                   turnover_pctile: float = 15.0,
                   min_listed_days: int = 250,
                   data_dir: str = None) -> set:
    """获取质量筛选后的股票池

    筛选流程:
    1. 基础过滤: 价格区间 + 排除负价/异常数据
    2. 次新股过滤: 上市时间门槛
    3. 僵尸股过滤: 换手率门槛 + 流动性地板(相对, bottom N%)
    4. 排雷: ST + 科创板 + 连续跌停 + 长期停牌
    5. 复合排名: 流动性(60%) + 价格稳定(25%) + 交易活跃(15%)

    所有通过质量筛选的股票全部纳入股票池（不再限制top-N数量）。

    Args:
        min_price: 最低价格（排除仙股，复权后价格）
        max_price: 最高价格
        min_turnover_rate: 最低日均换手率（排除僵尸股）
        turnover_pctile: 流动性地板百分位 (bottom N% 被排除)
        min_listed_days: 最少上市天数（排除次新股）
        data_dir: 数据目录路径

    Returns:
        set of stock codes
    """
    if data_dir is None:
        data_dir = _get_data_dir()

    if not os.path.exists(data_dir):
        return set()

    exclusion_set = get_exclusion_set()

    # 收集所有有效文件，优先_qfq，跳过._隐藏文件
    all_files_raw = os.listdir(data_dir)
    qfq_codes = set()
    hfq_files = []
    valid_files = []

    for f in all_files_raw:
        if f.startswith('._') or f.startswith('sh000001'):
            continue
        if f.endswith('_qfq.csv'):
            code = f[:-8]
            qfq_codes.add(code)
            valid_files.append((f, 'qfq'))
        elif f.endswith('_hfq.csv'):
            hfq_files.append(f)

    for f in hfq_files:
        code = f[:-8]
        if code not in qfq_codes:
            valid_files.append((f, 'hfq'))

    total_files = len(valid_files)
    filter_stats = {
        'total': total_files,
        'excluded_st_star': 0,
        'price_filtered': 0,
        'zombie_filtered': 0,
        'new_stock_filtered': 0,
        'turnover_floor': 0,
        'data_error': 0,
        'limit_down': 0,
        'long_suspended': 0,
    }

    raw_candidates = []  # 第一轮: 基本过滤后

    for item, ftype in valid_files:
        code = item[:-8]

        # === 排雷: ST + 科创板 ===
        if code in exclusion_set:
            filter_stats['excluded_st_star'] += 1
            continue

        filepath = os.path.join(data_dir, item)

        try:
            try:
                df = pd.read_csv(filepath, encoding='utf-8')
            except UnicodeDecodeError:
                try:
                    df = pd.read_csv(filepath, encoding='gbk')
                except Exception:
                    filter_stats['data_error'] += 1
                    continue

            if len(df) < 60:
                filter_stats['data_error'] += 1
                continue

            # === 次新股过滤 ===
            if 'datetime' in df.columns:
                df['datetime'] = pd.to_datetime(df['datetime'])
                first_date = df['datetime'].iloc[0]
                last_date = df['datetime'].iloc[-1]
                if (last_date - first_date).days < min_listed_days:
                    filter_stats['new_stock_filtered'] += 1
                    continue
            elif len(df) < min_listed_days:
                filter_stats['new_stock_filtered'] += 1
                continue

            # 用最后60个交易日做质量评估
            recent = df.tail(60)

            # === 价格过滤 ===
            last_price = recent['close'].iloc[-1]
            if last_price <= 0 or np.isnan(last_price):
                filter_stats['price_filtered'] += 1
                continue
            if last_price < min_price or last_price > max_price:
                filter_stats['price_filtered'] += 1
                continue

            # === 计算量价指标 ===
            if 'volume' not in recent.columns or 'close' not in recent.columns:
                filter_stats['data_error'] += 1
                continue

            daily_amount = recent['close'] * recent['volume']
            avg_amount = daily_amount.mean()

            avg_turnover = 0.0
            if 'turnover_rate' in recent.columns:
                avg_turnover = recent['turnover_rate'].mean()
                if avg_turnover <= 0:
                    filter_stats['zombie_filtered'] += 1
                    continue
                if avg_turnover < min_turnover_rate:
                    filter_stats['zombie_filtered'] += 1
                    continue

            avg_amplitude = 0.0
            if 'amplitude' in recent.columns:
                avg_amplitude = recent['amplitude'].mean()

            # === 异常数据过滤 ===
            # 连续跌停
            if 'change_percent' in recent.columns:
                recent_pct = recent['change_percent'].tail(10)
                cons_limit = 0
                for pct in recent_pct:
                    if pct < -9.0:
                        cons_limit += 1
                    else:
                        cons_limit = 0
                    if cons_limit >= 3:
                        break
                if cons_limit >= 3:
                    filter_stats['limit_down'] += 1
                    continue

            # 长期停牌
            if 'volume' in recent.columns:
                if (recent['volume'] <= 0).sum() > 30:
                    filter_stats['long_suspended'] += 1
                    continue

            raw_candidates.append({
                'code': code,
                'avg_amount': avg_amount,
                'avg_turnover': avg_turnover,
                'avg_amplitude': avg_amplitude,
            })

        except Exception:
            filter_stats['data_error'] += 1
            continue

    # === 第二层: 相对流动性地板 (bottom N% 排除) ===
    if raw_candidates and turnover_pctile > 0:
        amounts = [c['avg_amount'] for c in raw_candidates]
        floor = np.percentile(amounts, turnover_pctile)
        before = len(raw_candidates)
        raw_candidates = [c for c in raw_candidates if c['avg_amount'] >= floor]
        filter_stats['turnover_floor'] = before - len(raw_candidates)

    # === 复合评分排名 ===
    if raw_candidates:
        amounts_arr = np.array([c['avg_amount'] for c in raw_candidates])
        log_amounts = np.log10(amounts_arr + 1)
        # 流动性得分: 对数归一化
        la_min, la_max = log_amounts.min(), log_amounts.max()
        if la_max > la_min:
            liq_scores = (log_amounts - la_min) / (la_max - la_min)
        else:
            liq_scores = np.ones(len(raw_candidates)) * 0.5

        # 稳定性得分: 振幅
        amp_arr = np.array([c['avg_amplitude'] for c in raw_candidates])
        stab_scores = np.clip(1.0 - (amp_arr - 3.0) / 7.0, 0, 1)

        # 活跃度得分: 换手率
        to_arr = np.array([c['avg_turnover'] for c in raw_candidates])
        act_scores = np.zeros(len(raw_candidates))
        act_scores[(to_arr >= 0.005) & (to_arr <= 0.03)] = 1.0
        act_scores[(to_arr > 0.03) & (to_arr <= 0.08)] = 0.7
        act_scores[(to_arr > 0.08) & (to_arr <= 0.15)] = 0.4
        act_scores[(to_arr > 0.15) & (to_arr <= 0.25)] = 0.2
        act_scores[to_arr > 0.25] = 0.1
        act_scores[to_arr < 0.005] = 0.15

        for i, c in enumerate(raw_candidates):
            c['score'] = liq_scores[i] * 0.60 + stab_scores[i] * 0.25 + act_scores[i] * 0.15

        raw_candidates.sort(key=lambda x: -x['score'])

    selected = {c['code'] for c in raw_candidates}
    selected.add('sh000001')

    # 打印统计
    passed = len(raw_candidates)
    print(f"股票池筛选: {total_files} 总 -> ", end='')
    print(f"排雷{filter_stats['excluded_st_star']} | ", end='')
    print(f"次新{filter_stats['new_stock_filtered']} | ", end='')
    print(f"价格{filter_stats['price_filtered']} | ", end='')
    print(f"僵尸{filter_stats['zombie_filtered']} | ", end='')
    print(f"低流动{filter_stats['turnover_floor']} | ", end='')
    print(f"跌停{filter_stats['limit_down']} | ", end='')
    print(f"停牌{filter_stats['long_suspended']} | ", end='')
    print(f"异常{filter_stats['data_error']}")
    print(f"  -> {passed} 通过质量筛选, 全部纳入股票池 (流动性60%+稳定25%+活跃15%)")

    return selected


def load_st_stocks() -> set:
    """从stock_list.csv加载ST股票代码

    Returns:
        set of ST stock codes (不含前缀, 如 '000001')
    """
    metadata_dir = _get_metadata_dir()
    stock_list_path = os.path.join(metadata_dir, 'stock_list.csv')

    if not os.path.exists(stock_list_path):
        print("警告: stock_list.csv 不存在，无法过滤ST股票")
        return set()

    df = pd.read_csv(stock_list_path, dtype={'symbol': str})
    st_codes = set()
    for _, row in df.iterrows():
        name = str(row.get('name', ''))
        symbol = str(row.get('symbol', ''))
        if 'ST' in name and symbol:
            st_codes.add(symbol)

    print(f"ST股票过滤: 识别 {len(st_codes)} 只ST股票")
    return st_codes


def is_star_board(code: str) -> bool:
    """判断是否为科创板股票 (688xxx)"""
    return code.startswith('688')


def get_exclusion_set() -> set:
    """获取需要排除的股票集合: ST + 科创板

    Returns:
        set of stock codes to exclude
    """
    st_codes = load_st_stocks()

    # 科创板: 688xxx
    star_codes = set()
    data_dir = _get_data_dir()
    if os.path.exists(data_dir):
        for item in os.listdir(data_dir):
            if not (item.endswith('_qfq.csv') or item.endswith('_hfq.csv')):
                continue
            code = item[:-8] if item.endswith('_qfq.csv') else item[:-8]
            if is_star_board(code):
                star_codes.add(code)

    excluded = st_codes | star_codes
    overlap = len(st_codes & star_codes)
    print(f"股票排除: ST {len(st_codes)} + 科创板 {len(star_codes)} - 重叠{overlap} = 共排除 {len(excluded)} 只")
    return excluded
