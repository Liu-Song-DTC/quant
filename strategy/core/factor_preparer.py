# core/factor_preparer.py
"""
因子数据预计算模块

用于动态因子选择前的数据准备

注意：当 factor_mode='fixed' 时，此模块不参与计算，可跳过以加速回测
"""

import numpy as np
import pandas as pd
import os
from typing import Dict, List, Tuple
import multiprocessing
from tqdm import tqdm
import ctypes

# 全局题材热度计算器 (fork 前初始化, worker 进程继承)
_worker_concept_calc = None

from .config_loader import load_config
from .fundamental import FundamentalData
from .factor_calculator import calculate_indicators, compute_composite_factors, get_default_params, compress_fundamental_factor, board_limit_up_pct
from scipy.special import erfinv


# 全局变量用于 worker 进程 - 每个 worker 创建一个 FundamentalData 实例供所有股票复用
_worker_fd = None


def _init_factor_worker(fd, concept_calc=None):
    """Worker 进程初始化函数 - 共享父进程已加载的 FundamentalData + ConceptHeatCalculator

    使用 fork 后 COW 共享父进程的 FundamentalData，避免每个 worker 从磁盘重新加载
    4000+ 只股票的基本面数据（~8 workers × 4000 CSVs = 32000+ 次文件读取）。
    """
    global _worker_fd, _worker_concept_calc
    _worker_fd = fd
    if concept_calc is not None:
        _worker_concept_calc = concept_calc


def _preload_fundamental_cache(fd, code, dates):
    """预加载股票基本面数据缓存 — 指针推进法，每季度只计算一次而非每个日期查询一次"""
    cache = {}
    if fd is None or code not in fd.stock_data:
        return cache, None

    df = fd.stock_data[code]
    if '数据可用日期' not in df.columns or len(df) == 0:
        return cache, None

    df = df.copy()
    df['数据可用日期_str'] = df['数据可用日期'].astype(str)
    df = df.sort_values('数据可用日期_str').reset_index(drop=True)
    # 逐日期 iloc 会反复构造含上百列的 Series；一次转换即可保持同样的 as-of 语义。
    reports = df.to_dict('records')
    availability = df['数据可用日期_str'].tolist()

    sorted_dates = sorted(dates)
    report_idx = -1
    n_reports = len(df)
    stock_industry = None

    for d in sorted_dates:
        d_ts = pd.Timestamp(d)
        d_str = d_ts.strftime('%Y%m%d')

        while report_idx + 1 < n_reports and availability[report_idx + 1] <= d_str:
            report_idx += 1

        if report_idx >= 0:
            row = reports[report_idx]
            cache_key = d
            if cache_key not in cache:
                # 提取基本面值
                roe = _parse_pct(row.get('净资产收益率'))
                profit_growth = _parse_pct(row.get('净利润-同比增长'))
                revenue_growth = _parse_pct(row.get('营业总收入-同比增长'))
                gross_margin = _parse_pct(row.get('销售毛利率'))

                # 综合评分
                fund_score = 0.0
                if roe is not None:
                    fund_score += min(roe * 100, 30)
                if profit_growth is not None:
                    if profit_growth > 0.5: fund_score += 25
                    elif profit_growth > 0.2: fund_score += 15
                    elif profit_growth > 0: fund_score += 5
                eps = row.get('每股收益')
                if eps is not None:
                    try:
                        eps = float(eps)
                        if eps > 0: fund_score += min(eps * 10, 20)
                    except (ValueError, TypeError): pass
                if revenue_growth is not None:
                    if revenue_growth > 0.3: fund_score += 15
                    elif revenue_growth > 0.1: fund_score += 10

                # cf_to_profit
                cf_to_profit = None
                operating_cf = row.get('xjll_经营性现金流-现金流量净额')
                profit = row.get('lrb_净利润')
                if operating_cf is not None and profit is not None:
                    try:
                        if float(profit) > 0:
                            cf_to_profit = float(operating_cf) / float(profit)
                    except (ValueError, TypeError): pass

                cache[cache_key] = {
                    'roe': roe,
                    'profit_growth': profit_growth,
                    'revenue_growth': revenue_growth,
                    'fund_score': fund_score,
                    'gross_margin': gross_margin,
                    'cf_to_profit': cf_to_profit,
                    'industry': row.get('所处行业'),
                }

        if stock_industry is None and report_idx >= 0:
            stock_industry = reports[report_idx].get('所处行业')

    return cache, stock_industry


def _parse_pct(val):
    """解析百分比值"""
    if val is None:
        return None
    try:
        if isinstance(val, str):
            return float(val.strip('%')) / 100.0
        return float(val)
    except (ValueError, TypeError):
        return None


def _compute_stock_factors_worker(args):
    """多进程 worker: 计算单只股票的因子数据（从文件读取，避免主进程加载全量数据）"""
    global _worker_fd
    code, filepath, factor_dates, lookback, forward_period = args

    # 从文件读取数据（每个worker只持有一只股票的数据，内存可控）
    try:
        df = pd.read_csv(filepath, parse_dates=['datetime'])
        df = df.sort_values('datetime', kind='stable').drop_duplicates('datetime', keep='last')
    except Exception:
        return []

    # 使用 datetime 列而非 index
    if 'datetime' in df.columns:
        stock_dates = sorted(df['datetime'].tolist())
    else:
        stock_dates = sorted(df.index.tolist())

    n = len(stock_dates)
    if n < lookback + forward_period:
        return []

    # 一次性提取价格数据（float64 → 计算精度优于 float32）
    close_arr = df['close'].values.astype(np.float64, copy=False)
    high_arr = df['high'].values.astype(np.float64, copy=False) if 'high' in df.columns else close_arr
    low_arr = df['low'].values.astype(np.float64, copy=False) if 'low' in df.columns else close_arr
    vol_arr = df['volume'].values.astype(np.float64, copy=False) if 'volume' in df.columns else np.ones(n)
    open_arr = df['open'].values.astype(np.float64, copy=False) if 'open' in df.columns else close_arr
    turnover_arr = df['turnover_rate'].values.astype(np.float64, copy=False) if 'turnover_rate' in df.columns else None
    amount_arr = df['amount'].values.astype(np.float64, copy=False) if 'amount' in df.columns else None
    del df  # 释放DataFrame，仅保留numpy数组

    # === 使用统一的因子计算器计算所有基础指标 ===
    params = get_default_params()
    ind = calculate_indicators(close_arr, high_arr, low_arr, vol_arr, params,
                               open_arr=open_arr, turnover_rate=turnover_arr,
                               limit_up_pct=board_limit_up_pct(code))

    # === 构建日期到索引的映射（O(1) 查找）===
    date_to_idx = {d: i for i, d in enumerate(stock_dates)}

    # === 预加载基本面缓存（指针推进法，避免每日期查询DataFrame）===
    fund_cache, stock_industry = _preload_fundamental_cache(_worker_fd, code, factor_dates)

    # === 批量向量化构建结果 ===
    results = []
    for sample_date in factor_dates:
        # O(1) 查找索引
        idx = date_to_idx.get(sample_date)
        if idx is None or idx < lookback:
            continue

        # 使用统一的因子计算器计算组合因子
        fund_data = fund_cache.get(sample_date, {})

        # 先获取压缩后的基本面评分，传给 compute_composite_factors
        compressed_fund_score = 0.0
        if fund_data:
            raw_fund_score = fund_data.get('fund_score', 0) or 0
            if isinstance(raw_fund_score, (int, float)):
                compressed_fund_score = compress_fundamental_factor(raw_fund_score, 'fund_score')

        # 使用 factor_calculator 计算所有组合因子（含 tech_fund_combo）
        combo_factors = compute_composite_factors(ind, idx, fund_score=compressed_fund_score)

        # 行键模板: 条件添加的键(fund_*/concept_heat/ln_cap)先占位None,
        # 保证每行键集完全一致 -> 任何批次切分下列序恒等, 不会丢列/错位
        row = {'code': code, 'date': sample_date, 'industry': stock_industry or '',
               'label_end_date': pd.NaT,
               'ln_cap': None, 'concept_heat': None,
               'fund_roe': None, 'fund_profit_growth': None, 'fund_revenue_growth': None,
               'fund_score': None, 'fund_gross_margin': None, 'fund_cf_to_profit': None,
               'future_ret': None}
        row.update(combo_factors)

        # 市值因子 (size): A股小盘效应, 流通市值≈成交额/换手率
        if idx < n:
            amt = float(amount_arr[idx]) if amount_arr is not None else 0
            to_rate = float(turnover_arr[idx]) if turnover_arr is not None else 0
            if amt > 0 and to_rate > 0.001:
                est_circ_mv = amt / (to_rate / 100.0)
                row['ln_cap'] = float(np.tanh(np.log(max(est_circ_mv, 1e6)) / 10.0 - 0.5))
            else:
                row['ln_cap'] = 0.0

        # 题材热度因子 - 从真实概念数据计算
        global _worker_concept_calc
        if _worker_concept_calc is not None:
            try:
                _worker_concept_calc.set_daily_data(sample_date)
                row['concept_heat'] = _worker_concept_calc.get_concept_heat(code)
            except Exception:
                row['concept_heat'] = 0.5
        elif 'concept_heat' in ind and idx < len(ind['concept_heat']):
            row['concept_heat'] = float(ind['concept_heat'][idx])

        # 基本面因子 - 使用统一压缩函数（与signal_engine一致）
        if fund_data:
            raw_roe = fund_data.get('roe')
            raw_profit_growth = fund_data.get('profit_growth')
            raw_revenue_growth = fund_data.get('revenue_growth')
            raw_fund_score = fund_data.get('fund_score')
            raw_gross_margin = fund_data.get('gross_margin')
            raw_cf_to_profit = fund_data.get('cf_to_profit')

            if raw_roe is not None and isinstance(raw_roe, (int, float)):
                row['fund_roe'] = compress_fundamental_factor(raw_roe, 'fund_roe')
            if raw_profit_growth is not None and isinstance(raw_profit_growth, (int, float)):
                row['fund_profit_growth'] = compress_fundamental_factor(raw_profit_growth, 'fund_profit_growth')
            if raw_revenue_growth is not None and isinstance(raw_revenue_growth, (int, float)):
                row['fund_revenue_growth'] = compress_fundamental_factor(raw_revenue_growth, 'fund_revenue_growth')
            if raw_fund_score is not None and isinstance(raw_fund_score, (int, float)):
                row['fund_score'] = compress_fundamental_factor(raw_fund_score, 'fund_score')
            if raw_gross_margin is not None and isinstance(raw_gross_margin, (int, float)):
                row['fund_gross_margin'] = compress_fundamental_factor(raw_gross_margin, 'fund_gross_margin')
            if raw_cf_to_profit is not None and isinstance(raw_cf_to_profit, (int, float)):
                row['fund_cf_to_profit'] = compress_fundamental_factor(raw_cf_to_profit, 'fund_cf_to_profit')

        # 计算未来收益: 尾部forward_period日无未来价格, future_ret=NaN但行保留
        # (ML预测/实盘信号需要最新交易日的截面因子值)
        row['future_ret'] = np.nan
        if idx + forward_period < n:
            row['label_end_date'] = pd.Timestamp(stock_dates[idx + forward_period])
            future_price = close_arr[idx + forward_period]
            current_price = close_arr[idx]
            if current_price > 0:
                row['future_ret'] = (future_price - current_price) / current_price
        results.append(row)

    return results


def _data_fingerprint(stock_file_map: dict, fd, extra_paths: list) -> str:
    """数据内容指纹 (mtime_ns+size): 因子计算消费的全部输入文件。

    缓存键缺陷修复 (344复现陷阱): 旧键只含池代码/日期/参数, 不含数据内容 ->
    K线/基本面/题材数据刷新后键不变, 旧缓存被静默复用 (8/31新基本面缓存与
    8/28原版缓存键相同, 谁先建好谁占据该键)。加入指纹后, 任何输入文件变动
    都会使缓存失效并触发重算 — 失效方向安全 (只多算一次, 不会错用旧值)。
    """
    import hashlib
    h = hashlib.md5()

    def _feed(path):
        try:
            st = os.stat(path)
            h.update(f"{path}|{st.st_mtime_ns}|{st.st_size};".encode('utf-8'))
        except OSError:
            h.update(f"{path}|MISSING;".encode('utf-8'))

    for p in sorted(set(stock_file_map.values())):
        _feed(p)
    if fd is not None and getattr(fd, 'data_path', None):
        for root, _, files in os.walk(fd.data_path):
            for f in sorted(files):
                _feed(os.path.join(root, f))
    for p in sorted(set(extra_paths)):
        if os.path.isdir(p):
            for root, _, files in os.walk(p):
                for f in sorted(files):
                    _feed(os.path.join(root, f))
        else:
            _feed(p)
    return h.hexdigest()[:8]


# 因子值计算所消费的代码/配置 (改动任一 → 因子缓存失效重算, 方向安全)
_FACTOR_CODE_FILES = [
    'core/factor_preparer.py',
    'core/factor_calculator.py',
    'core/factor_neutralizer.py',
    'core/multi_timeframe.py',
    'core/concept_heat.py',
    'core/cmap_hist.py',  # 2026-10-17: 概念PIT钩子解析器 — 改它=改信号输入来源
    'core/industry_chain.py',
    'core/alternative_data.py',
    'core/fundamental.py',
    'core/market_regime_detector.py',
    'core/bom_chain.py',
    'core/gate_scorer.py',
    'core/chan_theory.py',  # 2026-10-11: B9涨停阈值等chan逻辑影响信号, 此前指纹缺口(仅改chan不触发重生成)
    'core/stock_pool.py',  # 0f: 日历池membership逻辑影响因子掩码/闸内容 (2026-09-17)
    'config/factor_config.yaml',
]

# 0f-v3 (2026-09-18): 因子缓存只存raw(掩码/中性化/rank前) — 池成员资格不影响
# raw计算(掩码按每次运行的membership_map应用), 故raw因子缓存键豁免stock_pool.py
# 与yaml的stock_pool节: granularity/floor-relax臂切换只换掩码不重算raw(~3h→秒级)。
# 信号代码指纹(_SIGNAL_CODE_FILES)不含豁免 — 池模式变化必须重生成信号。
_RAW_FACTOR_CODE_FILES = [f for f in _FACTOR_CODE_FILES
                          if f not in ('core/stock_pool.py',
                                       # 中性化在raw缓存之后跑, 不参与raw因子计算
                                       'core/factor_neutralizer.py',
                                       # MTF只被signal_engine import, 不参与raw因子计算
                                       'core/multi_timeframe.py')]


# 组合/执行层yaml节: 不参与因子计算也不参与信号生成, 因子缓存键与信号指纹均豁免
# (2026-09-06逐节验证: 9个因子文件+signal_engine+ml_predictor+bt_execution的信号
#  路径均不读这些节; portfolio.py/回测引擎消费它们)。
# backtest节必须保留(todate流入get_stock_pool → 股票池 → 因子/信号内容)。
_PORTFOLIO_LAYER_YAML_SECTIONS = {
    'portfolio', 'base_exposure', 'portfolio_stop_loss', 'stock_stop_loss',
    'hard_drawdown_stop', 'consecutive_loss_breaker', 'volatility_control',
    'risk_parity', 'dynamic_rebalance', 'enhanced_stop_loss',
    'mean_reversion_exit', 'cost_model', 'live_monitoring', 'account_risk_limits',
}

# raw因子缓存键专用豁免节 (0f-v3): 组合/执行层 + stock_pool (池节不参与raw计算)
# + ml (标签/特征集不参与raw因子值) + factor_neutralization (中性化在缓存之后跑,
#   改它只需要重算中性化/排名, 不需要重算全市场因子 —— 原先会白白触发~2h全量重算)
_RAW_YAML_SKIP_SECTIONS = _PORTFOLIO_LAYER_YAML_SECTIONS | {'stock_pool', 'ml',
                                                           'factor_neutralization'}


def _yaml_stripped_digest(yaml_path: str) -> str:
    """factor_config.yaml剥掉组合/执行层节后的内容md5[:8]。

    因子缓存键与信号代码指纹共用: portfolio层实验(如H5)只改这些节时,
    因子缓存/信号CSV均不失效 — 13min迭代的关键。信号层节(signal/ml/
    dynamic_factor/backtest等)仍全量计入。
    """
    import hashlib
    import re
    h = hashlib.md5()
    try:
        with open(yaml_path, encoding='utf-8') as _yf:
            _skip = ''
            for _ln in _yf:
                _m = re.match(r'^([A-Za-z_][A-Za-z0-9_]*):', _ln)
                if _m:
                    _skip = _m.group(1) if _m.group(1) in _PORTFOLIO_LAYER_YAML_SECTIONS else ''
                if not _skip:
                    h.update(_ln.encode('utf-8'))
    except OSError:
        h.update(b'MISSING')
    return h.hexdigest()[:8]


def _yaml_raw_digest(yaml_path: str) -> str:
    """raw因子缓存键专用yaml摘要: 在_stripped基础上再豁免stock_pool节。

    0f-v3: raw缓存=掩码前数据, 池节(pool_calendar/relax参数)不参与raw计算,
    granularity/relax臂切换不应重算raw。信号指纹继续用_yaml_stripped_digest
    (stock_pool节计入) — 池模式变化必须重生成信号。
    """
    import hashlib
    import re
    h = hashlib.md5()
    try:
        with open(yaml_path, encoding='utf-8') as _yf:
            _skip = ''
            for _ln in _yf:
                _m = re.match(r'^([A-Za-z_][A-Za-z0-9_]*):', _ln)
                if _m:
                    _skip = _m.group(1) if _m.group(1) in _RAW_YAML_SKIP_SECTIONS else ''
                if not _skip:
                    h.update(_ln.encode('utf-8'))
    except OSError:
        h.update(b'MISSING')
    return h.hexdigest()[:8]


def _file_content_digest(path: str) -> str:
    """文件内容 md5[:16] (内容寻址, 不用 mtime)。

    mtime 口径的三个问题: ①touch 一个文件就 bust 缓存 (45min~2h 白烧);
    ②跨树不可比 (研究 worktree 里 mtime 全变 ⇒ 指纹不同 ⇒ 身份闸只能走 iso 副本);
    ③指纹不是代码内容的纯函数, 与"同代码态同结果"的可复现要求冲突。
    """
    import hashlib
    try:
        with open(path, 'rb') as stream:
            return hashlib.md5(stream.read()).hexdigest()[:16]
    except OSError:
        return 'MISSING'


def _code_fingerprint() -> str:
    """代码指纹: 因子计算逻辑/配置的**内容**摘要。

    数据指纹只覆盖输入文件; 代码改动(如 concept_heat EMA污染修复)后缓存键
    若不变, 旧因子值会被静默复用 — 与344陷阱同款。代码指纹杜绝此类复用。
    yaml按剥组合/执行层节后的内容哈希(_yaml_stripped_digest) — 组合层参数
    不参与因子计算, 不应使因子缓存失效。
    """
    import hashlib
    h = hashlib.md5()
    base = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    for rel in _FACTOR_CODE_FILES:
        p = os.path.join(base, rel)
        if rel == 'config/factor_config.yaml':
            h.update(f"{rel}|{_yaml_stripped_digest(p)};".encode('utf-8'))
            continue
        h.update(f"{rel}|{_file_content_digest(p)};".encode('utf-8'))
    return h.hexdigest()[:8]


def _raw_code_fingerprint() -> str:
    """raw因子缓存键专用代码指纹 (0f-v3): 豁免stock_pool.py与yaml池节。

    raw缓存=掩码前数据 — 池成员资格(quarterly/monthly/daily/relax)在掩码阶段
    按membership_map每次运行应用, 不参与raw计算。granularity/relax臂切换只换
    掩码与闸, raw缓存应命中(~3h省为秒级)。信号指纹不走此豁免。
    """
    import hashlib
    h = hashlib.md5()
    base = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    for rel in _RAW_FACTOR_CODE_FILES:
        p = os.path.join(base, rel)
        if rel == 'config/factor_config.yaml':
            h.update(f"{rel}|{_yaml_raw_digest(p)};".encode('utf-8'))
            continue
        h.update(f"{rel}|{_file_content_digest(p)};".encode('utf-8'))
    return h.hexdigest()[:8]


# 信号生成所消费的代码 (因子文件 + 信号引擎 + ML + 执行入口)。
# 2026-09-06 797,071锚点危机教训: bt_execution.py/signal_engine.py/ml_predictor.py
# 不在旧10文件指纹内 — 9/5夜间bt_execution的ML purge与提交版不一致却指纹相同,
# 导致"恢复run"797,071与当天任何可复现代码态都对不上(因子parquet逐列相同、
# 同数据同参数进程内训练确定性验证、唯一解释=执行层代码漂移)。
# 此指纹用于信号复用门禁: 代码变则强制重生成信号, 杜绝混搭代码态。
_SIGNAL_CODE_FILES = _FACTOR_CODE_FILES + [
    'core/signal_engine.py',
    'core/ml_predictor.py',
    'core/forward_targets.py',
    'bt_execution.py',
]


def _signal_code_fingerprint() -> str:
    """信号代码指纹: 信号生成路径 (因子+信号引擎+ML+执行入口) 的 (mtime_ns+size)。

    与因子缓存键用的 _code_fingerprint 不同: 覆盖执行层三文件 (不在因子缓存键内,
    改它们不应使因子缓存失效, 但必须使信号CSV失效 — 信号复用门禁据此判断)。
    """
    import hashlib
    h = hashlib.md5()
    base = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    for rel in _SIGNAL_CODE_FILES:
        p = os.path.join(base, rel)
        if rel == 'config/factor_config.yaml':
            h.update(f"{rel}|{_yaml_stripped_digest(p)};".encode('utf-8'))
            continue
        try:
            st = os.stat(p)
            h.update(f"{rel}|{st.st_mtime_ns}|{st.st_size};".encode('utf-8'))
        except OSError:
            h.update(f"{rel}|MISSING;".encode('utf-8'))
    # 2026-10-20: 行为钩子必须计入指纹 — QUANT_EDGE_ASOF=off(全样本边, 对照臂)
    # 改变 concept_heat 因子值, 但代码/mtime 不变 ⇒ 门禁不可见 ⇒ 静默复用生产信号
    # = 假臂。同款先例: pit_map0531(QUANT_CONCEPT_MAP 不可见)、E-B1v2(warmup 走
    # sidecar)。只在 env **非空**时混入: 生产态(未设)指纹不变, 零扰动。
    _ea = os.environ.get('QUANT_EDGE_ASOF', '')
    if _ea:
        h.update(f"env.QUANT_EDGE_ASOF|{_ea};".encode('utf-8'))
    return h.hexdigest()[:8]


def _compute_factor_data_raw(stock_file_map: dict, factor_dates: list,
                             lookback: int, forward_period: int,
                             num_workers: int, fd) -> pd.DataFrame:
    """并行计算原始因子数据 → 临时CSV流式合并 → 返回未掩码DataFrame.

    0f-v3重构 (2026-09-18): 计算块从prepare_factor_data抽出 —
    缓存只存raw(掩码/中性化/rank前), 池掩码按每次运行的membership_map应用,
    避免"daily运行静默复用quarterly掩码缓存"的跨模式污染bug。
    """
    # 并行计算因子 - worker从文件读取数据，避免主进程加载全量stock_data_dict
    # fork + COW: 每个 worker 通过 _worker_fd 访问父进程已加载的基本面数据，无需重新从磁盘读取
    args_list = [
        (code, stock_file_map[code], factor_dates, lookback, forward_period)
        for code in stock_file_map.keys()
    ]

    # 分批构建 DataFrame：避免 list-of-dicts（~2GB 峰值内存）与 DataFrame 同时存在导致 OOM
    # chunksize=50 减少 IPC RPC 次数（4000只 / 50 = 80次，vs 10=400次）
    BATCH_SIZE = 20000  # 降低batch size以减少内存峰值
    batch_data = []
    total_results = 0

    # 使用临时CSV文件流式写入，避免df_chunks在内存中累积导致OOM
    tmp_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tmp_factor')
    # tmp_factor是->/tmp/quant_tmp_factor的symlink (WSL2 IO加速): /tmp重启即清空,
    # 悬空symlink时makedirs(exist_ok=True)抛FileExistsError — 重建目标目录恢复 (2026-09-18)
    if os.path.islink(tmp_dir):
        os.makedirs(os.path.realpath(tmp_dir), exist_ok=True)
    elif os.path.exists(tmp_dir) and not os.path.isdir(tmp_dir):
        os.unlink(tmp_dir)
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_csv_path = os.path.join(tmp_dir, 'factor_data_tmp.csv')
    tmp_csv_file = open(tmp_csv_path, 'w', encoding='utf-8')
    header_written = False
    column_order = None

    # 加载题材热度计算器（fork前, COW共享）
    concept_calc = None
    try:
        from .concept_heat import ConceptHeatCalculator
        concept_calc = ConceptHeatCalculator()
        concept_calc.load()
        print(f"题材热度计算器已加载: {len(concept_calc._concept_hist)} 概念板块历史")
    except Exception as e:
        print(f"题材热度计算器加载失败(fallback): {e}")

    def _flush_batch(batch):
        """将一批数据写入临时CSV，避免在内存中累积DataFrame"""
        nonlocal header_written, total_results, column_order
        if not batch:
            return
        df = pd.DataFrame(batch)
        # 关键: 不同batch的DataFrame列序可能不同(有/无基本面数据的行键集不同),
        # 直接to_csv会导致后续batch按各自列序写入 -> 读回时值错位。
        # 必须按首个batch固定的列序对齐后再写。
        if column_order is None:
            column_order = list(df.columns)
        else:
            df = df.reindex(columns=column_order)
        total_results += len(df)
        df.to_csv(tmp_csv_file, header=not header_written, index=False)
        header_written = True
        del df
        batch.clear()

    import platform
    if platform.system() == 'Windows':
        print("Windows: single-process factor computation (spawn overhead too high)")
        _init_factor_worker(fd, concept_calc)
        for args in tqdm(args_list, desc="计算因子"):
            res = _compute_stock_factors_worker(args)
            batch_data.extend(res)
            if len(batch_data) >= BATCH_SIZE:
                _flush_batch(batch_data)
    else:
        ctx = multiprocessing.get_context('fork')
        with ctx.Pool(num_workers, initializer=_init_factor_worker, initargs=(fd, concept_calc)) as pool:
            for res in tqdm(pool.imap(_compute_stock_factors_worker, args_list, chunksize=50),
                           total=len(args_list), desc="计算因子"):
                batch_data.extend(res)
                if len(batch_data) >= BATCH_SIZE:
                    _flush_batch(batch_data)

    # 最后一批
    _flush_batch(batch_data)
    tmp_csv_file.close()

    del args_list, batch_data  # 释放参数列表和临时数据

    # 从临时CSV读取合并后的因子数据（一次性加载，内存可控）
    print(f"从临时CSV加载因子数据: {tmp_csv_path}")
    factor_data = pd.read_csv(tmp_csv_path, parse_dates=['date'], dtype={'code': str, 'industry': str}) if os.path.getsize(tmp_csv_path) > 0 else pd.DataFrame()
    # 确保code保持为字符串（CSV读写可能转为int64导致与concept_map的isin不匹配）
    if len(factor_data) > 0 and factor_data['code'].dtype != object:
        factor_data['code'] = factor_data['code'].astype(str).str.zfill(6)
    # 清理临时文件 (tmp_dir若是symlink不删目录 — 保留给下次运行复用)
    try:
        os.remove(tmp_csv_path)
    except Exception:
        pass
    print(f"因子数据: {total_results} 条原始记录 → {len(factor_data)} 行")

    # === 内存优化：float64 → float32 (精度足够，内存减半) ===
    if len(factor_data) > 0:
        for col in factor_data.columns:
            if col in ('code', 'date', 'industry'):
                continue
            if factor_data[col].dtype == 'float64':
                factor_data[col] = pd.to_numeric(factor_data[col], downcast='float')
            elif factor_data[col].dtype == 'int64':
                factor_data[col] = pd.to_numeric(factor_data[col], downcast='integer')
    import gc
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(ctypes.c_int(0))
    except Exception:
        pass

    return factor_data


def _standardize_ranks_polars(factor_df, rank_factors):
    """截面排名标准化(polars, 2026-10-09) — 与原pandas逐日期groupby版逐位一致。

    语义(与旧版完全对齐):
      valid = 该日期全部因子非空的行; n_valid<5 该日期整组NaN
      rank_pct = 在valid子集内 rank(method='average', pct=True)  (分母=valid行数)
      normal = sqrt(2)*erfinv(2*clip(rank_pct,0.001,0.999)-1) → clip(-3,3)
    polars实现要点: 先mask外行置null再rank/count(排名集合与分母都只含valid行);
    'average'与pandas默认method一致。惰性import避免worker spawn侧开销。
    探针: /tmp/probe_polars_rankstd.py (30日期×63因子, 0/8,783,680单元不一致)。
    """
    import polars as pl
    pdf = pl.from_pandas(factor_df[['date'] + rank_factors])
    mask = pl.all_horizontal([pl.col(fc).is_not_null() for fc in rank_factors])
    # 表达式在pdf(含因子列)上下文中求值, 单遍select产出rank列
    exprs = []
    for fc in rank_factors:
        fcm = pl.when(mask).then(pl.col(fc)).otherwise(None)
        cnt = fcm.count().over('date')
        e = (fcm.rank('average').over('date') / cnt) \
            .clip(0.001, 0.999) \
            .map_batches(lambda s: np.sqrt(2.0) * erfinv(2.0 * s.to_numpy() - 1.0)) \
            .clip(-3.0, 3.0)
        e = pl.when(cnt >= 5).then(e).otherwise(None)
        exprs.append(e.fill_nan(None).alias(f'{fc}_rank'))
    res_pd = pdf.select('date', *exprs).drop('date').to_pandas()
    # 行序与输入一致(polars保持顺序), 直接按列赋值
    for fc in rank_factors:
        factor_df[f'{fc}_rank'] = res_pd[f'{fc}_rank'].to_numpy()


def prepare_factor_data(stock_file_map: dict, fd,
                       detailed_industries: dict,
                       all_dates: list,
                       num_workers: int = 8,
                       membership_map: dict = None) -> Tuple[pd.DataFrame, dict, list]:
    """预计算所有股票的因子数据（用于动态因子选择）

    Args:
        stock_file_map: {code: filepath} 股票文件路径映射（worker从磁盘读取，避免主进程加载全量数据）
        fd: FundamentalData 实例
        detailed_industries: 行业分类配置
        all_dates: 全市场交易日列表（可从sh000001获取，避免遍历所有股票）
        num_workers: 并行进程数
        membership_map: 0f季度日历池成员映射 {boundary_ts: set(codes)} —
            组装后按(date,boundary)掩码非成员码的因子值(NaN), 中性化/rank/ML自动继承。
            None=不掩码(off模式, 现状)

    Returns:
        tuple: (factor_data, industry_codes, all_dates)
            - factor_data: 所有股票在所有日期的因子值 DataFrame
            - industry_codes: {category: [codes]} 行业映射
            - all_dates: 所有交易日期列表
    """
    config_loader = load_config()
    lookback = config_loader.get('industry_factor_config.lookback_days', 250)
    forward_period = config_loader.get('dynamic_factor.forward_period', 20)

    # 构建行业映射（使用fundamental_data，无需加载价格数据）
    industry_codes = {cat: [] for cat in detailed_industries.keys()}
    unmatched_count = 0
    mid_date = all_dates[len(all_dates) // 2] if all_dates else None

    for code in stock_file_map.keys():
        matched = False
        try:
            ind = fd.get_industry(code, mid_date) if fd else None
            # 如果中期日期无数据，尝试用更晚的日期
            if ind is None and len(all_dates) > 200:
                for late_date in reversed(all_dates[-200:]):
                    ind = fd.get_industry(code, late_date) if fd else None
                    if ind:
                        break
            if ind:
                for cat, keywords in detailed_industries.items():
                    if any(kw in str(ind) for kw in keywords):
                        industry_codes[cat].append(code)
                        matched = True
                        break
        except Exception:
            pass
        if not matched:
            unmatched_count += 1

    # 日期采样：每N个交易日采样1次（减少计算量）
    # 尾部forward_period日不再截断: NaN标签由消费方(IC训练窗/ML purge)自行排除,
    # 保留尾部行使ML预测/实盘信号覆盖最新交易日
    date_step = config_loader.get('dynamic_factor.date_sample_step', 3)
    all_factor_dates = all_dates[lookback:]
    factor_dates = all_factor_dates[::date_step]
    del all_factor_dates  # 释放中间列表
    print(f"预计算因子数据: {len(factor_dates)} 个时间点 (每{date_step}日采样), {len(stock_file_map)} 只股票")

    # === 因子数据缓存: 避免重复计算 ===
    # 缓存键 = 池/日期/参数 + 数据内容指纹 (输入文件变化 → 键变化 → 缓存自动失效)
    from .cache_manager import load_factor_cache, save_factor_cache
    import hashlib
    _n_stocks = len(stock_file_map)
    _n_dates = len(factor_dates)
    _data_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                              '..', '..', 'data'))
    # 2026-10-17: 概念映射的**实际生效路径**必须进缓存键。旧版硬编码
    # data/stock_concept_map.pkl, 对 QUANT_CONCEPT_MAP 覆盖与版本表
    # (QUANT_CONCEPT_MAP_HIST)完全不可见 ⇒ 翻钩子会静默复用上一次的因子
    # = 假臂(实测 pit_map0531 即此机理: 与 pit_ident 指纹相同, md5 逐位相同)。
    from .cmap_hist import resolve_hist_path
    _concept_inputs = [
        os.environ.get('QUANT_CONCEPT_MAP')
        or os.path.join(_data_root, 'stock_concept_map.pkl'),
        os.path.join(_data_root, 'concept_hist.pkl'),
        os.path.join(_data_root, 'concept_daily.csv'),
    ]
    _hist_eff = resolve_hist_path()
    if _hist_eff:
        _concept_inputs.append(_hist_eff)
    _data_fp = _data_fingerprint(stock_file_map, fd, _concept_inputs)
    # raw缓存键用_raw_code_fingerprint (0f-v3: 豁免stock_pool.py与yaml池节 —
    # 掩码/闸变化不重算raw); 打印两者便于取证
    _code_fp = _raw_code_fingerprint()
    _cache_key_str = ','.join(sorted(stock_file_map.keys())[:100]) + str(_n_stocks) + \
                     str(factor_dates[0]) + str(factor_dates[-1]) + str(date_step) + str(lookback) + \
                     '_fp' + _data_fp + '_cfp' + _code_fp
    # 2026-10-20: QUANT_EDGE_ASOF 改变 concept_heat 边的拟合口径(as-of月末 vs
    # 全样本) ⇒ 改变因子值, 但对上面的数据/代码指纹均不可见 ⇒ 因子缓存命中
    # ⇒ 假臂。仅 env 非空时附加(生产态零扰动)。同款先例: QUANT_CONCEPT_MAP
    # 走 _concept_inputs 路径进 _data_fp (2026-10-17)。
    _edge_asof_env = os.environ.get('QUANT_EDGE_ASOF', '')
    if _edge_asof_env:
        _cache_key_str += '_ea' + _edge_asof_env
    _cache_hash = hashlib.md5(_cache_key_str.encode()).hexdigest()[:8]
    print(f"数据指纹: {_data_fp} (K线{len(stock_file_map)}文件 + 基本面 + 题材输入); "
          f"代码指纹(raw): {_code_fp}; 信号代码指纹: {_signal_code_fingerprint()}")
    _cached = load_factor_cache(_n_stocks, _n_dates, _cache_hash)
    _fresh = _cached is None or len(_cached) == 0
    if not _fresh:
        print(f"使用因子缓存，跳过因子计算")
        if _cached['code'].dtype != object:
            _cached['code'] = _cached['code'].astype(str).str.zfill(6)
        factor_data = _cached
    else:
        factor_data = _compute_factor_data_raw(stock_file_map, factor_dates,
                                               lookback, forward_period,
                                               num_workers, fd)

    # =========================================================================
    # 掩码/中性化/rank 每次运行应用 (0f-v3 2026-09-18):
    # 缓存只存raw因子值 — 池掩码与模式(quarterly/daily/off)相关, 掩码后数据
    # 入缓存会导致跨模式静默复用 (daily运行命中quarterly掩码缓存的bug);
    # 中性化/rank截面依赖掩码结果, 必须同路径每次重算。
    # =========================================================================

    # 预测截面不能按未来标签删行。异常标签仅可在各自训练窗口内处理。
    # 极端涨跌也是真实左/右尾；当前采用截面排名标签，无需按收益阈值删除。

    # 保存raw缓存 (掩码/中性化/rank前) — 仅fresh计算时; 写入失败不影响主流程
    if _fresh:
        try:
            save_factor_cache(factor_data, _n_stocks, _n_dates, _cache_hash)
        except Exception:
            pass

    # 0f日历池掩码 (2026-09-17; v3 2026-09-18: 每次运行按membership_map应用,
    # 缓存不存掩码后数据): 非成员日期的因子值置NaN —
    # 消除"晚入池码向历史截面注入因子值"的非PIT成分; 中性化/rank/ML经notna自动继承。
    # 指数(sh000001/sh000852/000001/399006)恒成员, 不掩码 (与bt_execution池口径一致)。
    if membership_map is not None and len(factor_data) > 0:
        _IDX_CODES = {'sh000001', 'sh000852', '000001', '399006'}
        _boundaries = sorted(pd.Timestamp(b) for b in membership_map.keys())
        _b_arr = pd.DatetimeIndex(_boundaries)
        _dates = pd.to_datetime(factor_data['date'])
        _b_idx = np.searchsorted(_b_arr.values, _dates.values, side='right') - 1
        _mask_cols = [c for c in factor_data.columns
                      if c not in ('code', 'date', 'industry', 'future_ret', 'label_end_date')]
        _masked_cells = 0
        for _bi, _b in enumerate(_boundaries):
            _rows = _b_idx == _bi
            if not _rows.any():
                continue
            _codes = factor_data['code'].values[_rows]
            _non_member = (~np.isin(_codes, list(membership_map[_b]))
                           & ~np.isin(_codes, list(_IDX_CODES)))
            if _non_member.any():
                _masked_cells += int(_non_member.sum())
                _mask_rows = _rows.copy()
                _mask_rows[_rows] = _non_member
                factor_data.loc[_mask_rows, _mask_cols] = np.nan
        _pre_rows = _b_idx < 0
        if _pre_rows.any():
            _pre_non_idx = _pre_rows & ~factor_data['code'].isin(_IDX_CODES).values
            _masked_cells += int(_pre_non_idx.sum())
            factor_data.loc[_pre_non_idx, _mask_cols] = np.nan
        print(f"因子数据日历池掩码: 边界{len(_boundaries)}个, "
              f"掩码{_masked_cells}行非成员因子值 (指数豁免)")

    # 因子中性化（行业+市值剥离）
    # 缠论/结构字段具有绝对含义（非截面相对值），不参与中性化
    _CHAN_STRUCTURAL_FIELDS = {
        # 中枢位置
        'pivot_position', 'pivot_present', 'pivot_zg', 'pivot_zd', 'pivot_zz',
        'pivot_level', 'pivot_count',
        'chan_pivot_present', 'chan_pivot_zg', 'chan_pivot_zd', 'chan_pivot_zz', 'chan_pivot_level',
        'breakout_above_pivot', 'breakout_below_pivot', 'consolidation_zone',
        # 走势结构
        'zhongyin', 'structure_complete', 'alignment_score',
        'trend_type', 'trend_strength',
        # 分型/笔/线段
        'top_fractals', 'bottom_fractals', 'fractal_type',
        'stroke_direction', 'stroke_id', 'stroke_count',
        'segment_direction', 'segment_id', 'segment_count',
        # 买卖点
        'buy_point', 'sell_point', 'buy_confidence', 'sell_confidence',
        'bi_buy_point', 'bi_sell_point', 'bi_buy_confidence', 'bi_sell_confidence', 'bi_td',
        'confirmed_buy', 'confirmed_sell', 'signal_level', 'buy_strength', 'sell_strength',
        'second_buy_point', 'second_buy_confidence', 'second_buy_b1_ref',
        # 背离
        'top_divergence', 'bottom_divergence', 'hidden_top_divergence',
        'hidden_bottom_divergence', 'divergence_active',
        # 分型质量
        'bottom_fractal_quality', 'bottom_fractal_strength',
        'bottom_fractal_vol_ratio', 'bottom_fractal_vol_spike', 'bottom_fractal_ema_dist',
        # 缠论信号强度 + 结构止损
        'chan_buy_score', 'chan_sell_score', 'structure_stop_price',
        # 独立系统 (资金流/情绪)
        'capital_flow_score', 'capital_flow_direction',
        'news_sentiment_score', 'news_sentiment_direction',
        'smart_money_flow',
    }
    neu_config = config_loader.get('factor_neutralization', {})
    if neu_config.get('enabled', False) and len(factor_data) > 0:
        from .factor_neutralizer import neutralize_factor_df

        # 删除噪声因子（|IC| < 0.01，纯噪声无预测力）
        _NOISE_FACTORS = {
            'turnover', 'turnover_ratio', 'turnover_5d_avg', 'turnover_20d_avg',
            'amplitude', 'amplitude_5d_avg', 'amplitude_20d_avg',
            'chan_divergence_strength', 'chan_buy_point',
            'avg_up_ret', 'up_capture',
            'max_gap_20d', 'max_ret_20d', 'gap_up_count_20d',
            'consecutive_up', 'pos_in_range_pos_20d',
        }
        dropped_noise = [c for c in _NOISE_FACTORS if c in factor_data.columns]
        if dropped_noise:
            factor_data.drop(columns=dropped_noise, inplace=True)
            print(f"删除噪声因子: {len(dropped_noise)} 个 (|IC|<0.01)")

        factor_cols = [c for c in factor_data.columns
                       if c not in ('code', 'date', 'future_ret', 'industry', 'label_end_date')
                       and c not in _CHAN_STRUCTURAL_FIELDS]
        skipped_chan = len([c for c in factor_data.columns if c in _CHAN_STRUCTURAL_FIELDS])
        print(f"因子中性化: {len(factor_cols)} 个因子列 (跳过{skipped_chan}个缠论/结构字段), "
              f"行业={'✓' if neu_config.get('neutralize_industry') else '✗'}, "
              f"市值={'✓' if neu_config.get('neutralize_market_cap') else '✗'}")
        factor_data = neutralize_factor_df(
            factor_data, factor_cols, industry_col='industry',
            market_cap_col='ln_cap' if neu_config.get('neutralize_market_cap', False) else None,
            market_cap_transform='compressed_log', date_col='date',
            neutralize_industry=neu_config.get('neutralize_industry', True),
            min_samples_per_industry=neu_config.get('min_samples_per_industry', 3),
        )
        # 用中性化后的列替换原始列
        for fc in factor_cols:
            neu_col = f'{fc}_neu'
            if neu_col in factor_data.columns:
                factor_data[fc] = factor_data[neu_col]
        factor_data.drop(columns=[c for c in factor_data.columns if c.endswith('_neu')],
                        inplace=True)

    # === 截面排名标准化：每个因子按日期 groupby → rank_pct → Normal变换 → [-1,1] ===
    # 解决原始因子量级差异导致的加权组合失效问题（IC权重被量级差异淹没）
    # 参考 Qlib/WorldQuant: cross-sectional ranking before factor combination
    if len(factor_data) > 0:
        _RANK_FACTORS = [c for c in factor_data.columns
                         if c not in ('code', 'date', 'future_ret', 'industry', 'label_end_date')
                         and c not in _CHAN_STRUCTURAL_FIELDS
                         and not c.endswith('_rank')]
        _rank_cols = [f'{fc}_rank' for fc in _RANK_FACTORS]
        for rc in _rank_cols:
            if rc not in factor_data.columns:
                factor_data[rc] = np.nan

        # 2026-10-09 polars向量化: 原pandas逐日期groupby = 816日期×63因子
        # 51,408次groupby切片+全帧.loc写入(单核~20min) → polars .over('date')
        # 单遍(~10s)。探针逐位一致: /tmp/probe_polars_rankstd.py 0/8.78M单元。
        _standardize_ranks_polars(factor_data, _RANK_FACTORS)
        if _rank_cols:
            print(f"截面排名标准化: {len(_RANK_FACTORS)} 个因子 → {len(_rank_cols)} 个 _rank 列 ([-3,3])")

    return factor_data, industry_codes, all_dates
