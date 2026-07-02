# ⚠️ 回测入口 — 修改因子选择/信号生成/组合构建逻辑时,
#    务必同步更新 core/live_init.py 的 init_live_engine/update_live_engine
#    确保回测与实盘等价。
import backtrader as bt
import pandas as pd
import numpy as np
import datetime
import os
from tqdm import tqdm
import math
from collections import defaultdict
import multiprocessing
from functools import partial
import gc
import warnings
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
import ctypes

from core.strategy import Strategy
from core.fundamental import FundamentalData
from core.signal_engine import SignalEngine
from core.factor_preparer import prepare_factor_data
from core.signal_store import SignalStore
from core.config_loader import load_config
from core.monitor import monitor, get_logger
from core.pipeline_logger import plog
from core.industry_mapping import INDUSTRY_KEYWORDS, build_fine_industry_map
from core.market_regime_detector import MarketRegimeDetector
from core.stock_pool import get_stock_pool, get_exclusion_set

# 情绪分析集成
try:
    from sentiment.orchestrator import SentimentOrchestrator
    SENTIMENT_AVAILABLE = True
except ImportError:
    SENTIMENT_AVAILABLE = False

# 加载配置
config = load_config()

# 回测参数 - 从配置文件读取
CASH = config.get('backtest.cash', 100000.0)
COMMISSION = config.get('backtest.commission', 0.0001)  # 万1佣金
STAMP_TAX = config.get('backtest.stamp_tax', 0.0005)  # A股印花税万5(卖出单边)
EFFECTIVE_COMM = COMMISSION + STAMP_TAX * 0.5  # 双边均摊: 买入万1, 卖出万6, 均摊万3.5
PERC = config.get('backtest.slippage', 0.001)
REBALANCE_DAYS = config.get('backtest.rebalance_days', 20)

# A股涨跌停限制数据（预计算，供BacktraderExecution使用）
_LIMIT_DATA = {}   # {(code, date): 'up' | 'down'}

NUM_WORKERS = config.get('backtest.num_workers', 2)

def _malloc_trim(pad=0):
    """将 Python 已释放但未归还 OS 的内存归还给内核（Linux only）。

    在 gc.collect() 之后调用，可显著降低进程 RSS，避免 fork 时子进程继承
    inflated 内存地址空间导致 OOM。
    """
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(ctypes.c_int(pad))
    except Exception as e:
        import traceback; print(f"[ERR] " + __file__ + ":" + str(e)); traceback.print_exc()  # 非 Linux / 权限不足时静默跳过

# 回测日期范围过滤 (加速测试: fromdate='2024-01-01', todate='2026-05-30')
FROMDATE = config.get('backtest.fromdate', None)
TODATE = config.get('backtest.todate', None)

# 数据路径 - 从配置文件读取，默认相对于策略目录（而非 CWD）
_STRATEGY_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_STRATEGY_DIR)
DATA_PATH = config.get('paths.data', os.path.join(_PROJECT_DIR, 'data/stock_data/backtrader_data/'))
FUNDAMENTAL_PATH = config.get('paths.fundamental', os.path.join(_PROJECT_DIR, 'data/stock_data/fundamental_data/'))


# 全局变量用于 worker 进程
_worker_engine = None
_worker_use_dynamic = False

# 模块级默认值（函数内通过 global 声明赋值，后处理代码读取）
dynamic_factor_stats = {'hit': 0, 'miss': 0, 'factor_names': {}}
use_dynamic = False
precomputed_cache = None
_ml_total_preds = 0
_ml_val_ic = None


def _build_concept_industry_codes(stock_codes, min_stocks=20):
    """从概念板块映射构建 industry_codes，与离线标定对齐。

    替代 prepare_factor_data 的关键词匹配（20大类），使用 stock_concept_map.pkl
    的精确概念板块，与 calibrate_industry_regime / factor_config.yaml 的分组方式完全一致。

    Returns:
        dict: {concept_name: [code1, code2, ...]}
    """
    import pickle
    concept_map_path = os.path.join(_PROJECT_DIR, 'data', 'stock_concept_map.pkl')
    if not os.path.exists(concept_map_path):
        print("[WARNING] stock_concept_map.pkl 不存在，回退到关键词映射")
        return {}

    with open(concept_map_path, 'rb') as f:
        raw = pickle.load(f)

    STYLE_KW = ['融资融券', '深股通', '沪股通', '富时罗素', '标准普尔', 'MSCI',
                '创业板综', '机构重仓', 'QFII', '破增发', '破发股', '昨日高',
                '中证500', '深成500', '中盘股', '小盘股', '央国企改革',
                '西部大开发', '年报预增', '专精特新', '上证380', 'HS300',
                '微盘股', '百元股', '大盘股', '小盘成长', '小盘价值',
                '转债标的', '长江三角', '深圳特区', '破净股', '创投',
                # v2: 过滤投资风格/主题标签，只保留产业类概念
                '养老金', '社保重仓', '基金重仓', '券商重仓', '保险重仓',
                '茅指数', '央视50', '大盘价值', '大盘成长', '中盘价值', '中盘成长',
                '价值股', '成长股', '红利股', '低市盈率', '高市盈率',
                '证金', '汇金', '国资云', '央企改革', '地方国资',
                '上证180', '上证50', '沪深300', '中证100', '中证1000',
                '深证100', '创业板指', '科创50', '科创100',
                'AB股', 'AH股', 'B股', 'ST股',
                '昨日涨停', '昨日首板', '昨日炸板', '昨日连板',
                '次新股', '预增', '预减', '扭亏', 'IPO受益', '壳资源',
                '参股券商', '参股银行', '参股保险', '参股期货', '参股新三板',
                '券商概念', '微盘精选', '百元股', '低安全分', '破发', '回购', '举牌',
                '长期破净', '高股息', '股权转让', '送转', '高送转',
                '新高', '权重股', '行业龙头', '近期新高', '历史新高', '百日新高',
                ]
    stock_codes_set = set(stock_codes)
    concept_to_codes = defaultdict(list)
    for code, concepts in raw.items():
        if code not in stock_codes_set:
            continue
        filtered = [c for c in concepts if not any(kw in c for kw in STYLE_KW)]
        for c in filtered:
            concept_to_codes[c].append(code)

    result = {
        c: clist for c, clist in concept_to_codes.items()
        if len(clist) >= min_stocks
    }
    print(f"概念板块映射: {len(result)} 个概念 (min_stocks={min_stocks}), "
          f"覆盖 {len(set(c for clist in result.values() for c in clist))} 只股票")
    return result


def _init_worker(fundamental_path, stock_codes, use_dynamic, industry_codes, factor_cache, all_dates, regime_df,
                 ml_model_path=None, ml_preds_path=None, rank_parquet_path=None):
    """Worker 进程初始化函数 (spawn 模式)

    每个 worker 独立创建 FundamentalData 和 ML 预测器，避免 fork 导致的
    OpenMP/XGBoost 死锁问题。
    ML 预测量改为按股票代码按需从 parquet 读取（不再加载 69MB pickle/进程）。
    因子排名同样按需从 parquet 读取。
    """
    global _worker_engine, _worker_use_dynamic, _worker_ml_preds_path, _worker_rank_parquet_path
    _worker_use_dynamic = use_dynamic
    _worker_ml_preds_path = ml_preds_path  # 只记路径, 按需读取
    _worker_rank_parquet_path = rank_parquet_path

    _worker_engine = SignalEngine()

    # spawn 模式：每个 worker 独立创建 FundamentalData
    if fundamental_path is not None and stock_codes is not None:
        from core.fundamental import FundamentalData
        _worker_fd = FundamentalData(fundamental_path, stock_codes=stock_codes)
        _worker_engine.set_fundamental_data(_worker_fd)

    if regime_df is not None:
        _worker_engine.set_market_regime(regime_df)

    if use_dynamic and factor_cache is not None and all_dates is not None:
        _worker_engine.set_industry_mapping(industry_codes)
        _worker_engine.dynamic_factor_selector.set_factor_cache(factor_cache, all_dates)

    if ml_model_path is not None and os.path.exists(ml_model_path):
        try:
            from core.ml_predictor import MLFactorPredictor
            worker_ml = MLFactorPredictor()
            worker_ml.load_model(ml_model_path)
            _worker_engine.set_ml_predictor(worker_ml)
        except Exception as e:
            import traceback; print(f"[ERR] " + __file__ + ":" + str(e)); traceback.print_exc()


def _generate_stock_signal_worker(args):
    """Worker 函数：为一个股票生成信号 — 直接从文件读取，避免 pickle 传大数据"""
    global _worker_engine, _worker_use_dynamic, _worker_diag_reported, _worker_ml_preds_path, _worker_rank_parquet_path
    code, filepath = args
    code = str(code).zfill(6)  # 归一化为6位字符串，确保与所有下游系统一致

    try:
        engine = _worker_engine

        if engine is None:
            engine = SignalEngine()

        # 按需加载本股票的 ML 预测（~500条, <10KB）, 替代 69MB 全量 pickle
        if _worker_ml_preds_path is not None and os.path.exists(_worker_ml_preds_path):
            _stock_preds_df = pd.read_parquet(_worker_ml_preds_path,
                                              filters=[('code', '==', code)])
            if len(_stock_preds_df) > 0:
                _stock_preds = {
                    (str(row['code']).zfill(6), pd.Timestamp(row['date'])): float(row['ml_pred'])
                    for _, row in _stock_preds_df.iterrows()
                }
                engine.set_ml_predictions(_stock_preds)
            else:
                engine.set_ml_predictions({})

        # 按需加载本股票的截面排名数据（~650行 × 50因子, ~130KB）, 用于因子组合评分
        if _worker_rank_parquet_path is not None and os.path.exists(_worker_rank_parquet_path):
            _stock_rank_df = pd.read_parquet(_worker_rank_parquet_path,
                                              filters=[('code', '==', code)])
            if len(_stock_rank_df) > 0:
                _stock_rank_df['date'] = pd.to_datetime(_stock_rank_df['date'])
                engine.set_stock_rank_data(code, _stock_rank_df)
            else:
                engine.set_stock_rank_data(code, None)

        store = SignalStore()
        data = pd.read_csv(filepath, parse_dates=['datetime'])
        # 日期范围过滤（与主进程一致的加速优化）
        if FROMDATE:
            data = data[data['datetime'] >= FROMDATE]
        if TODATE:
            data = data[data['datetime'] <= TODATE]
        if len(data) < 60:
            del data
            return (code, {}, {})
        engine.generate(code, data, store)

        if hasattr(engine, 'fundamental_data') and engine.fundamental_data is not None:
            engine.fundamental_data.clear_stock_cache(code)

        dyn_fail_snapshot = {}
        if hasattr(engine, '_dyn_fail'):
            dyn_fail_snapshot = dict(engine._dyn_fail)
            engine._dyn_fail = {k: 0 for k in engine._dyn_fail}

        # 收集 worker 的诊断数据（因子选择、买入信号等）
        diag_data = {}
        if hasattr(engine, '_diag') and engine._diag is not None:
            m = engine._diag.metrics
            diag_data = {
                'factor_selection': dict(m.get('factor_selection', {})),
                'buy_signals': m.get('buy_signals', 0),
                'buy_by_buy_point': dict(m.get('buy_by_buy_point', {})),
                'buy_by_factor_family': dict(m.get('buy_by_factor_family', {})),
                'gate_quality_samples': list(m.get('gate_quality_samples', [])),
                'hard_rejects': m.get('hard_rejects', 0),
                'alt_nb': m.get('alt_northbound_hits', 0),
                'alt_mg': m.get('alt_margin_hits', 0),
                'alt_dt': m.get('alt_dragon_tiger_hits', 0),
            }
            engine._diag.reset()

        # 收集 worker 的因子选择统计
        stats_snapshot = {}
        if hasattr(engine, '_stats'):
            stats_snapshot = dict(engine._stats)
            engine._stats = {k: 0 for k in engine._stats}

        # 收集 worker 的 BOM 统计
        bom_snapshot = {}
        if hasattr(engine, '_bom_diag'):
            bd = engine._bom_diag
            bom_snapshot = {
                'total': bd.get('total', 0),
                'hit': bd.get('hit', 0),
                'miss': bd.get('miss', 0),
                'moat': bd.get('moat', 0),
                'sum_score': bd.get('sum_score', 0.0),
                'n_unique': len(bd.get('_unique_codes', set())),
                'n_moat_codes': len(bd.get('_moat_codes', set())),
            }
            # Reset worker BOM counters
            bd['total'] = bd['hit'] = bd['miss'] = bd['moat'] = 0
            bd['sum_score'] = 0.0
            bd['_unique_codes'].clear()
            bd['_moat_codes'].clear()

        result = (code, store._store, dyn_fail_snapshot, diag_data, stats_snapshot, bom_snapshot)
        del data
        del store
        return result
    except Exception as e:
        print(f"[Worker Error] {code}: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return (code, {}, {}, {}, {})


def add_data_and_signal(cerebro, strategy, fundamental_data=None):
    """加载数据、生成信号、添加到cerebro。
    同时预计算涨跌停数据和ST股票集合。
    """
    global _LIMIT_DATA, _ml_total_preds, _ml_val_ic
    all_items = os.listdir(DATA_PATH)
    stock_codes = []

    # === Phase 0: 扫描文件构建路径映射（不加载数据，避免 ~200MB stock_data_dict） ===
    stock_file_map = {}  # code → filepath
    _LIMIT_DATA = {}     # 重置涨跌停数据
    _LOAD_COLS = ['datetime', 'open', 'high', 'low', 'close', 'volume', 'change_percent', 'amplitude']
    _DTYPE_MAP = {
        'open': 'float32', 'high': 'float32', 'low': 'float32', 'close': 'float32',
        'volume': 'float32', 'change_percent': 'float32', 'amplitude': 'float32',
    }
    for item in all_items:
        if item.startswith('._'):
            continue
        filepath = DATA_PATH + item
        if item.endswith('_qfq.csv'):
            name = item[:-8]
        elif item.endswith('_hfq.csv'):
            name = item[:-8]
        else:
            continue
        stock_file_map[name] = filepath

    # === 股票池过滤（先过滤再预计算，避免读不需要的文件） ===
    stock_pool_enabled = config.get('stock_pool.enabled', True)
    if stock_pool_enabled:
        stock_pool = get_stock_pool()
        pool_codes = stock_pool | {'sh000001'}
        before_count = len(stock_file_map)
        stock_file_map = {k: v for k, v in stock_file_map.items() if k in pool_codes}
        after_count = len(stock_file_map)
        print(f"股票池过滤: {before_count} -> {after_count} 只 (全部通过质量筛选)")
    else:
        print(f"股票池过滤: 已关闭，使用全市场 {len(stock_file_map)} 只股票")

    # 预计算涨跌停数据（流式读取，仅加载 change_percent + amplitude 列，读后即弃）
    print("预计算涨跌停数据（流式读取）...")
    for name, filepath in tqdm(list(stock_file_map.items()), desc="limit data"):
        try:
            data = pd.read_csv(filepath, parse_dates=['datetime'],
                              usecols=['datetime', 'change_percent', 'amplitude'])
        except Exception:
            continue
        if FROMDATE:
            data = data[data['datetime'] >= FROMDATE]
        if TODATE:
            data = data[data['datetime'] <= TODATE]
        if name == 'sh000001' or 'change_percent' not in data.columns:
            continue
        chg = data['change_percent'].values
        amp = data['amplitude'].values if 'amplitude' in data.columns else np.ones(len(data)) * 99
        dates = data['datetime'].values
        for i in range(len(data)):
            cp = chg[i]
            if pd.isna(cp):
                continue
            ap = amp[i] if not pd.isna(amp[i]) else 99
            # Fix: 原ap<3.0过严，大幅漏判真实涨停（开盘-2%拉到+10%振幅12%被漏掉）
            # 涨幅9.5%+在A股必然触及涨停板，振幅只用排除极端尾盘逆转(ap>=20%)
            if cp >= 9.5 and ap < 20.0:
                _LIMIT_DATA[(name, pd.Timestamp(dates[i]).date())] = 'up'
            elif cp <= -9.5 and ap < 20.0:
                _LIMIT_DATA[(name, pd.Timestamp(dates[i]).date())] = 'down'
            elif 4.5 <= cp <= 5.5 and ap < 12.0:  # ST涨停±5%, 放宽振幅
                _LIMIT_DATA[(name, pd.Timestamp(dates[i]).date())] = 'up'
            elif -5.5 <= cp <= -4.5 and ap < 12.0:
                _LIMIT_DATA[(name, pd.Timestamp(dates[i]).date())] = 'down'
        del data
    print(f"涨跌停数据: {len(_LIMIT_DATA)} 条 (涨停={sum(1 for v in _LIMIT_DATA.values() if v=='up')}, "
          f"跌停={sum(1 for v in _LIMIT_DATA.values() if v=='down')})")

    # === 科创板(688xxx) 不再剔除 — 2025-2026年妖股主要集中板块 ===
    star_codes = set()
    before_excl_star = len(stock_file_map)
    print(f"科创板保留: {before_excl_star} 只 (不再排除)")

    # ST股票不再静态排除 — 改为向量化回测中逐日调用 fundamental_data.is_st() (line ~1177)
    # 先加载sh000001获取交易日历（所有后续步骤依赖）
    sh000001_file = stock_file_map.get('sh000001')
    if sh000001_file:
        index_df = pd.read_csv(sh000001_file, parse_dates=['datetime'])
        if FROMDATE:
            index_df = index_df[index_df['datetime'] >= FROMDATE]
        if TODATE:
            index_df = index_df[index_df['datetime'] <= TODATE]
        calendar_index = pd.DatetimeIndex(sorted(index_df['datetime']))
    else:
        # fallback: 从任意一只股票获取日期
        calendar_index = None
        for fp in list(stock_file_map.values())[:1]:
            tmp = pd.read_csv(fp, parse_dates=['datetime'])
            calendar_index = pd.DatetimeIndex(sorted(tmp['datetime']))
            break

    # 处理指数数据 - 生成市场状态 (Fix#6: 多指数风格检测)
    # calendar_index 已在上方从 sh000001 构建
    regime_df = None
    # 加载小盘/成长指数（如果存在）
    small_cap_df = None
    growth_df = None
    if '000852' in stock_file_map:
        small_cap_df = pd.read_csv(stock_file_map['000852'], parse_dates=['datetime'])
    if '399006' in stock_file_map:
        growth_df = pd.read_csv(stock_file_map['399006'], parse_dates=['datetime'])

    if sh000001_file:
        strategy.generate_market_regime(
            index_df,
            small_cap_df=small_cap_df,
            growth_df=growth_df,
        )
        regime_df = strategy.index_data
        has_sc = small_cap_df is not None
        has_gr = growth_df is not None
        print(f"市场状态数据已生成，共 {len(regime_df)} 条记录"
              f" (中证1000={'✓' if has_sc else '✗'}, 创业板指={'✓' if has_gr else '✗'})")
    # 释放小盘/成长指数数据（已不需要）
    del small_cap_df, growth_df

    # 准备动态因子数据
    factor_mode = config.config.get('factor_mode', 'both')
    factor_df = None
    industry_codes = {}
    all_dates = sorted(calendar_index.tolist()) if calendar_index is not None else []
    # 因子训练需要历史数据: 扩展日期范围覆盖 lookback(250T) + IC warm-up(180T)
    # 430个交易日 ≈ 2年日历日，确保 factor_df 从回测起点就有足够的IC训练窗口
    _raw_sh_path = os.path.join(os.path.dirname(DATA_PATH.rstrip('/')), 'raw_data', '000001', 'qfq.csv')
    if os.path.exists(_raw_sh_path):
        _raw_dates = pd.to_datetime(pd.read_csv(_raw_sh_path, encoding='utf-8-sig', usecols=['日期'])['日期'])
        _early_dates = sorted(d for d in _raw_dates if pd.Timestamp(FROMDATE) - pd.Timedelta(days=730) <= d < pd.Timestamp(FROMDATE))
        all_dates = sorted(set(_early_dates) | set(all_dates))
    # 因子数据计算需要 factor_df: 动态模式(IC选择) 或 ML训练 时需要
    # fixed模式但ML启用时也需计算factor_df, 供ML训练使用
    _ml_enabled = config.config.get('ml', {}).get('enabled', False)
    _need_factor_df = factor_mode != 'fixed' or _ml_enabled
    if _need_factor_df:
        print(f"准备因子数据 (factor_mode={factor_mode})...")
        # 获取股票代码列表（排除指数）
        stock_codes = [name for name in stock_file_map.keys() if name != "sh000001"]
        factor_df, industry_codes, all_dates = prepare_factor_data(
            stock_file_map,
            fundamental_data,
            INDUSTRY_KEYWORDS,
            all_dates,
            NUM_WORKERS
        )
        strategy.set_factor_data(factor_df, industry_codes)
        strategy.set_sector_data({}, industry_codes)
        # ConceptHeat 使用概念板块（与行业映射独立，不覆盖因子选择的行业分组）
        concept_codes = _build_concept_industry_codes(stock_codes, min_stocks=30)
        if concept_codes:
            strategy.set_sector_data(concept_codes, industry_codes)
        print(f"因子模式: {factor_mode}, {len(industry_codes)} 个行业, {len(concept_codes)} 个概念")
    else:
        print(f"跳过IC计算: factor_mode={factor_mode} (fixed模式)")
        stock_codes = [name for name in stock_file_map.keys() if name != "sh000001"]
        # 使用关键词构建行业映射（prepare_factor_data 依赖INDUSTRY_KEYWORDS）
        industry_codes = build_fine_industry_map(fundamental_data, stock_codes)
        if industry_codes:
            strategy.set_industry_mapping(industry_codes)
            print(f"行业映射: {len(industry_codes)} 个行业")

    # factor_df 已由 prepare_factor_data 在工作进程中计算完成
    # stock_data_dict 不再存在 — 数据由worker从磁盘按需读取，无需清理
    import gc
    gc.collect()
    _malloc_trim()
    print("因子数据准备完成（流式加载，无 stock_data_dict 内存峰值）")

    # ===================================================================
    # Phase A: ML训练 + 因子预计算（依赖 factor_df，需在 factor_df 释放前完成）
    # 先于 Backtrader datafeed 创建，避免 reindexed 数据(~2GB) 与 factor_df(~4GB) 共存
    # ===================================================================

    # === ML预测层训练 (滚动窗口 Walk-Forward) ===
    ml_config = config.config.get('ml', {})
    _ml_model_path = None
    _ml_preds = {}
    # 检查 xgboost 可用性（避免逐 chunk 报错）
    _has_xgb = False
    try:
        import xgboost as _xgb
        _has_xgb = True
    except ImportError:
        pass
    if ml_config.get('enabled', False) and factor_df is not None and _has_xgb:
        try:
            from core.ml_predictor import MLFactorPredictor

            _train_window = ml_config.get('train_window_days', 750)  # 3年窗口
            _retrain_freq = ml_config.get('retrain_frequency', 60)   # 季度重训
            _pred_start = pd.Timestamp(ml_config.get('pred_start_date', '2021-01-01'))

            # 预测期: pred_start 之后的所有日期
            _all_dates = sorted(factor_df['date'].unique())
            _pred_dates = [d for d in _all_dates if d >= _pred_start]

            if len(_pred_dates) == 0:
                print("[ML] 无预测日期, 跳过")
            else:
                print(f"ML滚动窗口: train_window={_train_window}日 "
                      f"retrain_every={_retrain_freq}日 "
                      f"pred_dates={len(_pred_dates)}")

                # 按 retrain_frequency 分 chunk, 每 chunk 用前 train_window 日训练
                chunk_starts = list(range(0, len(_pred_dates), _retrain_freq))
                _val_ics = []
                _total_preds = 0

                for chunk_idx, chunk_start in enumerate(tqdm(chunk_starts, desc="ML滚动训练")):
                    chunk_end = min(chunk_start + _retrain_freq, len(_pred_dates))
                    chunk_dates = _pred_dates[chunk_start:chunk_end]
                    first_pred_date = chunk_dates[0]

                    # 训练集: first_pred_date 之前 train_window 日
                    train_start = first_pred_date - pd.Timedelta(days=_train_window)
                    train_mask = (factor_df['date'] >= train_start) & \
                                 (factor_df['date'] < first_pred_date)
                    train_df = factor_df[train_mask]

                    if len(train_df) < 50000:  # 最少5万样本
                        print(f"  chunk {chunk_idx}: 训练样本不足({len(train_df)}), 跳过")
                        continue

                    # 训练
                    ml_predictor = MLFactorPredictor(config.config)
                    val_ic = ml_predictor.train(train_df)
                    if val_ic is None or val_ic <= 0:
                        continue
                    _val_ics.append(val_ic)

                    # 保存最新模型
                    strategy_dir = os.path.dirname(os.path.abspath(__file__))
                    model_dir = os.path.join(strategy_dir, 'models')
                    os.makedirs(model_dir, exist_ok=True)
                    _ml_model_path = os.path.join(model_dir, 'xgb_strategy_model.json')
                    ml_predictor.save_model(_ml_model_path)

                    # 预测当前 chunk
                    for date in chunk_dates:
                        date_df = factor_df[factor_df['date'] == date]
                        if len(date_df) == 0:
                            continue
                        preds = ml_predictor.predict(date_df)
                        codes = date_df['code'].values
                        for j, code in enumerate(codes):
                            _ml_preds[(str(code).zfill(6), date)] = preds[j]
                    _total_preds += sum(1 for d in chunk_dates
                                        if d in factor_df['date'].values)

                _ml_total_preds = len(_ml_preds)
                _ml_val_ic = float(np.mean(_val_ics)) if _val_ics else 0.0
                print(f"ML滚动训练完成: {len(chunk_starts)} chunks, "
                      f"avg_IC={_ml_val_ic:.4f}, preds={_ml_total_preds:,}")
                print(f"ML模型已保存: {_ml_model_path}")

        except ImportError:
            print("[ML] xgboost未安装，跳过ML预测")
        except Exception as e:
            print(f"[ML] 训练失败: {e}")
            import traceback
            traceback.print_exc()

    # 准备信号生成用的参数
    global use_dynamic, precomputed_cache
    use_dynamic = factor_mode != 'fixed'
    stock_codes = [name for name in stock_file_map.keys() if name != "sh000001"]

    # 创建带动态因子的 SignalEngine + 预计算因子选择（如果启用）
    main_engine = None
    _rank_parquet_path = None  # 预定义，避免 use_dynamic=False 时 UnboundLocalError
    if use_dynamic:
        main_engine = SignalEngine()
        main_engine.set_factor_data(factor_df)
        main_engine.set_industry_mapping(industry_codes)
        main_engine.set_fundamental_data(fundamental_data)
        if _ml_model_path is not None and _ml_preds:
            from core.ml_predictor import MLFactorPredictor
            ml_predictor = MLFactorPredictor(config.config)
            ml_predictor.load_model(_ml_model_path)
            main_engine.set_ml_predictor(ml_predictor)
            main_engine.set_ml_predictions(_ml_preds)
        print(f"主引擎已设置动态因子数据")

        # === 截面排名缓存：从 factor_df 提取 _rank 列，写入 parquet 供 worker 按需读取 ===
        _rank_cols = [c for c in factor_df.columns if c.endswith('_rank')]
        if _rank_cols:
            strategy_dir = os.path.dirname(os.path.abspath(__file__))
            _rank_parquet_path = os.path.join(strategy_dir, 'rolling_validation_results',
                                              'factor_ranks_worker.parquet')
            factor_df[['code', 'date'] + _rank_cols].to_parquet(_rank_parquet_path, index=False)
            main_engine.set_rank_parquet(_rank_parquet_path)
            strategy.signal_engine.set_rank_parquet(_rank_parquet_path)
            print(f"截面排名缓存: {len(_rank_cols)} 个排名因子 → {_rank_parquet_path}")

        # 初始化因子库 (持久化IC评估 + 时变质量追踪)
        from core.factor_library import create_factor_library
        _factor_lib = create_factor_library()
        main_engine.dynamic_factor_selector.factor_library = _factor_lib

        # 预计算所有日期的因子选择（避免多进程中重复计算）
        print("预计算因子选择...")
        main_engine.dynamic_factor_selector.precompute_all_factor_selections(
            progress_callback=lambda curr, total: print(f"\r因子选择进度: {curr}/{total}", end="", flush=True),
            num_workers=NUM_WORKERS  # 内存已通过 _malloc_trim 控制，恢复并行
        )
        print(f"\n因子选择预计算完成，共 {len(main_engine.dynamic_factor_selector._factor_cache)} 个日期")

        # 提取预计算的缓存传递给workers
        precomputed_cache = main_engine.dynamic_factor_selector._factor_cache
        precomputed_all_dates = main_engine.dynamic_factor_selector._all_dates_cache

        # === 释放 factor_df（预计算完成，不再需要原始因子DataFrame） ===
        strategy.signal_engine.dynamic_factor_selector.factor_df = None
        main_engine.dynamic_factor_selector.factor_df = None
        main_engine.dynamic_factor_selector.industry_codes = {}
        del factor_df
        factor_df = None
        import gc
        gc.collect()
        gc.collect()
        _malloc_trim()  # 强制归还 freed pages 给 OS, 降低后续 fork 时的 RSS
        print("已释放 factor_df 内存（已归还 OS）")
    else:
        precomputed_cache = None
        precomputed_all_dates = None

    # ===================================================================
    # Phase B: 创建 Backtrader datafeeds（从磁盘重新加载，避免与 factor_df 共存）
    # factor_df 已释放，此时内存峰值最低
    # ===================================================================
    price_cols = ['open', 'high', 'low', 'close']
    for name in tqdm(stock_codes, desc="preparing datafeeds"):
        filepath = stock_file_map.get(name)
        if not filepath:
            continue
        data = pd.read_csv(filepath, parse_dates=['datetime'], dtype=_DTYPE_MAP)
        if FROMDATE:
            data = data[data['datetime'] >= FROMDATE]
        if TODATE:
            data = data[data['datetime'] <= TODATE]
        if 'datetime' in data.columns:
            data = data.set_index('datetime')
        data = data.reindex(calendar_index)
        data[price_cols] = data[price_cols].ffill()
        if 'volume' in data.columns:
            data['volume'] = data['volume'].fillna(0)
            if data['volume'].dtype != 'float64':
                data['volume'] = data['volume'].astype('float64')
        datafeed = bt.feeds.PandasData(dataname=data[price_cols + ['volume']])
        cerebro.adddata(datafeed, name=name)

    # 准备参数：传文件路径给worker（而非dict），避免内存翻倍
    stock_items = [
        (name, stock_file_map[name])
        for name in stock_codes
    ]

    # 增量写入信号CSV（而非内存中累积 ~1.3GB 的 all_signals 列表）
    strategy_dir = os.path.dirname(os.path.abspath(__file__))
    signals_output_path = os.path.join(strategy_dir, 'rolling_validation_results', 'backtest_signals.csv')
    os.makedirs(os.path.dirname(signals_output_path), exist_ok=True)
    signal_csv = open(signals_output_path, 'w', encoding='utf-8')
    signal_csv.write('code,date,buy,sell,score,pre_discount_score,factor_value,factor_name,industry,factor_quality,'
                     'chan_divergence_type,chan_divergence_strength,chan_structure_score,'
                     'chan_buy_point,chan_sell_point,signal_level,trend_type,'
                     'chan_pivot_zg,chan_pivot_zd,mom_60d,dist_ma60,max_dd_20d,vol_regime,'
                     'mtf_discount_factor,mtf_alignment_score,avg_trend_strength,'
                     'risk_vol,daily_return,volume_ratio,stroke_phase,exhaustion_risk,'
                     'gap_breakout_confirm,vol_opening_confirm,vol_opening_strength,'
                     'bom_quality_score,gate_quality,profit_declining,ma_trend_up\n')
    signal_count = [0]  # 用list实现闭包写入计数

    # 多进程并行生成信号
    print(f"多进程生成信号 ({NUM_WORKERS} workers)...")
    _worker_dyn_fail_agg = {k: 0 for k in ['total_calls', 'no_code_or_date', 'no_industry',
                                             'no_cache_data', 'no_dates', 'lookup_fail',
                                             'low_quality', 'no_factor_score']}

    # 动态因子统计
    global dynamic_factor_stats
    dynamic_factor_stats = {'hit': 0, 'miss': 0, 'factor_names': {}}

    # === 保存 ML 预测到 parquet（按code索引, worker按需读取, 避免69MB/进程pickle OOM）===
    _ml_preds_path = None
    if _ml_preds:
        strategy_dir = os.path.dirname(os.path.abspath(__file__))
        _ml_preds_path = os.path.join(strategy_dir, 'rolling_validation_results', 'ml_preds_worker.parquet')
        ml_preds_df = pd.DataFrame(
            [{'code': c, 'date': d, 'ml_pred': v} for (c, d), v in _ml_preds.items()],
            columns=['code', 'date', 'ml_pred']
        )
        ml_preds_df.to_parquet(_ml_preds_path, index=False)
        del ml_preds_df

    # 预初始化另类数据缓存（主进程单次执行，避免8个worker同时下载）
    try:
        from core.alternative_data import get_provider
        alt = get_provider()
        alt._ensure_dragon_tiger_history()
        alt.get_northbound_signal(pd.Timestamp('2024-01-02').date())
        alt.get_margin_signal(pd.Timestamp('2024-01-02').date())
    except Exception:
        pass

    ctx = multiprocessing.get_context('spawn')
    pool = ctx.Pool(
        processes=NUM_WORKERS,
        initializer=_init_worker,
        initargs=(FUNDAMENTAL_PATH, stock_codes, use_dynamic, industry_codes,
                  precomputed_cache, precomputed_all_dates, regime_df,
                  _ml_model_path, _ml_preds_path, _rank_parquet_path)
    )
    _sig_iter = pool.imap_unordered(_generate_stock_signal_worker, stock_items, chunksize=50)
    # 聚合 worker 因子选择统计
    _worker_stats_agg = {}
    _worker_bom_agg = {'total': 0, 'hit': 0, 'miss': 0, 'moat': 0, 'sum_score': 0.0,
                       'n_unique': 0, 'n_moat_codes': 0}
    for result in _sig_iter:
        code, store_data, dyn_fail, diag_data, stats_data, bom_data = (
            result if len(result) >= 6 else
            (result[0], result[1], result[2],
             result[3] if len(result) > 3 else {},
             result[4] if len(result) > 4 else {},
             result[5] if len(result) > 5 else {}))
        for k, v in dyn_fail.items():
            if k in _worker_dyn_fail_agg:
                _worker_dyn_fail_agg[k] += v
        # 合并 worker 因子选择统计（跳过列表类型字段）
        for k, v in stats_data.items():
            if isinstance(v, (int, float)):
                _worker_stats_agg[k] = _worker_stats_agg.get(k, 0) + v
        # 聚合 worker BOM 统计
        if bom_data:
            for k in ('total', 'hit', 'miss', 'moat', 'n_unique', 'n_moat_codes'):
                _worker_bom_agg[k] = _worker_bom_agg.get(k, 0) + bom_data.get(k, 0)
            _worker_bom_agg['sum_score'] = _worker_bom_agg.get('sum_score', 0.0) + bom_data.get('sum_score', 0.0)
        # 合并 worker 诊断数据到主进程
        if diag_data:
            try:
                from analysis.backtest_diagnostics import get_diagnostics
                main_diag = get_diagnostics()
                m = main_diag.metrics
                for k in ['factor_selection', 'buy_by_buy_point', 'buy_by_factor_family']:
                    if k in diag_data:
                        for kk, vv in diag_data[k].items():
                            m[k][kk] = m[k].get(kk, 0) + vv
                m['buy_signals'] = m.get('buy_signals', 0) + diag_data.get('buy_signals', 0)
                m['hard_rejects'] = m.get('hard_rejects', 0) + diag_data.get('hard_rejects', 0)
                m['gate_quality_samples'].extend(diag_data.get('gate_quality_samples', []))
                m['alt_northbound_hits'] = m.get('alt_northbound_hits', 0) + diag_data.get('alt_nb', 0)
                m['alt_margin_hits'] = m.get('alt_margin_hits', 0) + diag_data.get('alt_mg', 0)
                m['alt_dragon_tiger_hits'] = m.get('alt_dragon_tiger_hits', 0) + diag_data.get('alt_dt', 0)
            except Exception as e:
                import traceback; print(f"[ERR] " + __file__ + ":" + str(e)); traceback.print_exc()
        # 增量写入CSV（不在内存中累积 Signal 对象）
        for (c, date), sig in store_data.items():
            if hasattr(date, 'date'):
                date = date.date()
            row = (
                f'{c},{date},'
                f'{sig.buy},{sig.sell},{sig.score},{sig.pre_discount_score},'
                f'{sig.factor_value},{sig.factor_name},{sig.industry},'
                f'{getattr(sig, "factor_quality", 0.0)},'
                f'{getattr(sig, "chan_divergence_type", "")},'
                f'{getattr(sig, "chan_divergence_strength", 0.0)},'
                f'{getattr(sig, "chan_structure_score", 0.0)},'
                f'{getattr(sig, "chan_buy_point", 0)},'
                f'{getattr(sig, "chan_sell_point", 0)},'
                f'{getattr(sig, "signal_level", 0)},'
                f'{getattr(sig, "trend_type", 0)},'
                f'{getattr(sig, "chan_pivot_zg", float("nan"))},'
                f'{getattr(sig, "chan_pivot_zd", float("nan"))},'
                f'{getattr(sig, "mom_60d", 0.0)},'
                f'{getattr(sig, "dist_ma60", 0.0)},'
                f'{getattr(sig, "max_dd_20d", 0.0)},'
                f'{getattr(sig, "vol_regime", 1.0)},'
                f'{getattr(sig, "mtf_discount_factor", 1.0)},'
                f'{getattr(sig, "mtf_alignment_score", 0.0)},'
                f'{(getattr(sig, "weekly_trend_strength", 0.0) + getattr(sig, "monthly_trend_strength", 0.0)) / 2},'
                f'{getattr(sig, "risk_vol", 0.0)},'
                f'{getattr(sig, "daily_return", 0.0)},'
                f'{getattr(sig, "volume_ratio", 0.0)},'
                f'{getattr(sig, "stroke_phase", 0.0)},'
                f'{getattr(sig, "exhaustion_risk", 0.0)},'
                f'{getattr(sig, "gap_breakout_confirm", 0.0)},'
                f'{getattr(sig, "vol_opening_confirm", 0.0)},'
                f'{getattr(sig, "vol_opening_strength", 0.0)},'
                f'{getattr(sig, "bom_quality_score", 0.3)},'
                f'{getattr(sig, "_gate_quality", 0.5)},'
                f'{getattr(sig, "profit_declining", False)},'
                f'{getattr(sig, "ma_trend_up", False)}\n'
            )
            signal_csv.write(row)
            signal_count[0] += 1
            # 动态因子统计
            if sig.factor_name and sig.factor_name.startswith('DYN_'):
                dynamic_factor_stats['hit'] += 1
                fn = sig.factor_name.split('_')[1] if '_' in sig.factor_name else sig.factor_name
                dynamic_factor_stats['factor_names'][fn] = dynamic_factor_stats['factor_names'].get(fn, 0) + 1
            else:
                dynamic_factor_stats['miss'] += 1
        # 释放该股票返回的 dict（Signal 已写入 CSV，不再需要）
        store_data.clear()

    pool.close()
    pool.join()
    for k, v in _worker_dyn_fail_agg.items():
        strategy.signal_engine._dyn_fail[k] = v

    signal_csv.close()
    print(f"信号数据已保存: {signal_count[0]} 条 -> {signals_output_path}")

    # === 从CSV加载信号到 DataFrame-backed SignalStore（节省 ~800MB 内存） ===
    strategy.signal_store.finalize(signals_output_path)

    # 释放不再需要的大对象
    del stock_items
    import gc
    gc.collect()

    # 打印动态因子统计
    total = dynamic_factor_stats['hit'] + dynamic_factor_stats['miss']
    if total > 0:
        hit_rate = dynamic_factor_stats['hit'] / total * 100
        print(f"\n=== 动态因子统计 ===")
        print(f"动态因子命中: {dynamic_factor_stats['hit']:,} / {total:,} ({hit_rate:.1f}%)")
        print(f"非动态因子: {dynamic_factor_stats['miss']:,}")
        if dynamic_factor_stats['factor_names']:
            print("行业因子分布:")
            for fn, cnt in sorted(dynamic_factor_stats['factor_names'].items(), key=lambda x: -x[1])[:10]:
                print(f"  {fn}: {cnt:,}")

    # 更新策略的基本面数据加载范围（限制为实际回测池股票，节省查询内存）
    if hasattr(strategy, 'portfolio') and hasattr(strategy.portfolio, 'fundamental_data'):
        from core.fundamental import FundamentalData
        fundamental_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'data', 'stock_data', 'fundamental_data'
        )
        strategy.portfolio.fundamental_data = FundamentalData(fundamental_path + '/', stock_codes=stock_codes)
        print(f"基本面数据已限制为 {len(stock_codes)} 只股票")

    # 释放 stock_file_map + 最终清理
    stock_file_map.clear()
    del stock_file_map
    import gc
    gc.collect()
    _malloc_trim()
    print("已释放所有缓存内存")

    # 注入 worker 聚合的因子选择统计到主引擎（保留 ic_values 列表）
    if _worker_stats_agg:
        for k, v in _worker_stats_agg.items():
            if k != 'ic_values':
                strategy.signal_engine._stats[k] = v

    # 注入 worker 聚合的 BOM 统计到主引擎
    if _worker_bom_agg:
        bd = strategy.signal_engine._bom_diag
        bd['total'] = _worker_bom_agg.get('total', 0)
        bd['hit'] = _worker_bom_agg.get('hit', 0)
        bd['miss'] = _worker_bom_agg.get('miss', 0)
        bd['moat'] = _worker_bom_agg.get('moat', 0)
        bd['sum_score'] = _worker_bom_agg.get('sum_score', 0.0)
        bd['_unique_codes'] = set(range(_worker_bom_agg.get('n_unique', 0)))  # placeholder
        bd['_moat_codes'] = set(range(_worker_bom_agg.get('n_moat_codes', 0)))

    # 打印因子选择统计
    strategy.signal_engine.print_factor_stats()
    strategy.signal_engine.print_bom_stats()

    # === 回测诊断报告 ===
    try:
        from analysis.backtest_diagnostics import get_diagnostics
        diag = get_diagnostics()
        if _ml_total_preds > 0:
            diag.record_ml(total=_ml_total_preds, active=_ml_total_preds, val_ic=_ml_val_ic)
        diag.print_report()
        diag.save()
    except Exception as e:
        print(f"[诊断] 报告生成失败: {e}")

class BacktraderExecution(bt.Strategy):
    params = dict(
        real_strategy=None,
    )

    # 数据质量过滤参数
    MIN_PRICE = 2.0  # 最低价格限制（前复权后），避免低价股异常
    MIN_VOLUME = 100  # 最低成交量，过滤停牌股

    def __init__(self):
        self.universe = [d._name for d in self.datas]
        self.count = 0
        self.current_rebalance_period = REBALANCE_DAYS  # 动态周期，默认20天
        self.orders_list = defaultdict(list)
        self.last_date = None
        self.cost = defaultdict(list)
        self.portfolio_selections = []  # 记录每期选股结果
        self._prev_total_value = None  # 用于计算每日收益率
        # ST/涨跌停配置
        st_config = config.get('stock_pool', {})
        self._filter_st = st_config.get('filter_st', True)  # 默认过滤ST
        self._st_codes = getattr(self.p.real_strategy, 'st_codes', set()) if self.p.real_strategy else set()
        # 涨跌停数据
        self._limit_data = _LIMIT_DATA  # 引用全局预计算数据

        # === 成交率追踪 ===
        self._fill_stats = {
            'buy_attempted': 0, 'buy_filled': 0, 'buy_partial': 0, 'buy_rejected': 0,
            'sell_attempted': 0, 'sell_filled': 0, 'sell_partial': 0, 'sell_rejected': 0,
            'buy_limit_up_skip': 0, 'sell_limit_down_skip': 0,
            'sell_tplus1_blocked': 0, 'buy_cash_insufficient': 0,
        }

        # === T+1结算 ===
        cm_config = config.get('cost_model', {})
        self._tplus1 = cm_config.get('t_plus_1_enabled', True)
        self._today_buys = set()

        # === 冲击成本 ===
        self._impact_enabled = cm_config.get('impact_cost_enabled', False)
        self._impact_base = cm_config.get('impact_cost_base', 0.0003)
        self._impact_exp = cm_config.get('impact_cost_exponent', 0.5)
        self._impact_min = cm_config.get('min_impact_cost', 0.00005)
        self._adv_lookback = cm_config.get('adv_lookback', 20)
        self._adv_cache = {}  # 由全局_ADV_CACHE填充

    def _is_tradable(self, d):
        """检查股票是否可交易（非停牌、价格正常、非ST/科创板）"""
        price = d.close[0]
        volume = d.volume[0] if hasattr(d, 'volume') else 1

        # 1. 价格必须为正
        if price is None or math.isnan(price) or price <= 0:
            return False

        # 2. 价格不能太低（前复权后）- 避免低价股异常放大收益
        if price < self.MIN_PRICE:
            return False

        # 3. 成交量必须大于0（非停牌）
        if volume is None or math.isnan(volume) or volume < self.MIN_VOLUME:
            return False

        # 4. ST股票过滤（默认从strategy获取，可通过配置关闭）
        if self._filter_st and d._name in self._st_codes:
            return False

        return True

    def _is_limit_up(self, code, date):
        """检查股票当日是否涨停（无法买入）"""
        if hasattr(date, 'date'):
            date = date.date()
        return _LIMIT_DATA.get((code, date)) == 'up'

    def _is_limit_down(self, code, date):
        """检查股票当日是否跌停（无法卖出）"""
        if hasattr(date, 'date'):
            date = date.date()
        return _LIMIT_DATA.get((code, date)) == 'down'

    def next(self):
        self._today_buys = set()  # T+1: 每日重置当日买入集合
        if self.last_date is not None and self.last_date in self.orders_list:
            for order in self.orders_list[self.last_date]:
                self.cancel(order)
            del self.orders_list[self.last_date]
        self.count += 1
        date = self.datas[0].datetime.date(0)
        self.last_date = date

        prices = {}
        tradable_universe = []  # 可交易的股票池
        for d in self.datas:
            if not self._is_tradable(d):
                continue
            price = d.close[0]
            prices[d._name] = price
            tradable_universe.append(d._name)

        current_positions = {
            d._name: self.getposition(d).size * prices[d._name]
            for d in self.datas
            if d._name in prices and self.getposition(d).size != 0
        }

        rebalance = False
        if self.count >= self.current_rebalance_period:
            self.count = 1
            rebalance = True
        target = self.p.real_strategy.generate_positions(
            date=date,
            universe=tradable_universe,
            current_positions=current_positions,
            cash=self.broker.getcash(),
            prices=prices,
            cost=self.cost,
        )

        # 记录选股结果
        if rebalance:
            selection = self.p.real_strategy.portfolio.last_selection
            for s in selection:
                self.portfolio_selections.append({
                    'date': date,
                    'code': s['code'],
                    'score': s['score'],
                    'weight': s['weight'],
                    'industry': s.get('industry', ''),
                })

        active_codes = set(target.keys()) | set(current_positions.keys())
        for d in self.datas:
            code = d._name
            if code not in active_codes:
                continue

            # 再次检查可交易性（买入时必须可交易，卖出可以放宽）
            if not self._is_tradable(d) and code not in current_positions:
                continue

            price = d.close[0]
            if price is None or math.isnan(price) or price <= 0:
                continue

            pos = self.getposition(d)
            current_value = pos.size * price
            target_value = target.get(code, 0.0)

            diff_value = target_value - current_value

            # 忽略极小调整
            if abs(diff_value) < price * 100:
                continue

            raw = diff_value / price / 100
            # A 股最小 100 股
            size = max(int(raw), 1) * 100 if raw > 0 else min(int(raw), -1) * 100

            if size > 0:
                # 涨停检查：无法买入
                if self._is_limit_up(code, date):
                    self._fill_stats['buy_attempted'] += 1
                    self._fill_stats['buy_limit_up_skip'] += 1
                    continue
                # 冲击成本：从现金中预留
                impact = self._estimate_impact(code, size, price)
                impact_value = impact * size * price
                effective_cash = self.broker.getcash() - impact_value
                max_affordable = int(max(0, effective_cash) / price / 100) * 100
                if max_affordable < 100:
                    self._fill_stats['buy_attempted'] += 1
                    self._fill_stats['buy_cash_insufficient'] += 1
                    continue
                size = min(size, max_affordable)
                self._fill_stats['buy_attempted'] += 1
                if size < abs(raw) * 100 * 0.5:
                    self._fill_stats['buy_partial'] += 1
                order = self.buy(data=d, size=size)
                self.orders_list[date].append(order)
                self._today_buys.add(code)
            elif size < 0:
                # T+1: 当日买入不可卖出
                if self._tplus1 and code in self._today_buys:
                    self._fill_stats['sell_attempted'] += 1
                    self._fill_stats['sell_tplus1_blocked'] += 1
                    continue
                # 跌停检查：无法卖出
                if self._is_limit_down(code, date):
                    self._fill_stats['sell_attempted'] += 1
                    self._fill_stats['sell_limit_down_skip'] += 1
                    continue
                self._fill_stats['sell_attempted'] += 1
                order = self.sell(data=d, size=abs(size))
                self.orders_list[date].append(order)

        # 计算并记录每日收益率（供组合层波动率控制使用）
        total_value = self.broker.getvalue()
        if self._prev_total_value is not None and self._prev_total_value > 0:
            daily_return = (total_value - self._prev_total_value) / self._prev_total_value
            self.p.real_strategy.portfolio.update_returns(daily_return)
        self._prev_total_value = total_value

    def _estimate_impact(self, code: str, size: int, price: float) -> float:
        """估算冲击成本（交易量/日均量的平方根缩放）

        注意：向量化路径已替代 BT 路径，此方法暂未使用。
        如需启用 BT 路径，需先填充 self._adv_cache（从 datafeeds 预计算日均量）。
        """
        if not self._impact_enabled:
            return 0.0
        adv = self._adv_cache.get(code, 0)
        if adv <= 0:
            return 0.0
        participation = (abs(size) * price) / max(adv * price, 1)
        impact = self._impact_base * (participation ** self._impact_exp)
        return max(impact, self._impact_min)

    def print_fill_stats(self):
        """回测结束时输出成交率统计"""
        s = self._fill_stats
        print("\n" + "=" * 60)
        print("成交率统计 (Fill Rate)")
        print("=" * 60)
        ba = max(s['buy_attempted'], 1)
        sa = max(s['sell_attempted'], 1)
        print("买入: {}/{} = {:.1f}%  (涨停跳过={}, 现金不足={}, 部分成交={})".format(
            s['buy_filled'], s['buy_attempted'], s['buy_filled']/ba*100,
            s['buy_limit_up_skip'], s['buy_cash_insufficient'], s['buy_partial']))
        print("卖出: {}/{} = {:.1f}%  (跌停跳过={}, T+1拦截={})".format(
            s['sell_filled'], s['sell_attempted'], s['sell_filled']/sa*100,
            s['sell_limit_down_skip'], s['sell_tplus1_blocked']))

    def notify_order(self, order):
        # 未被处理的订单
        if order.status in [order.Submitted, order.Accepted]:
            return
        # 已经处理的订单
        if order.status in [order.Completed, order.Canceled, order.Margin]:
            if order.status == order.Completed:
                if order.isbuy():
                    self._fill_stats['buy_filled'] += 1
                if order.isbuy():
                    cost = self.cost[order.data._name]
                    if len(cost) == 0:
                        cost = [0, 0.0]
                    cost[1] = (cost[0] * cost[1] + order.executed.size * order.executed.price) / (cost[0] + order.executed.size)
                    cost[0] = cost[0] + order.executed.size
                    self.cost[order.data._name] = cost
                else:
                    cost = self.cost[order.data._name]
                    cost[0] = cost[0] + order.executed.size
                    if cost[0] == 0:
                        del self.cost[order.data._name]
                    else:
                        self.cost[order.data._name] = cost
            date = datetime.date.fromordinal(int(order.executed.dt))
            if order.isbuy():
                print(f'BUY EXECUTED, date {date}, ref: {order.ref}，Price: {order.executed.price}, '
                      f'Cost: {order.executed.value}, Comm {order.executed.comm}, Size: {order.executed.size}, Stock: {order.data._name}')
            else: # Sell
                print(f'SELL EXECUTED, date {date}, ref: {order.ref}, Price: {order.executed.price}, '
                      f'Cost: {order.executed.value}, Comm {order.executed.comm}, Size: {order.executed.size}, Stock: {order.data._name}')

def _vectorized_backtest(strategy, fundamental_data, fromdate, todate, initial_cash=100000):
    """向量化回测 — 替换 backtrader 逐 bar 循环。策略逻辑完全不变。"""
    import numpy as np
    from tqdm import tqdm

    # 1. 构建价格矩阵
    stock_codes = sorted([f.replace('_qfq.csv', '') for f in os.listdir(DATA_PATH)
                         if f.endswith('_qfq.csv') and f != 'sh000001_qfq.csv' and not f.startswith('._')])
    calendar = pd.bdate_range(start=fromdate, end=todate)
    n_dates = len(calendar)
    code_to_idx = {c: i for i, c in enumerate(stock_codes)}
    idx_to_code = {i: c for i, c in enumerate(stock_codes)}

    print(f"构建价格矩阵: {n_dates}天 × {len(stock_codes)}股 ...")
    close_px = np.full((n_dates, len(stock_codes)), np.nan, dtype=np.float32)
    volume_m = np.zeros_like(close_px)
    tradable = np.zeros_like(close_px, dtype=bool)

    for j, code in enumerate(tqdm(stock_codes, desc="loading")):
        fp = os.path.join(DATA_PATH, f'{code}_qfq.csv')
        if not os.path.exists(fp):
            continue
        df = pd.read_csv(fp, parse_dates=['datetime'], index_col='datetime',
                        usecols=['datetime', 'close', 'volume'])
        df = df[df.index >= pd.Timestamp(fromdate)]
        df = df[df.index <= pd.Timestamp(todate)]
        df = df.reindex(calendar)
        close_px[:, j] = df['close'].ffill().values.astype(np.float32)
        volume_m[:, j] = df['volume'].fillna(0).values.astype(np.float32)
    del df

    # 预计算滚动ADV矩阵（20日回溯，shift(1)排除当日，无前视偏差）
    _vol_df = pd.DataFrame(volume_m, index=calendar, columns=stock_codes)
    _adv_matrix = _vol_df.rolling(20, min_periods=5).mean().shift(1).values
    _adv_matrix = np.nan_to_num(_adv_matrix, nan=0.0)

    # 冲击成本参数
    cm_config = config.get('cost_model', {})
    impact_enabled = cm_config.get('impact_cost_enabled', True)
    impact_base = cm_config.get('impact_cost_base', 0.0003)
    impact_exp = cm_config.get('impact_cost_exponent', 0.5)
    impact_min = cm_config.get('min_impact_cost', 0.00005)

    def _impact(adv, size, price):
        if not impact_enabled or adv <= 0:
            return 0.0
        participation = abs(size) / max(adv, 1)
        return max(impact_base * (participation ** impact_exp), impact_min)

    # 2. 预计算可交易矩阵（含流动性过滤：日均成交额>500万）
    daily_value = close_px * volume_m  # 每日成交额
    # 20日滚动均值: 用 pandas rolling (沿 time axis=0)
    dv = pd.DataFrame(daily_value, index=calendar, columns=stock_codes)
    avg_daily_value = dv.rolling(20, min_periods=5).mean().values
    tradable = (~np.isnan(close_px)) & (close_px > 2.0) & (volume_m > 100) & (avg_daily_value > 5e6)
    # ST 过滤: 逐日检查（默认不过滤，除非 fundamental_data 可用）
    if fundamental_data is not None and hasattr(fundamental_data, 'is_st'):
        for i, d in enumerate(calendar):
            for j, code in enumerate(stock_codes):
                if tradable[i, j]:
                    try:
                        if fundamental_data.is_st(code, d.date()):
                            tradable[i, j] = False
                    except Exception as e:
                        import traceback; print(f"[ERR] " + __file__ + ":" + str(e)); traceback.print_exc()

    # 3. 主循环
    print("运行回测 ...")
    cash = float(initial_cash)
    positions = np.zeros(len(stock_codes), dtype=np.int32)
    nav = np.full(n_dates, np.nan)
    daily_ret = np.full(n_dates - 1, np.nan)
    selections = []
    cost_tracker = {}

    # 成交统计
    _fill_stats = {'buy_attempted': 0, 'buy_filled': 0, 'buy_partial': 0,
                   'sell_attempted': 0, 'sell_filled': 0, 'sell_partial': 0,
                   'buy_limit_up_skip': 0, 'sell_limit_down_skip': 0,
                   'sell_tplus1_blocked': 0, 'buy_cash_insufficient': 0}

    # T+1 结算：当日买入的股票不可卖出
    tplus1_enabled = config.get('cost_model', {}).get('t_plus_1_enabled', True)
    _today_buys = set()

    for i in tqdm(range(n_dates), desc="backtest"):
        date = calendar[i].date()
        px_today = close_px[i]
        ok = tradable[i]
        _today_buys.clear()  # T+1: 每日重置

        # 获取当日市场状态 (用于分市场统计)
        _regime_today = 0
        if strategy.index_data is not None:
            _idx_row = strategy.index_data[strategy.index_data['datetime'].dt.date == date]
            if not _idx_row.empty:
                _regime_today = int(_idx_row['regime'].values[0])

        # 当前持仓市值
        pos_value = float(np.dot(positions.astype(np.float64), np.nan_to_num(px_today, 0)))
        nav[i] = cash + pos_value
        if i > 0:
            daily_ret[i - 1] = (nav[i] - nav[i - 1]) / max(nav[i - 1], 1.0)
            if i >= 60:  # 市场状态检测需要60根bar
                plog.log_regime_execution(_regime_today, daily_ret[i - 1], nav[i])

        # 事件驱动：每天调用 generate_positions（止损检查 + 信号买入 + 补票）
        universe = [c for c in stock_codes if ok[code_to_idx[c]]]
        prices = {c: float(px_today[code_to_idx[c]]) for c in universe}
        cur_pos = {}
        for j in range(len(stock_codes)):
            if positions[j] > 0 and ok[j]:
                cur_pos[stock_codes[j]] = float(positions[j]) * float(px_today[j])

        try:
            target = strategy.generate_positions(
                date=date, universe=universe, current_positions=cur_pos,
                cash=cash, prices=prices, cost=cost_tracker)
        except Exception as e:
            import traceback
            print(f" [选股异常] date={date}: {e}")
            traceback.print_exc()
            target = {}

        # 记录选股
        for c, tv in target.items():
            if c in prices and c not in cur_pos:
                sig_score = 0.0
                sig = strategy.signal_store.get(c, date)
                if sig is not None:
                    sig_score = float(getattr(sig, 'score', 0.0))
                selections.append({'date': date, 'code': c,
                                   'weight': tv / max(nav[i], 1.0),
                                   'score': sig_score})

        # 事件驱动: 不强制卖出不在目标的持仓，只按 target 调整（仅止损卖出 + 新买入）
        for code, tv in target.items():
            j = code_to_idx.get(code)
            if j is None or not ok[j]:
                continue
            px = float(px_today[j])
            # 涨停当日排队失败惩罚: 若当日涨停(≥9.5%), 买入价上浮3%模拟无法成交的排队成本
            buy_px = px
            if i > 0:
                prev_px = close_px[i-1, j]
                if not np.isnan(prev_px) and prev_px > 0 and (px / prev_px - 1) > 0.095:
                    buy_px = px * 1.03
            target_shares = int(tv / buy_px / 100) * 100
            curr_shares = int(positions[j])
            diff = target_shares - curr_shares
            if diff >= 100:  # 买入
                _fill_stats['buy_attempted'] += 1
                impact = _impact(_adv_matrix[i, j], diff, buy_px)
                cost = diff * buy_px * (1.0 + COMMISSION + impact)
                if cost <= cash:
                    cash -= cost
                    positions[j] = target_shares
                    _fill_stats['buy_filled'] += 1
                    if tplus1_enabled:
                        _today_buys.add(code)
                elif diff < target_shares:
                    _fill_stats['buy_cash_insufficient'] += 1
            elif diff <= -100:  # 卖出（仅止损卖出，不强制换仓）
                _fill_stats['sell_attempted'] += 1
                if tplus1_enabled and code in _today_buys:
                    _fill_stats['sell_tplus1_blocked'] += 1
                    continue
                sell_px = px
                if i > 0:
                    prev_px = close_px[i-1, j]
                    if not np.isnan(prev_px) and prev_px > 0 and (px / prev_px - 1) < -0.095:
                        sell_px = px * 0.97
                impact = _impact(_adv_matrix[i, j], abs(diff), sell_px)
                cash += abs(diff) * sell_px * (1.0 - COMMISSION - STAMP_TAX - impact)
                positions[j] = target_shares
                _fill_stats['sell_filled'] += 1

        # 跟踪回撤
        running_peak = np.maximum.accumulate(nav[:i+1])
        running_dd = (nav[i] - running_peak[-1]) / max(running_peak[-1], 1.0)
        # 记录每日状态 (每20个交易日采样一次)
        if i % 20 == 0 or i == n_dates - 1:
            pos_count = int(np.sum(positions > 0))
            exposure_val = float(np.dot(positions.astype(np.float64), np.nan_to_num(px_today, 0))) / max(nav[i], 1.0)
            plog.log_backtest_daily(calendar[i].date(), nav[i], float(running_dd),
                                    pos_count, exposure_val, float(daily_ret[i-1]) if i > 0 else 0.0)

    # 记录执行成交汇总
    plog.log_execution(
        buy_filled=_fill_stats['buy_filled'],
        sell_filled=_fill_stats['sell_filled'],
        buy_attempted=_fill_stats['buy_attempted'],
        sell_attempted=_fill_stats['sell_attempted'],
        buy_limit_skip=_fill_stats['buy_limit_up_skip'],
        sell_limit_skip=_fill_stats['sell_limit_down_skip'],
        cash_insufficient=_fill_stats['buy_cash_insufficient'],
        tplus1_blocked=_fill_stats['sell_tplus1_blocked'],
    )

    # 4. 计算指标
    final_nav = nav[-1]
    total_return = final_nav / initial_cash - 1.0
    daily_ret_clean = np.nan_to_num(daily_ret, 0.0)
    mean_daily = np.mean(daily_ret_clean)
    std_daily = np.std(daily_ret_clean)
    sharpe = mean_daily / max(std_daily, 1e-10) * np.sqrt(252)

    peak = np.maximum.accumulate(nav)
    dd = (nav - peak) / peak
    max_dd = float(np.min(dd))

    annual_rets = {}
    for yr in sorted(set(d.year for d in calendar)):
        mask = np.array([d.year == yr for d in calendar])
        yr_nav = nav[mask]
        if len(yr_nav) > 1:
            annual_rets[yr] = float(yr_nav[-1] / yr_nav[0] - 1.0)

    # 成交统计: 每个调仓日的买卖操作 ≈ 选股数 × 2
    n_trades = len(selections) * 2

    print(f"\n=== 向量化回测结果 ===")
    print(f"初始资金: {initial_cash:,.0f}")
    print(f"最终净值: {final_nav:,.0f}  (总收益 {total_return*100:.2f}%)")
    for yr, r in sorted(annual_rets.items()):
        print(f"  {yr}: {r*100:.2f}%")
    print(f"Sharpe: {sharpe:.4f}")
    print(f"最大回撤: {abs(max_dd)*100:.2f}%")
    print(f"选股记录: {len(selections)} 次")
    print(f"日收益序列: {len(daily_ret_clean)} 点, 均值={mean_daily:.6f}, std={std_daily:.6f}")
    # 成交率统计
    ba = max(_fill_stats['buy_attempted'], 1)
    sa = max(_fill_stats['sell_attempted'], 1)
    print(f"成交率: 买入{_fill_stats['buy_filled']}/{_fill_stats['buy_attempted']}={_fill_stats['buy_filled']/ba*100:.1f}%"
          f" 卖出{_fill_stats['sell_filled']}/{_fill_stats['sell_attempted']}={_fill_stats['sell_filled']/sa*100:.1f}%"
          f" 资金不足={_fill_stats['buy_cash_insufficient']}")
    # 记录到诊断模块
    try:
        from analysis.backtest_diagnostics import get_diagnostics
        get_diagnostics().record_fill_stats(_fill_stats)
    except Exception as e:
        import traceback; print(f"[ERR] " + __file__ + ":" + str(e)); traceback.print_exc()

    return {'nav': nav, 'daily_returns': daily_ret_clean, 'sharpe': sharpe,
            'max_drawdown': max_dd, 'annual_returns': annual_rets,
            'final_value': final_nav, 'selections': selections,
            'n_dates': n_dates}


if __name__ == "__main__":
    # === 日志输出：同时输出到 stdout 和文件，方便监控进度 ===
    import sys as _sys
    from datetime import datetime as _dt
    _log_dir = os.path.join(os.path.dirname(__file__), 'logs')
    os.makedirs(_log_dir, exist_ok=True)
    _log_path = os.path.join(_log_dir, f'bt_execution_{_dt.now().strftime("%Y%m%d_%H%M%S")}.log')
    class _Tee:
        def __init__(self, *files): self.files = files
        def write(self, data):
            for f in self.files: f.write(data); f.flush()
        def flush(self):
            for f in self.files: f.flush()
    _log_f = open(_log_path, 'w', encoding='utf-8')
    _sys.stdout = _Tee(_sys.stdout, _log_f)
    print(f"回测日志: {_log_path}")

    # 加载基本面数据 (支持 _qfq.csv 和 _hfq.csv)
    stock_pool_enabled = config.get('stock_pool.enabled', True)
    stock_codes = []
    for f in os.listdir(DATA_PATH):
        if f.startswith('._'):
            continue
        if f.endswith('_qfq.csv') and f != 'sh000001_qfq.csv':
            stock_codes.append(f.replace('_qfq.csv', ''))
        elif f.endswith('_hfq.csv') and f != 'sh000001_hfq.csv':
            stock_codes.append(f.replace('_hfq.csv', ''))

    # 股票池过滤
    if stock_pool_enabled:
        stock_pool = get_stock_pool()
        stock_codes = [c for c in stock_codes if c in stock_pool]
        print(f"基本面数据加载(股票池): {len(stock_codes)} 只")
    else:
        print(f"基本面数据加载(全市场): {len(stock_codes)} 只")

    # 科创板(688xxx) 不再排除 — 与 add_data_and_signal 策略一致
    fundamental_data = FundamentalData(FUNDAMENTAL_PATH, stock_codes)

    # 初始化情绪分析编排器
    sentiment_orch = None
    sentiment_enabled = config.get('industry_sentiment.enabled', False)
    if sentiment_enabled and SENTIMENT_AVAILABLE:
        print("[Sentiment] 初始化情绪分析模块...")
        try:
            sentiment_orch = SentimentOrchestrator(config, backtest_mode=True)
        except Exception as e:
            print(f"[Sentiment] 初始化异常 (将跳过情绪调整): {e}")
            sentiment_orch = None

    cerebro = bt.Cerebro()
    strategy = Strategy(
        init_cash=CASH,
        fundamental_data=fundamental_data,
        sentiment_orchestrator=sentiment_orch,
    )

    # 强制 fresh generation
    import glob as _glob
    _existing = _glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         'rolling_validation_results', 'backtest_signals.csv'))
    if _existing:
        os.remove(_existing[0])
    add_data_and_signal(cerebro, strategy, fundamental_data)

    # ======  向量化回测引擎 (替代 backtrader 逐 bar 循环) ======
    # 释放 Backtrader datafeeds（向量化回测不依赖 cerebro），降低 OOM 风险
    del cerebro
    gc.collect()
    _malloc_trim()
    result = _vectorized_backtest(strategy, fundamental_data, FROMDATE, TODATE, CASH)

    # 保存选股结果
    if result['selections']:
        selections_df = pd.DataFrame(result['selections'])
        strategy_dir = os.path.dirname(os.path.abspath(__file__))
        selections_path = os.path.join(strategy_dir, 'rolling_validation_results', 'portfolio_selections.csv')
        os.makedirs(os.path.dirname(selections_path), exist_ok=True)
        selections_df.to_csv(selections_path, index=False)
        print(f"\n选股结果已保存: {len(selections_df)} 条 -> {selections_path}")

    # === 全链路追踪报告 ===
    plog.report()

    # === 妖股观察名单: 从信号中筛选潜在妖股候选 ===
    try:
        strategy_dir = os.path.dirname(os.path.abspath(__file__))
        signals_output_path = os.path.join(strategy_dir, 'rolling_validation_results', 'backtest_signals.csv')
        signals_df = pd.read_csv(signals_output_path, low_memory=False)
        yg = signals_df.copy()
        yg['volume_ratio'] = pd.to_numeric(yg['volume_ratio'], errors='coerce')
        yg['daily_return'] = pd.to_numeric(yg['daily_return'], errors='coerce')
        yg['score'] = pd.to_numeric(yg['score'], errors='coerce')
        # buy列修复: CSV中为字符串 'True'/'False', 需转bool
        yg['buy_bool'] = yg['buy'].astype(str).str.lower().map({'true': True, 'false': False})

        # 妖股特征: 高量比(>0.3, arctan压缩后≈2x原始量比) + 高日收益(>3%) + 正score, 但未触发买入
        yg_candidates = yg[
            (yg['volume_ratio'] > 0.3) &
            (yg['daily_return'] > 0.03) &
            (yg['score'] > 0) &
            (yg['buy_bool'] == False)
        ].copy()

        if len(yg_candidates) > 0:
            yg_candidates = yg_candidates.sort_values(['date', 'score'], ascending=[True, False])
            yg_path = os.path.join(strategy_dir, 'rolling_validation_results', 'yaogu_watchlist.csv')
            yg_candidates[['date', 'code', 'score', 'volume_ratio', 'daily_return',
                            'factor_name', 'industry']].to_csv(yg_path, index=False)
            print(f"妖股观察名单已保存: {len(yg_candidates)} 条 -> {yg_path}")
        del yg, signals_df
    except Exception as e:
        print(f"[妖股名单] 生成失败: {e}")
        import traceback; traceback.print_exc()
