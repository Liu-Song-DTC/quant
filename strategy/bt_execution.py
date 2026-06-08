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
import ctypes

from core.strategy import Strategy
from core.fundamental import FundamentalData
from core.signal_engine import SignalEngine
from core.factor_preparer import prepare_factor_data
from core.signal_store import SignalStore
from core.config_loader import load_config
from core.monitor import monitor, get_logger
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
_ST_CODES = set()  # ST股票代码集合
_ST_DATE_RANGES = {}  # {code: [(start_date, end_date), ...]} ST期间

# === 小说优化：动态调仓周期 ===
DYNAMIC_REBALANCE_CONFIG = config.get('dynamic_rebalance', {})
DYNAMIC_REBALANCE_ENABLED = DYNAMIC_REBALANCE_CONFIG.get('enabled', True)
REBALANCE_BULL = DYNAMIC_REBALANCE_CONFIG.get('bull_period', 30)
REBALANCE_NEUTRAL = DYNAMIC_REBALANCE_CONFIG.get('neutral_period', 20)
REBALANCE_BEAR = DYNAMIC_REBALANCE_CONFIG.get('bear_period', 15)
NUM_WORKERS = config.get('backtest.num_workers', 8)

def _malloc_trim(pad=0):
    """将 Python 已释放但未归还 OS 的内存归还给内核（Linux only）。

    在 gc.collect() 之后调用，可显著降低进程 RSS，避免 fork 时子进程继承
    inflated 内存地址空间导致 OOM。
    """
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(ctypes.c_int(pad))
    except Exception:
        pass  # 非 Linux / 权限不足时静默跳过

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


def _init_worker(fundamental_data, stock_codes, use_dynamic, industry_codes, factor_cache, all_dates, regime_df,
                 ml_model_path=None, ml_preds=None):
    """Worker 进程初始化函数

    注意: 不传递 factor_df (巨大DataFrame) 到每个worker，预计算的 factor_cache 已包含所有因子选择结果。
    fundamental_data 由父进程 fork 共享（COW），每个 worker 仅触发按需加载的页面复制。
    """
    global _worker_engine, _worker_use_dynamic
    _worker_use_dynamic = use_dynamic

    # fork 后共享父进程的 fundamental_data（COW 零拷贝）
    _worker_engine = SignalEngine()

    if fundamental_data is not None:
        _worker_engine.set_fundamental_data(fundamental_data)

    # 设置市场状态数据（关键：用于牛市优化）
    if regime_df is not None:
        _worker_engine.set_market_regime(regime_df)

    # 使用预计算的因子选择缓存（不传递原始factor_df，节省 ~400MB/worker）
    if use_dynamic and factor_cache is not None and all_dates is not None:
        _worker_engine.set_industry_mapping(industry_codes)
        _worker_engine.dynamic_factor_selector.set_factor_cache(factor_cache, all_dates)

    # 设置ML预测（每个worker加载模型副本）
    if ml_model_path is not None and os.path.exists(ml_model_path):
        try:
            from core.ml_predictor import MLFactorPredictor
            worker_ml = MLFactorPredictor()
            worker_ml.load_model(ml_model_path)
            _worker_engine.set_ml_predictor(worker_ml)
        except Exception:
            pass  # ML加载失败不阻塞回测
    if ml_preds is not None:
        _worker_engine.set_ml_predictions(ml_preds)


def _generate_stock_signal_worker(args):
    """Worker 函数：为一个股票生成信号 — 直接从文件读取，避免 pickle 传大数据"""
    global _worker_engine, _worker_use_dynamic, _worker_diag_reported
    code, filepath = args

    try:
        engine = _worker_engine

        if engine is None:
            engine = SignalEngine()

        store = SignalStore()
        data = pd.read_csv(filepath, parse_dates=['datetime'])
        # 日期范围过滤（与主进程一致的加速优化）
        if FROMDATE:
            data = data[data['datetime'] >= FROMDATE]
        if TODATE:
            data = data[data['datetime'] <= TODATE]
        if len(data) < 60:
            del data
            return (code, {})
        engine.generate(code, data, store)

        # 释放该股票的基本面数据缓存（避免每个worker累积 ~1100 只股票的基本面数据）
        if hasattr(engine, 'fundamental_data') and engine.fundamental_data is not None:
            engine.fundamental_data.clear_stock_cache(code)

        # 清理本股票数据，避免 worker 内 DataFrame 累积
        result = (code, store._store)
        del data
        del store
        return result
    except Exception as e:
        # 返回异常信息，避免worker静默失败
        print(f"[Worker Error] {code}: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return (code, {})


def add_data_and_signal(cerebro, strategy, fundamental_data=None):
    """加载数据、生成信号、添加到cerebro。
    同时预计算涨跌停数据和ST股票集合。
    """
    global _LIMIT_DATA, _ST_CODES, _ST_DATE_RANGES
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

    # === 股票池过滤 ===
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

    # === 科创板(688xxx) 不再剔除 — 2025-2026年妖股主要集中板块 ===
    star_codes = set()
    before_excl_star = len(stock_file_map)
    print(f"科创板保留: {before_excl_star} 只 (不再排除)")

    # === ST股票识别（基于基本面数据，使用指数中间日期作为参考） ===
    _ST_CODES = set()
    _ST_DATE_RANGES = {}
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
    all_dates_for_st = sorted(calendar_index.tolist()) if calendar_index is not None else []
    ref_mid_date = pd.Timestamp(all_dates_for_st[len(all_dates_for_st) // 2]) if all_dates_for_st else None

    if fundamental_data is not None and ref_mid_date is not None:
        try:
            st_candidates = [c for c in stock_file_map.keys() if c != 'sh000001']
            for code in tqdm(st_candidates, desc="ST detection"):
                if fundamental_data.is_st(code, ref_mid_date):
                    _ST_CODES.add(code)
            print(f"ST股票: {len(_ST_CODES)} 只")
            if _ST_CODES:
                strategy.st_codes = _ST_CODES
        except Exception as e:
            print(f"ST检测跳过: {e}")
            strategy.st_codes = set()
    else:
        strategy.st_codes = set()

    # 处理指数数据 - 生成市场状态 (Fix#6: 多指数风格检测)
    # calendar_index 已在上方ST检测阶段从 sh000001 构建
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
    # 只有当 factor_mode 不是 'fixed' 时才需要IC计算
    # reweight模式也需要IC计算（用于动态调整权重）
    if factor_mode != 'fixed':
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
        print(f"因子模式: {factor_mode}, {len(industry_codes)} 个行业")
    else:
        print(f"跳过IC计算: factor_mode={factor_mode} (fixed模式)")
        stock_codes = [name for name in stock_file_map.keys() if name != "sh000001"]
        # 构建细行业映射（128行业，替代20个大类关键词匹配）
        if fundamental_data is not None:
            industry_codes = build_fine_industry_map(fundamental_data, stock_codes, min_stocks=10)
            strategy.set_industry_mapping(industry_codes)
            print(f"细行业映射: {len(industry_codes)} 个行业")

    # factor_df 已由 prepare_factor_data 在工作进程中计算完成
    # stock_data_dict 不再存在 — 数据由worker从磁盘按需读取，无需清理
    import gc
    gc.collect()
    print("因子数据准备完成（流式加载，无 stock_data_dict 内存峰值）")

    # ===================================================================
    # Phase A: ML训练 + 因子预计算（依赖 factor_df，需在 factor_df 释放前完成）
    # 先于 Backtrader datafeed 创建，避免 reindexed 数据(~2GB) 与 factor_df(~4GB) 共存
    # ===================================================================

    # === ML预测层训练 ===
    ml_config = config.config.get('ml', {})
    _ml_model_path = None
    _ml_preds = {}
    if ml_config.get('enabled', False) and factor_df is not None:
        try:
            from core.ml_predictor import MLFactorPredictor
            print("训练XGBoost预测模型...")
            ml_predictor = MLFactorPredictor(config.config)
            val_ic = ml_predictor.train(factor_df)
            if val_ic is not None and val_ic > 0:
                strategy_dir = os.path.dirname(os.path.abspath(__file__))
                model_dir = os.path.join(strategy_dir, 'models')
                os.makedirs(model_dir, exist_ok=True)
                _ml_model_path = os.path.join(model_dir, 'xgb_strategy_model.json')
                ml_predictor.save_model(_ml_model_path)
                print(f"ML模型已保存: {_ml_model_path}")

                print("生成ML预测...")
                all_dates_sorted = sorted(factor_df['date'].unique())
                for date in tqdm(all_dates_sorted, desc="ML预测"):
                    date_df = factor_df[factor_df['date'] == date]
                    if len(date_df) == 0:
                        continue
                    preds = ml_predictor.predict(date_df)
                    codes = date_df['code'].values
                    for j, code in enumerate(codes):
                        _ml_preds[(code, date)] = preds[j]
                print(f"ML预测完成: {len(_ml_preds)} 条预测")
            else:
                print(f"[ML] 验证IC不足({val_ic}), 跳过ML预测")
        except ImportError:
            print("[ML] xgboost未安装，跳过ML预测")
        except Exception as e:
            print(f"[ML] 训练失败: {e}")
            import traceback
            traceback.print_exc()

    # 准备信号生成用的参数
    use_dynamic = factor_mode != 'fixed'
    stock_codes = [name for name in stock_file_map.keys() if name != "sh000001"]

    # 创建带动态因子的 SignalEngine + 预计算因子选择（如果启用）
    main_engine = None
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

        # 预计算所有日期的因子选择（避免多进程中重复计算）
        print("预计算因子选择...")
        main_engine.dynamic_factor_selector.precompute_all_factor_selections(
            progress_callback=lambda curr, total: print(f"\r因子选择进度: {curr}/{total}", end="", flush=True),
            num_workers=1  # 单线程避免fork时factor_df COW膨胀
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

    # === 预fork清理: 释放主进程缓存，最小化 COW 共享内存 ===
    if fundamental_data is not None:
        fundamental_data.clear_stock_cache()  # 清除所有惰性加载的基本面缓存
    if main_engine is not None:
        main_engine.set_fundamental_data(None)  # 解除 main_engine 对 fundamental_data 的引用
    del main_engine
    main_engine = None
    import gc
    gc.collect()
    print("预fork清理完成: 最小化COW共享内存")

    # 动态因子统计
    dynamic_factor_stats = {'hit': 0, 'miss': 0, 'factor_names': {}}

    ctx = multiprocessing.get_context('fork')
    with ctx.Pool(
        processes=NUM_WORKERS,
        initializer=_init_worker,
        initargs=(fundamental_data, stock_codes, use_dynamic, industry_codes, precomputed_cache, precomputed_all_dates, regime_df,
                  _ml_model_path, _ml_preds)
    ) as pool:
        for result in tqdm(
            pool.imap_unordered(_generate_stock_signal_worker, stock_items, chunksize=10),
            total=len(stock_items),
            desc="generating signals"
        ):
            code, store_data = result
            # 增量写入CSV（不在内存中累积 Signal 对象）
            for (c, date), sig in store_data.items():
                if hasattr(date, 'date'):
                    date = date.date()
                signal_csv.write(
                    f'{c},{date},{sig.buy},{sig.sell},{sig.score},{sig.pre_discount_score},'
                    f'{sig.factor_value},'
                    f'{sig.factor_name},{sig.industry},'
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
    print("已释放所有缓存内存")

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
        # === 动态调仓周期（小说：牛市捂股，熊市灵活）===
        if DYNAMIC_REBALANCE_ENABLED and self.p.real_strategy.index_data is not None:
            date_ts = pd.to_datetime(date)
            idx_row = self.p.real_strategy.index_data[
                self.p.real_strategy.index_data["datetime"].dt.date == date
            ]
            if not idx_row.empty:
                regime = int(idx_row["regime"].values[0])
                if regime == 1:
                    self.current_rebalance_period = REBALANCE_BULL
                elif regime == -1:
                    self.current_rebalance_period = REBALANCE_BEAR
                else:
                    self.current_rebalance_period = REBALANCE_NEUTRAL

        if self.count >= self.current_rebalance_period:
            self.count = 1
            rebalance = True
        target = self.p.real_strategy.generate_positions(
            date=date,
            universe=tradable_universe,  # 只传递可交易的股票
            current_positions=current_positions,
            cash=self.broker.getcash(),
            prices=prices,
            cost=self.cost,
            rebalance=rebalance,
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
                # 涨停检查：无法买入（封死涨停板时没有卖盘）
                if self._is_limit_up(code, date):
                    continue
                max_affordable = int(self.broker.getcash() / price / 100) * 100
                size = min(size, max_affordable)
                if size > 0:
                    order = self.buy(data=d, size=size)
                    self.orders_list[date].append(order)
            elif size < 0:
                # 跌停检查：无法卖出（封死跌停板时没有买盘）
                if self._is_limit_down(code, date):
                    continue
                order = self.sell(data=d, size=abs(size))  # Backtrader要求size为正
                self.orders_list[date].append(order)

        # 计算并记录每日收益率（供组合层波动率控制使用）
        total_value = self.broker.getvalue()
        if self._prev_total_value is not None and self._prev_total_value > 0:
            daily_return = (total_value - self._prev_total_value) / self._prev_total_value
            self.p.real_strategy.portfolio.update_returns(daily_return)
        self._prev_total_value = total_value

    def notify_order(self, order):
        # 未被处理的订单
        if order.status in [order.Submitted, order.Accepted]:
            return
        # 已经处理的订单
        if order.status in [order.Completed, order.Canceled, order.Margin]:
            if order.status == order.Completed:
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

    # 剔除科创板
    star_codes = {c for c in stock_codes if c.startswith('688')}
    stock_codes = [c for c in stock_codes if c not in star_codes]
    print(f"基本面数据(科创板过滤后): {len(stock_codes)} 只")

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

    add_data_and_signal(cerebro, strategy, fundamental_data)
    cerebro.broker.setcash(CASH)
    cerebro.broker.setcommission(commission=EFFECTIVE_COMM)
    cerebro.broker.set_slippage_perc(perc=PERC)

    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='pnl')  # 返回收益率时序数据
    cerebro.addanalyzer(bt.analyzers.AnnualReturn, _name='_AnnualReturn')  # 年化收益率
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='_SharpeRatio')  # 夏普比率
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='_DrawDown')  # 回撤

    cerebro.addstrategy(
        BacktraderExecution,
        real_strategy=strategy,
    )
    # 启动回测前最终GC: 清理信号生成阶段残留的临时对象
    import gc
    gc.collect()
    gc.collect()
    _malloc_trim()  # 归还所有碎片内存给 OS，确保 cerebro.run() 有充足空间
    print(f"启动回测引擎 (可用内存优化完毕)...")
    try:
        result = cerebro.run()
    except MemoryError:
        print("\n[ERROR] 内存不足！尝试减少股票数量或增大WSL2内存限制")
        import sys
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] 回测异常退出: {e}")
        import traceback
        traceback.print_exc()
        import sys
        sys.exit(1)

    # 从返回的 result 中提取回测结果
    strat = result[0]
    # 返回日度收益率序列
    daily_return = pd.Series(strat.analyzers.pnl.get_analysis())
    # 打印评价指标
    print("--------------- AnnualReturn -----------------")
    print(strat.analyzers._AnnualReturn.get_analysis())
    print("--------------- SharpeRatio -----------------")
    print(strat.analyzers._SharpeRatio.get_analysis())
    print("--------------- DrawDown -----------------")
    print(strat.analyzers._DrawDown.get_analysis())

    # 保存选股结果供验证使用
    if strat.portfolio_selections:
        selections_df = pd.DataFrame(strat.portfolio_selections)
        strategy_dir = os.path.dirname(os.path.abspath(__file__))
        selections_path = os.path.join(strategy_dir, 'rolling_validation_results', 'portfolio_selections.csv')
        os.makedirs(os.path.dirname(selections_path), exist_ok=True)
        selections_df.to_csv(selections_path, index=False)
        print(f"\n选股结果已保存: {len(selections_df)} 条 -> {selections_path}")

    # === 妖股观察名单: 从信号中筛选潜在妖股候选 ===
    try:
        strategy_dir = os.path.dirname(os.path.abspath(__file__))
        signals_output_path = os.path.join(strategy_dir, 'rolling_validation_results', 'backtest_signals.csv')
        signals_df = pd.read_csv(signals_output_path)
        yg = signals_df.copy()
        yg['volume_ratio'] = pd.to_numeric(yg['volume_ratio'], errors='coerce')
        yg['daily_return'] = pd.to_numeric(yg['daily_return'], errors='coerce')
        yg['score'] = pd.to_numeric(yg['score'], errors='coerce')

        # 妖股特征: 高量比(>2.0) + 高日收益(>3%) + 正score, 但未触发买入
        yg_candidates = yg[
            (yg['volume_ratio'] > 2.0) &
            (yg['daily_return'] > 0.03) &
            (yg['score'] > 0) &
            (yg['buy'] == False)
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

    # 打印因子选择统计
    strategy.signal_engine.print_factor_stats()
