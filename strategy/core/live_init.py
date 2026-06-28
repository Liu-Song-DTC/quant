# core/live_init.py
"""
实盘初始化 — 与回测使用完全相同的逻辑, 保证等价性。

用法:
    from core.live_init import init_live_engine

    engine = init_live_engine(
        stock_file_map, fundamental_data,
        industry_codes=None,  # 可选, 自动从 fundamental_data 构建
    )
    # 之后每个交易日:
    #   engine.generate(code, market_data, signal_store, latest_only=True)
    #   engine.dynamic_factor_selector.extend_to_date(today)

关键原则: 所有路径与 bt_execution.add_data_and_signal 保持一致,
修改回测时务必同步更新此文件。
"""

import os
import pandas as pd
from typing import Dict, List, Optional

from .signal_engine import SignalEngine
from .dynamic_factor_selector import init_live_factor_cache
from .config_loader import load_config
from .fundamental import FundamentalData


def init_live_engine(
    stock_file_map: Dict[str, str],
    fundamental_data: FundamentalData,
    industry_codes: Optional[Dict[str, List[str]]] = None,
    num_workers: int = 4,
) -> SignalEngine:
    """初始化实盘 SignalEngine — 与回测 bt_execution.add_data_and_signal 等价。

    执行顺序:
      1. 准备 factor_df (与回测相同的 prepare_factor_data)
      2. 预计算全局 IC → 存入 _factor_cache
      3. 设置 engine 的因子/行业/MR/factor_library

    Args:
        stock_file_map: {code: filepath} 历史日线数据
        fundamental_data: FundamentalData 实例
        industry_codes: 概念板块映射 (可选, 不传则从 fundamental_data 构建)
        num_workers: IC预计算并行度

    Returns:
        已初始化的 SignalEngine, 可直接用于实盘信号生成
    """
    from .factor_preparer import prepare_factor_data
    from .industry_mapping import INDUSTRY_KEYWORDS, build_fine_industry_map

    config = load_config()
    factor_mode = config.get('factor_mode', 'both')
    stock_codes = [name for name in stock_file_map.keys() if name != "sh000001"]

    # Step 1: 准备 factor_df (与回测完全相同的函数调用)
    if factor_mode != 'fixed':
        print(f"[实盘] 准备因子数据 (factor_mode={factor_mode})...")
        all_dates = sorted(
            pd.to_datetime(pd.read_csv(list(stock_file_map.values())[0])['datetime']).tolist()
        )
        factor_df, industry_codes_computed, all_dates = prepare_factor_data(
            stock_file_map, fundamental_data, INDUSTRY_KEYWORDS, all_dates, num_workers
        )
        if industry_codes is None:
            industry_codes = industry_codes_computed
    else:
        factor_df = None
        if industry_codes is None:
            industry_codes = build_fine_industry_map(fundamental_data, stock_codes)

    # Step 2: 创建 engine 并设置因子数据
    engine = SignalEngine()
    engine.set_fundamental_data(fundamental_data)

    if factor_df is not None and factor_mode != 'fixed':
        # 初始化全局IC缓存 (与回测 precompute_all_factor_selections 相同逻辑)
        selector = init_live_factor_cache(factor_df, industry_codes, num_workers=num_workers)
        engine.dynamic_factor_selector = selector

        # 初始化因子库 (时变质量追踪)
        from .factor_library import create_factor_library
        engine.dynamic_factor_selector.factor_library = create_factor_library()

        # ML模型 (如果已训练)
        _ml_model_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'models', 'xgb_strategy_model.json'
        )
        if os.path.exists(_ml_model_path):
            try:
                from .ml_predictor import MLFactorPredictor
                ml_predictor = MLFactorPredictor(config.config)
                ml_predictor.load_model(_ml_model_path)
                engine.set_ml_predictor(ml_predictor)
                print(f"[实盘] ML模型已加载: {_ml_model_path}")
            except Exception as e:
                print(f"[实盘] ML模型加载失败: {e}")

        print(f"[实盘] 引擎初始化完成: IC缓存={len(selector._factor_cache)}个日期, "
              f"factor_mode={factor_mode}, {len(industry_codes or {})}个行业")
    else:
        engine.set_industry_mapping(industry_codes or {})
        print(f"[实盘] 引擎初始化完成: factor_mode=fixed, {len(industry_codes or {})}个行业")

    return engine


def update_live_engine(
    engine: SignalEngine,
    new_stock_data: Dict[str, pd.DataFrame],
) -> None:
    """每日更新: 用新数据扩展 factor_df 并更新IC缓存。

    与回测的 extend_to_date 使用相同的 _compute_date_chunk 逻辑。

    Args:
        engine: 已初始化的 SignalEngine
        new_stock_data: {code: DataFrame} 新增的日线数据
    """
    from .factor_calculator import calculate_indicators, compute_composite_factors

    selector = engine.dynamic_factor_selector
    if selector is None or selector.factor_df is None:
        print("[实盘] 跳过更新: factor_df 不存在")
        return

    # 计算新日期的因子值 (与 prepare_factor_data 内 worker 逻辑一致)
    factor_mode = load_config().get('factor_mode', 'both')
    if factor_mode == 'fixed':
        selector.extend_to_date(
            pd.Timestamp.now().date(),
            None,
        )
        return

    latest_date = selector.factor_df['date'].max()
    new_rows = []
    today = pd.Timestamp.now().date()

    for code, df in new_stock_data.items():
        if len(df) < 60:
            continue
        indicators = calculate_indicators(
            df['close'].values, df['high'].values, df['low'].values, df['volume'].values
        )
        composites = compute_composite_factors(indicators, len(df) - 1)
        row = {'code': code, 'date': today}
        for k, v in composites.items():
            if isinstance(v, (int, float)):
                row[k] = v
        new_rows.append(row)

    if new_rows:
        new_df = pd.DataFrame(new_rows)
        selector.extend_to_date(today, new_df)
        print(f"[实盘] 已更新: {len(new_rows)} 只股票, 日期={today}")
    else:
        print("[实盘] 无新数据")
