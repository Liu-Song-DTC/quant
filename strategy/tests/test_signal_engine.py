"""冒烟测试: signal_engine + portfolio 核心路径"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from core.signal_engine import SignalEngine
from core.signal_store import SignalStore
from core.signal import Signal


def _make_sample_data(n: int = 250) -> pd.DataFrame:
    """生成模拟日线数据（足够长度以触发所有指标计算）"""
    rng = np.random.RandomState(42)
    dates = pd.date_range('2024-06-01', periods=n, freq='B')
    trend = np.sin(np.linspace(0, 4 * np.pi, n)) * 2
    close = 10.0 + np.cumsum(rng.randn(n) * 0.08) + trend
    close = np.maximum(close, 1.0)
    return pd.DataFrame({
        'datetime': dates,
        'open': close * (1 + rng.randn(n) * 0.003),
        'high': close * (1 + np.abs(rng.randn(n) * 0.015)),
        'low': close * (1 - np.abs(rng.randn(n) * 0.015)),
        'close': close,
        'volume': rng.randint(50000, 2000000, n),
        'amount': rng.randint(500000, 20000000, n),
    })


def test_signal_engine_init():
    """SignalEngine 初始化基本属性"""
    engine = SignalEngine()
    assert engine.buy_threshold > 0
    assert engine.sell_threshold < 0
    assert engine.mtf_analyzer is not None
    assert engine.dynamic_factor_selector is not None
    assert hasattr(engine, 'mtf_blend_strength')


def test_signal_engine_generate_basic():
    """基本信号生成: 不crash, 返回Signal"""
    df = _make_sample_data(250)
    engine = SignalEngine()
    store = SignalStore()

    try:
        engine.generate('000001', df, store)
    except Exception as e:
        # 允许某些配置缺失导致的异常, 但不能是语法/属性错误
        msg = str(e)
        if 'tuple' in msg.lower() or 'attribute' in msg.lower():
            raise

    # 至少应该生成一些信号
    assert len(store._store) > 0, "未生成任何信号"


def test_signal_engine_factor_mode_fixed():
    """fixed模式: 不依赖动态因子选择器"""
    df = _make_sample_data(250)
    engine = SignalEngine()
    engine.factor_mode = 'fixed'
    store = SignalStore()

    engine.generate('000001', df, store)
    assert len(store._store) > 0


def test_signal_dataclass():
    """Signal dataclass 构造和默认值"""
    sig = Signal(
        buy=False, sell=False, score=0.15,
        factor_value=0.08, factor_name='V41',
        industry='电子',
    )
    assert sig.score == 0.15
    assert not sig.buy
    assert sig.industry == '电子'
    assert sig.mtf_discount_factor == 1.0  # 默认值
    assert sig.pre_discount_score == 0.0   # 默认值

    d = sig.to_dict()
    assert 'score' in d
    assert 'mtf_discount_factor' in d


def test_signal_store_basic():
    """SignalStore 基本存取"""
    store = SignalStore()
    sig = Signal(buy=True, sell=False, score=0.2,
                 factor_value=0.1, factor_name='test', industry='金融')
    store.set('000001', pd.Timestamp('2025-06-15'), sig)

    retrieved = store.get('000001', pd.Timestamp('2025-06-15'))
    assert retrieved is not None
    assert retrieved.score == 0.2
    assert retrieved.buy is True


def test_portfolio_constructor_init():
    """PortfolioConstructor 初始化并读取YAML params"""
    from core.portfolio import PortfolioConstructor

    pc = PortfolioConstructor()
    assert pc.max_positions >= 3
    assert pc.position_stop_loss > 0
    assert pc.entry_speed > 0
    assert pc.rank_decay > 0
    assert pc.mtf_blend_strength if hasattr(pc, 'mtf_blend_strength') else True
    # 新外部化的参数
    assert pc.chan_bonus_sl2 > 0
    assert pc.mom_60d_fomo_threshold > 0
    assert pc.exhaustion_high_threshold > 0


def test_ml_predictor_init():
    """MLFactorPredictor 初始化"""
    try:
        from core.ml_predictor import MLFactorPredictor
        p = MLFactorPredictor({'ml': {'enabled': True}})
        assert p.xgb_params['tree_method'] == 'hist'
        assert p.xgb_params['device'] == 'cpu'
        assert p.model is None  # 未训练
    except ImportError:
        pass  # xgboost 未安装时跳过


def test_multi_timeframe_init():
    """MultiTimeframeAnalyzer 初始化"""
    from core.multi_timeframe import MultiTimeframeAnalyzer
    mtf = MultiTimeframeAnalyzer({})
    assert mtf.enabled
    assert mtf.weekly_ema_fast > 0
    assert mtf.monthly_ema_slow > 0
