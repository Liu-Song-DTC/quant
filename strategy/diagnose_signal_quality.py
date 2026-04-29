# diagnose_signal_quality.py
"""
诊断当前信号的质量 - 分别测试技术面和基本面
"""
import pandas as pd
import numpy as np
import sys
import os
sys.path.insert(0, '.')

from core.diagnostics import FactorDiagnostics, MarketRegimeDiagnostics
from core.market_regime_detector import MarketRegimeDetector
from core.signal_store import SignalStore
from core.signal_engine import SignalEngine
from core.fundamental import FundamentalData


def load_fundamental_data(stock_codes):
    """加载基本面数据"""
    FUNDAMENTAL_PATH = '../data/stock_data/fundamental_data/'
    return FundamentalData(FUNDAMENTAL_PATH, stock_codes)


if __name__ == "__main__":
    print("加载数据...")
    DATA_PATH = '../data/stock_data/backtrader_data/'

    # 加载指数
    index_data = pd.read_csv(DATA_PATH + 'sh000001_qfq.csv', parse_dates=['datetime'])

    # 生成市场状态
    print("生成市场状态...")
    detector = MarketRegimeDetector()
    index_with_regime = detector.generate(index_data)

    # 诊断市场状态
    print("\n" + "="*60)
    print("1. 市场状态诊断")
    print("="*60)
    regime_report = MarketRegimeDiagnostics.diagnose_regime(
        pd.Series(index_with_regime['regime'].values),
        index_data,
        look_ahead=20
    )
    regime_report.print_report()

    # 取前30只股票
    all_items = [f for f in os.listdir(DATA_PATH) if f.endswith('_qfq.csv') and f != 'sh000001_qfq.csv'][:30]

    # ====== 测试1：纯技术面信号 ======
    print("\n" + "="*60)
    print("测试1: 纯技术面信号")
    print("="*60)

    signal_engine1 = SignalEngine()  # 不加载基本面
    signal_store1 = SignalStore()

    for item in all_items:
        code = item.replace('_qfq.csv', '')
        try:
            df = pd.read_csv(DATA_PATH + item, parse_dates=['datetime'])
            if len(df) < 120:
                continue
            signal_engine1.generate(code, df, signal_store1)
        except Exception as e:
            continue

    tech_signals = []
    tech_returns = []

    for item in all_items:
        code = item.replace('_qfq.csv', '')
        try:
            df = pd.read_csv(DATA_PATH + item, parse_dates=['datetime'])
            for i, row in df.iterrows():
                if i < len(df) - 20:
                    date = row['datetime'].date() if hasattr(row['datetime'], 'date') else pd.to_datetime(row['datetime']).date()
                    sig = signal_store1.get(code, date)
                    # 使用有效的因子值（过滤异常值）
                    if sig and sig.factor_value is not None:
                        # 过滤异常动量值和收益
                        if abs(sig.factor_value) < 1.0:
                            future_price = df.iloc[i+20]['close'] if i+20 < len(df) else row['close']
                            ret = future_price / row['close'] - 1
                            if abs(ret) < 1.0:  # 过滤异常收益
                                tech_signals.append(sig.factor_value)
                                tech_returns.append(ret)
        except:
            continue

    # 打印分数分布
    if len(tech_signals) > 0:
        import numpy as np
        s = np.array(tech_signals)
        print(f"分数分布: min={s.min():.3f}, max={s.max():.3f}, mean={s.mean():.3f}")

    if len(tech_signals) > 100:
        sig_report = FactorDiagnostics.diagnose_factor(
            pd.Series(tech_signals), pd.Series(tech_returns), "技术面信号")
        sig_report.print_report()

        buy_signals = (pd.Series(tech_signals) > 0.30).astype(int)
        buy_report = FactorDiagnostics.diagnose_signal(
            buy_signals, pd.Series(tech_returns), "技术面买入信号")
        buy_report.print_report()

    # ====== 测试2：技术面+基本面 ======
    print("\n" + "="*60)
    print("测试2: 技术面+基本面信号")
    print("="*60)

    signal_engine2 = SignalEngine()
    signal_store2 = SignalStore()

    stock_codes = [f.replace('_qfq.csv', '') for f in all_items]
    fundamental_data = load_fundamental_data(stock_codes)
    signal_engine2.set_fundamental_data(fundamental_data)

    combined_signals = []
    combined_returns = []

    for item in all_items:
        code = item.replace('_qfq.csv', '')
        try:
            df = pd.read_csv(DATA_PATH + item, parse_dates=['datetime'])
            if len(df) < 120:
                continue
            signal_engine2.generate(code, df, signal_store2)
        except Exception as e:
            continue

    for item in all_items:
        code = item.replace('_qfq.csv', '')
        try:
            df = pd.read_csv(DATA_PATH + item, parse_dates=['datetime'])
            for i, row in df.iterrows():
                if i < len(df) - 20:
                    date = row['datetime'].date() if hasattr(row['datetime'], 'date') else pd.to_datetime(row['datetime']).date()
                    sig = signal_store2.get(code, date)
                    if sig:
                        combined_signals.append(sig.score)
                        future_price = df.iloc[i+20]['close'] if i+20 < len(df) else row['close']
                        ret = future_price / row['close'] - 1
                        combined_returns.append(ret)
        except:
            continue

    if len(combined_signals) > 100:
        sig_report2 = FactorDiagnostics.diagnose_factor(
            pd.Series(combined_signals), pd.Series(combined_returns), "技术面+基本面信号")
        sig_report2.print_report()

        buy_signals2 = (pd.Series(combined_signals) > 0.40).astype(int)
        buy_report2 = FactorDiagnostics.diagnose_signal(
            buy_signals2, pd.Series(combined_returns), "技术面+基本面买入信号")
        buy_report2.print_report()

    # ====== 对比结果 ======
    print("\n" + "="*60)
    print("对比结论")
    print("="*60)
    tech_ic = np.corrcoef(tech_signals, tech_returns)[0, 1] * 100 if len(tech_signals) > 10 else 0
    comb_ic = np.corrcoef(combined_signals, combined_returns)[0, 1] * 100 if len(combined_signals) > 10 else 0
    print(f"技术面 IC: {tech_ic:.2f}%")
    print(f"技术面+基本面 IC: {comb_ic:.2f}%")
    print(f"基本面提升: {comb_ic - tech_ic:+.2f}%")
