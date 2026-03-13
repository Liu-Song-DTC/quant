# diagnose_signal_quality.py
"""
诊断当前信号的质量
"""
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '.')

from core.diagnostics import FactorDiagnostics, MarketRegimeDiagnostics
from core.market_regime_detector import MarketRegimeDetector
from core.signal_store import SignalStore
from core.signal_engine import SignalEngine


def load_stock_data():
    """加载股票数据"""
    DATA_PATH = '../data/stock_data/backtrader_data/'
    all_items = list(set(os.listdir(DATA_PATH)) - {'sh000001_qfq.csv'})[:50]  # 取50只股票

    stock_data = {}
    dates = set()

    for item in all_items:
        try:
            df = pd.read_csv(DATA_PATH + item, parse_dates=['datetime'])
            code = item.replace('_qfq.csv', '')
            stock_data[code] = df
            dates.update(df['datetime'])
        except:
            continue

    calendar_index = pd.DatetimeIndex(sorted(dates))
    return stock_data, calendar_index


if __name__ == "__main__":
    import os

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

    # 生成信号
    print("\n生成交易信号...")
    signal_engine = SignalEngine()
    signal_store = SignalStore()

    # 取前20只股票
    all_items = [f for f in os.listdir(DATA_PATH) if f.endswith('_qfq.csv') and f != 'sh000001_qfq.csv'][:20]

    all_signals = []
    all_returns = []

    for item in all_items:
        code = item.replace('_qfq.csv', '')
        try:
            df = pd.read_csv(DATA_PATH + item, parse_dates=['datetime'])
            if len(df) < 120:
                continue

            signal_engine.generate(code, df, signal_store)

            # 收集信号和收益
            for i, row in df.iterrows():
                if i < len(df) - 20:
                    date = row['datetime'].date() if hasattr(row['datetime'], 'date') else pd.to_datetime(row['datetime']).date()
                    sig = signal_store.get(code, date)
                    if sig:
                        all_signals.append(sig.score)
                        # 20天后收益
                        future_price = df.iloc[i+20]['close'] if i+20 < len(df) else row['close']
                        ret = future_price / row['close'] - 1
                        all_returns.append(ret)
        except Exception as e:
            continue

    if len(all_signals) > 100:
        # 诊断信号质量
        print("\n" + "="*60)
        print("2. 信号质量诊断")
        print("="*60)

        sig_series = pd.Series(all_signals)
        ret_series = pd.Series(all_returns)

        sig_report = FactorDiagnostics.diagnose_factor(sig_series, ret_series, "技术面信号")
        sig_report.print_report()

        # 诊断买入信号
        print("\n" + "="*60)
        print("3. 买入信号诊断")
        print("="*60)

        buy_signals = (sig_series > 0.25).astype(int)
        buy_report = FactorDiagnostics.diagnose_signal(buy_signals, ret_series, "买入信号")
        buy_report.print_report()
