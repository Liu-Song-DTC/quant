# 诊断脚本：检查信号分布
import pandas as pd
import numpy as np
import os
from core.signal_engine import SignalEngine
from core.signal_store import SignalStore

DATA_PATH = "../data/stock_data/backtrader_data/"

# 只分析少量股票
sample_stocks = ['600000', '600016', '600028', '600036', '600050']

signal_engine = SignalEngine()
signal_store = SignalStore()

# 生成信号
for item in os.listdir(DATA_PATH):
    if not item.endswith('.csv'):
        continue
    code = item[:-8]
    if code not in sample_stocks:
        continue
    if code == 'sh000001':
        continue

    df = pd.read_csv(DATA_PATH + item, parse_dates=['datetime'])
    signal_engine.generate(code, df, signal_store)

# 统计信号分布
buy_count = 0
sell_count = 0
neutral_count = 0
scores = []
sell_scores = []

for (code, date), sig in signal_store._store.items():
    if code not in sample_stocks:
        continue
    scores.append(sig.score)
    if sig.buy:
        buy_count += 1
    elif sig.sell:
        sell_count += 1
        sell_scores.append(sig.score)
    else:
        neutral_count += 1

print(f"股票: {sample_stocks}")
print(f"总信号数: {len(scores)}")
print(f"买入信号: {buy_count} ({buy_count/len(scores)*100:.1f}%)")
print(f"卖出信号: {sell_count} ({sell_count/len(scores)*100:.1f}%)")
print(f"中性信号: {neutral_count} ({neutral_count/len(scores)*100:.1f}%)")
print(f"平均分数: {np.mean(scores):.3f}")
print(f"分数>0.25: {sum(1 for s in scores if s > 0.25)}")
print(f"分数>0.30: {sum(1 for s in scores if s > 0.30)}")
print(f"分数>0.35: {sum(1 for s in scores if s > 0.35)}")
print(f"卖出信号分数: {sell_scores[:10]}")  # 前10个卖出信号


# ====================== 信号质量诊断函数 ======================

def calculate_ic_ir(signal_store, price_data: dict, look_ahead: int = 20) -> dict:
    """
    计算IC (Information Coefficient) 和 IR (Information Ratio)

    IC: 因子值与未来收益的相关系数
    IR: IC均值/IC标准差 (稳定性)
    """
    factor_values = []
    future_returns = []

    for code, df in price_data.items():
        df = df.copy()
        df['date'] = pd.to_datetime(df['datetime']).dt.date

        for idx in range(len(df) - look_ahead):
            date = df['date'].iloc[idx]
            sig = signal_store.get(code, date)

            if sig is None or sig.score == 0:
                continue

            future_price = df['close'].iloc[idx + look_ahead]
            current_price = df['close'].iloc[idx]
            ret = future_price / current_price - 1

            factor_values.append(sig.score)
            future_returns.append(ret)

    # 计算IC
    ic = np.corrcoef(factor_values, future_returns)[0, 1]

    # 分段计算IC（用于IR）
    n = len(factor_values)
    chunk_size = n // 10
    ic_list = []

    for i in range(0, n - chunk_size, chunk_size):
        chunk_factors = factor_values[i:i+chunk_size]
        chunk_returns = future_returns[i:i+chunk_size]
        if len(chunk_factors) > 10:
            ic_chunk = np.corrcoef(chunk_factors, chunk_returns)[0, 1]
            if not np.isnan(ic_chunk):
                ic_list.append(ic_chunk)

    ir = np.mean(ic_list) / np.std(ic_list) if len(ic_list) > 1 and np.std(ic_list) > 0 else 0

    return {
        'ic': ic,
        'ir': ir,
        'ic_list': ic_list,
        'sample_count': n,
    }


def diagnose_signal_comprehensive(signal_store, price_data: dict, look_ahead: int = 20) -> dict:
    """
    综合诊断信号质量

    Args:
        signal_store: SignalStore对象
        price_data: dict {code: DataFrame} 包含datetime, close列
        look_ahead: 持有期

    Returns:
        诊断结果字典
    """
    results = {
        'total_signals': 0,
        'buy_signals': 0,
        'sell_signals': 0,
        'avg_return': 0.0,
        'buy_return': 0.0,
        'sell_return': 0.0,
    }

    all_returns = []
    buy_returns = []
    sell_returns = []

    # 按市场状态分析
    regime_returns = {-1: [], 0: [], 1: []}

    for code, df in price_data.items():
        df = df.copy()
        df['date'] = pd.to_datetime(df['datetime']).dt.date

        for idx in range(len(df) - look_ahead):
            date = df['date'].iloc[idx]
            sig = signal_store.get(code, date)

            if sig is None or sig.score == 0:
                continue

            # 计算未来收益
            future_price = df['close'].iloc[idx + look_ahead]
            current_price = df['close'].iloc[idx]
            ret = future_price / current_price - 1

            all_returns.append(ret)

            if sig.buy:
                buy_returns.append(ret)
                results['buy_signals'] += 1
            if sig.sell:
                sell_returns.append(ret)
                results['sell_signals'] += 1

            # 按市场状态分组
            regime_returns[sig.risk_regime].append(ret)

    results['total_signals'] = len(all_returns)
    results['avg_return'] = np.mean(all_returns) if all_returns else 0

    if buy_returns:
        results['buy_return'] = np.mean(buy_returns)
    if sell_returns:
        results['sell_return'] = np.mean(sell_returns)

    # 市场状态分析
    results['regime_analysis'] = {}
    for regime, returns in regime_returns.items():
        if returns:
            regime_name = {-1: '熊市', 0: '震荡市', 1: '牛市'}[regime]
            results['regime_analysis'][regime_name] = {
                'count': len(returns),
                'avg_return': np.mean(returns),
                'win_rate': (np.array(returns) > 0).mean(),
            }

    # 比较原始分数和调整后分数
    raw_returns = []
    adjusted_returns = []

    for code, df in price_data.items():
        df = df.copy()
        df['date'] = pd.to_datetime(df['datetime']).dt.date

        for idx in range(len(df) - look_ahead):
            date = df['date'].iloc[idx]
            sig = signal_store.get(code, date)

            if sig is None:
                continue

            future_price = df['close'].iloc[idx + look_ahead]
            current_price = df['close'].iloc[idx]
            ret = future_price / current_price - 1

            if sig.score > 0.01:
                raw_returns.append(ret)
            if sig.adjusted_score > 0.01:
                adjusted_returns.append(ret)

    results['score_comparison'] = {
        'raw_count': len(raw_returns),
        'raw_return': np.mean(raw_returns) if raw_returns else 0,
        'adjusted_count': len(adjusted_returns),
        'adjusted_return': np.mean(adjusted_returns) if adjusted_returns else 0,
    }

    return results


def run_signal_diagnostic():
    """运行完整信号诊断"""
    import sys
    sys.path.insert(0, '.')

    from core.signal_store import SignalStore
    from core.signal_engine import SignalEngine
    from core.market_regime_detector import MarketRegimeDetector

    DATA_PATH = "../data/stock_data/backtrader_data/"

    # 加载市场状态
    print("加载市场状态数据...")
    index_data = pd.read_csv(DATA_PATH + 'sh000001_qfq.csv', parse_dates=['datetime'])
    regime_detector = MarketRegimeDetector()
    regime_df = regime_detector.generate(index_data)

    # 初始化
    signal_engine = SignalEngine()
    signal_engine.set_market_regime(regime_df)

    signal_store = SignalStore()

    # 加载股票数据 - 增加样本量
    sample_stocks = []
    for item in os.listdir(DATA_PATH):
        if not item.endswith('.csv'):
            continue
        code = item[:-8]
        if code == 'sh000001':
            continue
        sample_stocks.append(code)
        if len(sample_stocks) >= 50:  # 增加到50只
            break

    price_data = {}

    print(f"生成信号... ({len(sample_stocks)}只股票)")
    for code in sample_stocks:
        try:
            df = pd.read_csv(DATA_PATH + code + '_qfq.csv', parse_dates=['datetime'])
            price_data[code] = df
            signal_engine.generate(code, df, signal_store)
        except:
            pass

    # 诊断信号
    print("\n" + "="*70)
    print("【信号质量诊断报告】")
    print("="*70)

    results = diagnose_signal_comprehensive(signal_store, price_data, look_ahead=20)

    print(f"\n总信号数: {results['total_signals']}")
    print(f"买入信号: {results['buy_signals']}")
    print(f"卖出信号: {results['sell_signals']}")
    print(f"平均收益: {results['avg_return']:.2%}")
    print(f"买入收益: {results['buy_return']:.2%}")
    print(f"卖出收益: {results['sell_return']:.2%}")

    # IC/IR分析
    print("\n" + "="*70)
    print("【因子质量评价 - IC/IR】")
    print("="*70)

    ic_results = calculate_ic_ir(signal_store, price_data, look_ahead=20)
    print(f"\nIC (Information Coefficient): {ic_results['ic']:.2%}")
    print(f"IR (Information Ratio): {ic_results['ir']:.2f}")
    print(f"样本数: {ic_results['sample_count']}")

    # IC分布
    if ic_results['ic_list']:
        print(f"\nIC分布 (分10段):")
        for i, ic_val in enumerate(ic_results['ic_list']):
            bar = "█" * int(abs(ic_val) * 50)
            sign = "+" if ic_val > 0 else "-"
            print(f"  段{i+1}: {sign}{bar} {ic_val:.2%}")

    print("\n【按市场状态分析】")
    for regime_name, stats in results.get('regime_analysis', {}).items():
        print(f"  {regime_name}: 样本{stats['count']:4d}, 平均收益{stats['avg_return']:7.2%}, 胜率{stats['win_rate']:.1%}")

    print("\n【原始分数 vs 调整后分数】")
    comp = results.get('score_comparison', {})
    print(f"  原始分数: 样本{comp.get('raw_count', 0):4d}, 平均收益{comp.get('raw_return', 0):.2%}")
    print(f"  调整后: 样本{comp.get('adjusted_count', 0):4d}, 平均收益{comp.get('adjusted_return', 0):.2%}")

    # IC解读
    print("\n" + "="*70)
    print("【IC评价标准】")
    print("="*70)
    print("  |IC| > 3%   : 因子有效")
    print("  |IC| > 5%   : 因子效果较好")
    print("  |IC| > 10%  : 因子效果优秀")
    print("  IR > 0.5   : 因子稳定")
    print("  IR > 1.0   : 因子非常稳定")


if __name__ == "__main__":
    run_signal_diagnostic()
