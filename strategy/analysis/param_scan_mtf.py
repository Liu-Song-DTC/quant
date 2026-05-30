"""
多时间框架折扣参数网格扫描 (快速版)

对 multi_timeframe.discount 参数进行网格搜索,
找出能使信号准确率最大化的最优参数组合。

原理:
    1. 读取 backtest_signals.csv (需包含 pre_discount_score + MTF 成分字段)
    2. 计算 future_ret (通过读取 stock price data)
    3. 对每组参数, 纯数学重算折扣 → new_score → buy/sell
    4. 计算 buy_accuracy, signal_IC, Sharpe 代理指标
    5. 输出最优参数

用法:
    # 先跑一次回测 (生成包含 MTF 字段的 backtest_signals.csv)
    cd strategy && python bt_execution.py

    # 然后跑参数扫描
    python strategy/analysis/param_scan_mtf.py

    # 预览参数空间
    python strategy/analysis/param_scan_mtf.py --dry-run
"""

import sys
import os
import yaml
import itertools
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from tqdm import tqdm

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "strategy"))


def load_config():
    config_path = ROOT / "strategy/config/factor_config.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def compute_future_ret(signals_df: pd.DataFrame, data_path: str,
                       forward_days: int = 20) -> pd.Series:
    """计算每个信号的 forward return

    对 buy 信号所在股票, 读取价格数据计算 forward_period 后的收益率。
    只对 buy=True 或 score>0 的信号计算, 其余返回 NaN。
    """
    # 只对有可能成为 buy 信号的条目计算 future_ret
    relevant = signals_df[
        (signals_df['buy']) | (signals_df['pre_discount_score'] > 0.03)
    ].copy()

    if len(relevant) == 0:
        return pd.Series(np.nan, index=signals_df.index)

    # 按股票分组, 读取价格数据
    stock_dates = defaultdict(list)
    for idx, row in relevant.iterrows():
        stock_dates[row['code']].append(pd.Timestamp(row['date']))

    future_ret_map = {}

    for code, dates in tqdm(stock_dates.items(), desc="computing future_ret"):
        # 尝试多种文件命名
        filepath = None
        for suffix in ['_qfq.csv', '_hfq.csv']:
            fp = os.path.join(data_path, f'{code}{suffix}')
            if os.path.exists(fp):
                filepath = fp
                break
        if filepath is None:
            continue

        try:
            df = pd.read_csv(filepath, parse_dates=['datetime'])
            df = df.set_index('datetime').sort_index()
            close = df['close']

            for signal_date in dates:
                # 找到信号日之后 forward_days 个交易日的价格
                future_idx = close.index[close.index > signal_date]
                if len(future_idx) >= forward_days:
                    future_date = future_idx[forward_days - 1]
                    current_price = close.loc[close.index[close.index <= signal_date][-1]]
                    future_price = close.loc[future_date]
                    if current_price > 0:
                        ret = (future_price - current_price) / current_price
                        if abs(ret) < 0.5:
                            future_ret_map[(code, signal_date.date())] = ret
        except Exception:
            continue

    # 映射回原始 DataFrame 的索引
    result = pd.Series(np.nan, index=signals_df.index)
    for idx, row in signals_df.iterrows():
        key = (row['code'], pd.Timestamp(row['date']).date())
        if key in future_ret_map:
            result[idx] = future_ret_map[key]

    return result


def base_discount(alignment: float, ct: float, pt: float, full: float = 1.0) -> float:
    """计算基础折扣因子 (从 multi_timeframe.py 复制公式)"""
    if alignment >= 0.3:
        base = pt + (full - pt) * (alignment - 0.3) / 0.7
    elif alignment <= -0.3:
        base = ct + (pt - ct) * (alignment + 1.0) / 0.7
    else:
        base = 0.60 + 0.18 * (alignment + 0.3) / 0.6
    return float(np.clip(base, ct, full))


def recompute_discount(df: pd.DataFrame, ct: float, pt: float, wt: float,
                       full: float = 1.0) -> np.ndarray:
    """根据新参数重新计算折扣因子

    使用保存的 MTF 成分字段精确重算:
      new_discount = new_base * new_trend_adj * other_adj

    其中 other_adj 从原始 discount_factor 中反推, 保留形态/支撑阻力调整。
    """
    alignment = df['mtf_alignment_score'].values
    avg_strength = df['avg_trend_strength'].values
    old_discount = df['mtf_discount_factor'].values

    # 旧参数: ct=0.50, pt=0.72, full=1.0, wt=0.85
    old_ct, old_pt, old_wt = 0.50, 0.72, 0.85

    n = len(df)
    new_discount = np.ones(n)

    for i in range(n):
        align = alignment[i]
        strength = avg_strength[i]
        old_d = old_discount[i]

        if np.isnan(align) or np.isnan(old_d) or old_d <= 0:
            new_discount[i] = 1.0
            continue

        # 旧折扣的组成部分
        old_base = base_discount(align, old_ct, old_pt, full)
        old_trend = old_wt if strength < 0.3 else 1.0
        old_combined = old_base * old_trend

        # 其他调整 (形态+支撑阻力, 约为1.0)
        other_adj = old_d / max(old_combined, 0.01)
        other_adj = np.clip(other_adj, 0.85, 1.15)

        # 新折扣
        new_base = base_discount(align, ct, pt, full)
        new_trend = wt if strength < 0.3 else 1.0
        new_d = new_base * new_trend * other_adj
        new_discount[i] = float(np.clip(new_d, 0.30, 1.10))

    return new_discount


def scan_parameters(dry_run: bool = False):
    """主扫描函数"""
    config = load_config()

    # 路径配置
    data_path = config.get('paths', {}).get('data', str(ROOT / 'data/stock_data/backtrader_data/'))
    signals_path = ROOT / "strategy/rolling_validation_results/backtest_signals.csv"

    if not signals_path.exists():
        print(f"ERROR: 信号文件不存在: {signals_path}")
        print("请先运行回测: cd strategy && python bt_execution.py")
        sys.exit(1)

    # 参数网格定义 (满足 counter_trend < partial 约束)
    param_grid = {
        'counter_trend': [0.30, 0.40, 0.50, 0.60],
        'partial': [0.55, 0.65, 0.72, 0.80, 0.90],
        'weak_trend': [0.75, 0.80, 0.85, 0.90, 0.95],
    }

    # 生成满足约束的组合
    combinations = []
    for ct in param_grid['counter_trend']:
        for pt in param_grid['partial']:
            if pt <= ct:
                continue
            for wt in param_grid['weak_trend']:
                combinations.append((ct, pt, wt))

    print(f"{'=' * 60}")
    print(f"  MTF 折扣参数网格扫描 (快速信号验证)")
    print(f"  参数空间: {len(combinations)} 个组合")
    print(f"  counter_trend: {param_grid['counter_trend']}")
    print(f"  partial:       {param_grid['partial']}")
    print(f"  weak_trend:    {param_grid['weak_trend']}")
    print(f"  约束: counter_trend < partial")
    print(f"{'=' * 60}")

    if dry_run:
        print("\n[Dry Run] 参数组合预览:")
        for i, combo in enumerate(combinations):
            print(f"  {i+1}. counter_trend={combo[0]}, partial={combo[1]}, weak_trend={combo[2]}")
        print(f"\n总计: {len(combinations)} 个有效组合")
        return

    # ── Step 1: 读取信号数据 ──
    print(f"\n[1/3] 读取信号数据...")
    df = pd.read_csv(signals_path, parse_dates=['date'])
    print(f"  总信号: {len(df):,} 条")
    print(f"  buy信号: {df['buy'].sum():,} 条")

    # 检查新字段是否存在
    required_cols = ['pre_discount_score', 'mtf_discount_factor',
                     'mtf_alignment_score', 'avg_trend_strength']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"\nERROR: 信号文件缺少必需字段: {missing}")
        print("请重新运行回测生成包含 MTF 字段的信号文件。")
        sys.exit(1)

    # ── Step 2: 计算 future_ret ──
    print(f"\n[2/3] 计算 forward returns...")
    df['future_ret'] = compute_future_ret(df, data_path, forward_days=20)
    valid = df['future_ret'].notna()
    print(f"  有效 future_ret: {valid.sum():,} / {len(df):,} ({valid.mean()*100:.1f}%)")

    if valid.sum() < 100:
        print("ERROR: 有效信号太少, 无法可靠评估。请检查数据路径。")
        sys.exit(1)

    # 只对有 future_ret 的 buy 相关信号评估
    eval_df = df[valid].copy()
    # 只看 pre_discount_score > 0 的 (买入方向)
    buy_related = eval_df[eval_df['pre_discount_score'] > 0.03].copy()
    print(f"  买入方向信号 (pre_score>0.03): {len(buy_related):,}")

    if len(buy_related) < 50:
        print("ERROR: 买入方向信号太少。")
        sys.exit(1)

    # ── Step 3: 网格搜索 ──
    print(f"\n[3/3] 网格搜索 ({len(combinations)} 个组合)...")

    results = []
    best_sharpe = -999
    best_params = None

    buy_threshold = config.get('signal', {}).get('buy_threshold', 0.15)

    for i, (ct, pt, wt) in enumerate(combinations):
        new_discount = recompute_discount(buy_related, ct, pt, wt)
        new_score = buy_related['pre_discount_score'].values * new_discount

        # 新 buy 判定: score > buy_threshold
        new_buy = new_score > buy_threshold

        # 度量 1: Buy Accuracy (买入后上涨的概率)
        if new_buy.sum() >= 10:
            buy_ret = buy_related['future_ret'].values[new_buy]
            accuracy = (buy_ret > 0).mean()
            avg_ret = buy_ret.mean()
        else:
            accuracy = np.nan
            avg_ret = np.nan

        # 度量 2: Signal IC (new_score 与 future_ret 的秩相关)
        future_arr = buy_related['future_ret'].values
        if len(new_score) >= 20:
            try:
                from scipy.stats import spearmanr
                ic, _ = spearmanr(new_score, future_arr)
            except Exception:
                ic = np.corrcoef(new_score, future_arr)[0, 1] if len(new_score) > 1 else 0
        else:
            ic = 0

        # 度量 3: Sharpe 代理 (avg_ret / std_ret for buy signals)
        if new_buy.sum() >= 10:
            buy_rets = buy_related['future_ret'].values[new_buy]
            sharpe_proxy = buy_rets.mean() / (buy_rets.std() + 1e-10)
        else:
            sharpe_proxy = np.nan

        results.append({
            'counter_trend': ct,
            'partial': pt,
            'weak_trend': wt,
            'n_buy': int(new_buy.sum()),
            'buy_accuracy': round(accuracy * 100, 2) if not np.isnan(accuracy) else np.nan,
            'avg_return': round(avg_ret * 100, 3) if not np.isnan(avg_ret) else np.nan,
            'signal_ic': round(ic, 4),
            'sharpe_proxy': round(sharpe_proxy, 4) if not np.isnan(sharpe_proxy) else np.nan,
        })

        if not np.isnan(sharpe_proxy) and sharpe_proxy > best_sharpe:
            best_sharpe = sharpe_proxy
            best_params = (ct, pt, wt)

        if (i + 1) % 20 == 0:
            print(f"  进度: {i+1}/{len(combinations)}")

    # ── 输出结果 ──
    print(f"\n{'=' * 60}")
    print(f"  扫描完成")
    print(f"{'=' * 60}")

    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values('sharpe_proxy', ascending=False, na_position='last')

    print(f"\nTop 15 参数组合 (按 Sharpe 代理):")
    print(result_df.head(15).to_string(index=False))

    if best_params:
        print(f"\n最优参数: counter_trend={best_params[0]}, "
              f"partial={best_params[1]}, weak_trend={best_params[2]}")
        print(f"最优 Sharpe 代理: {best_sharpe}")

    # 找出 buy_accuracy 最高的
    best_acc = result_df.sort_values('buy_accuracy', ascending=False, na_position='last').iloc[0]
    print(f"\n最高准确率: {best_acc['buy_accuracy']:.2f}% "
          f"(counter_trend={best_acc['counter_trend']}, "
          f"partial={best_acc['partial']}, weak_trend={best_acc['weak_trend']})")

    # 保存结果
    output_path = ROOT / f"strategy/analysis/mtf_param_scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    result_df.to_csv(output_path, index=False)
    print(f"\n结果已保存至: {output_path}")

    # 打印推荐配置
    if best_params:
        print(f"\n推荐 YAML 配置:")
        print(f"  multi_timeframe:")
        print(f"    discount:")
        print(f"      counter_trend: {best_params[0]}")
        print(f"      partial: {best_params[1]}")
        print(f"      weak_trend: {best_params[2]}")

        # 自动更新配置
        update = input(f"\n是否自动更新 factor_config.yaml? [y/N]: ").strip().lower()
        if update == 'y':
            config_path = ROOT / "strategy/config/factor_config.yaml"
            with open(config_path, 'r', encoding='utf-8') as f:
                cfg = yaml.safe_load(f)
            cfg['multi_timeframe']['discount']['counter_trend'] = best_params[0]
            cfg['multi_timeframe']['discount']['partial'] = best_params[1]
            cfg['multi_timeframe']['discount']['weak_trend'] = best_params[2]
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False)
            print(f"配置已更新: {config_path}")

    return result_df, best_params


if __name__ == "__main__":
    dry_run = '--dry-run' in sys.argv or '-n' in sys.argv
    scan_parameters(dry_run=dry_run)
