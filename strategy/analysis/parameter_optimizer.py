#!/usr/bin/env python3
"""
Walk-Forward参数优化器

两阶段优化策略：
  Stage 1 (快速筛选): 利用已有信号数据，在信号层面评估参数组合质量（秒级）
  Stage 2 (精确验证): 对 Top-5 参数组合运行完整回测验证（分钟级）

核心思路: 因子计算/缠论分析等底层逻辑不变，只调整阈值/权重/过滤参数。
因此可以用同一份信号数据，通过改变买入/卖出阈值、排名过滤、权重分配
等参数来评估效果，无需每次重新生成信号。

使用方式:
    python analysis/parameter_optimizer.py [--stage2] [--windows 4]
"""

import sys
import os
import gc
import json
import itertools
import multiprocessing
import numpy as np
import pandas as pd
import yaml
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from tqdm import tqdm
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STRATEGY_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, STRATEGY_DIR)

from core.config_loader import load_config
from utils.utils import safe_spearmanr

# ======================== 参数搜索空间 ========================

PARAM_GRID = {
    # 信号阈值
    'signal.buy_threshold': [0.12, 0.15, 0.18, 0.22, 0.25],
    'signal.sell_threshold': [-0.20, -0.17, -0.15, -0.12, -0.10],

    # 组合构建
    'portfolio.max_single_weight': [0.08, 0.10, 0.12, 0.15, 0.18],
    'portfolio.position_stop_loss': [0.05, 0.07, 0.10, 0.12, 0.15],
    'portfolio.trailing_stop_pct': [0.10, 0.15, 0.20, 0.25],
    'portfolio.entry_speed': [0.6, 0.8, 1.0],

    # 选股过滤 (在 portfolio.selection 下)
    'selection.min_rank_pct': [0.40, 0.50, 0.55, 0.60],
    'selection.min_absolute_score': [0.10, 0.15, 0.20],
    'selection.min_confidence': [0.70, 0.80, 0.90],

    # 调仓周期
    'backtest.rebalance_days': [15, 20, 25, 30],

    # 市场敞口 (在 portfolio.fv_exposure_params 下)
    'fv_exposure.exposure_min': [0.25, 0.30, 0.40],
    'fv_exposure.exposure_max': [0.90, 0.95, 1.0],

    # 波动率控制
    'volatility_control.enabled': [True, False],
    'portfolio.target_volatility': [0.12, 0.15, 0.18],
}

# 减少组合数：从全组合(~5M)降为随机采样或因子优先级采样
MAX_STAGE1_COMBOS = 500  # Stage 1 最多评估 500 组参数
STAGE2_TOP_N = 5         # Stage 2 验证 Top 5

# Walk-Forward 窗口配置
WF_TRAIN_YEARS = 2       # 训练窗口长度（年）
WF_TEST_MONTHS = 6       # 测试窗口长度（月）
WF_MIN_WINDOWS = 3       # 最少窗口数


# ======================== 数据结构 ========================

@dataclass
class ParamResult:
    params: Dict
    buy_accuracy: float = 0.0
    sell_accuracy: float = 0.0
    signal_ic: float = 0.0
    signal_ir: float = 0.0
    win_rate: float = 0.0
    avg_return: float = 0.0
    sharpe_estimate: float = 0.0
    n_signals: int = 0
    combined_score: float = 0.0


# ======================== Stage 1: 信号级快速筛选 ========================

def _compute_signal_quality(signals_df: pd.DataFrame, params: Dict) -> ParamResult:
    """在已有信号数据上，用给定的阈值/过滤参数计算信号质量指标。

    不做完整回测，而是通过改变 buy/sell 判定阈值和选股过滤来评估。
    """
    df = signals_df.copy()
    buy_threshold = params.get('signal.buy_threshold', 0.18)
    sell_threshold = params.get('signal.sell_threshold', -0.15)
    min_score = params.get('selection.min_absolute_score', 0.15)
    min_rank_pct = params.get('selection.min_rank_pct', 0.50)
    min_conf = params.get('selection.min_confidence', 0.80)

    # 应用新的买卖阈值
    df['_buy'] = (df['score'] >= buy_threshold) & (df['score'] >= min_score)
    df['_sell'] = (df['score'] <= sell_threshold)

    buy_sigs = df[df['_buy']]
    sell_sigs = df[df['_sell']]

    n_buy = len(buy_sigs)
    n_sell = len(sell_sigs)

    result = ParamResult(params=params, n_signals=n_buy + n_sell)

    if n_buy == 0 and n_sell == 0:
        return result

    # 买入准确率
    if n_buy > 0 and 'future_ret' in buy_sigs.columns:
        result.buy_accuracy = (buy_sigs['future_ret'] > 0).mean()

    # 卖出准确率（卖出后下跌=正确）
    if n_sell > 0 and 'future_ret' in sell_sigs.columns:
        result.sell_accuracy = (sell_sigs['future_ret'] < 0).mean()

    # 信号IC：score vs future_ret
    if 'future_ret' in df.columns:
        valid = df[['score', 'future_ret']].dropna()
        if len(valid) >= 30:
            ic_raw = safe_spearmanr(valid['score'], valid['future_ret'])
            ic = ic_raw[0] if isinstance(ic_raw, (tuple, list)) else ic_raw
            if ic is not None and np.isscalar(ic) and not np.isnan(ic):
                result.signal_ic = float(ic)

    # 信号IR：按日期分组的IC稳定性
    if 'date' in df.columns and 'future_ret' in df.columns:
        ic_list = []
        for _, g in df.groupby('date'):
            v = g[['score', 'future_ret']].dropna()
            if len(v) >= 20:
                ic_v_raw = safe_spearmanr(v['score'], v['future_ret'])
                ic_v = ic_v_raw[0] if isinstance(ic_v_raw, (tuple, list)) else ic_v_raw
                if ic_v is not None and np.isscalar(ic_v) and not np.isnan(ic_v):
                    ic_list.append(ic_v)
        if ic_list:
            result.signal_ir = np.mean(ic_list) / (np.std(ic_list) + 1e-10)

    # 胜率（买入后盈利>0的比例）
    if n_buy > 0 and 'future_ret' in buy_sigs.columns:
        result.win_rate = result.buy_accuracy

    # 平均收益
    if n_buy > 0 and 'future_ret' in buy_sigs.columns:
        result.avg_return = buy_sigs['future_ret'].mean()

    # 综合评分：IC * 准确率 * log(信号数) — 平衡预测力和实用性
    result.combined_score = (
        abs(result.signal_ic) * 0.35 +
        result.buy_accuracy * 0.25 +
        result.sell_accuracy * 0.15 +
        result.win_rate * 0.15 +
        min(np.log1p(result.n_signals) / 10, 0.1)
    )

    return result


def _sample_param_combinations(grid: Dict, max_n: int = None) -> List[Dict]:
    """从参数网格中采样组合（全组合太大时用随机采样）"""
    if max_n is None:
        max_n = MAX_STAGE1_COMBOS
    keys = list(grid.keys())
    values = list(grid.values())

    total = 1
    for v in values:
        total *= len(v)
    print(f"参数空间: {total:,} 组合")

    if total <= max_n:
        # 全量枚举
        combos = []
        for combo in itertools.product(*values):
            combos.append(dict(zip(keys, combo)))
        return combos

    # 随机采样
    print(f"随机采样 {max_n} 组...")
    rng = np.random.RandomState(42)
    combos = []
    seen = set()
    while len(combos) < max_n:
        combo = tuple(rng.randint(0, len(v)) for v in values)
        if combo not in seen:
            seen.add(combo)
            combos.append({k: values[i][c] for i, (k, c) in enumerate(zip(keys, combo))})
    return combos


def stage1_signal_screening(signals_df: pd.DataFrame,
                            param_combos: List[Dict],
                            n_workers: int = 8) -> List[ParamResult]:
    """Stage 1: 并行评估所有参数组合的信号质量"""
    print(f"\n{'='*60}")
    print(f"Stage 1: 信号级快速筛选 ({len(param_combos)} 组合)")
    print(f"{'='*60}")

    results = []
    for combo in tqdm(param_combos, desc="评估参数"):
        r = _compute_signal_quality(signals_df, combo)
        results.append(r)

    results.sort(key=lambda x: -x.combined_score)
    return results


# ======================== Stage 2: 回测验证 ========================

def stage2_backtest_validation(top_params: List[Dict],
                               train_start: str,
                               train_end: str) -> List[Dict]:
    """Stage 2: 对 Top-N 参数组合运行实际回测验证

    修改 factor_config.yaml 中的参数，运行 bt_execution.py 的简化版回测。
    """
    print(f"\n{'='*60}")
    print(f"Stage 2: 回测验证 Top-{len(top_params)} 参数")
    print(f"训练期: {train_start} → {train_end}")
    print(f"{'='*60}")

    validated = []
    for i, params in enumerate(top_params):
        print(f"\n--- 验证 {i+1}/{len(top_params)} ---")
        print(f"参数: {params}")

        try:
            result = _run_mini_backtest(params, train_start, train_end)
            validated.append({**params, 'backtest_result': result})
            print(f"  Sharpe: {result.get('sharpe', 'N/A')}, "
                  f"年化收益: {result.get('annual_return', 'N/A')}")
        except Exception as e:
            print(f"  回测失败: {e}")
            validated.append({**params, 'backtest_result': {'error': str(e)}})

    validated.sort(key=lambda x: x.get('backtest_result', {}).get('sharpe', -99), reverse=True)
    return validated


def _run_mini_backtest(params: Dict, start_date: str, end_date: str) -> Dict:
    """运行迷你回测：修改config并执行backtrader回测"""
    import subprocess
    import tempfile
    import shutil

    # 备份原配置
    config_path = os.path.join(STRATEGY_DIR, 'config', 'factor_config.yaml')
    backup_path = config_path + '.bak'
    shutil.copy(config_path, backup_path)

    try:
        # 修改配置
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)

        # 应用参数
        for key_path, value in params.items():
            _set_nested(cfg, key_path, value)

        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

        # 运行回测
        bt_script = os.path.join(STRATEGY_DIR, 'bt_execution.py')
        env = os.environ.copy()
        env['PYTHONPATH'] = STRATEGY_DIR

        result = subprocess.run(
            [sys.executable, bt_script],
            capture_output=True, text=True,
            cwd=STRATEGY_DIR,
            env=env,
            timeout=600  # 10分钟超时
        )

        # 解析回测输出获取Sharpe
        sharpe = _parse_sharpe_from_output(result.stdout + result.stderr)
        return {'sharpe': sharpe, 'exit_code': result.returncode}

    except subprocess.TimeoutExpired:
        return {'error': 'timeout', 'sharpe': -99}
    except Exception as e:
        return {'error': str(e), 'sharpe': -99}
    finally:
        # 恢复原配置
        shutil.copy(backup_path, config_path)
        os.remove(backup_path)


def _set_nested(cfg: dict, key_path: str, value):
    """设置嵌套配置项，如 'portfolio.max_single_weight' → cfg['portfolio']['max_single_weight']"""
    parts = key_path.split('.')
    target = cfg
    for part in parts[:-1]:
        if part not in target:
            target[part] = {}
        target = target[part]
    target[parts[-1]] = value


def _parse_sharpe_from_output(output: str) -> float:
    """从回测输出中提取Sharpe ratio"""
    for line in output.split('\n'):
        if 'sharpe' in line.lower() or 'Sharpe Ratio' in line:
            try:
                parts = line.split()
                for i, p in enumerate(parts):
                    if 'sharpe' in p.lower():
                        val = parts[i + 1].strip(',:')
                        return float(val)
                    try:
                        if abs(float(p)) > 0.01 and abs(float(p)) < 5:
                            return float(p)
                    except ValueError:
                        continue
            except (ValueError, IndexError):
                continue
    return 0.0


# ======================== Walk-Forward 主流程 ========================

def _get_wf_windows(signals_df: pd.DataFrame, n_windows: int = WF_MIN_WINDOWS) -> List[Tuple[str, str, str, str]]:
    """生成 walk-forward 窗口 (train_start, train_end, test_start, test_end)"""
    if 'date' not in signals_df.columns:
        raise ValueError("signals_df 缺少 date 列")

    dates = pd.to_datetime(signals_df['date']).sort_values()
    min_date = dates.min()
    max_date = dates.max()
    train_days = WF_TRAIN_YEARS * 365
    test_days = WF_TEST_MONTHS * 30

    # 从数据末尾向前推，生成窗口
    windows = []
    test_end = max_date
    for _ in range(n_windows):
        test_start = test_end - pd.Timedelta(days=test_days)
        train_end = test_start - pd.Timedelta(days=1)
        train_start = train_end - pd.Timedelta(days=train_days)

        if train_start < min_date:
            train_start = min_date
        if (train_end - train_start).days < 180:
            break  # 训练期不足6个月

        windows.insert(0, (
            train_start.strftime('%Y-%m-%d'),
            train_end.strftime('%Y-%m-%d'),
            test_start.strftime('%Y-%m-%d'),
            test_end.strftime('%Y-%m-%d'),
        ))
        test_end = test_start - pd.Timedelta(days=1)

    return windows


def run_walk_forward_optimization(signals_path: str = None,
                                  n_windows: int = WF_MIN_WINDOWS,
                                  run_stage2: bool = False) -> Dict:
    """完整的 Walk-Forward 参数优化

    Args:
        signals_path: backtest_signals.csv 路径（None=自动查找）
        n_windows: walk-forward 窗口数
        run_stage2: 是否运行回测验证（耗时长）

    Returns:
        优化结果字典，含每窗口最优参数和共识推荐
    """
    # 1. 加载信号数据
    if signals_path is None:
        signals_path = os.path.join(STRATEGY_DIR, 'rolling_validation_results', 'backtest_signals.csv')

    if not os.path.exists(signals_path):
        raise FileNotFoundError(f"信号文件不存在: {signals_path}。请先运行 bt_execution.py。")

    print(f"加载信号数据: {signals_path}")
    signals_df = pd.read_csv(signals_path, parse_dates=['date'])
    print(f"信号: {len(signals_df)} 条, {signals_df['code'].nunique()} 只股票, "
          f"{signals_df['date'].nunique()} 个交易日")

    # 合并 future_ret（从 validation_results 获取）
    val_path = os.path.join(STRATEGY_DIR, 'rolling_validation_results', 'validation_results.csv')
    if os.path.exists(val_path):
        val_df = pd.read_csv(val_path, parse_dates=['date'])
        if 'future_ret' in val_df.columns:
            val_df['date'] = pd.to_datetime(val_df['date'])
            signals_df['date'] = pd.to_datetime(signals_df['date'])
            signals_df = signals_df.merge(
                val_df[['code', 'date', 'future_ret']],
                on=['code', 'date'], how='left'
            )
            print(f"已合并 future_ret: {signals_df['future_ret'].notna().sum()} 条有效")

    # 2. 生成 walk-forward 窗口
    windows = _get_wf_windows(signals_df, n_windows)
    print(f"\nWalk-Forward 窗口: {len(windows)} 个")
    for i, (tr_s, tr_e, te_s, te_e) in enumerate(windows):
        print(f"  窗口{i+1}: 训练 [{tr_s}, {tr_e}] → 测试 [{te_s}, {te_e}]")

    # 3. 采样参数组合
    param_combos = _sample_param_combinations(PARAM_GRID)

    # 4. 每个窗口运行 Stage 1 + (可选)Stage 2
    wf_results = []
    all_top_params = []

    for wi, (tr_s, tr_e, te_s, te_e) in enumerate(windows):
        print(f"\n{'='*70}")
        print(f"Walk-Forward 窗口 {wi+1}/{len(windows)}")
        print(f"训练: {tr_s} → {tr_e}  |  测试: {te_s} → {te_e}")
        print(f"{'='*70}")

        # 过滤训练期信号
        train_mask = (signals_df['date'] >= tr_s) & (signals_df['date'] <= tr_e)
        train_df = signals_df[train_mask]
        print(f"训练信号: {len(train_df)} 条")

        if len(train_df) < 1000:
            print("  跳过（数据不足）")
            continue

        # Stage 1: 信号级筛选
        results = stage1_signal_screening(train_df, param_combos)
        top5 = results[:STAGE2_TOP_N]

        print(f"\nTop-5 参数 (Stage 1 评分):")
        for i, r in enumerate(top5):
            print(f"  #{i+1}: score={r.combined_score:.4f}, IC={r.signal_ic:.4f}, "
                  f"buy_acc={r.buy_accuracy:.2%}, n_signals={r.n_signals}")

        wf_result = {
            'window': wi + 1,
            'train_period': f"{tr_s}→{tr_e}",
            'test_period': f"{te_s}→{te_e}",
            'top5_stage1': [{'params': r.params, 'score': r.combined_score} for r in top5],
        }

        if run_stage2:
            # Stage 2: 回测验证
            top_params = [r.params for r in top5]
            stage2_results = stage2_backtest_validation(top_params, tr_s, tr_e)
            wf_result['stage2_results'] = stage2_results

            # 选最优
            best = stage2_results[0]
            wf_result['best_params'] = best
            wf_result['best_sharpe'] = best.get('backtest_result', {}).get('sharpe', 0)
            all_top_params.append(best)
        else:
            # 无 Stage 2：直接用 Stage 1 第一名
            wf_result['best_params'] = top5[0].params
            all_top_params.append(top5[0].params)

        wf_results.append(wf_result)

    # 5. 汇总共识参数
    consensus = _compute_consensus(all_top_params)
    print(f"\n{'='*70}")
    print(f"Walk-Forward 参数优化完成")
    print(f"{'='*70}")
    print(f"有效窗口: {len(wf_results)}/{len(windows)}")
    print(f"\n共识参数（多窗口一致最优值）:")
    for key, stats in sorted(consensus.items()):
        if stats['stability'] >= 0.5:
            print(f"  {key}: {stats['best_value']} "
                  f"(选择率={stats['stability']:.0%}, 窗口值={stats['values']})")

    # 保存结果
    output_path = os.path.join(STRATEGY_DIR, 'analysis_results', 'param_optimization.json')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    output = {
        'consensus': {k: {'best_value': v['best_value'], 'stability': v['stability']}
                      for k, v in consensus.items()},
        'windows': wf_results,
    }
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n结果已保存: {output_path}")

    return output


def _compute_consensus(all_top_params: List[Dict]) -> Dict:
    """从各窗口最优参数中提取共识值

    对每个参数，统计各窗口的选择情况。
    数值型参数取中位数，类别型参数取众数。
    """
    if not all_top_params:
        return {}

    # 收集各窗口的参数值
    param_values = defaultdict(list)
    for params in all_top_params:
        p = params.get('params', params) if isinstance(params, dict) else params
        for key, val in p.items():
            param_values[key].append(val)

    consensus = {}
    for key, vals in param_values.items():
        n = len(vals)
        if n == 0:
            continue

        # 数值型：取中位数
        if all(isinstance(v, (int, float)) for v in vals):
            best_value = float(np.median(vals))
        elif all(isinstance(v, bool) for v in vals):
            # 布尔型：取众数
            best_value = max(set(vals), key=vals.count)
        else:
            best_value = vals[0]

        # 稳定性 = 等于best_value的窗口数 / 总窗口数（容差）
        if isinstance(best_value, (int, float)) and not isinstance(best_value, bool):
            # 数值型：±10%内算一致
            consistent = sum(1 for v in vals if abs(v - best_value) / max(abs(best_value), 1e-10) < 0.10)
            stability = consistent / n
        else:
            consistent = sum(1 for v in vals if v == best_value)
            stability = consistent / n

        consensus[key] = {
            'best_value': best_value,
            'stability': stability,
            'values': vals,
        }

    return consensus


# ======================== 应用最优参数 ========================

def apply_best_params(config_path: str = None, params: Dict = None):
    """将最优参数写入 factor_config.yaml"""
    if config_path is None:
        config_path = os.path.join(STRATEGY_DIR, 'config', 'factor_config.yaml')

    if params is None:
        # 尝试从优化结果加载
        result_path = os.path.join(STRATEGY_DIR, 'analysis_results', 'param_optimization.json')
        if not os.path.exists(result_path):
            print("无优化结果，请先运行 run_walk_forward_optimization()")
            return
        with open(result_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        params = data.get('consensus', {})
        params = {k: v['best_value'] for k, v in params.items()}

    # 备份
    import shutil
    shutil.copy(config_path, config_path + '.preopt')

    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    for key_path, value in params.items():
        _set_nested(cfg, key_path, value)

    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    print(f"已应用 {len(params)} 个优化参数到 {config_path}")
    print(f"原配置备份: {config_path}.preopt")


# ======================== 命令行入口 ========================

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Walk-Forward参数优化器')
    parser.add_argument('--stage2', action='store_true', help='运行Stage 2回测验证（耗时长）')
    parser.add_argument('--windows', type=int, default=WF_MIN_WINDOWS, help=f'Walk-forward窗口数（默认{WF_MIN_WINDOWS}）')
    parser.add_argument('--apply', action='store_true', help='优化后自动应用最优参数到config')
    parser.add_argument('--combos', type=int, default=MAX_STAGE1_COMBOS, help=f'Stage 1评估组合数（默认{MAX_STAGE1_COMBOS}）')
    args = parser.parse_args()

    MAX_STAGE1_COMBOS = args.combos

    result = run_walk_forward_optimization(
        n_windows=args.windows,
        run_stage2=args.stage2,
    )

    if args.apply:
        apply_best_params()
