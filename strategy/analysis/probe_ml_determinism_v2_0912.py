#!/usr/bin/env python3
"""2026-09-12 ML确定性探针v2: 真实密集日历 + 全23chunk滚动循环复刻 (决定性实验)

背景: 9/12冷跑(B, 082730) vs PIT基线(A, 9/11夜223431) 四指标-35.9%。证据链:
  - 数据指纹0f2ae158✓ 代码指纹619adafb✓ 池2718✓ 因子缓存同文件d814a206✓
  - A现算并保存缓存(22:44), B加载同一缓存; save→return间无变异(705→709)
  - 两run 23个chunk训练样本数逐chunk全同(203233...638726) → 日历/窗口/purge全同
  - 但验证IC几乎每chunk差第4位小数(0.0796 vs 0.0809, ...) → 微小差异级联
  - v1探针(193,988行近似窗口)跨进程4跑bit级一致 → 近似窗口上train()确定

v2: 完全复刻bt_execution的ML滚动循环(密集日历purge + 23chunk + 3种子ensemble),
    独立进程跑两遍, 每chunk打印验证IC。三方比对:
      若两遍互相一致且=其中一run的序列 → 另一run的ML输入数据与缓存有差异, 转查数据路径
      若两遍互相不一致 → 实锤XGBoost真实窗口上的运行间非确定

内置校验: 23个chunk的训练样本数必须逐chunk命中两run日志的序列
  (203233 259483 316659 375408 436968 467410 476509 492344 508497 523040
   535951 550620 566159 576517 588214 596622 607303 613222 620568 629284
   634172 635561 638726) — 不命中则日历复刻有误, IC比对无效。

只读探针: 不写任何回测产物 (跳过save_model, 不碰rolling_validation_results)。
执行: cd strategy && python analysis/probe_ml_determinism_v2_0912.py > logs/probe_ml_v2_runA.log 2>&1
"""
import os
import sys
import time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config_loader import load_config
from core.ml_predictor import MLFactorPredictor
from core.strategy import Strategy

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(BASE, '..', 'data')
BT_DATA = os.path.join(DATA_ROOT, 'stock_data', 'backtrader_data')
CACHE = os.path.join(BASE, 'cache', 'factor_df_2718s_809d_d814a206.parquet')

# 两run日志记录的chunk样本数 (唯一外部锚)
LOGGED_COUNTS = [203233, 259483, 316659, 375408, 436968, 467410, 476509, 492344,
                 508497, 523040, 535951, 550620, 566159, 576517, 588214, 596622,
                 607303, 613222, 620568, 629284, 634172, 635561, 638726]
# A(pit_0911): 0.0796 0.1146 0.0882 0.0451 0.0815 0.0426 0.0842 0.0992 0.0977
#              0.1186 0.0650 0.1367 0.0987 0.1096 0.1109 0.1324 0.1016 0.1109
#              0.1100 0.1130 0.0937 0.0888 0.0136(复用)
# B(082730):   0.0809 0.1166 0.0865 0.0464 0.0826 0.0436 0.0833 0.0985 0.0993
#              0.1184 0.0648 0.1374 0.0995 0.1111 0.1113 0.1319 0.1010 0.1106
#              0.1100 0.1135 0.0921 0.0893 0.0141(复用)
IC_A = [0.0796, 0.1146, 0.0882, 0.0451, 0.0815, 0.0426, 0.0842, 0.0992, 0.0977,
        0.1186, 0.0650, 0.1367, 0.0987, 0.1096, 0.1109, 0.1324, 0.1016, 0.1109,
        0.1100, 0.1130, 0.0937, 0.0888, 0.0136]
IC_B = [0.0809, 0.1166, 0.0865, 0.0464, 0.0826, 0.0436, 0.0833, 0.0985, 0.0993,
        0.1184, 0.0648, 0.1374, 0.0995, 0.1111, 0.1113, 0.1319, 0.1010, 0.1106,
        0.1100, 0.1135, 0.0921, 0.0893, 0.0141]


def build_calendar():
    """复刻bt_execution线540-544(calendar_index) + 584-588(早段补齐)"""
    config = load_config()
    fromdate = pd.Timestamp(config.get('backtest.fromdate', '2021-01-01'))
    # calendar_index: sh000001全部日期 (stock_file_map首条目=sh000001, 注释+样本数校验兜底)
    idx = pd.read_csv(os.path.join(BT_DATA, 'sh000001_qfq.csv'), parse_dates=['datetime'])
    calendar = sorted(pd.Timestamp(t) for t in idx['datetime'])
    # 早段补齐: raw 000001 在 [fromdate-730d, fromdate) 的日期
    raw_path = os.path.join(DATA_ROOT, 'stock_data', 'raw_data', '000001', 'qfq.csv')
    raw_dates = pd.to_datetime(pd.read_csv(raw_path, encoding='utf-8-sig',
                                           usecols=['日期'])['日期'])
    early = sorted(d for d in raw_dates
                   if fromdate - pd.Timedelta(days=730) <= d < fromdate)
    all_dates = sorted(set(early) | set(calendar))
    return all_dates, fromdate


def build_regime_map():
    """复刻bt_execution线530-573: index_df[FROMDATE,TODATE] → Strategy.generate_market_regime"""
    config = load_config()
    fromdate = pd.Timestamp(config.get('backtest.fromdate', '2021-01-01'))
    todate = config.get('backtest.todate', None)
    index_df = pd.read_csv(os.path.join(BT_DATA, 'sh000001_qfq.csv'),
                           parse_dates=['datetime'])
    if fromdate is not None:
        index_df = index_df[index_df['datetime'] >= fromdate]
    if todate is not None:
        index_df = index_df[index_df['datetime'] <= pd.Timestamp(todate)]
    small_cap_df = pd.read_csv(os.path.join(BT_DATA, 'sh000852_qfq.csv'),
                               parse_dates=['datetime'])
    growth_df = pd.read_csv(os.path.join(BT_DATA, '399006_qfq.csv'),
                            parse_dates=['datetime'])
    strategy = Strategy(init_cash=1000000)
    strategy.generate_market_regime(index_df, small_cap_df=small_cap_df,
                                    growth_df=growth_df)
    regime_df = strategy.index_data
    print(f"[regime] index_data {len(regime_df)} 行 (两run日志=1380), "
          f"index类型={type(regime_df.index).__name__}")
    _regime_map = {}
    if regime_df is not None and 'regime' in regime_df.columns:
        for _idx, _row in regime_df.iterrows():
            k = _idx.strftime('%Y-%m-%d') if isinstance(_idx, pd.Timestamp) else str(_idx)
            _regime_map[k] = int(_row['regime'])
    return _regime_map


def main():
    t0 = time.time()
    config = load_config()
    ml_config = config.config.get('ml', {})
    print(f"[load] 缓存: {CACHE} ({os.path.getsize(CACHE) / 1e9:.2f}GB)")
    factor_df = pd.read_parquet(CACHE)
    # 复刻factor_preparer线487-489的code列处理
    if factor_df['code'].dtype != object:
        factor_df['code'] = factor_df['code'].astype(str).str.zfill(6)
    print(f"[load] factor_df: {len(factor_df)} 行, {factor_df['code'].nunique()} 只, "
          f"{factor_df['date'].min().date()}~{factor_df['date'].max().date()} "
          f"({time.time() - t0:.0f}s)")

    all_dates, fromdate = build_calendar()
    print(f"[cal] 密集日历 {len(all_dates)} 天 ({all_dates[0].date()}~{all_dates[-1].date()})")
    _regime_map = build_regime_map()

    # === 复刻bt_execution线636-796的ML滚动循环 ===
    _ml_df = factor_df  # margin_features.enabled=False → 无两融接入
    _train_window = ml_config.get('train_window_days', 750)
    _retrain_freq = ml_config.get('retrain_frequency', 60)
    _pred_start = pd.Timestamp(ml_config.get('pred_start_date', '2021-01-01'))

    _all_dates = sorted(_ml_df['date'].unique())
    _pred_dates = [d for d in _all_dates if d >= _pred_start]
    print(f"[ml] train_window={_train_window} retrain_every={_retrain_freq} "
          f"pred_dates={len(_pred_dates)} (日志=690)")
    assert len(_pred_dates) == 690, f"pred_dates={len(_pred_dates)} != 690, 日历/缓存复刻有误"

    chunk_starts = list(range(0, len(_pred_dates), _retrain_freq))
    _val_ics = []
    _last_good_predictor = None
    _reused_chunks = 0
    _max_reuse = int(ml_config.get('max_consecutive_reuse', 3))
    _consecutive_reuse = 0
    _min_val_ic = ml_config.get('min_val_ic', 0.03)
    _fp_days = int(config.get('dynamic_factor.forward_period', 10))
    chunk_counts = []
    chunk_ics = []

    for chunk_idx, chunk_start in enumerate(chunk_starts):
        chunk_end = min(chunk_start + _retrain_freq, len(_pred_dates))
        chunk_dates = _pred_dates[chunk_start:chunk_end]
        first_pred_date = chunk_dates[0]

        _d0_ts = pd.Timestamp(first_pred_date)
        _d0_pos = int(np.searchsorted(np.asarray(all_dates), _d0_ts))
        _purge_end = all_dates[max(0, _d0_pos - _fp_days)]
        train_start = first_pred_date - pd.Timedelta(days=_train_window)
        train_mask = (_ml_df['date'] >= train_start) & (_ml_df['date'] < _purge_end)
        train_df = _ml_df[train_mask]
        chunk_counts.append(len(train_df))

        if len(train_df) < 50000:
            print(f"  chunk {chunk_idx}: 训练样本不足({len(train_df)}), 跳过")
            chunk_ics.append(None)
            continue

        ml_predictor = MLFactorPredictor(config.config)
        _train_regime = 0
        if _regime_map:
            _train_regs = [_regime_map.get(d.strftime('%Y-%m-%d'), 0)
                           for d in train_df['date'].unique()]
            _train_regime = int(np.median(_train_regs)) if _train_regs else 0
        val_ic = ml_predictor.train(train_df, regime_info={'regime': _train_regime})
        chunk_ics.append(val_ic)
        if val_ic is None or val_ic < _min_val_ic:
            if _last_good_predictor is None:
                if val_ic is not None and val_ic > 0:
                    print(f"  chunk {chunk_idx}: 验证IC={val_ic:.4f} < {_min_val_ic}, 跳过")
                continue
            if _consecutive_reuse < _max_reuse:
                print(f"  chunk {chunk_idx}: 验证IC={val_ic}, <{_min_val_ic}, 复用上一有效模型")
                ml_predictor = _last_good_predictor
                _reused_chunks += 1
                _consecutive_reuse += 1
            elif val_ic is not None and val_ic > 0:
                print(f"  chunk {chunk_idx}: 验证IC={val_ic:.4f} <{_min_val_ic}, "
                      f"连续复用{_consecutive_reuse}次达上限, 采用新模型")
                _last_good_predictor = ml_predictor
                _consecutive_reuse = 0
            else:
                print(f"  chunk {chunk_idx}: 验证IC={val_ic}, "
                      f"复用链已达{_consecutive_reuse}次且新模型无正向IC, 继续复用")
                ml_predictor = _last_good_predictor
                _reused_chunks += 1
                _consecutive_reuse += 1
        else:
            _val_ics.append(val_ic)
            _last_good_predictor = ml_predictor
            _consecutive_reuse = 0
            # 只读探针: 跳过save_model, 不覆盖models/产物

        # 探针只关心训练IC, 跳过predict落盘 (不写ml_preds)

    print(f"\n[result] 23 chunks: 有效={len(_val_ics)} 复用={_reused_chunks} "
          f"avg_IC={np.mean(_val_ics) if _val_ics else 0:.4f} "
          f"总耗时={(time.time() - t0) / 60:.1f}min")
    print("[result] chunk样本数: " + ' '.join(map(str, chunk_counts)))
    print("[result] chunk验证IC: " + ' '.join(f"{x:.10f}" if x is not None else 'None'
                                              for x in chunk_ics))

    # 校验: 样本数必须逐chunk命中日志
    if chunk_counts == LOGGED_COUNTS:
        print("[check] 样本数23/23命中两run日志 → 日历/窗口/purge复刻精确")
    else:
        bad = [(i, a, b) for i, (a, b) in enumerate(zip(chunk_counts, LOGGED_COUNTS))
               if a != b]
        print(f"[check] 样本数不符 {len(bad)}/23 (chunk, 探针, 日志): {bad[:5]}")

    # 三方比对: 探针IC序列 vs A vs B
    for tag, ref in (('A(pit)', IC_A), ('B(082730)', IC_B)):
        diffs = [abs(x - r) for x, r in zip(chunk_ics, ref) if x is not None]
        print(f"[cmp] vs {tag}: max|Δ|={max(diffs):.6f} mean|Δ|={np.mean(diffs):.6f} "
              f"一致chunk数={sum(1 for x, r in zip(chunk_ics, ref) if x is not None and abs(x - r) < 1e-6)}/23")


if __name__ == '__main__':
    main()
