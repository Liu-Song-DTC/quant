"""E-seq1a烟测: 生产signals CSV的score/adjusted_score注入GRU z (2026-09-15)

机制镜像: 信号层 score ← score + w*z_gru (逐日截面z, 2025-01-01+行)。
生产事实: score与adjusted_score在CSV中逐位相等(组合层消费score), 两列同改。
不改buy/sell旗标 — 烟测隔离"GRU排序/定注"效应; 旗标联动属全机制阶段。
协议: 写新文件, 由run脚本负责swap(备份→替换→跑→复原), 本脚本不碰生产文件。
用法: python edit_signals_gru_20260915.py W [GRU_PKL]   (W=0.32≈z-sum等权)
"""
import os
import sys
import pandas as pd

OUT = '/mnt/d/quant/strategy/rolling_validation_results'
SIG = os.path.join(OUT, 'backtest_signals.csv')
GRU = os.path.join(OUT, 'gru_daily_preds_v2.pkl')


def main():
    w = float(sys.argv[1]) if len(sys.argv) > 1 else 0.32
    g = pd.read_pickle(sys.argv[2] if len(sys.argv) > 2 else GRU)
    mode = sys.argv[3] if len(sys.argv) > 3 else 'all'
    g['code'] = pd.Series(g['code']).astype(str).str.zfill(6).values
    g['date'] = pd.to_datetime(g['date'])
    zcol = 'gru_z' if 'gru_z' in g else 'z'
    print(f'GRU z条目: {len(g["code"])} ({g["date"].min().date()}~{g["date"].max().date()}), '
          f'z列={zcol}, 模式={mode}', flush=True)

    sig = pd.read_csv(SIG, dtype={'code': str}, low_memory=False)
    sig['date'] = pd.to_datetime(sig['date'])
    sig['code'] = sig['code'].str.zfill(6)
    n_all = len(sig)
    mask = (sig.date >= '2025-01-01').values
    if mode in ('buys',):
        mask = mask & (sig.buy == True).values
    gdf = pd.DataFrame({'code': pd.Series(g['code']).astype(str).str.zfill(6),
                        'date': pd.to_datetime(g['date']), 'z': g[zcol]})
    sub = sig.loc[mask, ['code', 'date']].merge(gdf, on=['code', 'date'], how='left')
    z = sub['z'].values
    hit = ~np.isnan(z)
    z = np.clip(z, -3.0, 3.0)  # 小截面日期z可达22σ, winsorize防单点注入爆炸
    print(f'2025+行 {mask.sum()}, 有GRU z {hit.sum()} ({100*hit.mean():.1f}%), '
          f'|z|分布: max={np.nanmax(np.abs(z)):.2f} (winsor±3)', flush=True)
    hit_full = np.zeros(n_all, dtype=bool)
    hit_full[mask] = hit
    # 注入 (两列同改保持逐位相等不变式)
    if mode == 'mlchan':
        # 生产ML通道同款: adjusted = (1-w)*score + w*tanh(z) (有界收缩blend)
        s_cur = sig.loc[hit_full, 'score'].to_numpy()
        s_new = (1 - w) * s_cur + w * np.tanh(z[hit])
        sig.loc[hit_full, 'score'] = s_new
        sig.loc[hit_full, 'adjusted_score'] = s_new
    else:
        sig.loc[hit_full, 'score'] += w * z[hit]
        sig.loc[hit_full, 'adjusted_score'] += w * z[hit]
    # 仅score列绝对值超历史极值|1.79|的少数行截断, 防极端z引入超range分数
    clip = 2.0
    over = (sig.loc[mask, 'score'].abs() > clip)
    over_full = np.zeros(n_all, dtype=bool)
    over_full[mask] = over.values
    print(f'score超{clip}行(截断): {int(over_full.sum())}', flush=True)
    sig.loc[over_full, 'score'] = np.sign(sig.loc[over_full, 'score']) * clip
    sig.loc[over_full, 'adjusted_score'] = np.sign(sig.loc[over_full, 'adjusted_score']) * clip
    # 非注入行原样保留(逐位): 校验未注入行score与原始一致
    orig = pd.read_csv(SIG, dtype={'code': str}, low_memory=False)['score'].to_numpy()
    untouched_ok = (sig.loc[~hit_full, 'score'].to_numpy() == orig[~hit_full]).all()
    print(f'未注入行逐位一致: {untouched_ok} (共{int((~hit_full).sum())}行)', flush=True)

    suffix = {'all': '', 'buys': '_buys', 'mlchan': '_mlchan'}[mode]
    out_path = os.path.join(OUT, f'backtest_signals.gru_w{w:g}{suffix}.csv')
    sig.to_csv(out_path, index=False)
    print(f'已写 {out_path} ({n_all} 行)', flush=True)


if __name__ == '__main__':
    import numpy as np
    main()
