"""阶段8b: 个股主力资金流 增量检验探针 (2026-09-15)

问题: 东财每日个股主力净流入(超大单+大单口径)是否在 smart_money_flow(价量代理)
之外提供增量预测力? 若只是价量代理的翻版, 接线无意义; 若增量显著 → 接线候选。

方法:
  1. 从当前factor parquet(2689只)确定性抽样300只, 拉取个股资金流历史(东财, akshare)
  2. 标签 = parquet.future_ret (ML同款, fwd forward_period日 close/close)
  3. 月度截面: IC(main_net_ratio, fwd) + IR; 逐年
  4. 增量检验: 月内 rank(smart_money_flow) 回归剥离后, main_net_ratio 的偏相关/t值
判据: IC>0.02 且 IR>0.3 且 增量t>2 → 接线候选; 增量不显著 → 否决(价量代理已覆盖)

产出: rolling_validation_results/fundflow_probe.pkl + 打印判决
"""
import os
import sys
import time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import requests
import pyarrow.parquet as pq

SINA_URL = ('https://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/'
            'MoneyFlow.ssl_qsfx_zjlrqs')
HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
           'Referer': 'https://finance.sina.com.cn/'}

PQ_PATH = '/mnt/d/quant/strategy/cache/factor_df_2689s_810d_e492a643.parquet'
OUT = '/mnt/d/quant/strategy/rolling_validation_results'
RAW = os.path.join(OUT, 'fundflow_raw_300.pkl')
N_SAMPLE = 300


def load_parquet_panel():
    t = pq.read_table(PQ_PATH, columns=['code', 'date', 'smart_money_flow', 'future_ret'])
    df = t.to_pandas()
    df['code'] = df['code'].astype(str).str.zfill(6)
    return df


def fetch_sample(codes):
    """新浪个股资金流历史(东财push2his被网络级封锁, 换源).
    返回 {code: df[date, main_net_ratio, xl_net_ratio]}
    main_net_ratio←ratioamount(主力净流入占比), xl_net_ratio←r0_ratio(超大单占比).
    断点续跑: 每50只落盘; 连续12只失败 → 保存部分退出(限流保护)."""
    out = {}
    if os.path.exists(RAW):
        out = pd.read_pickle(RAW)
    streak = 0
    t0 = time.time()
    for i, c in enumerate(codes):
        if c in out:
            continue
        ok = False
        daima = ('sh' if c.startswith(('6', '9')) else 'sz') + c
        for attempt in range(3):
            try:
                r = requests.get(SINA_URL, headers=HEADERS, timeout=20,
                                 params={'page': '1', 'num': '3000', 'sort': 'opendate',
                                         'asc': '0', 'daima': daima})
                j = r.json()
                if isinstance(j, list) and len(j):
                    df = pd.DataFrame(j)
                    df = df.rename(columns={'opendate': 'date', 'ratioamount': 'main_net_ratio',
                                            'r0_ratio': 'xl_net_ratio'})
                    df = df[['date', 'main_net_ratio', 'xl_net_ratio']].copy()
                    df['date'] = pd.to_datetime(df['date'])
                    for col in ('main_net_ratio', 'xl_net_ratio'):
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    out[c] = df
                    ok = True
                break
            except Exception as e:
                if attempt == 2:
                    print(f'  [FAIL {c}] {e}', flush=True)
                time.sleep(3 * (attempt + 1))
        streak = 0 if ok else streak + 1
        if streak >= 12:
            print(f'连续{streak}只失败, 限流保护: 保存{len(out)}只退出', flush=True)
            pd.to_pickle(out, RAW)
            return out
        if (i + 1) % 50 == 0:
            pd.to_pickle(out, RAW)
            print(f'  拉取 {i+1}/{len(codes)} 只 已存{len(out)} {time.time()-t0:.0f}s', flush=True)
        time.sleep(0.4)
    pd.to_pickle(out, RAW)
    return out


def main():
    panel = load_parquet_panel()
    codes_all = sorted(panel['code'].unique())
    step = max(1, len(codes_all) // N_SAMPLE)
    codes = codes_all[::step][:N_SAMPLE]
    print(f'池 {len(codes_all)} 只 → 抽样 {len(codes)} 只', flush=True)

    if os.path.exists(RAW):
        hist = pd.read_pickle(RAW)
        print(f'复用缓存 {RAW}: {len(hist)} 只', flush=True)
    else:
        print('拉取个股资金流历史...', flush=True)
        hist = fetch_sample(codes)
        print(f'拉取完成 {len(hist)} 只', flush=True)
    if len(hist) < 150:
        print(f'样本不足({len(hist)}<150), 限流中, 稍后重跑续传', flush=True)
        sys.exit(0)

    flow = pd.concat(hist.values(), keys=hist.keys(), names=['code']).reset_index(level=0)
    flow = flow.dropna(subset=['main_net_ratio'])
    print(f'资金流行数 {len(flow)}, 日期范围 {flow.date.min()} ~ {flow.date.max()}', flush=True)

    merged = panel.merge(flow, on=['code', 'date'], how='inner')
    merged = merged.dropna(subset=['future_ret', 'smart_money_flow', 'main_net_ratio'])
    merged['y'] = merged['future_ret']
    print(f'合并后 {len(merged)} 行, {merged.code.nunique()} 只', flush=True)

    # 月度IC
    print('\n[月度IC] main_net_ratio / xl_net_ratio / smart_money_flow vs future_ret:', flush=True)
    from scipy import stats as _st
    months = pd.date_range('2021-01-01', '2026-09-01', freq='MS')
    rows = []
    for m in months:
        m_end = m + pd.DateOffset(months=1) - pd.DateOffset(days=1)
        sub = merged[merged['date'] <= m_end].tail(0)
        sub = merged[(merged['date'] >= m) & (merged['date'] <= m_end)]
        if len(sub) < 40:
            continue
        ic1 = _st.spearmanr(sub['main_net_ratio'], sub['y'])[0]
        ic2 = _st.spearmanr(sub['xl_net_ratio'], sub['y'])[0]
        ic0 = _st.spearmanr(sub['smart_money_flow'], sub['y'])[0]
        rows.append((m.strftime('%Y-%m'), len(sub), ic0, ic1, ic2))
    r = pd.DataFrame(rows, columns=['month', 'n', 'ic_smf', 'ic_main', 'ic_xl'])
    for col in ('ic_smf', 'ic_main', 'ic_xl'):
        v = r[col]
        print(f'  {col:10s}: mean={v.mean():+.4f} IR={v.mean()/v.std():+.2f} '
              f'正率={100*(v > 0).mean():.0f}%', flush=True)
    print('\n逐年 IC_main:')
    for y in sorted(merged.date.dt.year.unique()):
        sub = merged[merged.date.dt.year == y]
        if len(sub) < 100:
            continue
        v = sub.groupby(sub.date.dt.month).apply(
            lambda g: _st.spearmanr(g['main_net_ratio'], g['y'])[0] if len(g) >= 40 else np.nan)
        v = v.dropna()
        if len(v):
            print(f'  {y}: n月={len(v)} mean={v.mean():+.4f} IR={v.mean()/v.std():+.2f}', flush=True)

    # 增量检验: 月内 rank(smf) 剥离后 main 的偏相关
    print('\n[增量检验] future_ret ~ a*rank(smf) + b*rank(main):', flush=True)
    tstats = []
    for m in months:
        m_end = m + pd.DateOffset(months=1) - pd.DateOffset(days=1)
        sub = merged[(merged['date'] >= m) & (merged['date'] <= m_end)]
        if len(sub) < 40:
            continue
        X = pd.DataFrame({
            'rs': sub['smart_money_flow'].rank(pct=True),
            'rm': sub['main_net_ratio'].rank(pct=True),
        })
        X = np.column_stack([np.ones(len(X)), X])
        y = sub['y'].values.astype(float)
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        # 月度回归的t统计量 (SSE手算, numpy2兼容)
        n, k = X.shape
        dof = n - k
        pred = X @ beta
        sse = float(np.sum((y - pred) ** 2))
        mse = sse / dof
        cov = mse * np.linalg.inv(X.T @ X)
        se_b = np.sqrt(cov[2, 2]) if np.all(np.isfinite(cov)) else np.nan
        tstats.append((m.strftime('%Y-%m'), beta[2], se_b, beta[2] / se_b if se_b > 0 else np.nan))
    tr = pd.DataFrame(tstats, columns=['month', 'b_main', 'se', 't'])
    t = tr['t'].dropna()
    print(f'  b_main月均值={tr.b_main.mean():+.4f} | t均值={t.mean():+.2f} '
          f'| |t|>2月份占比={100*((t.abs() > 2).mean()):.0f}% | t>2占={100*((t > 2).mean()):.0f}% '
          f'| t<0占={100*((t < 0).mean()):.0f}%', flush=True)
    print(f'  自相关调整 t/sqrt(1+2*rho) ≈ {t.mean()/np.sqrt(1 + 2*t.autocorr()):+.2f} '
          f'(rho={t.autocorr():+.2f})' if len(t) > 12 else '', flush=True)

    pd.to_pickle({'merged': merged, 'monthly_ic': r, 'incremental': tr},
                 os.path.join(OUT, 'fundflow_probe.pkl'))
    print('\nDONE', flush=True)


if __name__ == '__main__':
    main()
