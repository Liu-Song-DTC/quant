"""阶段8d: 市场级宏观/情绪指标 regime层前瞻探针 (2026-09-15)

背景: regime_detector 是纯价格型(指数技术指标). 外部市场级指标能否提供前瞻增量?
若某指标对指数未来20d/1月收益有稳定预测力 → 可作regime先验候选(仍需机制实验).

指标(全部单次API, 无分页):
  A. 社融增量(macro_china_shrzgm, 月) — z-score/同比
  B. M1-M2剪刀差(macro_china_money_supply, 月) — 经典A股领先指标
  C. 制造业PMI(macro_china_pmi, 月) — 荣枯线
  D. PPI同比(macro_china_ppi, 月) — 盈利周期代理
  E. 期指基差(futures_main_sina IF0/IC0/IM0/IH0, 日) — 年化基差率, 对冲成本/情绪
测度: 每指标信号 → 指数未来20d收益 Spearman (日频基差) / 未来1月收益相关 (月度),
  2021-2026, 逐年稳定性
判据: 单指标 |IC|>0.05 且 逐年正率≥4/6 → 有价值, 排队机制实验; 否则关闭该指标
产出: rolling_validation_results/macro_regime_probe.pkl
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy import stats as _st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import akshare as ak

OUT = '/mnt/d/quant/strategy/rolling_validation_results'
IDX = pd.read_csv('/mnt/d/quant/data/stock_data/backtrader_data/sh000001_qfq.csv',
                  parse_dates=['datetime']).set_index('datetime')['close']


def report(name, s, idx_ret_fwd, freq='monthly'):
    """s=信号序列(对齐日期), idx_ret_fwd=同期未来指数收益. 输出相关+逐年."""
    if freq == 'monthly':
        s = s.copy(); s.index = s.index.to_period('M')
        idx_ret_fwd = idx_ret_fwd.copy(); idx_ret_fwd.index = idx_ret_fwd.index.to_period('M')
    df = pd.DataFrame({'s': s, 'f': idx_ret_fwd}).dropna()
    if len(df) < 20:
        print(f'  [{name}] 样本不足({len(df)})', flush=True)
        return
    ic = _st.spearmanr(df['s'], df['f'])[0]
    yrs = df.index.year
    pos = 0
    nyr = 0
    ics = []
    for y in sorted(set(yrs)):
        m = yrs == y
        if m.sum() >= 8:
            ic_y = _st.spearmanr(df['s'][m], df['f'][m])[0]
            nyr += 1
            pos += ic_y > 0
            ics.append(f'{y}:{ic_y:+.2f}')
    print(f'  [{name}] n={len(df)} 相关={ic:+.3f} 逐年正率={pos}/{nyr} 逐年IC[{",".join(ics)}]', flush=True)


def _parse_month(col):
    from datetime import datetime
    s = col.astype(str).str.replace('年', '-').str.replace('月份', '')
    out = []
    for x in s:
        digits = ''.join(ch for ch in x if ch.isdigit())
        out.append(datetime.strptime(digits[:8], '%Y%m%d') if len(digits) >= 8
                   else datetime.strptime(digits[:6], '%Y%m'))
    return pd.DatetimeIndex(out)


def main():
    f1 = IDX.resample('ME').last().pct_change().shift(-1)

    print('=== A. 社融增量 ===', flush=True)
    try:
        sz = ak.macro_china_shrzgm()
        sz.index = _parse_month(sz['月份'])
        sz = sz[sz.index >= '2021-01-01'].copy()
        sz['v'] = pd.to_numeric(sz['社会融资规模增量'], errors='coerce')
        sz['s'] = sz['v'].pct_change(12)  # 同比
        report('社融增量同比→次月指数收益', sz['s'], f1)
    except Exception as e:
        print(f'  [A] 失败 {type(e).__name__}: {e}', flush=True)

    print('=== B. M1-M2剪刀差 ===', flush=True)
    try:
        ms = ak.macro_china_money_supply()
        m1c = [c for c in ms.columns if 'M1' in c and '同比' in c]
        m2c = [c for c in ms.columns if 'M2' in c and '同比' in c]
        if '月份' in ms.columns and m1c and m2c:
            ms.index = _parse_month(ms['月份'])
            ms = ms[ms.index >= '2021-01-01'].copy()
            ms['m1'] = pd.to_numeric(ms[m1c[0]], errors='coerce')
            ms['m2'] = pd.to_numeric(ms[m2c[0]], errors='coerce')
            ms['scissor'] = ms['m1'] - ms['m2']
            report('M1-M2剪刀差→次月', ms['scissor'], f1)
        else:
            print(f'  M1/M2列缺失: {list(ms.columns)}', flush=True)
    except Exception as e:
        print(f'  [B] 失败 {type(e).__name__}: {e}', flush=True)

    print('=== C. PMI ===', flush=True)
    try:
        pmi = ak.macro_china_pmi()
        dcol = '日期' if '日期' in pmi.columns else '月份'
        c1 = '今值' if '今值' in pmi.columns else pmi.columns[1]
        pmi.index = _parse_month(pmi[dcol])
        pmi = pmi[pmi.index >= '2021-01-01'].copy()
        pmi['v'] = pd.to_numeric(pmi[c1], errors='coerce')
        report('制造业PMI→次月', pmi['v'], f1)
    except Exception as e:
        print(f'  [C] 失败 {type(e).__name__}: {e}', flush=True)

    print('=== D. PPI同比 ===', flush=True)
    try:
        ppi = ak.macro_china_ppi()
        dcol2 = '日期' if '日期' in ppi.columns else '月份'
        c2 = '今值' if '今值' in ppi.columns else ppi.columns[1]
        ppi.index = _parse_month(ppi[dcol2])
        ppi = ppi[ppi.index >= '2021-01-01'].copy()
        ppi['v'] = pd.to_numeric(ppi[c2], errors='coerce')
        report('PPI同比→次月', ppi['v'], f1)
    except Exception as e:
        print(f'  [D] 失败 {type(e).__name__}: {e}', flush=True)

    print('=== E. 期指基差(日频→未来20d) ===', flush=True)
    spots = {'IF0': 399300, 'IC0': 399905, 'IH0': 999016, 'IM0': 399852}
    for sym, idx_code in spots.items():
        try:
            fut = ak.futures_main_sina(symbol=sym, start_date='20210101', end_date='20260915')
        except Exception as e:
            print(f'  [{sym}] 拉取失败 {e}', flush=True)
            continue
        if '日期' not in fut.columns or '收盘价' not in fut.columns:
            print(f'  [{sym}] 列缺失', flush=True)
            continue
        fut['date'] = pd.to_datetime(fut['日期'])
        fut['close'] = pd.to_numeric(fut['收盘价'], errors='coerce')
        fut = fut.set_index('date')['close'].dropna()
        if idx_code == 399300:
            sp = ak.stock_zh_index_daily(symbol='sh000300')
        elif idx_code == 399905:
            sp = ak.stock_zh_index_daily(symbol='sh000905')
        elif idx_code == 999016:
            sp = ak.stock_zh_index_daily(symbol='sh000016')
        else:
            sp = ak.stock_zh_index_daily(symbol='sh000852')
        sp = sp.rename(columns={'date': 'd', 'close': 'c'})
        sp['d'] = pd.to_datetime(sp['d'])
        sp = sp.set_index('d')['c']
        # 年化基差率 (主连 vs 现货, 当月合约近似; 主连滚动会引入跳变, 用20d滚动中位平滑)
        common = fut.index.intersection(sp.index)
        basis = (fut[common] / sp[common] - 1) * 100  # %
        basis_s = basis.rolling(20).median()
        fwd20 = IDX.pct_change(20).shift(-20)
        s = basis_s.reindex(fwd20.index).ffill()
        report(f'{sym}年化基差(20d中位, 日)→指数20d', s, fwd20, freq='daily')

    pd.to_pickle({'done': True}, os.path.join(OUT, 'macro_regime_probe.pkl'))
    print('\nDONE', flush=True)


if __name__ == '__main__':
    main()
