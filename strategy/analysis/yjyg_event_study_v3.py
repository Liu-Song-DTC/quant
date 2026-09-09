#!/usr/bin/env python
"""业绩预告事件研究 v3 (2026-09-09): 干净数据 + 市场调整 + 分年分层

v1/v2 (8/25, segval_20260822) 遗产: NEG队列exc均值1e11=脏价格未滤净; v1结论
"未市场调整时NEG跑赢POS(每年负spread)"疑为beta/规模混杂。v3用当前干净数据
(backtrader_data qfq)重做, 判据同v2: 超额口径POS-NEG spread方向+梯度分层。

关键PIT: notice_date=预告公告日(盘后知), 基准价=公告日收盘, fwd从下一交易日算。
市场基准=事件股票等权日收益(小盘口径, 同v2)。
输出: cohort×horizon / cohort×year / POS内change_pct梯度 / 行业无关性检验省略。
输入: data/alternative_data/yjyg_records.pkl
      data/stock_data/backtrader_data/*_qfq.csv
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, '/mnt/d/quant/strategy')
from core.alternative_data import get_provider

DATA_DIR = '/mnt/d/quant/data/stock_data/backtrader_data'
HORIZONS = [5, 10, 20, 40, 60]
IND_PRIO = ['归属于上市公司股东的净利润', '扣除非经常性损益后的净利润', '净利润',
            '每股收益', '营业收入']
TYPE_COHORT = {
    '预增': 'POS', '扭亏': 'POS', '略增': 'POS',
    '预减': 'NEG', '首亏': 'NEG', '略减': 'NEG', '增亏': 'NEG', '续亏': 'NEG',
    '减亏': 'MIX', '续盈': 'NEU', '不确定': 'NEU',
}


def load_events():
    df = get_provider().load_yjyg()
    df = df[df['indicator'].isin(IND_PRIO)].copy()
    df['prio'] = df['indicator'].map({k: i for i, k in enumerate(IND_PRIO)})
    df = df.sort_values(['code', 'report_period', 'prio', 'notice_date'])
    ev = df.groupby(['code', 'report_period'], as_index=False).apply(
        lambda g: g.sort_values(['prio', 'notice_date']).iloc[-1],
        include_groups=False)
    ev = ev[['code', 'notice_date', 'report_period', 'forecast_type', 'change_pct']]
    ev['cohort'] = ev['forecast_type'].map(TYPE_COHORT)
    ev['code'] = ev['code'].astype(str).str.zfill(6)
    return ev.reset_index(drop=True)


def main():
    ev = load_events()
    print(f"事件(每[code,period]最新): {len(ev):,} | 股票 {ev['code'].nunique():,} | "
          f"区间 {ev['notice_date'].min()} → {ev['notice_date'].max()}")
    ev = ev[(ev['notice_date'] >= '2021-01-01') & (ev['notice_date'] <= '2026-08-31')].copy()
    print(f"2021+事件: {len(ev):,}")

    # pass1: 事件股票价格 + 等权市场日收益
    codes = sorted(set(ev['code']))
    px_map, mkt_daily = {}, []
    n_ok = 0
    for code in codes:
        p = os.path.join(DATA_DIR, f'{code}_qfq.csv')
        if not os.path.exists(p):
            continue
        try:
            dfp = pd.read_csv(p, usecols=['datetime', 'close'])
        except Exception:
            continue
        dfp['datetime'] = pd.to_datetime(dfp['datetime'])
        dfp = dfp.sort_values('datetime').reset_index(drop=True)
        cl = dfp['close'].values.astype(float)
        if len(cl) < 10:
            continue
        ret = np.diff(cl) / np.clip(cl[:-1], 1e-9, None)
        keep = np.isfinite(ret) & (np.abs(ret) <= 0.25)
        dates = dfp['datetime'].values
        px_map[code] = (dates, cl, dates[:-1][keep], ret[keep])
        # 等权日收益累积用 (日期序列长, 直接全量)
        mkt_daily.append(pd.Series(ret[keep], index=pd.to_datetime(dates[:-1][keep])))
        n_ok += 1
    mkt = pd.concat(mkt_daily, axis=1).mean(axis=1, skipna=True).sort_index()
    print(f"价格载入: {n_ok}/{len(codes)} | 等权市场日收益: {len(mkt)} 天")

    def fwd_series(code, notice_dt):
        if code not in px_map:
            return None
        dates, cl, rdates, rrets = px_map[code]
        i = dates.searchsorted(np.datetime64(notice_dt))
        if i >= len(dates) or i + 1 >= len(dates):
            return None
        base = cl[i]
        if base <= 0:
            return None
        # fwd = 下一交易日close起 1..h
        out = {}
        for h in HORIZONS:
            j = i + 1 + h
            if j >= len(cl):
                out[h] = np.nan
            else:
                out[h] = cl[j] / cl[i + 1] - 1
        return out

    def mkt_fwd(notice_dt, h):
        dts = mkt.index
        i = dts.searchsorted(np.datetime64(notice_dt))
        seg = dts[i:i + h + 1]
        if len(seg) < 2:
            return np.nan
        r = mkt.loc[seg].values
        return float(np.prod(1 + r) - 1)

    rows = []
    for code, nd in zip(ev['code'], ev['notice_date']):
        nd = pd.Timestamp(nd)
        f = fwd_series(code, nd)
        if f is None:
            continue
        for h in HORIZONS:
            if np.isnan(f[h]):
                continue
            rows.append((code, nd, h, f[h] - mkt_fwd(nd, h)))
    ex = pd.DataFrame(rows, columns=['code', 'notice_date', 'h', 'exc'])
    # 事件元数据按[code,notice_date]去重再合并, 防多period同日公告引爆多对多
    evu = ev.drop_duplicates(['code', 'notice_date'])[
        ['code', 'notice_date', 'cohort', 'change_pct']]
    ev2 = ex.merge(evu, on=['code', 'notice_date'], how='inner')
    ev2['year'] = ev2['notice_date'].dt.year
    print(f"超额收益计算: {len(ev2):,} 事件×horizon | 事件数 {ev2[['code','notice_date']].drop_duplicates().shape[0]:,}")

    print("\n=== ① cohort × horizon 市场调整超额 ===")
    g = ev2.groupby(['cohort', 'h'])['exc'].agg(['mean', 'median', 'count'])
    g[['mean', 'median']] = (g[['mean', 'median']] * 100).round(2)
    print(g.to_string())

    print("\n=== ② POS−NEG spread 逐年 (exc20) ===")
    piv = ev2[ev2['h'] == 20].pivot_table(index='year', columns='cohort',
                                          values='exc', aggfunc='mean')
    piv['spread'] = piv.get('POS', np.nan) - piv.get('NEG', np.nan)
    print((piv * 100).round(2).to_string())

    print("\n=== ③ POS内 change_pct 梯度 (exc20, 惊喜程度分层) ===")
    pos = ev2[(ev2['h'] == 20) & (ev2['cohort'] == 'POS')].copy()
    pos['cp'] = pd.to_numeric(pos['change_pct'], errors='coerce')
    pos = pos[pos['cp'].notna()]
    pos['g'] = pd.cut(pos['cp'], [-np.inf, 30, 50, 100, 200, np.inf],
                      labels=['<30', '30-50', '50-100', '100-200', '>200'])
    gp = pos.groupby('g', observed=True)['exc'].agg(['mean', 'median', 'count'])
    gp[['mean', 'median']] = (gp[['mean', 'median']] * 100).round(2)
    print(gp.to_string())
    print("\nNEG内同梯度:")
    neg = ev2[(ev2['h'] == 20) & (ev2['cohort'] == 'NEG')].copy()
    neg['cp'] = pd.to_numeric(neg['change_pct'], errors='coerce')
    neg = neg[neg['cp'].notna()]
    neg['g'] = pd.cut(neg['cp'], [-np.inf, -100, -50, 0, np.inf],
                      labels=['<-100', '-100~-50', '-50~0', '>0?'])
    gn = neg.groupby('g', observed=True)['exc'].agg(['mean', 'median', 'count'])
    gn[['mean', 'median']] = (gn[['mean', 'median']] * 100).round(2)
    print(gn.to_string())

    print("\n=== ④ 非调整口径对照 (原始fwd, 检验v1的NEG>POS是否仍在) ===")
    rows2 = []
    for code, nd in zip(ev['code'], ev['notice_date']):
        nd = pd.Timestamp(nd)
        f = fwd_series(code, nd)
        if f is None or np.isnan(f[20]):
            continue
        rows2.append((code, nd, f[20]))
    raw = pd.DataFrame(rows2, columns=['code', 'notice_date', 'fwd20'])
    raw = raw.merge(ev[['code', 'notice_date', 'cohort']], on=['code', 'notice_date'])
    gr = raw.groupby('cohort')['fwd20'].agg(['mean', 'median', 'count'])
    gr[['mean', 'median']] = (gr[['mean', 'median']] * 100).round(2)
    print(gr.to_string())


if __name__ == '__main__':
    main()
