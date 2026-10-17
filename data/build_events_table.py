#!/usr/bin/env python
"""事件表: 东财(官方口径) × 新浪(完备性) 融合 (2026-10-17) —— 生产版

替代 `corporate_actions.pkl` 作为 `rebuild_table_qfq.py --events` 的输入。

**为什么需要融合**
  东财 `stock_fhps_detail_em`(分红送配表)**不是完备事件源**:
    · 洞一 230 个股票日 = **配股 / 重整转增** —— 分红送配表结构性不覆盖;
    · 洞二 14 个 = `现金分红-现金分红比例` **漏「特别股息」**
      (601088 2016 年报列 10派4.60, 反解 D=2.970 = 0.46 + 2.51特别股息, 精确到 3 位小数)。
  漏一个事件 ⇒ `W(t)` 凭空造出 −17%~−60% 假跌幅, 且**永久留在历史里**。

**已确立的事实(三个探针)**
 ① 交易所参考价口径模型 `m_conv = P_前收·(1+S)/(P_前收 − D)` 复现新浪因子:
    35,919 个事件日 **中位相对差 0.0000% / p90 0.0061% / p99 0.172%** ⇒ 我们的 (S,D) 99.6% 正确。
 ② 新浪/东财 qfq 在除权日的**收益率**一致到 1e-5 ⇒ 两者用**同一个"交易所除权参考价"口径**,
    而非真实财富比。两者与真实财富比的系统偏差在 601088 处达 **1.33pp**(D=2.97/22.29=13.3%)
    ⇒ **不能直接采用新浪乘数**, 要用它**反解参数**再走精确公式。
 ③ 冲突(>0.5%) 91 个; 我们完全没有的事件 230 个。

**融合规则(逐事件, 无自由阈值)**
 A. 两边都有且 |m_conv/m_sina − 1| ≤ 0.5%  → 用我们的 (S,D)                    [精确财富比]
 B. 两边都有且冲突 > 0.5%                  → 与**东财 qfq 除权日当日比值**比谁更近
      (除权日 C 的跳变恰等于该次分红 ⇒ 加性口径在那里有效, 实测送转日中位误差 0.092%)
      新浪更近 ⇒ 反解修 D: D' = P_前收(1 − (1+S)/m_sina), 若 D' ≥ 0 且 D' > D
      否则**维持我方** —— 修 S **无解释**: 东财文字明写 "10转4.50股"(显式实施比例),
      且 300708 2025-08-15 当日**最低价 8.88 = 我们口径参考价 round(12.88/1.45,2)=8.88**
      (逐分吻合), 而新浪隐含参考价 9.0179 非圆整。⇒ 新浪/东财 qfq 的 Π 在此是**同一对象**
      (实测 e_sina=1e-5), 不构成独立裁判 ⇒ 分歧时维持官方文字。
      代价: 27 个股票日残留 1~3pp 误差(11 年 / 全市场) —— 记录在案。
 C. 只有我们 → 用我们的
 D. 只有新浪 → 直接用新浪乘数 (230 个: 配股/重整转增 —— 分红送配表结构性不覆盖)

**输入**(须先跑 `fetch_actions_table.py` 与 `fetch_sina_factors.py`):
    data/alternative_data/corporate_actions.pkl   (code, ex_date, s, D)
    data/alternative_data/sina_factors.pkl        (code, ex_date, mult)
    data/stock_data/raw_data/{code}/none.csv      (交易日 + 裸收盘)
    data/stock_data/raw_data/{code}/qfq.csv       (厂商 qfq, 仅用于除权日裁判)

**产出** `data/alternative_data/events_table.pkl`: code, ex_date, s, D, mult, src
    `mult` 非空 ⇒ `rebuild_table_qfq.py` 按**直接乘数**消费(`r = mult·raw(t)/raw(t−1)`);
    em 行的 mult = `(1+s) + D/raw(t)` ⇒ 展开后与 (s,D) 公式**逐位等价**。
"""
import os
import argparse
import numpy as np
import pandas as pd

from market_codes import INDEX_PSEUDO, is_stock_code, audit

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = f'{PROJ}/data/stock_data/raw_data'
ALT = f'{PROJ}/data/alternative_data'
CA = f'{ALT}/corporate_actions.pkl'
SINA = f'{ALT}/sina_factors.pkl'
OUT = f'{ALT}/events_table.pkl'
TOL = 0.005


def _vendor_close(code):
    """厂商 qfq 的 {日期: 收盘}。用于除权日裁判; 非正价**不入表** ⇒ 该日退化为"维持我方"。

    那 52 只被 `repair_qfq_negatives.py` 维护的股票在原始文件里仍是负价(补丁已撤),
    负价下比值无意义 ⇒ 直接丢弃该日, 不参与裁判。
    """
    p = f'{RAW}/{code}/qfq.csv'
    if not os.path.exists(p):
        return {}
    try:
        d = pd.read_csv(p, encoding='utf-8-sig', usecols=['日期', '收盘'])
    except Exception:
        return {}
    d['日期'] = pd.to_datetime(d['日期'])
    d['收盘'] = pd.to_numeric(d['收盘'], errors='coerce')
    d = d[np.isfinite(d['收盘']) & (d['收盘'] > 0)]
    return dict(zip(d['日期'].values, d['收盘'].values.astype(float)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=OUT)
    ap.add_argument('--ca', default=CA)
    ap.add_argument('--sina', default=SINA)
    a = ap.parse_args()

    print('排除的指数伪码(沿用生产文件, 不参与事件构建):')
    for c, nm in audit(RAW):
        print(f'    {c}  {nm}')
    print()

    ca = pd.read_pickle(a.ca)
    ca['code'] = ca['code'].astype(str).str.zfill(6)
    sn = pd.read_pickle(a.sina).rename(columns={'code': 'sym', 'ex_date': 'd'})
    sn['sym'] = sn.sym.astype(str).str.zfill(6)
    sn['d'] = pd.to_datetime(sn.d)

    n_drop_ca = (~ca.code.map(is_stock_code)).sum()
    n_drop_sn = (~sn.sym.map(is_stock_code)).sum()
    ca = ca[ca.code.map(is_stock_code)]
    sn = sn[sn.sym.map(is_stock_code)]
    print(f'剔除伪码事件: 东财 {n_drop_ca} 条, 新浪 {n_drop_sn} 条\n')

    ca_by = {c: g[['ex_date', 's', 'D']].values.tolist() for c, g in ca.groupby('code')}
    sn_by = {c: dict(zip(g['d'], g['mult'])) for c, g in sn.groupby('sym')}
    codes = sorted(set(ca_by) | set(sn_by))

    rows = []
    n_conf = n_repD = n_repS = n_only_sn = 0
    n_no_arb = 0                      # 冲突但厂商裁判不可用(负价/缺行) ⇒ 维持我方
    for i, code in enumerate(codes):
        try:
            r = pd.read_csv(f'{RAW}/{code}/none.csv', encoding='utf-8-sig', usecols=['日期', '收盘'])
            r['日期'] = pd.to_datetime(r['日期'])
            r = r.sort_values('日期').reset_index(drop=True)
        except Exception:
            continue
        days = r['日期'].values
        raw = r['收盘'].values.astype(float)
        n = len(raw)
        if n < 2:
            continue
        idx = pd.Series(np.arange(n), index=pd.DatetimeIndex(days))

        def to_day(ts):
            p = int(idx.index.searchsorted(pd.Timestamp(ts), side='left'))
            return p if 0 < p < n else None

        ours = {}
        for ex, s, D in ca_by.get(code, []):
            p = to_day(ex)
            if p is None:
                continue
            s0, D0 = ours.get(p, (0.0, 0.0))
            ours[p] = ((1 + s0) * (1 + s) - 1, D0 + D)
        sina = {}
        for k, m in sn_by.get(code, {}).items():
            p = to_day(k)
            if p is not None:
                sina[p] = float(m)

        vclose = None                     # 懒加载: 只在真有冲突时才读厂商文件
        for p in sorted(set(ours) | set(sina)):
            pp, cl = raw[p - 1], raw[p]
            if pp <= 0:
                continue
            if p in ours and p in sina:
                s, D = ours[p]
                ms = sina[p]
                mc = pp * (1 + s) / (pp - D) if pp > D else np.inf
                if abs(mc / ms - 1) <= TOL:
                    rows.append((code, pd.Timestamp(days[p]), s, D, (1 + s) + D / cl, 'em'))
                    continue
                n_conf += 1
                r_ours = ((1 + s) * cl + D) / pp - 1
                r_sina = ms * cl / pp - 1
                if vclose is None:
                    vclose = _vendor_close(code)
                c0, c1 = vclose.get(days[p - 1]), vclose.get(days[p])
                if c0 is None or c1 is None:
                    n_no_arb += 1
                    use_sina = False
                else:
                    rv = c1 / c0 - 1
                    use_sina = abs(r_sina - rv) < abs(r_ours - rv)
                if use_sina:
                    Dp = pp * (1 - (1 + s) / ms)
                    if Dp >= 0 and Dp > D:
                        rows.append((code, pd.Timestamp(days[p]), s, Dp, (1 + s) + Dp / cl, 'em_repD'))
                        n_repD += 1
                        continue
                    n_repS += 1
                rows.append((code, pd.Timestamp(days[p]), s, D, (1 + s) + D / cl, 'em'))
            elif p in ours:
                s, D = ours[p]
                rows.append((code, pd.Timestamp(days[p]), s, D, (1 + s) + D / cl, 'em'))
            else:
                rows.append((code, pd.Timestamp(days[p]), np.nan, np.nan, sina[p], 'sina'))
                n_only_sn += 1
        if i % 800 == 0:
            print(f'  ...{i}/{len(codes)}  事件={len(rows)}', flush=True)

    m = pd.DataFrame(rows, columns=['code', 'ex_date', 's', 'D', 'mult', 'src'])
    m = m.sort_values(['code', 'ex_date']).reset_index(drop=True)
    m.to_pickle(a.out)
    print(f'\n=== 落盘 {a.out} ===')
    print(f'  事件 {len(m)} 条 / {m.code.nunique()} 只')
    print(m.src.value_counts().to_string())
    print(f'  冲突(>0.5%) {n_conf}   其中修D {n_repD} / 维持我方(含修S不采纳) {n_conf - n_repD}'
          f'   (厂商裁判不可用 {n_no_arb})')
    print(f'  仅新浪填补 {n_only_sn}')
    print(f'  乘数 |m−1| > 0.20 的: {(m["mult"].sub(1).abs() > 0.20).sum()}')
    print(f'  日期范围 {m.ex_date.min().date()} ~ {m.ex_date.max().date()}')


if __name__ == '__main__':
    main()
