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


META = f'{ALT}/.events_table_meta.json'


def _backdate_guard(m, out_path):
    """**铁律守卫**: 事后新增的历史事件 = 改写历史 (2026-10-17)。

    表驱动重建是**前向**递推 `W(t)=W(t−1)·r(t)`, 只在走到 ex_date 那一格时消费事件
    ⇒ 在 ex_date **之前**进表零副作用; 但在 ex_date **之后**补进来, 就会把
    `W(t)` 对**所有 t ≥ ex_date** 改掉 ⇒ 历史格子被重写, 且 `P = W·k` 全段位移。
    (frozen 锚的 k 也会随之漂移, 由 `rebuild_table_qfq.py` 的 drift 报告兜住 ——
     但那是**事后**, 且默认采用新锚。本守卫把同一件事**提前到写盘前**, 并点名。)

    判据: 与上一版表逐 (code, ex_date) 比对, 新表**多出来**的、且 ex_date 早于
    上一版构建时**已覆盖到的数据末日**的事件 = 回溯新增。
    非新增(改 s/D/mult)不算改历史 —— 那是同一格上的值修正, 仍会动历史, 故一并报。

    首建(无上一版) ⇒ 无基准, 跳过。
    meta 缺失时**降级但不瞎**: 用"上一版表自己的最大事件日"当界 —— 上一版在该日
    **有事件** ⇒ 构建时数据必然已覆盖到它 ⇒ 该界以下的回溯新增是**零假阳性**的,
    只是窗口比 meta 的 data_end 窄。
    """
    import json
    prev, prev_meta = None, {}
    if os.path.exists(out_path):
        try:
            prev = pd.read_pickle(out_path)
            prev['code'] = prev['code'].astype(str).str.zfill(6)
        except Exception as e:
            print(f'  [守卫] 上一版表读取失败({type(e).__name__}: {e}) ⇒ 无基准, 跳过比对')
            prev = None
        if prev is not None:
            if os.path.exists(META):
                try:
                    with open(META) as fh:
                        prev_meta = json.load(fh)
                except Exception as e:
                    print(f'  [守卫] meta 读取失败({type(e).__name__}) ⇒ 回溯判据降级')
            else:
                print('  [守卫] 无 meta(本守卫首次运行?) ⇒ 回溯判据降级为"上一版最大事件日"')

    # 数据末日: 本版实测(最新 raw 末行), 作为**写盘时**的"现在"
    ends = []
    for c in list(m.code.unique())[:400]:
        try:
            d = pd.read_csv(f'{RAW}/{c}/none.csv', encoding='utf-8-sig', usecols=['日期'])
            ends.append(pd.to_datetime(d['日期']).max())
        except Exception:
            pass
    now_end = max(ends) if ends else None

    if prev is not None:
        key = ['code', 'ex_date']
        pm = prev.set_index(key)
        nm = m.set_index(key)
        added = nm.index.difference(pm.index)
        removed = pm.index.difference(nm.index)
        both = nm.index.intersection(pm.index)
        # 全向量化 —— 逐键 .loc 在 36k 事件上要 40s, refresh 第2步扛不住
        ca_ = nm.loc[both, ['s', 'D', 'mult']].fillna(-999).values
        cb_ = pm.loc[both, ['s', 'D', 'mult']].fillna(-999).values
        changed = both[~(ca_ == cb_).all(axis=1)]
        # 回溯新增 = 新增且 ex_date <= 上一版构建时已覆盖到的数据末日
        prev_end = prev_meta.get('data_end')
        if prev_end is not None:
            base, how = pd.Timestamp(prev_end), f'上一版数据末日 {prev_end}'
        else:
            base = pd.Timestamp(prev['ex_date'].max())      # 降级: 零假阳性但窗口窄
            how = f'降级界=上一版最大事件日 {base.date()}'
        retro = added[added.get_level_values(1) <= base] if len(added) else added
        # ⚠ 2026-10-17: 把回溯新增按"**上一版表里这只股有没有任何事件**"拆成两类 ——
        #   否则新纳入宇宙的整只股票(如 115 只退市股补裸价后首次进表, ~1,500 条历史事件)
        #   会全部报成"改写历史", 把**真信号**(已有码被补进旧事件)淹没在噪声里。
        #   区别在**该码的 bt 文件过去是不是表驱动产出的**:
        #     整只缺席 ⇒ 它过去走"沿用生产"分支(vendor 加性文件), 从没被表驱动过
        #                ⇒ 进表不构成"改写"任何表驱动历史, 只是换了口径(仍须登记)。
        #     码在表内 ⇒ 这些格子过去确实没有事件, 现在有了 ⇒ **真回溯新增, 铁律违反**。
        if len(retro):
            _prev_codes = set(pm.index.get_level_values(0))
            # "上一版表里没有这只码"**不足以**判定它没被表驱动过 —— 零事件的码也缺席
            # (W=raw, k=1, 表里查不到)。精确判据 = 上一版**重建报告**:
            #   本步(build_events_table)在 refresh 里排在重建**之前** ⇒ 此刻 bt 目录里的
            #   `_rebuild_table_report.json` 还是**上一版**的, 正好记录"哪些码是表驱动产出的"。
            #   在报告里 ⇒ 表驱动过(零事件) ⇒ 补进旧事件 = **真回溯新增**。
            #   不在报告里 ⇒ 走的"沿用生产"分支(vendor 加性) ⇒ 只是口径切换。
            _built = set()
            try:
                import json as _json
                with open(f'{PROJ}/data/stock_data/backtrader_data/_rebuild_table_report.json') as _fh:
                    _built = set(_json.load(_fh))
            except Exception as _e:
                print(f'  [守卫] 上一版重建报告读取失败({type(_e).__name__}) ⇒ '
                      f'新码分类降级为"表内是否出现过"')
            _lev0 = retro.get_level_values(0)
            _newcode = ~_lev0.isin(_prev_codes)
            retro_old = retro[(_lev0.isin(_prev_codes)) | (_lev0.isin(_built))]
            retro_new = retro[_newcode & ~_lev0.isin(_built)]
        else:
            retro_new = retro_old = retro
        print(f'\n  [守卫] vs 上一版: 新增 {len(added)}  删除 {len(removed)}  改值 {len(changed)}'
              f'  ({how})')
        if len(retro_new):
            _c = sorted(set(retro_new.get_level_values(0)))
            print(f'  [说明] **整只新码进表 {len(retro_new)} 条 / {len(_c)} 只**'
                  f' (既不在上一版事件表、也不在上一版重建报告里, 如 {_c[:5]})')
            print('      ⇒ 这些码过去走"沿用生产"分支(vendor 加性文件), 从未被表驱动过')
            print('      ⇒ 不算改写表驱动历史, 但是**口径切换** ⇒ 数据重锚, 须登记。')
        if len(retro_old):
            print(f'  !!! **回溯新增 {len(retro_old)} 条**(已有码被补进旧事件) = 改写历史(铁律违反) !!!')
            for k in retro_old[:15]:
                print(f'      {k[0]}  ex_date={pd.Timestamp(k[1]).date()}'
                      f'  s={nm.loc[k, "s"]} D={nm.loc[k, "D"]} src={nm.loc[k, "src"]}')
            if len(retro_old) > 15:
                print(f'      ... 另 {len(retro_old) - 15} 条')
            print('      ⇒ 这些格子在过去的重建里**没有事件**, 现在有了 ⇒ 历史复权价被改写。')
            print('      ⇒ 冻结锚 k 会随之漂移(rebuild 会报"锚漂移"); 四指标铁律豁免, 但须登记。')
        if len(removed):
            print(f'  !!! 事件**消失** {len(removed)} 条 ⇒ 同样改写历史, 上游数据源不稳?')
            for k in list(removed)[:8]:
                print(f'      {k[0]}  ex_date={pd.Timestamp(k[1]).date()}')

    tmp = META + '.tmp'
    with open(tmp, 'w') as fh:
        json.dump({'built_at': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                   'data_end': str(now_end.date()) if now_end is not None else None,
                   'n_events': int(len(m)), 'n_codes': int(m.code.nunique())}, fh)
    os.replace(tmp, META)


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

    # ── 冻结事件pin (2026-10-24, navlaw −27% 事故修复) ─────────────────────────
    # 锚态已登记的事件乘数, 不随源漂移/撤回而改写 (铁律: 数据只append)。
    # 来源: .arms/fix_restore_mult_1023.py 从锚态 bt 字节反解 (逐字节复现验证)。
    # 融合值与pin差>0.5% ⇒ 大声报警(可能是真修正), 但仍按pin执行, 请人工核查。
    _FROZEN = f'{ALT}/events_frozen.pkl'
    if os.path.exists(_FROZEN):
        fz = pd.read_pickle(_FROZEN)
        n_pin = n_ins = 0
        for r in fz.itertuples():
            _ex = pd.Timestamp(r.ex_date)
            sel = (m['code'] == str(r.code).zfill(6)) & (m['ex_date'] == _ex)
            if sel.any():
                old = m.loc[sel, 'mult'].iloc[0]
                if isinstance(old, float) and np.isfinite(old) \
                        and abs(old / r.mult - 1) > 0.005:
                    print(f'  ⚠ [事件pin] {r.code}@{r.ex_date}: 融合值 {old:.6f} 与冻结值 '
                          f'{r.mult:.6f} 差>0.5% — 可能是真修正, 请人工核查 events_frozen.pkl!',
                          flush=True)
                m.loc[sel, ['s', 'D', 'mult', 'src']] = [r.s, r.D, r.mult, 'frozen']
                n_pin += 1
            else:
                m = pd.concat([m, pd.DataFrame([{
                    'code': str(r.code).zfill(6), 'ex_date': _ex, 's': r.s,
                    'D': r.D, 'mult': r.mult, 'src': 'frozen'}])],
                    ignore_index=True)
                n_ins += 1
        m = m.sort_values(['code', 'ex_date']).reset_index(drop=True)
        print(f'  [事件pin] 冻结 {len(fz)} 条: 覆写 {n_pin} / 补插 {n_ins}')

    _backdate_guard(m, a.out)
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
