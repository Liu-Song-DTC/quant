#!/usr/bin/env python3
"""census_1020 判读 (2026-10-20夜 04:00, 判据先于结果写定).

背景: 修复态(cb245b3)下重扫旧采纳旋钮 — bp/mp/fec/slk/ea2 全部在
加性数据+#365冻结bug时代(≤10/15)采纳, bi_buy 活化后 bp_confirm 闸
首次以设计语义运行(buy=True −10.6%) ⇒ 面形全部作废须复扫。

判据校准(烟测于 fix365 vs 生产, 先于臂结果):
  · 终值比 block-bootstrap CI: −41% 真效应 ⇒ CI [0.257,1.277] 含 1 ⇒ 无判别力;
  · 月度Δ均值 CI: −0.76%/月 ⇒ CI [−2.02%,+0.60%] 含 0 ⇒ 同样无判别力。
  ⇒ NAV 级统计检验在本系统对路径依赖型差异**结构性失明**(与 warmup 臂族
    结论一致)。改用四条判据, 统计牙齿放票级配对 α:

预注册采纳判据(同时满足):
  (a) 效应量: ΔNAV ≥ +20k (≈5.5% 基线; 按机制普查 150k/2.4M≈6% 比例外推);
  (b) 逐年: 同号 或 最大单年 |Δ| < 60%·|总Δ| (防单年彩票);
  (c) 票级: 臂独有票 fwd10 α ≥ 基线独有票 α (旋钮没有"换进更差的票");
  (d) 机制方向与旋钮语义一致(人工)。
  CI/参考数值仍打印但不作门槛。CI含0的臂 = 明日探针候选, 不今晚采纳。

用法: cd /mnt/d/quant/strategy && ../.venv/bin/python \
        /mnt/d/quant/.arms/probe/judge_census_1020.py
"""
import json
import os

import numpy as np
import pandas as pd

RVD = 'rolling_validation_results'
TAGS = ['bp1', 'bp3', 'mp3', 'mp5', 'fec0', 'fec2', 'slk15', 'ea2old']
BASE = 'base1020fix'
PROD_FROZEN = 615042.97   # 生产冻结态(10-19, 旧bug) 参考
FIX365 = 363724.62        # fix365臂(todate 10-19, 同代码) 参考
BT = '../data/stock_data/backtrader_data/{}_qfq.csv'
FWD = 10
ADOPT_DELTA = 20_000      # 判据(a) 效应量门槛


def load_nav(tag):
    df = pd.read_csv(f'{RVD}/equity_curve.csv.{tag}')
    df['date'] = pd.to_datetime(df['date'])
    return df.set_index('date')['nav'].astype(float)


def load_json(tag):
    try:
        with open(f'{RVD}/backtest_report.json.{tag}') as fh:
            return json.load(fh)
    except FileNotFoundError:
        return None


def load_picks(tag):
    try:
        df = pd.read_csv(f'{RVD}/portfolio_selections.csv.{tag}',
                         dtype={'code': str})
        return set(zip(df['code'], df['date'].astype(str)))
    except FileNotFoundError:
        return None


_cache = {}


def fwd_alpha(keys):
    """(code, 'YYYY-MM-DD') 集合 → fwd10 收盘收益均值(逐股CSV缓存)。"""
    out = []
    for c, d in keys:
        if c not in _cache:
            try:
                _cache[c] = pd.read_csv(BT.format(c), usecols=['datetime', 'close'])
            except Exception:
                _cache[c] = None
        df = _cache[c]
        if df is None:
            continue
        m = df.index[df['datetime'] == d]
        if len(m) == 0:
            continue
        j = m[0]
        if j + FWD < len(df):
            out.append(df['close'].iloc[j + FWD] / df['close'].iloc[j] - 1)
    return np.array(out)


def block_bootstrap_ratio(bnav, nnav, reps=1000, seed=7):
    """配对月块bootstrap 终值比(仅参考, 已证无判别力)。"""
    common = bnav.index.intersection(nnav.index)
    b, n = bnav.loc[common].values, nnav.loc[common].values
    br = np.diff(b) / b[:-1]
    nr = np.diff(n) / n[:-1]
    keys = np.array([f'{d.year}-{d.month:02d}' for d in common[1:]])
    uk = pd.unique(keys)
    blocks = {k: np.where(keys == k)[0] for k in uk}
    rng = np.random.default_rng(seed)
    ratios = np.empty(reps)
    for r in range(reps):
        pb = pn = 1.0
        for k in rng.choice(uk, size=len(uk), replace=True):
            i = blocks[k]
            pb *= float(np.prod(1.0 + br[i]))
            pn *= float(np.prod(1.0 + nr[i]))
        ratios[r] = pn / pb
    lo, hi = np.percentile(ratios, [2.5, 97.5])
    return lo, hi


def yearly_rets(nav):
    g = nav.groupby(nav.index.year)
    first_val = nav.iloc[0]
    out, prev = {}, None
    for yr, s in g:
        end = s.iloc[-1]
        start = prev if prev is not None else first_val
        out[yr] = end / start - 1.0
        prev = end
    return out


def main():
    base = load_nav(BASE)
    jb = load_json(BASE) or {}
    mb = jb.get('metrics', {})
    print(f"基线 {BASE}: NAV {base.iloc[-1]:,.0f}  "
          f"sharpe {mb.get('sharpe', float('nan')):.3f}  "
          f"mdd {mb.get('mdd', float('nan')):.2%}")
    print(f"参考: 生产冻结态 {PROD_FROZEN:,.0f} | fix365臂(10-19) {FIX365:,.0f}\n")

    base_yr = yearly_rets(base)
    base_picks = load_picks(BASE) or set()
    base_a = fwd_alpha(base_picks)

    for t in TAGS:
        if not os.path.exists(f'{RVD}/equity_curve.csv.{t}'):
            continue
        nav = load_nav(t)
        j = load_json(t) or {}
        m = j.get('metrics', {})
        lo, hi = block_bootstrap_ratio(base, nav)
        pk = load_picks(t) or set()
        jac = len(pk & base_picks) / max(1, len(pk | base_picks))
        d = nav.iloc[-1] - base.iloc[-1]
        print(f"── {t}: NAV {nav.iloc[-1]:,.0f}  Δ {d:+,.0f}  "
              f"sharpe {m.get('sharpe', float('nan')):.3f}  "
              f"mdd {m.get('mdd', float('nan')):.2%}  "
              f"Jacc {jac:.2f}  CI比[{lo:.2f},{hi:.2f}]")
        # (a) 效应量
        ok_a = d >= ADOPT_DELTA
        # (b) 逐年
        yr = yearly_rets(nav)
        dy = {y: yr.get(y, 0) - base_yr.get(y, 0) for y in sorted(base_yr)}
        tot = sum(dy.values())
        signs = {s for s in np.sign(list(dy.values())) if s != 0}
        same_sign = len(signs) <= 1
        mx = max((abs(v) for v in dy.values()), default=0)
        dom = (mx / abs(tot) > 0.60) if abs(tot) > 1e-9 else False
        ok_b = same_sign or not dom
        print(f"   逐年Δ: " + "  ".join(f"{y}:{v:+.1%}" for y, v in dy.items())
              + f"  同号={same_sign} 单年主导={dom}")
        # (c) 票级 α — 纯超集臂(bp1: lost≈0)/纯子集臂(mp3: un≈0)时换对照
        un = pk - base_picks
        lost = base_picks - pk
        ua, la = fwd_alpha(un), fwd_alpha(lost)
        ua_m = float(np.nanmean(ua)) if len(ua) else float('nan')
        la_m = float(np.nanmean(la)) if len(la) else float('nan')
        ba_m = float(np.nanmean(base_a))
        if len(la) >= 30:
            ref, ref_n = la_m, len(la)
        else:                     # 纯超集: 新增票对照基线全体
            ref, ref_n = ba_m, len(base_a)
        ok_c = (len(ua) >= 30 and ua_m >= ref) if len(ua) >= 30 else None
        print(f"   票级: 臂独有 n={len(ua)} α={ua_m:+.3%} | "
              f"基线独有 n={len(la)} α={la_m:+.3%} | "
              f"基线全体 n={len(base_a)} α={ba_m:+.3%} | 对照={ref:+.3%}(n={ref_n})")
        c_txt = {True: '过', False: '不过', None: 'n<30信息性'}[ok_c]
        verdict = '★过(a)(b)(c)' if (ok_a and ok_b and ok_c) else \
                  f"(a)={ok_a} (b)={ok_b} (c)={c_txt}"
        print(f"   判据: {verdict}  (d)机制=人工\n")

    print("采纳=(a)+(b)+(c)+(d)全过; 否则基线态出单, CI含0臂记为明日探针候选。")


if __name__ == '__main__':
    main()
