#!/usr/bin/env python3
"""S4 独立 sleeve 探针 v2 — 生产口径执行层 (2026-10-22).

v1 (s4_sleeve_1022.py) 结论看似强 (s4_only_n2 与主账簿 w20 = 1.196M/Sharpe1.52),
但执行层比生产宽松, 而 S4=底背离 天然落在下跌/小票/ST 上. v2 补齐生产同源闸:

  1. 可交易: raw VWAP(amount/(volume*100)) > 2.0, volume > 100手,
     20日均成交额(shift1) > 5e6, ST 冻结时间线 (get_st_timeline 同一 pkl);
  2. 涨跌停: 开盘判据, 板别 20/30/10%, 容差 0.003 (与 bt_execution L2114 一致);
  3. 跳空罚: 买价 open*(1+slip), 若 gap > 板别阈值(19.5/29.5/9.5%) 再 ×1.03;
     卖价 open*(1-slip), 若 gap < -阈值 再 ×0.97;
  4. 成本: 佣金万1; 印花 2023-08-28 前千1/后万5 (与 _STAMP_ARR 一致);
     impact_cost_enabled=false ⇒ 无冲击成本;
  5. 买卖时点: 调仓日收盘定目标, **次日开盘**执行 (与生产 T+1 open 一致);
     买: 不可交易/涨停/现金不足 ⇒ **当次作废** (生产 _pending={});
     卖: 不可交易/跌停 ⇒ 当日不卖, 到**下一个调仓执行日**再试
     (生产 Step A 也只在 _pending 非空的日子跑);
  6. 停牌行: bt 文件无零成交占位行, 缺行=停牌; 估价沿上一收盘.

其余与 v1 相同: 主账簿 130 个调仓日, S4 子池按 score top-N 等权, 保留交集不重复买卖.
输出 v1 同结构 + 每年混合收益 + 单票贡献 (为 ex-top 彩票判据做准备).

用法: .venv/bin/python .arms/probe/s4_sleeve_1022b.py
"""
import os
import sys
import pickle
import numpy as np
import pandas as pd

PROJ = '/mnt/d/quant'
BT = f'{PROJ}/data/stock_data/backtrader_data'
OUT = f'{PROJ}/.arms/probe/s4_sleeve_1022b'
CELLS = f'{PROJ}/.arms/probe/bp_deep_1021/cells_full.parquet'
SEL = f'{PROJ}/.arms/armout/bp1022_base3/portfolio_selections.csv'
EQ = f'{PROJ}/.arms/armout/bp1022_base3/equity_curve.csv'
INDEX = f'{BT}/sh000001_qfq.csv'
ST_PKL = f'{PROJ}/data/alternative_data/st_timeline_frozen.pkl'

CAP0 = 250_000.0
SLIP = 0.001
COMMISSION = 0.0001
N_PRIMARY = 2
STAMP_CUT = pd.Timestamp('2023-08-28')

os.makedirs(OUT, exist_ok=True)

with open(ST_PKL, 'rb') as _fh:
    ST = pickle.load(_fh)
ST.pop('__meta__', None)


def limit_rate(code):
    if code[:2] in ('30', '68'):
        return 0.20
    if code[0] in ('4', '8'):
        return 0.30
    return 0.10


def gap_thr(code):
    if code[:2] in ('30', '68'):
        return 0.195
    if code[0] in ('4', '8'):
        return 0.295
    return 0.095


def stamp_rate(d):
    return 0.001 if d < STAMP_CUT else 0.0005


def load_index():
    d = pd.read_csv(INDEX, usecols=['datetime', 'close'], parse_dates=['datetime'])
    d = d.dropna().drop_duplicates('datetime').sort_values('datetime')
    return d.set_index('datetime')['close']


_CAL = None
_PX = {}


def get_px(code):
    if code in _PX:
        return _PX[code]
    p = f'{BT}/{code}_qfq.csv'
    if not os.path.exists(p):
        _PX[code] = None
        return None
    d = pd.read_csv(p, usecols=['datetime', 'open', 'close', 'volume', 'amount'],
                    parse_dates=['datetime'])
    d = d.dropna(subset=['open', 'close']).drop_duplicates('datetime').sort_values('datetime')
    d = d.set_index('datetime')
    # ADV20: 与生产同口径 — 指数日历上缺失=0成交额, rolling(20, min5).mean().shift(1)
    full = d.reindex(_CAL)
    adv = full['amount'].fillna(0.0).rolling(20, min_periods=5).mean().shift(1)
    d['prev_close'] = d['close'].shift(1)
    d['pct_open'] = d['open'] / d['prev_close'] - 1.0
    d['raw_vwap'] = np.where((d['amount'].fillna(0) > 0) & (d['volume'].fillna(0) > 0),
                             d['amount'] / (d['volume'].fillna(0) * 100.0), d['close'])
    d['adv20'] = adv.reindex(d.index)
    _PX[code] = d
    return d


def is_st(code, d):
    tl = ST.get(code)
    if not tl:
        return False
    avail, flags = tl
    k = int(np.searchsorted(avail, d.strftime('%Y%m%d'), side='right')) - 1
    return bool(flags[k]) if k >= 0 else False


def bar_info(code, d):
    """返回 (tradable, bar) — 与 bt_execution.tradable 同源."""
    px = get_px(code)
    if px is None or d not in px.index:
        return False, None
    b = px.loc[d]
    ok = (np.isfinite(b['open']) and b['open'] > 0
          and b['volume'] > 100 and b['raw_vwap'] > 2.0
          and np.isfinite(b['adv20']) and b['adv20'] > 5e6
          and not is_st(code, d))
    return bool(ok), b


def buy_fill(code, b):
    p = float(b['open']) * (1.0 + SLIP)
    pc = float(b['prev_close']) if np.isfinite(b['prev_close']) else np.nan
    if np.isfinite(pc) and pc > 0 and p / pc - 1 > gap_thr(code):
        p *= 1.03
    return p


def sell_fill(code, b):
    p = float(b['open']) * (1.0 - SLIP)
    pc = float(b['prev_close']) if np.isfinite(b['prev_close']) else np.nan
    if np.isfinite(pc) and pc > 0 and p / pc - 1 < -gap_thr(code):
        p *= 0.97
    return p


def simulate(mask_col, N, name, cap=CAP0, drop=None):
    """drop = {(signal_date, code)}; 剔除后由下一名候选补位 (彩票判据)."""
    pool = ALL[ALL[mask_col]]
    if drop:
        key = pd.MultiIndex.from_arrays([pool['date'], pool['code']])
        pool = pool[~key.isin(drop)]
    top = (pool.sort_values(['date', 'score'], ascending=[True, False])
               .groupby('date').head(N)[['date', 'code', 'score']])
    tgt_by_date = {d: g.code.tolist() for d, g in top.groupby('date')}

    cal = [d for d in _CAL if START <= d <= END]
    pending = None          # 前一日收盘生成的目标, 次日开盘执行
    holdings = {}
    open_trades = {}
    last_close = {}
    cash = cap
    trades = []
    eq_rows = []
    n_cancel = {'buy_notradable': 0, 'buy_limitup': 0, 'buy_cash': 0,
                'sell_blocked': 0, 'sell_ok': 0, 'buy_ok': 0}

    def eq_at():
        v = cash
        for c, sh in holdings.items():
            v += sh * last_close.get(c, 0.0)
        return v

    for d in cal:
        # ---------- 1) 开盘执行昨日目标 ----------
        if pending is not None:
            sig_d, target = pending
            # 卖: 不在目标中的 (排序保证跨进程确定性)
            for c in sorted(holdings):
                if c in target:
                    continue
                ok, b = bar_info(c, d)
                if not ok:
                    n_cancel['sell_blocked'] += 1
                    continue
                if b['pct_open'] <= -(limit_rate(c) - 0.003):
                    n_cancel['sell_blocked'] += 1
                    continue
                px = sell_fill(c, b)
                sh = holdings.pop(c)
                net = sh * px * (1.0 - COMMISSION - stamp_rate(d))
                cash += net
                tr = open_trades.pop(c, None)
                if tr is not None:
                    tr['exit_date'] = d
                    tr['exit_px'] = px
                    tr['ret'] = (px * (1.0 - COMMISSION - stamp_rate(d))) / tr['entry_gross'] - 1
                    trades.append(tr)
                n_cancel['sell_ok'] += 1
            # 买: 目标中未持有的 (排序保证跨进程确定性)
            buys = sorted(c for c in target if c not in holdings)
            for bi, c in enumerate(buys):
                ok, b = bar_info(c, d)
                if not ok:
                    n_cancel['buy_notradable'] += 1
                    continue
                if b['pct_open'] >= (limit_rate(c) - 0.003):
                    n_cancel['buy_limitup'] += 1
                    continue
                eq_prev = eq_at()
                n_left = max(1, len(buys) - bi)
                budget = min(eq_prev / N, cash / n_left)
                if budget <= 0:
                    n_cancel['buy_cash'] += 1
                    continue
                px = buy_fill(c, b)
                shares = int(budget / (px * (1.0 + COMMISSION)) / 100) * 100
                if shares <= 0:
                    n_cancel['buy_cash'] += 1
                    continue
                cost = shares * px * (1.0 + COMMISSION)
                if cost > cash:
                    shares = int(cash / (px * (1.0 + COMMISSION)) / 100) * 100
                    if shares <= 0:
                        n_cancel['buy_cash'] += 1
                        continue
                    cost = shares * px * (1.0 + COMMISSION)
                cash -= cost
                holdings[c] = shares
                # 必须立即入 mark 表: 否则同日第二笔买入时 eq_at() 把新仓估值为 0,
                # 预算被压缩 (v2 首版 bug, 且 set 顺序随机 ⇒ 跨进程 NAV 漂移)
                last_close[c] = float(b['close'])
                open_trades[c] = {'code': c, 'entry_date': d, 'entry_px': px,
                                  'entry_gross': px * (1.0 + COMMISSION),
                                  'shares': shares, 'sleeve': name,
                                  'signal_date': sig_d}
                n_cancel['buy_ok'] += 1
            pending = None

        # ---------- 2) 收盘 mark ----------
        for c in list(holdings):
            px = get_px(c)
            if px is not None and d in px.index:
                last_close[c] = float(px.loc[d, 'close'])
        eq_rows.append((d, eq_at()))

        # ---------- 3) 调仓日收盘: 生成次日目标 ----------
        if d in tgt_by_date:
            pending = (d, set(tgt_by_date[d]))

    eqs = pd.Series(dict(eq_rows)).sort_index()
    tr = pd.DataFrame(trades)
    if len(tr):
        tr['hold_days'] = (tr.exit_date - tr.entry_date).dt.days
        tr['pnl_yuan'] = tr.shares * (tr.exit_px * (1 - COMMISSION)
                                      - tr.entry_px * (1 + COMMISSION))
    return eqs, tr, n_cancel


def metrics(eq, label, cap=CAP0):
    r = eq.pct_change().dropna()
    yrs = max(len(eq) / 252.0, 1e-9)
    nav = float(eq.iloc[-1])
    cagr = (nav / cap) ** (1 / yrs) - 1 if yrs > 0 else np.nan
    vol = r.std() * np.sqrt(252)
    sharpe = r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else np.nan
    dn = r[r < 0].std() * np.sqrt(252)
    mdd = float((eq / eq.cummax() - 1).min())

    def win_h(h):
        x = eq.iloc[h::h] / eq.shift(h).iloc[h::h] - 1
        return float((x > 0).mean()) if len(x) else np.nan

    m = eq.resample('ME').last().pct_change().dropna()
    y = eq.resample('YE').last().pct_change().dropna()
    return {'arm': label, 'nav': nav, 'total': nav / cap - 1, 'cagr': cagr,
            'vol': vol, 'sharpe': sharpe,
            'sortino': r.mean() * 252 / dn if dn > 0 else np.nan,
            'mdd': mdd, 'calmar': cagr / abs(mdd) if mdd < 0 else np.nan,
            'win_d': float((r > 0).mean()), 'win_5d': win_h(5),
            'win_20d': win_h(20), 'win_m': float((m > 0).mean()),
            'win_y': float((y > 0).mean()), 'n_days': len(eq)}


def trade_stats(tr):
    if not len(tr):
        return {}
    win = tr.ret > 0
    losses = -tr.loc[~win, 'ret'].sum()
    return {'n': len(tr), 'win': float(win.mean()), 'mean': float(tr.ret.mean()),
            'med': float(tr.ret.median()),
            'pf': float(tr.loc[win, 'ret'].sum() / losses) if losses > 0 else np.inf,
            'hold': float(tr.hold_days.mean())}


def main():
    global ALL, _CAL, START, END
    index_series = load_index()
    _CAL = list(index_series.index)

    cells = pd.read_parquet(CELLS, columns=['date', 'code', 'score', 'chan_sell_point',
                                            'S1', 'S2', 'S3', 'S4', 'k'])
    cells = cells[cells.chan_sell_point == 0].copy()
    sel = pd.read_csv(SEL, parse_dates=['date'])
    rebal = sorted(sel.date.unique())
    cells = cells[cells.date.isin(rebal)].copy()
    cells['_u_s4_any'] = cells.S4 == 1
    cells['_u_s4_only'] = (cells.S4 == 1) & (cells.S1 + cells.S2 + cells.S3 == 0)
    cells['_u_s4_all4'] = cells.k == 4
    cells['_u_ctrl_non4'] = cells.S4 == 0
    cells['_u_ctrl_all'] = True
    ALL = cells

    eq_main = pd.read_csv(EQ, parse_dates=['date']).set_index('date')['nav']
    START, END = eq_main.index.min(), eq_main.index.max()

    runs = [('_u_s4_any', 1, 's4_any_n1', CAP0),
            ('_u_s4_any', 2, 's4_any_n2', CAP0),
            ('_u_s4_any', 3, 's4_any_n3', CAP0),
            ('_u_s4_only', 1, 's4_only_n1', CAP0),
            ('_u_s4_only', 2, 's4_only_n2', CAP0),
            ('_u_s4_only', 3, 's4_only_n3', CAP0),
            ('_u_s4_all4', 2, 's4_all4_n2', CAP0),
            ('_u_ctrl_non4', 2, 'ctrl_non4_n2', CAP0),
            ('_u_ctrl_all', 2, 'ctrl_all_n2', CAP0),
            ('_u_s4_only', 2, 's4_only_n2_cap50', 50_000.0),
            ('_u_s4_only', 2, 's4_only_n2_cap75', 75_000.0),
            ('_u_s4_any', 2, 's4_any_n2_cap50', 50_000.0)]
    rows, trows, eqs, cancels = [], [], {}, {}
    for mask, n, label, cap in runs:
        print(f'[sleeve] {label} cap={cap:,.0f} ...', flush=True)
        e, tr, cn = simulate(mask, n, label, cap)
        e.name = label
        eqs[label] = e
        m = metrics(e, label, cap)
        m.update(trade_stats(tr))
        m['cap'] = cap
        m['n_cancel'] = sum(v for k, v in cn.items() if k.startswith(('buy_', 'sell_blocked')))
        rows.append(m)
        cancels[label] = cn
        if len(tr):
            tt = tr.copy()
            tt['sse_entry'] = [index_series.asof(d) for d in tt.entry_date]
            tt['sse_exit'] = [index_series.asof(d) for d in tt.exit_date]
            tt['sse_ret'] = tt.sse_exit / tt.sse_entry - 1
            tt['alpha'] = tt.ret - tt.sse_ret
            tt['sleeve'] = label
            trows.append(tt)

    # ---- ex-top 彩票判据: 剔除贡献最大的 (signal_date, code) 后由下一名补位 ----
    base_tr = pd.concat(trows, ignore_index=True) if trows else pd.DataFrame()
    if len(base_tr):
        for src in ['s4_only_n2', 's4_any_n2']:
            g = base_tr[base_tr.sleeve == src].sort_values('pnl_yuan', ascending=False)
            for k in (1, 3, 5):
                if k > len(g):
                    continue
                drop = set(zip(g.head(k).signal_date, g.head(k).code))
                label = f'{src}_extop{k}'
                mask = '_u_s4_only' if src.startswith('s4_only') else '_u_s4_any'
                print(f'[sleeve] {label} drop={sorted((str(a.date()), b) for a, b in drop)} ...',
                      flush=True)
                e, trx, cn = simulate(mask, 2, label, CAP0, drop=drop)
                e.name = label
                eqs[label] = e
                m = metrics(e, label, CAP0)
                m.update(trade_stats(trx))
                m['cap'] = CAP0
                m['n_cancel'] = sum(v for kk, v in cn.items() if kk.startswith(('buy_', 'sell_blocked')))
                rows.append(m)
                cancels[label] = cn
                if len(trx):
                    tt = trx.copy()
                    tt['sse_entry'] = [index_series.asof(dd) for dd in tt.entry_date]
                    tt['sse_exit'] = [index_series.asof(dd) for dd in tt.exit_date]
                    tt['sse_ret'] = tt.sse_exit / tt.sse_entry - 1
                    tt['alpha'] = tt.ret - tt.sse_ret
                    tt['sleeve'] = label
                    trows.append(tt)

    pd.DataFrame(eqs).to_csv(f'{OUT}/sleeve_equity.csv', float_format='%.4f')
    res = pd.DataFrame(rows).set_index('arm')
    pd.set_option('display.width', 260)
    pd.set_option('display.max_columns', 60)
    print('\n=== SLEEVE STANDALONE (生产口径执行) ===')
    show = ['cap', 'nav', 'total', 'cagr', 'vol', 'sharpe', 'sortino', 'mdd', 'calmar',
            'win_d', 'win_5d', 'win_20d', 'win_m', 'win_y',
            'n', 'win', 'mean', 'med', 'pf', 'hold', 'n_cancel']
    print(res[[c for c in show if c in res.columns]].to_string(float_format=lambda v: f'{v:.4f}'))
    res.to_csv(f'{OUT}/sleeve_metrics.csv', float_format='%.6f')

    tr = pd.concat(trows, ignore_index=True) if trows else pd.DataFrame()
    if len(tr):
        tr.to_csv(f'{OUT}/sleeve_trades.csv', index=False)
        print('\n=== SLEEVE TRADE ALPHA vs SSE ===')
        g = tr.groupby('sleeve').agg(n=('ret', 'size'), win=('ret', lambda x: (x > 0).mean()),
                                     mean=('ret', 'mean'), alpha=('alpha', 'mean'),
                                     alpha_win=('alpha', lambda x: (x > 0).mean()),
                                     pf=('ret', lambda x: x[x > 0].sum() / max(-x[x < 0].sum(), 1e-12)),
                                     hold=('hold_days', 'mean'), pnl=('pnl_yuan', 'sum'))
        print(g.to_string(float_format=lambda v: f'{v:+.4f}'))
        print('\n=== 逐年 alpha (mean) ===')
        tr['y'] = pd.to_datetime(tr.entry_date).dt.year
        print(tr.pivot_table(index='y', columns='sleeve', values='alpha', aggfunc='mean')
                .to_string(float_format=lambda v: f'{v:+.4f}'))
        print('\n=== s4_only_n2 单票 P&L 贡献 top10 ===')
        top10 = (tr[tr.sleeve == 's4_only_n2']
                 .sort_values('pnl_yuan', ascending=False)
                 .head(10)[['code', 'entry_date', 'exit_date', 'ret', 'hold_days', 'pnl_yuan']])
        print(top10.to_string(index=False))
        print('\n=== s4_only_n2 P&L 集中度 ===')
        _g = tr[tr.sleeve == 's4_only_n2'].sort_values('pnl_yuan', ascending=False).copy()
        _g['ey'] = pd.to_datetime(_g.entry_date).dt.year
        _tot = _g.pnl_yuan.sum()
        for _k in (1, 3, 5, 10, 20):
            _s = _g.head(_k).pnl_yuan.sum()
            print(f'top{_k:>2}: {_s:+,.0f} / {_tot:+,.0f} = {_s/_tot*100:5.1f}%')
        _y = _g.groupby('ey').pnl_yuan.sum()
        print('逐年 P&L 占比: ' + ', '.join(f'{y}:{v/_tot*100:+.1f}%' for y, v in _y.items()))
        print('\n=== 逐年逐笔 P&L (元) ===')
        tr['ey'] = pd.to_datetime(tr.entry_date).dt.year
        print(tr.pivot_table(index='ey', columns='sleeve', values='pnl_yuan', aggfunc='sum')
                .to_string(float_format=lambda v: f'{v:+,.0f}'))
        print('\n=== 取消/阻塞统计 ===')
        print(pd.DataFrame(cancels).T.to_string())

    # ---- 混合 (只混同 cap 的 sleeve) ----
    r_main_full = eq_main.pct_change().fillna(0.0)
    blend_rows = []
    blend_rows.append(metrics(pd.Series(eq_main.values, index=eq_main.index), 'main'))
    for label in ['s4_any_n2', 's4_only_n2', 's4_any_n2_extop3', 's4_only_n2_extop1',
                  's4_only_n2_extop3', 's4_only_n2_extop5',
                  's4_all4_n2', 'ctrl_non4_n2', 'ctrl_all_n2']:
        e = eqs[label]
        r_s = e.pct_change().reindex(eq_main.index).fillna(0.0)
        ov = e.index.intersection(eq_main.index)
        print(f'\ncorr({label}, main) = {r_s.loc[ov].corr(r_main_full.loc[ov]):+.4f}')
        for w in (0.10, 0.20, 0.30):
            rb = (1 - w) * r_main_full + w * r_s
            eqt = pd.Series(CAP0 * (1 + rb).cumprod(), index=eq_main.index)
            blend_rows.append(metrics(eqt, f'{label}_w{int(w*100)}'))
    # 小资金可行性: w20 用 cap50 的真实路径 (50k 是 250k 的 20%), w30 用 cap75
    for label, w in [('s4_only_n2_cap50', 0.20), ('s4_any_n2_cap50', 0.20),
                     ('s4_only_n2_cap75', 0.30)]:
        r_s = eqs[label].pct_change().reindex(eq_main.index).fillna(0.0)
        rb = (1 - w) * r_main_full + w * r_s
        eqt = pd.Series(CAP0 * (1 + rb).cumprod(), index=eq_main.index)
        blend_rows.append(metrics(eqt, f'{label}_w{int(w*100)}_realcap'))
    bl = pd.DataFrame(blend_rows).set_index('arm')
    print('\n=== BLEND (main + sleeve, 日再平衡近似) ===')
    bshow = ['nav', 'total', 'cagr', 'vol', 'sharpe', 'sortino', 'mdd', 'calmar',
             'win_d', 'win_5d', 'win_20d', 'win_m', 'win_y']
    print(bl[bshow].to_string(float_format=lambda v: f'{v:.4f}'))
    bl.to_csv(f'{OUT}/blend_metrics.csv', float_format='%.6f')

    # 混合年度收益 (从混合净值重算)
    print('\n=== 混合年度收益 ===')
    by = {}
    for label, row in bl.iterrows():
        # 由 nav/cagr 无法还原路径, 这里重算所选组合
        pass
    chosen = ['main', 's4_any_n2_w20', 's4_only_n2_w20', 's4_only_n2_w30',
              's4_only_n2_extop1_w20', 's4_only_n2_extop3_w20', 's4_any_n2_extop3_w20']
    for lab in chosen:
        if lab == 'main':
            s = eq_main
        else:
            base = lab.rsplit('_w', 1)[0]
            w = int(lab.rsplit('_w', 1)[1]) / 100.0
            r_s = eqs[base].pct_change().reindex(eq_main.index).fillna(0.0)
            rb = (1 - w) * r_main_full + w * r_s
            s = pd.Series(CAP0 * (1 + rb).cumprod(), index=eq_main.index)
        by[lab] = s.resample('YE').last().pct_change().dropna()
    ydf = pd.DataFrame(by)
    ydf.index = ydf.index.year
    print(ydf.to_string(float_format=lambda v: f'{v*100:+.2f}%'))

    # 小资金可行性: cap50/cap75 vs cap250 的同策略收益差
    print('\n=== 资金粒度检查 (同策略不同 cap) ===')
    for a, b in [('s4_only_n2', 's4_only_n2_cap50'),
                 ('s4_only_n2', 's4_only_n2_cap75'),
                 ('s4_any_n2', 's4_any_n2_cap50')]:
        ra = res.loc[a, 'cagr']; rb = res.loc[b, 'cagr']
        print(f'{a} CAGR {ra:+.4f} vs {b} {rb:+.4f}  diff {rb-ra:+.4f}')

    print(f'\n输出: {OUT}')


if __name__ == '__main__':
    main()
