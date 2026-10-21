#!/usr/bin/env python3
"""S4 独立 sleeve 探针 (2026-10-22).

问题 (承接 bp 源级轴):
  信号层 S4(底背离) 是唯一有截面 α 的源 (ex10 ~+0.8%, 三 regime 全正),
  但在主账簿内改排序/闸门全部亏 (账簿位移). 唯一未否证的收割形式 =
  **独立 sleeve / 独立资金** (不动主账簿的选股与权重).

本探针只回答 go/no-go, 不是生产臂:
  - 独立子账簿在**主账簿同一调仓日** (portfolio_selections 的 130 个日期) 选股,
    次交易日开盘执行 (与生产 T+1 open 一致);
  - 每个 universe 按 score 取 top N, 等权; 不在目标的持仓次日开盘卖出;
    目标中未持有的买入 (保留交集不重复买卖, 降换手);
  - 涨停不追 (按板 20/30/10%), 跌停/停牌顺延卖出 (最多10个交易日);
  - 成本: 买 0.0011 (滑点0.001+佣金0.0001), 卖 0.0016 (+印花0.0005);
  - 现金 0 收益; 每日收盘 mark-to-market.

Universe:
  s4_any      S4=1 (任意其他源), chan_sell=0
  s4_only     S4=1 且 S1=S2=S3=0
  s4_all4     k=4 (S1..S4 全确认)
  ctrl_non4   S4=0
  ctrl_all    全部 buy (无 S4 过滤)  ← 负控: 若它不差, S4 不是增量

敏感性: s4_any 的 N=1/2/3; 其余 N=2.

混合: 用主账簿日收益 r_main 与 sleeve 日收益 r_s 做 w=10/20/30% 日再平衡混合,
只作规模近似 (假设主账簿收益可按比例缩放).

用法: .venv/bin/python .arms/probe/s4_sleeve_1022.py
"""
import os
import numpy as np
import pandas as pd

PROJ = '/mnt/d/quant'
BT = f'{PROJ}/data/stock_data/backtrader_data'
OUT = f'{PROJ}/.arms/probe/s4_sleeve_1022'
CELLS = f'{PROJ}/.arms/probe/bp_deep_1021/cells_full.parquet'
SEL = f'{PROJ}/.arms/armout/bp1022_base3/portfolio_selections.csv'
EQ = f'{PROJ}/.arms/armout/bp1022_base3/equity_curve.csv'
INDEX = f'{BT}/sh000001_qfq.csv'

CAP0 = 250_000.0
COST_BUY = 0.0011
COST_SELL = 0.0016
MAX_DEFER = 10          # 最多顺延交易日
N_PRIMARY = 2

os.makedirs(OUT, exist_ok=True)


def limit_thr(code):
    if code.startswith(('300', '301', '688', '689')):
        return 0.195
    if code.startswith(('4', '8')):
        return 0.295
    return 0.095


def load_index():
    d = pd.read_csv(INDEX, usecols=['datetime', 'close'], parse_dates=['datetime'])
    d = d.dropna().drop_duplicates('datetime').sort_values('datetime')
    return d.set_index('datetime')['close']


_PX = {}


def get_px(code):
    if code in _PX:
        return _PX[code]
    p = f'{BT}/{code}_qfq.csv'
    if not os.path.exists(p):
        _PX[code] = None
        return None
    d = pd.read_csv(p, usecols=['datetime', 'open', 'close'], parse_dates=['datetime'])
    d = d.dropna(subset=['open', 'close']).drop_duplicates('datetime').sort_values('datetime')
    d = d.set_index('datetime')
    d['prev_close'] = d['close'].shift(1)
    _PX[code] = d
    return d


def simulate(univ_mask, N, name):
    """返回 (equity Series, trades DataFrame, exposure Series)."""
    cells = ALL
    pool = cells[cells['_u_' + name]]
    top = (pool.sort_values(['date', 'score'], ascending=[True, False])
               .groupby('date').head(N)[['date', 'code', 'score']])
    tgt_by_date = {d: g.code.tolist() for d, g in top.groupby('date')}
    rebal = set(tgt_by_date)

    cal = list(INDEX_SERIES.index)
    cal = [d for d in cal if d >= START and d <= END]
    target = set()
    pending = []          # {'side','code','tries'}
    holdings = {}         # code -> shares
    open_trades = {}      # code -> trade dict (未平仓)
    last_close = {}
    cash = CAP0
    trades = []
    eq_rows, exp_rows = [], []

    def equity_at(price_map):
        v = cash
        for c, sh in holdings.items():
            v += sh * price_map.get(c, last_close.get(c, 0.0))
        return v

    for i, d in enumerate(cal):
        # ---------- 1) 开盘执行 pending ----------
        still = []
        # 先卖 (释放现金)
        for a in [x for x in pending if x['side'] == 'sell']:
            code = a['code']
            px = get_px(code)
            bar = px.loc[d] if (px is not None and d in px.index) else None
            if bar is None or not np.isfinite(bar['open']) or bar['open'] <= 0:
                a['tries'] += 1
                if a['tries'] <= MAX_DEFER:
                    still.append(a)
                continue
            pc = bar['prev_close']
            if np.isfinite(pc) and pc > 0 and bar['open'] / pc - 1 <= -limit_thr(code):
                a['tries'] += 1           # 跌停卖不出, 顺延
                if a['tries'] <= MAX_DEFER:
                    still.append(a)
                continue
            sh = holdings.pop(code, 0)
            if sh > 0:
                net = sh * float(bar['open']) * (1 - COST_SELL)
                cash += net
                tr = open_trades.pop(code, None)
                if tr is not None and 'entry_px_gross' in tr:
                    tr['exit_date'] = d
                    tr['exit_px'] = float(bar['open'])
                    tr['exit_net'] = float(bar['open']) * (1 - COST_SELL)
                    tr['ret'] = tr['exit_net'] / tr['entry_px_gross'] - 1
                    trades.append(tr)
            last_close[code] = float(bar['close'])
        # 再买
        buys = [x for x in pending if x['side'] == 'buy']
        for bi, a in enumerate(buys):
            code = a['code']
            px = get_px(code)
            bar = px.loc[d] if (px is not None and d in px.index) else None
            if bar is None or not np.isfinite(bar['open']) or bar['open'] <= 0:
                a['tries'] += 1
                if a['tries'] <= MAX_DEFER:
                    still.append(a)
                continue
            pc = bar['prev_close']
            if np.isfinite(pc) and pc > 0 and bar['open'] / pc - 1 >= limit_thr(code):
                continue                  # 涨停买不到, 放弃该票
            if code in holdings:
                continue
            eq_prev = equity_at(last_close)
            n_left = max(1, len(buys) - bi)
            budget = min(eq_prev / N, cash / n_left)
            if budget <= 0:
                continue
            gross = float(bar['open']) * (1 + COST_BUY)
            shares = int(budget / gross / 100) * 100
            if shares <= 0:
                continue
            cash -= shares * gross
            holdings[code] = holdings.get(code, 0) + shares
            open_trades[code] = {
                'code': code, 'entry_date': d, 'entry_px': float(bar['open']),
                'entry_px_gross': gross, 'shares': shares,
                'sleeve': name, 'signal_date': a.get('signal_date')}
            last_close[code] = float(bar['close'])
        pending = still

        # ---------- 2) 收盘 mark ----------
        for c in list(holdings):
            px = get_px(c)
            if px is not None and d in px.index:
                last_close[c] = float(px.loc[d, 'close'])
        eq = equity_at(last_close)
        eq_rows.append((d, eq))
        exp_rows.append((d, (eq - cash) / eq if eq > 0 else 0.0))

        # ---------- 3) 调仓日: 设置次日 pending ----------
        if d in rebal:
            target = set(tgt_by_date.get(d, []))
            _pend_sell = {x['code'] for x in pending if x['side'] == 'sell'}
            _pend_buy = {x['code'] for x in pending if x['side'] == 'buy'}
            # 卖出不在目标的
            for c in list(holdings):
                if c not in target and c not in _pend_sell:
                    pending.append({'side': 'sell', 'code': c, 'tries': 0,
                                    'trade': open_trades.get(c)})
            # 买入目标中未持有的
            for c in target:
                if c not in holdings and c not in _pend_buy:
                    pending.append({'side': 'buy', 'code': c, 'tries': 0,
                                    'signal_date': d})
    eqs = pd.Series(dict(eq_rows)).sort_index()
    exp = pd.Series(dict(exp_rows)).sort_index()
    tr = pd.DataFrame(trades)
    if len(tr):
        tr = tr[tr['exit_date'].notna()].copy()
        tr['hold_days'] = (tr.exit_date - tr.entry_date).dt.days
    return eqs, tr, exp


def metrics(eq, label):
    r = eq.pct_change().dropna()
    yrs = len(eq) / 252.0
    nav = float(eq.iloc[-1])
    cagr = (nav / CAP0) ** (1 / yrs) - 1 if yrs > 0 else np.nan
    vol = r.std() * np.sqrt(252)
    sharpe = r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else np.nan
    dn = r[r < 0].std() * np.sqrt(252)
    mdd = float((eq / eq.cummax() - 1).min())
    def win_h(h):
        x = eq.iloc[h::h] / eq.shift(h).iloc[h::h] - 1
        return float((x > 0).mean()) if len(x) else np.nan
    m = eq.resample('ME').last().pct_change().dropna()
    y = eq.resample('YE').last().pct_change().dropna()
    return {'arm': label, 'nav': nav, 'total': nav / CAP0 - 1, 'cagr': cagr,
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
    global ALL, INDEX_SERIES, START, END
    INDEX_SERIES = load_index()
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

    runs = [('s4_any', 1), ('s4_any', 2), ('s4_any', 3),
            ('s4_only', 2), ('s4_all4', 2), ('ctrl_non4', 2), ('ctrl_all', 2)]
    rows, trows, eqs = [], [], {}
    for univ, n in runs:
        label = f'{univ}_n{n}'
        print(f'[sleeve] {label} ...', flush=True)
        e, tr, exp = simulate(univ, n, univ)
        e.name = label
        eqs[label] = e
        m = metrics(e, label)
        ts = trade_stats(tr)
        m.update(ts)
        m['avg_exp'] = float(exp.mean())
        m['turnover_trades_per_year'] = len(tr) / (len(e) / 252.0)
        rows.append(m)
        if len(tr):
            tt = tr.copy()
            ss = INDEX_SERIES
            tt['sse_entry'] = [ss.asof(d) for d in tt.entry_date]
            tt['sse_exit'] = [ss.asof(d) for d in tt.exit_date]
            tt['sse_ret'] = tt.sse_exit / tt.sse_entry - 1
            tt['alpha'] = tt.ret - tt.sse_ret
            tt['sleeve'] = label
            trows.append(tt)

    res = pd.DataFrame(rows).set_index('arm')
    pd.set_option('display.width', 260)
    pd.set_option('display.max_columns', 60)
    print('\n=== SLEEVE STANDALONE ===')
    show = ['nav', 'total', 'cagr', 'vol', 'sharpe', 'sortino', 'mdd', 'calmar',
            'win_d', 'win_5d', 'win_20d', 'win_m', 'win_y',
            'n', 'win', 'mean', 'med', 'pf', 'hold', 'avg_exp',
            'turnover_trades_per_year']
    print(res[show].to_string(float_format=lambda v: f'{v:.4f}'))
    res.to_csv(f'{OUT}/sleeve_metrics.csv', float_format='%.6f')

    tr = pd.concat(trows, ignore_index=True) if trows else pd.DataFrame()
    if len(tr):
        tr.to_csv(f'{OUT}/sleeve_trades.csv', index=False)
        print('\n=== SLEEVE TRADE ALPHA vs SSE ===')
        g = tr.groupby('sleeve').agg(n=('ret', 'size'), win=('ret', lambda x: (x > 0).mean()),
                                     mean=('ret', 'mean'), alpha=('alpha', 'mean'),
                                     alpha_win=('alpha', lambda x: (x > 0).mean()),
                                     pf=('ret', lambda x: x[x > 0].sum() / max(-x[x < 0].sum(), 1e-12)),
                                     hold=('hold_days', 'mean'))
        print(g.to_string(float_format=lambda v: f'{v:+.4f}'))
        print('\n=== 逐年 alpha (mean) ===')
        tr['y'] = pd.to_datetime(tr.entry_date).dt.year
        print(tr.pivot_table(index='y', columns='sleeve', values='alpha', aggfunc='mean')
                .to_string(float_format=lambda v: f'{v:+.4f}'))

    # ---- 混合 ----
    # 日再平衡近似: 主账簿收益按 (1-w) 缩放, sleeve 按 w; 首行=CAP0 (收益0).
    r_main_full = eq_main.pct_change().fillna(0.0)
    blend_rows = []
    base_m = metrics(pd.Series(eq_main.values, index=eq_main.index), 'main')
    blend_rows.append(base_m)
    for label in ['s4_any_n2', 's4_only_n2', 's4_all4_n2', 'ctrl_non4_n2', 'ctrl_all_n2']:
        e = eqs[label]
        r_s = e.pct_change().reindex(eq_main.index).fillna(0.0)
        ov = e.index.intersection(eq_main.index)
        print(f'\ncorr({label}, main) = {r_s.loc[ov].corr(r_main_full.loc[ov]):+.4f}')
        for w in (0.10, 0.20, 0.30):
            rb = (1 - w) * r_main_full + w * r_s
            eqt = pd.Series(CAP0 * (1 + rb).cumprod(), index=eq_main.index)
            m = metrics(eqt, f'{label}_w{int(w*100)}')
            blend_rows.append(m)
    bl = pd.DataFrame(blend_rows).set_index('arm')
    print('\n=== BLEND (main + sleeve, daily rebalanced) ===')
    print(bl[['nav', 'total', 'cagr', 'vol', 'sharpe', 'sortino', 'mdd', 'calmar',
              'win_d', 'win_5d', 'win_20d', 'win_m', 'win_y']]
          .to_string(float_format=lambda v: f'{v:.4f}'))
    bl.to_csv(f'{OUT}/blend_metrics.csv', float_format='%.6f')

    # 年度收益
    yd = {}
    for label in ['main'] + [x for x in bl.index if x != 'main']:
        pass
    print('\n=== 年度收益 (main vs best blends) ===')
    ye = {}
    for label, series in [('main', eq_main)] + [(k, eqs[k]) for k in eqs]:
        ye[label] = series.resample('YE').last().pct_change().dropna()
    print(pd.DataFrame(ye).set_index(pd.DataFrame(ye).index.year)
          .to_string(float_format=lambda v: f'{v*100:+.2f}%'))

    print(f'\n输出: {OUT}')


if __name__ == '__main__':
    main()
