#!/usr/bin/env python3
"""转债hybrid10 sleeve (实盘路径预备, 默认关闭)。

研究依据: strategy/analysis/cov_research_20260926.md v3裁决 —
hybrid = rank(债价) + w×rank(纯债溢价率), w=1, n=10, cap=130:
月Sharpe 1.02 (lp10 0.90), 年化+14.8%, MDD −15.5%, 相关vs生产 0.087,
危机月12/12跑赢生产, 年分解无单年驱动, 参数面60点宽平台无彩票签名。
出资模型: 生产组合不动, 闲置现金w%×hybrid (MV网格 w=0.10→1.86/0.20→1.92,
生产1.78; 接入起步w=10-20%)。月频调仓(半月频0.48 << 月频0.90)。

机制 (与回测引擎 cov_mech_v3.py run() 逐口径对齐):
- 选入: 债价<cap(130) 且 纯债价值>0 且 最近交易日距今≤max_stale_days(停牌/退市跳过)
- 评分: rank(债价) + w×rank(债价/纯债价值−1), 取最小n只
- 换仓: 全换仓差分 — 卖 held−picks, 买 picks−held; held∩picks 不动(不回补不trim)
  (回测逐月等权持仓、成本只按换手计55%; 差分换仓=回测经济学的忠实实盘化,
  还省掉回测里对延续持仓的"全卖全买"隐含成本)
- 成本: 佣金万3双侧(转债无印花税), 回测已计入
- 强赎安全(实盘专用; 回测cap<130下零触发): 近30日转股价值≥130天数
  ≥redemption_days → 跳过买入(临近强赎公告, 不建新仓)
- 单位: 张(1000元面值), 最小1手=10张; 债价报价=每张

本模块只产建议块(写入trade_orders.json的cb_sleeve键), 不下真实单 —
QMT侧执行脚本不在本仓库。开关: yaml live_monitoring.cb_sleeve.enabled
(live_monitoring是信号指纹豁免节, 加键零fp漂移)。任何异常都不raise —
cb失败只降级跳过, 绝不影响股票订单(调用方也包try/except双保险)。

数据: data/alternative_data/cb_daily/{code}.csv (sina日线) +
cb_value/{code}.csv (EM价值分析: 转股价值/纯债价值, PIT正确含下修分红调整) +
cb_list.csv (债券列表) — 由 data/cb_sleeve_refresh.py 维护, akshare中继代理链路。
状态: cb_sleeve_state.json (项目根) — 月度调仓节奏 + 持仓张数跟踪。
"""

import os
import json

import pandas as pd
import numpy as np

_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CACHE_DIR = os.path.join(_PROJECT_DIR, 'data', 'alternative_data')
STATE_FILE = os.path.join(_PROJECT_DIR, 'cb_sleeve_state.json')

DEFAULT_CFG = dict(
    enabled=False,
    n=10,                 # 持仓券数 (参数面宽平台内点)
    w=1.0,                # 纯债溢价率rank权重 (hybrid机理)
    cap=130,              # 债价上限 (挡强赎区, 偏差上界=0)
    alloc_ratio=0.10,     # 闲置现金出资比例 (MV网格稳健起点)
    min_total=20000,      # 最低出资额, 低于此只卖不买 (10券×1手≈1.1万)
    lot=10,               # 1手=10张 (沪深转债最小交易单位)
    max_stale_days=5,     # 停牌/退市判定: 最新交易日距今超过N天
    redemption_days=15,   # 强赎检查: 近30日转股价值≥130天数阈值
)

LOT_MIN_ZHANG = 10        # 最小申报单位(张)
COST_NOTE = '佣金万3双侧(转债无印花税, 回测已计)'


def _sym(code: str) -> str:
    """裸码 → QMT下单格式 113050.SH / 123456.SZ (11xxxx=沪, 12xxxx=深)。"""
    return f"{code}.SH" if code.startswith('11') else f"{code}.SZ"


def load_state() -> dict:
    """读 cb_sleeve_state.json; 缺失/损坏 → 空态。"""
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, encoding='utf-8') as f:
                st = json.load(f)
            if isinstance(st, dict) and isinstance(st.get('positions'), dict):
                return st
        except Exception:
            pass
    return {'month': None, 'positions': {}}


def save_state(state: dict) -> None:
    with open(STATE_FILE, 'w', encoding='utf-8') as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def load_snapshot(cache_dir: str, asof, strict: bool = False) -> dict:
    """加载当前截面 {code: {price, bv, cv_recent, last_date}}。

    只读cb_daily/cb_value缓存, 无网络。默认(live口径): 日线与价值都取
    ≤asof 的最后一行 — 实盘调仓日上, 滞后1-2天的价值行是真实可得信息。
    strict=True(身份验证口径): 只在 t0 当日有行时才取值, 否则视为缺失 —
    与回测引擎 reindex 后的 P.loc[t0]/BV.loc[t0] 逐位对齐, 供恒等烟测用。
    缓存缺失/损坏的券静默跳过 — 与回测"可交易集=日线存在性"同口径。
    """
    daily_dir = os.path.join(cache_dir, 'cb_daily')
    value_dir = os.path.join(cache_dir, 'cb_value')
    snap = {}
    if not os.path.isdir(daily_dir):
        return snap
    for fn in sorted(os.listdir(daily_dir)):
        if not fn.endswith('.csv'):
            continue
        code = fn[:-4]
        try:
            d = pd.read_csv(os.path.join(daily_dir, fn), parse_dates=['date'])
            d = d[d['date'] <= asof]
            if d.empty:
                continue
            if strict:
                d = d[d['date'] == d['date'].max()]  # 仅取最后一日
                if d.empty or d['date'].iloc[-1] != asof:
                    continue  # t0当日无行情 → 与回测tradable=NaN同义
            else:
                d = d.sort_values('date')
            price = float(d['close'].iloc[-1])
            if not np.isfinite(price) or price <= 0:
                continue
            last_date = d['date'].iloc[-1]
            bv = np.nan
            cv_recent = []
            vf = os.path.join(value_dir, fn)
            if os.path.exists(vf):
                v = pd.read_csv(vf, parse_dates=['日期'])
                v = v[v['日期'] <= asof]
                if strict:
                    v = v[v['日期'] == v['日期'].max()]  # 与BV.loc[t0]对齐
                    if v.empty or v['日期'].iloc[-1] != asof:
                        v = v.iloc[0:0]
                if not v.empty:
                    bv = float(v['纯债价值'].iloc[-1])
                    v30 = v[v['日期'] > asof - pd.Timedelta(days=45)]
                    cv_recent = v30['转股价值'].dropna().tolist()
            snap[code] = dict(price=price, bv=bv, cv_recent=cv_recent,
                              last_date=last_date)
        except Exception:
            continue
    return snap


def select_hybrid(snap: dict, asof, n: int = 10, w: float = 1.0,
                  cap: float = 130, max_stale_days: int = 5,
                  redemption_days: int = 15) -> list:
    """hybrid10选择器 — cov_mech_v3.py mk_hybrid 的忠实移植 + 实盘安全层。

    返回按评分升序的裸码列表(≤n只)。过滤: 纯债价值缺失/≤0、债价≥cap、
    停牌(日线陈旧)、强赎触发中。
    """
    rows = []
    for code, s in snap.items():
        p = s['price']
        bv = s['bv']
        if not (np.isfinite(bv) and bv > 0):
            continue
        if not (p < cap):
            continue
        if (asof - s['last_date']).days > max_stale_days:
            continue
        if redemption_days and len(s['cv_recent']) >= 30 and \
                sum(1 for c in s['cv_recent'] if c >= 130) >= redemption_days:
            continue  # redemption_days=0 → 安全检查关闭(身份烟测/对照)
        rows.append((code, p, p / bv - 1.0))
    if not rows:
        return []
    df = pd.DataFrame(rows, columns=['code', 'p', 'prem'])
    score = df['p'].rank() + w * df['prem'].rank()
    return df.assign(score=score).sort_values('score').head(n)['code'].tolist()


def compute_cb_sleeve(post_order_cash: float, cfg: dict, state: dict,
                      snap: dict, asof, dry_run: bool = False) -> dict:
    """主入口: 生成转债调仓建议块(不落盘, 落盘由调用方按dry_run裁决)。

    Args:
        post_order_cash: 股票订单执行后预计剩余可用资金(与repo_sweep同口径)。
        cfg: yaml live_monitoring.cb_sleeve 子节 ({} → 默认关闭)。
        state: load_state() 结果。
        snap: load_snapshot() 结果。
        asof: 目标调仓日 (pd.Timestamp/date)。
        dry_run: True时state不写入返回值之外 (调用方不落盘)。

    Returns:
        {'enabled', 'rebalanced', 'month', 'picks', 'orders', 'alloc_total',
         'spend_est', 'reason', 'notes', 'new_state'}
        orders元素: {'action': open/close, 'code': 113050.SH格式,
                     'qty_zhang', 'est_price', 'est_amount'}。
    """
    cfg = dict(DEFAULT_CFG, **(cfg or {}))
    empty = dict(enabled=False, rebalanced=False, picks=[], orders=[],
                 alloc_total=0.0, spend_est=0.0, reason='', notes=[])
    if not cfg.get('enabled', False):
        empty['reason'] = 'cb_sleeve关闭(yaml live_monitoring.cb_sleeve.enabled=false)'
        return empty

    month = str(asof)[:7]
    state = state or {'month': None, 'positions': {}}
    base = dict(empty, enabled=True, month=month, new_state=state)
    if state.get('month') == month:
        base['reason'] = f'本月({month})已调仓, 月频节奏跳过'
        return base
    if post_order_cash is None or post_order_cash <= 0:
        base['reason'] = '订单后无剩余现金, 无法出资转债'
        return base

    n = int(cfg.get('n', 10))
    picks = select_hybrid(snap, asof, n=n, w=float(cfg.get('w', 1.0)),
                          cap=float(cfg.get('cap', 130)),
                          max_stale_days=int(cfg.get('max_stale_days', 5)),
                          redemption_days=int(cfg.get('redemption_days', 15)))

    held = {str(c): int(q) for c, q in (state.get('positions') or {}).items()}
    alloc_ratio = min(max(float(cfg.get('alloc_ratio', 0.10)), 0.0), 1.0)
    alloc_total = post_order_cash * alloc_ratio
    min_total = float(cfg.get('min_total', 20000))
    lot = int(cfg.get('lot', 10))
    stale = int(cfg.get('max_stale_days', 5))

    orders, notes = [], []
    new_positions = dict(held)

    # ── 卖出: held − picks (全换仓差分; 停牌持仓无法卖, 保留继续跟踪) ──
    suspended_held = []
    for c in [c for c in held if c not in picks]:
        s = snap.get(c)
        if s is None or s['price'] <= 0 or (asof - s['last_date']).days > stale:
            suspended_held.append(c)
            continue
        qty = held[c]
        orders.append({'action': 'close', 'code': _sym(c), 'qty_zhang': qty,
                       'est_price': round(s['price'], 3),
                       'est_amount': round(qty * s['price'], 2)})
        new_positions.pop(c, None)

    # ── 买入: picks − held, 每券等额预算, 不足1手跳过(不追垫) ──
    spend_est = 0.0
    budget_cut = alloc_total < min_total
    if not budget_cut:
        per_pick = alloc_total / max(len(picks), 1)
        for c in [c for c in picks if c not in held]:
            s = snap.get(c)
            if s is None or s['price'] <= 0:
                continue
            qty = int(per_pick / s['price'] / lot) * lot
            if qty < LOT_MIN_ZHANG:
                continue
            orders.append({'action': 'open', 'code': _sym(c), 'qty_zhang': qty,
                           'est_price': round(s['price'], 3),
                           'est_amount': round(qty * s['price'], 2)})
            spend_est += qty * s['price']
            new_positions[str(c)] = qty

    new_state = dict(state, month=month, positions=new_positions)
    base.update(rebalanced=True, picks=picks, orders=orders,
                alloc_total=round(alloc_total, 2), spend_est=round(spend_est, 2),
                new_state=new_state)
    if budget_cut:
        base['reason'] = (f'出资{alloc_total:,.0f}元 < 门槛{min_total:,.0f}元, '
                          f'本轮只卖不买({len(picks)}只候选已算)')
    else:
        base['reason'] = (f'{month}月频调仓: 候选{len(picks)}只, '
                          f'卖{sum(1 for o in orders if o["action"] == "close")} '
                          f'买{sum(1 for o in orders if o["action"] == "open")}, '
                          f'出资预算{alloc_total:,.0f}元 实花{spend_est:,.0f}元')
    notes = [
        '执行顺序: 股票买单全部成交后, 再下转债单',
        f'单位=张(1000元面值), 最小1手=10张; 下单数量按qty_zhang',
        f'{COST_NOTE}',
        '下单前确认QMT账户已开通可转债交易权限(独立于股票)',
        '估价为建议参考价, 以市价委托为准(低价券价差宽, 勿追价)',
        f'强赎检查: 近30日转股价值≥130天数≥{cfg.get("redemption_days", 15)}的券已跳过买入',
    ]
    if suspended_held:
        notes.append(f'停牌/无行情持仓未生成卖单(留待下月): {", ".join(suspended_held)}')
    if not picks:
        notes.append('候选为0: 已清仓全部可卖持仓, 下月再选(回测同口径: 无候选→空仓月)')
    base['notes'] = notes
    return base
