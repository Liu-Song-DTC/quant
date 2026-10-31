#!/usr/bin/env python3
"""实盘选股前数据新鲜度门禁 (fail-closed: 任一硬指标不过 → 退出码1)

检查范围与期望:
  硬失败 (数据落后 → 拒绝出单):
    K线 (backtrader_data):        全量逐只检查 ≥ 预期交易日
                                  指数(sh000001/399006, 市场状态输入)同样硬检查;
                                  退市/停牌股用 kline_stale_allowlist.txt 豁免;
                                  北交所(43/82/83/87/88/92段)仅报告(已排除出股票池)
    融资融券 margin_daily:        最大日期 ≥ 上一交易日 (T+1晨发布, 周一晚最新=上周五)
    北向 northbound_daily:        同上
    概念历史 concept_hist:        各板块最大日期 ≥ 预期交易日 (白名单: 东财下架板块)
    基本面 fundamental_data:      最大报告期 ≥ 应披露的最新报告期
    龙虎榜上榜史 dragon_tiger_history: 最新上榜日 ≥ 上一交易日 (信号层输入, #353)
    复权事件表 events_table:  事件数/覆盖股数 ≥ 结构性地板 (塌陷=复权面损坏, #353)
  软警告 (事件驱动型, 不阻断):
    龙虎榜/减持记录/减持计划/业绩预告/解禁: 仅报告最后日期
    概念映射 stock_concept_map:   mtime 超过14天提醒

  数值正确性探针 (2026-09-03 新增, 与新鲜度同批硬检查):
    K线末bar:        close>0 且 high≥low (负价qfq/脏行类bug, 全量)
    margin:          最近3行融资余额非空且≥0
    northbound:      停披(2024-08-19)前历史行net_flow完整率≥90% (停披后NaN属政策)
    概念历史:        各板块最近5行return无NaN
    基本面:          最新报告期净利润-同比增长 NaN率≤2% (列漂移bug探测器, 全量)

预期交易日: 今天若为工作日且已过17:30 → 今天; 否则最近一个工作日。
            节假日误报请用 --expected YYYY-MM-DD 指定。

用法:
    python strategy/check_data_freshness.py [--expected 2026-09-02]
"""
import argparse
import os
import pickle
import sys
from datetime import date as date_type, datetime, timedelta
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
BT_DIR = PROJECT / 'data' / 'stock_data' / 'backtrader_data'
FUND_DIR = PROJECT / 'data' / 'stock_data' / 'fundamental_data'
ALT_DIR = PROJECT / 'data' / 'alternative_data'
CONCEPT_HIST = PROJECT / 'data' / 'concept_hist.pkl'
CONCEPT_MAP = PROJECT / 'data' / 'stock_concept_map.pkl'
KLINE_ALLOWLIST = PROJECT / 'data' / 'stock_data' / 'kline_stale_allowlist.txt'
# 上市名册对账的允许清单 = "在册但整个价格文件缺失"的例外 (新上市待建库);
# 与上面的**滞后**清单语义不同 ⇒ 不可混用 (混用会让缺文件的股票被静默放过)。
LISTING_ALLOWLIST = PROJECT / 'data' / 'stock_data' / 'listing_reconciliation_allowlist.txt'

# 东财已下架、真实终止的板块 (2026-09-03 核实): 无后续数据, 豁免检查
CONCEPT_FROZEN = {'2025年报预增'}
# 报告期披露截止: 季度末 + 62天 (Q1 4/30, Q2 8/31, Q3 10/31, 年报 4/30)
REPORT_DEADLINE_DAYS = 62


def expected_trading_day() -> date_type:
    now = datetime.now()
    d = date_type.today()
    if now.hour * 60 + now.minute >= 17 * 60 + 30 and d.weekday() < 5:
        return d
    d -= timedelta(days=1)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def prev_trading_day(d: date_type) -> date_type:
    """上一交易日 (跳过周末). margin T+1晨发布 → 周一晚最新=上周五;
    日历日-1 会落在周末造成误报。节假日用 --expected 配合。"""
    d -= timedelta(days=1)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def max_dt(series) -> str:
    s = pd.to_datetime(series, errors='coerce')
    return s.max().date().isoformat() if s.notna().any() else 'N/A'


def _file_last_bar(fpath):
    """快速读CSV最后一行 → (日期, open, high, low, close), 解析失败返回None值"""
    with open(fpath, 'rb') as fh:
        fh.seek(0, 2)
        size = fh.tell()
        fh.seek(max(0, size - 200))
        line = fh.read().decode('utf-8', 'ignore').rstrip().splitlines()[-1]
    parts = line.split(',')
    try:
        return (parts[0].strip()[:10], float(parts[1]), float(parts[2]),
                float(parts[3]), float(parts[4]))
    except (ValueError, IndexError):
        return (parts[0].strip()[:10], None, None, None, None)


def _file_has_bad_close(fpath):
    """全历史 close≤0 扫描 (逐行读, 5494只约30-60s; 捕获xtquant加性复权负价复活)"""
    try:
        with open(fpath, encoding='utf-8', errors='ignore') as fh:
            fh.readline()
            for line in fh:
                parts = line.rstrip('\n').split(',')
                if len(parts) > 4 and parts[4]:
                    try:
                        if float(parts[4]) <= 0:
                            return True
                    except ValueError:
                        continue
    except OSError:
        return False
    return False


def _load_kline_allowlist(expected=None) -> set:
    """Accept dated suspension exemptions only through their verified expiry.

    Legacy undated entries remain supported; new suspension entries should use
    ``until=YYYY-MM-DD`` to force periodic review if the share stays stale.
    """
    if not os.path.exists(KLINE_ALLOWLIST):
        return set()
    allowed = set()
    with open(KLINE_ALLOWLIST, encoding='utf-8') as f:
        for line in f:
            parts = line.split()
            if not parts or parts[0].startswith('#'):
                continue
            deadlines = [field[6:] for field in parts[1:] if field.startswith('until=')]
            if deadlines:
                if len(deadlines) != 1:
                    continue
                try:
                    deadline = date_type.fromisoformat(deadlines[0])
                except ValueError:
                    continue  # malformed expiry must never grant an exception
                if expected is not None and expected > deadline:
                    continue
            allowed.add(parts[0])
    return allowed


def _load_listing_allowlist() -> set:
    if not os.path.exists(LISTING_ALLOWLIST):
        return set()
    with open(LISTING_ALLOWLIST, encoding='utf-8') as f:
        return {ln.split()[0] for ln in f if ln.strip() and not ln.strip().startswith('#')}


def _propose_stale_allowlist(stale_codes):
    """为"已不在上游在册名册、且K线滞后"的股票生成可直接粘贴的滞后白名单建议行。

    动机(2026-10-28): 两只停止交易的股票未登记滞后白名单 ⇒ 新鲜度门禁拦停整条链
    ⇒ 需人工考古才知道该登记什么。自动提名 ⇒ 门禁失败时直接给出建议行。

    返回 (建议行列表, 建议文件路径或None)。非"名册外"的滞后码不提名(那是真数据问题)。
    """
    meta = PROJECT / 'data' / 'stock_data' / 'stock_metadata'
    rosters = sorted(meta.glob('listed_codes_*.csv'), reverse=True)
    if not rosters:
        return [], None
    try:
        listed = {ln.strip() for ln in rosters[0].read_text(encoding='utf-8').splitlines() if ln.strip()}
    except OSError:
        return [], None
    raw_dir = PROJECT / 'data' / 'stock_data' / 'raw_data'
    lines = []
    for code in stale_codes:
        if code in listed:
            continue                      # 仍在册 ⇒ 滞后是**真数据问题** ⇒ 不提名
        name, last = '', ''
        try:
            import csv as _csv
            with open(raw_dir / code / 'none.csv', encoding='utf-8-sig') as fh:
                rows = list(_csv.DictReader(fh))
            if rows:
                name = (rows[-1].get('股票名称') or '').strip()
                last = (rows[-1].get('日期') or '').strip()
        except (OSError, ValueError):
            pass
        lines.append(f'{code} {name} {last} 停止交易(退市/长期停牌): 上游在册名册已不含该码, '
                     f'末根bar={last}; 若复牌请从本文件移除')
    out = PROJECT / 'data' / 'stock_data' / 'kline_stale_proposals.txt'
    try:
        out.write_text(
            '# K线滞后白名单**建议行** (check_data_freshness.py 自动生成, 每次门禁检查覆盖)\n'
            '# 条件: 已在册名册里没有 + K线滞后 ⇒ 停止交易(退市/长期停牌) ⇒ 无后续K线是真实状态\n'
            '# 处置: 人工确认后把下列行追加到 data/stock_data/kline_stale_allowlist.txt\n'
            + ('\n'.join(lines) + '\n' if lines else '(本次无建议)\n'),
            encoding='utf-8')
    except OSError:
        out = None
    return lines, out


def check_kline(exp):
    """K线全量检查 (2026-09-03 升级: 抽样→全量, 新鲜度+数值健全性一路扫描):
    - 指数输入(sh000001上证/399006创业板指, 市场状态检测用) → 硬失败
    - 沪深股票 → 硬失败 (退市/停牌白名单豁免)
    - 北交所(43/82/83/87/88/92段) → 仅报告 (已整体排除出股票池, 2026-09-03用户指令)
    - 数值健全性(全量): 最后bar close≤0 / high<low → 硬失败 (负价qfq/脏行类bug)
    - 全历史负收盘扫描(全量): 任一 close≤0 → 硬失败 (2026-09-04: Windows侧每次重跑
      --bt-only 都会把 repair_qfq_negatives 的sina修复覆盖回xtquant加性版, 9128行负价
      原样复活; 门禁必须能抓出来 → 跑 data/repair_qfq_negatives.py)
    """
    from core.stock_pool import is_bse_code
    files = [f for f in os.listdir(BT_DIR) if f.endswith('_qfq.csv')]
    if not files:
        return False, f'K线: {BT_DIR} 无 *_qfq.csv 文件!'
    allow = _load_kline_allowlist(exp)
    stale_hard = []
    bad_bar = []
    neg_close = []
    bse_total = bse_stale = 0
    for f in files:
        code = f[:-8]
        d, o, h, l, c = _file_last_bar(os.path.join(BT_DIR, f))
        if c is not None and (c <= 0 or (h is not None and l is not None and h < l)):
            bad_bar.append(f'{code}(c={c})')
        if _file_has_bad_close(os.path.join(BT_DIR, f)):
            neg_close.append(code)
        if is_bse_code(code):
            bse_total += 1
            if d < exp.isoformat():
                bse_stale += 1
            continue
        if d >= exp.isoformat() or code in allow:
            continue
        stale_hard.append(f'{code}({d})')
    if stale_hard:
        s = ', '.join(stale_hard[:8]) + ('...' if len(stale_hard) > 8 else '')
        _codes = [x.split('(')[0] for x in stale_hard]
        _prop, _pfile = _propose_stale_allowlist(_codes)
        _extra = ''
        if _prop:
            _extra = (f' | 已自动提名 {len(_prop)} 只"停止交易"股 ⇒ 追加到 '
                      f'data/stock_data/kline_stale_allowlist.txt (建议行: {_pfile}): '
                      f'{_prop[0][:60]}')
        return False, f'K线: {len(files)}只, 滞后{len(stale_hard)}只(硬): {s}{_extra}'
    if bad_bar:
        s = ', '.join(bad_bar[:8]) + ('...' if len(bad_bar) > 8 else '')
        return False, f'K线数值异常(硬): {s}'
    if neg_close:
        s = ', '.join(neg_close[:8]) + ('...' if len(neg_close) > 8 else '')
        return False, (f'K线全历史负收盘(硬): {len(neg_close)}只: {s} '
                       f'(sina修复被覆盖, 跑 data/repair_qfq_negatives.py)')
    # 2026-10-28 (P0-a): 上市名册对账 —— 本检查此前只枚举**现存**文件,
    # 新上市/从未建档的股票下载失败时不可见 (股票池静默缩小)。
    # 名册由下载器落盘; 只在名册可信时把"缺股"当硬失败。
    _listing_note = _check_listing_reconciliation()
    if _listing_note:
        return False, _listing_note
    return True, (f'K线: {len(files)}只全部新鲜且末bar数值健全'
                  f' (北交所{bse_total}只已排除, 其中滞后{bse_stale}只仅报告)')


def _check_listing_reconciliation():
    """生产名册必须是今天生成且可信的；无法对账即拒绝出单。"""
    import sys as _sys
    _meta = PROJECT / 'data' / 'stock_data' / 'stock_metadata'
    # MMDD 文件名不能排序表示时间；明确要求今天的文件，避免跨年误选旧名册。
    _roster = _meta / f'listed_codes_{date_type.today():%m%d}.csv'
    if not _roster.is_file():
        return f'上市名册对账(硬): 缺少今日名册 {_roster}，拒绝出单'
    try:
        if datetime.fromtimestamp(_roster.stat().st_mtime).date() != date_type.today():
            return f'上市名册对账(硬): 今日名册文件未在今天生成 {_roster}，拒绝出单'
    except OSError as exc:
        return f'上市名册对账(硬): 无法检查名册时间 {_roster}: {exc}'
    _sys.path.insert(0, str(PROJECT / 'data'))
    try:
        from check_listing_reconciliation import load_roster
        from listing_reconciliation import reconcile_listing
        codes, reference_count = load_roster(_roster)
        _raw_dir = PROJECT / 'data' / 'stock_data' / 'raw_data'
        _raw = {d.name for d in _raw_dir.iterdir() if d.is_dir()} if _raw_dir.is_dir() else set()
        _qfq = {f[:-8] for f in os.listdir(BT_DIR) if f.endswith('_qfq.csv')}
        _rep = reconcile_listing(codes, _raw, _qfq,
                                 allowlist=_load_listing_allowlist(),
                                 reference_count=reference_count)
    except (OSError, UnicodeError, ValueError, ImportError) as exc:
        return f'上市名册对账(硬): 无法完成对账 ({exc})，拒绝出单'
    if _rep.roster_anomaly:
        return f'上市名册对账(硬): 名册不可信 ({_rep.roster_note})，拒绝出单'
    if _rep.ok:
        return ''
    missing = sorted(set(_rep.missing_raw) | set(_rep.missing_qfq))
    preview = ', '.join(missing[:8]) + ('...' if len(missing) > 8 else '')
    return (f'上市名册对账(硬): 在册{_rep.expected_count}只, 缺数据{len(missing)}只: '
            f'{preview} ⇒ 股票池会静默缩小, 拒绝放行')


def check_fundamental(exp):
    """基本面全量检查 (2026-09-03 升级: 抽样20只→全量):
    1. 报告期新鲜度: 全量最大报告期 ≥ 应披露的最新报告期
    2. 列漂移探针: 最新报告期的 净利润-同比增长 NaN率 ≤ 2% (2026年列漂移bug探测器)
    """
    files = [f for f in os.listdir(FUND_DIR) if f.endswith('.csv')]
    if not files:
        return False, f'基本面: {FUND_DIR} 无 CSV!'
    # 应披露的最新报告期: 季度末 + 62天 ≤ 今天 的最近一个
    today = date_type.today()
    candidates = []
    y = today.year
    for qe in [date_type(y - 1, 12, 31), date_type(y, 3, 31), date_type(y, 6, 30),
               date_type(y, 9, 30), date_type(y, 12, 31)]:
        if today >= qe + timedelta(days=REPORT_DEADLINE_DAYS):
            candidates.append(qe)
    exp_rp = max(candidates).strftime('%Y%m%d') if candidates else 'N/A'
    best = '0'
    latest_rows = nan_cnt = 0
    ind_nan_cnt = 0
    for f in files:
        try:
            df = pd.read_csv(os.path.join(FUND_DIR, f),
                             usecols=lambda c: c in ('报告期', '净利润-同比增长', '所处行业'))
            if '报告期' not in df.columns:
                continue
            s = df['报告期'].astype(str).str.extract(r'(\d{8})', expand=False)
            m = s.dropna().max()
            if m and m > best:
                best = m
            if exp_rp != 'N/A':
                latest = df[s == exp_rp]
                if not latest.empty:
                    latest_rows += 1
                    if pd.isna(latest['净利润-同比增长'].iloc[-1]):
                        nan_cnt += 1
                    # 2026-09-04: akshare新版yjbb无'所处行业'列 → 全市场最新期NaN,
                    # signal_engine回退路径崩溃。探针与同比并列, 防漂移再发。
                    if '所处行业' in df.columns and pd.isna(latest['所处行业'].iloc[-1]):
                        ind_nan_cnt += 1
        except Exception:
            continue
    date_ok = best >= exp_rp
    nan_frac = nan_cnt / max(latest_rows, 1)
    nan_ok = nan_frac <= 0.02
    ind_nan_frac = ind_nan_cnt / max(latest_rows, 1)
    ind_nan_ok = ind_nan_frac <= 0.02
    msg = (f'基本面: {len(files)}只, 最大报告期 {best} (期望≥{exp_rp}), '
           f'最新期同比NaN {nan_cnt}/{latest_rows} ({nan_frac:.1%}), '
           f'行业NaN {ind_nan_cnt}/{latest_rows} ({ind_nan_frac:.1%})')
    if not date_ok:
        msg += ' [报告期落后]'
    if not nan_ok:
        msg += ' [同比NaN超标, 疑似列漂移!]'
    if not ind_nan_ok:
        msg += ' [行业NaN超标, 疑似列漂移!]'
    return date_ok and nan_ok and ind_nan_ok, msg


def check_alt_pkl(name, col, exp, slack_days=0):
    p = ALT_DIR / name
    if not os.path.exists(p):
        return False, f'{name}: 文件不存在!'
    try:
        df = pd.read_pickle(p)
        if isinstance(df, dict):
            return True, f'{name}: dict {len(df)}键 (不校验日期)'
        m = pd.to_datetime(df[col], errors='coerce').max()
        if pd.isna(m):
            return True, f'{name}: 无有效日期 (不校验)'
        exp_d = exp - timedelta(days=slack_days)
        ok = m.date() >= exp_d
        return ok, f'{name}: 最大 {m.date()} (期望≥{exp_d})'
    except Exception as e:
        return False, f'{name}: 读取失败 {str(e)[:50]}'


# 北向每日流量交易所2024-08-19起停披 → 停披后net_flow=NaN是政策, 非数据问题 (代码有兜底)
NORTHBOUND_STOP = '2024-08-19'


def check_margin_sanity():
    """margin数值完整性: 最近3行融资余额非空且≥0"""
    try:
        df = pd.read_pickle(ALT_DIR / 'margin_daily.pkl').sort_values('date')
        tail = df.tail(3)
        bad = tail[tail['margin_balance'].isna() | (tail['margin_balance'] < 0)]
        ok = bad.empty and len(tail) > 0
        return ok, f'margin数值: 最近3行融资余额 {"正常" if ok else "异常(空/负)!"}'
    except Exception as e:
        return False, f'margin数值: 检查失败 {str(e)[:40]}'


def check_northbound_sanity():
    """northbound数值完整性: 停披前的历史行net_flow完整率≥90% (停披后NaN属政策)"""
    try:
        df = pd.read_pickle(ALT_DIR / 'northbound_daily.pkl')
        cutoff = pd.Timestamp(NORTHBOUND_STOP).date()
        dates = pd.to_datetime(df['date']).dt.date
        hist = df[dates < cutoff]
        frac = hist['net_flow'].notna().mean() if len(hist) else 1.0
        ok = frac >= 0.9
        return ok, (f'northbound数值: 停披前{len(hist)}行net_flow完整率{frac:.1%}'
                    f' (停披后NaN属政策, 仅报告)')
    except Exception as e:
        return False, f'northbound数值: 检查失败 {str(e)[:40]}'


def check_concept_sanity():
    """concept_hist数值完整性: 各板块(冻结板块除外)最近5行return无NaN"""
    try:
        with open(CONCEPT_HIST, 'rb') as f:
            hist = pickle.load(f)
    except Exception as e:
        return False, f'概念数值: 读取失败 {str(e)[:40]}'
    bad = []
    for name, df in hist.items():
        if name in CONCEPT_FROZEN:
            continue
        try:
            if df.tail(5)['return'].isna().any():
                bad.append(name)
        except Exception:
            bad.append(name)
    ok = not bad
    return ok, (f'概念数值: {len(hist)}板块最近5行return'
                f' {"全部正常" if ok else f"异常: {bad[:5]}"}')


def check_concept_hist(exp):
    if not os.path.exists(CONCEPT_HIST):
        return False, 'concept_hist.pkl: 文件不存在!'
    with open(CONCEPT_HIST, 'rb') as f:
        hist = pickle.load(f)
    stale = []
    for name, df in hist.items():
        if name in CONCEPT_FROZEN:
            continue
        if df['date'].max().date() < exp:
            stale.append(f'{name}({df["date"].max().date()})')
    ok = not stale
    msg = f'概念历史: {len(hist)}板块'
    if stale:
        msg += f', 落后: {stale[:5]}...' if len(stale) > 5 else f', 落后: {stale}'
    else:
        msg += f', 全部≥{exp}'
    return ok, msg


def check_concept_map():
    if not os.path.exists(CONCEPT_MAP):
        return False, 'stock_concept_map.pkl: 文件不存在!'
    age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(CONCEPT_MAP))
    ok = age.days <= 14
    return ok, f'概念映射: mtime {age.days}天前 (软警告, 阈值14天)'


def check_unlock_versions():
    """Prevent orders if the as-of unlock table diverges from the live table."""
    try:
        try:
            from core.unlock_schedule_versions import verify_current
        except ModuleNotFoundError:
            from strategy.core.unlock_schedule_versions import verify_current
        day = verify_current(ALT_DIR / 'unlock_schedule.pkl')
        return True, f'解禁日程版本: 已发布表与{day}观测版本一致'
    except Exception as exc:
        return False, f'解禁日程版本: {str(exc)[:150]}'


def check_events_table():
    """#353: events_table.pkl (表驱动复权的事件表) 注册门禁。
    结构性塌陷(n_events/n_codes 骤减) ⇒ 硬失败 — 全史事件表只增不减,
    37k 事件跌到 <1k = 复权面静默损坏(新事件全部不复权), 必须拦。
    data_end 落后 ⇒ 仅在消息里报告(事件稀疏, 滞后1-2天无害, 阻断会误伤)。"""
    import json
    meta_p = ALT_DIR / '.events_table_meta.json'
    pkl_p = ALT_DIR / 'events_table.pkl'
    if not meta_p.exists() or not pkl_p.exists():
        return False, 'events_table: 表或meta不存在 (data/build_events_table.py 未跑?)'
    try:
        with open(meta_p) as fh:
            meta = json.load(fh)
        n_ev, n_cd = int(meta['n_events']), int(meta['n_codes'])
        de = meta.get('data_end') or '?'
    except Exception as e:
        return False, f'events_table: meta 读取失败 {str(e)[:50]}'
    ok = n_ev >= 1000 and n_cd >= 3000
    return ok, f'events_table(复权事件表): {n_ev:,}事件/{n_cd:,}只, data_end={de}'


def check_dragon_tiger_history(exp):
    """#353: dragon_tiger_history.pkl 是信号层四文件之一 (bt_execution._SIGNAL_ALT_FILES,
    get_dragon_tiger_signal 直接消费), 但此前不在门禁 — 若重建步静默失败而
    dragon_tiger.pkl 本体新鲜, 信号层会持续用旧上榜史而无任何告警。
    schema = {code: set(datetime.date)} (check_alt_pkl 只认 DataFrame, 需专用检查)。
    期望 ≥ 上一交易日 (同 margin/northbound: T+1 晨发布口径的下界, 真抓取失败
    max 会停在更早 ⇒ 照样 FAIL)。"""
    name = 'dragon_tiger_history.pkl'
    p = ALT_DIR / name
    if not os.path.exists(p):
        return False, f'{name}: 文件不存在!'
    try:
        obj = pd.read_pickle(p)
        if not isinstance(obj, dict) or not obj:
            return False, f'{name}: 非 dict 或全空!'
        mx = None
        for v in obj.values():
            if v:
                m = max(v)
                if mx is None or m > mx:
                    mx = m
        if mx is None:
            return False, f'{name}: 所有股票上榜集合全空!'
        exp_d = prev_trading_day(exp)
        ok = pd.Timestamp(mx).date() >= exp_d
        return ok, f'{name}(信号层): 最新上榜 {mx} (期望≥{exp_d})'
    except Exception as e:
        return False, f'{name}: 读取失败 {str(e)[:50]}'


def main():
    ap = argparse.ArgumentParser(description='实盘选股数据新鲜度门禁')
    ap.add_argument('--expected', type=str, default=None,
                    help='预期交易日 YYYY-MM-DD (节假日误报时指定)')
    args = ap.parse_args()

    exp = pd.Timestamp(args.expected).date() if args.expected else expected_trading_day()
    print(f'=== 数据新鲜度检查 (预期交易日 ≥ {exp}) ===')

    hard = [
        check_kline(exp),
        check_unlock_versions(),
        check_alt_pkl('margin_daily.pkl', 'date', prev_trading_day(exp)),
        # 北向: 东财源头**本身**滞后一个交易日 (2026-10-19 实测: 北向资金/沪股通/
        # 深股通三个 symbol 在 20:15 查询, max 全部 = 2026-10-16 = 上一交易日;
        # 我们的 pkl 2770 行与源逐行一致)。故期望值取上一交易日, 不是当日。
        # 注意这仍是**下界**: 若某天抓取真失败, max 会停在更早 ⇒ 照样 FAIL。
        check_alt_pkl('northbound_daily.pkl', 'date', prev_trading_day(exp)),
        check_concept_hist(exp),
        check_fundamental(exp),
        check_margin_sanity(),
        check_northbound_sanity(),
        check_concept_sanity(),
        check_dragon_tiger_history(exp),
        check_events_table(),
    ]
    soft = [
        check_alt_pkl('dragon_tiger.pkl', '最近上榜日', exp),
        check_alt_pkl('reduction_plans.pkl', 'ann_date', exp),
        check_alt_pkl('reduction_records.pkl', 'start_date', exp),
        check_alt_pkl('yjyg_records.pkl', 'notice_date', exp),
        check_alt_pkl('unlock_schedule.pkl', 'unlock_date', exp),
        check_concept_map(),
    ]

    fails = 0
    for ok, msg in hard:
        mark = 'PASS' if ok else 'FAIL'
        print(f'  [{mark}] {msg}')
        fails += 0 if ok else 1
    print('  -- 软检查 (仅报告) --')
    for ok, msg in soft:
        print(f'  [{"OK " if ok else "WARN"}] {msg}')
        if not ok and '概念映射' in msg:
            print('        提示: 板块成员变化慢, 非阻塞; 超14天请跑 data/refresh_concept_map_0903.py')

    if fails:
        print(f'\n=== 门禁未通过: {fails} 项硬指标落后 ===')
        print('请先刷新数据: bash data/refresh_all.sh (另需Windows侧更新K线)')
        print('节假日误报: 用 --expected 指定上一交易日')
        sys.exit(1)
    print('\n=== 门禁通过: 所有硬指标新鲜 ===')
    sys.exit(0)


if __name__ == '__main__':
    main()
