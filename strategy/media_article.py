#!/usr/bin/env python3
"""
自媒体文章生成器 — 把每日选股结果转为公众号/知乎风格的 Markdown 文章.

用法:
    .venv/bin/python strategy/media_article.py                    # 生成今日复盘文章
    .venv/bin/python strategy/media_article.py --date 2026-08-13  # 指定日期
    .venv/bin/python strategy/media_article.py --intro            # 生成首篇方法论文章

输出: strategy/media_out/复盘_YYYY-MM-DD.md
"""

import os, sys, json, argparse
from datetime import date as date_type, datetime, timedelta
from pathlib import Path

import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
DATA_DIR = os.path.join(_PROJECT_DIR, 'data', 'stock_data', 'backtrader_data')
RAW_DIR = os.path.join(_PROJECT_DIR, 'data', 'stock_data', 'raw_data')
OUTPUT_DIR = os.path.join(_SCRIPT_DIR, 'media_out')
CHART_DIR = os.path.join(_SCRIPT_DIR, 'live_charts')

INDEX_CODE = 'sh000001'

DISCLAIMER = (
    "免责声明：本文所有内容仅为个人量化交易系统的学习与记录，"
    "不构成任何投资建议。股市有风险，入市需谨慎。"
)


def _load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


_name_cache = None


def _load_name_cache():
    """加载 stock_list_full.csv 名称缓存 (去除 XD/XR/DR 前缀)."""
    global _name_cache
    if _name_cache is not None:
        return
    _name_cache = {}
    cache_fp = os.path.join(_PROJECT_DIR, 'data', 'stock_data', 'stock_metadata', 'stock_list_full.csv')
    if os.path.exists(cache_fp):
        try:
            df = pd.read_csv(cache_fp, dtype={'symbol': str})
            if 'symbol' in df.columns and 'name' in df.columns:
                for code, n in zip(df['symbol'].str.zfill(6), df['name']):
                    name = str(n).strip()
                    if name[:2] in ('XD', 'XR', 'DR'):
                        name = name.lstrip('XDXRDR')
                    if name:
                        _name_cache[code] = name
        except Exception:
            pass


def _stock_name(code):
    """取股票名称: stock_list 缓存优先, raw_data 回退 (raw可能截断如'XD健信超')."""
    _load_name_cache()
    name = _name_cache.get(code)
    if name:
        return name
    fp = os.path.join(RAW_DIR, code, 'qfq.csv')
    if os.path.exists(fp):
        try:
            with open(fp, 'r', encoding='utf-8') as f:
                header = f.readline().strip().split(',')
                first = f.readline().strip().split(',')
            if '股票名称' in header and len(first) > header.index('股票名称'):
                name = first[header.index('股票名称')].strip()
                if name[:2] in ('XD', 'XR', 'DR'):
                    name = name.lstrip('XDXRDR')
                if name:
                    return name
        except Exception:
            pass
    return ''


def _fmt_px(p):
    if p is None:
        return '-'
    if p < 10:
        return f'{p:.3f}'
    if p < 100:
        return f'{p:.2f}'
    return f'{p:.1f}'


def _last_close(code, target_date=None):
    """某股票最近收盘价."""
    fp = os.path.join(DATA_DIR, f'{code}_qfq.csv')
    if not os.path.exists(fp):
        return None
    try:
        df = pd.read_csv(fp, parse_dates=['datetime'])
        if target_date:
            df = df[df['datetime'] <= target_date]
        if df.empty:
            return None
        return float(df['close'].iloc[-1])
    except Exception:
        return None


def _selection_close(code, entry_date):
    """选股日收盘价 = 建仓日前最后一个交易日的收盘价."""
    fp = os.path.join(DATA_DIR, f'{code}_qfq.csv')
    if not os.path.exists(fp):
        return None
    try:
        df = pd.read_csv(fp, parse_dates=['datetime'])
        df = df[df['datetime'] < pd.Timestamp(entry_date)]
        if df.empty:
            return None
        return float(df['close'].iloc[-1])
    except Exception:
        return None


def _market_overview(target_date):
    """市场概况: 指数涨跌."""
    fp = os.path.join(DATA_DIR, f'{INDEX_CODE}_qfq.csv')
    if not os.path.exists(fp):
        return None
    try:
        df = pd.read_csv(fp, parse_dates=['datetime'])
        df = df[df['datetime'] <= target_date]
        if len(df) < 2:
            return None
        today = df.iloc[-1]
        prev = df.iloc[-2]
        chg = (today['close'] - prev['close']) / prev['close'] * 100
        chg_5d = (today['close'] - df['close'].iloc[-6]) / df['close'].iloc[-6] * 100 if len(df) >= 6 else None
        return {
            'date': target_date,
            'close': today['close'],
            'chg': chg,
            'chg_5d': chg_5d,
            'high': today['high'],
            'low': today['low'],
        }
    except Exception:
        return None


def _prev_trading_day(target_date):
    """上一个交易日."""
    d = pd.Timestamp(target_date) - timedelta(days=1)
    fp = os.path.join(DATA_DIR, f'{INDEX_CODE}_qfq.csv')
    try:
        df = pd.read_csv(fp, parse_dates=['datetime'])
        dates = sorted(df['datetime'].dt.date)
        prev = [x for x in dates if x < pd.Timestamp(target_date).date()]
        return prev[-1].strftime('%Y-%m-%d') if prev else None
    except Exception:
        return d.strftime('%Y-%m-%d')


# ── 选股理由解析 ──────────────────────────────────────────────

FACTOR_HUMAN = {
    'trend_lowvol': '趋势低波',
    'fund_revenue_growth': '营收增长',
    'fund_profit_growth': '利润增长',
    'fund_pg_improve': '利润增速改善',
    'momentum_reversal': '动量反转',
    'mom_x_lowvol': '动量×低波组合',
    'turnover_stability': '换手稳定性',
    'V41': '基础综合因子',
}

TREND_HUMAN = {0: '横盘整理', 1: '温和上行', 2: '明确上行', 3: '强势上行'}

_signal_cache = {}
_score_map_cache = {}


def _load_score_map():
    """加载信号CSV的 (code, 日期) → 评分 映射."""
    if _score_map_cache:
        return _score_map_cache
    fp = os.path.join(_SCRIPT_DIR, 'rolling_validation_results', 'backtest_signals.csv')
    if os.path.exists(fp):
        try:
            df = pd.read_csv(fp, dtype={'code': str}, usecols=['code', 'date', 'score'])
            df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
            _score_map_cache.update(
                {(r.code, r.date): r.score for r in df.itertuples()})
        except Exception:
            pass
    return _score_map_cache


def _signal_score(code, date_str):
    """某股票在指定日期的系统评分."""
    score_map = _load_score_map()
    v = score_map.get((code, str(date_str)[:10]))
    return float(v) if pd.notna(v) else None


def _latest_signals(target_date):
    """加载信号CSV, 返回最新信号日(≤target_date)的 DataFrame."""
    if 'df' in _signal_cache and _signal_cache.get('date') == target_date:
        return _signal_cache['df']
    fp = os.path.join(_SCRIPT_DIR, 'rolling_validation_results', 'backtest_signals.csv')
    if not os.path.exists(fp):
        _signal_cache['df'] = None
        _signal_cache['date'] = target_date
        return None
    try:
        df = pd.read_csv(fp, dtype={'code': str})
        df['date'] = pd.to_datetime(df['date'])
        df = df[df['date'] <= pd.Timestamp(target_date)]
        if df.empty:
            _signal_cache['df'] = None
            _signal_cache['date'] = target_date
            return None
        latest = df['date'].max()
        df = df[df['date'] == latest].set_index('code')
        _signal_cache['df'] = df
        _signal_cache['date'] = target_date
        return df
    except Exception:
        _signal_cache['df'] = None
        _signal_cache['date'] = target_date
        return None


def _factor_human(factor_name):
    """把因子名翻译成人话: fund_revenue_growth+trend_lowvol_FSM → 营收增长+趋势低波"""
    if not factor_name or not isinstance(factor_name, str):
        return ''
    parts = factor_name.replace('_FSM', '').replace('_FSMV', '').split('+')
    parts = [p for p in parts if p]
    translated = []
    for p in parts:
        for key, human in FACTOR_HUMAN.items():
            if key in p:
                translated.append(human)
                break
        else:
            # 未知因子: 保留原名但不带后缀
            translated.append(p.split('_')[0])
    return '+'.join(dict.fromkeys(translated))  # 去重保序


def _concept_heat_map():
    """当前各股票题材热度 (0~1)."""
    try:
        sys.path.insert(0, _SCRIPT_DIR)
        from core.concept_heat import get_calculator
        calc = get_calculator()
        heat = {}
        for code in (calc._stock_concepts or {}):
            try:
                h = calc.get_concept_heat(code)
                if h is not None:
                    heat[code] = round(float(h), 2)
            except Exception:
                pass
        return heat
    except Exception:
        return {}


_fund_industry_cache = {}


def _fund_industry(code):
    """从基本面数据取真实申万行业 (所处行业), 清理Ⅱ/Ⅲ/Ⅳ后缀."""
    if code in _fund_industry_cache:
        return _fund_industry_cache[code]
    result = ''
    fp = os.path.join(_PROJECT_DIR, 'data', 'stock_data', 'fundamental_data', f'{code}.csv')
    if os.path.exists(fp):
        try:
            df = pd.read_csv(fp, usecols=lambda c: '行业' in str(c))
            col = [c for c in df.columns if '所处行业' in str(c)]
            if col:
                vals = df[col[0]].dropna()
                if not vals.empty:
                    result = str(vals.iloc[-1]).replace('Ⅱ', '').replace('Ⅲ', '').replace('Ⅳ', '').strip()
        except Exception:
            pass
    _fund_industry_cache[code] = result
    return result


# 非题材类概念标签 (机构持仓/指数成分等, 不适合作为选股理由)
NON_THEMATIC_CONCEPTS = {
    '机构重仓', 'QFII重仓', '基金重仓', '融资融券', '沪股通', '深股通',
    '创业板综', '专精特新', '标普概念', '富时罗素', 'MSCI概念', '预盈预增',
    '预亏预减', '高送转', '次新股', '转融券标的', '大盘价值', '中俄贸易概念',
}

# 热门题材优先展示 (概念数据排序不可靠, 把市场热点方向提到前面)
PRIORITY_CONCEPTS = ['半导体', '光刻机', '存储', '芯片', '算力', 'AI', '机器人',
                     '创新药', '减肥药', '固态电池', '低空经济', '商业航天']

_concept_cache = {}


def _stock_concepts(code):
    """股票所属概念标签 (过滤非题材类)."""
    if code in _concept_cache:
        return _concept_cache[code]
    concepts = []
    try:
        from core.concept_heat import get_calculator
        calc = get_calculator()
        raw = (calc._stock_concepts or {}).get(code) or []
        concepts = [c for c in raw if c not in NON_THEMATIC_CONCEPTS]
        # 热门题材优先 (概念数据排序不可靠)
        concepts.sort(key=lambda c: 0 if any(k in c for k in PRIORITY_CONCEPTS) else 1)
    except Exception:
        pass
    _concept_cache[code] = concepts
    return concepts


def _build_reason(code, sig, heat_map):
    """为一支入选股票生成选股理由列表."""
    reasons = []
    if sig is None:
        return reasons
    # 1. 命中因子 (行业动态验证后当前有效的因子; 行业用基本面真实申万行业)
    factor_h = _factor_human(sig.get('factor_name', ''))
    industry = _fund_industry(code)
    score = sig.get('score', 0)
    if factor_h:
        ind_s = f" · 所属行业「{industry}」" if industry else ''
        reasons.append(f"命中当前有效因子：**{factor_h}**{ind_s}（综合评分 {score:.2f}）")
    elif score:
        reasons.append(f"综合评分 **{score:.2f}**")
    # 2. 趋势状态 (归一化动量/均线距离不展示原始数值, 只保留趋势定性)
    trend = sig.get('trend_type', 0)
    trend_h = TREND_HUMAN.get(int(trend) if pd.notna(trend) else 0, '')
    if trend_h:
        reasons.append(f"趋势状态：**{trend_h}**")
    # 3. 量能
    vr = sig.get('volume_ratio', 0)
    if pd.notna(vr) and float(vr) > 0.3:
        reasons.append(f"量能：成交量较均量放大 **{float(vr):.1f} 倍**")
    # 4. 市场关注方向 (概念标签, 前3个)
    concepts = _stock_concepts(code)
    if concepts:
        reasons.append(f"市场关注方向：{'、'.join(concepts[:3])}")
    # 5. 题材热度 (0.5是系统默认回退值, 跳过)
    heat = heat_map.get(code)
    if heat is not None and heat > 0.3 and abs(heat - 0.5) > 0.01:
        reasons.append(f"所属题材当前热度 **{heat:.2f}**（市场情绪聚集区）")
    return reasons


def generate_review(target_date=None, compliance=False):
    """生成每日复盘文章.

    Args:
        target_date: 复盘日期 (默认今天)
        compliance: 合规模式 — 进行中的持仓不写具体金额/价位数字,
                    已完成的批次才写 (复盘性质, 非操作指令)
    """
    if target_date is None:
        target_date = date_type.today().strftime('%Y-%m-%d')

    orders_path = os.path.join(_PROJECT_DIR, 'trade_orders.json')
    positions_path = os.path.join(_PROJECT_DIR, 'current_positions.json')

    if not os.path.exists(orders_path) or not os.path.exists(positions_path):
        print(f'[ERROR] 缺少 trade_orders.json 或 current_positions.json')
        return None

    orders_data = _load_json(orders_path)
    positions = _load_json(positions_path)
    orders = orders_data.get('orders', [])

    # ── 市场概况 ──
    market = _market_overview(target_date)
    if market is None:
        print('[WARN] 指数数据不可用, 跳过市场概况')
        return None

    # ── 标题 ──
    title = f"量化选股复盘 | {target_date[5:].replace('-', '月')}日，今天选出了什么？"

    lines = []
    lines.append(f"# {title}")
    lines.append('')
    lines.append(f"> 我的量化系统每天自动扫描全市场 5000+ 只股票，"
                 f"今天它给出了这些信号。所有内容仅供学习交流。")
    lines.append('')

    # ── 一、市场环境 ──
    lines.append('## 一、市场环境')
    lines.append('')
    lines.append(f"- 上证指数：**{market['close']:.2f}** 点"
                 f"（{'上涨' if market['chg'] >= 0 else '下跌'} {abs(market['chg']):.2f}%）")
    if market['chg_5d'] is not None:
        lines.append(f"- 近5日：{market['chg_5d']:+.2f}%")
    lines.append(f"- 当日区间：{market['low']:.2f} ~ {market['high']:.2f}")
    lines.append('')

    # ── 二、今日新入选 ──
    opens = [o for o in orders if o.get('action') == 'open']
    adjusts = [o for o in orders if o.get('action') == 'adjust']

    # 预加载信号 + 题材热度 (选股理由用)
    sig_df = _latest_signals(target_date)
    heat_map = _concept_heat_map() if opens else {}

    lines.append('## 二、今日新入选股票')
    lines.append('')
    if opens:
        lines.append('系统今天给出以下买入信号：')
        lines.append('')
        lines.append('> ⚠️ 以下为量化系统的自动交易记录，仅供学习交流。'
                     '不构成任何投资建议，请勿据此进行买卖操作。')
        lines.append('')
        if compliance:
            # 合规模式: 不写金额/价位, 但系统评分属于方法输出可展示
            # 评分取选股日信号 (订单日的前一交易日), 与批次表口径一致
            sel_date_full = _prev_trading_day(orders_data.get('date'))
            lines.append('| 代码 | 名称 | 系统评分 |')
            lines.append('|------|------|---------|')
            for o in opens:
                code = o['stock_code'].split('.')[0]
                name = _stock_name(code)
                score = _signal_score(code, sel_date_full)
                score_s = f"**{score:.2f}**" if score is not None else '-'
                lines.append(f"| {code} | {name} | {score_s} |")
            lines.append('')
        else:
            lines.append('| 代码 | 名称 | 买入金额 | 止损位 | 止盈位 | 盈亏比 |')
            lines.append('|------|------|---------|--------|--------|--------|')
            for o in opens:
                code = o['stock_code'].split('.')[0]
                name = _stock_name(code)
                amount = o.get('amount', 0)
                sl = o.get('stop_loss_price')
                tp = o.get('take_profit_price')
                entry = positions.get(code, {}).get('entry_price')
                if entry and sl and tp:
                    rr = (tp - entry) / (entry - sl) if entry > sl else 0
                    rr_str = f'1:{rr:.1f}'
                else:
                    rr_str = '-'
                lines.append(f"| {code} | {name} | {amount:,} | {_fmt_px(sl)} | "
                             f"{_fmt_px(tp)} | {rr_str} |")
            lines.append('')

        # ── 选股理由 ──
        lines.append('### 为什么选它们')
        lines.append('')
        for o in opens:
            code = o['stock_code'].split('.')[0]
            name = _stock_name(code)
            sig = sig_df.loc[code] if sig_df is not None and code in sig_df.index else None
            reasons = _build_reason(code, sig, heat_map)
            if not reasons:
                continue
            lines.append(f"**{name}（{code}）**")
            lines.append('')
            for r in reasons:
                lines.append(f"- {r}")
            lines.append('')
    else:
        lines.append('今天系统没有给出新的买入信号，继续持有现有仓位。')
        lines.append('')

    # ── 三、各批次5日表现 (与展示页截图对齐: 调仓周期5个交易日) ──
    HOLD_DAYS = 5
    lines.append('## 三、各批次5日表现')
    lines.append('')
    lines.append('> 调仓周期 5 个交易日。成本 = 选股日收盘价，盈利金额 = 买入金额 × 盈亏。')
    lines.append('')
    if positions:
        # 按选股日分批
        batches = {}
        for code, pos in positions.items():
            entry_date = pos.get('entry_date')
            if not entry_date:
                continue
            fp = os.path.join(DATA_DIR, f'{code}_qfq.csv')
            if not os.path.exists(fp):
                continue
            d = pd.read_csv(fp, parse_dates=['datetime'])
            if d.empty:
                continue
            sel_mask = d['datetime'] < pd.Timestamp(entry_date)
            if not sel_mask.any():
                continue
            cost = float(d.loc[sel_mask, 'close'].iloc[-1])
            sel_idx = int(sel_mask.sum()) - 1
            day5_idx = min(sel_idx + HOLD_DAYS, len(d) - 1)
            pnl = (float(d['close'].iloc[day5_idx]) - cost) / cost * 100
            days = day5_idx - sel_idx
            sel_date_full = d['datetime'].iloc[sel_idx].strftime('%Y-%m-%d')
            sel_date = sel_date_full[5:]
            score = _signal_score(code, sel_date_full)
            batches.setdefault(sel_date, []).append({
                'code': code,
                'name': _stock_name(code),
                'amount': pos.get('amount', 0),
                'pnl': pnl,
                'days': days,
                'done': days >= HOLD_DAYS,
                'score': score,
            })

        for bdate in sorted(batches):
            stocks = sorted(batches[bdate], key=lambda x: -x['pnl'])
            n = len(stocks)
            avg = sum(s['pnl'] for s in stocks) / n
            win = sum(1 for s in stocks if s['pnl'] > 0)
            amt = sum(s['amount'] for s in stocks)
            days = max(s['days'] for s in stocks)
            done = all(s['done'] for s in stocks)
            prog = '已完成' if done else f'第{days}/5天'
            if compliance and not done:
                # 合规模式: 进行中的批次不写金额数字, 有股票/评分/盈亏比例
                lines.append(f"### {bdate} 批次 · {prog} · 平均 {avg:+.1f}%（{win}/{n}只盈利）")
                lines.append('')
                lines.append('| 股票 | 系统评分 | 5日盈亏 |')
                lines.append('|------|---------|--------|')
                for s in stocks:
                    pnl_s = f"**{s['pnl']:+.1f}%**"
                    sc = f"**{s['score']:.2f}**" if s['score'] is not None else '-'
                    lines.append(f"| {s['name']} {s['code']} | {sc} | {pnl_s} |")
                lines.append('')
            else:
                head = f"### {bdate} 批次 · {prog} · 平均 {avg:+.1f}%（{win}/{n}只盈利"
                if done:
                    head += f"，投入 {amt/10000:.1f} 万"
                head += '）'
                lines.append(head)
                lines.append('')
                if done:
                    lines.append('| 股票 | 系统评分 | 买入金额 | 5日盈亏 | 盈利金额 |')
                    lines.append('|------|---------|---------|--------|---------|')
                    for s in stocks:
                        profit = s['amount'] * s['pnl'] / 100
                        pnl_s = f"**{s['pnl']:+.1f}%**"
                        sc = f"**{s['score']:.2f}**" if s['score'] is not None else '-'
                        profit_s = f"**{profit:+,.0f} 元**" if profit >= 0 else f"{profit:,.0f} 元"
                        lines.append(f"| {s['name']} {s['code']} | {sc} | {s['amount']/10000:.1f} 万 "
                                     f"| {pnl_s} | {profit_s} |")
                else:
                    lines.append('| 股票 | 系统评分 | 5日盈亏 |')
                    lines.append('|------|---------|--------|')
                    for s in stocks:
                        pnl_s = f"**{s['pnl']:+.1f}%**"
                        sc = f"**{s['score']:.2f}**" if s['score'] is not None else '-'
                        lines.append(f"| {s['name']} {s['code']} | {sc} | {pnl_s} |")
                lines.append('')

    # ── 四、昨日选股回顾 ──
    prev_day = _prev_trading_day(target_date)
    prev_entries = {c: p for c, p in positions.items()
                    if p.get('entry_date') == prev_day}
    lines.append(f'## 四、昨日选股回顾')
    lines.append('')
    if prev_entries:
        lines.append(f'昨天（{prev_day}）系统选出的股票，今天的表现：')
        lines.append('')
        wins = 0
        for code, pos in sorted(prev_entries.items()):
            cost = _selection_close(code, prev_day)
            last = _last_close(code, target_date)
            if cost and last:
                pnl = (last - cost) / cost * 100
                if pnl > 0:
                    wins += 1
                lines.append(f"- {code} {_stock_name(code)}：{pnl:+.1f}%")
        if prev_entries:
            hit = wins / len(prev_entries) * 100
            lines.append('')
            lines.append(f"昨日命中率：{wins}/{len(prev_entries)}（{hit:.0f}%）")
    else:
        lines.append(f'昨日（{prev_day}）没有新增持仓。')
    lines.append('')

    # ── 免责声明 ──
    lines.append('---')
    lines.append('')
    lines.append(DISCLAIMER)
    lines.append('')

    # ── 输出 ──
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f'复盘_{target_date}.md')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f'文章已生成: {out_path}')
    print(f'  新入选 {len(opens)} 只, 调整 {len(adjusts)} 只, 持仓 {len(positions)} 只')
    return out_path


# ═══════════════════════════════════════════════════════════════
# 首篇方法论文章
# ═══════════════════════════════════════════════════════════════

INTRO_ARTICLE = """# 我如何用一套量化系统在A股中选出好股

> 不荐股，只讲方法。这篇文章完整介绍我的量化选股系统是怎么工作的。

## 为什么要做量化

散户炒股最大的敌人是情绪：追涨杀跌、拿不住、割在地板上。

我的做法是把选股这件事交给程序——用数据说话，用纪律约束情绪。系统每天自动完成：

1. **扫描全市场 5000+ 只股票**，计算几百个因子
2. **动态筛选最优因子**，而不是拍脑袋定指标
3. **打分排序**，只买得分最高的
4. **严格风控**，每只票都带止损止盈

## 系统的选股逻辑（核心思路）

### 第一步：因子计算

系统对每只股票计算三大类因子：

- **技术因子**：动量、均线、波动率、成交量异常等
- **基本面因子**：估值（PE/PB）、盈利质量、成长性
- **另类数据**：龙虎榜机构动向、北向资金流向、融资余额变化、题材热度

A股有一个特色：龙虎榜、北向资金这些"聪明钱"数据，是散户也能拿到的公开信息，但很少有人系统地利用它们。

### 第二步：动态因子选择（关键创新）

传统量化策略的问题是：**因子会失效**。一个因子今年有效，明年可能就废了。

我的系统用 walk-forward IC 验证解决这个问题：

- 每过一段时间，用最近的历史数据重新检验每个因子的预测能力（IC）
- **只保留当前仍然有效的因子**，淘汰失效的
- 而且是**分行业**验证——医药股的有效因子和半导体完全不一样

这意味着系统会自适应市场变化，而不是拿着一套过时指标刻舟求剑。

### 第三步：信号生成与打分

有效因子汇总成综合评分，触发买卖信号。买卖信号都要经过多重确认，不是单因子一拍脑袋。

### 第四步：组合与风控

- 风险预算分配：得分高的票配更多仓位
- 目标波动率控制：市场剧烈波动时自动降仓
- 市场环境感知：牛/熊/震荡三种状态下采用不同的仓位暴露
- 每只票严格设止损止盈，盈亏比至少 1:2

## 这套系统实际怎么跑

每天盘前，系统自动完成：

1. 更新全市场数据（通过券商接口，免费）
2. 计算因子、验证有效性、生成信号
3. 输出选股清单 + 每只票的止损止盈位
4. 盘中按纪律执行，绝不手滑

## 我想在这里分享什么

- **每日复盘**：系统今天选了什么、为什么、表现如何（纯学习交流）
- **量化教程**：因子、IC验证、回测这些概念，用大白话讲清楚
- **踩坑记录**：我在这条路上犯过的错，帮你少交学费

---

""" + DISCLAIMER


def generate_intro():
    """生成首篇方法论文章."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, '首篇_量化选股方法论.md')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(INTRO_ARTICLE)
    print(f'文章已生成: {out_path}')
    return out_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='自媒体文章生成器')
    parser.add_argument('--date', type=str, default=None, help='复盘日期 (默认今天)')
    parser.add_argument('--intro', action='store_true', help='生成首篇方法论文章')
    parser.add_argument('--compliance', action='store_true',
                        help='合规模式: 进行中持仓不写具体金额/价位, 仅已完成批次写数字')
    args = parser.parse_args()

    if args.intro:
        generate_intro()
    else:
        generate_review(args.date, compliance=args.compliance)
