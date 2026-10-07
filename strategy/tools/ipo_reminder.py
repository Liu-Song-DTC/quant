#!/usr/bin/env python3
"""打新提醒 (2026-10-07, #253落地件) — 每日流信息块, 零订单零写入(除缓存/ipo_today.json).

量化结论: 新股打新EV +3.53%/年(50万市值配售, 非科创板, 首日收盘卖, 2021-26实测),
转债打新EV +0.3~0.6%/年(信用申购顶格, 中签率假设带0.005-0.02%).
用途: 交易日盘前提示今日可申购新股/新债 — 提醒块, 申购操作由用户在券商App/QMT手动执行.
用法: python strategy/tools/ipo_reminder.py [--date 2026-10-09]
数据: stock_xgsglb_em('全部股票') 全史申购日历(含未来) + bond_cov_issue_cninfo(新债公告, 尽力而为).
"""
import argparse
import json
import os
import sys
import warnings
import datetime as dt

warnings.filterwarnings('ignore')
import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)  # strategy/
CACHE = os.path.join(_SCRIPT_DIR, '.ipo_cache.csv')
OUT = os.path.join(_PROJECT_DIR, 'ipo_today.json')
CACHE_TTL_HOURS = 12


def _patch():
    try:
        import yaml
        with open(os.path.join(_PROJECT_DIR, '..', 'trade', 'config.yaml'), 'r', encoding='utf-8') as f:
            p = yaml.safe_load(f)['proxy']
        import akshare_proxy_patch
        akshare_proxy_patch.install_patch(
            p['host'], auth_token=p['auth_token'], retry=8,
            hook_domains=['fund.eastmoney.com', 'push2.eastmoney.com',
                          'push2his.eastmoney.com', 'emweb.securities.eastmoney.com'])
    except Exception:
        pass


def fetch_stock_calendar(force=False):
    if not force and os.path.exists(CACHE):
        age = dt.datetime.now() - dt.datetime.fromtimestamp(os.path.getmtime(CACHE))
        if age.total_seconds() < CACHE_TTL_HOURS * 3600:
            return pd.read_csv(CACHE, dtype=str)
    _patch()
    import akshare as ak
    df = ak.stock_xgsglb_em(symbol='全部股票')
    df.to_csv(CACHE, index=False)
    return df


def fetch_cb_announcements():
    try:
        _patch()
        import akshare as ak
        return ak.bond_cov_issue_cninfo()
    except Exception as e:
        print(f'[新债公告] 获取失败(不影响新股提醒): {e}')
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--date', default=None, help='提醒日期 (默认今天), 格式YYYY-MM-DD')
    ap.add_argument('--force', action='store_true', help='强制刷新缓存')
    args = ap.parse_args()
    target = pd.Timestamp(args.date).date() if args.date else dt.date.today()

    print(f'=== 打新提醒 {target} ===')
    items = []

    # ── 新股 ──
    ip = fetch_stock_calendar(force=args.force)
    ip['申购日期'] = pd.to_datetime(ip['申购日期'], errors='coerce')
    for c in ['发行价格', '申购上限', '顶格申购需配市值', '发行市盈率', '行业市盈率']:
        ip[c] = pd.to_numeric(ip[c], errors='coerce')
    m = ip[ip['申购日期'].dt.date == target].copy()
    if len(m):
        print(f'\n[新股] {len(m)} 只今日申购:')
        for r in m.itertuples():
            board = str(r.板块)
            if board in ('科创板', '北交所'):
                print(f'  {r.股票简称}({r.股票代码}) {board} — 无权限, 跳过')
                continue
            # 板块列仅三值(非科创板/科创板/北交所), 细分从代码前缀推断
            _c = str(r.股票代码)
            if _c.startswith(('300', '301', '302')):
                board = '创业板'
            elif _c.startswith(('600', '601', '603', '605')):
                board = '沪市主板'
            elif _c.startswith(('000', '001', '002', '003')):
                board = '深市主板'
            need = r.顶格申购需配市值 if pd.notna(r.顶格申购需配市值) else 0
            ratio = min(1.0, 50.0 / need) if need and need > 0 else 1.0
            max_shares = r.申购上限 if pd.notna(r.申购上限) else 0
            sub_shares = int(max_shares * ratio / 100) * 100  # 整手
            px = r.发行价格 if pd.notna(r.发行价格) else None
            amt = f'约{sub_shares * px:,.0f}元' if px and sub_shares else '待定(发行价未公告)'
            # 破发风险提示: 创业板高PE发行
            risk = ''
            if board == '创业板':
                pe_i = r.发行市盈率 if pd.notna(r.发行市盈率) else None
                pe_m = r.行业市盈率 if pd.notna(r.行业市盈率) else None
                if pe_i is None or pe_m is None:
                    risk = ' [提示] 市盈率数据未全, 注意破发风险'
                elif pe_i > pe_m * 1.3:
                    risk = f' [提示] 发行PE {pe_i:.0f} > 行业PE {pe_m:.0f}×1.3, 破发风险偏高'
                else:
                    risk = f' (发行PE {pe_i:.0f} vs 行业 {pe_m:.0f})'
            elif board in ('沪市主板', '深市主板'):
                risk = ' — 主板首日涨跌幅保护, 建议申购'
            print(f'  申购: {r.股票简称}({r.股票代码}) {board} 顶格配市值{need:g}万 '
                  f'→ 申购{max_shares:g}股 {amt}{risk}')
            items.append({'type': 'stock', 'code': r.股票代码, 'name': r.股票简称,
                          'board': board, 'max_shares': max_shares, 'sub_shares': sub_shares,
                          'price': px, 'risk': risk.strip()})
    else:
        print('\n[新股] 今日无新股申购')

    # ── 新债 (公告窗口内尽力而为) ──
    cb = fetch_cb_announcements()
    cb_hits = []
    if cb is not None:
        cb['网上申购日期'] = pd.to_datetime(cb['网上申购日期'], errors='coerce')
        cb_h = cb[cb['网上申购日期'].dt.date == target]
        cb_hits = cb_h.to_dict('records')
        if cb_hits:
            print(f'\n[新债] {len(cb_hits)} 只今日申购 (顶格=信用申购, 零成本零资金占用):')
            for r in cb_hits:
                print(f"  顶格申购: {r['债券简称']}({r['债券代码']}) "
                      f"上限{r['网上申购数量上限']}手 — 建议顶格(中签后缴款)")
                items.append({'type': 'bond', 'code': r['债券代码'], 'name': r['债券简称'],
                              'max_hand': r['网上申购数量上限']})
        else:
            print('\n[新债] 公告窗口内今日无新债申购 (数据源仅覆盖近期公告, 可能漏)')

    if not items:
        print('\n今日无可申购项。')

    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump({'date': str(target), 'items': items}, f, ensure_ascii=False, indent=2)
    print(f'\n已写入 {OUT}')


if __name__ == '__main__':
    main()
