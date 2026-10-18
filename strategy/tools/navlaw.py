#!/usr/bin/env python
"""净值铁律哨兵 — "回测收益能复现"的日常判据 (2026-10-18)

**为什么必须有这个**

用户要的是"稳定可复现的选股系统"。铁律的形式是

    f(t; T) ≡ f(t; t)     对所有 T > t

即: 数据向前推进一天, **历史部分一格不动**。日常流 `run_daily_live.sh`
每天推进 todate 一天、重跑一次回测 —— 这本该是**每天一次天然的铁律实验**。

但 `equity_curve.csv` / `portfolio_selections.csv` 是**原地覆盖**的,
前一天的产物当场就没了 ⇒ **这个实验从来没做过**。不是没通过, 是没测。

本工具把"昨天的曲线"存下来, 于是每天都能零成本地做一次:
    共有日期区间上, 历史段的 NAV / 选股集合 是否逐位不变。

**两个量必须拆开报** (见 history-rewrite-ledger-20261017)

  · **选股集合** — 离散, 决定 α。一只票进没进, 是非题。
  · **净值/权重** — 连续。同样的股, 仓位大小可以不同。

两者可以独立违反。只报一个会漏。

**尾段 vs 历史段**

实测(#339)证明"尾部定稿"是数据供应商的固有行为: 最后 1 个交易日的 bar
会在下次刷新时被重新下载定稿, 波及最后 ~10 日。这部分**不是违规**。
故默认把最后 `--tail`(10) 个共有交易日单列, 判据只压**历史段**。

**用法**

    navlaw.py snap [label]      # 存档当前产物 (默认 label = 产物末日期)
    navlaw.py diff [A] [B]      # 比两个存档; 省略 = 最近两个
    navlaw.py ls                # 列出存档
    navlaw.py watch             # snap + diff 最近两个 (日常流用; 只报不拦)

存档放 `.arms/navlaw/<label>/`, **不放 /tmp** (tmpfs, 已实际 OOM 掉一次回测)。
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import time

import pandas as pd

ROOT = '/mnt/d/quant'
RVD = os.path.join(ROOT, 'strategy', 'rolling_validation_results')
STORE = os.path.join(ROOT, '.arms', 'navlaw')

# 存档哪些产物。equity=净值(连续) / selections=选股集合(离散) / realized=已实现笔
FILES = ['equity_curve.csv', 'portfolio_selections.csv',
         'trade_realized.csv', 'backtest_report.json']


def _md5(p):
    h = hashlib.md5()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def _snapdir(label):
    return os.path.join(STORE, label)


def _load(label):
    d = _snapdir(label)
    meta_p = os.path.join(d, 'meta.json')
    if not os.path.exists(meta_p):
        return None
    meta = json.load(open(meta_p))
    meta['dir'] = d
    return meta


# 判据容差: NAV 量级 ~1e6 ⇒ 1e-12 相对 = 1.4e-6 元, 远低于一分钱。
# 任何真实的数据改动都远在这之上; 只有 CSV 往返的 float64 重序列化会落在下面。
# (实测: 用 pandas 读→改一格→to_csv 往返, 会污染同列其他行的**文本**,
#  但浮点值相对差 ~4e-16 = 机器精度 ⇒ 那不是改写。)
REL_EPS = 1e-12


def _read_equity(d):
    p = os.path.join(d, 'equity_curve.csv')
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p)
    df['date'] = df['date'].astype(str).str[:10]
    df['nav_str'] = df['nav'].astype(str)      # 文本口径(严格)
    return df


def _read_sel(d):
    p = os.path.join(d, 'portfolio_selections.csv')
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p)
    df['date'] = df['date'].astype(str).str[:10]
    df['code'] = df['code'].astype(str).str.zfill(6)
    return df


def cmd_snap(args):
    label = args.label
    if not label:
        eq = _read_equity(RVD)
        if eq is None or eq.empty:
            print('✗ 没有 equity_curve.csv, 先跑回测')
            return 1
        label = eq['date'].iloc[-1]
    d = _snapdir(label)
    if os.path.exists(d) and not args.force:
        # 同 label 再 snap = 同 todate 重跑。这本身就是一次铁律检验:
        # 同代码+同数据+同 todate ⇒ 产物必须逐位相同。md5 不同就是改写,
        # 不能静默覆盖掉 —— 加时间戳后缀留证, 让 diff 去报。
        old = _load(label)
        cur_eq = _md5(os.path.join(RVD, 'equity_curve.csv')) \
            if os.path.exists(os.path.join(RVD, 'equity_curve.csv')) else None
        old_eq = (old or {}).get('files', {}).get('equity_curve.csv', {}).get('md5')
        if cur_eq and old_eq and cur_eq != old_eq:
            suf = time.strftime('%m%d_%H%M%S')
            print(f'⚠ 同 label({label}) 重跑但 equity md5 变了: '
                  f'{old_eq[:8]} → {cur_eq[:8]}')
            print(f'  ⇒ 同 todate 重跑本应逐位相同 —— 另存为 {label}__{suf} 留证')
            label = f'{label}__{suf}'
            d = _snapdir(label)
        else:
            print(f'· 存档 {label} 已存在且 md5 未变 (加 --force 覆盖)')
            return 0
    os.makedirs(d, exist_ok=True)
    meta = {'label': label, 'ts': time.strftime('%Y-%m-%d %H:%M:%S'),
            'files': {}, 'src': RVD}
    for f in FILES:
        src = os.path.join(RVD, f)
        if not os.path.exists(src):
            continue
        shutil.copy2(src, os.path.join(d, f))
        meta['files'][f] = {'md5': _md5(src), 'size': os.path.getsize(src)}
    eq = _read_equity(d)
    if eq is not None and not eq.empty:
        meta['last_date'] = eq['date'].iloc[-1]
        meta['first_date'] = eq['date'].iloc[0]
        meta['n_days'] = int(len(eq))
        meta['nav_end'] = float(eq['nav'].iloc[-1])
    json.dump(meta, open(os.path.join(d, 'meta.json'), 'w'), indent=1)
    print(f'✓ snap {label}: {meta.get("first_date")}→{meta.get("last_date")} '
          f'({meta.get("n_days")} 日) NAV={meta.get("nav_end", 0):,.2f}')
    return 0


def _labels_sorted():
    if not os.path.isdir(STORE):
        return []
    out = []
    for n in os.listdir(STORE):
        m = _load(n)
        if m and m.get('last_date'):
            out.append((m['last_date'], m.get('ts', ''), n))
    return [n for _, _, n in sorted(out)]


def cmd_ls(args):
    ls = _labels_sorted()
    if not ls:
        print('(无存档)')
        return 0
    w = max(16, max(len(n) for n in ls) + 2)
    print(f'{"label":<{w}}{"存档时间":<21}{"区间":<26}{"日数":>6}{"NAV":>16}  md5(eq)')
    print('-' * (w + 72))
    for n in ls:
        m = _load(n)
        eq = m['files'].get('equity_curve.csv', {}).get('md5', '')[:8]
        print(f'{n:<{w}}{m["ts"]:<21}'
              f'{m.get("first_date","?")}→{m.get("last_date","?"):<14}'
              f'{m.get("n_days",0):>6}{m.get("nav_end",0):>16,.0f}  {eq}')
    return 0


def _cmp_nav(a, b, tail):
    """比净值曲线。返回 (rows, verdict)。"""
    ea, eb = _read_equity(a['dir']), _read_equity(b['dir'])
    if ea is None or eb is None:
        return None, None
    m = ea.merge(eb, on='date', how='inner', suffixes=('_A', '_B'))
    m = m.sort_values('date').reset_index(drop=True)
    if m.empty:
        return [], None
    m['dn'] = (m['nav_B'] - m['nav_A']).abs()
    m['rel'] = m['dn'] / m['nav_A'].abs().replace(0, pd.NA)
    # 实质改写 = 浮点值真的变了(超过机器精度), 而非文本重序列化
    m['viol'] = m['rel'].fillna(0) > REL_EPS
    # 严格口径: 文本逐位不同 (含浮点噪声, 单列只作参考)
    m['strict'] = m['nav_str_A'] != m['nav_str_B']
    # 尾段 = 较晚存档的最后 tail 个共有交易日
    cut = max(0, len(m) - tail)
    m['seg'] = ['历史' if i < cut else '尾段' for i in range(len(m))]
    rows = m
    return rows, cut


def _cmp_sel(a, b, tail_dates):
    """比选股集合(离散)。返回逐日差异 DataFrame。"""
    sa, sb = _read_sel(a['dir']), _read_sel(b['dir'])
    if sa is None or sb is None:
        return None
    ga = {d: set(g['code']) for d, g in sa.groupby('date')}
    gb = {d: set(g['code']) for d, g in sb.groupby('date')}
    out = []
    for d in sorted(set(ga) & set(gb)):
        A, B = ga[d], gb[d]
        if A != B:
            out.append({'date': d, 'seg': '尾段' if d in tail_dates else '历史',
                        'lost': ','.join(sorted(A - B))[:60],
                        'gain': ','.join(sorted(B - A))[:60]})
    return pd.DataFrame(out)


def cmd_diff(args):
    ls = _labels_sorted()
    if args.a and args.b:
        A, B = args.a, args.b
    elif len(ls) >= 2:
        A, B = ls[-2], ls[-1]
    else:
        print(f'✗ 需要至少 2 个存档 (现有 {len(ls)}); 先 navlaw.py snap')
        return 1
    ma, mb = _load(A), _load(B)
    if ma is None or mb is None:
        print(f'✗ 存档缺失: {A if ma is None else B}')
        return 1

    print('=' * 84)
    print(f'净值铁律哨兵  A={A} ({ma["ts"]}, 末 {ma.get("last_date")})')
    print(f'                B={B} ({mb["ts"]}, 末 {mb.get("last_date")})')
    print(f'判据: 共有日期区间上 **历史段** 的净值与选股集合必须逐位不变 '
          f'(尾段 {args.tail} 日单列, 属供应商定稿)')
    print('=' * 84)

    rows, cut = _cmp_nav(ma, mb, args.tail)
    if rows is None:
        print('✗ 缺 equity_curve.csv')
        return 1
    if not rows.empty:
        tail_dates = set(rows[rows['seg'] == '尾段']['date'])
        print(f'\n── 净值 (连续量) ──  共有 {len(rows)} 日 '
              f'(历史 {cut} / 尾段 {len(rows)-cut})')
        for seg in ('历史', '尾段'):
            s = rows[rows['seg'] == seg]
            if s.empty:
                continue
            bad = s[s['viol']]
            noise = s[s['strict'] & ~s['viol']]
            tag = '✓' if seg == '历史' and bad.empty else ('✗' if seg == '历史' else '·')
            extra = f', 浮点噪声 {len(noise)} 日(不算违规)' if len(noise) else ''
            print(f'  {tag} {seg}: {len(s)} 日, 被改写 {len(bad)} 日{extra}, '
                  f'最大相对={s["rel"].max():.3e}')
            if not bad.empty:
                for _, r in bad.head(5).iterrows():
                    print(f'      {"⚠" if seg=="历史" else "·"} {r["date"]}  '
                          f'{r["nav_A"]:,.2f} → {r["nav_B"]:,.2f}  '
                          f'(Δ={r["nav_B"]-r["nav_A"]:+,.2f})')
                if len(bad) > 5:
                    print(f'      ... 另有 {len(bad)-5} 日被改写')

        sd = _cmp_sel(ma, mb, tail_dates)
        print(f'\n── 选股集合 (离散量, 决定 α) ──')
        if sd is None:
            print('  · 缺 portfolio_selections.csv, 跳过')
            sd_hist = pd.DataFrame()
        else:
            sd_hist = sd[sd['seg'] == '历史'] if not sd.empty else sd
            sd_tail = sd[sd['seg'] == '尾段'] if not sd.empty else sd
            print(f'  {"✓" if sd_hist.empty else "✗"} 历史: {len(sd_hist)} 日选股集合被改写')
            for _, r in sd_hist.head(5).iterrows():
                print(f'      ⚠ {r["date"]}  剔:{r["lost"]}  加:{r["gain"]}')
            if len(sd_hist) > 5:
                print(f'      ... 另有 {len(sd_hist)-5} 日')
            print(f'  · 尾段: {len(sd_tail)} 日 (仅记录)')
    else:
        print('\n  (无共有日期)')

    # ── 判定 ────────────────────────────────────────────────────────────────
    print('\n' + '=' * 84)
    hist = rows[rows['seg'] == '历史'] if not rows.empty else pd.DataFrame()
    nav_viol = int(hist['viol'].sum()) if not hist.empty else 0
    sel_viol = len(sd_hist) if 'sd_hist' in dir() and sd_hist is not None else 0
    if nav_viol or sel_viol:
        print(f'✗ 铁律违反: 历史段净值 {nav_viol} 日被改写 / 选股集合 {sel_viol} 日被改写')
        print('  ⇒ 今天的数据动了历史。按 histcells.py diff 定位到数据层, 再按列归因。')
        return 2
    print(f'✓ 铁律成立: 共有历史段 {len(hist)} 日净值与选股集合 **零改写** '
          f'(尾段 {len(rows)-len(hist)} 日定稿已单列)')
    print(f'  A NAV={ma.get("nav_end",0):,.2f}  B NAV={mb.get("nav_end",0):,.2f}')
    return 0


def cmd_watch(args):
    """日常流用: snap 当前 + diff 最近两个。只报不拦 (返回码恒 0)。"""
    rc = cmd_snap(argparse.Namespace(label=None, force=False))
    if rc != 0:
        print('[navlaw] snap 失败, 跳过 diff')
        return 0
    print()
    rc = cmd_diff(argparse.Namespace(a=None, b=None, tail=args.tail))
    if rc == 2:
        print('\n[navlaw] ⚠ 上面是铁律违反 —— 不拦出单, 但必须当天查清并记账。')
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)

    s = sub.add_parser('snap'); s.add_argument('label', nargs='?')
    s.add_argument('--force', action='store_true'); s.set_defaults(fn=cmd_snap)

    d = sub.add_parser('diff')
    d.add_argument('a', nargs='?'); d.add_argument('b', nargs='?')
    d.add_argument('--tail', type=int, default=10)
    d.set_defaults(fn=cmd_diff)

    l = sub.add_parser('ls'); l.set_defaults(fn=cmd_ls)

    w = sub.add_parser('watch'); w.add_argument('--tail', type=int, default=10)
    w.set_defaults(fn=cmd_watch)

    a = ap.parse_args()
    sys.exit(a.fn(a))


if __name__ == '__main__':
    main()
