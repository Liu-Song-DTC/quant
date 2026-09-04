#!/usr/bin/env python3
"""qfq 负价修复 (通用版, 挂在 refresh_all.sh 每次刷新后自动重跑)

背景 (2026-09-03 定案): 50只高分红股 xtquant 加性前复权越界 -> 负收盘价;
  sina stock_zh_a_daily(qfq 乘性) vs xtquant raw(none.csv) 裁判通过 -> 50只用sina整文件重建。
  但每次Windows下载器重跑 --bt-only 都会重新用xtquant生成qfq -> 修复被覆盖,
  9128行负价原样回来 (2026-09-04 实测)。故本脚本做成幂等通用版:
    - 扫描全部 *_qfq.csv, 只修 close<=0 的文件
    - 盘中运行(15:30前)时 sina 只取到昨日, 避免当天半截bar; 盘后取到今天
    - 旧文件尾部若是当天半截bar, 允许新文件尾日=最近完整交易日(不判跳过)
    - 备份 {file}.preQfqFix (仅首次)
"""
import os, shutil, time
import numpy as np
import pandas as pd
import akshare as ak

BT = '/mnt/d/quant/data/stock_data/backtrader_data'
RAW = '/mnt/d/quant/data/stock_data/raw_data'
COLS = ['datetime', 'open', 'high', 'low', 'close', 'volume', 'openinterest',
        'amount', 'amplitude', 'change_percent', 'change_amount', 'turnover_rate']


def last_complete_day():
    """盘中(15:30前)返回昨日, 盘后返回今日 (周末让sina自然回落到最后交易日)"""
    now = pd.Timestamp.now()
    if now.weekday() >= 5:
        now -= pd.Timedelta(days=now.weekday() - 4)
    if now.time() < pd.Timestamp('15:30').time():
        now -= pd.Timedelta(days=1)
    return now.strftime('%Y%m%d')


def sina_prefix(code):
    return ('sh' if code[0] in '69' else 'sz' if code[0] in '03' else 'bj') + code


def fetch_sina(code, end):
    last = None
    for i in range(3):
        try:
            df = ak.stock_zh_a_daily(symbol=sina_prefix(code),
                                     start_date='20160101', end_date=end,
                                     adjust='qfq')
            if df is not None and len(df) > 0:
                return df
            last = '空返回'
        except Exception as e:
            last = str(e)[:80]
        time.sleep(3)
    raise RuntimeError(last)


def to_xtquant(df):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    df['prev_close'] = df['close'].shift(1)
    out = pd.DataFrame({
        'datetime': df['date'].dt.strftime('%Y-%m-%d'),
        'open': df['open'].round(4),
        'high': df['high'].round(4),
        'low': df['low'].round(4),
        'close': df['close'].round(4),
        'volume': (df['volume'] / 100.0).round(4),
        'openinterest': 0.0,
        'amount': df['amount'].round(4),
        'amplitude': ((df['high'] - df['low']) / df['prev_close'] * 100).round(4),
        'change_percent': (df['close'].pct_change() * 100).round(4),
        'change_amount': df['close'].diff().round(4),
        'turnover_rate': (df['turnover'] * 100).round(4),
    })
    return out[COLS]


def referee(code, new_df):
    """裁判: sina复权收益 vs xtquant未复权raw收益, 中位|Δret|≤0.001 且 p99≤0.05"""
    none_path = os.path.join(RAW, code, 'none.csv')
    if not os.path.exists(none_path):
        return False, '无raw none.csv'
    raw = pd.read_csv(none_path, parse_dates=['日期'])
    m = new_df.copy()
    m['dt'] = pd.to_datetime(m['datetime'])
    m = m.merge(raw[['日期', '收盘']], left_on='dt', right_on='日期')
    r_s = m['close'].pct_change()
    r_raw = m['收盘'].pct_change()
    d = (r_s - r_raw).abs().dropna()
    med = d.median()
    p99 = d.quantile(0.99)
    big = int((d > 0.05).sum())
    ok = med <= 0.001 and p99 <= 0.05
    return ok, f'重叠{len(m)}天 中位|Δret|={med:.6f} p99={p99:.4f} >5%天数={big}'


def main():
    end = last_complete_day()
    print(f'[qfq修复] sina取数截止: {end} (盘中自动截断)', flush=True)
    files = sorted(f for f in os.listdir(BT) if f.endswith('_qfq.csv'))
    targets = []
    for f in files:
        try:
            df = pd.read_csv(os.path.join(BT, f), usecols=['close'])
        except Exception:
            continue
        if (df['close'] <= 0).any():
            targets.append(f)
    print(f'负价文件: {len(targets)} 只', flush=True)
    ok_n, fail_n, skip_n = 0, 0, 0
    for i, f in enumerate(targets):
        code = f.replace('_qfq.csv', '')
        fp = os.path.join(BT, f)
        try:
            old = pd.read_csv(fp, parse_dates=['datetime'])
            new = to_xtquant(fetch_sina(code, end))
            if (new['close'] <= 0).any():
                print(f'[{i+1}/{len(targets)}] {code}: sina 仍有非正价!? 跳过', flush=True)
                fail_n += 1
                continue
            # 旧文件尾部可能是当天半截bar, 有效参照=min(旧尾日, sina截止日)
            old_tail = old['datetime'].iloc[-1].strftime('%Y-%m-%d')
            eff = min(old_tail, end[:4] + '-' + end[4:6] + '-' + end[6:])
            if new['datetime'].iloc[-1] < eff:
                print(f'[{i+1}/{len(targets)}] {code}: sina尾日 {new["datetime"].iloc[-1]} '
                      f'早于 {eff}, 跳过', flush=True)
                fail_n += 1
                continue
            ok, msg = referee(code, new)
            if not ok:
                print(f'[{i+1}/{len(targets)}] {code}: 裁判未过 ({msg}), 跳过', flush=True)
                fail_n += 1
                continue
            bak = fp + '.preQfqFix'
            if not os.path.exists(bak):
                shutil.copy2(fp, bak)
            new.to_csv(fp, index=False)
            print(f'[{i+1}/{len(targets)}] {code}: {len(old)}行->{len(new)}行, '
                  f'{new["datetime"].iloc[0]}~{new["datetime"].iloc[-1]}, {msg}', flush=True)
            ok_n += 1
        except Exception as e:
            print(f'[{i+1}/{len(targets)}] {code}: FAIL {str(e)[:90]}', flush=True)
            fail_n += 1
        time.sleep(1.2)
    print(f'=== qfq修复完成: 成功{ok_n} 失败{fail_n} ===', flush=True)
    return 0 if fail_n == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
