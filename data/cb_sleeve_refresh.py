#!/usr/bin/env python3
"""转债sleeve数据刷新 (2026-10-01, cb_sleeve实盘预备配套, 零回测影响)

下载 chain (经akshare中继代理链路, 与research 9/26-27同款端点):
  bond_zh_cov()                 → cb_list.csv   (全量债券列表, 含退市)
  bond_zh_hs_cov_daily(symbol)  → cb_daily/{code}.csv  (sina日线, volume单位=张)
  bond_zh_cov_value_analysis(code) → cb_value/{code}.csv (EM: 转股价值/纯债价值,
                                   PIT正确含下修/分红调整, 退市券全覆盖)

用法:
  python data/cb_sleeve_refresh.py            # 增量: 跳过今天已更新的文件
  python data/cb_sleeve_refresh.py --full     # 全量重下
月频调仓只需月内刷新一次; cb_sleeve选择器对陈旧日线的券自动跳过(max_stale_days)。
失败券跳过计数(126/127类分离交易债/可交换债日线端点无date列=预期失败, 研究同口径排除)。
"""
import os, sys, time, socket, argparse
import warnings
warnings.filterwarnings('ignore')
socket.setdefaulttimeout(20)

_ALT = '/mnt/d/quant/data/alternative_data'
DAILY_DIR = os.path.join(_ALT, 'cb_daily')
VALUE_DIR = os.path.join(_ALT, 'cb_value')
LIST_FILE = os.path.join(_ALT, 'cb_list.csv')
SLEEP = 0.3  # 研究同款限速(保护中继代理)


def fetch_list():
    import akshare as ak
    for i in range(4):
        try:
            bl = ak.bond_zh_cov()
            if bl is not None and len(bl) > 100:
                return bl
        except Exception as e:
            print(f'[list] 重试{i+1}: {str(e)[:80]}', flush=True)
            time.sleep(3)
    raise RuntimeError('bond_zh_cov 获取失败')


def _today_mtime(p: str) -> bool:
    try:
        st = os.stat(p)
        return time.strftime('%Y-%m-%d', time.localtime(st.st_mtime)) == \
               time.strftime('%Y-%m-%d')
    except OSError:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--full', action='store_true', help='全量重下(默认增量: 跳过今日已更新)')
    ap.add_argument('--only', type=str, default='', help='只刷指定券(逗号分隔裸码, 调试用)')
    args = ap.parse_args()

    os.makedirs(DAILY_DIR, exist_ok=True)
    os.makedirs(VALUE_DIR, exist_ok=True)
    import akshare as ak
    import pandas as pd

    bl = fetch_list()
    bl['债券代码'] = bl['债券代码'].astype(str).str.zfill(6)
    tmp = LIST_FILE + '.tmp'
    bl.to_csv(tmp, index=False)
    os.replace(tmp, LIST_FILE)
    print(f'[list] {len(bl)} 只 → {LIST_FILE}', flush=True)

    codes = sorted(bl['债券代码'].unique())
    if args.only:
        want = set(args.only.replace(' ', '').split(','))
        codes = [c for c in codes if c in want]
        print(f'[only] 限定 {len(codes)} 只: {codes[:10]}', flush=True)

    ok_d = ok_v = fail_d = fail_v = skip = 0
    for i, code in enumerate(codes):
        df = os.path.join(DAILY_DIR, f'{code}.csv')
        vf = os.path.join(VALUE_DIR, f'{code}.csv')
        if not args.full and _today_mtime(df) and _today_mtime(vf):
            skip += 1
            continue
        sym = ('sh' if code.startswith('11') else 'sz') + code
        if args.full or not _today_mtime(df):
            try:
                d = ak.bond_zh_hs_cov_daily(symbol=sym)
                if d is not None and len(d):
                    d.to_csv(df, index=False)
                    ok_d += 1
                else:
                    fail_d += 1
            except Exception:
                fail_d += 1
            time.sleep(SLEEP)
        if args.full or not _today_mtime(vf):
            try:
                v = ak.bond_zh_cov_value_analysis(symbol=code)
                if v is not None and len(v):
                    v.to_csv(vf, index=False)
                    ok_v += 1
                else:
                    fail_v += 1
            except Exception:
                fail_v += 1
            time.sleep(SLEEP)
        if i % 100 == 0:
            print(f'  progress {i}/{len(codes)} d{ok_d}/{fail_d} v{ok_v}/{fail_v} '
                  f'skip{skip}', flush=True)

    print(f'[done] 日线 ok={ok_d} fail={fail_d} | 价值 ok={ok_v} fail={fail_v} | '
          f'跳过(今日已更) {skip}', flush=True)


if __name__ == '__main__':
    main()
