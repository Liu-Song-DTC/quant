#!/usr/bin/env python3
"""K线下载器 — akshare/efinance 版 (2026-09-30, QMT权限撤销后的替代链路)

模式:
  --full    全量迁移: 全部沪/深上市码 2016-01-01 起全历史重下 (none/qfq/hfq + fq_factors),
            退市码冻结不动; 下载前对旧raw做重叠收益差报告(重锚对账用)
  默认      每日增量: 尾部追加 + EM侧重叠复权变化检测(>0.01 → 全量重载)

链: trade/config.yaml 中继代理补丁 → efinance get_quote_history(fqt=0/1/2)
   → raw_data/{code}/{none,qfq,hfq}.csv + fq_factors.csv (tmp+os.replace 原子写)
bt文件重建 = convert_raw_to_backtrader.py (本脚本不管)
"""
import os, sys, time, random, socket, json, threading
import warnings
warnings.filterwarnings('ignore')
socket.setdefaulttimeout(30)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import yaml
import pandas as pd
import numpy as np

CFG = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'trade', 'config.yaml')
RAW = '/mnt/d/quant/data/stock_data/raw_data'
BT = '/mnt/d/quant/data/stock_data/backtrader_data'
META = '/mnt/d/quant/data/stock_data/stock_metadata'
BEG = '20160101'
FQ_TYPES = {'none': 0, 'qfq': 1, 'hfq': 2}

_RL_LOCK = threading.Lock()
_RL_LAST = [0.0]
RL_MIN_GAP = 0.18  # 全worker共享限速 (~5.5 req/s, 保护中继代理IP)

def rate_limit():
    with _RL_LOCK:
        wait = _RL_LAST[0] + RL_MIN_GAP - time.time()
        if wait > 0:
            time.sleep(wait)
        _RL_LAST[0] = time.time()

def install_patch():
    with open(CFG, 'r', encoding='utf-8') as f:
        p = yaml.safe_load(f)['proxy']
    import akshare_proxy_patch
    akshare_proxy_patch.install_patch(
        p['host'], auth_token=p['auth_token'], retry=8,
        hook_domains=['fund.eastmoney.com', 'push2.eastmoney.com',
                      'push2his.eastmoney.com', 'emweb.securities.eastmoney.com'])
    print(f'[patch] 中继代理已安装 {p["host"]}', flush=True)

def get_listed_codes():
    """沪/深上市码 (排除北交所4/8/92、B股900、退市'-'报价行), 返回zfill6 set"""
    import efinance as ef
    spot = None
    for i in range(4):
        try:
            rate_limit()
            spot = ef.stock.get_realtime_quotes()
            if spot is not None and len(spot) > 3000:
                break
        except Exception as e:
            print(f'[list] 重试{i+1}: {str(e)[:80]}', flush=True)
            time.sleep(3)
    if spot is None or len(spot) <= 3000:
        raise RuntimeError('实时行情获取失败, 无法确定股票列表')
    spot = spot.copy()
    spot['code'] = spot['股票代码'].astype(str).str.zfill(6)
    # 退市股仍在行情列表但最新价='-': 剔除 (0c截断语义: 数据集只保留在册股)
    active = pd.to_numeric(spot['最新价'], errors='coerce') > 0
    spot = spot[active]
    codes = spot['code']
    sh = codes.str[:2].isin(['60', '68'])            # 600/601/603/605/688/689
    sz = codes.str[:3].isin(['000', '001', '002', '003', '300', '301', '302'])  # 302=创业板新段(2026启用, 302132中航成飞)
    out = set(codes[sh | sz].tolist())
    print(f'[list] 在册码总数 {len(out)} (沪{int(sh.sum())} 深{int(sz.sum())}), '
          f'剔除退市行 {int((~active).sum())}', flush=True)
    return out

def fetch_one(code, beg, end, fqt):
    import efinance as ef
    rate_limit()
    return ef.stock.get_quote_history(code, beg=beg, end=end, klt=101, fqt=fqt)

def fetch_with_retry(code, beg, end, fqt, tries=4):
    last = None
    for i in range(tries):
        try:
            df = fetch_one(code, beg, end, fqt)
            if df is not None and len(df) > 0:
                return df
            last = '空返回'
        except Exception as e:
            last = f'{type(e).__name__}:{str(e)[:60]}'
        time.sleep(2 * (i + 1) + random.random())
    raise RuntimeError(last)

def atomic_write(df, path):
    tmp = path + '.tmp'
    df.to_csv(tmp, index=False, encoding='utf-8')
    os.replace(tmp, path)

def coerce_dates(df):
    """EM返回的日期可能是str; 统一datetime64便于merge/去重"""
    df = df.copy()
    df['日期'] = pd.to_datetime(df['日期'])
    return df

def read_tail_date(path):
    """读CSV最后一行日期 (快)"""
    try:
        with open(path, 'rb') as f:
            size = f.seek(0, 2)
            if size < 50:
                return None
            f.seek(max(0, size - 300))
            tail = f.read().decode('utf-8', errors='ignore').strip().split('\n')
        tail = [l for l in tail if l.strip()]  # 去掉尾部空行
        if not tail:
            return None
        cols = tail[-1].split(',')
        # raw格式: 股票名称,股票代码,日期,... → 日期在第3列(index 2);
        # 兼容日期在首列的旧格式(指数目录等): 先试[2]再试[0]
        for ci in (2, 0):
            if len(cols) > ci:
                try:
                    return pd.to_datetime(cols[ci].strip()).date()
                except Exception:
                    continue
        return None
    except Exception:
        return None

def last_complete_trading_day():
    """最近一个**已收盘**的交易日 (2026-10-17 修周末叠加 bug)。

    旧实现两条规则**叠加**: `weekday()>=5 → 退到周五`, 再 `time()<15:30 → 再退一天`。
    对工作日这两条是"今天没收盘就用昨天", 正确; 但周末日期已被第一条改写成周五,
    第二条再退就多退一天 ⇒ **周五的数据被静默跳过**。
    实测 2026-10-17(周六)跑 refresh_all.sh: 5215 只全部 'skip', raw 停在 10-15,
    而中继源上 10-16 数据已存在(600519=1329.00 / 000001=11.50)。
    危害 = **静默陈旧一天**: 下游看到的是"刷新成功"而不是"少了一天"。
    (该次被 `check_data_freshness.py` 的 fail-closed 门禁拦下 — 门禁逻辑本来就是
     对的, 只有本函数错。)

    修正后与门禁 `expected_trading_day()` 同构, 只把收盘界从 17:30 提前到 15:30:
      工作日 >=15:30 → 今天; 否则回退一天并跳过周末。
    """
    now = pd.Timestamp.now()
    if now.weekday() < 5 and now.time() >= pd.Timestamp('15:30').time():
        return now.strftime('%Y%m%d')           # 工作日已收盘 = 今天
    d = now - pd.Timedelta(days=1)
    while d.weekday() >= 5:                     # 回退到最近工作日
        d -= pd.Timedelta(days=1)
    return d.strftime('%Y%m%d')

def compute_fq_factors(code, none_df, qfq_df, hfq_df):
    n = none_df[['日期', '收盘']].copy()
    q = qfq_df[['日期', '收盘']].copy()
    h = hfq_df[['日期', '收盘']].copy()
    m = n.merge(q, on='日期', suffixes=('_n', '_q')).merge(h, on='日期')
    m = m.rename(columns={'收盘': '收盘_h'})
    m['qfq_factor'] = m['收盘_q'] / m['收盘_n']
    m['hfq_factor'] = m['收盘_h'] / m['收盘_n']
    out = pd.DataFrame({'日期': pd.to_datetime(m['日期']).dt.date, 'symbol': code,
                        'qfq_factor': m['qfq_factor'], 'hfq_factor': m['hfq_factor']})
    return out

def full_download(code, beg, end, old_raw_dir=None):
    """全量: none/qfq/hfq + fq_factors 原子写. 返回(qfq_df, 状态full/recycled/short_ipo)"""
    out = {}
    for t, fqt in FQ_TYPES.items():
        out[t] = fetch_with_retry(code, beg, end, fqt)
    for t, df in out.items():
        df = coerce_dates(df)
        df = df[df['日期'] >= '2016-01-04'].reset_index(drop=True)  # 与旧raw深度对齐(2016起)
        df = df.drop_duplicates(subset=['日期'], keep='last').sort_values('日期')
        out[t] = df
    min_rows = min(len(d) for d in out.values())
    if min_rows < 100:
        old_path = os.path.join(RAW, code, 'qfq.csv')
        if os.path.exists(old_path):
            try:
                old = pd.read_csv(old_path)
                old_name = old['股票名称'].iloc[0] if '股票名称' in old.columns else None
            except Exception:
                old_name = None
            new_name = out['qfq']['股票名称'].iloc[0] if '股票名称' in out['qfq'].columns else None
            if old_name == new_name and len(old) > min_rows + 15:
                # 同名且EM明显短于旧raw=EM侧截断异常: 保护旧数据, 报FAIL
                raise RuntimeError(f'EM仅{min_rows}行<旧raw{len(old)}行且名称未变({new_name}), 保护旧raw')
            if old_name is not None and old_name != new_name:
                # 名称变了=代码回收(新公司复用旧代码): 以EM当前实体为准
                status = 'recycled'
            else:
                # 双短历史(旧raw≈EM)=近期IPO: EM为准(覆盖更新)
                status = 'short_ipo'
        else:
            status = 'short_ipo'  # 新上市码: 短历史合法
    else:
        status = 'full'
    for t, df in out.items():
        atomic_write(df, os.path.join(RAW, code, f'{t}.csv'))
    fq = compute_fq_factors(code, out['none'], out['qfq'], out['hfq'])
    atomic_write(fq, os.path.join(RAW, code, 'fq_factors.csv'))
    return out['qfq'], status

def incremental(code, end_ymd):
    """每日增量: 尾部追加; EM侧重叠复权变化>0.01 → 全量重载"""
    raw_dir = os.path.join(RAW, code)
    qfq_path = os.path.join(raw_dir, 'qfq.csv')
    tail = read_tail_date(qfq_path)
    if tail is None:
        _, st = full_download(code, BEG, end_ymd)  # 新上市码: 全量
        return st if st != 'full' else 'new'
    if tail >= pd.to_datetime(end_ymd).date():
        return 'skip'
    beg_ov = (pd.to_datetime(tail) - pd.Timedelta(days=12)).strftime('%Y%m%d')
    new_q = coerce_dates(fetch_with_retry(code, beg_ov, end_ymd, 1))
    old_q = pd.read_csv(qfq_path, parse_dates=['日期'])
    ov_new = new_q[new_q['日期'] <= pd.to_datetime(tail)]
    ov = old_q.merge(ov_new, on='日期', suffixes=('_o', '_n'))
    if len(ov) >= 3:
        diff = (ov['收盘_n'] - ov['收盘_o']).abs().max()
        if diff > 0.01:
            full_download(code, BEG, end_ymd)
            return 'fq_reload'
    # 尾部追加 none/qfq/hfq (去重)
    for t, fqt in FQ_TYPES.items():
        df = coerce_dates(fetch_with_retry(code, beg_ov, end_ymd, fqt))
        old = pd.read_csv(os.path.join(raw_dir, f'{t}.csv'), parse_dates=['日期'])
        both = pd.concat([old, df], ignore_index=True)
        both = both.drop_duplicates(subset=['日期'], keep='last').sort_values('日期')
        atomic_write(both, os.path.join(raw_dir, f'{t}.csv'))
    return 'incr'

def overlap_report(code, new_qfq):
    """迁移对账: 新EM qfq vs 旧raw qfq 重叠日收益差 (写前调用)"""
    old_path = os.path.join(RAW, code, 'qfq.csv')
    if not os.path.exists(old_path):
        return None
    try:
        old = pd.read_csv(old_path, parse_dates=['日期'])
        ov = old.merge(coerce_dates(new_qfq), on='日期', suffixes=('_o', '_n'))
        if len(ov) < 100:
            return None
        ro = ov['收盘_o'].pct_change()
        rn = ov['收盘_n'].pct_change()
        d = (rn - ro).abs().dropna()
        lvl = (ov['收盘_n'] / ov['收盘_o']).median()
        if abs(lvl - 1.0) < 1e-9:
            return None  # 同源(旧文件已是EM数据): 不产生对账行
        return {'code': code, 'n': len(ov), 'level_ratio': round(lvl, 6),
                'med_abs_dret': round(d.median(), 6), 'p99_abs_dret': round(d.quantile(0.99), 6),
                'big05': int((d > 0.05).sum())}
    except Exception:
        return None

def download_index(end_ymd):
    import akshare as ak
    # 三个指数都是回测输入: 上证(regime检测)+创业板指+中证1000(行业链/小盘代理)
    for ak_sym, dir_name in [('sh000001', 'sh000001'), ('sz399006', '399006'), ('sh000852', 'sh000852')]:
        d = None
        for i in range(4):
            try:
                rate_limit()
                d = ak.stock_zh_index_daily_em(symbol=ak_sym,
                                               start_date=BEG, end_date=end_ymd)
                if d is not None and len(d) > 500:
                    break
            except Exception as e:
                print(f'[idx] {dir_name} 重试{i+1}: {str(e)[:80]}', flush=True)
                time.sleep(3)
        if d is None or len(d) <= 500:
            raise RuntimeError(f'指数下载失败: {dir_name}')
        os.makedirs(os.path.join(RAW, dir_name), exist_ok=True)
        atomic_write(d, os.path.join(RAW, dir_name, 'qfq.csv'))
        print(f'[idx] {dir_name} {len(d)}行 → {d["date"].iloc[-1]}', flush=True)

def run(full=False, test_n=0):
    end = last_complete_trading_day()
    print(f'=== K线下载 {"全量迁移" if full else "增量"} 截止 {end} ===', flush=True)
    install_patch()
    download_index(end)
    listed = get_listed_codes()
    # 2026-10-28 (P0-a): 名册落盘, 供换名前/新鲜度门禁做集合对账。
    # 此前下载失败只写状态日志并正常结束, 新上市/从未建档的股票可能无声缺席
    # ⇒ 股票池缩小而不报错 (隔离样例: 只留 sh000001 时新鲜度检查仍报"全部新鲜")。
    _stamp = time.strftime('%m%d')
    _roster_path = os.path.join(META, f'listed_codes_{_stamp}.csv')
    try:
        with open(_roster_path, 'w', encoding='utf-8') as _f:
            for _c in sorted(listed):
                _f.write(f'{_c}\n')
        print(f'[roster] 在册名册已落盘 {len(listed)} 只 → {_roster_path}', flush=True)
    except OSError as _e:
        print(f'[roster] 名册落盘失败({_e}) ⇒ 门禁将无法对账', flush=True)
    raw_codes = {d for d in os.listdir(RAW)
                 if os.path.isdir(os.path.join(RAW, d)) and d != 'sh000001'}
    frozen = sorted(raw_codes - listed)
    targets = sorted(listed | raw_codes)  # 含退市码(冻结跳过)
    if test_n:
        # 跨板块代表码: 各前缀首个 + 随机补足 (688科创板/301次新创业板/60主板全覆盖)
        prefixes = ['000', '002', '300', '301', '600', '601', '603', '605', '688']
        picks = []
        for p in prefixes:
            hit = sorted([c for c in listed if c.startswith(p)])
            if hit:
                picks.append(hit[len(hit) // 2])  # 中位码避免头部老股
        rng = random.Random(42)
        rest = sorted(listed - set(picks))
        picks += rng.sample(rest, max(0, test_n - len(picks)))
        targets = sorted(picks)[:test_n]
    print(f'[plan] 目标 {len(targets)} (上市 {len(listed)}, 退市冻结 {len(frozen)})', flush=True)

    results = []
    report_rows = []
    st_lock = threading.Lock()

    def work(code):
        try:
            if code in frozen and not full:
                return (code, 'frozen')
            if full and code in frozen:
                return (code, 'frozen_full')  # 全量迁移也只处理上市码, 退市冻结
            os.makedirs(os.path.join(RAW, code), exist_ok=True)
            if full:
                q, fst = full_download(code, BEG, end)
                if fst == 'full':
                    rep = overlap_report(code, q)
                    if rep:
                        with st_lock:
                            report_rows.append(rep)
                return (code, fst)
            st = incremental(code, end)
            return (code, st)
        except Exception as e:
            return (code, f'FAIL:{str(e)[:70]}')

    from concurrent.futures import ThreadPoolExecutor
    done = 0
    with ThreadPoolExecutor(max_workers=12) as ex:
        futs = {ex.submit(work, c): c for c in targets}
        for f in futs:
            code, st = f.result()
            done += 1
            if done % 200 == 0 or st.startswith('FAIL'):
                print(f'[{done}/{len(targets)}] {code}: {st}', flush=True)
            results.append((code, st))

    # FAIL码重试一轮 (串行, 多数为瞬时网络问题)
    fails = [c for c, s in results if s.startswith('FAIL')]
    if fails and not test_n:
        print(f'[retry] {len(fails)}个FAIL码重试...', flush=True)
        for c in fails:
            try:
                os.makedirs(os.path.join(RAW, c), exist_ok=True)
                q, fst = full_download(c, BEG, end)
                if fst == 'full':
                    rep = overlap_report(c, q)
                    if rep:
                        report_rows.append(rep)
                results = [(x, 'retry_ok') if x == c else (x, s) for x, s in results]
            except Exception as e:
                print(f'[retry FAIL] {c}: {str(e)[:70]}', flush=True)

    if report_rows:
        rp = pd.DataFrame(report_rows)
        rp.to_csv(os.path.join(META, 'migration_overlap_20260930.csv'), index=False)
        print(f'[overlap] 重叠报告 {len(rp)}码 → stock_metadata/migration_overlap_20260930.csv', flush=True)
    log = pd.DataFrame(results, columns=['code', 'status'])
    log.to_csv(os.path.join(META, f'kline_download_log_{time.strftime("%m%d")}.csv'), index=False)
    from collections import Counter
    c = Counter(s for _, s in results)
    print(f'=== 完成: {dict(c)} ===', flush=True)
    # 2026-10-28 (P0-a): 重试后仍失败的代码 ⇒ 写失败清单并**非零退出**。
    # 真实失败率历史上为 0 (5219 skip + 201 frozen + 1 incr), 因此非零退出不会造成
    # 日常阻断; 但一旦有代码取不到历史, 必须显式失败而不是静默缩小股票池。
    _still_failed = sorted(c for c, s in results if str(s).startswith('FAIL'))
    if _still_failed:
        _fail_path = os.path.join(META, f'kline_download_failures_{time.strftime("%m%d")}.csv')
        pd.DataFrame({'code': _still_failed}).to_csv(_fail_path, index=False)
        print(f'!!! {len(_still_failed)} 只股票下载失败(重试后仍失败) ⇒ 清单: {_fail_path}',
              flush=True)
        print(f'    前20: {_still_failed[:20]}', flush=True)
        raise SystemExit(2)

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--full', action='store_true')
    ap.add_argument('--test', type=int, default=0)
    args = ap.parse_args()
    run(full=args.full, test_n=args.test)
