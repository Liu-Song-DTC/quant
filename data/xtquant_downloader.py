#!/usr/bin/env python3
"""
XTQuant 数据下载器 — 通过国金证券 QMT 客户端接口下载 A 股日线数据
替代 efinance/akshare, 免费无积分限制, 数据来自券商官方接口.

要求: 必须在 Windows 上运行 (xtquant 依赖 Windows DLL), QMT 客户端需已登录.

用法:
    python data/xtquant_downloader.py                       # 增量更新全部A股
    python data/xtquant_downloader.py --full                # 全量下载(从2015-01-01)
    python data/xtquant_downloader.py --since 2026-01-01    # 从指定日期开始
    python data/xtquant_downloader.py --stock 000001        # 单只股票
    python data/xtquant_downloader.py --failed              # 重试失败记录
    python data/xtquant_downloader.py --bt                  # 下载后同步更新backtrader数据
"""

import os, sys, time, json, argparse, re
from datetime import datetime, timedelta, date
from pathlib import Path
import numpy as np
import pandas as pd

# --- 路径 ---
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(__file__).resolve().parent / 'stock_data'
RAW_DATA_DIR = DATA_DIR / 'raw_data'
BT_DATA_DIR = DATA_DIR / 'backtrader_data'
FAILED_LOG = DATA_DIR / 'failed_stocks.json'
METADATA_FILE = DATA_DIR / 'metadata.json'

INDEX_CODE = 'sh000001'
DEFAULT_START = '2015-01-01'

# --- 检查 xtquant 可用性 ---
try:
    from xtquant import xtdata
    HAS_XTQUANT = True
except ImportError:
    HAS_XTQUANT = False

# efinance 输出的 CSV 列顺序, 保持兼容
STOCK_COLUMNS = ['股票名称', '股票代码', '日期', '开盘', '收盘', '最高', '最低',
                 '成交量', '成交额', '振幅', '涨跌幅', '涨跌额', '换手率']
INDEX_COLUMNS = ['date', 'open', 'close', 'high', 'low', 'volume', 'amount']

DIVIDEND_MAP = {'none': 'none', 'qfq': 'front', 'hfq': 'back'}

# 日期格式: efinance输出 YYYY-MM-DD
DATE_RE = re.compile(r'^\d{4}-\d{2}-\d{2}$')

# 常见的中文乱码/空文件名
INVALID_NAMES = {'0', '1', '', 'nan', 'None', 'null'}


def to_xt(code):
    """000001 -> 000001.SZ, 600000 -> 600000.SH, 920000 -> 920000.BJ"""
    code = str(code).zfill(6)
    if code.startswith(('4', '8', '92')):
        return f"{code}.BJ"
    if code.startswith(('6', '9', '5')):
        return f"{code}.SH"
    return f"{code}.SZ"


def pick_date_from_line(line):
    """从CSV行中提取日期字符串(YYYY-MM-DD)"""
    for part in line.split(','):
        part = part.strip().strip('"')
        if DATE_RE.match(part):
            return part
    return None


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


class XTQuantDownloader:
    """通过 xtquant 下载 A 股日线数据到 raw_data 目录"""

    def __init__(self, data_dir=None):
        base = Path(data_dir) if data_dir else DATA_DIR
        self.raw_dir = base / 'raw_data'
        self.bt_dir = base / 'backtrader_data'
        self.failed_log = base / 'failed_stocks.json'
        self.metadata_file = base / 'metadata.json'
        self._code_map = {}  # {bare: xt_code}, 由 run() 填充
        self._float_vol_cache = {}  # {bare: 流通股本(股)}
        ensure_dir(self.raw_dir)
        ensure_dir(self.bt_dir)

    # ── 基础查询 ──────────────────────────────────────────────

    def get_a_shares(self):
        """获取沪深A股+北交所列表, 返回 {裸代码: xt_code} 映射"""
        xt_codes = list(xtdata.get_stock_list_in_sector('沪深A股'))
        try:
            bj = xtdata.get_stock_list_in_sector('北交所')
            if bj:
                xt_codes += list(bj)
        except Exception:
            pass
        mapping = {}
        for c in xt_codes:
            parts = c.split('.')
            if len(parts) == 2 and parts[0].isdigit() and len(parts[0]) == 6:
                mapping[parts[0]] = c
        return mapping  # {'000001': '000001.SZ', '600000': '600000.SH', '920000': '920000.BJ'}

    def _xt_code(self, code, code_map=None):
        """获取 xtquant 格式代码, 优先使用从 get_a_shares 拿到的映射"""
        if code_map and code in code_map:
            return code_map[code]
        return to_xt(code)

    def get_stock_name(self, code):
        try:
            xt_code = self._code_map.get(code, to_xt(code))
            detail = xtdata.get_instrument_detail(xt_code)
            if detail:
                name = str(detail.get('InstrumentName', ''))
                if name and name not in INVALID_NAMES:
                    return name
        except Exception:
            pass
        return ''

    def get_ipo_date(self, code):
        """获取上市日期 YYYYMMDD, 失败返回 None"""
        try:
            xt_code = self._code_map.get(code, to_xt(code))
            detail = xtdata.get_instrument_detail(xt_code)
            if detail:
                d = detail.get('OpenDate', '')
                s = str(d).strip()
                if len(s) >= 8 and s[:8].isdigit():
                    return s[:8]
        except Exception:
            pass
        return None

    # ── 读取已有数据的最后日期 ────────────────────────────────

    def read_last_date(self, code, adj_type):
        """读取已有 raw CSV 的最后一条数据日期, 返回 date 或 None"""
        code = str(code).zfill(6)
        file_path = self.raw_dir / code / f'{adj_type}.csv'
        if not file_path.exists():
            return None
        size = file_path.stat().st_size
        if size < 80:
            return None
        try:
            with open(file_path, 'rb') as f:
                f.seek(max(0, size - 1024))
                tail = f.read().decode('utf-8', errors='ignore')
            lines = [l.strip() for l in tail.split('\n') if l.strip()]
            if len(lines) < 2:
                return None
            date_str = pick_date_from_line(lines[-1])
            return datetime.strptime(date_str, '%Y-%m-%d').date() if date_str else None
        except Exception:
            return None

    # ── 下载核心 ──────────────────────────────────────────────

    def download_one(self, code, start_date, end_date, adj_types=('none', 'qfq', 'hfq')):
        """下载单只股票全部复权类型数据.

        Returns:
            dict like {'qfq': DataFrame, 'hfq': DataFrame, 'none': DataFrame}
            或 None (下载失败/无数据).
        """
        start_str = start_date.replace('-', '') if isinstance(start_date, str) else start_date
        end_str = end_date.replace('-', '') if isinstance(end_date, str) else end_date

        if code == INDEX_CODE:
            return self._download_index(start_str, end_str)

        xt_code = self._code_map.get(code, to_xt(code))

        # 1) 下载到 QMT 本地缓存 (只需一次, 复权类型在读取时指定)
        try:
            xtdata.download_history_data(
                xt_code, period='1d',
                start_time=start_str, end_time=end_str,
            )
            time.sleep(0.25)
        except Exception as e:
            print(f"  download_history_data 失败 {code}: {e}")
            return None

        # 2) 按复权类型分别读取 (只取基础价量字段, 派生字段在第3步统一计算)
        result = {}
        for adj_type in adj_types:
            dividend = DIVIDEND_MAP[adj_type]
            try:
                raw = xtdata.get_market_data(
                    field_list=['open', 'high', 'low', 'close', 'volume', 'amount', 'preClose'],
                    stock_list=[xt_code],
                    period='1d',
                    start_time=start_str,
                    end_time=end_str,
                    dividend_type=dividend,
                    fill_data=False,
                )
                if not raw or 'close' not in raw or raw['close'].empty:
                    continue

                # raw['close'] index=股票代码, columns=日期; 取第一行(唯一股票)转为日期索引Series
                close_s = raw['close'].iloc[0]
                idx = close_s.index

                df = pd.DataFrame({
                    '日期': pd.to_datetime(idx).date,
                    '开盘': raw['open'].iloc[0].values,
                    '收盘': close_s.values,
                    '最高': raw['high'].iloc[0].values,
                    '最低': raw['low'].iloc[0].values,
                    '成交量': raw['volume'].iloc[0].values,
                    '成交额': raw['amount'].iloc[0].values,
                    'preClose': raw['preClose'].iloc[0].values,
                })

                # 过滤全零行 (停牌/无数据)
                df = df[(df['开盘'] != 0) | (df['收盘'] != 0) |
                        (df['最高'] != 0) | (df['最低'] != 0)]

                if not df.empty:
                    result[adj_type] = df
            except Exception as e:
                # 单类型失败不致命, 继续尝试其他类型
                pass

        # 3) 计算派生字段: 涨跌幅/振幅/涨跌额 (交易所口径) + 换手率
        #    交易所涨跌幅 = qfq收益率 (前复权序列的日收益=官方涨跌幅, 含除权参考价修正)
        #    换手率 = 成交量(手)×100 ÷ 流通股本(股) × 100%
        if result:
            self._add_derived_fields(code, result)

        return result if result else None

    def _add_derived_fields(self, code, result):
        """统一计算涨跌幅/振幅/涨跌额/换手率 (与数据源无关的交易所口径)."""
        try:
            none_df = result.get('none')
            qfq_df = result.get('qfq')
            if none_df is None or qfq_df is None or none_df.empty or qfq_df.empty:
                # 只有单类型时退化: 用preClose字段
                for df in result.values():
                    pre = df['preClose'].values
                    df['涨跌额'] = np.round(df['收盘'].values - pre, 4)
                    df['涨跌幅'] = np.round(
                        np.where(pre > 0, (df['收盘'].values - pre) / pre * 100, 0), 4)
                    df['振幅'] = np.round(
                        np.where(pre > 0, (df['最高'].values - df['最低'].values) / pre * 100, 0), 4)
                return

            # 对齐日期, 用 qfq 收益率计算官方涨跌幅
            merged = pd.merge(
                none_df[['日期', '收盘']].rename(columns={'收盘': 'none_close'}),
                qfq_df[['日期', '收盘']].rename(columns={'收盘': 'qfq_close'}),
                on='日期', how='inner')
            merged = merged.sort_values('日期').reset_index(drop=True)
            merged['none_ret'] = merged['none_close'] / merged['none_close'].shift(1)
            merged['qfq_ret'] = merged['qfq_close'] / merged['qfq_close'].shift(1)

            # 除权参考价 = 昨收 × none_ret/qfq_ret (非除权日等于昨收)
            pre_close_ref = merged['none_close'].shift(1) * merged['none_ret'] / merged['qfq_ret']
            pre_close_ref = pre_close_ref.replace([np.inf, -np.inf], np.nan)

            # 首行无昨收: 用 xtquant 的 preClose 兜底
            first_pre = none_df['preClose'].iloc[0] if 'preClose' in none_df.columns else np.nan
            pre_close_ref = pre_close_ref.fillna(first_pre)

            # 涨跌额/振幅/涨跌幅是交易所官方统计量, 三个复权序列应完全相同
            # → 只从 none 序列计算一次, 同值赋给所有序列
            none_aligned = none_df.merge(
                merged[['日期', 'qfq_ret']], on='日期', how='left').sort_values('日期')
            none_aligned['pre_ref'] = pre_close_ref.values
            official = pd.DataFrame({
                '日期': none_aligned['日期'],
                '涨跌幅': np.round((none_aligned['qfq_ret'] - 1) * 100, 4),
                '涨跌额': np.round(none_aligned['收盘'] - none_aligned['pre_ref'], 4),
                '振幅': np.round(
                    np.where(none_aligned['pre_ref'] > 0,
                             (none_aligned['最高'] - none_aligned['最低'])
                             / none_aligned['pre_ref'] * 100, 0), 4),
            })
            # 首行无昨收: 用 xtquant preClose 兜底
            if 'preClose' in none_df.columns:
                first_pre = none_df['preClose'].iloc[0]
                if pd.notna(first_pre) and first_pre > 0:
                    f = official.index[0]
                    first_none = none_df.iloc[0]
                    official.loc[f, '涨跌幅'] = round(
                        (first_none['收盘'] - first_pre) / first_pre * 100, 4)
                    official.loc[f, '涨跌额'] = round(first_none['收盘'] - first_pre, 4)
                    official.loc[f, '振幅'] = round(
                        (first_none['最高'] - first_none['最低']) / first_pre * 100, 4)

            for adj_type, df in result.items():
                d = df.merge(official, on='日期', how='left').sort_values('日期')
                d = d.drop(columns=['preClose'], errors='ignore')
                result[adj_type] = d

            # 换手率 = 成交量(手)×10000 ÷ 流通股本(股), 用 none 序列计算
            float_vol = self.get_float_volume(code)
            if float_vol and float_vol > 0:
                for adj_type, df in result.items():
                    df['换手率'] = np.round(
                        np.where(df['成交量'] > 0, df['成交量'] * 10000 / float_vol, 0), 4)
        except Exception as e:
            print(f"  [WARN] {code} 派生字段计算失败: {e}")

    def _last_trading_day(self):
        """最新交易日 (从指数raw数据取最后日期), 格式 YYYYMMDD."""
        try:
            fp = self.raw_dir / 'sh000001' / 'qfq.csv'
            if not fp.exists():
                return None
            last = self.read_last_date(INDEX_CODE, 'qfq')
            return last.strftime('%Y%m%d') if last else None
        except Exception:
            return None

    def get_float_volume(self, code):
        """流通股本 (股), 带缓存."""
        if code in self._float_vol_cache:
            return self._float_vol_cache[code]
        try:
            detail = xtdata.get_instrument_detail(self._code_map.get(code, to_xt(code)))
            fv = 0.0
            if detail:
                fv = float(detail.get('FloatVolumn') or detail.get('FloatVolume') or 0)
            self._float_vol_cache[code] = fv
        except Exception:
            self._float_vol_cache[code] = 0.0
        return self._float_vol_cache[code]

    def _download_index(self, start_str, end_str):
        """下载上证指数 (000001.SH)"""
        try:
            xtdata.download_history_data(
                '000001.SH', period='1d',
                start_time=start_str, end_time=end_str,
            )
            time.sleep(0.25)
            raw = xtdata.get_market_data(
                field_list=['open', 'high', 'low', 'close', 'volume', 'amount'],
                stock_list=['000001.SH'],
                period='1d',
                start_time=start_str,
                end_time=end_str,
                dividend_type='none',
                fill_data=False,
            )
            if not raw or 'close' not in raw or raw['close'].empty:
                return None
            close_s = raw['close'].iloc[0]
            idx = close_s.index
            df = pd.DataFrame({
                'date': pd.to_datetime(idx).date,
                'open': raw['open'].iloc[0].values,
                'close': close_s.values,
                'high': raw['high'].iloc[0].values,
                'low': raw['low'].iloc[0].values,
                'volume': raw['volume'].iloc[0].values,
                'amount': raw['amount'].iloc[0].values,
            })
            return {'qfq': df} if not df.empty else None
        except Exception as e:
            print(f"  下载指数失败: {e}")
            return None

    # ── 保存与合并 ────────────────────────────────────────────

    def save_to_raw(self, code, data_dict, overwrite=False):
        """将下载数据写入 raw_data.

        Args:
            overwrite: True = 全量覆盖 (--full 模式), False = 与已有数据合并去重.

        Returns: 实际保存的复权类型列表.
        """
        code = str(code).zfill(6)
        out_dir = self.raw_dir / code
        ensure_dir(out_dir)

        stock_name = '' if code == INDEX_CODE else self.get_stock_name(code)
        saved = []

        for adj_type, df in data_dict.items():
            if df is None or df.empty:
                continue

            file_path = out_dir / f'{adj_type}.csv'

            if code == INDEX_CODE:
                df = df.sort_values('date')
                df.to_csv(file_path, index=False, encoding='utf-8')
                saved.append(adj_type)
                continue

            # 组装 efinance 兼容列顺序
            df_out = df.copy()
            df_out['股票名称'] = stock_name
            df_out['股票代码'] = code
            df_out = df_out[STOCK_COLUMNS]
            df_out['日期'] = pd.to_datetime(df_out['日期']).dt.date
            df_out = df_out.sort_values('日期')

            if overwrite:
                # 全量覆盖: 直接写, 不合并
                df_out.to_csv(file_path, index=False, encoding='utf-8')
                saved.append(adj_type)
                continue

            if file_path.exists():
                try:
                    existing = pd.read_csv(file_path, parse_dates=['日期'], encoding='utf-8')
                    existing['日期'] = existing['日期'].dt.date
                    existing_dates = set(existing['日期'])
                    new_rows = df_out[~df_out['日期'].isin(existing_dates)]
                    if new_rows.empty:
                        saved.append(adj_type)
                        continue
                    combined = pd.concat([existing, new_rows], ignore_index=True)
                    combined = combined.sort_values('日期')
                    combined = combined[STOCK_COLUMNS]  # 统一列顺序
                except Exception:
                    combined = df_out
            else:
                combined = df_out

            combined.to_csv(file_path, index=False, encoding='utf-8')
            saved.append(adj_type)

        return saved

    # ── backtrader 格式转换 ──────────────────────────────────

    def convert_to_backtrader(self, codes=None, start_date='2016-01-01',
                               end_date=None, min_days=100):
        """将 raw_data 转换为 backtrader 格式 (不依赖 efinance).

        直接读取 raw_data CSV, 清洗后写入 backtrader_data 目录.
        """
        if codes is None:
            codes = sorted(
                p.name for p in self.raw_dir.iterdir()
                if p.is_dir() and (p / 'qfq.csv').exists()
            )
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')

        skipped = 0
        updated = 0
        required_cols = ['日期', '开盘', '最高', '最低', '收盘', '成交量']
        total = len(codes)
        t0 = time.time()
        bar_width = 40

        for i, code in enumerate(codes):
            code = str(code).zfill(6) if code != INDEX_CODE else INDEX_CODE
            raw_path = self.raw_dir / code / 'qfq.csv'
            bt_path = self.bt_dir / f'{code}_qfq.csv'

            if not raw_path.exists():
                continue
            if raw_path.stat().st_size < 200:
                continue

            # 指数数据: 列名为英文 (date,open,close,high,low,volume,amount)
            if code == INDEX_CODE:
                try:
                    idx_df = pd.read_csv(raw_path, parse_dates=['date'], encoding='utf-8')
                    idx_df = idx_df[(idx_df['date'] >= start_date) & (idx_df['date'] <= end_date)]
                    if len(idx_df) < min_days:
                        continue
                    bt_df = pd.DataFrame({
                        'datetime': pd.to_datetime(idx_df['date']),
                        'open': idx_df['open'].astype(float),
                        'high': idx_df['high'].astype(float),
                        'low': idx_df['low'].astype(float),
                        'close': idx_df['close'].astype(float),
                        'volume': idx_df['volume'].astype(float),
                        'openinterest': 0,
                        'amount': idx_df['amount'].astype(float),
                    })
                    bt_df = bt_df.sort_values('datetime')
                    # 删除可能存在的WSL symlink, 用真实文件替换
                    if bt_path.is_symlink():
                        bt_path.unlink()
                    bt_df.to_csv(bt_path, index=False, encoding='utf-8')
                    updated += 1
                except OSError as e:
                    print(f'\n[WARN] {code} 文件访问失败: {e}, 跳过')
                except Exception as e:
                    print(f'\n[WARN] {code} 转换失败: {e}, 跳过')
                continue

            # 检查 bt 文件是否已是最新
            try:
                raw_mtime = raw_path.stat().st_mtime
                if bt_path.exists() and bt_path.stat().st_mtime >= raw_mtime:
                    skipped += 1
                    # 进度条
                    n = i + 1
                    if n % 50 == 0 or n == total:
                        pct = n / total
                        filled = int(bar_width * pct)
                        bar = '#' * filled + '-' * (bar_width - filled)
                        elapsed = time.time() - t0
                        rate = n / elapsed if elapsed > 0 else 0
                        eta = (total - n) / rate if rate > 0 else 0
                        print(f'\r  [{bar}] {pct*100:.0f}% {n}/{total} '
                              f'| {updated}新/{skipped}跳过 | {rate:.0f}只/s | ETA {eta:.0f}s  ',
                              end='', flush=True)
                    continue
            except OSError as e:
                print(f'\n[WARN] {code} 文件访问失败: {e}, 跳过')
            else:
                try:
                    df = pd.read_csv(raw_path, parse_dates=['日期'], encoding='utf-8')
                except Exception:
                    continue

                df = df[(df['日期'] >= start_date) & (df['日期'] <= end_date)]
                if len(df) < min_days:
                    continue

                if not all(c in df.columns for c in required_cols):
                    continue

                bt_df = pd.DataFrame({
                    'datetime': pd.to_datetime(df['日期']),
                    'open': df['开盘'].astype(float),
                    'high': df['最高'].astype(float),
                    'low': df['最低'].astype(float),
                    'close': df['收盘'].astype(float),
                    'volume': df['成交量'].astype(float),
                    'openinterest': 0,
                    'amount': df.get('成交额', 0).astype(float),
                })

                for col, bt_col in [('振幅', 'amplitude'), ('涨跌幅', 'change_percent'),
                                    ('涨跌额', 'change_amount'), ('换手率', 'turnover_rate')]:
                    if col in df.columns:
                        bt_df[bt_col] = df[col].astype(float)

                bt_df = bt_df.sort_values('datetime')
                bt_df.to_csv(bt_path, index=False, encoding='utf-8')
                updated += 1

            # 进度条
            n = i + 1
            if n % 50 == 0 or n == total:
                pct = n / total
                filled = int(bar_width * pct)
                bar = '#' * filled + '-' * (bar_width - filled)
                elapsed = time.time() - t0
                rate = n / elapsed if elapsed > 0 else 0
                eta = (total - n) / rate if rate > 0 else 0
                print(f'\r  [{bar}] {pct*100:.0f}% {n}/{total} '
                      f'| {updated}新/{skipped}跳过 | {rate:.0f}只/s | ETA {eta:.0f}s  ',
                      end='', flush=True)

        print()  # newline
        print(f"backtrader数据: 更新 {updated} 只, 跳过 {skipped} 只")
        return updated

    def calc_fq_factors(self, code, overwrite=False):
        """根据 none/qfq/hfq 收盘价计算复权因子并保存.

        Args:
            overwrite: True = 全量覆盖 (--full), False = 增量合并
        """
        if code == INDEX_CODE:
            return

        code = str(code).zfill(6)
        data_dir = self.raw_dir / code
        fq_path = data_dir / 'fq_factors.csv'

        paths = {t: data_dir / f'{t}.csv' for t in ['none', 'qfq', 'hfq']}
        if not all(p.exists() for p in paths.values()):
            return

        try:
            dfs = {}
            for t, p in paths.items():
                df = pd.read_csv(p, parse_dates=['日期'], encoding='utf-8')
                df = df[['日期', '收盘']].copy()
                df.rename(columns={'收盘': f'收盘_{t}'}, inplace=True)
                dfs[t] = df

            merged = dfs['none'].merge(dfs['qfq'], on='日期', how='inner') \
                                .merge(dfs['hfq'], on='日期', how='inner')

            factors = pd.DataFrame({
                '日期': merged['日期'].dt.date,
                'symbol': code,
                'qfq_factor': np.round(
                    merged['收盘_qfq'] / merged['收盘_none'].replace(0, np.nan), 6),
                'hfq_factor': np.round(
                    merged['收盘_hfq'] / merged['收盘_none'].replace(0, np.nan), 6),
            }).dropna()

            if not overwrite and fq_path.exists():
                old = pd.read_csv(fq_path, parse_dates=['日期'])
                old['日期'] = old['日期'].dt.date
                existing_dates = set(old['日期'])
                new_rows = factors[~factors['日期'].isin(existing_dates)]
                if new_rows.empty:
                    return
                factors = pd.concat([old, new_rows], ignore_index=True).sort_values('日期')

            factors.to_csv(fq_path, index=False, encoding='utf-8')
        except Exception as e:
            if '收盘_none' not in str(e) and '收盘_qfq' not in str(e):
                print(f"  计算 fq_factors 失败 {code}: {e}")

    # ── 主循环 ────────────────────────────────────────────────

    def run(self, codes=None, start_date=None, end_date=None, full=False,
            adj_types=('none', 'qfq', 'hfq'), batch_size=50):
        """主下载循环.

        Args:
            codes: 裸代码列表, None = 全部A股
            start_date: 'YYYY-MM-DD' 或 'YYYYMMDD'
            end_date: 同上, 默认今天
            full: True = 强制从 DEFAULT_START 下载
            adj_types: 需要下载的复权类型
            batch_size: 进度报告间隔
        Returns:
            failed_codes: list of codes that failed
        """
        # 获取股票列表
        if codes is None:
            print("获取沪深A股列表...", end=' ', flush=True)
            self._code_map = self.get_a_shares()
            codes = [INDEX_CODE] + sorted(self._code_map.keys())
            print(f"共 {len(codes)} 只 (含指数)")
        elif isinstance(codes, str):
            codes = [codes]

        if end_date is None:
            end_date = datetime.now().strftime('%Y%m%d')
        else:
            end_date = end_date.replace('-', '') if '-' in str(end_date) else end_date

        if start_date:
            start_date = start_date.replace('-', '') if '-' in str(start_date) else start_date

        total = len(codes)
        stats = {'upd': 0, 'new': 0, 'skip': 0}
        failed = []
        t0 = time.time()

        for i, code in enumerate(codes):
            code = str(code).zfill(6) if code != INDEX_CODE else INDEX_CODE

            # ── 确定起止日期 ──
            if full or start_date:
                since = start_date if start_date else DEFAULT_START.replace('-', '')
            else:
                last = self.read_last_date(code, 'qfq')
                since = (last + timedelta(days=1)).strftime('%Y%m%d') if last else DEFAULT_START.replace('-', '')

            # 部分类型可能落后, 取最早的需要日期
            if not full and not start_date:
                for at in adj_types:
                    if at == 'qfq':
                        continue
                    lt = self.read_last_date(code, at)
                    if lt is None:
                        # 该类型文件不存在 → 从 DEFAULT_START 开始
                        since = min(since, DEFAULT_START.replace('-', ''))
                    elif lt < datetime.strptime(since, '%Y%m%d').date():
                        since = min(since, (lt + timedelta(days=1)).strftime('%Y%m%d'))

            # 跳过判断: 用最新交易日而非今天 (周末/节假日无新数据时直接跳过)
            last_trading_day = self._last_trading_day()
            if since > (last_trading_day if last_trading_day else datetime.now().strftime('%Y%m%d')):
                stats['skip'] += 1
                continue

            # ── IPO 日期检查(仅全量/新股票时) ──
            if since < '20200101':
                ipo = self.get_ipo_date(code)
                if ipo and since < ipo:
                    since = max(since, ipo)

            # ── 下载 ──
            had_data = self.read_last_date(code, 'qfq') is not None
            try:
                data = self.download_one(code, since, end_date, adj_types)
            except Exception as e:
                print(f"[{i+1}/{total}] {code} 异常: {e}")
                failed.append(code)
                continue

            if data is None:
                # 不一定是失败: 可能是新股尚未上市 / 停牌 / 无交易日
                failed.append(code)
                continue

            # ── 保存 ──
            saved = self.save_to_raw(code, data, overwrite=full)
            if saved:
                if had_data:
                    stats['upd'] += 1
                else:
                    stats['new'] += 1
                self.calc_fq_factors(code, overwrite=full)

            # ── 进度 ──
            n = i + 1
            if n % batch_size == 0 or n == total:
                elapsed = time.time() - t0
                rate = n / elapsed if elapsed > 0 else 0
                eta = (total - n) / rate if rate > 0 else 0
                print(f"[{n}/{total}] {stats['upd']}更新/{stats['new']}新增/{stats['skip']}跳过"
                      f" | {rate:.1f}只/s | ETA {eta:.0f}s   ", end='\r')
                sys.stdout.flush()

        print()  # newline after progress

        # ── 失败记录 ──
        if failed:
            existing = {}
            if self.failed_log.exists():
                try:
                    loaded = json.loads(self.failed_log.read_text(encoding='utf-8'))
                    if isinstance(loaded, dict):
                        existing = loaded
                except Exception:
                    pass
            existing[datetime.now().strftime('%Y%m%d_%H%M')] = failed
            self.failed_log.write_text(
                json.dumps(existing, ensure_ascii=False, indent=2), encoding='utf-8')
            print(f"{len(failed)} 只失败, 记录在 {self.failed_log}")

        # ── metadata ──
        self.metadata_file.write_text(json.dumps({
            'last_update': datetime.now().strftime('%Y%m%d'),
            'source': 'xtquant',
            'codes': total,
        }, ensure_ascii=False))

        elapsed = time.time() - t0
        print(f"完成: {stats['upd']}更新/{stats['new']}新增/{stats['skip']}跳过/{len(failed)}失败"
              f" | 耗时 {elapsed/60:.1f}分钟")
        return failed


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

def main():
    if not HAS_XTQUANT:
        print("=" * 60)
        print("  错误: xtquant 未安装")
        print("  xtquant 是 QMT 客户端的 Python API, 需在 Windows 上运行.")
        print("  请确保:")
        print("    1. 国金证券 QMT 客户端已安装并登录")
        print("    2. Python 可导入 xtquant (pip install xtquant)")
        print("    3. 本脚本在 Windows 上运行")
        print("=" * 60)
        sys.exit(1)

    parser = argparse.ArgumentParser(
        description='XTQuant A股数据下载器 (国金证券QMT)')
    parser.add_argument('--full', action='store_true',
                        help='全量下载 (从 %s 开始)' % DEFAULT_START)
    parser.add_argument('--since', type=str, default=None,
                        help='起始日期 YYYY-MM-DD 或 YYYYMMDD')
    parser.add_argument('--stock', type=str, default=None,
                        help='只下载指定股票代码')
    parser.add_argument('--failed', action='store_true',
                        help='重试最近一次失败记录中的股票')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='数据目录 (默认 data/stock_data)')
    parser.add_argument('--batch', type=int, default=50,
                        help='进度报告间隔 (默认50)')
    parser.add_argument('--bt', action='store_true',
                        help='下载后同步更新 backtrader 数据')
    parser.add_argument('--bt-only', action='store_true',
                        help='只做 backtrader 转换, 跳过下载阶段')
    args = parser.parse_args()

    dl = XTQuantDownloader(data_dir=args.data_dir)

    if args.bt_only:
        print("======> 只执行 backtrader 转换...")
        dl.convert_to_backtrader()
        return

    # 确定 code 列表
    codes = None
    if args.stock:
        codes = [args.stock]
    elif args.failed:
        if dl.failed_log.exists():
            try:
                fd = json.loads(dl.failed_log.read_text(encoding='utf-8'))
                latest = sorted(fd.keys())[-1]
                codes = fd[latest]
                print(f"重试 {len(codes)} 只失败股票 (批次: {latest})")
            except Exception:
                print("读取失败记录出错")
                return
        else:
            print("无失败记录")
            return

    failed = dl.run(
        codes=codes,
        start_date=args.since,
        full=args.full,
        batch_size=args.batch,
    )

    # 可选: 同步更新 backtrader 数据
    if args.bt:
        print("\n======> 更新 backtrader 数据...")
        dl.convert_to_backtrader(codes=codes)


if __name__ == '__main__':
    main()
