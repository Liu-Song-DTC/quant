import akshare_proxy_patch

# 从 trade/config.yaml 读取代理配置
try:
    import yaml
    from pathlib import Path
    _cfg_path = Path(__file__).parent.parent / "trade" / "config.yaml"
    if _cfg_path.exists():
        with open(_cfg_path, "r", encoding="utf-8") as _f:
            _cfg = yaml.safe_load(_f) or {}
        _proxy = _cfg.get("proxy", {})
        if _proxy.get("host") and _proxy.get("auth_token"):
            akshare_proxy_patch.install_patch(
                _proxy["host"],
                auth_token=_proxy["auth_token"],
                retry=_proxy.get("retry", 30),
                hook_domains=[
                    "fund.eastmoney.com",
                    "push2.eastmoney.com",
                    "push2his.eastmoney.com",
                    "emweb.securities.eastmoney.com",
                ],
            )
            print(f"akshare代理已安装: {_proxy['host']}")
except Exception:
    pass

import akshare as ak
import efinance as ef
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import json
import pickle
from pathlib import Path
import hashlib
import shutil
import time
import random
from tqdm import tqdm


INDEX = "sh000001"

class StockDataManager:
    def __init__(self, data_dir=None):
        """初始化数据管理器"""
        if data_dir is None:
            # 默认路径: 脚本所在目录的 stock_data/
            data_dir = Path(__file__).parent / "stock_data"
        self.data_dir = Path(data_dir)
        self.raw_data_dir = self.data_dir / "raw_data"
        self.backtrader_data_dir = self.data_dir / "backtrader_data"
        self.stock_metadata_dir = self.data_dir / "stock_metadata"
        self.fundamental_data_dir = self.data_dir / "fundamental_data"

        # 基本面源数据目录（按日期存储）
        self.fundamental_source_dir = self.stock_metadata_dir / "fundamental_source"

        # 创建目录
        self.data_dir.mkdir(exist_ok=True)
        self.raw_data_dir.mkdir(exist_ok=True)
        self.backtrader_data_dir.mkdir(exist_ok=True)
        self.stock_metadata_dir.mkdir(exist_ok=True)
        self.fundamental_data_dir.mkdir(exist_ok=True)
        self.fundamental_source_dir.mkdir(exist_ok=True)
        self.metadata_path = self.data_dir / "metadata.json"

        # 加载配置
        self.config = self._load_config()

        self.metadata = self._load_metadata()

    def _load_metadata(self):
        """加载元数据"""
        if self.metadata_path.exists():
            with open(self.metadata_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {
            "last_update": "19000101"
        }

    def _save_metadata(self):
        """保存元数据"""
        with open(self.metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

    def _load_config(self):
        """加载配置"""
        config_path = self.data_dir / "config.json"
        default_config = {
            "update_interval": 30,  # 更新间隔（天）
            "min_market_cap": -1,   # 最小市值过滤（亿元）
            "min_price": -1,        # 最低价格过滤（元）
            "min_amount": -1, # 最低成交量过滤（万元）
            "max_stock_count": -1,  # 最大股票数量
            "exclude_st": True,    # 排除ST股票
            "exclude_suspended": True,  # 排除停牌股票
            "exclude_star_board": False,  # 排除科创板(688xxx)
            "backtest_format": "backtrader",  # 回测格式
            "data_version": "1.0"
        }

        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                user_config = json.load(f)
                default_config.update(user_config)

        # 保存配置
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(default_config, f, ensure_ascii=False, indent=2)

        return default_config

    def get_stock_list(self, force_update=False):
        """获取股票列表"""
        stock_full_list_file = self.stock_metadata_dir / "stock_list_full.csv"
        stock_list_file = self.stock_metadata_dir / "stock_list.csv"
        current_date = datetime.today()

        def _filter_stock(data):
            # 应用过滤规则
            filtered_data = self._apply_filters(data)
            # 保存股票列表
            filtered_data.to_csv(stock_list_file, index=False, encoding="utf-8")
            return filtered_data

        if not force_update and stock_full_list_file.exists():
            # 检查是否需要更新
            mtime = datetime.strptime(self.metadata["last_update"], "%Y%m%d")
            if (current_date - mtime).days < self.config["update_interval"]:
                print("股票列表在更新间隔内，使用缓存")
                data = pd.read_csv(stock_full_list_file, dtype={'symbol': str})
                return _filter_stock(data)

        try:
            print("正在获取股票列表...")

            # 使用 efinance 获取A股实时数据（积分 5-10 vs akshare 12-18）
            spot_data = ef.stock.get_realtime_quotes()

            # 标准化列名（efinance 列名）
            column_mapping = {
                '股票代码': 'symbol',
                '股票名称': 'name',
                '最新价': 'close',
                '涨跌幅': 'change_pct',
                '涨跌额': 'change',
                '成交量': 'volume',
                '成交额': 'amount',
                '最高': 'high',
                '最低': 'low',
                '今开': 'open',
                '昨日收盘': 'pre_close',
                '量比': 'volume_ratio',
                '换手率': 'turnover',
                '动态市盈率': 'pe_ttm',
                '总市值': 'total_market_cap',
                '流通市值': 'circulating_market_cap',
            }

            # 重命名列
            for old_col, new_col in column_mapping.items():
                if old_col in spot_data.columns:
                    spot_data.rename(columns={old_col: new_col}, inplace=True)

            # 确保代码是字符串格式
            spot_data['symbol'] = spot_data['symbol'].astype(str).str.zfill(6)

            # 保存完整数据用于参考
            spot_data.to_csv(stock_full_list_file, index=False, encoding="utf-8")
            filtered_data = _filter_stock(spot_data)

            self.metadata["last_update"] = current_date.strftime("%Y%m%d")
            self._save_metadata()
            print(f"股票列表获取完成，过滤后共 {len(filtered_data)} 只股票")
            return filtered_data

        except Exception as e:
            print(f"获取股票列表失败: {e}")
            # 降级: 优先用全量缓存, 其次用过滤后缓存
            if stock_full_list_file.exists():
                print("使用缓存的股票全量列表")
                data = pd.read_csv(stock_full_list_file, dtype={'symbol': str})
                return _filter_stock(data)
            if stock_list_file.exists():
                print("使用缓存的股票过滤列表")
                return pd.read_csv(stock_list_file, dtype={'symbol': str})
            print("无缓存可用，股票列表为空")
            return pd.DataFrame()

    def _apply_filters(self, df):
        """应用过滤规则"""
        if df.empty:
            return df

        filtered_df = df.copy()

        # 1. 排除ST股票
        if self.config["exclude_st"]:
            mask = ~filtered_df['name'].str.contains('ST')
            filtered_df = filtered_df[mask]
            print(f"排除ST股票后剩余: {len(filtered_df)}")

        # 2. 排除科创板 (688xxx)
        if self.config.get("exclude_star_board", False):
            mask = ~filtered_df['symbol'].str.startswith('688')
            filtered_df = filtered_df[mask]
            print(f"排除科创板后剩余: {len(filtered_df)}")

        # 3. 排除停牌股票
        if self.config["exclude_suspended"]:
            # 假设停牌股票成交量为0
            mask = filtered_df['volume'] > 0
            filtered_df = filtered_df[mask]
            print(f"排除停牌股票后剩余: {len(filtered_df)}")

        # 4. 按市值过滤
        if self.config["min_market_cap"] > 0:
            mask = filtered_df['total_market_cap'] >= self.config["min_market_cap"] * 1e8
            filtered_df = filtered_df[mask]
            print(f"按市值过滤后剩余: {len(filtered_df)}")

        # 5. 按价格过滤
        if self.config["min_price"] > 0:
            mask = filtered_df['close'] >= self.config["min_price"]
            filtered_df = filtered_df[mask]
            print(f"按价格过滤后剩余: {len(filtered_df)}")

        # 6. 按成交量过滤
        if self.config["min_amount"] > 0:
            mask = filtered_df['volume'] * filtered_df['close'] >= self.config["min_amount"] * 1e4
            filtered_df = filtered_df[mask]
            print(f"按成交量过滤后剩余: {len(filtered_df)}")

        # 7. 限制股票数量
        if self.config["max_stock_count"] > 0 and len(filtered_df) > self.config["max_stock_count"]:
            # 按市值排序，选择大市值股票
            filtered_df = filtered_df.nlargest(self.config["max_stock_count"], 'total_market_cap')
            print(f"限制数量后剩余: {len(filtered_df)}")

        return filtered_df

    @staticmethod
    def _retry_call(fn, fn_name, max_retries=3, base_delay=2.0):
        """带指数退避的重试调用，仅针对连接类异常"""
        last_err = None
        for attempt in range(max_retries + 1):
            try:
                if attempt > 0:
                    delay = base_delay * (2 ** (attempt - 1)) + random.uniform(0, 1)
                    time.sleep(delay)
                return fn()
            except Exception as e:
                last_err = e
                err_str = str(e)
                is_conn = any(k in type(e).__name__ or k in err_str
                              for k in ('Connection', 'RemoteDisconnected', 'Protocol',
                                        'Timeout', 'ReadTimeout', 'ConnectTimeout'))
                if not is_conn or attempt == max_retries:
                    raise
        raise last_err  # unreachable

    def download_raw_data(self, symbol, start_time, adj_types=None, start_times=None):
        """下载原始数据（不做任何处理）

        adj_types: 需要下载的复权类型列表, 默认全部 ['none','qfq','hfq']
                   增量更新时可只传 ['qfq'] 以减少API调用
        start_times: dict[adj_type -> datetime], 按类型指定起始日期, 避免重复下载
        """
        symbol = str(symbol).zfill(6)
        if adj_types is None:
            adj_types = ['none', 'qfq', 'hfq']

        def _type_start(adj_type):
            """获取某类型的起始日期"""
            if start_times and adj_type in start_times:
                return start_times[adj_type]
            return start_time

        try:
            if symbol == INDEX:
                st = _type_start('qfq')
                start_date = st.strftime("%Y%m%d")
                end_date = datetime.today().strftime("%Y%m%d")
                if start_date >= end_date:
                    return None
                print(f"下载 {symbol} 数据: {start_date} 到 {end_date}")
                sh_index_df = self._retry_call(
                    lambda: ak.stock_zh_index_daily_em(
                        symbol=symbol,
                        start_date=start_date,
                        end_date=end_date,
                    ),
                    fn_name="stock_zh_index_daily_em",
                )
                return {"qfq": sh_index_df}

            # 用所有请求类型的最大起始日期来判断是否需要IPO查询
            effective_start = start_time
            if start_times:
                effective_start = min(start_times.values())
            if effective_start >= pd.to_datetime('2020-01-01'):
                pass  # 增量场景, 跳过IPO查询
            else:
                ipo_date = self._get_ipo_date(symbol)
                if not ipo_date:
                    print(f"无法获取 {symbol} 的上市日期")
                    return None
                # 确保各类型的起始日期不早于IPO日期
                ipo_dt = pd.to_datetime(ipo_date)
                if start_times:
                    for k in start_times:
                        start_times[k] = max(start_times[k], ipo_dt)
                else:
                    start_time = max(start_time, ipo_dt)

            end_date = datetime.today().strftime("%Y%m%d")

            # 紧凑日志: 按类型显示各自下载区间
            if start_times:
                parts = sorted(start_times.keys())  # none, qfq 固定顺序
                ranges = " ".join(f"{t}:{start_times[t].strftime('%Y%m%d')}→{end_date}" for t in parts)
                print(f"下载 {symbol}: {ranges}")
            else:
                print(f"下载 {symbol}: {start_time.strftime('%Y%m%d')} → {end_date}")

            data_dict = {}
            # range_days 用最短的请求区间来估算sleep
            if start_times:
                min_start = min(start_times.values())
            else:
                min_start = start_time
            range_days = (pd.to_datetime(end_date) - min_start).days
            base_sleep = 0.05 if range_days < 30 else 0.5

            # 不复权数据（原始数据）- efinance（积分 1-2）
            if 'none' in adj_types:
                try:
                    type_start = _type_start('none')
                    start_date = type_start.strftime("%Y%m%d")
                    if start_date < end_date:
                        time.sleep(random.uniform(0, base_sleep))
                        df_none = self._retry_call(
                            lambda: ef.stock.get_quote_history(
                                symbol,
                                beg=start_date,
                                end=end_date,
                                klt=101,
                                fqt=0,
                            ),
                            fn_name=f"get_quote_history({symbol}, none)",
                        )
                        if not df_none.empty:
                            data_dict['none'] = df_none
                except Exception:
                    print(f"警告：无法获取 {symbol} 的不复权数据")

            # 前复权数据 - efinance（积分 1-2）
            if 'qfq' in adj_types:
                try:
                    type_start = _type_start('qfq')
                    start_date = type_start.strftime("%Y%m%d")
                    if start_date < end_date:
                        time.sleep(random.uniform(0, base_sleep))
                        df_qfq = self._retry_call(
                            lambda: ef.stock.get_quote_history(
                                symbol,
                                beg=start_date,
                                end=end_date,
                                klt=101,
                                fqt=1,
                            ),
                            fn_name=f"get_quote_history({symbol}, qfq)",
                        )
                        if not df_qfq.empty:
                            data_dict['qfq'] = df_qfq
                except Exception:
                    print(f"警告：无法获取 {symbol} 的前复权数据")

            # 后复权数据 - efinance（积分 1-2）
            if 'hfq' in adj_types:
                try:
                    type_start = _type_start('hfq')
                    start_date = type_start.strftime("%Y%m%d")
                    if start_date < end_date:
                        time.sleep(random.uniform(0, base_sleep))
                        df_hfq = self._retry_call(
                            lambda: ef.stock.get_quote_history(
                                symbol,
                                beg=start_date,
                                end=end_date,
                                klt=101,
                                fqt=2,
                            ),
                            fn_name=f"get_quote_history({symbol}, hfq)",
                        )
                        if not df_hfq.empty:
                            data_dict['hfq'] = df_hfq
                except Exception:
                    print(f"警告：无法获取 {symbol} 的后复权数据")

            if not data_dict:
                print(f"{symbol}: 所有复权类型数据都为空")
                return None

            # 计算并保存复权因子
            self._calculate_adjustment_factors(symbol, data_dict)

            return data_dict

        except Exception as e:
            # 单行简略报错, 不打印完整traceback(避免批量下载时刷屏)
            err_msg = str(e)[:100]
            print(f"下载 {symbol} 失败: {err_msg}")
            return None

    def _get_ipo_date(self, symbol):
        """获取上市日期（efinance）"""
        try:
            info = ef.stock.get_base_info(symbol)
            ipo_date = info.get('上市时间', None)
            if ipo_date and str(ipo_date) != 'nan' and str(ipo_date) != '-':
                return str(ipo_date).split()[0]
        except Exception:
            pass

        # 如果无法获取，使用默认值
        return "1990-01-01"

    def _calculate_adjustment_factors(self, symbol, data_dict):
        """计算复权因子"""
        if 'none' not in data_dict or 'qfq' not in data_dict or 'hfq' not in data_dict:
            return

        try:
            # 合并数据
            df_none = data_dict['none'].copy()
            df_qfq = data_dict['qfq'].copy()
            df_hfq = data_dict['hfq'].copy()

            # 确保日期列一致
            df_none['日期'] = pd.to_datetime(df_none['日期'])
            df_qfq['日期'] = pd.to_datetime(df_qfq['日期'])
            df_hfq['日期'] = pd.to_datetime(df_hfq['日期'])

            # 计算前复权因子（相对不复权的调整）
            merged_qfq = pd.merge(df_none[['日期', '收盘']],
                                df_qfq[['日期', '收盘']],
                                on='日期',
                                suffixes=('_none', '_qfq'))

            # 计算后复权因子（相对不复权的调整）
            merged_hfq = pd.merge(df_none[['日期', '收盘']],
                                df_hfq[['日期', '收盘']],
                                on='日期',
                                suffixes=('_none', '_hfq'))

            # 计算因子
            merged_qfq['qfq_factor'] = merged_qfq['收盘_qfq'] / merged_qfq['收盘_none']
            merged_hfq['hfq_factor'] = merged_hfq['收盘_hfq'] / merged_hfq['收盘_none']

            # 创建因子DataFrame
            factors_df = pd.DataFrame({
                '日期': merged_qfq['日期'].dt.date,
                'symbol': symbol,
                'qfq_factor': merged_qfq['qfq_factor'],
                'hfq_factor': merged_hfq['hfq_factor']
            })

            # 保存因子
            data_dict['fq_factors'] = factors_df

        except Exception as e:
            print(f"计算复权因子失败 {symbol}: {e}")

    def _update_fq_factors_incremental(self, symbol, existing_data, new_data, date_col, raw_data_path):
        """增量追加 fq_factors: 用已下载的 none+qfq 计算新日期的复权因子"""
        try:
            # 取合并后的最新 qfq 和 none 数据
            if 'qfq' not in existing_data or 'none' not in existing_data:
                return
            qfq = existing_data['qfq'].copy()
            none = existing_data['none'].copy()
            qfq[date_col] = pd.to_datetime(qfq[date_col])
            none[date_col] = pd.to_datetime(none[date_col])

            # 只计算 qfq_factor（hfq 需要单独下载，不常用，沿用上期值或跳过）
            merged = pd.merge(
                none[['日期', '收盘']].rename(columns={'收盘': '收盘_none'}),
                qfq[['日期', '收盘']].rename(columns={'收盘': '收盘_qfq'}),
                on='日期', how='inner')
            merged['qfq_factor'] = merged['收盘_qfq'] / merged['收盘_none']

            # hfq_factor: 从已有 fq_factors 取最后一个值沿用
            fq_path = raw_data_path / 'fq_factors.csv'
            last_hfq_factor = 1.0
            if fq_path.exists():
                old_fq = pd.read_csv(fq_path, parse_dates=['日期'])
                if 'hfq_factor' in old_fq.columns and not old_fq.empty:
                    last_hfq_factor = old_fq['hfq_factor'].iloc[-1]

            new_fq = pd.DataFrame({
                '日期': merged['日期'].dt.date,
                'symbol': symbol,
                'qfq_factor': merged['qfq_factor'],
                'hfq_factor': last_hfq_factor,
            })

            # 合并已有 fq_factors，去重保存
            if fq_path.exists():
                old_fq = pd.read_csv(fq_path, parse_dates=['日期'])
                existing_dates = set(old_fq['日期'].astype(str))
                new_fq = new_fq[~new_fq['日期'].astype(str).isin(existing_dates)]
                if new_fq.empty:
                    return
                combined_fq = pd.concat([old_fq, new_fq], ignore_index=True)
                combined_fq = combined_fq.sort_values('日期')
            else:
                combined_fq = new_fq
            combined_fq.to_csv(fq_path, index=False, encoding='utf-8')
        except Exception:
            pass  # fq_factors 非关键路径，失败不影响主流程

    def incremental_update(self, symbol):
        """增量更新数据(自适应)

        逻辑:
        1. 只读日期列快速判断新鲜度(周末看周五, 工作日看今天)
        2. 已是最新则跳过(无I/O浪费); 否则下载增量(起始=最后日期)
        3. 复权校验检测分红/送转, 触发全量重载
        4. 合并保存, 日志只显示实际新增天数
        """
        symbol = str(symbol).zfill(6)
        raw_data_path = self.raw_data_dir / symbol
        raw_data_path.mkdir(exist_ok=True)

        # 快速读取: 只读日期列判断新鲜度(避免读全文件后跳过浪费I/O)
        existing_data = {}
        last_dates = {}
        types = ['none', 'qfq', 'hfq', 'fq_factors']
        date_col = 'date' if symbol == INDEX else '日期'
        if symbol == INDEX:
            types = ['qfq']
        if raw_data_path.exists():
            for adj_type in types:
                file_path = raw_data_path / f"{adj_type}.csv"
                if file_path.exists():
                    # 只读日期列取最大值, 比读全文件快10x
                    date_series = pd.read_csv(file_path, usecols=[date_col])[date_col]
                    last_dates[adj_type] = pd.to_datetime(date_series).max()

        # 智能判断是否需要更新：考虑周末
        today = datetime.today().date()
        weekday = today.weekday()
        if weekday >= 5:  # Sat=5, Sun=6: 无新交易数据，到周五即跳过
            friday = today - timedelta(days=weekday - 4)
            if last_dates and all(v.date() >= friday for v in last_dates.values()):
                return "skip"
        else:
            # 工作日：数据到「今天」才跳过 (盘后更新场景)
            if last_dates and all(v.date() >= today for v in last_dates.values()):
                return "skip"

        # 计算落后天数（用于统计）
        if last_dates:
            gap_days = (today - min(last_dates.values()).date()).days
        else:
            gap_days = None  # 新股，无历史数据

        # 确定下载起始日: 每种类型用自己的最后日期, 避免已最新的类型重复下载浪费API
        if last_dates:
            download_start = min(last_dates.values())
            per_type_start = {t: last_dates[t] for t in ['qfq', 'none'] if t in last_dates}
            new_data = self.download_raw_data(symbol, start_time=download_start,
                                              adj_types=['qfq', 'none'], start_times=per_type_start)
        else:
            download_start = pd.to_datetime('1990-01-01')
            # 新股: 全量下载所有复权类型
            new_data = self.download_raw_data(symbol, start_time=download_start)

        if new_data is None:
            return "fail"

        # 生成状态标签
        if gap_days is None:
            status = "new"
        elif gap_days <= 3:
            status = "ok"
        else:
            status = f"behind:{gap_days}"

        # 确认需要更新后, 再加载全量已有数据(合并用)
        if raw_data_path.exists():
            for adj_type in types:
                file_path = raw_data_path / f"{adj_type}.csv"
                if file_path.exists() and adj_type not in existing_data:
                    existing_data[adj_type] = pd.read_csv(file_path, parse_dates=[date_col])

        # 检测复权变化: 对比重叠日期的qfq收盘价（仅非指数股票）
        fq_changed = False
        if symbol != INDEX and 'qfq' in existing_data and 'qfq' in new_data:
            old_qfq = existing_data['qfq']
            new_qfq = new_data['qfq']
            overlap_dates = set(old_qfq[date_col].astype(str)) & set(new_qfq[date_col].astype(str))
            if overlap_dates:
                old_subset = old_qfq[old_qfq[date_col].astype(str).isin(overlap_dates)]
                new_subset = new_qfq[new_qfq[date_col].astype(str).isin(overlap_dates)]
                old_subset = old_subset.sort_values(date_col).tail(3)
                new_subset = new_subset.sort_values(date_col).tail(3)
                if len(old_subset) > 0 and len(new_subset) > 0:
                    old_close = old_subset['收盘'].values
                    new_close = new_subset['收盘'].values
                    if len(old_close) == len(new_close):
                        diff = abs(old_close - new_close).max()
                        if diff > 0.01:
                            fq_changed = True

        # 复权变了, 全量重新下载所有类型
        if fq_changed:
            full_data = self.download_raw_data(symbol, start_time=pd.to_datetime('1990-01-01'))
            if full_data is not None:
                new_data = full_data

        # 合并数据(仅处理有数据的类型)
        merge_types = [t for t in types if t in existing_data or t in new_data]
        new_dates_count = 0
        for adj_type in merge_types:
            changed = True
            if adj_type not in existing_data:
                combined = new_data[adj_type]
                if adj_type == 'qfq' and not combined.empty:
                    new_dates_count = len(combined)
            elif adj_type not in new_data:
                # 该类型本次未下载, 文件不变, 跳过保存
                continue
            elif adj_type in ('qfq', 'hfq') and fq_changed:
                combined = new_data[adj_type]
                if adj_type == 'qfq':
                    new_dates_count = len(combined)
            else:
                old_df = existing_data[adj_type]
                new_df = new_data[adj_type]
                # 确保日期列类型一致
                old_df[date_col] = pd.to_datetime(old_df[date_col])
                new_df[date_col] = pd.to_datetime(new_df[date_col])
                old_count = len(old_df)
                combined = pd.concat([old_df, new_df], ignore_index=True)
                combined = combined.drop_duplicates(subset=[date_col], keep='last')
                combined = combined.sort_values(date_col)
                if len(combined) == old_count:
                    changed = False  # 无新数据, 跳过写入
                elif adj_type == 'qfq':
                    new_dates_count = len(combined) - old_count

            if changed:
                raw_data_path.mkdir(exist_ok=True)
                combined.to_csv(raw_data_path / f"{adj_type}.csv",
                              index=False, encoding="utf-8")

        # 增量更新 fq_factors: 用新下载的 none+qfq 计算复权因子追加
        if symbol != INDEX and new_dates_count > 0 and 'none' in new_data and 'qfq' in new_data:
            self._update_fq_factors_incremental(
                symbol, existing_data, new_data, date_col, raw_data_path)

        # 精简日志: 只显示实际新增天数
        if new_dates_count > 0:
            new_last = combined[date_col].max()
            print(f"  {symbol}: +{new_dates_count}天 → {new_last.date()}")

        return status

    def batch_download(self, symbols=None, force=False, max_workers=30):
        """批量下载数据"""
        if symbols is None:
            stock_list = self.get_stock_list()
            if stock_list.empty:
                print("股票列表为空")
                return
            symbols = stock_list['symbol'].tolist()

        print(f"开始批量更新 {len(symbols)} 只股票数据...")

        import concurrent.futures
        from tqdm import tqdm

        # 连接健康检查：挑2只普通股票测试（排除指数，API端点不同）
        print("检查数据源连接...")
        regular_symbols = [s for s in symbols if s != INDEX]
        test_symbols = regular_symbols[:2] if len(regular_symbols) >= 2 else regular_symbols
        if not test_symbols:
            print("  无测试标可测，跳过在线更新\n")
            return
        test_failures = 0
        for sym in test_symbols:
            try:
                test_result = self.download_raw_data(sym, start_time=pd.to_datetime('2026-05-01'))
                if test_result is None:
                    test_failures += 1
            except Exception:
                test_failures += 1

        if test_failures == len(test_symbols):
            print(f"⚠ 数据源不可用 (测试 {len(test_symbols)} 只全部失败)")
            print("  跳过在线更新，使用本地缓存数据继续...\n")
            return
        print(f"  连接正常\n")

        # 统计计数器(线程安全用共享list)
        stats = {"skip": 0, "ok": 0, "new": 0, "fail": 0, "behind": []}

        def download_single(symbol):
            try:
                status = self.incremental_update(symbol)
                if status is None:
                    status = "fail"
                return (symbol, status)
            except Exception as e:
                return (symbol, f"error: {str(e)}")

        results = []
        fail_count = [0]
        fail_fast_threshold = min(50, len(symbols) // 10)

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(download_single, symbol): symbol for symbol in symbols}

            for future in tqdm(concurrent.futures.as_completed(futures),
                             total=len(futures), desc="更新进度"):
                result = future.result()
                results.append(result)
                status = str(result[1])
                if "error" in status:
                    fail_count[0] += 1
                    if fail_count[0] >= fail_fast_threshold:
                        for f in futures:
                            f.cancel()
                        print(f"\n⚠ 连续 {fail_count[0]} 次下载失败, API 不可用, 提前退出")
                        break

        # 汇总统计
        behind_list = []
        for _, status in results:
            s = str(status)
            if s == "skip":
                stats["skip"] += 1
            elif s == "ok":
                stats["ok"] += 1
            elif s == "new":
                stats["new"] += 1
            elif s == "fail":
                stats["fail"] += 1
            elif s.startswith("behind:"):
                days = int(s.split(":")[1])
                behind_list.append(days)
            # error:xxx 归入 fail

        n_behind = len(behind_list)
        behind_detail = ""
        if behind_list:
            behind_list.sort(reverse=True)
            max_gap = behind_list[0]
            avg_gap = sum(behind_list) // n_behind
            behind_detail = f", 落后补数据 {n_behind} 只 (最大 {max_gap} 天, 平均 {avg_gap} 天)"

        print(f"批量更新完成: 已最新 {stats['skip']}, 增量 {stats['ok']}, "
              f"新股 {stats['new']}, 失败 {stats['fail']}{behind_detail}")

        # 保存下载日志
        log_df = pd.DataFrame(results, columns=['symbol', 'status'])
        log_df.to_csv(self.stock_metadata_dir / "download_log.csv", index=False)

        # 数据新鲜度终检
        self._check_data_freshness(symbols)

        return results

    def _check_data_freshness(self, symbols: list):
        """数据新鲜度终检 — 确认所有股票数据到最新交易日"""
        today = datetime.today().date()
        wd = today.weekday()
        if wd == 5:
            expected = today - timedelta(days=1)
        elif wd == 6:
            expected = today - timedelta(days=2)
        else:
            expected = today  # 工作日：盘后更新，预期数据到今天

        current_count = 0
        stale = []

        for sym in symbols:
            sym = str(sym).zfill(6)
            qfq_file = self.raw_data_dir / sym / 'qfq.csv'
            if not qfq_file.exists():
                continue
            try:
                # 指数数据用 'date' 列, 个股用 '日期' 列
                date_col = 'date' if sym == INDEX else '日期'
                date_series = pd.read_csv(qfq_file, usecols=[date_col])[date_col]
                last = pd.to_datetime(date_series.iloc[-1]).date()
                gap = (expected - last).days
                if gap <= 1:
                    current_count += 1
                else:
                    stale.append((sym, last, gap))
            except Exception:
                pass

        total = current_count + len(stale)
        if total == 0:
            return

        pct = current_count / total * 100
        print(f"\n数据新鲜度终检 (预期={expected}): {current_count}/{total} 只 ({pct:.1f}%)")
        if stale:
            stale.sort(key=lambda x: -x[2])
            print(f"  ⚠ 落后>1交易日: {len(stale)} 只")
            for sym, last, gap in stale[:10]:
                print(f"    {sym}: 最后数据 {last} (落后{gap}个交易日)")
            if len(stale) > 10:
                print(f"    ... 共 {len(stale)} 只")
        else:
            print(f"  ✓ 全部数据已到最新交易日")

    def _has_data(self, symbol):
        """检查是否有数据"""
        raw_data_path = self.raw_data_dir / str(symbol).zfill(6)
        if not raw_data_path.exists():
            return False

        # 检查是否有至少一种复权类型的数据
        for adj_type in ['none', 'qfq', 'hfq']:
            file_path = raw_data_path / f"{adj_type}.csv"
            if file_path.exists() and file_path.stat().st_size > 100:  # 文件大小大于100字节
                return True

        return False

    def create_backtrader_data(self, symbols, start_date, end_date, adj_type='qfq'):
        """创建Backtrader格式数据（增量追加：仅有新增日期时才更新bt文件）"""
        skipped = 0
        updated = 0
        for symbol in symbols:
            symbol = str(symbol).zfill(6)
            raw_data_path = self.raw_data_dir / symbol

            # 检查数据文件
            data_file = raw_data_path / f"{adj_type}.csv"
            if not data_file.exists():
                continue
            if data_file.stat().st_size < 200:
                continue

            bt_file = self.backtrader_data_dir / f"{symbol}_{adj_type}.csv"
            date_col = 'date' if symbol == INDEX else '日期'
            bt_last_date = None
            bt_tail = None

            # ── 快速判断是否需要更新：只读raw和bt的尾部 ──
            if bt_file.exists():
                try:
                    # 读raw最后几行取最后日期
                    raw_tail = pd.read_csv(data_file, nrows=5)
                    raw_last_date = pd.to_datetime(raw_tail.iloc[-1].iloc[0])
                    # 读bt最后几行取最后日期
                    bt_tail = pd.read_csv(bt_file)
                    bt_tail['datetime'] = pd.to_datetime(bt_tail['datetime'])
                    bt_last_date = bt_tail['datetime'].max()
                    # 无新增 → 跳过
                    if raw_last_date == bt_last_date or raw_last_date == pd.NaT:
                        skipped += 1
                        continue
                except Exception:
                    pass  # 读尾部失败，做全量重建

            try:
                if symbol == INDEX:
                    # 指数: 逻辑简单, 保持全量读写
                    df = pd.read_csv(data_file, parse_dates=['date'])
                    df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
                    bt_df = pd.DataFrame({
                        'datetime': pd.to_datetime(df['date']),
                        'open': df['open'].astype(float),
                        'high': df['high'].astype(float),
                        'low': df['low'].astype(float),
                        'close': df['close'].astype(float),
                        'volume': df['volume'].astype(float),
                        'openinterest': 0,
                        'amount': df['amount'].astype(float),
                    })
                else:
                    if bt_file.exists() and bt_last_date is not None and raw_last_date is not None and raw_last_date > bt_last_date:
                        # ── 增量追加: 只读新增日期 + 追加到已有bt文件 ──
                        new_raw = pd.read_csv(data_file, parse_dates=[date_col])
                        new_raw = new_raw[new_raw[date_col] > bt_last_date]
                        if new_raw.empty:
                            skipped += 1
                            continue

                        new_raw.columns = [col.strip() for col in new_raw.columns]
                        required_cols = ['日期', '开盘', '最高', '最低', '收盘', '成交量']
                        if not all(c in new_raw.columns for c in required_cols):
                            continue

                        new_bt = pd.DataFrame({
                            'datetime': pd.to_datetime(new_raw['日期']),
                            'open': new_raw['开盘'].astype(float),
                            'high': new_raw['最高'].astype(float),
                            'low': new_raw['最低'].astype(float),
                            'close': new_raw['收盘'].astype(float),
                            'volume': new_raw['成交量'].astype(float),
                            'openinterest': 0,
                            'amount': new_raw['成交额'].astype(float),
                            'amplitude': new_raw['振幅'].astype(float),
                            'change_percent': new_raw['涨跌幅'].astype(float),
                            'change_amount': new_raw['涨跌额'].astype(float),
                            'turnover_rate': new_raw['换手率'].astype(float)
                        })
                        # 对新增行做简化清洗（跳过quantile，行太少无意义）
                        new_bt = new_bt[(new_bt['open'] > 0) & (new_bt['high'] >= new_bt['low'])]
                        # 拼接已有 + 新增
                        bt_df = pd.concat([bt_tail, new_bt], ignore_index=True)
                    else:
                        # ── 全量构建(新股或尾部读取失败) ──
                        df = pd.read_csv(data_file, parse_dates=[date_col])
                        df = df[(df[date_col] >= start_date) & (df[date_col] <= end_date)]
                        df.columns = [col.strip() for col in df.columns]
                        required_cols = ['日期', '开盘', '最高', '最低', '收盘', '成交量']
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
                            'amount': df['成交额'].astype(float),
                            'amplitude': df['振幅'].astype(float),
                            'change_percent': df['涨跌幅'].astype(float),
                            'change_amount': df['涨跌额'].astype(float),
                            'turnover_rate': df['换手率'].astype(float)
                        })
                        bt_df = self._clean_backtrader_data(bt_df, symbol)

                # 创建完整的日期序列（处理停牌）
                bt_df = self._create_complete_series(bt_df)

                # 保存
                bt_df.to_csv(bt_file, index=False)
                updated += 1

            except Exception as e:
                print(f"创建Backtrader数据失败 {symbol}: {e}")

        print(f"Backtrader数据: 更新 {updated} 只, 跳过 {skipped} 只(已是最新)")

    def _clean_backtrader_data(self, df, symbol):
        """清理Backtrader数据"""
        if df.empty:
            return df

        df_clean = df.copy()

        # 1. 移除无效价格
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            df_clean = df_clean[df_clean[col] > 0]

        # 2. 检查价格一致性
        df_clean = df_clean[
            (df_clean['high'] >= df_clean['low']) &
            (df_clean['high'] >= df_clean['close']) &
            (df_clean['high'] >= df_clean['open']) &
            (df_clean['close'] >= df_clean['low']) &
            (df_clean['open'] >= df_clean['low'])
        ]

        # 3. 处理极端值（使用分位数）
        for col in price_cols:
            q_low = df_clean[col].quantile(0.001)
            q_high = df_clean[col].quantile(0.999)
            df_clean = df_clean[(df_clean[col] >= q_low) & (df_clean[col] <= q_high)]

        # 4. 处理成交量异常
        if 'volume' in df_clean.columns:
            # 成交量不能为负
            df_clean = df_clean[df_clean['volume'] >= 0]

            # 处理极端成交量
            q_high_vol = df_clean['volume'].quantile(0.999)
            df_clean.loc[df_clean['volume'] > q_high_vol, 'volume'] = q_high_vol

        return df_clean

    def _create_complete_series(self, df):
        """创建完整的时间序列"""
        if df.empty:
            return df

        df = df.sort_values('datetime')

        # 创建完整的日期范围
        start_date = df['datetime'].min()
        end_date = df['datetime'].max()

        # 获取交易日历（这里简化处理，实际应该使用交易所日历）
        from pandas.tseries.offsets import BDay
        all_dates = pd.date_range(start=start_date, end=end_date, freq='B')

        # 创建完整序列
        date_df = pd.DataFrame({'datetime': all_dates})
        merged = pd.merge(date_df, df, on='datetime', how='left')

        # 填充缺失的交易日（停牌）
        # 价格使用前值填充
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            merged[col] = merged[col].ffill()

        # 成交量填0
        if 'volume' in merged.columns:
            merged['volume'] = merged['volume'].fillna(0)

        # openinterest填0
        merged['openinterest'] = merged['openinterest'].fillna(0)

        return merged

    def validate_data_quality(self, symbol):
        """验证数据质量"""
        symbol = str(symbol).zfill(6)
        raw_data_path = self.raw_data_dir / symbol

        if not raw_data_path.exists():
            print(f"股票 {symbol} 数据不存在")
            return False

        quality_report = {
            'symbol': symbol,
            'data_exists': True,
            'data_files': [],
            'records_count': {},
            'date_range': {},
            'issues': []
        }

        # 检查每种复权类型的数据
        for adj_type in ['none', 'qfq', 'hfq']:
            file_path = raw_data_path / f"{adj_type}.csv"
            if file_path.exists():
                try:
                    df = pd.read_csv(file_path)
                    quality_report['data_files'].append(f"{adj_type}.csv")
                    quality_report['records_count'][adj_type] = len(df)

                    if '日期' in df.columns:
                        dates = pd.to_datetime(df['日期'])
                        quality_report['date_range'][adj_type] = {
                            'start': dates.min(),
                            'end': dates.max(),
                            'days': (dates.max() - dates.min()).days
                        }

                    # 检查数据质量问题
                    issues = self._check_data_issues(df, adj_type)
                    if issues:
                        quality_report['issues'].extend(issues)

                except Exception as e:
                    quality_report['issues'].append(f"读取 {adj_type} 数据失败: {e}")
            else:
                quality_report['issues'].append(f"缺少 {adj_type} 数据文件")

        # 保存质量报告
        report_path = self.stock_metadata_dir / "quality_reports"
        report_path.mkdir(exist_ok=True)

        with open(report_path / f"{symbol}.json", 'w', encoding='utf-8') as f:
            json.dump(quality_report, f, default=str, ensure_ascii=False, indent=2)

        # 打印摘要
        if quality_report['issues']:
            print(f" {symbol} 发现问题: {len(quality_report['issues'])} 个")
            for issue in quality_report['issues'][:3]:  # 只显示前3个问题
                print(f"    - {issue}")

        return len(quality_report['issues']) == 0

    def _check_data_issues(self, df, adj_type):
        """检查数据问题"""
        issues = []

        if df.empty:
            issues.append(f"{adj_type}数据为空")
            return issues

        # 检查必要的列
        required_cols = ['日期', '开盘', '收盘', '最高', '最低', '成交量']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            issues.append(f"缺少列: {', '.join(missing_cols)}")

        # 检查缺失值
        numeric_cols = ['开盘', '收盘', '最高', '最低', '成交量']
        for col in numeric_cols:
            if col in df.columns:
                null_count = df[col].isnull().sum()
                if null_count > 0:
                    issues.append(f"{col} 有 {null_count} 个缺失值")

        # 检查价格合理性
        if all(col in df.columns for col in ['最高', '最低', '开盘', '收盘']):
            invalid_high_low = df[df['最高'] < df['最低']]
            if len(invalid_high_low) > 0:
                issues.append(f"最高价 < 最低价 的记录: {len(invalid_high_low)}")

            invalid_open = df[(df['开盘'] < df['最低']) | (df['开盘'] > df['最高'])]
            if len(invalid_open) > 0:
                issues.append(f"开盘价不在[最低价, 最高价]范围内的记录: {len(invalid_open)}")

            invalid_close = df[(df['收盘'] < df['最低']) | (df['收盘'] > df['最高'])]
            if len(invalid_close) > 0:
                issues.append(f"收盘价不在[最低价, 最高价]范围内的记录: {len(invalid_close)}")

        return issues

    def create_universe_for_backtest(self, symbols, start_date, end_date, adj_type='qfq', min_days=100):
        """创建回测股票池"""
        # 先检查有多少股票有足够数据
        universe = []

        for symbol in symbols:
            try:
                if symbol == INDEX:
                    universe.append(symbol)
                    continue
                raw_data_path = self.raw_data_dir / str(symbol).zfill(6) / (adj_type + ".csv")
                if not raw_data_path.exists():
                    continue

                df = pd.read_csv(raw_data_path, parse_dates=['日期'])
                df_filtered = df[(df['日期'] >= start_date) & (df['日期'] <= end_date)]

                if len(df_filtered) >= min_days:
                    if self.validate_data_quality(symbol):
                        universe.append(symbol)

            except Exception as e:
                print(f"检查 {symbol} 时出错: {e}")

        if len(universe) <= 1:  # 只有 index 或空
            print(f"有效股票池太小 ({len(universe)}只)，跳过重建，保留现有回测数据")
            return universe

        # 只有在有足够股票时才更新
        print(f"增量更新回测数据目录 (股票池 {len(universe)} 只)...")
        os.makedirs(self.backtrader_data_dir, exist_ok=True)

        # 保存股票池
        self.create_backtrader_data(universe, start_date, end_date, adj_type=adj_type)
        print(f"创建回测股票池完成: {len(universe)} 只股票")
        return universe

    # ==================== 基本面数据管理 ====================

    def get_needed_financial_dates(self):
        """智能获取需要更新的财务日期（自动检测已有数据）"""
        # 需要4个文件都存在才算完整
        required_files = ['yjbb.csv', 'zcfz.csv', 'lrb.csv', 'xjll.csv']

        # 已有完整数据的日期
        existing_dates = set()
        for date_dir in self.fundamental_source_dir.iterdir():
            if date_dir.is_dir():
                files = [f.name for f in date_dir.iterdir()]
                if all(rf in files for rf in required_files):
                    existing_dates.add(date_dir.name)

        # 需要获取的日期范围（从2010年到现在）
        current_year = datetime.today().year
        current_month = datetime.today().month
        # 披露时间线: 年报/一季报→4月底, 中报→8月底, 三季报→10月底
        if current_month >= 10:
            max_quarter = f"{current_year}0930"
        elif current_month >= 8:
            max_quarter = f"{current_year}0630"
        elif current_month >= 4:
            max_quarter = f"{current_year}0331"
        else:
            max_quarter = f"{current_year - 1}1231"
        all_dates = []
        for year in range(2010, current_year + 1):
            for month in ['0331', '0630', '0930', '1231']:
                date = f"{year}{month}"
                if date > max_quarter:
                    continue
                # 只返回不完整的日期
                if date not in existing_dates:
                    all_dates.append(date)

        return all_dates

    def download_financial_data_by_date(self, date):
        """按日期下载所有股票的财务数据（源文件）"""
        os.makedirs(self.fundamental_source_dir / f"{date}", exist_ok=True)
        def help(source_file, func):
            try:
                if source_file.exists():
                    return True
                df = func(date=date)
                time.sleep(0.3)  # 请求间隔，避免被限流

                if df is not None and len(df) > 0:
                    df.to_csv(source_file, index=False, encoding="utf-8")
                    print(f"  {date}: 获取 {len(df)} 条记录")
                    return True
            except Exception as e:
                print(f"  警告: {date} 获取失败: {e}")
                return False
        # efinance 日期格式为 YYYY-MM-DD
        _ef_date = f"{date[:4]}-{date[4:6]}-{date[6:8]}"
        help(self.fundamental_source_dir / f"{date}" / "yjbb.csv", lambda date=date: ef.stock.get_all_company_performance(_ef_date))
        help(self.fundamental_source_dir / f"{date}" / "zcfz.csv", ak.stock_zcfz_em)
        help(self.fundamental_source_dir / f"{date}" / "lrb.csv", ak.stock_lrb_em)
        help(self.fundamental_source_dir / f"{date}" / "xjll.csv", ak.stock_xjll_em)
        return True

    def update_fundamental_source(self):
        """更新基本面源数据（按日期，自动检测已有数据）"""
        dates = self.get_needed_financial_dates()

        if not dates:
            print("基本面源数据已是最新，无需更新")
            return 0, 0

        print(f"开始更新基本面源数据 ({len(dates)} 个日期)...")

        success = 0
        fail = 0
        for date in tqdm(dates, desc="下载财务数据"):
            if self.download_financial_data_by_date(date):
                success += 1
            else:
                fail += 1

        print(f"源数据更新完成: 成功 {success}, 失败 {fail}")
        return success, fail

    def build_stock_fundamental_history(self):
        """从源数据构建每只股票的历史财务数据（按日期分别合并4个表，再汇总）"""
        # 获取股票列表（只保留当前上市股票）
        stock_list_file = self.stock_metadata_dir / "stock_list_full.csv"
        if not stock_list_file.exists():
            print("未找到股票列表文件 stock_list_full.csv")
            return

        stock_df = pd.read_csv(stock_list_file, dtype={'symbol': str})
        valid_codes = set(stock_df['symbol'].tolist())
        print(f"只保留 stock_list_full.csv 中的 {len(valid_codes)} 只股票")

        print("步骤1: 读取并合并数据（按日期）...")

        # 获取所有日期目录
        date_dirs = sorted([d for d in self.fundamental_source_dir.iterdir() if d.is_dir()])

        # 先读取所有文件，按日期合并
        all_merged = []

        for date_dir in tqdm(date_dirs, desc="处理日期"):
            date = date_dir.name

            # 读取4个报表文件
            yjbb_df = None
            zcfz_df = None
            lrb_df = None
            xjll_df = None

            # 业绩报表
            yjbb_file = date_dir / "yjbb.csv"
            if yjbb_file.exists():
                yjbb_df = pd.read_csv(yjbb_file, dtype={'股票代码': str})

            # 资产负债表
            zcfz_file = date_dir / "zcfz.csv"
            if zcfz_file.exists():
                zcfz_df = pd.read_csv(zcfz_file, dtype={'股票代码': str})
                zcfz_df = zcfz_df.add_prefix('zcfz_')
                zcfz_df = zcfz_df.rename(columns={'zcfz_股票代码': '股票代码'})

            # 利润表
            lrb_file = date_dir / "lrb.csv"
            if lrb_file.exists():
                lrb_df = pd.read_csv(lrb_file, dtype={'股票代码': str})
                lrb_df = lrb_df.add_prefix('lrb_')
                lrb_df = lrb_df.rename(columns={'lrb_股票代码': '股票代码'})

            # 现金流量表
            xjll_file = date_dir / "xjll.csv"
            if xjll_file.exists():
                xjll_df = pd.read_csv(xjll_file, dtype={'股票代码': str})
                xjll_df = xjll_df.add_prefix('xjll_')
                xjll_df = xjll_df.rename(columns={'xjll_股票代码': '股票代码'})

            # 合并该日期的所有数据
            if yjbb_df is not None:
                yjbb_df['报告期'] = date
                yjbb_df['数据可用日期'] = date  # 添加数据可用日期，防止数据泄露
                merged = yjbb_df

                # 按股票代码合并其他表
                if zcfz_df is not None and not zcfz_df.empty:
                    merged = merged.merge(zcfz_df, on='股票代码', how='left')
                if lrb_df is not None and not lrb_df.empty:
                    merged = merged.merge(lrb_df, on='股票代码', how='left')
                if xjll_df is not None and not xjll_df.empty:
                    merged = merged.merge(xjll_df, on='股票代码', how='left')

                all_merged.append(merged)

        if not all_merged:
            print("没有数据可处理")
            return

        # 合并所有日期
        print("步骤2: 合并所有日期数据...")
        df_all = pd.concat(all_merged, ignore_index=True)

        # 获取所有股票代码（只保留 stock_list_full.csv 中的股票）
        all_codes = df_all['股票代码'].unique().tolist()
        stock_codes = [c for c in all_codes if c in valid_codes]
        print(f"共 {len(stock_codes)} 只股票需要处理")

        # 按股票保存
        print("步骤3: 按股票保存...")
        success = 0

        for code in tqdm(stock_codes, desc="保存股票数据"):
            output_file = self.fundamental_data_dir / f"{code}.csv"

            # 已存在则跳过（增量更新）
            if output_file.exists():
                continue

            stock_data = df_all[df_all['股票代码'] == code]
            if len(stock_data) > 0:
                stock_data.to_csv(output_file, index=False, encoding="utf-8")
                success += 1

        print(f"构建完成: 成功 {success}")
        return success

    def incremental_update_fundamental(self):
        """增量更新基本面数据（只更新有新数据的部分）"""
        print("\n======> 增量更新基本面数据...")

        # Step 1: 检查是否有缺失的源数据
        needed = self.get_needed_financial_dates()
        if not needed:
            print("基本面源数据已是最新，跳过更新")
            return

        # Step 2: 更新缺失的源数据
        print(f"需要更新 {len(needed)} 个季度的源数据...")
        self.update_fundamental_source()

        # Step 3: 从源文件构建股票历史数据
        print("\n构建股票历史数据...")
        self.build_stock_fundamental_history()

        print("\n基本面数据更新完成!")


def main():
    """主函数"""
    # 初始化数据管理器
    manager = StockDataManager()

    print("=" * 50)
    print("股票数据管理系统")
    print("=" * 50)

    #  获取股票列表
    print("\n======> 获取股票列表...")
    stock_list = manager.get_stock_list()
    print(f"股票列表共 {len(stock_list)} 只股票")

    if len(stock_list) == 0:
        print("股票列表为空，可能网络不可用且无缓存，退出")
        return

    sample_symbols = stock_list['symbol'].tolist()
    sample_symbols.insert(0, INDEX)

    #  批量更新数据 (增量, 网络失败时跳过)
    print("\n======> 增量更新数据...")
    from datetime import datetime as dt
    today = dt.today().strftime('%Y-%m-%d')
    try:
        manager.batch_download(symbols=sample_symbols, force=False)
    except Exception as e:
        print(f"数据下载失败 (网络不可用): {e}")
        print("将使用本地缓存数据继续...")

    #  创建/更新回测格式数据
    print("\n======> 创建回测股票池...")
    universe = manager.create_universe_for_backtest(
        symbols=sample_symbols,
        start_date='2024-01-01',
        end_date=today,
        min_days=100,
    )

    print("\n" + "=" * 50)
    print("数据管理完成")
    print("=" * 50)

    #  打印汇总信息
    print(f"\n汇总信息:")
    print(f"- 股票列表: {len(stock_list)} 只")
    print(f"- 回测股票池: {len(universe)} 只")
    print(f"- 数据目录: {manager.data_dir}")
    print(f"- 原始数据: {len(list(manager.raw_data_dir.glob('*/*.csv')))} 个文件")
    print(f"- 处理数据: {len(list(manager.backtrader_data_dir.glob('**/*.csv')))} 个文件")

    # 增量更新基本面数据
    print("\n======> 增量更新基本面数据...")
    manager.incremental_update_fundamental()


if __name__ == "__main__":
    main()
