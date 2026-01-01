import akshare as ak
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import json
import pickle
from pathlib import Path
import warnings
import hashlib
warnings.filterwarnings('ignore')

class StockDataManager:
    def __init__(self, data_dir="./stock_data"):
        """初始化数据管理器"""
        self.data_dir = Path(data_dir)
        self.raw_data_dir = self.data_dir / "raw_data"
        self.backtrader_data_dir = self.data_dir / "backtrader_data"
        self.stock_metadata_dir = self.data_dir / "stock_metadata"

        # 创建目录
        self.data_dir.mkdir(exist_ok=True)
        self.raw_data_dir.mkdir(exist_ok=True)
        self.backtrader_data_dir.mkdir(exist_ok=True)
        self.stock_metadata_dir.mkdir(exist_ok=True)
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
            "last_update": ""
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
            "min_volume": -1, # 最低成交量过滤（股）
            "max_stock_count": -1,  # 最大股票数量
            "exclude_st": True,    # 排除ST股票
            "exclude_suspended": True,  # 排除停牌股票
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

    def _get_stock_list(self, force_update=False):
        """获取股票列表"""
        stock_list_file = self.stock_metadata_dir / "stock_list.csv"
        current_date = datetime.today()

        if not force_update and stock_list_file.exists():
            # 检查是否需要更新
            mtime = datetime.strptime(self.metadata["last_update"], "%Y%m%d")
            if (current_date - mtime).days < self.config["update_interval"]:
                print("股票列表在更新间隔内，使用缓存")
                return pd.read_csv(stock_list_file, dtype={'symbol': str})

        self.metadata["last_update"] = current_date.strftime("%Y%m%d")
        self._save_metadata()
        try:
            print("正在获取股票列表...")

            # 获取A股实时数据
            spot_data = ak.stock_zh_a_spot_em()

            # 清理列名
            spot_data.columns = [col.strip().replace(' ', '_').replace('\u200b', '')
                               for col in spot_data.columns]

            # 标准化列名
            column_mapping = {
                '代码': 'symbol',
                '名称': 'name',
                '最新价': 'close',
                '涨跌幅': 'change_pct',
                '涨跌额': 'change',
                '成交量': 'volume',
                '成交额': 'amount',
                '振幅': 'amplitude',
                '最高': 'high',
                '最低': 'low',
                '今开': 'open',
                '昨收': 'pre_close',
                '量比': 'volume_ratio',
                '换手率': 'turnover',
                '市盈率-动态': 'pe_ttm',
                '市净率': 'pb',
                '总市值': 'total_market_cap',
                '流通市值': 'circulating_market_cap',
                '涨速': 'change_speed',
                '5分钟涨跌': 'change_5min',
                '60日涨跌幅': 'change_60d',
                '年初至今涨跌幅': 'change_ytd'
            }

            # 重命名列
            for old_col, new_col in column_mapping.items():
                if old_col in spot_data.columns:
                    spot_data.rename(columns={old_col: new_col}, inplace=True)

            # 确保代码是字符串格式
            spot_data['symbol'] = spot_data['symbol'].astype(str).str.zfill(6)

            # 应用过滤规则
            filtered_data = self._apply_filters(spot_data)

            # 保存股票列表
            filtered_data.to_csv(stock_list_file, index=False, encoding="utf-8")

            # 保存完整数据用于参考
            spot_data.to_csv(self.stock_metadata_dir / "stock_list_full.csv",
                           index=False, encoding="utf-8")

            print(f"股票列表获取完成，过滤后共 {len(filtered_data)} 只股票")
            return filtered_data

        except Exception as e:
            print(f"获取股票列表失败: {e}")
            if stock_list_file.exists():
                return pd.read_csv(stock_list_file, dtype={'symbol': str})
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

        # 2. 排除停牌股票
        if self.config["exclude_suspended"]:
            # 假设停牌股票成交量为0
            mask = filtered_df['volume'] > 0
            filtered_df = filtered_df[mask]
            print(f"排除停牌股票后剩余: {len(filtered_df)}")

        # 3. 按市值过滤
        if self.config["min_market_cap"] > 0:
            mask = filtered_df['total_market_cap'] >= self.config["min_market_cap"] * 1e8
            filtered_df = filtered_df[mask]
            print(f"按市值过滤后剩余: {len(filtered_df)}")

        # 4. 按价格过滤
        if self.config["min_price"] > 0:
            mask = filtered_df['close'] >= self.config["min_price"]
            filtered_df = filtered_df[mask]
            print(f"按价格过滤后剩余: {len(filtered_df)}")

        # 5. 按成交量过滤
        if self.config["min_volume"] > 0:
            mask = filtered_df['volume'] >= self.config["min_volume"]
            filtered_df = filtered_df[mask]
            print(f"按成交量过滤后剩余: {len(filtered_df)}")

        # 6. 限制股票数量
        if self.config["max_stock_count"] > 0 and len(filtered_df) > self.config["max_stock_count"]:
            # 按市值排序，选择大市值股票
            filtered_df = filtered_df.nlargest(self.config["max_stock_count"], 'total_market_cap')
            print(f"限制数量后剩余: {len(filtered_df)}")

        return filtered_df

    def download_raw_data(self, symbol, start_time, save_raw=True):
        """下载原始数据（不做任何处理）"""
        symbol = str(symbol).zfill(6)

        try:
            # 获取上市日期
            ipo_date = self._get_ipo_date(symbol)
            if not ipo_date:
                print(f"无法获取 {symbol} 的上市日期")
                return None

            # 计算开始日期（上市日期或配置的起始日期）
            start_date = max(
                pd.to_datetime(ipo_date),
                start_time
            ).strftime("%Y%m%d")

            end_date = datetime.today().strftime("%Y%m%d")
            if start_date >= end_date:
                return None

            print(f"下载 {symbol} 数据: {start_date} 到 {end_date}")

            # 获取不同复权类型的数据
            data_dict = {}

            # 不复权数据（原始数据）
            try:
                df_none = ak.stock_zh_a_hist(
                    symbol=symbol,
                    period="daily",
                    start_date=start_date,
                    end_date=end_date,
                    adjust=""
                )
                if not df_none.empty:
                    data_dict['none'] = df_none
            except:
                print(f"警告：无法获取 {symbol} 的不复权数据")

            # 前复权数据
            try:
                df_qfq = ak.stock_zh_a_hist(
                    symbol=symbol,
                    period="daily",
                    start_date=start_date,
                    end_date=end_date,
                    adjust="qfq"
                )
                if not df_qfq.empty:
                    data_dict['qfq'] = df_qfq
            except:
                print(f"警告：无法获取 {symbol} 的前复权数据")

            # 后复权数据
            try:
                df_hfq = ak.stock_zh_a_hist(
                    symbol=symbol,
                    period="daily",
                    start_date=start_date,
                    end_date=end_date,
                    adjust="hfq"
                )
                if not df_hfq.empty:
                    data_dict['hfq'] = df_hfq
            except:
                print(f"警告：无法获取 {symbol} 的后复权数据")

            if not data_dict:
                print(f"{symbol}: 所有复权类型数据都为空")
                return None

            # 保存原始数据
            if save_raw:
                raw_data_path = self.raw_data_dir / f"{symbol}"
                raw_data_path.mkdir(exist_ok=True)

                for adj_type, df in data_dict.items():
                    # 标准化列名
                    df.columns = [col.strip() for col in df.columns]
                    df['symbol'] = symbol

                    # 保存为CSV
                    df.to_csv(raw_data_path / f"{adj_type}.csv",
                            index=False, encoding="utf-8")

            # 计算并保存复权因子
            self._calculate_adjustment_factors(symbol, data_dict)

            return data_dict

        except Exception as e:
            print(f"下载 {symbol} 原始数据失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _get_ipo_date(self, symbol):
        """获取上市日期"""
        try:
            stock_info = ak.stock_individual_info_em(symbol=symbol)
            for _, row in stock_info.iterrows():
                if '上市时间' in str(row['item']):
                    return str(row['value']).split()[0]
        except:
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

    def incremental_update(self, symbol):
        """增量更新数据"""
        symbol = str(symbol).zfill(6)
        raw_data_path = self.raw_data_dir / symbol
        raw_data_path.mkdir(exist_ok=True)
        # 检查是否有现有数据
        existing_data = {}
        last_dates = {}
        last_date = pd.to_datetime('1990-01-01')
        if raw_data_path.exists():
            for adj_type in ['none', 'qfq', 'hfq', 'fq_factors']:
                file_path = raw_data_path / f"{adj_type}.csv"
                if file_path.exists():
                    existing_data[adj_type] = pd.read_csv(file_path, parse_dates=['日期'])
                    last_dates[adj_type] = existing_data[adj_type]['日期'].max()
                    existing_data[adj_type]['日期'] = existing_data[adj_type]['日期'].dt.date
            if len(last_dates) == 4:
                last_date = min(last_dates.values()) + timedelta(days=1)

        # 获取最新数据
        new_data = self.download_raw_data(symbol, start_time=last_date, save_raw=False)
        if new_data is None:
            return

        # 合并数据
        for adj_type in ['none', 'qfq', 'hfq', 'fq_factors']:
            if adj_type not in existing_data:
                combined = new_data[adj_type]
            elif adj_type not in new_data:
                combined = existing_data[adj_type]
            else:
                combined = pd.concat([existing_data[adj_type], new_data[adj_type]],
                                   ignore_index=True)
                # 去重
                combined = combined.drop_duplicates(subset=['日期'], keep='last')
                combined = combined.sort_values('日期')

            # 保存
            raw_data_path.mkdir(exist_ok=True)
            combined.to_csv(raw_data_path / f"{adj_type}.csv",
                          index=False, encoding="utf-8")

        return

    def batch_download(self, symbols=None, force=False, max_workers=5):
        """批量下载数据"""
        if symbols is None:
            stock_list = self._get_stock_list()
            if stock_list.empty:
                print("股票列表为空")
                return
            symbols = stock_list['symbol'].tolist()

        print(f"开始批量下载 {len(symbols)} 只股票数据...")

        import concurrent.futures
        from tqdm import tqdm

        def download_single(symbol):
            try:
                self.incremental_update(symbol)
                return (symbol, "success")
            except Exception as e:
                return (symbol, f"error: {str(e)}")

        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(download_single, symbol): symbol for symbol in symbols}  # 限制前100只

            for future in tqdm(concurrent.futures.as_completed(futures),
                             total=len(futures), desc="下载进度"):
                results.append(future.result())

        #  统计结果
        success = sum(1 for _, status in results if status == "success")
        exists = sum(1 for _, status in results if status == "already_exists")
        errors = sum(1 for _, status in results if "error" in status)

        print(f"批量下载完成: 成功 {success}, 已存在 {exists}, 失败 {errors}")

        # 保存下载日志
        log_df = pd.DataFrame(results, columns=['symbol', 'status'])
        log_df.to_csv(self.stock_metadata_dir / "download_log.csv", index=False)

        return results

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

    def create_backtrader_data(self, symbols, adj_type='qfq'):
        """创建Backtrader格式数据"""
        for symbol in symbols:
            symbol = str(symbol).zfill(6)
            raw_data_path = self.raw_data_dir / symbol

            # 检查数据文件
            data_file = raw_data_path / f"{adj_type}.csv"
            if not data_file.exists():
                print(f"{symbol} 的 {adj_type} 数据不存在")
                continue

            try:
                # 读取数据
                df = pd.read_csv(data_file)

                # 标准化列名
                df.columns = [col.strip() for col in df.columns]

                # 确保有必要的列
                required_cols = ['日期', '开盘', '最高', '最低', '收盘', '成交量']
                for col in required_cols:
                    if col not in df.columns:
                        print(f"{symbol} 数据缺少必要列: {col}")
                        continue

                # 准备Backtrader格式
                bt_df = pd.DataFrame({
                    'datetime': pd.to_datetime(df['日期']),
                    'open': df['开盘'].astype(float),
                    'high': df['最高'].astype(float),
                    'low': df['最低'].astype(float),
                    'close': df['收盘'].astype(float),
                    'volume': df['成交量'].astype(float),
                    'openinterest': 0
                })

                # 处理异常值
                bt_df = self._clean_backtrader_data(bt_df, symbol)

                # 创建完整的日期序列（处理停牌）
                bt_df = self._create_complete_series(bt_df)

                # 保存处理后的数据
                output_file = self.backtrader_data_dir/ f"{symbol}_{adj_type}.csv"
                bt_df.to_csv(output_file, index=False)

                print(f"已保存 {symbol} 的Backtrader格式数据: {len(bt_df)} 条记录")

            except Exception as e:
                print(f"创建Backtrader数据失败 {symbol}: {e}")

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
            print(f"  发现问题: {len(quality_report['issues'])} 个")
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
        # 获取股票列表

        universe = []

        for symbol in symbols:
            try:
                # 检查是否有足够的数据
                raw_data_path = self.raw_data_dir / str(symbol).zfill(6) / "qfq.csv"
                if not raw_data_path.exists():
                    continue

                df = pd.read_csv(raw_data_path, parse_dates=['日期'])
                df_filtered = df[(df['日期'] >= start_date) & (df['日期'] <= end_date)]

                # 检查数据天数
                if len(df_filtered) >= min_days:
                    # 检查数据质量
                    if self.validate_data_quality(symbol):
                        universe.append(symbol)

            except Exception as e:
                print(f"检查 {symbol} 时出错: {e}")

        # 保存股票池
        self.create_backtrader_data(universe, adj_type=adj_type)
        print(f"创建回测股票池完成: {len(universe)} 只股票")
        return universe


def main():
    """主函数"""
    # 初始化数据管理器
    manager = StockDataManager()

    print("=" * 50)
    print("股票数据管理系统")
    print("=" * 50)

    # 获取股票列表
    print("\n======> 获取股票列表...")
    stock_list = manager._get_stock_list()
    print(f"股票列表共 {len(stock_list)} 只股票")

    # 批量下载数据（示例：只下载前20只）
    print("\n======> 批量下载数据...")
    sample_symbols = stock_list['symbol'].head(20).tolist()
    manager.batch_download(symbols=sample_symbols, force=False)

    # 创建回测股票池
    print("\n======> 创建回测股票池...")
    universe = manager.create_universe_for_backtest(
        symbols=sample_symbols,
        start_date='2020-01-01',
        end_date='2023-12-31',
        min_days=100
    )

    print("\n" + "=" * 50)
    print("数据管理完成")
    print("=" * 50)

    # 打印汇总信息
    print(f"\n汇总信息:")
    print(f"- 股票列表: {len(stock_list)} 只")
    print(f"- 回测股票池: {len(universe)} 只")
    print(f"- 数据目录: {manager.data_dir}")
    print(f"- 原始数据: {len(list(manager.raw_data_dir.glob('*/*.csv')))} 个文件")
    print(f"- 处理数据: {len(list(manager.backtrader_data_dir.glob('**/*.csv')))} 个文件")


if __name__ == "__main__":
    main()
