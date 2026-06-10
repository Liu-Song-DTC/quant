# core/alternative_data.py
"""
A股另类数据信号源 — 龙虎榜、北向资金、融资融券。

这些是A股独有的高价值信号，远超传统技术因子：
- 龙虎榜: 机构/游资买卖动向，最强短线信号
- 北向资金: 外资通过沪深港通的资金流向，聪明钱指标
- 融资融券: 杠杆资金情绪，融资余额变化=散户情绪

回测中从本地缓存读取，避免API限流。
"""

import numpy as np
import pandas as pd
import os
from datetime import date as date_type
from pathlib import Path


class AlternativeDataProvider:
    """另类数据统一接口 — 从本地缓存提供因子级信号"""

    def __init__(self, data_dir: str = None):
        if data_dir is None:
            strategy_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            data_dir = os.path.join(strategy_dir, '..', 'data', 'alternative_data')
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # 缓存
        self._dragon_tiger: pd.DataFrame = None      # 龙虎榜
        self._northbound: pd.DataFrame = None         # 北向资金日度
        self._margin: pd.DataFrame = None             # 融资融券

    # ========== 龙虎榜 ==========

    def load_dragon_tiger(self) -> pd.DataFrame:
        """加载龙虎榜数据缓存。若无缓存则尝试下载。

        Returns DataFrame with columns:
            code, date, buy_amount, sell_amount, net_amount,
            institution_buy, institution_sell, top_institution_count
        """
        cache_path = self.data_dir / 'dragon_tiger.parquet'
        if cache_path.exists():
            self._dragon_tiger = pd.read_parquet(cache_path)
            return self._dragon_tiger

        try:
            import akshare as ak
            print("[AltData] 下载龙虎榜数据...")
            df = ak.stock_lhb_detail_em(date='')  # 返回最近一期
            # akshare龙虎榜接口返回日期范围数据，分批下载
            all_data = []
            for year in range(2022, 2027):
                for month in range(1, 13):
                    try:
                        monthly = ak.stock_lhb_stock_statistic_em(
                            symbol=f"{year}{month:02d}")
                        if monthly is not None and len(monthly) > 0:
                            all_data.append(monthly)
                    except Exception:
                        continue
            if all_data:
                df = pd.concat(all_data, ignore_index=True)
                df.to_parquet(cache_path)
                self._dragon_tiger = df
                print(f"[AltData] 龙虎榜: {len(df)} 条 -> {cache_path}")
            return df if all_data else pd.DataFrame()
        except ImportError:
            print("[AltData] akshare未安装，跳过龙虎榜")
            return pd.DataFrame()
        except Exception as e:
            print(f"[AltData] 龙虎榜下载失败: {e}")
            return pd.DataFrame()

    def get_dragon_tiger_signal(self, code: str, date) -> float:
        """获取个股龙虎榜信号 [-1, 1]。

        正值=机构净买入(利好)，负值=游资出货(利空)。
        """
        if self._dragon_tiger is None:
            self._dragon_tiger = self.load_dragon_tiger()
        if self._dragon_tiger is None or len(self._dragon_tiger) == 0:
            return 0.0

        date_str = str(date)[:10]
        # 查找该股票最近20日内的龙虎榜记录
        df = self._dragon_tiger
        code_col = next((c for c in df.columns if '代码' in str(c) or c == 'code'), None)
        date_col = next((c for c in df.columns if '日期' in str(c) or c == 'date'), None)
        if code_col is None or date_col is None:
            return 0.0

        mask = (df[code_col].astype(str).str.contains(code[:6])) if code_col else pd.Series([False]*len(df))
        recent = df[mask]
        if recent.empty:
            return 0.0

        # 最近记录的机构净买入占比
        buy_col = next((c for c in df.columns if '买入' in str(c)), None)
        sell_col = next((c for c in df.columns if '卖出' in str(c)), None)
        if buy_col and sell_col:
            net = recent.iloc[-1][buy_col] - recent.iloc[-1][sell_col]
            total = recent.iloc[-1][buy_col] + recent.iloc[-1][sell_col] + 1e-10
            return float(np.tanh(net / total * 3))
        return 0.0

    # ========== 北向资金 ==========

    def load_northbound(self) -> pd.DataFrame:
        """加载北向资金日度数据。Returns DataFrame with columns:
            date, net_flow(净流入/亿), cumulative_flow, sh_flow, sz_flow
        """
        cache_path = self.data_dir / 'northbound_daily.parquet'
        if cache_path.exists():
            self._northbound = pd.read_parquet(cache_path)
            return self._northbound

        try:
            import akshare as ak
            print("[AltData] 下载北向资金数据...")
            df = ak.stock_hsgt_hist_em(symbol="北向资金")
            if df is not None and len(df) > 0:
                df.columns = ['date', 'net_flow', 'sh_flow', 'sz_flow', 'cumulative']
                df['date'] = pd.to_datetime(df['date'])
                df.to_parquet(cache_path)
                self._northbound = df
                print(f"[AltData] 北向资金: {len(df)} 天 -> {cache_path}")
            return df if df is not None else pd.DataFrame()
        except ImportError:
            print("[AltData] akshare未安装，跳过北向资金")
            return pd.DataFrame()
        except Exception as e:
            print(f"[AltData] 北向资金下载失败: {e}")
            return pd.DataFrame()

    def get_northbound_signal(self, date) -> float:
        """获取北向资金市场级信号 [-1, 1]。

        正值=外资持续流入(利好市场)，负值=外资流出(利空)。
        使用5日累计净流入 / 20日均值 标准化。
        """
        if self._northbound is None:
            self._northbound = self.load_northbound()
        if self._northbound is None or len(self._northbound) == 0:
            return 0.0

        df = self._northbound
        date_str = str(date)[:10]
        mask = df['date'] <= date_str
        if not mask.any():
            return 0.0

        recent = df[mask].tail(20)
        if len(recent) < 5:
            return 0.0

        flow_5d = recent['net_flow'].tail(5).sum()
        flow_avg = recent['net_flow'].mean() + 0.01
        return float(np.tanh(flow_5d / max(abs(flow_avg) * 5, 1)))

    # ========== 融资融券 ==========

    def load_margin(self) -> pd.DataFrame:
        """加载融资融券日度数据。Returns DataFrame with columns:
            date, margin_balance(融资余额/亿), short_balance(融券余额/亿),
            margin_change, margin_buy, total_balance
        """
        cache_path = self.data_dir / 'margin_daily.parquet'
        if cache_path.exists():
            self._margin = pd.read_parquet(cache_path)
            return self._margin

        try:
            import akshare as ak
            print("[AltData] 下载融资融券数据...")
            df = ak.stock_margin_detail_sse(start_date='20220101')
            if df is not None and len(df) > 0:
                df['date'] = pd.to_datetime(df['date'])
                df.to_parquet(cache_path)
                self._margin = df
                print(f"[AltData] 融资融券: {len(df)} 天 -> {cache_path}")
            return df if df is not None else pd.DataFrame()
        except ImportError:
            print("[AltData] akshare未安装，跳过融资融券")
            return pd.DataFrame()
        except Exception as e:
            print(f"[AltData] 融资融券下载失败: {e}")
            return pd.DataFrame()

    def get_margin_signal(self, date) -> float:
        """获取融资融券市场级信号 [-1, 1]。

        正值=融资余额上升(散户看多)，但极度看多=反向信号。
        负值=融资余额下降(去杠杆)，持续下降=恐慌。
        """
        if self._margin is None:
            self._margin = self.load_margin()
        if self._margin is None or len(self._margin) == 0:
            return 0.0

        df = self._margin
        date_str = str(date)[:10]
        mask = df['date'] <= date_str
        if not mask.any():
            return 0.0

        recent = df[mask].tail(10)
        if len(recent) < 5:
            return 0.0

        # 融资余额5日变化率
        bal_col = next((c for c in df.columns if '余额' in str(c) or 'margin_balance' in str(c)), None)
        if bal_col is None:
            return 0.0

        bal_change = (recent[bal_col].iloc[-1] - recent[bal_col].iloc[0]) / (abs(recent[bal_col].iloc[0]) + 1e-10)
        # 温和上升=乐观，剧烈上升=过热(反向)，下降=恐慌
        if abs(bal_change) < 0.03:
            signal = bal_change * 3
        else:
            signal = -bal_change * 2  # 极端=反向
        return float(np.tanh(signal))


# ========== 单例 ==========
_provider: AlternativeDataProvider = None

def get_provider() -> AlternativeDataProvider:
    global _provider
    if _provider is None:
        _provider = AlternativeDataProvider()
    return _provider
