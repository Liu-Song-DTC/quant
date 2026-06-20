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

        # 龙虎榜历史明细: 懒加载, 首次get_dragon_tiger_dates时自动下载
        self._dt_history = None

    # ========== 龙虎榜 ==========

    def load_dragon_tiger(self) -> pd.DataFrame:
        """加载龙虎榜数据缓存。若无缓存则尝试下载。

        Returns DataFrame with columns:
            code, date, buy_amount, sell_amount, net_amount,
            institution_buy, institution_sell, top_institution_count
        """
        cache_path = self.data_dir / 'dragon_tiger.pkl'
        if cache_path.exists():
            self._dragon_tiger = pd.read_pickle(cache_path)
            return self._dragon_tiger

        try:
            import akshare as ak
            print("[AltData] 下载龙虎榜数据...")
            # akshare 1.18: symbol 参数改为 '近一月'/'近三月'/'近一年'
            df = ak.stock_lhb_stock_statistic_em(symbol='近一年')
            if df is not None and len(df) > 0:
                df.to_pickle(cache_path)
                self._dragon_tiger = df
                print(f"[AltData] 龙虎榜: {len(df)} 条 -> {cache_path}")
            return df if df is not None else pd.DataFrame()
        except ImportError:
            print("[AltData] akshare未安装，跳过龙虎榜")
            return pd.DataFrame()
        except Exception as e:
            print(f"[AltData] 龙虎榜下载失败: {e}")
            return pd.DataFrame()

    def _ensure_dragon_tiger_history(self):
        """确保龙虎榜历史明细缓存存在, 不存在则从聚合数据构建或自动下载。"""
        history_path = self.data_dir / 'dragon_tiger_history.pkl'
        if history_path.exists():
            if self._dt_history is None:
                import pickle
                self._dt_history = pickle.loads(history_path.read_bytes())
            return

        # 优先从已有聚合缓存构建（秒级，避免重复下载）
        agg_path = self.data_dir / 'dragon_tiger.pkl'
        import pickle
        if agg_path.exists():
            try:
                df = pd.read_pickle(agg_path)
                if df is not None and len(df) > 0:
                    code_col = next((c for c in df.columns if '代码' in str(c) or c == 'code'), None)
                    date_col = next((c for c in df.columns if '日期' in str(c) or '上榜' in str(c) or c == 'date'), None)
                    if code_col and date_col:
                        df['code_6'] = df[code_col].astype(str).str.extract(r'(\d{6})', expand=False)
                        df['dt'] = pd.to_datetime(df[date_col]).dt.date
                        self._dt_history = {}
                        for code, grp in df.groupby('code_6'):
                            if code and len(str(code)) == 6:
                                self._dt_history[code] = set(grp['dt'].unique())
                        history_path.write_bytes(pickle.dumps(self._dt_history))
                        return
            except Exception:
                pass

        # 下载（仅当无缓存时，应在主进程预先调用避免worker竞争）
        try:
            import akshare as ak
            import time as _time
            import re
            print("[AltData] 下载龙虎榜历史明细 (2024-01 ~ 2026-06)...")
            all_records = []
            for year in [2024, 2025, 2026]:
                for month in range(1, 13):
                    if year == 2026 and month > 6:
                        break
                    date_str = f'{year}-{month:02d}'
                    try:
                        df = ak.stock_lhb_detail_daily_em(date=date_str)
                        if df is not None and len(df) > 0:
                            all_records.append(df)
                        _time.sleep(0.3)
                    except Exception:
                        pass
            if all_records:
                merged = pd.concat(all_records, ignore_index=True)
                code_col = next((c for c in merged.columns if '代码' in str(c)), None)
                date_col = next((c for c in merged.columns if '日期' in str(c)), None)
                if code_col and date_col:
                    merged['code_6'] = merged[code_col].astype(str).str.extract(r'(\d{6})', expand=False)
                    merged['dt'] = pd.to_datetime(merged[date_col]).dt.date
                    self._dt_history = {}
                    for code, grp in merged.groupby('code_6'):
                        if code and len(str(code)) == 6:
                            self._dt_history[code] = set(grp['dt'].unique())
                    history_path.write_bytes(pickle.dumps(self._dt_history))
                    print(f"[AltData] 龙虎榜历史: {len(self._dt_history)} 只股票 -> {history_path}")
        except ImportError:
            pass  # akshare未安装, 静默跳过
        except Exception as e:
            print(f"[AltData] 龙虎榜历史下载失败: {e}")

    def get_dragon_tiger_dates(self, code: str, query_date=None):
        """返回该股票出现在龙虎榜的日期集合（<= query_date），用于逐bar快速判断是否有信号。

        优先使用历史明细缓存(dragon_tiger_history.pkl), 回退到聚合数据。
        """
        self._ensure_dragon_tiger_history()

        if self._dt_history is not None:
            code_key = str(code)[:6].zfill(6) if len(str(code)) >= 6 else str(code).zfill(6)
            dates_set = self._dt_history.get(code_key, set())
            if query_date is not None:
                try:
                    cutoff = pd.to_datetime(str(query_date)[:10]).date()
                    dates_set = {d for d in dates_set if d <= cutoff}
                except Exception:
                    pass
            return dates_set

        # 回退: 聚合数据 (只存最近上榜日)
        if self._dragon_tiger is None:
            self._dragon_tiger = self.load_dragon_tiger()
        if self._dragon_tiger is None or len(self._dragon_tiger) == 0:
            return set()

        df = self._dragon_tiger
        code_col = next((c for c in df.columns if '代码' in str(c) or c == 'code'), None)
        date_col = next((c for c in df.columns if '日期' in str(c) or '上榜' in str(c) or c == 'date'), None)
        if code_col is None or date_col is None:
            return set()

        code_strs = df[code_col].astype(str).str.extract(r'(\d{6})', expand=False)
        mask = code_strs == code[:6]
        if not mask.any():
            return set()
        if query_date is not None:
            try:
                dt_series = pd.to_datetime(df[date_col])
                cutoff = pd.to_datetime(str(query_date)[:10])
                mask = mask & (dt_series <= cutoff)
            except Exception:
                print(f"[AltData] 龙虎榜日期解析失败(get_dragon_tiger_dates), date_col={date_col}")
        dates = pd.to_datetime(df.loc[mask, date_col]).dt.date
        return set(dates)

    def get_dragon_tiger_signal(self, code: str, date) -> float:
        """获取个股龙虎榜信号 [-1, 1]。

        正值=机构净买入(利好)，负值=游资出货(利空)。
        """
        if self._dragon_tiger is None:
            self._dragon_tiger = self.load_dragon_tiger()
        if self._dragon_tiger is None or len(self._dragon_tiger) == 0:
            return 0.0

        date_str = str(date)[:10]
        df = self._dragon_tiger
        code_col = next((c for c in df.columns if '代码' in str(c) or c == 'code'), None)
        date_col = next((c for c in df.columns if '日期' in str(c) or '上榜' in str(c) or c == 'date'), None)
        if code_col is None or date_col is None:
            return 0.0

        code_strs = df[code_col].astype(str).str.extract(r'(\d{6})', expand=False)
        code_mask = code_strs == code[:6] if code_col else pd.Series([False]*len(df))
        # 日期过滤：仅使用 <= 查询日期的记录，防止前视偏差
        try:
            dt_series = pd.to_datetime(df[date_col])
            date_mask = dt_series <= pd.to_datetime(date_str)
        except Exception:
            print(f"[AltData] 龙虎榜日期解析失败，date_col={date_col}")
            date_mask = pd.Series([True] * len(df))
        mask = code_mask & date_mask
        recent = df[mask].sort_values(date_col) if date_col and not df[mask].empty else df[mask]
        if recent.empty:
            return 0.0

        # 按日期排序后取最近记录的机构净买入占比
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
        cache_path = self.data_dir / 'northbound_daily.pkl'
        if cache_path.exists():
            self._northbound = pd.read_pickle(cache_path)
            return self._northbound

        try:
            import akshare as ak
            print("[AltData] 下载北向资金数据...")
            df = ak.stock_hsgt_hist_em(symbol="北向资金")
            if df is not None and len(df) > 0:
                # akshare 1.18: 列名变更, 映射到标准列名
                col_map = {
                    '日期': 'date', '当日成交净买额': 'net_flow',
                    '买入成交额': 'buy_amount', '卖出成交额': 'sell_amount',
                    '历史累计净买额': 'cumulative', '当日资金流入': 'inflow',
                    '持股市值': 'hold_value',
                }
                df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
                df['date'] = pd.to_datetime(df['date'])
                df.to_pickle(cache_path)
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

        主信号：5日累计净买额 / 20日均值 → tanh 标准化
        兜底信号（hist API 2024-08-19 起断数据）：
          - 领涨股-涨跌幅：北向当天买最多的股票涨了多少
          - 沪深300-涨跌幅：大盘方向辅助
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

        recent = df[mask].sort_values('date').tail(20)
        if len(recent) < 5:
            return 0.0

        flow_5d = recent['net_flow'].tail(5).fillna(0).sum()
        net_flow_valid = recent['net_flow'].dropna()
        if len(net_flow_valid) == 0:
            # hist API 无 net_flow → 用领涨股涨跌幅 + 沪深300涨跌幅替代
            return self._northbound_fallback_signal(recent)
        flow_avg = net_flow_valid.mean() + 0.01
        if abs(flow_avg) * 5 < 1e-6:
            return 0.0
        return float(np.tanh(flow_5d / max(abs(flow_avg) * 5, 1)))

    def _northbound_fallback_signal(self, recent: pd.DataFrame) -> float:
        """net_flow 缺失时的兜底信号，使用可用字段推导北向情绪。

        使用 recent 的最近20条记录，按以下优先级：
          1. 领涨股-涨跌幅: 北向当天买最多的股票涨跌幅 → 直接反映北向攻击性
          2. 沪深300-涨跌幅: 大盘走向 → 辅助确认

        每项独立 tanh 压缩到 [-1,1]，等权融合。
        """
        signals = []
        weights = []

        # 领涨股-涨跌幅：北向资金最偏好的股票的当日表现
        lead_col = next((c for c in recent.columns if '领涨' in str(c) and '跌幅' in str(c)), None)
        if lead_col:
            lead_chg = recent[lead_col].dropna()
            if len(lead_chg) >= 3:
                lead_mean = lead_chg.tail(5).mean()  # 近5日均值
                # 涨跌幅 0~10% → -10%~+10% 范围，tanh(chg/5) 将 +5% 压缩到 ~0.76
                signals.append(float(np.tanh(lead_mean / 5.0)))
                weights.append(0.6)

        # 沪深300-涨跌幅：大盘环境
        hs300_col = next((c for c in recent.columns if '沪深300' in str(c) and '跌幅' in str(c)), None)
        if hs300_col:
            hs300_chg = recent[hs300_col].dropna()
            if len(hs300_chg) >= 3:
                hs300_mean = hs300_chg.tail(5).mean()
                signals.append(float(np.tanh(hs300_mean / 3.0)))
                weights.append(0.4)

        if not signals:
            return 0.0

        total_w = sum(weights) or 1.0
        return float(np.clip(
            sum(s * w for s, w in zip(signals, weights)) / total_w,
            -1.0, 1.0
        ))

    # ========== 融资融券 ==========

    def load_margin(self) -> pd.DataFrame:
        """加载融资融券日度数据。Returns DataFrame with columns:
            date, margin_balance(融资余额/亿), short_balance(融券余额/亿),
            margin_change, margin_buy, total_balance
        """
        cache_path = self.data_dir / 'margin_daily.pkl'
        if cache_path.exists():
            self._margin = pd.read_pickle(cache_path)
            return self._margin

        try:
            import akshare as ak
            print("[AltData] 下载融资融券数据...")
            # akshare 1.18: 改用 stock_margin_sse (支持日期范围)
            df = ak.stock_margin_sse(start_date='20220101', end_date='20260530')
            if df is not None and len(df) > 0:
                col_map = {
                    '信用交易日期': 'date', '融资余额': 'margin_balance',
                    '融资买入额': 'margin_buy', '融券余量': 'short_balance',
                    '融券余量金额': 'short_amount',
                }
                df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
                df['date'] = pd.to_datetime(df['date'])
                df.to_pickle(cache_path)
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

        recent = df[mask].sort_values('date').tail(10)
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
