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

        # 信号级逐日memo (2026-09-01性能优化): 北向/两融市场级信号是date的纯函数,
        # 每股逐bar重复计算 → 按date_str缓存, 跨股票复用
        self._nb_sig_cache: dict = {}
        self._mg_sig_cache: dict = {}
        # 龙虎榜预解析缓存: (df, code_col, date_col, code_strs, dt_arr, buy_col, sell_col)
        # 原实现每次调用都对全表 astype+extract+to_datetime, 均为df的纯函数, 一次性缓存等价
        self._dt_parsed = None

        # 股东减持: 懒加载; _reduction_idx_*为get_reduction_codes的按日缓存
        self._reduction: pd.DataFrame = None
        self._reduction_idx: set = None
        self._reduction_idx_date = None
        self._reduction_idx_days = None

        # 限售解禁: 懒加载; _unlock_idx_key为get_unlock_codes的按日缓存
        self._unlock: pd.DataFrame = None
        self._unlock_idx: set = None
        self._unlock_idx_key = None

        # 业绩预告: 懒加载(超预期事件研究/因子)
        self._yjyg: pd.DataFrame = None

        # 两融个股明细特征frame: 懒加载(build一次, ML每次复放重建provider会重建)
        self._margin_feat: pd.DataFrame = None

        # 减持计划公告(start/end事件): 懒加载+过期自动增量
        # 减持计划公告缓存(计划层过滤, 2026-08-27)
        self._reduction_plans: pd.DataFrame = None
        self._plan_codes_cache: dict = {}

    # ========== 龙虎榜 ==========

    def load_dragon_tiger(self) -> pd.DataFrame:
        """加载龙虎榜数据缓存。若无缓存则尝试下载。

        Returns DataFrame with columns:
            code, date, buy_amount, sell_amount, net_amount,
            institution_buy, institution_sell, top_institution_count
        """
        cache_path = self.data_dir / 'dragon_tiger.pkl'
        if cache_path.exists():
            _age_hours = (pd.Timestamp.now() - pd.Timestamp.fromtimestamp(cache_path.stat().st_mtime)).total_seconds() / 3600
            if _age_hours < 24:
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
            import calendar
            import re
            _now = pd.Timestamp.now()
            print(f"[AltData] 下载龙虎榜历史明细 (2024-01 ~ {_now:%Y-%m})...")
            all_records = []
            for year in [2024, 2025, _now.year]:
                for month in range(1, 13):
                    if year == _now.year and month > _now.month:
                        break
                    _start = f'{year}{month:02d}01'
                    _end = f'{year}{month:02d}{calendar.monthrange(year, month)[1]}'
                    try:
                        # akshare 1.18: stock_lhb_detail_daily_em 已移除, 改用区间版
                        df = ak.stock_lhb_detail_em(start_date=_start, end_date=_end)
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

        # 预解析 (2026-09-01性能优化): 代码提取/日期解析/买卖列名均为df的纯函数,
        # 一次性缓存; 每次调用仅剩 numpy mask + 小子集 sort_values (与原实现逐位等价)
        parsed = self._dt_parsed
        if parsed is None or parsed[0] is not df:
            code_col = next((c for c in df.columns if '代码' in str(c) or c == 'code'), None)
            date_col = next((c for c in df.columns if '日期' in str(c) or '上榜' in str(c) or c == 'date'), None)
            if code_col is None or date_col is None:
                parsed = self._dt_parsed = (df, None, None, None, None, None, None)
            else:
                code_strs = df[code_col].astype(str).str.extract(r'(\d{6})', expand=False)
                try:
                    dt_arr = pd.to_datetime(df[date_col]).to_numpy()
                except Exception:
                    print(f"[AltData] 龙虎榜日期解析失败，date_col={date_col}")
                    dt_arr = None
                buy_col = next((c for c in df.columns if '买入' in str(c)), None)
                sell_col = next((c for c in df.columns if '卖出' in str(c)), None)
                parsed = self._dt_parsed = (df, code_col, date_col, code_strs, dt_arr, buy_col, sell_col)
        _, code_col, date_col, code_strs, dt_arr, buy_col, sell_col = parsed
        if code_col is None or date_col is None:
            return 0.0

        code_mask = code_strs.to_numpy() == code[:6]  # NaN → False, 与原Series比较一致
        # 日期过滤：仅使用 <= 查询日期的记录，防止前视偏差
        if dt_arr is not None:
            date_mask = dt_arr <= pd.to_datetime(date_str)
        else:
            date_mask = np.ones(len(df), dtype=bool)
        mask = code_mask & date_mask
        recent = df[mask].sort_values(date_col) if date_col and not df[mask].empty else df[mask]
        if recent.empty:
            return 0.0

        # 按日期排序后取最近记录的机构净买入占比
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
            _age_hours = (pd.Timestamp.now() - pd.Timestamp.fromtimestamp(cache_path.stat().st_mtime)).total_seconds() / 3600
            if _age_hours < 24:
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
        # 逐日memo (2026-09-01性能优化): 信号是date的纯函数, 每股逐bar重复计算
        if date_str in self._nb_sig_cache:
            return self._nb_sig_cache[date_str]
        result = self._nb_signal_impl(df, date_str)
        self._nb_sig_cache[date_str] = result
        return result

    def _nb_signal_impl(self, df, date_str) -> float:
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
            _age_hours = (pd.Timestamp.now() - pd.Timestamp.fromtimestamp(cache_path.stat().st_mtime)).total_seconds() / 3600
            if _age_hours < 24:
                self._margin = pd.read_pickle(cache_path)
                return self._margin

        try:
            import akshare as ak
            print("[AltData] 下载融资融券数据...")
            # akshare 1.18: 改用 stock_margin_sse (支持日期范围)
            # 2026-08-23修复: end_date曾硬编码'20260530'导致两融数据永远停在5/29
            df = ak.stock_margin_sse(start_date='20220101',
                                     end_date=pd.Timestamp.now().strftime('%Y%m%d'))
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
        # 逐日memo (2026-09-01性能优化): 信号是date的纯函数, 每股逐bar重复计算
        if date_str in self._mg_sig_cache:
            return self._mg_sig_cache[date_str]
        result = self._mg_signal_impl(df, date_str)
        self._mg_sig_cache[date_str] = result
        return result

    def _mg_signal_impl(self, df, date_str) -> float:
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


    # ========== 股东减持 ==========

    def _fetch_reduction_em(self, since: str = None) -> pd.DataFrame:
        """东财datacenter: RPT_SHARE_HOLDER_INCREASE, DIRECTION=减持.
        since='YYYY-MM-DD'时只拉公告时间>=since的增量记录."""
        import requests
        url = 'https://datacenter-web.eastmoney.com/api/data/v1/get'
        flt = '(DIRECTION="减持")'
        if since:
            flt += f"(EITIME>='{since}')"
        params = {
            'sortColumns': 'EITIME', 'sortTypes': '-1',
            'pageSize': '500', 'pageNumber': '1',
            'reportName': 'RPT_SHARE_HOLDER_INCREASE',
            'columns': 'ALL', 'source': 'WEB', 'client': 'WEB',
            'filter': flt,
        }
        frames = []
        page = 1
        while True:
            params['pageNumber'] = str(page)
            r = requests.get(url, params=params, timeout=30,
                             headers={'User-Agent': 'Mozilla/5.0'})
            j = r.json()
            res = j.get('result') or {}
            data = res.get('data') or []
            if not data:
                break
            frames.append(pd.DataFrame(data))
            if page >= (res.get('pages') or 1):
                break
            page += 1
            if page > 2000:
                break
        if not frames:
            return pd.DataFrame()
        raw = pd.concat(frames, ignore_index=True)
        df = pd.DataFrame({
            'code': raw['SECURITY_CODE'].astype(str).str.zfill(6),
            'name': raw.get('SECURITY_NAME_ABBR', ''),
            'holder': raw.get('HOLDER_NAME', ''),
            'change_num': pd.to_numeric(raw.get('CHANGE_NUM'), errors='coerce'),
            'change_ratio': pd.to_numeric(raw.get('CHANGE_RATIO'), errors='coerce'),
            'start_date': pd.to_datetime(raw['START_DATE'], errors='coerce'),
            'end_date': pd.to_datetime(raw['END_DATE'], errors='coerce'),
            'eitime': pd.to_datetime(raw['EITIME'], errors='coerce'),
        })
        return df.dropna(subset=['code', 'eitime'])

    def load_reduction(self) -> pd.DataFrame:
        """加载股东减持记录(全历史, 增量更新).
        Returns DataFrame: code, name, holder, change_num(万股), change_ratio(占总股本),
                           start_date(变动开始), end_date(变动截止), eitime(公告时刻)"""
        cache_path = self.data_dir / 'reduction_records.pkl'
        old = None
        if cache_path.exists():
            if self._reduction is not None:
                return self._reduction
            old = pd.read_pickle(cache_path)
            _age_h = (pd.Timestamp.now() - pd.Timestamp.fromtimestamp(cache_path.stat().st_mtime)).total_seconds() / 3600
            if _age_h < 24:
                self._reduction = old
                return old
        try:
            since = None
            if old is not None and len(old):
                since = (old['eitime'].max() - pd.Timedelta(days=2)).strftime('%Y-%m-%d')
            print(f"[AltData] 下载股东减持数据{'(增量>=' + since + ')' if since else '(全量)'}...")
            new = self._fetch_reduction_em(since)
            if old is None:
                df = new
            elif len(new):
                df = pd.concat([old, new], ignore_index=True)
                df = df.drop_duplicates(subset=['code', 'holder', 'start_date', 'end_date', 'eitime'])
            else:
                df = old
            if df is not None and len(df):
                df = df.sort_values('eitime').reset_index(drop=True)
                df.to_pickle(cache_path)
            self._reduction = df
            print(f"[AltData] 股东减持: {len(df)} 条 -> {cache_path}")
            return df if df is not None else pd.DataFrame()
        except Exception as e:
            print(f"[AltData] 股东减持下载失败: {e}")
            if old is not None:
                self._reduction = old
                return old
            return pd.DataFrame()

    def get_reduction_codes(self, date, recent_days: int = 30) -> set:
        """date时点处于减持期的股票代码集合(时点安全: 仅用公告时刻<=date的记录).
        规则(满足其一):
          1) 减持公告披露于date前recent_days天内(披露滞后于实际窗口, 计划常持续数月)
          2) date落在[变动开始日, 变动截止日]窗口内(多批次减持进行中)"""
        if self._reduction is None:
            self._reduction = self.load_reduction()
        df = self._reduction
        if df is None or len(df) == 0:
            return set()
        t = pd.Timestamp(str(date)[:10])
        if self._reduction_idx_date != t or self._reduction_idx_days != recent_days:
            eit = df['eitime'].dt.normalize()
            known = eit <= t
            recent = known & (eit >= t - pd.Timedelta(days=recent_days))
            inwin = known & (df['start_date'] <= t) & (df['end_date'] >= t)
            self._reduction_idx = set(df.loc[recent | inwin, 'code'])
            self._reduction_idx_date = t
            self._reduction_idx_days = recent_days
        return self._reduction_idx


    # ========== 限售解禁 ==========

    def _fetch_unlock_em(self, start: str, end: str) -> pd.DataFrame:
        """东财解禁明细: start/end='YYYYMMDD'."""
        import akshare as ak
        raw = ak.stock_restricted_release_detail_em(start_date=start, end_date=end)
        if raw is None or len(raw) == 0:
            return pd.DataFrame()
        df = pd.DataFrame({
            'code': raw['股票代码'].astype(str).str.zfill(6),
            'unlock_date': pd.to_datetime(raw['解禁时间'], errors='coerce'),
            'ratio': pd.to_numeric(raw['占解禁前流通市值比例'], errors='coerce'),
            'shares': pd.to_numeric(raw['实际解禁数量'], errors='coerce'),
            'market_value': pd.to_numeric(raw['实际解禁市值'], errors='coerce'),
            'type': raw.get('限售股类型', ''),
        })
        return df.dropna(subset=['code', 'unlock_date'])

    def load_unlock(self) -> pd.DataFrame:
        """加载限售解禁日程(2021至今+已公告的未来日程).
        Returns DataFrame: code, unlock_date, ratio(占解禁前流通市值比例), shares, market_value, type"""
        cache_path = self.data_dir / 'unlock_schedule.pkl'
        if self._unlock is not None:
            return self._unlock
        if cache_path.exists():
            _age_h = (pd.Timestamp.now() - pd.Timestamp.fromtimestamp(cache_path.stat().st_mtime)).total_seconds() / 3600
            if _age_h < 24:
                self._unlock = pd.read_pickle(cache_path)
                return self._unlock
        try:
            # 按月分段拉取(接口限制日期范围), 覆盖回测期+未来3个月日程
            frames = []
            cur = pd.Timestamp('2021-01-01')
            end = pd.Timestamp.now() + pd.DateOffset(months=4)
            while cur < end:
                nxt = min(cur + pd.DateOffset(months=3), end)
                frames.append(self._fetch_unlock_em(cur.strftime('%Y%m%d'), nxt.strftime('%Y%m%d')))
                cur = nxt
            df = pd.concat([f for f in frames if len(f)], ignore_index=True)
            df = df.drop_duplicates(subset=['code', 'unlock_date', 'type', 'shares'])
            df = df.sort_values('unlock_date').reset_index(drop=True)
            df.to_pickle(cache_path)
            self._unlock = df
            print(f"[AltData] 限售解禁: {len(df)} 条 ({df['unlock_date'].min().date()}~{df['unlock_date'].max().date()}) -> {cache_path}")
            return df
        except Exception as e:
            print(f"[AltData] 限售解禁下载失败: {e}")
            self._unlock = pd.DataFrame()
            return self._unlock

    def get_unlock_codes(self, date, ahead_days: int = 30, min_ratio: float = 0.05) -> set:
        """date时点'解禁临近'的股票集合: 未来ahead_days天内存在解禁且规模>=min_ratio(占流通市值比例).
        解禁日程在发行/增发时即确定, 提前数月可知, 前视风险低."""
        if self._unlock is None:
            self._unlock = self.load_unlock()
        df = self._unlock
        if df is None or len(df) == 0:
            return set()
        t = pd.Timestamp(str(date)[:10])
        key = (str(date)[:10], ahead_days, min_ratio)
        if self._unlock_idx_key != key:
            m = (df['unlock_date'] >= t) & (df['unlock_date'] <= t + pd.Timedelta(days=ahead_days)) \
                & (df['ratio'] >= min_ratio)
            self._unlock_idx = set(df.loc[m, 'code'])
            self._unlock_idx_key = key
        return self._unlock_idx


    # ========== 业绩预告 ==========

    @staticmethod
    def _yjyg_periods(start='2008-12-31', end=None):
        """全部季度末报告期列表(YYYYMMDD), 对齐真实季度末(0331/0630/0930/1231)."""
        end = end or pd.Timestamp.now()
        out, cur = [], pd.Timestamp(start) + pd.offsets.QuarterEnd(0)
        while cur <= end:
            out.append(cur.strftime('%Y%m%d'))
            cur = cur + pd.DateOffset(months=3) + pd.offsets.QuarterEnd(0)
        return out

    def _fetch_yjyg_em(self, period: str) -> pd.DataFrame:
        """东财业绩预告: period=报告期('YYYYMMDD'). 全历史自20081231.
        时点安全字段=公告日期; 预测数值为区间中值."""
        import akshare as ak
        raw = ak.stock_yjyg_em(date=period)
        if raw is None or len(raw) == 0:
            return pd.DataFrame()
        df = pd.DataFrame({
            'code': raw['股票代码'].astype(str).str.zfill(6),
            'name': raw.get('股票简称', ''),
            'indicator': raw['预测指标'].astype(str),
            'forecast_type': raw['预告类型'].astype(str),
            'change_pct': pd.to_numeric(raw.get('业绩变动幅度'), errors='coerce'),
            'predict_value': pd.to_numeric(raw.get('预测数值'), errors='coerce'),
            'base_value': pd.to_numeric(raw.get('上年同期值'), errors='coerce'),
            'notice_date': pd.to_datetime(raw['公告日期'], errors='coerce'),
            'report_period': pd.Timestamp(period),
        })
        return df.dropna(subset=['code', 'notice_date'])

    def load_yjyg(self) -> pd.DataFrame:
        """加载业绩预告事件表(全历史2008至今, 增量刷新近3个季度).
        Returns DataFrame: code, name, indicator, forecast_type, change_pct(%),
                           predict_value(区间中值), base_value(上年同期), notice_date, report_period"""
        cache_path = self.data_dir / 'yjyg_records.pkl'
        old = None
        if cache_path.exists():
            if self._yjyg is not None:
                return self._yjyg
            old = pd.read_pickle(cache_path)
            _age_h = (pd.Timestamp.now() - pd.Timestamp.fromtimestamp(cache_path.stat().st_mtime)).total_seconds() / 3600
            if _age_h < 24:
                self._yjyg = old
                return old
        try:
            import time
            # 增量: 预告最迟在报告期后~4.5个月内公告, 刷新近200天内的报告期即可
            if old is not None and len(old):
                cut = pd.Timestamp.now() - pd.Timedelta(days=200)
                periods = [p for p in self._yjyg_periods()
                           if pd.Timestamp(p) >= max(cut, old['report_period'].min())]
            else:
                periods = self._yjyg_periods()
            frames = []
            for i, p in enumerate(periods):
                try:
                    frames.append(self._fetch_yjyg_em(p))
                    time.sleep(1)
                except Exception as e:
                    print(f"[AltData] 业绩预告 {p} 拉取失败: {e}")
            df = pd.concat([f for f in frames if len(f)], ignore_index=True) if frames else pd.DataFrame()
            if old is not None and len(old):
                df = pd.concat([old, df], ignore_index=True)
            if df is None or len(df) == 0:
                self._yjyg = old if old is not None else pd.DataFrame()
                return self._yjyg
            df = df.drop_duplicates(subset=['code', 'report_period', 'indicator', 'notice_date'])
            df = df.sort_values('notice_date').reset_index(drop=True)
            df.to_pickle(cache_path)
            self._yjyg = df
            print(f"[AltData] 业绩预告: {len(df)} 条 ({df['notice_date'].min().date()}~{df['notice_date'].max().date()}) -> {cache_path}")
            return df
        except Exception as e:
            print(f"[AltData] 业绩预告下载失败: {e}")
            if old is not None:
                self._yjyg = old
                return old
            return pd.DataFrame()

    # ========== 两融个股明细(ML截面特征) ==========
    # 数据由 data/margin_detail_backfill.py 回填/增量维护(每日收盘后披露, T-1滞后使用)

    def load_margin_detail(self) -> pd.DataFrame:
        """加载个股融资融券明细(全标的日度, 2021至今). 不存在时返回空框."""
        cache_path = self.data_dir / 'margin_detail.pkl'
        if not cache_path.exists():
            print(f"[AltData] 两融明细缺失: {cache_path}")
            return pd.DataFrame()
        df = pd.read_pickle(cache_path)
        if len(df) == 0:
            return df
        df['date'] = df['date'].astype(str)
        df['code'] = df['code'].astype(str).str.zfill(6)
        for c in ('rzye', 'rzmre', 'rqyl'):
            df[c] = pd.to_numeric(df[c], errors='coerce')
        return df

    def get_margin_feature_frame(self) -> pd.DataFrame:
        """两融截面特征frame: (date, code, rz_chg5, rz_chg20, rz_buy_ratio, rqyl_chg5).

        融资余额变化率=杠杆资金拥挤度/投机需求; 融券余量变化=看空压力.
        调用方负责T-1滞后(merge_asof allow_exact_matches=False);
        非两融标的不在此frame中(join后为NaN, XGBoost原生处理).
        """
        if getattr(self, '_margin_feat', None) is not None:
            return self._margin_feat
        df = self.load_margin_detail()
        if df is None or len(df) == 0:
            self._margin_feat = pd.DataFrame()
            return self._margin_feat
        df = df.dropna(subset=['rzye']).sort_values(['code', 'date'])
        g = df.groupby('code', sort=False)
        feat = pd.DataFrame({
            'date': df['date'].values,
            'code': df['code'].values,
            'rz_chg5': g['rzye'].pct_change(5).values,
            'rz_chg20': g['rzye'].pct_change(20).values,
            'rz_buy_ratio': (df['rzmre'] / df['rzye']).values,
            'rqyl_chg5': g['rqyl'].pct_change(5).values,
        })
        # pct_change对0余额产生inf; 极值截断到±3
        num_cols = ['rz_chg5', 'rz_chg20', 'rz_buy_ratio', 'rqyl_chg5']
        feat[num_cols] = feat[num_cols].replace([np.inf, -np.inf], np.nan).clip(-3, 3).astype('float32')
        feat = feat.sort_values('date')
        self._margin_feat = feat
        print(f"[AltData] 两融特征frame: {len(feat)} 行, {feat['code'].nunique()} 标的, "
              f"{feat['date'].min()}~{feat['date'].max()}")
        return feat

    # ========== 减持计划公告(计划层过滤, 2026-08-27) ==========
    # 背景(301257事件): 增减持明细只记录已执行减持, 预披露的减持计划不进库,
    # 刚预披露的标的仍会被选入. 计划层靠公告标题状态机补齐:
    # start(预披露/拟减持) -> end(实施完成/期限届满/结果公告/实施完毕/终止).
    # 全量数据由 data/reduction_plan_backfill.py 回填(2021至今); 此处懒加载+自动增量.

    def _fetch_reduction_plans_recent(self, months: int = 2) -> pd.DataFrame:
        """拉取近N个月持股变动公告中的减持类公告(东财公告大全), 供增量补齐."""
        import time as _time
        import calendar
        from akshare.stock_fundamental.stock_notice import _stock_notice_report
        now = pd.Timestamp.now()
        ym_list = []
        y, m = now.year, now.month
        for _ in range(months):
            ym_list.append((y, m))
            m -= 1
            if m < 1:
                y, m = y - 1, 12
        rows = []
        for y, m in reversed(ym_list):
            last = calendar.monthrange(y, m)[1]
            b, e = f'{y:04d}-{m:02d}-01', f'{y:04d}-{m:02d}-{last:02d}'
            try:
                raw = _stock_notice_report(symbol='持股变动', begin_date=b, end_date=e)
            except Exception as ex:
                print(f"[AltData] 减持计划增量拉取{b[:7]}失败: {str(ex)[:60]}")
                continue
            _time.sleep(0.8)
            if raw is None or len(raw) == 0:
                continue
            jc = raw[raw['公告标题'].astype(str).str.contains('减持')]
            if len(jc):
                rows.append(pd.DataFrame({
                    'ann_date': jc['公告日期'].astype(str).str[:10],
                    'code': jc['代码'].astype(str).str.zfill(6),
                    'name': jc['名称'].astype(str),
                    'title': jc['公告标题'].astype(str),
                }))
        if not rows:
            return pd.DataFrame()
        df = pd.concat(rows, ignore_index=True)
        end_keys = ('实施完成', '期限届满', '结果公告', '实施完毕', '终止')

        def _kind(t):
            t = str(t)
            if '预披露' in t or '拟减持' in t or '拟计划减持' in t:
                return 'start'
            if any(k in t for k in end_keys):
                return 'end'
            return 'other'

        df['kind'] = df['title'].map(_kind)
        return df

    def load_reduction_plans(self) -> pd.DataFrame:
        """减持计划公告frame(懒加载+自动增量). 列: ann_date/code/name/title/kind.

        增量触发: 最后公告日<今天(落后), 或pkl超过4小时未刷新(当天日间新发公告,
        18:00实盘前会重拉一次; 拉取失败fail-open用已有数据).
        """
        if self._reduction_plans is not None:
            return self._reduction_plans
        pkl = self.data_dir / 'reduction_plans.pkl'
        df = pd.read_pickle(pkl) if pkl.exists() else pd.DataFrame()
        today = str(pd.Timestamp.now().date())
        last = df['ann_date'].max() if len(df) else None
        pkl_age_h = (pd.Timestamp.now().timestamp() - pkl.stat().st_mtime) / 3600 if pkl.exists() else 99
        if last is None or last < today or pkl_age_h > 4:
            try:
                fresh = self._fetch_reduction_plans_recent()
                if len(fresh):
                    df = fresh if not len(df) else pd.concat([df, fresh], ignore_index=True)
                    df = df.drop_duplicates(subset=['code', 'ann_date', 'title']).reset_index(drop=True)
                if len(df):
                    df.to_pickle(pkl)
                new_last = df['ann_date'].max() if len(df) else 'NA'
                print(f"[AltData] 减持计划已增量更新: {len(df)} 条, 最新公告 {new_last}")
            except Exception as e:
                print(f"[AltData] 减持计划增量失败(用已有数据): {e}")
        else:
            print(f"[AltData] 减持计划: {len(df)} 条, 最新公告 {last}")
        self._reduction_plans = df
        return df

    def get_reduction_plan_codes(self, date) -> set:
        """当日处于减持计划期内的股票集合(计划层).

        状态机: 取ann_date<=date的最近一条start/end事件; 该事件为start且
        距date<=210天 => 计划期内(预披露后未见到完成/届满公告).
        210天兜底: 一般减持计划期限<=6个月, 无end公告的start不会永久污染.
        """
        d = str(date)[:10]
        if d in self._plan_codes_cache:
            return self._plan_codes_cache[d]
        df = self.load_reduction_plans()
        codes = set()
        if len(df):
            ev = df[df['kind'].isin(['start', 'end']) & (df['ann_date'] <= d)]
            ev = ev.sort_values('ann_date').groupby('code', sort=False).tail(1)
            active = ev[ev['kind'] == 'start']
            if len(active):
                age = (pd.Timestamp(d) - pd.to_datetime(active['ann_date'])).dt.days
                codes = set(active.loc[age <= 210, 'code'])
        self._plan_codes_cache[d] = codes
        return codes


# ========== 单例 ==========
_provider: AlternativeDataProvider = None

def get_provider() -> AlternativeDataProvider:
    global _provider
    if _provider is None:
        _provider = AlternativeDataProvider()
    return _provider
