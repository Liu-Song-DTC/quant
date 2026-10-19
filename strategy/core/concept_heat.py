# core/concept_heat.py
"""
题材热度因子 — 基于东方财富概念板块数据 + 产业链传导

三层次信号:
1. 概念板块涨跌幅: stock_board_concept_name_em() 日更
2. 产业链传导: 上游涨→埋伏下游 (core/industry_chain.py)
3. 妖股信号: volume_surge + turnover_burst + limit_up_freq

回测: 使用 concept_hist.pkl 缓存的历史数据
实盘: 每日调用一次 stock_board_concept_name_em()
"""

import os
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd

from .industry_chain import compute_chain_signals, get_chain_lead_score, discover_chain_edges
from .cmap_hist import resolve_hist_path


def _cal_month_end(date) -> Optional[str]:
    """决策日 → **该日之前(含)最近的日历月末**, 返回 'YYYY-MM-DD'。

    ⚠ 必须是**日历**月末, 不是"数据里该月最后一个交易日": 后者在当月会随新数据
    推移(今天=10-19 ⇒ 十月末=10-19, 明天 ⇒ 10-20), 于是拟合窗口又依赖 T,
    等于把铁律违反换个地方重新引入。日历月末对所有 T 恒定。
    """
    try:
        ts = pd.Timestamp(date)
    except Exception:
        return None
    if pd.isna(ts):
        return None
    me = ts + pd.offsets.MonthEnd(0)
    if me > ts:
        me = me - pd.offsets.MonthEnd(1)
    return me.strftime('%Y-%m-%d')


class ConceptHeatCalculator:
    """题材热度计算器 — 支持回测和实盘双模式"""

    def __init__(self, map_path: str = None):
        if map_path is None:
            # 研究钩子: QUANT_CONCEPT_MAP 覆盖概念映射路径(默认空=生产路径, 逐位零改动)
            map_path = os.environ.get('QUANT_CONCEPT_MAP') or str(
                Path(__file__).parent.parent.parent / "data" / "stock_concept_map.pkl")

        self._map_path = map_path
        self._stock_concepts: Dict[str, List[str]] = {}
        self._concept_stocks: Dict[str, List[str]] = {}
        self._concept_scores: Dict[str, float] = {}
        self._chain_signals: Dict[str, float] = {}
        self._concept_hist: Dict[str, pd.DataFrame] = {}
        self._auto_edges: dict = {}  # 自动发现的产业链传导边 (仅 as-of 关闭时的全样本态)
        self._auto_edges_computed = False
        # 2026-10-19 (铁律 #350): 传导边改为**按决策日 as-of 拟合**, 见 _edges_asof。
        # key = 日历月末 'YYYY-MM-DD' → edges。单次拟合 0.266s, 全窗口 ~45 次 ≈ 12s。
        self._edges_cache: Dict[str, dict] = {}
        self._scores_cache: Dict[str, tuple] = {}  # 日期→(concept_scores, chain_signals)
        self._ema_concept_scores: Dict[str, float] = {}  # EMA平滑后的概念分数 (仅实盘增量模式)
        self._hist_ema: Dict[str, pd.Series] = {}  # 回测: 概念→截至每日的EMA序列(确定性)
        self._loaded = False
        # 2026-10-16: 概念映射**版本化** (as-of 成分股归属)
        # 前视根因: _concept_scores(概念涨跌幅)是 as-of 的, 但 _stock_concepts(哪只股票
        # 属于哪个概念)是**今天的静态名单** ⇒ 拿今天的成分股回测历史。
        # 实测 churn: 10/4→10/16 仅12天, 5656只共同股票中 26.3% 的归属集合变了
        # (成员 Jaccard 中位 1.000, 但每股12.7个概念里增删1个就翻转集合判定)
        # ⇒ concept_heat 每次刷新全史重写 ⇒ 回测曲线不可复现。
        # 修法: 按决策日 τ 取"截止 τ 的最近版本"的成分股名单, 与 concept_inception
        # 的成立日 gates 同构。**默认开启**(2026-10-17, 见 core/cmap_hist.py):
        # 此前是"研究钩子默认空=关闭", 但生产与回测必须是同一个东西。
        self._cmap_versions: Optional[List[tuple]] = None  # [(date_str, map_dict), ...] 升序
        self._cmap_vcache: Dict[str, tuple] = {}           # date_str → (stock_concepts, concept_stocks)
        self._cmap_ver_cache: Dict[str, str] = {}          # 决策日 → 版本号 (memo)
        self._cmap_cur: Optional[str] = None               # 当前已应用的版本日期

    def load(self, hist_path: str = None):
        """加载股票→概念映射 + 历史概念数据"""
        if self._loaded:
            return

        with open(self._map_path, 'rb') as f:
            self._stock_concepts = pickle.load(f)

        for code, concepts in self._stock_concepts.items():
            for c in concepts:
                self._concept_stocks.setdefault(c, []).append(code)

        # 2026-10-17: 版本化映射**默认开启** — 见 core/cmap_hist.py 的模块说明。
        # 此前是"研究钩子, 默认空=关闭", 但"设了 env 才 PIT 正确"本身就是铁律
        # 禁止的"两个东西", 且是静默失效的温床(参 live-regime-blind-bug)。
        # 显式关闭仅用于对照臂: QUANT_CONCEPT_MAP_HIST=off
        _hist_p = resolve_hist_path()
        if _hist_p:
            try:
                with open(_hist_p, 'rb') as f:
                    _h = pickle.load(f)
                vs = sorted((v for v in _h.get('versions', []) if v.get('map')),
                            key=lambda v: str(v['date']))
                if vs:
                    self._cmap_versions = [(str(v['date']), v['map']) for v in vs]
                    print(f"[ConceptHeat] 版本化映射已启用: {len(vs)} 版 "
                          f"({self._cmap_versions[0][0]} → {self._cmap_versions[-1][0]})")
            except Exception as _e:
                print(f"[ConceptHeat] ⚠ 版本化映射加载失败({str(_e)[:60]}), 退回静态映射")
                self._cmap_versions = None

        # 加载概念历史数据(回测用)
        if hist_path is None:
            hist_path = str(Path(__file__).parent.parent.parent / "data" / "concept_hist.pkl")
        try:
            with open(hist_path, 'rb') as f:
                self._concept_hist = pickle.load(f)
            print(f"[ConceptHeat] 已加载 {len(self._stock_concepts)} 股票映射 + {len(self._concept_hist)} 概念历史")
        except FileNotFoundError:
            print(f"[ConceptHeat] 已加载 {len(self._stock_concepts)} 股票映射 (无历史数据)")

        # 预计算每概念"截至每日"的EMA序列(回测确定性查询, 与增量公式逐位等价:
        #   e_t = e_{t-1}*decay + x_t*(1-decay), decay=0.5^(1/3), e_0=x_0)
        decay = 0.5 ** (1.0 / 3.0)
        for name, df in self._concept_hist.items():
            try:
                s = (df.drop_duplicates('date').set_index('date')['return']
                       .astype(float).sort_index())
                self._hist_ema[name] = s.ewm(alpha=1.0 - decay, adjust=False).mean()
            except Exception:
                pass

        # 2026-10-19 (铁律 #350): 此前这里调用 _compute_auto_edges() —— 用**当时全部**
        # concept_hist 拟合一次并锁死, 于是 todate=T 与 T+1 得到不同的边, 同一历史日的
        # chain_score 随 T 变。改为按决策日 as-of 拟合(见 _edges_asof), 此处不再预计算。
        self._loaded = True

    def _version_for(self, date) -> Optional[str]:
        """决策日 date 对应的版本号 ("截止该日的最近版本"; 早于最早版本 ⇒ 最早版本)。

        版本化映射关闭 ⇒ None。结果按日期串 memo —— _get_bom_score 每个
        (股票×bar) 都调一次(实测 ~7.4M 次), 不能每次构造 Timestamp。
        """
        if not self._cmap_versions:
            return None
        if isinstance(date, str):
            d = date[:10]
        else:
            try:
                d = date.strftime('%Y-%m-%d')       # datetime.date / pd.Timestamp
            except AttributeError:
                d = str(pd.Timestamp(date).date())
        hit = self._cmap_ver_cache.get(d)
        if hit is not None:
            return hit
        ver = self._cmap_versions[0][0]
        for vd, _ in self._cmap_versions:
            if vd <= d:
                ver = vd
            else:
                break
        self._cmap_ver_cache[d] = ver
        return ver

    def _version_payload(self, ver: str):
        """版本号 → (stock_concepts, concept_stocks); 构造结果按版本缓存"""
        cached = self._cmap_vcache.get(ver)
        if cached is None:
            m = next(mm for vd, mm in self._cmap_versions if vd == ver)
            cs: Dict[str, List[str]] = {}
            for code, concepts in m.items():
                for c in concepts:
                    cs.setdefault(c, []).append(code)
            cached = (m, cs)
            self._cmap_vcache[ver] = cached
        return cached

    def stock_concepts_asof(self, date):
        """纯查询: 决策日 date 的 (stock_concepts, concept_stocks)。**不改共享状态**。

        2026-10-18 新增。与 _apply_map_asof 的唯一区别就是"不改状态" —— 这一点是
        承重的:

        `_apply_map_asof` 由 set_daily_data(bar_date) 在**题材热度那个循环**里推进,
        所以 `self._stock_concepts` 的语义是"**最后一次 set_daily_data 那一版**",
        而不是"当前这个 bar 的版本"。任何在别处读它的代码拿到的都是残留版本。

        `signal_engine._get_bom_score` 正是这么读的 —— 它在**下一个循环**里按 bar
        日期取 BOM 分, 却让整段历史共用那个残留名单 ⇒
          (a) **前视**: 2021 年的 BOM 乘数用了 2026 年的概念名单;
          (b) **铁律违反**: todate 推进一天 ⇒ 该股最后一根 bar 的版本跳一版 ⇒
              整段历史的 BOM 乘数被重标。
        实测(#344 扰动测试, todate 10-15 vs 10-16): 11,696 格改写 / 28 只票,
        其中 11 只整段历史被**一个常数**缩放(ratio 变异系数 ~1e-4); 用两版名单
        分别算 compute_stock_bom_score 可复现该常数 ✓。
        """
        ver = self._version_for(date)
        if ver is None:
            return self._stock_concepts, self._concept_stocks
        return self._version_payload(ver)

    def _apply_map_asof(self, date):
        """按决策日 τ 切换到"截止 τ 的最近版本"成分股名单 (版本化映射关闭时=no-op)

        τ 早于最早版本 ⇒ 用最早版本(能拿到的最早名单, 恒定 ⇒ 可复现)。
        切换只换两个属性的**引用**(构造结果按版本缓存), 下游消费者
        (get_concept_heat / get_chain_lead_score)读的都是属性。

        ⚠ 只对"先 set_daily_data(τ) 再读"的消费者成立。**按 bar 日期取数、且与
        set_daily_data 不在同一个循环**的消费者必须改用 stock_concepts_asof(τ),
        否则拿到的是残留版本 —— 见该方法说明(2026-10-18 BOM 残留状态修复)。
        """
        ver = self._version_for(date)
        if ver is None or ver == self._cmap_cur:
            return
        self._stock_concepts, self._concept_stocks = self._version_payload(ver)
        self._cmap_cur = ver

    def _edges_asof(self, date):
        """决策日 τ → **截止 τ 的最近日历月末**拟合出的产业链传导边。

        2026-10-19 (铁律 #350)。原 `_compute_auto_edges` 在 load() 时用**当时全部**
        concept_hist 拟合一次并锁死 ⇒ 边依赖序列末尾 T。实测
        (`.arms/probe/autoedge_ironlaw_1019.py`, 复刻口径后按截断点拟合; 数字测于
        同日修复首行脏数据**之前**, 但"边随 T 变"这一性质与数据无关):
            T=2026-10-19: 30 边 / 12 个目标概念
            T=2025-12-31: 30 边 /  1 个目标概念   ← 12 条边被换掉
            T=2024-12-31: 30 边 /  1 个目标概念   ← 14 条边被换掉
            T=2023-12-29:  0 边
        ⇒ 11 个概念只在"进程跑到 2026-10-19"时才拿到那 +0.30 信号
        (chain_score = tanh(0.30×1.5) = 0.4219 ⇒ 贡献 0.127, 是概念涨幅项典型值
        0.035 的 3.6 倍, 且对这些股票**恒定**) ⇒ 更新今天的数据改写了历史选股。
        实测影响 (`.arms/probe/edgeimpact_1019.py`, 1,500 股 × 12 日抽样):
        改 as-of 后 **57.4%** 的 (股票,日期) 格 concept_heat 变, 改变量恒为
        ±0.1266 —— 旧态下 13 个目标概念覆盖 **70.4%** 的股票池, 全体吃同一份加成。

        修法: 每个**日历**月末用截至该月末的数据拟合一次, 决策日取 τ 之前最近的
        那个月末。⇒ 边只依赖 ≤ τ 的数据, 与 T 无关。
        ⚠ 必须用**日历**月末, 不能用"数据里该月最后一个交易日" —— 后者在当月会随
        新数据推移, 又变回依赖 T。
        ⚠ 起点受限: concept_hist 起于 2023-01-03, 首个可用拟合点 2023-01-31;
        更早的决策日返回 {} (与"边还不存在"一致)。
        成本: 单次拟合 0.266s, 全窗口 ~45 次 ≈ 12s, 按月末缓存。
        """
        if os.environ.get('QUANT_EDGE_ASOF', '').lower() in ('off', '0', 'false'):
            return self._full_sample_edges()          # 仅对照臂用
        key = _cal_month_end(date)
        if key is None:
            return {}
        hit = self._edges_cache.get(key)
        if hit is None:
            hit = self._fit_edges_upto(key)
            self._edges_cache[key] = hit
        return hit

    def _fit_edges_upto(self, cut: str) -> dict:
        """用 concept_hist 中 date <= cut 的行拟合传导边 (口径与旧实现逐字一致)。

        使用3日滚动收益替代日频：单日噪音太大导致相关性不显著，
        3日聚合后信噪比提升，同时保留足够的时序分辨率。
        """
        if len(self._concept_hist) < 10:
            return {}
        try:
            concept_3d_rets = {}
            for name, df in self._concept_hist.items():
                if 'return' not in df.columns or 'date' not in df.columns:
                    continue
                sub = df[df['date'] <= cut]
                if len(sub) < 20:
                    continue
                daily = sub['return'].dropna().values
                if len(daily) < 20:
                    continue
                # 3日滚动累计收益: r[t-2]+r[t-1]+r[t] 的累积
                rolling_3d = np.array([
                    np.prod(1 + daily[max(0, i-2):i+1]) - 1
                    for i in range(2, len(daily))
                ])
                if len(rolling_3d) >= 10:
                    concept_3d_rets[name] = rolling_3d
            if len(concept_3d_rets) < 10:
                return {}
            names = list(concept_3d_rets.keys())
            return discover_chain_edges(
                concept_3d_rets, names,
                min_correlation=0.35,  # 3日频阈值（日频0.55太高，周频0.30太低）
                max_lag=3,             # 3日频滞后1~3期 ≈ 3~9个自然日
                top_n=30,
            )
        except Exception:
            return {}

    def _full_sample_edges(self) -> dict:
        """旧行为(全样本拟合, 锁死) —— **仅供 QUANT_EDGE_ASOF=off 的对照臂**。"""
        if self._auto_edges_computed:
            return self._auto_edges
        self._auto_edges = self._fit_edges_upto('9999-12-31')
        self._auto_edges_computed = True
        total_edges = sum(len(v) for v in self._auto_edges.values())
        print(f"[ConceptHeat] ⚠ 对照臂: 全样本拟合 {total_edges} 条传导边 (非 PIT)")
        return self._auto_edges

    def set_daily_data(self, date, concept_returns: Dict[str, float] = None):
        """设置当日概念涨跌幅并计算产业链传导

        - 实盘模式: 优先从 data/concept_daily.csv 自动读取
        - 回测模式: 从 concept_hist 按日期查找
        - 手动模式: 传入 concept_returns
        """
        # 2026-10-16: 先按决策日切换成分股名单, 再走缓存(缓存存的是涨跌幅/链信号,
        # 与名单无关; 但名单必须在 get_concept_heat 之前就位)
        self._apply_map_asof(date)
        # 统一规范化日期键，确保所有路径使用一致缓存键
        cache_key = str(pd.Timestamp(date).date())

        if concept_returns is not None:
            # 实盘/手动模式: 传入当日全量概念涨跌幅, 进程内增量EMA平滑
            self._concept_scores = concept_returns
            self._apply_heat_ema()
            self._chain_signals = compute_chain_signals(
                self._concept_scores, auto_discovered_edges=self._edges_asof(date)
            )
            self._scores_cache[cache_key] = (self._concept_scores, self._chain_signals)
            return
        elif cache_key in self._scores_cache:
            self._concept_scores, self._chain_signals = self._scores_cache[cache_key]
            return
        elif self._hist_ema:
            # 回测模式: 从预计算EMA序列按日确定性取值
            # 不读concept_daily.csv(那是当日实盘数据, 回测引用=前视泄漏)
            # 不依赖全局增量状态(旧实现首日之后冻结/跨股票污染)
            date_ts = pd.Timestamp(date)
            self._concept_scores = {}
            for name, ema_s in self._hist_ema.items():
                v = ema_s.asof(date_ts)
                if pd.notna(v):
                    self._concept_scores[name] = float(v)
        else:
            # 实盘模式(无历史缓存): 尝试从当日概念数据CSV读取
            if not hasattr(self, '_daily_csv_path'):
                self._daily_csv_path = Path(__file__).parent.parent.parent / "data" / "concept_daily.csv"
                self._daily_csv_exists = self._daily_csv_path.exists()
            if self._daily_csv_exists:
                try:
                    df = pd.read_csv(self._daily_csv_path)
                    if '涨跌幅' in df.columns and '板块名称' in df.columns:
                        self._concept_scores = dict(zip(
                            df['板块名称'], df['涨跌幅'].astype(float) / 100.0))
                except Exception:
                    self._daily_csv_exists = False
            self._apply_heat_ema()

        # 计算产业链传导信号 (基于EMA平滑后的分数)
        if self._concept_scores:
            self._chain_signals = compute_chain_signals(
                self._concept_scores, auto_discovered_edges=self._edges_asof(date)
            )
        else:
            self._chain_signals = {}

        # 缓存结果供同日期其他股票复用
        # 性能(2026-10-09): 上限300→2000 — 回测单股1394个日期, 300上限下每股
        # 复算~1100个日期(asof×60概念/日, cProfile每股~2.2s×5585股≈3.4h总wall)。
        # 2000上限=全日期覆盖, 每worker每日期只算一次; 内存≈1334×(60浮点+链信号)≈3MB/worker。
        if len(self._scores_cache) > 2000:
            oldest = min(self._scores_cache.keys())
            del self._scores_cache[oldest]
        self._scores_cache[cache_key] = (self._concept_scores, self._chain_signals)

    def _apply_heat_ema(self):
        """对 _concept_scores 应用EMA衰减 (halflife=3天)，冷却过时的概念热度。"""
        decay = 0.5 ** (1.0 / 3.0)  # ~0.7937, 每天衰减, 3天半衰
        if self._ema_concept_scores:
            for k in list(self._ema_concept_scores.keys()):
                if k in self._concept_scores:
                    self._ema_concept_scores[k] = self._ema_concept_scores[k] * decay + self._concept_scores[k] * (1 - decay)
                else:
                    self._ema_concept_scores[k] *= decay  # 无新数据时持续衰减
            for k, v in self._concept_scores.items():
                if k not in self._ema_concept_scores:
                    self._ema_concept_scores[k] = v  # 新概念初始化
        else:
            self._ema_concept_scores = dict(self._concept_scores)
        # 用EMA平滑后的分数替换原始分数
        self._concept_scores = dict(self._ema_concept_scores)

    def get_concept_heat(self, code: str) -> float:
        """获取单只股票的题材热度 (概念涨幅 + 产业链埋伏)"""
        if not self._loaded:
            self.load()

        concepts = self._stock_concepts.get(code, [])
        if not concepts or not self._concept_scores:
            return 0.5

        # 维度1: 概念板块涨幅
        scores = []
        for c in concepts:
            ret = self._concept_scores.get(c)
            if ret is not None:
                scores.append(np.tanh(max(ret, 0) * 5.0))

        concept_score = float(np.clip(np.mean(scores), 0.0, 1.0)) if scores else 0.5

        # 维度2: 产业链埋伏加分
        chain_score = get_chain_lead_score(code, self._stock_concepts, self._chain_signals)

        # 融合: 70%概念涨幅 + 30%产业链埋伏
        return float(np.clip(0.7 * concept_score + 0.3 * chain_score, 0.0, 1.0))

    def get_all_concept_heat(self, codes: List[str], date=None) -> np.ndarray:
        """批量获取股票题材热度"""
        if not self._loaded:
            self.load()

        if date is not None:
            self.set_daily_data(date)

        result = np.full(len(codes), 0.5)
        if not self._concept_scores:
            return result

        for i, code in enumerate(codes):
            result[i] = self.get_concept_heat(code)
        return result

    def get_top_concepts(self, n: int = 30) -> List[str]:
        sorted_concepts = sorted(self._concept_scores.items(), key=lambda x: -x[1])
        return [c for c, _ in sorted_concepts[:n]]

    def get_chain_lead_concepts(self, min_score: float = 0.05) -> Dict[str, float]:
        """获取产业链埋伏信号中分数>threshold的概念"""
        return {k: v for k, v in sorted(self._chain_signals.items(),
                key=lambda x: -x[1]) if v > min_score}


# 全局单例
_calculator: Optional[ConceptHeatCalculator] = None

def get_calculator() -> ConceptHeatCalculator:
    global _calculator
    if _calculator is None:
        _calculator = ConceptHeatCalculator()
        _calculator.load()
    return _calculator


def compute_concept_heat_for_date(date, codes: List[str]):
    """为指定日期批量计算股票的题材热度（回测用）

    Args:
        date: 日期 (str or Timestamp)
        codes: 股票代码列表

    Returns:
        np.ndarray of heat scores 0~1
    """
    calc = get_calculator()
    return calc.get_all_concept_heat(codes, date=date)
