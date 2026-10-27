# core/cmap_hist.py
"""版本化概念映射路径解析 (2026-10-17)

**为什么需要这个模块**

概念层是全系统**唯一没有日期维度**的信号输入。它有两条消费路径, 都直接读
`data/stock_concept_map.pkl` —— 那是**今天**的成分股名单, 却被拿去回测全部历史:

  ① `core/concept_heat.py` 的 `_stock_concepts`(哪只股票属于哪个概念)
     ⇒ 决定 concept_heat 因子值
  ② `core/signal_engine.py` 的 `_get_specific_industry`(概念名 → 季度因子配置)
     ⇒ 决定该股套用哪套因子权重(比 ① 更一阶, 非 scale-invariant)

而 `stock_concept_map.pkl` **每天被 `refresh_all.sh` 的 7/9 步改写**。实测:

  · 2026-10-16 → 10-17(1 天): 930/5,659 只归属变 = **16.4%**
  · 2026-05-31 版 vs 今日版(4.5 月): **88.5%** 的股票归属不同, Jaccard 中位 0.80
    (概念名词汇相同 ⇒ 是真标签漂移, 不是格式差异)

⇒ 每天刷新一次就把**全部历史**重写一遍 ⇒ 直接违反铁律
   "更新今天的数据, 不应该改变历史"。

**修法**

按决策日 τ 取"截止 τ 的最近版本"的成分股名单, 数据源
`data/concept_map_history.pkl` —— 由 `data/refresh_concept_map_0903.py` 每次刷新
**追加一版**(`record_version` 拒绝覆盖历史日期, 只允许覆盖"今天"), 因此历史恒定。
τ 早于最早版本 ⇒ 固定使用首版作为历史代理；之后按时间线更新。
首版前的代理关系可复现，但不构成当时真实成员的证明，历史评估须标注覆盖边界。

**默认开启 (2026-10-17)**

此前它是"研究钩子, 默认空=关闭"。但生产与回测必须是**同一个东西** ——
"设了 env 才 PIT 正确"本身就是铁律禁止的"两个东西", 且是静默失效的温床
(参 live-regime-blind-bug: 查表失败返回默认 ⇒ 静默 bug)。

显式关闭仅用于对照臂: `QUANT_CONCEPT_MAP_HIST=off`(或 '' / none / 0 / false)。
"""
import os

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_HIST = os.path.join(PROJECT_DIR, 'data', 'concept_map_history.pkl')

_OFF = ('', 'off', 'none', '0', 'false')


def resolve_hist_path() -> str:
    """返回版本化概念映射路径; 只有显式关闭才允许静态快照。

    未设 env  ⇒ 用 DEFAULT_HIST(**默认开启**)。
    显式设为 _OFF 之一 ⇒ None。
    """
    v = os.environ.get('QUANT_CONCEPT_MAP_HIST')
    if v is None:
        v = DEFAULT_HIST
    if str(v).strip().lower() in _OFF:
        return None
    if not os.path.exists(v):
        raise FileNotFoundError(f'concept history missing; refusing static-map fallback: {v}')
    return v
