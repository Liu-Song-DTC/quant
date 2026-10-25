"""ST 状态时间线 — 纯 PIT 版 (2026-10-17)

## 为什么需要它

`get_st_timeline` 原来从财务 CSV 的 **`股票简称` 列**推 ST 状态。但那一列**不是 PIT 值** ——
它是**文件构建时**的名称快照, 被写进该股**全部历史行**。后果:

  · 107 只"当前是ST"的股票, 其**整段历史**(含未ST的年份)都被 `tradable` 排除
  · 反向: 历史上是ST、现已摘帽的股票, 名字干净 ⇒ **从未被排除** ⇒ 在其ST年份里被交易
  · `f(t;T) ≠ f(t;t)`: 全量重建(或将来某天 ST 状态变化)会改写全部历史的 ST 判定

证据: 5585 只中 5344 单段 / 241 双段, **全部 241 个边界都落在报告期 20251231**
(= `refresh_fundamental_recent.py` 的 QUARTERS[0])。`refresh_fundamental_recent.py`
文件头注释自己就写着 "2026-08-31 ST漏判修复: ST判定依据的'股票简称'列陈旧" ——
8/31 打过一次补丁, 但只重刷最近 3 季 ⇒ 只修了"最近变成ST"那一类。

## 修法: 从裸价挖, 不依赖任何快照

`raw_data/{code}/none.csv`(裸价)**不重述** ⇒ 是唯一 PIT 安全的价格输入。主板 ST 限 ±5%,
非ST ±10% ⇒ 两条硬证据:

  · `|ret| > 5.5%`  ⇒ **证明**当天不是 5% 板 ⇒ 非ST (除新股首日/复牌/退市整理期)
  · `收盘 == round(昨收 × 1.05, 2)` (或 ×0.95) ⇒ 证明当天是 5% 板 ⇒ ST

**因果状态机(trailing-only)**: 连续 K 次 5% 板 ⇒ ST; 遇到 >5.5% 波动 ⇒ 解除; 否则保持。
只看过去 ⇒ `f(t;T) ≡ f(t;t)` 成立。

## 校准 (3312 只主板, K=2)

prevalence **3.08%** 股票-日(真实量级); 对照组 `000001 平安银行` / `600519 茅台` /
`000004`(指数数据) **全部 0 假阳性**; `600745 *ST闻泰` ST 2026-05-07→07-09、
`000010 *ST美丽` 2019-04-29→2026-07-15、`002199 东晶电子` 2016-08-12→2026-07-03。
K=1 假阳性 93.7% 不可用(单次巧合命中 ~0.05%/日)。

## 覆盖边界

**2042 只创业板/科创板不可检测** —— 它们的 ST 与常态同为 20% 板, 裸价无判别力。
这些股票**返回 None**, 由调用方回退到旧逻辑(名字快照)。这是已知的、无法用价格修复的
残留前视, 见记忆 st-snapshot-lookahead-20261017。
"""
from __future__ import annotations

import os
import pickle

import numpy as np

# 主板: ST=±5%, 非ST=±10% ⇒ 可判别。创业板30x/科创板68x 同为20% ⇒ 不可判别。
MAIN_BOARD = ('600', '601', '603', '605', '000', '001', '002', '003')

# 连续 K 次 5% 板命中 ⇒ 判 ST。K=1 假阳性率 93.7%(单次巧合 ~0.05%/日), K=2 → 0 假阳性。
CONFIRM_K = 2
# 超过此幅度 ⇒ 证明当天不是 5% 板 ⇒ 硬证据"非ST"。留 0.5pp 容差覆盖 2 位小数舍入。
NEG_THRESHOLD = 0.055


def is_detectable(code: str) -> bool:
    """该股能否用裸价判别 ST(仅主板)。"""
    return len(code) == 6 and code[:3] in MAIN_BOARD


def build_st_flags(close: np.ndarray) -> np.ndarray:
    """因果状态机: 由按日期升序的裸收盘价序列算出逐日 ST 标记。

    **只读 t 及之前的数据** ⇒ 对任意 T>t, 在 T 时刻重算 t 的值逐位相同。
    """
    close = np.asarray(close, dtype=float)
    n = len(close)
    st = np.zeros(n, dtype=bool)
    if n < 2:
        return st

    prev = close[:-1]
    cur = close[1:]
    with np.errstate(invalid='ignore', divide='ignore'):
        # 昨收为 0/NaN 的格子不构成证据(停牌、缺数据)
        valid = np.isfinite(prev) & np.isfinite(cur) & (prev > 0)
        ret = np.where(valid, cur / np.where(prev > 0, prev, 1.0) - 1.0, 0.0)
        neg = valid & (np.abs(ret) > NEG_THRESHOLD)
        # 涨跌停价恒等式: 交易所按 2 位小数四舍五入
        pos = valid & ((cur == np.round(prev * 1.05, 2)) | (cur == np.round(prev * 0.95, 2)))

    run = 0
    state = False
    for t in range(1, n):
        if neg[t - 1]:
            state = False
            run = 0
        elif pos[t - 1]:
            run += 1
            if run >= CONFIRM_K:
                state = True
        else:
            run = 0
        st[t] = state
    return st


def build_all(raw_root: str, codes=None, verbose: bool = True) -> dict:
    """扫描 raw_data, 返回 {code: (dates: np.ndarray[str], st: np.ndarray[bool])}。

    只包含**可检测(主板)**且成功读到的股票; 其余不在字典里 ⇒ 调用方回退。
    """
    import pandas as pd

    if codes is None:
        codes = sorted(os.listdir(raw_root))
    out = {}
    total = 0
    for code in codes:
        if not (len(code) == 6 and code.isdigit()) or not is_detectable(code):
            continue
        fp = os.path.join(raw_root, code, 'none.csv')
        if not os.path.exists(fp):
            continue
        try:
            d = pd.read_csv(fp, usecols=['日期', '收盘'], dtype={'日期': str})
        except Exception:
            continue
        d = d.dropna(subset=['收盘'])
        if len(d) < 50:
            continue
        d = d.sort_values('日期').reset_index(drop=True)
        dates = d['日期'].to_numpy()
        close = d['收盘'].to_numpy(dtype=float)
        out[code] = (dates, build_st_flags(close))
        total += int(out[code][1].sum())
        if verbose and len(out) % 1000 == 0:
            print(f'  st_timeline: {len(out)} 只', flush=True)
    if verbose:
        n_days = sum(len(v[0]) for v in out.values())
        print(f'st_timeline: {len(out)} 只主板股, ST股票-日 {total:,} / {n_days:,} '
              f'= {100 * total / max(n_days, 1):.2f}%', flush=True)
    return out


def default_path() -> str:
    return os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), 'data', 'stock_data', 'st_timeline.pkl')


def save(tl: dict, path: str | None = None) -> str:
    path = path or default_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'wb') as f:
        pickle.dump(tl, f, protocol=4)
    os.replace(tmp, path)
    return path


_LOADED = None


def load(path: str | None = None):
    """惰性加载; 文件不存在 ⇒ None(调用方回退到旧逻辑)。"""
    global _LOADED
    if _LOADED is None:
        p = path or default_path()
        if not os.path.exists(p):
            _LOADED = {}
        else:
            try:
                with open(p, 'rb') as f:
                    _LOADED = pickle.load(f)
            except Exception:
                _LOADED = {}
    return _LOADED or None
