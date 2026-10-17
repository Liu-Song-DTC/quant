#!/usr/bin/env python3
"""基本面可用日口径 — 法定披露截止日 (PIT)。

## 为什么

系统原约定 `数据可用日期 ≡ 报告期`(见 data_manager.py / refresh_fundamental_recent.py)。
后果: **回测**里 2010-03-31 当天就能看到 2010Q1 财报, 而该报告实际 4 月下旬才披露
⇒ 回测系统性提前 1~4 个月使用未公开财报。**实盘**则诚实(库里物理上没有未披露的行)。
⇒ 两边用的是不同的信息集 = 违反铁律"实盘就是回测, 不应该有两个东西"。

实测(10/17 复核, 与 9/13 一致): `数据可用日期 == 报告期` **100%**(5725/5725 行)。

## 为什么用**日历规则**而不是真实公告日

`zcfz_公告日期` 的 Q1/H1/Q3 是干净的(中位滞后 28/57/28 天), 但 FY 被快照污染
(+476 天 ≈ 次年年报的公告日); `lrb_/xjll_/最新公告日期` 三列整体污染(中位 392~478 天)。

关键实测: **法定截止日与真实公告日中位只差 2~5 天**(A 股公司扎堆在截止日披露):

| 季  | 真实公告日中位 | 法定截止 | 差  |
|-----|---------------|---------|-----|
| Q1  | +28 天        | 4/30    | −2  |
| H1  | +57 天        | 8/31    | −5  |
| Q3  | +28 天        | 10/31   | −3  |
| FY  | (不可恢复)    | 次年4/30 | —   |

⇒ 用日历规则的代价只有 2~5 天新鲜度, 换来的是 **可证明 PIT 安全 + 零数据依赖**
(污染列不可能再伤到我们, 因为根本不用它)。规则只依赖 `报告期`, 纯函数。

## 与 9/13 的关系

9/13 的 P0-1 审计已建影子目录 `data/stock_data/fundamental_data_deadline/`(5585 文件),
实测 NAV 1,143,938 → 1,038,615 (−9.2%), 但 **Sharpe 1.5555→1.5635、MDD 27.41%→16.81%**
(风险调整后更好)。当时判定"实盘本就诚实 ⇒ 只污染回测画像", 降级为 V2 候选。
**10/16 铁律裁定后此判定作废**: 回测≠实盘本身就是不允许的, 不分"只污染画像"。

## 用法

    from fundamental_pit import pit_available_date, DEADLINE_MMDD
    pit_available_date('20260331')  # -> '20260430'
"""
from __future__ import annotations

# 报告期 (月日) → 法定披露截止 (月日, 见《证券法》/交易所规则)
#   Q1  4/30   H1  8/31   Q3 10/31   FY 次年 4/30
DEADLINE_MMDD = {
    '0331': ('0430', 0),
    '0630': ('0831', 0),
    '0930': ('1031', 0),
    '1231': ('0430', 1),   # 年报: 次年
}

#: 非标报告期(数据源偶发)的兜底滞后天数 — 取最长的年报口径, 保守
FALLBACK_LAG_DAYS = 120


def _norm(period) -> str:
    """报告期归一为 8 位 'YYYYMMDD'。容忍 int / float / '2026-03-31' / '20260331'。"""
    if period is None:
        return ''
    if isinstance(period, float):
        if period != period:      # NaN
            return ''
        period = int(period)
    s = ''.join(ch for ch in str(period) if ch.isdigit())
    return s if len(s) == 8 else ''


def pit_available_date(period) -> str:
    """报告期 → 法定披露截止日 'YYYYMMDD'。

    非法/非标报告期返回 ''(调用方须显式处理, 不要静默回退成报告期)。
    """
    s = _norm(period)
    if not s:
        return ''
    mmdd = s[4:]
    if mmdd in DEADLINE_MMDD:
        end_mmdd, year_shift = DEADLINE_MMDD[mmdd]
        return f'{int(s[:4]) + year_shift:04d}{end_mmdd}'
    # 非标报告期: 报告期 + 兜底滞后(保守), 用 pandas 免得手写日历
    import pandas as pd
    return (pd.Timestamp(s) + pd.Timedelta(days=FALLBACK_LAG_DAYS)).strftime('%Y%m%d')


def pit_available_series(periods):
    """向量化版本: Series/array of 报告期 → Series of 可用日(str)。

    非法报告期 → ''(不静默回退)。
    """
    import numpy as np
    import pandas as pd

    s = pd.Series(periods).astype(str).str.replace(r'\D', '', regex=True)
    ok = s.str.len() == 8
    out = pd.Series('', index=s.index, dtype=object)
    if not ok.any():
        return out
    mmdd = s[ok].str[4:]
    for key, (end_mmdd, shift) in DEADLINE_MMDD.items():
        m = mmdd == key
        if not m.any():
            continue
        yr = s[ok][m].str[:4].astype(int) + shift
        out.loc[out.index[ok][m]] = yr.astype(str).str.zfill(4) + end_mmdd
    rest = ok & (out == '')
    if rest.any():
        ts = pd.to_datetime(s[rest], format='%Y%m%d', errors='coerce')
        fb = (ts + pd.Timedelta(days=FALLBACK_LAG_DAYS)).dt.strftime('%Y%m%d')
        out.loc[rest] = fb.fillna('')
    return out


if __name__ == '__main__':
    import sys
    for a in sys.argv[1:]:
        print(f'{a} -> {pit_available_date(a)}')
