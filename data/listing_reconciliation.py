"""上市名册对账 (2026-10-28, P0-a): 应有上市代码 − 实际原始文件 − 实际QFQ文件。

为什么需要 (交接计划 §5 P0-a):
- `data/akshare_kline_downloader.py` 把每只股票的下载失败转成状态日志后**正常结束**;
- `strategy/check_data_freshness.py:check_kline()` 只枚举**现存** `*_qfq.csv`;
  ⇒ 新上市/从未建档的股票下载失败时, 既不报错也不留痕 ⇒ **股票池静默缩小**。
  隔离样例已复现: 只放一份 `sh000001_qfq.csv` 时新鲜度检查仍返回"1只全部新鲜"。

本模块只做集合对账, **不联网**、不改任何文件, 便于单测:
- 上游名册由下载器落盘 (`stock_metadata/listed_codes_<MMDD>.csv`), 门禁离线消费;
- 名册本身可能异常(上游只返回一部分) ⇒ 先判名册可信度, 不可信时结论为"无法判定",
  绝不把上游抖动报成"成千上万只缺数据"。

允许清单 (allowlist) 覆盖**应当缺席**的代码: 新上市首日上游无数据、长期停牌、
已隔离的错误实体。退市/停牌的历史文件属于"多出来的"一类, 只报告不阻断。
"""
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Set

# 上游名册低于此数量 ⇒ 判定为名册异常(而不是"全市场只剩这点股票")
ROSTER_FLOOR = 3000
# 名册比上一次少这个比例 ⇒ 判定为上游异常(不据此报缺股)
ROSTER_DROP_RATIO = 0.90


@dataclass
class ListingReconciliation:
    """一次名册对账的结果。"""

    expected_count: int = 0
    missing_raw: List[str] = field(default_factory=list)
    missing_qfq: List[str] = field(default_factory=list)
    extra_raw: List[str] = field(default_factory=list)
    extra_qfq: List[str] = field(default_factory=list)
    roster_anomaly: bool = False
    roster_note: str = ''

    @property
    def conclusive(self) -> bool:
        """名册可信 ⇒ 缺失集合可作为硬失败依据。"""
        return not self.roster_anomaly

    @property
    def ok(self) -> bool:
        return self.conclusive and not self.missing_raw and not self.missing_qfq

    def summary(self) -> str:
        if self.roster_anomaly:
            return (f'上市名册不可信({self.roster_note}) ⇒ 无法对账, '
                    f'不做缺股判定(名册数={self.expected_count})')
        parts = [f'名册{self.expected_count}只']
        if self.missing_raw:
            parts.append(f'缺原始文件{len(self.missing_raw)}')
        if self.missing_qfq:
            parts.append(f'缺QFQ{len(self.missing_qfq)}')
        if self.extra_raw:
            parts.append(f'名册外原始文件{len(self.extra_raw)}')
        if self.extra_qfq:
            parts.append(f'名册外QFQ{len(self.extra_qfq)}')
        if not (self.missing_raw or self.missing_qfq):
            parts.append('无缺股')
        return '上市名册对账: ' + ', '.join(parts)

    def to_dict(self) -> dict:
        return {
            'expected_count': self.expected_count,
            'missing_raw': list(self.missing_raw),
            'missing_qfq': list(self.missing_qfq),
            'extra_raw': list(self.extra_raw),
            'extra_qfq': list(self.extra_qfq),
            'roster_anomaly': self.roster_anomaly,
            'roster_note': self.roster_note,
            'conclusive': self.conclusive,
            'ok': self.ok,
        }


def roster_is_plausible(expected: Iterable[str],
                         reference_count: Optional[int] = None,
                         floor: int = ROSTER_FLOOR,
                         drop_ratio: float = ROSTER_DROP_RATIO) -> tuple:
    """判断上游名册本身是否可信。返回 (是否可信, 说明)。"""
    n = len(set(expected))
    if n < floor:
        return False, f'名册仅{n}只(下限{floor})'
    if reference_count and n < reference_count * drop_ratio:
        return False, (f'名册{n}只较上次{reference_count}只下降'
                       f'{100 * (1 - n / max(reference_count, 1)):.1f}%'
                       f'(阈值{100 * (1 - drop_ratio):.0f}%)')
    return True, ''


def reconcile_listing(expected: Iterable[str],
                       raw_codes: Iterable[str],
                       qfq_codes: Iterable[str],
                       allowlist: Iterable[str] = (),
                       reference_count: Optional[int] = None,
                       floor: int = ROSTER_FLOOR,
                       drop_ratio: float = ROSTER_DROP_RATIO) -> ListingReconciliation:
    """应有上市代码 − 实际原始文件 − 实际QFQ文件。

    allowlist: 应当缺席的代码(新上市首日无数据/长期停牌/已隔离错误实体)。
    多出来的文件(退市旧文件等)只报告, 不阻断。
    """
    exp: Set[str] = {str(c).strip() for c in expected if str(c).strip()}
    raw: Set[str] = {str(c).strip() for c in raw_codes if str(c).strip()}
    qfq: Set[str] = {str(c).strip() for c in qfq_codes if str(c).strip()}
    allow: Set[str] = {str(c).strip() for c in allowlist if str(c).strip()}

    report = ListingReconciliation(expected_count=len(exp))
    plausible, note = roster_is_plausible(exp, reference_count, floor, drop_ratio)
    if not plausible:
        report.roster_anomaly = True
        report.roster_note = note
        # 名册不可信: 缺失集合不可作为硬失败依据, 但仍记录以便人工查看
        report.missing_raw = sorted(exp - raw - allow)
        report.missing_qfq = sorted(exp - qfq - allow)
        report.extra_raw = sorted(raw - exp)
        report.extra_qfq = sorted(qfq - exp)
        return report

    report.missing_raw = sorted(exp - raw - allow)
    report.missing_qfq = sorted(exp - qfq - allow)
    report.extra_raw = sorted(raw - exp)
    report.extra_qfq = sorted(qfq - exp)
    return report
