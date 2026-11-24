"""上市名册对账的回归样例 (2026-10-28, 交接计划 §5 P0-a 交付项3)。

三个必测样例:
1. 新增代码首日失败 ⇒ 缺原始文件 ⇒ 硬失败(不能静默缩小股票池);
2. 退市旧文件 ⇒ 属于"名册外" ⇒ 只报告不阻断;
3. 上游名单异常(只回传一部分) ⇒ 结论为"无法判定" ⇒ 不得报成千上万只缺股。

另测: 新上市首日无数据在允许清单里时不阻断; 名册外QFQ同样只报告。
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'data'))
from listing_reconciliation import (  # noqa: E402
    reconcile_listing, roster_is_plausible)


def _roster(n=4000, extra=()):
    return [f'{i:06d}' for i in range(n)] + list(extra)


class ListingReconciliationTests(unittest.TestCase):
    def test_new_listing_first_day_failure_is_hard(self):
        """样例1: 新代码首日下载失败 ⇒ 缺原始文件 ⇒ 不可 ok。"""
        expected = _roster()
        raw = [c for c in expected if c != '000399']
        report = reconcile_listing(expected, raw, raw, floor=10)
        self.assertTrue(report.conclusive)
        self.assertFalse(report.ok)
        self.assertIn('000399', report.missing_raw)
        self.assertIn('缺原始文件', report.summary())

    def test_delisted_old_file_only_reported(self):
        """样例2: 退市旧文件(名册外)⇒ 只报告, 不阻断。"""
        expected = _roster()
        raw = expected + ['999999']          # 已退市但保留历史
        report = reconcile_listing(expected, raw, raw, floor=10)
        self.assertTrue(report.ok)
        self.assertIn('999999', report.extra_raw)
        self.assertNotIn('999999', report.missing_raw)

    def test_upstream_roster_anomaly_is_inconclusive(self):
        """样例3: 上游只回传一小部分 ⇒ 无法判定, 不报缺股。"""
        full = _roster(n=4000)
        truncated = full[:120]                # 上游只回传一小部分(抖动/限流)
        report = reconcile_listing(truncated, full, full, floor=3000)
        self.assertFalse(report.conclusive)
        self.assertFalse(report.ok)
        self.assertTrue(report.roster_anomaly)
        self.assertIn('无法对账', report.summary())
        # 参考值下降检测: 名册比上次少 10% 以上 ⇒ 异常 (边界: 恰好少10%不算)
        ok, note = roster_is_plausible(_roster(n=3500), reference_count=4000)
        self.assertFalse(ok)
        self.assertIn('下降', note)
        ok_edge, _ = roster_is_plausible(_roster(n=3600), reference_count=4000)
        self.assertTrue(ok_edge)

    def test_new_listing_without_data_may_be_allowlisted(self):
        """新上市首日上游无数据 ⇒ 在允许清单里 ⇒ 不阻断。"""
        expected = _roster()
        raw = [c for c in expected if c != '000399']
        report = reconcile_listing(expected, raw, raw, allowlist=['000399'], floor=10)
        self.assertTrue(report.ok)
        self.assertEqual(report.missing_raw, [])

    def test_missing_qfq_is_hard_and_extra_qfq_only_reported(self):
        expected = _roster()
        raw = expected + ['999999']
        qfq = [c for c in expected if c != '000399']
        report = reconcile_listing(expected, raw, qfq, floor=10)
        self.assertFalse(report.ok)
        self.assertIn('000399', report.missing_qfq)
        self.assertIn('999999', report.extra_raw)
        self.assertNotIn('999999', report.missing_qfq)


class ListingGateIntegrationTests(unittest.TestCase):
    """隔离样例 (计划 §5 P0-a): 只有一份 sh000001_qfq.csv 时,
    新鲜度检查此前返回 (True, 'K线: 1只全部新鲜…') ⇒ 缺股不可见。
    修复后必须硬失败。"""

    def test_check_kline_fails_when_listed_codes_are_absent(self):
        import tempfile
        from unittest.mock import patch
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        import check_data_freshness as cdf

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            sd = root / 'data' / 'stock_data'
            bt = sd / 'backtrader_data'
            raw = sd / 'raw_data'
            meta = sd / 'stock_metadata'
            for p in (bt, raw, meta):
                p.mkdir(parents=True)
            (bt / 'sh000001_qfq.csv').write_text(
                'datetime,open,close,high,low,volume,amount\n'
                '2026-10-28,1.0,1.0,1.0,1.0,1.0,1.0\n', encoding='utf-8')
            (raw / 'sh000001').mkdir()
            (meta / f'listed_codes_{__import__("datetime").date.today():%m%d}.csv').write_text(
                '\n'.join(f'{i:06d}' for i in range(4000)) + '\n', encoding='utf-8')
            with patch.object(cdf, 'BT_DIR', bt), \
                    patch.object(cdf, 'PROJECT', root), \
                    patch.object(cdf, 'KLINE_ALLOWLIST', meta / 'none.txt'), \
                    patch.object(cdf, 'LISTING_ALLOWLIST', meta / 'none.txt'):
                ok, msg = cdf.check_kline(__import__('datetime').date(2026, 10, 28))
            self.assertFalse(ok, msg)
            self.assertIn('上市名册对账', msg)

    def test_stale_allowlist_does_not_excuse_a_missing_file(self):
        """两个允许清单语义不同: 滞后清单(末bar旧)不得豁免"整个文件缺失",
        否则缺文件的股票会被静默放过 ⇒ 股票池静默缩小。"""
        import tempfile
        from unittest.mock import patch
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        import check_data_freshness as cdf

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            sd = root / 'data' / 'stock_data'
            bt = sd / 'backtrader_data'
            raw = sd / 'raw_data'
            meta = sd / 'stock_metadata'
            for p in (bt, raw, meta):
                p.mkdir(parents=True)
            (bt / 'sh000001_qfq.csv').write_text(
                'datetime,open,close,high,low,volume,amount\n'
                '2026-10-28,1.0,1.0,1.0,1.0,1.0,1.0\n', encoding='utf-8')
            (raw / 'sh000001').mkdir()
            codes = [f'{i:06d}' for i in range(4000)]
            (meta / f'listed_codes_{__import__("datetime").date.today():%m%d}.csv').write_text(
                '\n'.join(codes) + '\n', encoding='utf-8')
            stale_allow = meta / 'stale.txt'
            stale_allow.write_text('000399 退市股末bar滞后\n', encoding='utf-8')
            listing_allow = meta / 'listing.txt'
            listing_allow.write_text('', encoding='utf-8')
            with patch.object(cdf, 'BT_DIR', bt), \
                    patch.object(cdf, 'PROJECT', root), \
                    patch.object(cdf, 'KLINE_ALLOWLIST', stale_allow), \
                    patch.object(cdf, 'LISTING_ALLOWLIST', listing_allow):
                ok, msg = cdf.check_kline(__import__('datetime').date(2026, 10, 28))
            self.assertFalse(ok, msg)
            self.assertIn('上市名册对账', msg)


if __name__ == '__main__':
    unittest.main()
