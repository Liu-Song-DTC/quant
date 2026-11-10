"""Cash/share replay prerequisites are measured against held event dates."""
import pickle
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from strategy.analysis.audit_action_record_coverage import audit


class RecordCoverageTests(unittest.TestCase):
    def test_hold_interval_and_missing_decomposition(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            trades, events, parts = base / 'trades.csv', base / 'events.pkl', base / 'parts'
            parts.mkdir()
            pd.DataFrame({'code': ['300953'], 'entry_date': ['2025-06-02'],
                          'exit_date': ['2025-07-01']}).to_csv(trades, index=False)
            pd.DataFrame({'code': ['300953'] * 3,
                          'ex_date': ['2025-06-01', '2025-06-13', '2025-06-20'],
                          's': [0., 0., float('nan')],
                          'D': [0.1, 0.3, float('nan')],
                          'src': ['em', 'em_repD', 'sina']}).to_pickle(events)
            with (parts / '300953.pkl').open('wb') as file:
                pickle.dump([('300953', pd.Timestamp('2025-06-13'),
                              pd.Timestamp('2025-06-12'))], file)
            detail, summary = audit(trades, events, parts)
            self.assertEqual(summary['held_event_rows'], 2)
            self.assertEqual(summary['events_with_record_date'], 1)
            self.assertEqual(summary['events_missing_record_date'], 1)
            self.assertEqual(summary['events_without_cash_share_decomposition'], 1)
            self.assertEqual(summary['events_without_unmodified_provider_terms'], 2)
            self.assertEqual(detail.record_date_known.tolist(), [True, False])

    def test_issuer_provenance_fills_missing_without_imitating_provider(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            trades, events, parts, manual = (base / 'trades.csv', base / 'events.pkl',
                                              base / 'parts', base / 'issuer.csv')
            parts.mkdir()
            pd.DataFrame({'code': ['300580'], 'entry_date': ['2023-06-19'],
                          'exit_date': ['2023-06-23']}).to_csv(trades, index=False)
            pd.DataFrame({'code': ['300580'], 'ex_date': ['2023-06-21'],
                          's': [0.5], 'D': [0.4], 'src': ['em']}).to_pickle(events)
            notice = 'https://static.cninfo.com.cn/finalpage/2023-06-14/1217053542.PDF'
            pd.DataFrame({'code': ['300580'], 'ex_date': ['2023-06-21'],
                          'record_date': ['2023-06-20'],
                          'source_url': [notice]}).to_csv(manual, index=False)
            detail, counts = audit(trades, events, parts, manual)
            self.assertEqual(counts['events_with_record_date'], 1)
            self.assertEqual(counts['events_from_provider'], 0)
            self.assertEqual(counts['events_from_issuer_only'], 1)
            self.assertEqual(counts['events_without_unmodified_provider_terms'], 0)
            self.assertEqual(detail.record_date_source.tolist(), ['issuer'])
            self.assertEqual(detail.issuer_notice_url.tolist(), [notice])
            with (parts / '300580.pkl').open('wb') as file:
                pickle.dump([('300580', pd.Timestamp('2023-06-21'),
                              pd.Timestamp('2023-06-19'))], file)
            with self.assertRaisesRegex(ValueError, 'provider/issuer record conflict'):
                audit(trades, events, parts, manual)

    def test_issuer_date_must_precede_event(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            trades, events, parts, manual = (base / 'trades.csv', base / 'events.pkl',
                                              base / 'parts', base / 'issuer.csv')
            parts.mkdir()
            pd.DataFrame({'code': ['300580'], 'entry_date': ['2023-06-19'],
                          'exit_date': ['2023-06-23']}).to_csv(trades, index=False)
            pd.DataFrame({'code': ['300580'], 'ex_date': ['2023-06-21'],
                          's': [0.5], 'D': [0.4]}).to_pickle(events)
            pd.DataFrame({'code': ['300580'], 'ex_date': ['2023-06-21'],
                          'record_date': ['2023-06-21'],
                          'source_url': ['https://static.cninfo.com.cn/notice.pdf']}).to_csv(manual, index=False)
            with self.assertRaisesRegex(ValueError, 'invalid issuer record'):
                audit(trades, events, parts, manual)


if __name__ == '__main__':
    unittest.main()
