"""Previously published price rows must survive daily refresh byte for byte."""
import tempfile
import unittest
from pathlib import Path

from data.guard_qfq_history import audit


class QfqHistoryGuardTests(unittest.TestCase):
    def test_append_allowed_but_price_and_volume_edits_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            old, new = Path(directory) / 'old', Path(directory) / 'new'
            old.mkdir()
            new.mkdir()
            header = 'datetime,open,volume\n'
            previous = header + '2026-10-27,10,100\n'
            (old / '300953_qfq.csv').write_text(previous)
            (new / '300953_qfq.csv').write_text(previous + '2026-10-28,11,120\n')
            result = audit(old, new)
            self.assertEqual((result['changed_files'], result['new_rows_appended']), (0, 1))
            (new / '300953_qfq.csv').write_text(header + '2026-10-27,10,101\n2026-10-28,11,120\n')
            result = audit(old, new)
            self.assertEqual(result['changed_files'], 1)
            self.assertEqual(result['examples'][0]['columns'], ['volume'])
            (new / '300953_qfq.csv').write_text(header + '2026-10-27,9,100\n')
            self.assertEqual(audit(old, new)['examples'][0]['columns'], ['open'])

    def test_only_turnover_can_be_preserved_and_other_drift_leaves_candidate_untouched(self):
        with tempfile.TemporaryDirectory() as directory:
            old, new = Path(directory) / 'old', Path(directory) / 'new'
            old.mkdir()
            new.mkdir()
            header = 'datetime,close,turnover_rate\n'
            (old / '000001_qfq.csv').write_text(header + '2026-10-27,10,1.72\n')
            candidate = new / '000001_qfq.csv'
            candidate.write_text(header + '2026-10-27,10,1.73\n2026-10-28,11,1.80\n')
            report = audit(old, new)
            self.assertEqual((report['changed_files'], report['auxiliary_only_rows']), (1, 1))
            report = audit(old, new, preserve_auxiliary=True)
            self.assertEqual((report['changed_files'], report['auxiliary_rows_preserved']), (0, 1))
            self.assertEqual(candidate.read_text(), header + '2026-10-27,10,1.72\n2026-10-28,11,1.80\n')
            candidate.write_text(header + '2026-10-27,10,1.73\n2026-10-28,11,1.80\n')
            (old / '000002_qfq.csv').write_text(header + '2026-10-27,10,1.72\n')
            (new / '000002_qfq.csv').write_text(header + '2026-10-27,9,1.72\n')
            report = audit(old, new, preserve_auxiliary=True)
            self.assertEqual((report['blocked_files'], report['auxiliary_rows_preserved']), (1, 0))
            self.assertIn('1.73', candidate.read_text())
            (old / '000002_qfq.csv').unlink()
            (new / '000002_qfq.csv').unlink()
            candidate.write_text(header + '2026-10-27,9,1.73\n')
            report = audit(old, new, preserve_auxiliary=True)
            self.assertEqual(report['changed_files'], 1)
            self.assertEqual(report['auxiliary_rows_preserved'], 0)
            self.assertEqual(candidate.read_text(), header + '2026-10-27,9,1.73\n')

    def test_missing_or_truncated_history_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            old, new = Path(directory) / 'old', Path(directory) / 'new'
            old.mkdir()
            new.mkdir()
            text = 'datetime,close\n2026-10-27,10\n'
            (old / '000001_qfq.csv').write_text(text)
            self.assertEqual(audit(old, new)['examples'][0]['reason'], 'missing_file')
            (new / '000001_qfq.csv').write_text('datetime,close\n')
            self.assertEqual(audit(old, new)['examples'][0]['reason'], 'old_row_changed')
            (new / '000001_qfq.csv').write_text('date,close\n2026-10-27,10\n')
            self.assertEqual(audit(old, new)['examples'][0]['reason'], 'header_changed')

    def test_amplitude_and_change_percent_preserve_only_historical_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            old, new = Path(directory) / 'old', Path(directory) / 'new'
            old.mkdir()
            new.mkdir()
            header = 'datetime,open,high,low,close,volume,amount,amplitude,change_percent,change_amount,turnover_rate\n'
            prior = '2026-10-27,10,11,9,10,100,1000,2.31,1.87,0.18,1.72\n'
            latest = '2026-10-27,10,11,9,10,100,1000,2.30,1.88,0.18,1.72\n'
            appended = '2026-10-28,10,11,9,11,120,1200,3.00,10.00,1.00,2.00\n'
            (old / '000001_qfq.csv').write_text(header + prior)
            candidate = new / '000001_qfq.csv'
            candidate.write_text(header + latest + appended)
            result = audit(old, new, preserve_auxiliary=True)
            self.assertEqual(result['auxiliary_rows_preserved'], 1)
            self.assertEqual(result['auxiliary_columns'], {'amplitude': 1, 'change_percent': 1})
            self.assertEqual(result['auxiliary_file_names'], ['000001_qfq.csv'])
            self.assertEqual(candidate.read_text(), header + prior + appended)


if __name__ == '__main__':
    unittest.main()
