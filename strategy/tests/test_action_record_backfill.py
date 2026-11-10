"""Backfill checks frozen held ex-dates before writing independent metadata."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import requests

import pandas as pd

from strategy.analysis.backfill_action_record_dates import backfill


class ActionRecordBackfillTests(unittest.TestCase):
    def test_only_matching_held_event_is_saved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trades, events, parts = root / 'trades.csv', root / 'events.pkl', root / 'parts'
            pd.DataFrame({'code': ['300953'], 'entry_date': ['2026-05-01'],
                          'exit_date': ['2026-06-01']}).to_csv(trades, index=False)
            pd.DataFrame({'code': ['300953'], 'ex_date': ['2026-05-28'],
                          's': [.4], 'D': [.3]}).to_pickle(events)
            good = pd.DataFrame({'股权登记日': ['2026-05-27'],
                                 '除权除息日': ['2026-05-28']})
            seen = []
            def provider(**_):
                requests.get('https://example.invalid')
                return good
            with patch('requests.get', side_effect=lambda *a, **kw: seen.append(kw['timeout'])), \
                 patch('strategy.analysis.backfill_action_record_dates.ak.stock_fhps_detail_em',
                       side_effect=provider):
                auto, _ = backfill(trades, events, parts, workers=1)
            self.assertEqual(auto[0]['status'], 'ok', auto[0])
            self.assertEqual(seen, [(6, 18)])
            (parts / '300953.pkl').unlink()
            result, coverage = backfill(trades, events, parts, workers=1,
                                        fetch=lambda _: good)
            self.assertEqual(result[0]['status'], 'ok')
            self.assertEqual(coverage['events_with_record_date'], 1)
            self.assertTrue((parts / '300953.pkl').exists())
            again, coverage2 = backfill(trades, events, parts, workers=1,
                                         fetch=lambda _: self.fail('should skip covered event'))
            self.assertEqual(again, [])
            self.assertEqual(coverage2['events_with_record_date'], 1)
            self.assertEqual(pd.read_pickle(events).iloc[0]['D'], .3)
            (parts / '300953.pkl').unlink()
            wrong = good.assign(除权除息日='2026-05-29')
            result, coverage = backfill(trades, events, parts, workers=1,
                                        fetch=lambda _: wrong)
            self.assertEqual(result[0]['status'], 'failed')
            self.assertEqual(coverage['events_with_record_date'], 0)
            self.assertFalse((parts / '300953.pkl').exists())


if __name__ == '__main__':
    unittest.main()
