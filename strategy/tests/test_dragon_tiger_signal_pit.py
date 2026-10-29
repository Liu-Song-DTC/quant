"""Bad provider dates must not expose later events to earlier signals."""
from datetime import date
import pickle
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from strategy.core.alternative_data import AlternativeDataProvider


class DragonTigerSignalPitTests(unittest.TestCase):
    def test_malformed_provider_date_does_not_disable_asof_filter(self):
        with tempfile.TemporaryDirectory() as directory:
            provider = AlternativeDataProvider(directory)
            provider._dragon_tiger = pd.DataFrame({
                '股票代码': ['000001', '000001', '000001'],
                '最近上榜日': ['2026-10-27', 'broken-date', '2026-10-30'],
                '机构买入': [10.0, 0.0, 0.0],
                '机构卖出': [0.0, 10.0, 10.0],
            })
            self.assertGreater(provider.get_dragon_tiger_signal('000001', '2026-10-28'), 0)
            self.assertEqual(provider.get_dragon_tiger_signal('000001', 'bad-date'), 0)

    def test_invalid_query_date_never_returns_unfiltered_event_set(self):
        with tempfile.TemporaryDirectory() as directory:
            provider = AlternativeDataProvider(directory)
            (Path(directory) / 'dragon_tiger_history.pkl').write_bytes(pickle.dumps({
                '000001': {date(2026, 10, 27), date(2026, 10, 30)}}))
            self.assertEqual(provider.get_dragon_tiger_dates('000001', 'bad-date'), set())
            self.assertEqual(provider.get_dragon_tiger_dates('000001', '2026-10-28'),
                             {date(2026, 10, 27)})

    def test_aggregate_fallback_filters_bad_and_future_dates(self):
        with tempfile.TemporaryDirectory() as directory:
            provider = AlternativeDataProvider(directory)
            provider._ensure_dragon_tiger_history = lambda: None
            provider._dragon_tiger = pd.DataFrame({
                '股票代码': ['000001', '000001', '000001'],
                '最近上榜日': ['2026-10-27', 'broken-date', '2026-10-30'],
            })
            self.assertEqual(provider.get_dragon_tiger_dates('000001', '2026-10-28'),
                             {date(2026, 10, 27)})
            self.assertEqual(provider.get_dragon_tiger_dates('000001', 'bad-date'), set())


if __name__ == '__main__':
    unittest.main()
