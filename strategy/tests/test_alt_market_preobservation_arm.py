"""Research arm zeroes unsupported market signals without changing risk filters."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from strategy.core.alt_data_versions import record_snapshot
from strategy.core.alternative_data import AlternativeDataProvider


class AltMarketPreobservationArmTests(unittest.TestCase):
    def test_before_first_observation_is_zero_only_in_research_arm(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dates = pd.date_range('2024-01-02', periods=6, freq='B')
            north = root / 'northbound_daily.pkl'
            margin = root / 'margin_daily.pkl'
            pd.DataFrame({'date': dates, 'net_flow': [10.] * 6}).to_pickle(north)
            pd.DataFrame({'date': dates, 'margin_balance': [100., 110., 120., 130., 140., 150.]}).to_pickle(margin)
            for source in (north, margin):
                record_snapshot(source, observed_on='2026-10-29')
            default = AlternativeDataProvider(str(root))
            with patch.dict('os.environ', {'QUANT_ALT_ZERO_PREOBSERVED_MARKET': '1'}):
                strict = AlternativeDataProvider(str(root))
            for accessor in ('get_northbound_signal', 'get_margin_signal'):
                with self.subTest(accessor=accessor):
                    self.assertNotEqual(getattr(default, accessor)('2024-01-09'), 0)
                    self.assertEqual(getattr(strict, accessor)('2024-01-09'), 0)
            self.assertEqual(default._zero_preobserved_market_signals, False)
            self.assertEqual(strict._zero_preobserved_market_signals, True)
            self.assertIsNone(strict._reduction)  # Arm does not mutate or disable filters.


if __name__ == '__main__':
    unittest.main()
