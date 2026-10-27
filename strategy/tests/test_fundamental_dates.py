"""Scalar and vector date contracts must agree and reject malformed periods."""
from datetime import date
import importlib.util
from pathlib import Path
import unittest

import pandas as pd


spec = importlib.util.spec_from_file_location(
    'fundamental_date_contract', Path(__file__).resolve().parents[2] / 'data/fundamental_pit.py')
pit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pit)


class FundamentalDateContractTests(unittest.TestCase):
    def test_scalar_vector_agreement_with_duplicate_indexes_and_numeric_dates(self):
        values = [20260331., '2026-06-30', pd.Timestamp('2026-09-30'), date(2025, 12, 31),
                  '20260331.0', None, float('nan'), pd.NaT, '20260231',
                  20260331.5, 'garbage20260331', '00000331', '99991231', -20260331]
        series = pd.Series(values, index=['same'] * len(values))
        result = pit.pit_available_series(series)
        self.assertEqual(result.index.tolist(), series.index.tolist())
        self.assertEqual(result.tolist(), [pit.pit_available_date(v) for v in values])
        self.assertEqual(result.iloc[:5].tolist(),
                         ['20260430', '20260831', '20261031', '20260430', '20260430'])
        self.assertTrue(result.iloc[5:].eq('').all())

    def test_valid_historical_quarters_keep_existing_deadline_contract(self):
        values, expected = [], []
        for year in range(2010, 2027):
            for period, deadline in [('0331', f'{year}0430'), ('0630', f'{year}0831'),
                                     ('0930', f'{year}1031'), ('1231', f'{year + 1}0430')]:
                values.append(f'{year}{period}')
                expected.append(deadline)
        self.assertEqual(pit.pit_available_series(values).tolist(), expected)

    def test_nonstandard_valid_period_uses_same_fallback(self):
        value = '20260228'
        self.assertEqual(pit.pit_available_date(value), '20260628')
        self.assertEqual(pit.pit_available_series([value]).tolist(), ['20260628'])


if __name__ == '__main__':
    unittest.main()
