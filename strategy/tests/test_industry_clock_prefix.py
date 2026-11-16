"""Industry-clock contracts through the real collector and cached resolver."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import signal_engine as module


class IndustryClockPrefixTests(unittest.TestCase):
    code = '999999'
    old, new = '合成旧概念', '合成新概念'
    inception = pd.Timestamp('2021-05-20')
    target = pd.Timestamp('2021-05-10')

    def setUp(self):
        self.dates = list(pd.bdate_range('2021-01-04', periods=200))
        self.queries = []
        def concept_map(day):
            self.queries.append(pd.Timestamp(day))
            return {self.code: [self.new, self.old]}, 'unchanged_snapshot'
        for context in [patch.object(module, '_concept_map_asof', concept_map),
                patch.object(module, '_resolve_factor_config', lambda day: {self.new: {}, self.old: {}}),
                patch.dict(module.CONCEPT_INCEPTION, {self.new: self.inception}, clear=True)]:
            context.start()
            self.addCleanup(context.stop)

    def engine(self):
        owner = module.SignalEngine.__new__(module.SignalEngine)
        owner.fundamental_data = None
        owner.industry_codes = {}
        owner._fund_cache = {}
        owner._has_fund_data_cache = False
        owner._spec_ind_cache = {}
        return owner

    def collect(self, dates):
        owner = self.engine()
        observed = {}
        def select_factor(*args, **kwargs):
            observed[pd.Timestamp(kwargs['current_date'])] = kwargs['specific_industry']
            # Other score/financial processing is unrelated to this metadata
            # boundary and is deliberately bypassed by its existing None path.
            return None
        owner._select_factor = select_factor
        owner._collect_bar_scalars({'close': np.ones(len(dates))}, self.code, dates, len(dates))
        return observed

    def test_pit_prefix_metadata_is_invariant_to_future_rows(self):
        with patch.dict('os.environ', {'QUANT_INDUSTRY_PIT': '1'}):
            prefix = self.collect([date for date in self.dates if date <= self.target])
            full = self.collect(self.dates)
        self.assertEqual(prefix[self.target], self.old)
        self.assertEqual(full[self.target], self.old)

    def test_pit_cache_recognizes_an_inception_inside_the_quarter(self):
        with patch.dict('os.environ', {'QUANT_INDUSTRY_PIT': '1'}):
            owner = self.engine()
            self.assertEqual(owner._get_specific_industry(self.code, self.target), self.old)
            self.assertEqual(owner._get_specific_industry(self.code, pd.Timestamp('2021-06-01')), self.new)

    def test_pit_collector_does_not_seed_from_a_future_midpoint(self):
        with patch.dict('os.environ', {'QUANT_INDUSTRY_PIT': '1'}):
            self.collect(self.dates)
        self.assertEqual(self.queries[0], self.dates[60])

    def test_pit_cache_recognizes_report_industry_changes_inside_quarter(self):
        with patch.dict('os.environ', {'QUANT_INDUSTRY_PIT': '1'}), \
                patch.object(module, '_concept_map_asof', lambda day: ({}, 'same')):
            owner = self.engine()
            owner.fundamental_data = SimpleNamespace(get_industry=lambda code, day:
                self.old if pd.Timestamp(day) < self.inception else self.new)
            self.assertEqual(owner._get_specific_industry(self.code, self.target), self.old)
            self.assertEqual(owner._get_specific_industry(self.code, pd.Timestamp('2021-06-01')), self.new)

    def test_pit_cache_storage_does_not_grow_with_history_length(self):
        with patch.dict('os.environ', {'QUANT_INDUSTRY_PIT': '1'}):
            owner = self.engine()
            for day in pd.bdate_range('2021-01-04', periods=1500):
                owner._get_specific_industry(self.code, day)
            self.assertEqual(len(owner._spec_ind_cache), 1)
            # A historical query after the latest query must still resolve
            # that day's metadata, rather than return the latest cache entry.
            self.assertEqual(owner._get_specific_industry(self.code, self.target), self.old)

    def test_explicit_legacy_clock_matches_default(self):
        with patch.dict('os.environ') as environment:
            environment.pop('QUANT_INDUSTRY_PIT', None)
            default = self.collect(self.dates)
        with patch.dict('os.environ', {'QUANT_INDUSTRY_PIT': '0'}):
            explicit = self.collect(self.dates)
        self.assertEqual(default, explicit)


if __name__ == '__main__':
    unittest.main()
