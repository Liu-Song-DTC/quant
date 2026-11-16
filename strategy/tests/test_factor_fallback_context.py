"""Fallback factor selection must carry the requested date without changing valid paths."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.factor_library import FactorLibrary
from core.signal_engine import SignalEngine


class FactorFallbackContextTests(unittest.TestCase):
    def engine(self, library=None, industry=''):
        return SimpleNamespace(
            dynamic_factor_selector=SimpleNamespace(factor_library=library),
            _get_specific_industry=lambda *args: industry,
            _safe_get=SignalEngine._safe_get,
            _lookup_rank=lambda *args: float('nan'),
            _get_fundamental_factor_value=lambda *args: None,
            _get_cached_scoring_factors=lambda *args: [],
            _stats={'fixed_industry': 0})

    def score(self, engine, regime=0, date=pd.Timestamp('2022-03-01')):
        return SignalEngine._calculate_default_factor(
            engine, {'trend_lowvol': np.full(3, .4), 'momentum_reversal': np.full(3, .5)},
            1, regime, 'default', code='600000', current_date=date)

    def test_unknown_industry_library_receives_requested_date_and_quarter(self):
        day = pd.Timestamp('2022-03-01')
        quarter = {'asof_identity': '2022Q1'}
        for regime in (0, 1):
            with self.subTest(regime=regime):
                lib = SimpleNamespace(get_scoring_factors=Mock(return_value=[('trend_lowvol', .7, 1)]))
                with patch('core.signal_engine.INDUSTRY_FACTOR_CONFIG', {}), patch(
                        'core.signal_engine._resolve_factor_config', return_value=quarter) as resolve:
                    result = self.score(self.engine(lib), regime)
                self.assertAlmostEqual(result[1], .28)
                resolve.assert_called_once_with(day)
                lib.get_scoring_factors.assert_called_once_with('', as_of_date=day, fallback_config=quarter)

    def test_primary_or_absent_library_does_not_load_quarter(self):
        config = {'KNOWN': {'factors': ['trend_lowvol'], 'weights': [1.]}}
        with patch('core.signal_engine.INDUSTRY_FACTOR_CONFIG', config), patch(
                'core.signal_engine._resolve_factor_config', side_effect=AssertionError('eager quarter load')):
            lib = SimpleNamespace(get_scoring_factors=Mock(side_effect=AssertionError('unnecessary fallback')))
            self.assertAlmostEqual(self.score(self.engine(lib, 'KNOWN'))[1], .4)
            self.assertEqual(self.score(self.engine())[0], 'trend_lowvol+relative_strength+low_downside')

    def test_bear_quarter_is_reused_for_existing_library_fallback(self):
        day = pd.Timestamp('2022-03-01')
        quarter = {'asof_identity': '2022Q1'}
        lib = SimpleNamespace(get_scoring_factors=Mock(return_value=[('trend_lowvol', .7, 1)]))
        with patch('core.signal_engine.INDUSTRY_FACTOR_CONFIG', {}), patch(
                'core.signal_engine._resolve_factor_config', return_value=quarter) as resolve:
            self.assertAlmostEqual(self.score(self.engine(lib), regime=-1)[1], .28)
        resolve.assert_called_once_with(day)
        lib.get_scoring_factors.assert_called_once_with('', as_of_date=day, fallback_config=quarter)

    def test_undated_real_library_skips_ic_and_preserves_default_direction(self):
        lib = FactorLibrary.__new__(FactorLibrary)
        lib.store = SimpleNamespace(get_all_factors=Mock(side_effect=AssertionError('undated IC lookup')))
        with patch('core.signal_engine.INDUSTRY_FACTOR_CONFIG', {}), patch(
                'core.signal_engine._resolve_factor_config', side_effect=AssertionError('undated quarter')):
            for regime in (0, -1):
                with self.subTest(regime=regime):
                    self.assertAlmostEqual(self.score(self.engine(lib), regime, date=None)[1], .22)


if __name__ == '__main__':
    unittest.main()
