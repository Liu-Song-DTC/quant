"""Candidate-derived chain cropping is a distinct, explicit policy choice."""
import contextlib
import io
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import portfolio as module
import pandas as pd


class CandidateChainFocusTests(unittest.TestCase):
    def construct(self, policy=None):
        # A has the strongest median among groups with >=2 candidates. B has
        # the strongest individual name. A count/median crop removes that name.
        specs = [('000001', .60, 'A'), ('000002', .59, 'A'), ('000003', .58, 'A'),
            ('000004', 1., 'B'), ('000005', .10, 'B')]
        signals = {code: SimpleNamespace(buy=True, sell=False, score=score,
            factor_value=.5, industry=industry, signal_level=3, chan_buy_point=3,
            chan_sell_point=0, bi_buy_point=1, chan_divergence_type='bottom',
            chan_divergence_strength=.3, ma_trend_up=True, trend_type=2,
            exhaustion_risk=0., risk_vol=.03, _gate_quality=1., factor_name='SYNTHETIC',
            volume_ratio=0., daily_return=0., mom_60d=0., dist_ma60=0., bom_quality_score=.3)
            for code, score, industry in specs}
        store = SimpleNamespace(get=lambda code, date: signals.get(code))
        with contextlib.redirect_stdout(io.StringIO()), patch.object(module, 'plog', MagicMock()), \
                patch('core.macro_data.load_macro_data', return_value=None), \
                patch('core.industry_chain.get_chain_concepts', return_value={'A'}) as lookup, \
                patch.dict(os.environ):
            os.environ.pop('QUANT_CANDIDATE_CHAIN_FOCUS', None)
            if policy is not None:
                os.environ['QUANT_CANDIDATE_CHAIN_FOCUS'] = policy
            owner = module.PortfolioConstructor()
            owner._reduction_codes = lambda date: set()
            owner._reduction_plan_codes = lambda date: set()
            owner._unlock_codes = lambda date: set()
            targets = owner._build_desired_value(date=pd.Timestamp('2024-12-18').date(),
                universe=list(signals), current_positions={code: 10000. for code in signals},
                cash=50000., signal_store=store, prices={code: 10. for code in signals},
                trend_score=1., index_volume_ratio=1.)
            return targets, lookup.call_count

    def test_default_preserves_existing_candidate_crop(self):
        targets, calls = self.construct()
        self.assertEqual(set(targets), {'000001', '000002', '000003'})
        self.assertEqual(calls, 1)

    def test_explicit_enabled_matches_default(self):
        default, _ = self.construct()
        enabled, _ = self.construct('1')
        self.assertEqual(default, enabled)

    def test_disabled_policy_keeps_strong_name_outside_candidate_chain(self):
        targets, calls = self.construct('0')
        self.assertIn('000004', targets)
        self.assertEqual(calls, 0)


if __name__ == '__main__':
    unittest.main()
