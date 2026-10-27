import importlib.util
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

path=Path(__file__).resolve().parents[1]/'analysis/probe_ml_objectives.py'
spec=importlib.util.spec_from_file_location('objective_probe',path)
probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)


class ObjectiveProbeTests(unittest.TestCase):
    def test_purging_excludes_labels_crossing_fit_and_validation_boundaries(self):
        dates=pd.date_range('2023-01-01','2024-03-01',freq='7D')
        frame=pd.DataFrame({'date':dates,'label_end_date':dates+pd.Timedelta(days=15)})
        fit,val,test=probe.purged_split(frame,'2024-01-01','2025-01-01')
        self.assertLess(frame.loc[fit,'label_end_date'].max(),pd.Timestamp('2024-01-01')-pd.Timedelta(days=120))
        self.assertLess(frame.loc[val,'label_end_date'].max(),pd.Timestamp('2024-01-01'))
        self.assertGreaterEqual(frame.loc[test,'date'].min(),pd.Timestamp('2024-01-01'))
        self.assertFalse((fit&val).any());self.assertFalse((val&test).any())

    def test_top_five_metrics_use_same_dates_and_downside_direction_is_explicit(self):
        days=np.repeat(pd.date_range('2026-01-01',periods=2),10)
        scores=np.tile(np.arange(10),2);returns=(scores-4.5)/100
        high=probe.summarize(scores,returns,days)
        low=probe.summarize(scores,returns,days,higher_is_better=False)
        self.assertEqual(high['top5_samples'],10)
        self.assertEqual(high['top5_win_rate'],1.)
        self.assertEqual(low['top5_win_rate'],0.)
        self.assertAlmostEqual(high['universe_win_rate'],.5)
        self.assertAlmostEqual(high['top5_mean_downside'],0.)

    def test_unavailable_best_stock_is_selected_then_cash_without_hindsight_refill(self):
        result=probe.summarize([6,5,4,3,2,1],[np.nan,.03,.02,.01,-.01,.50],['2026-01-01']*6)
        self.assertEqual(result['top5_samples'],5)
        self.assertEqual(result['top5_entry_unavailable'],1)
        self.assertAlmostEqual(result['top5_win_rate'],.75)
        self.assertAlmostEqual(result['top5_mean_net_return_per_slot'],.01)

    def test_calibration_interval_is_separate_from_early_stopping_and_test(self):
        dates=pd.date_range('2022-01-01','2024-02-01',freq='7D')
        frame=pd.DataFrame({'date':dates,'label_end_date':dates+pd.Timedelta(days=15)})
        fit,val,cal,test=probe.purged_four_split(frame,'2024-01-01','2025-01-01')
        self.assertLess(frame.loc[fit,'label_end_date'].max(),pd.Timestamp('2024-01-01')-pd.Timedelta(days=180))
        self.assertLess(frame.loc[val,'label_end_date'].max(),pd.Timestamp('2024-01-01')-pd.Timedelta(days=90))
        self.assertLess(frame.loc[cal,'label_end_date'].max(),pd.Timestamp('2024-01-01'))
        self.assertTrue(all(not (a&b).any() for a,b in [(fit,val),(val,cal),(cal,test)]))


if __name__=='__main__':unittest.main()
