import sys
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from analysis.ml_combination_rules import (choose_candidates,selected_metrics,
                                           fit_probability_calibration,calibrated_probability)
from core.forward_targets import short_net_targets


class CombinationRuleTests(unittest.TestCase):
    def test_selection_is_independent_of_future_returns_and_keeps_cash_slots(self):
        frame=pd.DataFrame({'date':['2026-01-01']*25,'code':[f'{i:06}' for i in range(25)],
                            'rank_prediction':np.arange(25)[::-1],
                            'downside_prediction':np.arange(25),
                            'calibrated_win_probability':[.6]*3+[.4]*22,
                            'net_return':[np.nan]+[.01]*24})
        changed=frame.copy();changed['net_return']=np.arange(25)
        for recipe in ['rank','rank_then_risk','rank_then_probability']:
            a,b=choose_candidates(frame,recipe),choose_candidates(changed,recipe)
            self.assertEqual(a.code.tolist(),b.code.tolist())
        chosen=choose_candidates(frame,'rank_then_probability')
        result=selected_metrics(chosen,1)
        self.assertEqual(result['selected_candidates'],3)
        self.assertEqual(result['selected_entry_unavailable'],1)
        self.assertAlmostEqual(result['filled_slot_fraction'],.4)
        self.assertAlmostEqual(result['mean_net_return_per_slot'],.004)

    def test_probability_calibration_is_monotone_and_handles_one_class(self):
        p=np.linspace(.05,.95,200);y=(p>.6).astype(float)
        state=fit_probability_calibration(p,y)
        values=calibrated_probability(p,state)
        self.assertTrue((np.diff(values)>=0).all());self.assertTrue(((values>=0)&(values<=1)).all())
        state=fit_probability_calibration(p,np.zeros(200))
        self.assertFalse(state['fitted']);np.testing.assert_array_equal(calibrated_probability(p,state),np.zeros(200))

    def test_future_label_details_are_separate_from_features_and_default_output_is_unchanged(self):
        dates=pd.bdate_range('2026-01-01',periods=12)
        frame=pd.DataFrame({'date':dates,'code':['600000']*12,'feature':np.arange(12)})
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'600000_qfq.csv'
            pd.DataFrame({'datetime':dates,'open':[10.,11.]+[10.]*10,
                          'close':[10.]*12,'volume':[100.]*12}).to_csv(path,index=False)
            ordinary=short_net_targets(frame,{'600000':str(path)},dates)
            labels,details=short_net_targets(frame,{'600000':str(path)},dates,return_details=True)
        pd.testing.assert_frame_equal(ordinary,labels)
        self.assertNotIn('entry_available',labels)
        self.assertTrue(details.label_matured.iloc[0])
        self.assertFalse(details.entry_available.iloc[0])
        self.assertTrue(pd.isna(details.net_return_3.iloc[0]))
        self.assertFalse(details.label_matured.iloc[-1])


if __name__=='__main__':unittest.main()
