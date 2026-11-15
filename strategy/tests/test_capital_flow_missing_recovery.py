"""Numerical contracts for missing-window recovery, not threshold optimization."""
import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).parents[1]))
from core import capital_flow as cf


class CapitalFlowMissingRecoveryTests(unittest.TestCase):
    def test_finite_helpers_preserve_arithmetic(self):
        rng=np.random.default_rng(6514)
        for n in [0,1,19,20,80,200]:
            values=rng.normal(size=n)
            for window in [1,5,20,61]:
                expected=np.full(n,np.nan)
                if n>=window:
                    cs=np.cumsum(np.insert(values.astype(float),0,0))
                    expected[window-1:]=cs[window:]-cs[:-window]
                np.testing.assert_array_equal(cf._rolling_sum(values,window),expected)
            expected=values.copy()
            for i in range(1,n):expected[i]=(2/6)*values[i]+(1-2/6)*expected[i-1]
            np.testing.assert_array_equal(cf._ema_smooth(values,2/6),expected)

    def test_rolling_sum_recovers_only_after_complete_finite_window(self):
        for invalid in [np.nan,np.inf,-np.inf]:
            values=np.arange(120,dtype=float)/10;values[[0,1,30,68]]=invalid
            for window in [1,5,20]:
                expected=np.full(len(values),np.nan)
                for i in range(window-1,len(values)):
                    block=values[i-window+1:i+1]
                    if np.isfinite(block).all():expected[i]=block.sum()
                np.testing.assert_allclose(cf._rolling_sum(values,window),expected,rtol=0,atol=1e-11,equal_nan=True)

    def test_ema_unknown_remains_unknown_and_next_value_restarts(self):
        values=np.array([np.nan,2.,4.,np.inf,8.,10.,-np.inf,np.nan,6.])
        expected=np.array([np.nan,2.,3.,np.nan,8.,9.,np.nan,np.nan,6.])
        np.testing.assert_array_equal(cf._ema_smooth(values,.5),expected)

    @staticmethod
    def inputs():
        i=np.arange(180,dtype=float)
        close=10+i*.015+np.sin(i/4)*.2
        return [close,close+.1,close-.1,1000+200*np.sin(i/7),close-.02]

    def test_finite_price_history_yields_mature_finite_scores(self):
        result=cf.compute_capital_flow_signal(*self.inputs())
        self.assertTrue(np.isfinite(result['net_flow_score'][23:]).all())
        score=result['capital_flow_score'][83:]
        self.assertTrue(np.isfinite(score).all())
        self.assertTrue(((score>=0)&(score<=1)).all())

    def test_recovered_values_do_not_depend_on_later_bars(self):
        values=self.inputs();full=cf.compute_capital_flow_signal(*values)
        for cut in [24,60,84,120,179]:
            prefix=cf.compute_capital_flow_signal(*(v[:cut] for v in values))
            for field in full:
                np.testing.assert_array_equal(full[field][:cut],prefix[field],err_msg=f'{field}:{cut}')
        for cut in [24,45,90]:
            arr=np.arange(120,dtype=float);arr[[0,17,60]]=np.nan
            np.testing.assert_array_equal(cf._rolling_sum(arr,5)[:cut],cf._rolling_sum(arr[:cut],5))
            np.testing.assert_array_equal(cf._ema_smooth(arr,.4)[:cut],cf._ema_smooth(arr[:cut],.4))


if __name__=='__main__':unittest.main()
