"""Optional next-open admission experiments, separate from signal definitions.

Default off. The B2 experiment freezes the decision-visible B1 reference and
checks only the later opening price, never the later day's high/low/close.
This is a new admission hypothesis, not an original holding-exit requirement.
"""
from functools import lru_cache
from pathlib import Path
import math


class B2HoldingThesis:
    """Experimental, fill-committed B2 support; separate from admission.

    Missing basis retains the old exit behavior. Closing support failure is
    terminal for this filled lifecycle, even if a blocked sale later rebounds.
    """
    def __init__(self, provider):
        if provider.mode != 'b2_native_floor':
            raise ValueError('Holding thesis requires native B2 references')
        self.provider = provider
        self.active = {}
        self.observed_date = None
        self.current_exits = {}

    def prepare(self, date, target, positions, signals):
        return {code: dict(row, decision_date=str(date)[:10])
                for code, row in self.provider.prepare(date, target, positions, signals).items()}

    def first_fill(self, code, date, constraint):
        if not constraint:
            return None
        if code in self.active:
            raise ValueError(f'B2 lifecycle was not closed before new first fill: {code}')
        if str(date)[:10] <= constraint['decision_date']:
            raise ValueError('B2 first fill must follow its decision')
        basis = dict(constraint)
        if basis.get('known') and (not math.isfinite(basis['floor']) or basis['floor'] <= 0):
            raise ValueError('Invalid frozen B2 support')
        basis.update(first_fill_date=str(date)[:10], trigger_date=None,
                     holding_exit_premise='New experiment: decision-frozen native floor, later close<=floor')
        self.active[code] = basis
        return dict(basis)

    def closed(self, code):
        self.current_exits.pop(code, None)
        return self.active.pop(code, None)

    def observe(self, date, positions, fresh_closes):
        day = str(date)[:10]
        self.observed_date = day
        self.current_exits = {}
        observations = []
        for code, basis in self.active.items():
            if positions.get(code, 0) <= 0:
                raise ValueError(f'Frozen B2 lifecycle without actual inventory: {code}')
            if day < basis['first_fill_date']:
                raise ValueError('B2 observation predates actual entry')
            close = fresh_closes.get(code)
            known_quote = close is not None and math.isfinite(close) and close > 0
            broken = bool(basis.get('known') and known_quote and close <= basis['floor'])
            if broken and basis['trigger_date'] is None:
                basis['trigger_date'] = day
            committed = basis['trigger_date'] is not None
            record = dict(basis, code=code, observation_date=day, decision_close=close,
                          fresh_quote_known=known_quote, support_broken_now=broken,
                          exit_committed=committed)
            if committed:
                self.current_exits[code] = record
            observations.append(record)
        return observations

    def exits_for(self, date):
        if str(date)[:10] != self.observed_date:
            raise ValueError('B2 thesis requires same-session observed inventory and closes')
        return {code: dict(row) for code, row in self.current_exits.items()}


class EntryAdmission:
    MODES={'off','b2_native_floor','b1_price_confirmation'}

    def __init__(self,mode='off',data_path=None,start_date='2021-01-01'):
        if mode not in self.MODES:raise ValueError(f'Unknown entry admission mode: {mode}')
        self.mode=mode
        self.data_path=Path(data_path) if data_path is not None else None
        self.start_date=start_date

    @lru_cache(maxsize=128)
    def _native_b2(self,code,decision_date):
        import numpy as np
        import pandas as pd
        from . import chan_theory as ct
        from .b2_observation import observe_second_buy_at
        from .factor_calculator import _ema,_macd
        path=self.data_path/f'{code}_qfq.csv'
        frame=pd.read_csv(path,usecols=['datetime','close','high','low','volume'],parse_dates=['datetime'])
        frame=frame.loc[frame.datetime.between(pd.Timestamp(self.start_date),pd.Timestamp(decision_date))].sort_values('datetime')
        if frame.empty or frame.datetime.iloc[-1]!=pd.Timestamp(decision_date):
            return dict(known=False,reason='no_decision_bar')
        if frame.datetime.duplicated().any():raise ValueError(f'Duplicate decision history: {code}')
        c,h,l,v=[frame[k].to_numpy(float) for k in ['close','high','low','volume']]
        if len(c)<60:return dict(known=False,reason='insufficient_native_history')
        e20,e60,e120=[_ema(c,n) for n in (20,60,120)];macd=_macd(c)[2]
        primary=ct.compute_chan_signal(c,h,l,e20,e60,e120,macd,v)
        observed=observe_second_buy_at(c,h,l,v,e20,e60,macd,primary['buy_point'],len(c)-1)
        anchor=int(observed['second_buy_b1_ref'])
        if not observed['second_buy_point'] or not 0<=anchor<len(c):
            return dict(known=False,reason='native_B2_not_recovered_from_prefix')
        floor=float(l[anchor]*.98)
        if not math.isfinite(floor) or floor<=0:return dict(known=False,reason='invalid_native_floor')
        return dict(known=True,reason='native_B2_prefix_confirmed',floor=floor,
            origin_date=str(frame.datetime.iloc[anchor].date()),decision_date=str(pd.Timestamp(decision_date).date()),
            source='original_B2_B1_low_times_0.98',validation_scope='B2 formation-reference survival at admission, not full holding-exit validity')

    def prepare(self,date,target,current_positions,signal_store):
        if self.mode=='off':return {}
        constraints={}
        for code,value in target.items():
            if value<=0 or current_positions.get(code,0)>0:continue
            sig=signal_store.get(code,date)
            required_bp=1 if self.mode=='b1_price_confirmation' else 2
            if sig is None or int(getattr(sig,'chan_buy_point',0))!=required_bp:continue
            result=dict(self._b1_confirmation(str(code),str(date)) if self.mode=='b1_price_confirmation'
                        else self._native_b2(str(code),str(date)))
            constraints[code]=dict(result,policy=self.mode,
                unknown_behavior='Retain original eligibility; unknown is not reported as a passed native check.')
        return constraints

    @lru_cache(maxsize=128)
    def _b1_confirmation(self,code,decision_date):
        """Only decision-close price and own last20 volumes; never extend a signal."""
        import pandas as pd
        path=self.data_path/f'{code}_qfq.csv'
        frame=pd.read_csv(path,usecols=['datetime','close','high','volume'],parse_dates=['datetime'])
        frame=frame.loc[frame.datetime.between(pd.Timestamp(self.start_date),pd.Timestamp(decision_date))].sort_values('datetime')
        if frame.empty or frame.datetime.iloc[-1]!=pd.Timestamp(decision_date):
            return dict(known=False,reason='no_decision_bar')
        if frame.datetime.duplicated().any():raise ValueError(f'Duplicate decision history: {code}')
        if len(frame)<20:return dict(known=False,reason='insufficient_confirmation_history')
        close=float(frame.close.iloc[-1]);previous_high=float(frame.high.iloc[-2])
        volume=float(frame.volume.iloc[-1]);mean_volume=float(frame.volume.tail(20).mean())
        if frame.volume.tail(20).isna().any() or volume<0:
            return dict(known=False,reason='missing_or_negative_confirmation_volume')
        if not all(math.isfinite(x) for x in (close,previous_high,volume,mean_volume)) or min(close,previous_high,mean_volume)<=0:
            return dict(known=False,reason='invalid_confirmation_inputs')
        return dict(known=True,confirmed=close>previous_high and volume>=mean_volume,
            decision_date=str(pd.Timestamp(decision_date).date()),decision_close=close,
            previous_own_bar_high=previous_high,volume=volume,mean_volume20=mean_volume,
            volume_multiple=volume/mean_volume,
            source='B1 decision close>previous own high AND volume>=own20bar mean',
            validation_scope='New-entry confirmation hypothesis only; not a strict Chan B1 classifier, state pool or holding exit.')

    @staticmethod
    def opening_verdict(constraint,opening_price,current_shares,execution_date):
        if not constraint or current_shares>0:return dict(allowed=True,checked=False,reason='not_new_constrained_entry')
        if not constraint.get('known'):return dict(allowed=True,checked=False,reason='decision_reference_unknown_original_eligibility_retained')
        if str(execution_date)[:10]<=constraint['decision_date']:
            raise ValueError('Opening admission requires a session after the decision')
        if not math.isfinite(opening_price) or opening_price<=0:
            return dict(allowed=False,checked=True,reason='opening_price_unavailable')
        if constraint.get('policy')=='b1_price_confirmation':
            confirmed=bool(constraint['confirmed'])
            return dict(allowed=confirmed,checked=True,
                reason='B1_decision_price_confirmation_passed' if confirmed else 'B1_decision_price_confirmation_missing')
        if opening_price<=constraint['floor']:
            return dict(allowed=False,checked=True,reason='opening_below_or_at_frozen_B2_floor')
        return dict(allowed=True,checked=True,reason='opening_above_frozen_B2_floor')
