"""Optional source-specific profit protection; no admission or ranking changes.

Research only, default off. Lifecycle evidence is committed on real first
fills and removed on real full closes. Admission support is not reused as a
permanent holding stop. A blocked exit remains committed.
"""
from functools import lru_cache
from pathlib import Path
import hashlib
import json
import math
import os


def _truth(value):
    return type(value).__name__ in ('bool','bool_') and bool(value)


class DailyExitContext:
    """Bounded price cache and one monthly independent peer table."""
    def __init__(self,data_path,peer_path,calendar,start_date='2021-01-01'):
        import pandas as pd
        self.data_path=Path(data_path);self.peer_path=Path(peer_path);self.start_date=start_date
        self.calendar=pd.DatetimeIndex(calendar).normalize()
        if self.calendar.has_duplicates or not self.calendar.is_monotonic_increasing:
            raise ValueError('Profit-guard calendar must be unique and ordered')
        self.manifest=json.loads((self.peer_path/'manifest.json').read_text())
        if self.manifest.get('all_market_universe') is not True:
            raise ValueError('Profit protection requires independent all-market peer coverage')
        if self.manifest['validation']['known_mask_mismatches'] or self.manifest['validation']['ebb_mismatches']:
            raise ValueError('Peer context did not pass canonical classification validation')

    @lru_cache(maxsize=64)
    def _prices(self,code):
        import pandas as pd
        frame=pd.read_csv(self.data_path/f'{code}_qfq.csv',usecols=['datetime','high','low','close','volume'],parse_dates=['datetime']).set_index('datetime').sort_index()
        if frame.index.has_duplicates:raise ValueError(f'Duplicate price dates: {code}')
        frame['ma20']=frame.close.rolling(20,min_periods=20).mean()
        frame=frame.loc[frame.index>=pd.Timestamp(self.start_date)].copy()
        prior=frame.close.shift()
        tr=pd.concat([frame.high-frame.low,(frame.high-prior).abs(),(frame.low-prior).abs()],axis=1).max(axis=1)
        frame['true_range']=tr
        return frame

    @lru_cache(maxsize=1)
    def _peers(self,month):
        import pandas as pd
        relative=f'months/month={month}/context.parquet'
        expected=self.manifest['output_sha256'].get(relative)
        if expected is None:raise ValueError(f'Missing declared peer month: {month}')
        path=self.peer_path/relative
        with path.open('rb') as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
        if actual!=expected:raise ValueError(f'Peer context changed: {month}')
        table=pd.read_parquet(path,columns=['code','date','peer_known5','peer_ebb','peer_return5','peer_relative5','peer_breadth5'])
        table['code']=table.code.astype(str);table['date']=pd.to_datetime(table.date)
        table=table.set_index(['code','date']).sort_index()
        if not table.index.is_unique:raise ValueError('Duplicate stock-date peer evidence')
        return table

    def prepare(self,code,date):
        import pandas as pd
        day=pd.Timestamp(date).normalize();frame=self._prices(str(code))
        if day not in frame.index:return dict(known=False,reason='decision_quote_missing')
        known=frame.loc[:day,'true_range']
        atr=float(known.tail(20).mean()) if len(known)>=20 else float('nan')
        if not math.isfinite(atr) or atr<=0:return dict(known=False,reason='entry_ATR_unknown')
        return dict(known=True,ATR20=atr,reason='decision_known_ATR20')

    def observe(self,code,date):
        import pandas as pd
        day=pd.Timestamp(date).normalize();frame=self._prices(str(code));result=dict(quote_known=False,price_two_below=False,peer_known=False,peer_ebb=False)
        if day not in frame.index:return result
        close=float(frame.at[day,'close']);volume=float(frame.at[day,'volume'])
        if not math.isfinite(close) or close<=0 or not math.isfinite(volume) or volume<=0:return result
        result.update(quote_known=True,close=close)
        k=self.calendar.get_indexer([day])[0]
        if k>0:
            previous=self.calendar[k-1]
            if previous in frame.index:
                ma=float(frame.at[day,'ma20']);old_ma=float(frame.at[previous,'ma20']);old_close=float(frame.at[previous,'close'])
                result['price_two_below']=bool(all(math.isfinite(v) for v in [ma,old_ma,old_close]) and close<ma and old_close<old_ma)
        daystr=day.strftime('%Y-%m-%d')
        if not self.manifest['start']<=daystr<=self.manifest['end']:return result
        peers=self._peers(day.strftime('%Y%m'));key=(str(code),day)
        if key not in peers.index:return result
        row=peers.loc[key]
        result.update(peer_known=_truth(row.peer_known5),peer_ebb=_truth(row.peer_ebb),
            peer_return5=row.peer_return5,peer_relative5=row.peer_relative5,peer_breadth5=row.peer_breadth5)
        return result


class B2ContextProfitGuard:
    """Primary B2 by default; B4 scope is an isolated replay experiment.

    Historical class/event identifiers retain B2 for compatibility. Every
    lifecycle carries its actual primary type and the explicit research scope.
    """
    def __init__(self,provider,primary_points=None):
        if primary_points is None:
            scope=os.environ.get('QUANT_RESEARCH_PROFIT_GUARD_POINTS','2')
            if scope not in ('2','2,4'):
                raise ValueError('Profit-guard research scope must be 2 or 2,4')
            if scope!='2' and (os.environ.get('QUANT_SIGNAL_REPLAY')!='1' or
                              os.environ.get('QUANT_B2_CONTEXT_PROFIT_GUARD')!='1'):
                raise ValueError('Extended profit guard requires an explicitly enabled sealed replay')
            primary_points=tuple(int(p) for p in scope.split(','))
        if tuple(primary_points) not in ((2,),(2,4)):
            raise ValueError('Unsupported profit-guard primary types')
        self.primary_points=tuple(primary_points)
        self.provider=provider;self.active={};self.observed_date=None;self.current_exits={}

    def prepare(self,date,target,positions,signals):
        prepared={}
        for code,value in target.items():
            if value<=0 or positions.get(code,0)>0:continue
            signal=signals.get(code,date)
            if signal is None:continue
            point=int(getattr(signal,'chan_buy_point',0))
            if point not in self.primary_points:continue
            row=self.provider.prepare(code,date)
            prepared[code]=dict(row,decision_date=str(date)[:10],source_chan_buy_point=point,
                guard_primary_points=self.primary_points,
                source_bi_buy_point=int(getattr(signal,'bi_buy_point',0)),
                premise=f'B{point} profit protection after gain plus daily-price and independent-peer deterioration')
        return prepared

    def first_fill(self,code,date,constraint,all_in_unit_cost):
        if not constraint:return None
        if code in self.active:raise ValueError('Profit-guard lifecycle not closed before new first fill')
        day=str(date)[:10]
        if day<=constraint['decision_date']:raise ValueError('First fill must follow its decision')
        if not math.isfinite(all_in_unit_cost) or all_in_unit_cost<=0:raise ValueError('Invalid actual entry cost')
        row=dict(constraint,first_fill_date=day,first_fill_unit_cost=float(all_in_unit_cost),peak_close=None,trigger_date=None)
        if row.get('known') and (not math.isfinite(row['ATR20']) or row['ATR20']<=0):raise ValueError('Invalid frozen ATR')
        self.active[code]=row
        return dict(row)

    def closed(self,code):
        self.current_exits.pop(code,None)
        return self.active.pop(code,None)

    def observe(self,date,positions,fresh_closes):
        day=str(date)[:10]
        if self.observed_date is not None and day<self.observed_date:raise ValueError('Profit-guard time moved backwards')
        self.observed_date=day;self.current_exits={};observations=[]
        for code,basis in self.active.items():
            if positions.get(code,0)<=0:raise ValueError('Profit-guard lifecycle without actual inventory')
            if day<basis['first_fill_date']:raise ValueError('Observation before actual first fill')
            supplied=fresh_closes.get(code)
            fresh=supplied is not None and math.isfinite(supplied) and supplied>0
            context=self.provider.observe(code,date) if fresh else dict(quote_known=False,price_two_below=False,peer_known=False,peer_ebb=False)
            triggered=False
            if basis.get('known') and context.get('quote_known'):
                close=float(context['close'])
                if not math.isclose(close,supplied,rel_tol=2e-6,abs_tol=1e-4):raise ValueError('Execution/context decision quotes disagree')
                basis['peak_close']=max(basis['peak_close'] if basis['peak_close'] is not None else close,close)
                atr=basis['ATR20'];peak=basis['peak_close']
                triggered=bool(peak>basis['first_fill_unit_cost']+atr and close<=peak-atr and
                    context.get('price_two_below') and context.get('peer_known') and context.get('peer_ebb'))
                if triggered and basis['trigger_date'] is None:basis['trigger_date']=day
            record=dict(basis,**context,code=code,observation_date=day,fresh_quote_known=fresh,
                triggered_now=triggered,exit_committed=basis['trigger_date'] is not None)
            if record['exit_committed']:self.current_exits[code]=record
            observations.append(record)
        return observations

    def exits_for(self,date):
        if str(date)[:10]!=self.observed_date:raise ValueError('Profit protection needs same-session observation')
        return {code:dict(row) for code,row in self.current_exits.items()}
