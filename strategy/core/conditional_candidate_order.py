"""Default-off, sealed walk-forward expected-payoff ordering research.

Predictions only reorder already qualified fresh candidates of the same primary
type and existing timing stratum. Missing coverage preserves the entire group;
unmodeled stocks are never silently assigned a bad prediction.
"""
from collections import defaultdict
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


SOURCE_FIELDS={'pure_factor':'pre_discount_score','ml_score':'ml_score',
               'score':'score','gate_quality':'_gate_quality','bom_quality':'bom_quality_score'}
BLOCK_FIELDS = tuple(f'prediction_block_{i}' for i in range(5))


def consensus_order(rows):
    """Reverse a pair only when the full fit and every past-block fit agree."""
    ordered = list(rows)
    for i in range(1, len(ordered)):
        j = i
        while j > 0:
            before, after = ordered[j-1], ordered[j]
            if not (after['prediction'] > before['prediction'] and
                    all(a > b for a, b in zip(after['block_predictions'], before['block_predictions']))):
                break
            ordered[j-1], ordered[j] = after, before
            j -= 1
    return ordered


class ConditionalCandidateOrder:
    def __init__(self,path,expected_sha256,preserve_b1_origin=False,b1_block_consensus=False,b1_origin_scope='all'):
        path=Path(path)
        with path.open('rb') as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
        if actual!=expected_sha256:raise ValueError('Conditional-order prediction hash mismatch')
        table=pd.read_parquet(path)
        required=['date','code','chan_buy_point','prediction','last_label_date',*SOURCE_FIELDS]
        self.preserve_b1_origin=bool(preserve_b1_origin)
        self.b1_block_consensus=bool(b1_block_consensus)
        self.b1_origin_scope=b1_origin_scope
        if b1_origin_scope not in ('all','V_reversal','close_rebound_not_open_gap'):
            raise ValueError('Unknown B1 ranking origin scope')
        if b1_origin_scope!='all' and not self.preserve_b1_origin:
            raise ValueError('B1 origin scope requires origin-preserving order')
        if self.b1_block_consensus and not self.preserve_b1_origin:
            raise ValueError('B1 block consensus requires origin-preserving order')
        if self.preserve_b1_origin:required.append('b1_origin')
        if self.b1_block_consensus:required.extend(BLOCK_FIELDS)
        if not set(required).issubset(table.columns):raise ValueError('Incomplete conditional-order schema')
        table=table[required].copy()
        table['date']=pd.to_datetime(table.date).dt.normalize()
        table['last_label_date']=pd.to_datetime(table.last_label_date).dt.normalize()
        table['code']=table.code.astype(str)
        if table.empty or table.duplicated(['date','code']).any():raise ValueError('Empty/duplicate conditional predictions')
        if not table.last_label_date.lt(table.date).all():raise ValueError('Conditional predictions contain immature labels')
        if not np.isfinite(table[['prediction',*SOURCE_FIELDS]].to_numpy(dtype=float)).all():
            raise ValueError('Nonfinite source scores or conditional predictions')
        if self.b1_block_consensus and not np.isfinite(table.loc[table.chan_buy_point.eq(1),list(BLOCK_FIELDS)].to_numpy(dtype=float)).all():
            raise ValueError('Nonfinite B1 block predictions')
        if not table.chan_buy_point.isin(range(1,10)).all():raise ValueError('Invalid primary buy point')
        if self.preserve_b1_origin:
            allowed={'standard_pivot_divergence','V_reversal','close_rebound_not_open_gap'}
            if not table.loc[table.chan_buy_point.eq(1),'b1_origin'].isin(allowed).all():
                raise ValueError('Unknown B1 origin in subtype-preserving prediction table')
        self.rows=table.set_index(['date','code']).to_dict('index')
        # Observation metadata only: an intentionally unmodeled family or
        # model warmup must not look like a failed prediction pipeline.
        self.prediction_dates_by_type = {
            int(point): frozenset(group.date)
            for point, group in table.groupby('chan_buy_point')}
        self.sha256=actual

    def reorder(self,qualified,date,held_codes,timing_evidence=None):
        day=pd.Timestamp(date).normalize();groups=defaultdict(list)
        fresh={};points=set()
        if timing_evidence is not None:
            fresh={row['code']:row['fresh'] for row in timing_evidence['rows']}
            scope=timing_evidence['priority_points']
            points=set(scope) if scope is not None else {int(c.get('chan_buy_point',0)) for c in qualified}
        for i,candidate in enumerate(qualified):
            if candidate['code'] in held_codes or candidate.get('is_held') or candidate.get('_locked'):continue
            bp=int(candidate.get('chan_buy_point',0))
            stratum=bool(fresh[candidate['code']]) if bp in points else None
            groups[(bp,stratum)].append(i)
        result=list(qualified);records=[]
        for (bp,stratum),indices in groups.items():
            rows=[];failure=None
            for i in indices:
                c=qualified[i];row=self.rows.get((day,c['code']))
                if row is None:
                    failure='incomplete_group_prediction_coverage';break
                if int(row['chan_buy_point'])!=bp:raise ValueError('Conditional-order source type mismatch')
                sig=c.get('sig')
                for field,attribute in SOURCE_FIELDS.items():
                    value=getattr(sig,attribute,None)
                    if value is None or not np.isfinite(float(value)) or not np.isclose(float(value),float(row[field]),rtol=2e-6,atol=1e-7):
                        raise ValueError(f'Conditional-order source score mismatch: {day} {c["code"]} {field}')
                rows.append(dict(code=c['code'],original_index=i,prediction=float(row['prediction']),
                                 last_label_date=row['last_label_date'],effective_score=float(c['effective_score'])))
                if self.preserve_b1_origin and bp==1:rows[-1]['b1_origin']=row['b1_origin']
                if self.b1_block_consensus and bp==1:
                    rows[-1]['block_predictions']=[float(row[f]) for f in BLOCK_FIELDS]
            record=dict(buy_point=bp,fresh_stratum=stratum,indices=indices,
                        codes=[qualified[i]['code'] for i in indices],reason=failure or 'covered',rows=rows)
            missing_codes = [qualified[i]['code'] for i in indices
                             if (day, qualified[i]['code']) not in self.rows]
            prediction_dates = self.prediction_dates_by_type.get(bp, frozenset())
            record['prediction_coverage'] = dict(
                requested=len(indices), available=len(indices)-len(missing_codes),
                missing_codes=missing_codes,
                domain=('unmodeled_buy_point' if not prediction_dates else
                        'before_first_prediction_date' if day < min(prediction_dates) else
                        'date_not_in_prediction_table' if day not in prediction_dates else
                        'partial_candidate_coverage' if missing_codes else 'covered'),
                whole_group_retained=bool(missing_codes))
            if failure is None:
                partitions=defaultdict(list)
                for row in rows:partitions[row.get('b1_origin')].append(row)
                for partition in partitions.values():
                    destinations=[r['original_index'] for r in partition]
                    active=(bp!=1 or self.b1_origin_scope=='all' or
                            partition[0]['b1_origin']==self.b1_origin_scope)
                    for row in partition:
                        row['ranking_applied']=active
                    ordered=(partition if not active else
                             consensus_order(partition) if self.b1_block_consensus and bp==1
                             else sorted(partition,key=lambda r:-r['prediction']))
                    for destination,origin in zip(destinations,ordered):
                        result[destination]=qualified[origin['original_index']]
                        origin['new_index']=destination
                record['changed']=sum(row['original_index']!=row['new_index'] for row in rows)
            else:record['changed']=0
            record['preserve_b1_origin']=self.preserve_b1_origin and bp==1
            record['b1_block_consensus']=self.b1_block_consensus and bp==1
            record['b1_origin_scope']=self.b1_origin_scope
            records.append(record)
        return result,dict(prediction_sha256=self.sha256,groups=records,
            changed=sum(r['changed'] for r in records),
            preserve_b1_origin=self.preserve_b1_origin,
            b1_block_consensus=self.b1_block_consensus,
            b1_origin_scope=self.b1_origin_scope,
            scope='Same-type and same-timing-stratum fresh ordering only; optional B1 origin positions also retained. Source/numeric scores, holdings, sizing and exits remain unchanged. Whole primary/timing group retains order when predictions are incomplete.')
