"""Observe sector context before inspecting any buy flags or future labels.

No ranking policy, order generation or portfolio decisions are changed. Price
and snapshot sources retain their inherited historical-provenance limitations.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import ast
import hashlib
import io
import json
import multiprocessing
from pathlib import Path
import pickle
import sys

import numpy as np
import pandas as pd
import yaml

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'strategy'))
from core.stock_pool import _prefix_nonstock_flags, _is_non_stock_series, _monthly_boundaries


def save(path,value):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    temp.replace(path)


def state(out,stage,**extra):
    save(out/'state.json',dict(stage=stage,updated_at=datetime.now().isoformat(),**extra))


def asset(code):
    return len(code)==6 and code.isdigit() and not code.startswith(('4','8','92'))


def read_stock(args):
    code,path,start,end,bounds,floor=args
    path=Path(path)
    before=path.stat(); payload=path.read_bytes(); after=path.stat()
    assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
    frame=pd.read_csv(io.BytesIO(payload),usecols=['datetime','close','volume','amount'])
    frame['datetime']=pd.to_datetime(frame.datetime)
    if frame.datetime.duplicated().any() or not frame.datetime.is_monotonic_increasing:
        raise ValueError(f'unordered/duplicate price dates: {code}')
    # Values at a prefix cannot depend on later rows, even when the full-file
    # scalar classifier changes. Audit that difference before interpreting it.
    flags=_prefix_nonstock_flags(frame)
    old=_is_non_stock_series(frame)
    dates=frame.datetime.to_numpy(dtype='datetime64[ns]')
    changes=[]
    if code not in {'000001','399006'}:
        threshold=40e6 if code.startswith('688') else 80e6 if code.startswith('300') else 160e6
        for day in bounds:
            cut=int(np.searchsorted(dates,np.datetime64(day),'right'))
            if cut>=60 and old!=bool(flags[cut-1]):
                eligible=bool(frame.close.iloc[cut-1]>=2 and frame.amount.iloc[cut-20:cut].mean()>=threshold*floor)
                changes.append(dict(code=code,date=str(pd.Timestamp(day).date()),
                    old_nonstock=old,asof_nonstock=bool(flags[cut-1]),other_pool_conditions_pass=eligible))
    close=frame.close
    result=pd.DataFrame(dict(code=code,date=frame.datetime,amount=frame.amount,
        ret_1d=close.pct_change(fill_method=None),ret_5d=close/close.shift(5)-1,
        ret_20d=close/close.shift(20)-1,above_ma20=(close>close.rolling(20).mean()).astype(float)))
    valid=(np.arange(len(frame))>=59)&~flags&close.gt(0)&frame.volume.gt(0)&frame.amount.gt(0)
    result=result.loc[valid & frame.datetime.between(start,end)]
    return result,changes,dict(path=str(path),sha256=hashlib.sha256(payload).hexdigest(),
        size=after.st_size,mtime_ns=after.st_mtime_ns)


def calendar_sector_features(sectors, market):
    """Use exchange-session windows and preserve missing observations.

    Each sector starts at its first observed date. A sector introduced later
    cannot manufacture historical keys or values before it was observed.
    """
    calendar = pd.DatetimeIndex(market.date.sort_values().unique())
    if market.date.duplicated().any() or sectors.duplicated(['sector', 'date']).any():
        raise ValueError('Duplicate market or sector dates')
    observed_dates = pd.DatetimeIndex(sectors.date.unique())
    if not observed_dates.isin(calendar).all():
        raise ValueError('Sector date absent from market calendar')
    parts = []
    context = market.set_index('date')
    for name, part in sectors.groupby('sector', sort=True):
        dates = calendar[calendar >= part.date.min()]
        frame = part.set_index('date').reindex(dates)
        frame.index.name = 'date'
        frame['sector'] = name
        frame['benchmark_return'] = context.benchmark_return.reindex(dates)
        frame['market_equalweight_return'] = context.equal_weight_return.reindex(dates)
        frame['excess_log_return'] = np.log1p(frame.ret_1d) - np.log1p(frame.benchmark_return)
        frame['market_excess_log_return'] = np.log1p(frame.ret_1d) - np.log1p(frame.market_equalweight_return)
        for horizon in (5, 20):
            frame[f'relative_compound_{horizon}d'] = np.expm1(
                frame.excess_log_return.rolling(horizon, min_periods=horizon).sum())
            frame[f'relative_to_market_{horizon}d'] = np.expm1(
                frame.market_excess_log_return.rolling(horizon, min_periods=horizon).sum())
        positive = frame.excess_log_return.gt(0).where(frame.excess_log_return.notna())
        frame['positive_relative_days_5'] = positive.rolling(5, min_periods=5).sum()
        frame['amount_share_vs20'] = frame.amount_share / frame.amount_share.rolling(20, min_periods=20).mean()
        parts.append(frame.reset_index())
    return pd.concat(parts, ignore_index=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path)
    p.add_argument('--start',default='2026-09-01');p.add_argument('--end',default='2026-11-13')
    p.add_argument('--workers',default=2,type=int);a=p.parse_args();out=a.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    cfg=yaml.safe_load((ROOT/'strategy/config/factor_config.yaml').read_text())
    history=ROOT/'data/concept_map_history.pkl'; inception=ROOT/'data/concept_inception.pkl'
    versions=sorted(pickle.loads(history.read_bytes())['versions'],key=lambda v:str(v['date']))
    computation_start=max(pd.Timestamp(a.start)-pd.Timedelta(days=60),pd.Timestamp(versions[0]['date']))
    inceptions={k:pd.Timestamp(v) for k,v in pickle.loads(inception.read_bytes()).items()}
    engine=ROOT/'strategy/core/signal_engine.py'
    node=next(n for n in ast.parse(engine.read_text()).body if isinstance(n,ast.Assign)
              and any(isinstance(t,ast.Name) and t.id=='STYLE_KW' for t in n.targets))
    excluded=ast.literal_eval(node.value)+['AH股','上证','沪深300','深成','中证']
    protected={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in
        [Path(__file__),ROOT/'strategy/core/stock_pool.py',engine,history,inception,ROOT/'strategy/config/factor_config.yaml']}
    save(out/'manifest.json',dict(source_sha256=protected,start=a.start,end=a.end,
        computation_start=str(computation_start.date()),workers=a.workers,production_policy_changed=False,
        old_computed_cache_reused=False,future_labels_read=False,buy_flags_read=False,
        inputs='Current QFQ prices, actual turnover amounts, dated concept-map snapshots; no candidate-derived sector universe.',
        limitations=['Concept snapshots use supplied version dates; not a certification of original collection/publication time.',
            'Market mother universe includes ST; no uncertified historical limit-up/down counts.',
            'Overlapping concept amount shares cannot be added together. No true cash-flow interpretation.',
            'Previously researched dates are descriptive, not untouched out-of-sample evaluation.'],
        feature_clock='5/20 global market sessions; missing sector observations remain unknown, not skipped or filled with zero.',
        benchmarks='Report both Shanghai-index-relative and all-stock-equal-weight-relative strength; do not choose a benchmark after labels.',
        excluded_theme_substrings=excluded))
    data=ROOT/'data/stock_data/backtrader_data'
    bounds=_monthly_boundaries(pd.Timestamp('2021-01-01'),pd.Timestamp(a.end))
    tasks=[(f.name[:-8],str(f),str(computation_start.date()),a.end,bounds,float(cfg['stock_pool']['pool_relax_floor']))
           for f in sorted(data.glob('*_qfq.csv')) if asset(f.name[:-8])]
    state(out,'reading_all_stock_prices',total=len(tasks))
    parts=[]; differences=[]; stamps=[]
    with ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context('spawn')) as executor:
        for i,(frame,diff,stamp) in enumerate(executor.map(read_stock,tasks,chunksize=20),1):
            parts.append(frame);differences.extend(diff);stamps.append(stamp)
            if i%200==0:state(out,'reading_all_stock_prices',completed=i,total=len(tasks))
    stocks=pd.concat(parts,ignore_index=True);del parts
    stocks.to_parquet(out/'stock_context.parquet',index=False)
    save(out/'pool_prefix_real_input_audit.json',dict(files=len(stamps),boundaries=len(bounds),
        changed_classification_boundaries=len(differences),
        potentially_changed_memberships=sum(x['other_pool_conditions_pass'] for x in differences),
        differences=differences,scope='Classifier-only delta, price/liquidity conditions evaluated with current configuration; index exceptions retained.'))
    save(out/'price_inputs.json',stamps)
    index=pd.read_csv(data/'sh000001_qfq.csv',usecols=['datetime','close'],parse_dates=['datetime']).set_index('datetime')
    benchmark=index.close.pct_change(fill_method=None)
    rows=[];market_rows=[];maps={}
    for day,part in stocks.groupby('date',sort=True):
        eligible=[v for v in versions if str(v['date'])[:10]<=str(day.date())]
        if not eligible:raise ValueError(f'no known membership version at {day}')
        version=eligible[-1];key=str(version['date'])
        if key not in maps:
            inverse={}
            for code,names in version['map'].items():
                code=str(code).zfill(6)
                if not asset(code):continue
                for name in set(names):
                    if any(token in name for token in excluded):continue
                    inverse.setdefault(name,set()).add(code)
            maps[key]=inverse
        part=part.set_index('code');total=float(part.amount.sum())
        market_rows.append(dict(date=day,stocks=len(part),amount=total,
            advancing_fraction=float(part.ret_1d.gt(0).mean()),declining_fraction=float(part.ret_1d.lt(0).mean()),
            above_ma20_fraction=float(part.above_ma20.mean()),benchmark_return=float(benchmark.loc[day]),
            equal_weight_return=float(part.ret_1d.mean())))
        for name,members in maps[key].items():
            if inceptions.get(name,pd.Timestamp.min)>day:continue
            matched=part.loc[part.index.intersection(members)]
            if matched.empty:continue
            rows.append(dict(date=day,sector=name,membership_version=key,members=len(members),
                observed=len(matched),coverage=len(matched)/len(members),
                ret_1d=float(matched.ret_1d.mean()),amount_share=float(matched.amount.sum()/total),
                advancing_fraction=float(matched.ret_1d.gt(0).mean()),
                above_ma20_fraction=float(matched.above_ma20.mean()),
                benchmark_return=float(benchmark.loc[day])))
    sectors=calendar_sector_features(pd.DataFrame(rows),pd.DataFrame(market_rows))
    sectors=sectors.loc[sectors.date.between(a.start,a.end)]
    sectors.to_parquet(out/'sector_context.parquet',index=False)
    pd.DataFrame(market_rows).to_csv(out/'market_context.csv',index=False)
    latest=sectors.loc[sectors.date.eq(a.end)&sectors.observed.ge(5)].sort_values(
        ['relative_compound_5d','sector'],ascending=[False,True])
    latest.head(30).to_csv(out/'latest_sector_observations.csv',index=False,encoding='utf-8-sig')
    assert all(hashlib.sha256(Path(f).read_bytes()).hexdigest()==digest for f,digest in protected.items())
    assert all((Path(s['path']).stat().st_size,Path(s['path']).stat().st_mtime_ns)==(s['size'],s['mtime_ns']) for s in stamps)
    save(out/'summary.json',dict(production_policy_changed=False,future_labels_read=False,buy_flags_read=False,
        source_and_input_stamps_unchanged=True,stock_rows=len(stocks),sector_rows=len(sectors),
        sectors=sectors.sector.nunique(),dates=sectors.date.nunique(),
        hypothesis='Sector direction is computed from all contemporaneously observable constituents before any stock buy-point filter.',
        profit_improvement_claimed=False))
    state(out,'completed',sector_rows=len(sectors))


if __name__=='__main__':main()
