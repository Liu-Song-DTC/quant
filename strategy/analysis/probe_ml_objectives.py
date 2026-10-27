#!/usr/bin/env python3
"""Controlled research probe: rank, net return, win probability and downside.

All heads share features, purged intervals, model capacity and seeds. Evaluations
are on a deterministic stock subsample, NOT a tradable portfolio backtest or
untouched out-of-sample evidence. Probability outputs are explicitly uncalibrated.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core.forward_targets import short_net_targets
from core.ml_predictor import MLFactorPredictor
from core.input_identity import file_content_sha256
from analysis.ml_combination_rules import (fit_probability_calibration,calibrated_probability,
                                           choose_candidates,selected_metrics)


def attach_feature_panel(frame, features, panel_path, columns, data_sha256, cache_sha256):
    """Join declared event features without changing candidate rows or order."""
    panel_path = Path(panel_path)
    manifest_path = panel_path.with_name('report.json')
    metadata = json.loads(manifest_path.read_text())
    if (metadata['data_sha256'] != data_sha256 or metadata['factor_cache_sha256'] != cache_sha256
            or metadata['features_sha256'] != file_content_sha256(panel_path)):
        raise ValueError('event panel does not match the sealed experiment inputs')
    if not columns or len(columns) != len(set(columns)) or any(not c.startswith('fresh_') for c in columns):
        raise ValueError('explicit unique fresh_ feature columns required')
    if set(columns) & set(frame.columns):
        raise ValueError('event features overlap existing inputs')
    panel = pd.read_parquet(panel_path, columns=['code', 'date', *columns])
    if len(panel) != len(frame) or panel.duplicated(['code', 'date']).any():
        raise ValueError('event panel changed the candidate universe')
    if any(not pd.api.types.is_numeric_dtype(panel[c]) for c in columns):
        raise ValueError('event features must be numeric')
    joined = frame.merge(panel, on=['code', 'date'], how='left', sort=False,
                         validate='one_to_one', indicator='_panel_join')
    if not joined['_panel_join'].eq('both').all():
        raise ValueError('event panel has missing candidate keys')
    pd.testing.assert_frame_equal(frame[['code', 'date']].reset_index(drop=True),
                                  joined[['code', 'date']].reset_index(drop=True))
    return joined.drop(columns='_panel_join'), features + list(columns), [panel_path, manifest_path]


def purged_split(frame, test_start, test_end, validation_days=120, training_days=750):
    start,end=pd.Timestamp(test_start),pd.Timestamp(test_end)
    if end<=start or validation_days<=0 or training_days<=validation_days:
        raise ValueError('invalid chronological evaluation intervals')
    val_start=start-pd.Timedelta(days=validation_days)
    dates=pd.to_datetime(frame['date']);ends=pd.to_datetime(frame['label_end_date'])
    fit=(dates>=start-pd.Timedelta(days=training_days))&(dates<val_start)&(ends<val_start)
    validation=(dates>=val_start)&(dates<start)&(ends<start)
    test=(dates>=start)&(dates<end)
    return fit.to_numpy(),validation.to_numpy(),test.to_numpy()


def purged_four_split(frame,test_start,test_end):
    fit,_,test=purged_split(frame,test_start,test_end,validation_days=180)
    dates=pd.to_datetime(frame['date']);ends=pd.to_datetime(frame['label_end_date'])
    val_start=pd.Timestamp(test_start)-pd.Timedelta(days=180)
    cal_start=pd.Timestamp(test_start)-pd.Timedelta(days=90)
    validation=((dates>=val_start)&(dates<cal_start)&(ends<cal_start)).to_numpy()
    calibration=((dates>=cal_start)&(dates<pd.Timestamp(test_start))&(ends<pd.Timestamp(test_start))).to_numpy()
    return fit,validation,calibration,test


def summarize(predictions, values, days, higher_is_better=True):
    rows=pd.DataFrame({'date':days,'prediction':predictions,'net_return':values})
    rows=rows.dropna(subset=['date','prediction'])
    chosen=[];ics=[]
    for _,group in rows.groupby('date'):
        chosen.append(group.sort_values('prediction',ascending=not higher_is_better,kind='stable').head(5))
        known=group.dropna(subset=['net_return'])
        if len(known)>=10 and known['prediction'].nunique()>1:
            ic=known['prediction'].corr(known['net_return'],method='spearman')
            if np.isfinite(ic):ics.append(float(ic))
    if not chosen:
        raise ValueError('empty objective evaluation')
    top=pd.concat(chosen)
    filled=top.dropna(subset=['net_return']);known=rows.dropna(subset=['net_return'])
    return {'days':int(rows['date'].nunique()),'samples':len(rows),'top5_samples':len(top),
            'universe_win_rate':float((known.net_return>0).mean()) if len(known) else None,
            'universe_mean_net_return':float(known.net_return.mean()) if len(known) else None,
            'top5_win_rate':float((filled.net_return>0).mean()) if len(filled) else None,
            'top5_mean_net_return':float(filled.net_return.mean()) if len(filled) else None,
            'top5_entry_unavailable':int(top.net_return.isna().sum()),
            'top5_mean_net_return_per_slot':float(top.net_return.fillna(0).sum()/(rows['date'].nunique()*5)),
            'top5_mean_downside':float(np.maximum(-top.net_return.fillna(0),0).mean()),
            'daily_rank_ic':float(np.mean(ics)) if ics else None}


def run(run_dir,output,stocks=256,combinations=False,feature_panel=None,feature_columns=None,
        rank_method='regression'):
    from xgboost import XGBRegressor, XGBRanker
    import xgboost,scipy,polars
    run_dir,output=Path(run_dir),Path(output)
    if rank_method not in ['regression','ndcg_top5']:
        raise ValueError('unknown rank learning method')
    if not isinstance(stocks,int) or stocks<20:raise ValueError('at least 20 sampled stocks required')
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True)
    root=run_dir/'strategy'
    cache=list((root/'cache').glob('factor_df_*.parquet'))
    if len(cache)!=1:raise ValueError('exactly one archived factor cache required')
    config=yaml.safe_load((root/'config/factor_config.yaml').read_text())
    names=pq.read_schema(cache[0]).names
    features=[c for c in config['ml']['feature_whitelist'] if c in names]
    paths={p.name[:-8]:str(p) for p in (run_dir/'data/stock_data/backtrader_data').glob('*_qfq.csv')
           if p.name!='sh000001_qfq.csv'}
    codes=sorted(paths,key=lambda c:hashlib.sha256(c.encode()).hexdigest())[:stocks]
    input_paths=[cache[0],run_dir/'data/stock_data/backtrader_data/sh000001_qfq.csv',
                 *[Path(paths[c]) for c in codes]]
    if (feature_panel is None) != (not feature_columns):
        raise ValueError('feature panel and explicit feature columns must be supplied together')
    if feature_panel is not None:
        input_paths.extend([Path(feature_panel),Path(feature_panel).with_name('report.json')])
    before={str(path):file_content_sha256(path) for path in input_paths}
    frame=pd.read_parquet(cache[0],columns=['code','date',*features],filters=[('code','in',codes)])
    if frame.duplicated(['code','date']).any():raise ValueError('duplicate factor rows')
    frame=frame.sort_values(['date','code'],kind='stable').reset_index(drop=True)
    if feature_panel is not None:
        frame,features,_=attach_feature_panel(
            frame,features,feature_panel,feature_columns,
            json.loads((run_dir/'manifest.json').read_text())['data_sha256'],file_content_sha256(cache[0]))
        config['ml']['feature_whitelist']=list(features)
    calendar=pd.read_csv(run_dir/'data/stock_data/backtrader_data/sh000001_qfq.csv',usecols=['datetime'])['datetime']
    frame,details=short_net_targets(frame,{c:paths[c] for c in codes},calendar,return_details=True)
    raw=frame['future_ret'].to_numpy(dtype=float)
    rank=frame.groupby('date')['future_ret'].rank(pct=True).to_numpy()-.5
    windows=[('2022-01-01','2023-01-01'),('2024-01-01','2025-01-01'),('2026-01-01','2027-01-01')]
    report={'scope':'deterministic stock subsample; raw factor cache; no industry neutralization or production pool mask',
            'data_sha256':json.loads((run_dir/'manifest.json').read_text())['data_sha256'],
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'factor_cache_sha256':file_content_sha256(cache[0]),
            'dependency_sha256':{name:file_content_sha256(Path(__file__).resolve().parents[1]/'core'/name)
                                 for name in ['ml_predictor.py','forward_targets.py']},
            'stocks':codes,'features':features,'seeds':[42,123,777],
            'feature_panel_columns':list(feature_columns or []),
            'rank_method':rank_method,
            'versions':{'python':sys.version,'numpy':np.__version__,'pandas':pd.__version__,
                        'xgboost':xgboost.__version__,'scipy':scipy.__version__,'polars':polars.__version__},
            'input_sha256':before,'combinations':combinations,
            'label':'average next-open-entry 3/5 day net marked return; blocked exits retained',
            'limitations':['No portfolio weights, capacity or actual exit execution.',
                           'Entry-limit label checks are approximate; full ST/IPO/board-date execution parity is not established.',
                           'Historical data already used for research; not untouched OOS.',
                           'Standalone probability head uncalibrated; combination has a separate prior calibration interval.',
                           'Subsample ranks and raw-factor distributions differ from production.'],
            'windows':[]}
    params=dict(config['ml']);common={k:params[k] for k in ['max_depth','learning_rate','subsample','colsample_bytree','reg_lambda','reg_alpha','min_child_weight']}
    common.update(n_estimators=80,n_jobs=1,tree_method='hist',verbosity=0,early_stopping_rounds=12)
    for start,end in windows:
        calibration=None
        if combinations:
            fit,val,calibration,test=purged_four_split(frame,start,end)
            calibration=calibration & np.isfinite(raw)
        else:
            fit,val,test=purged_split(frame,start,end)
        # pandas 3 can expose read-only boolean arrays; keep split masks immutable.
        fit=fit & np.isfinite(raw);val=val & np.isfinite(raw)
        # Censor only globally immature reporting dates. Entry availability must
        # never filter the prediction universe before stocks are selected.
        last_evaluable=pd.DatetimeIndex(calendar).sort_values().unique()[-6]
        test=test & (pd.to_datetime(frame['date'])<=last_evaluable).to_numpy()
        if min(fit.sum(),val.sum(),test.sum())<100:raise ValueError('insufficient objective split samples')
        encoder=MLFactorPredictor(config)
        encoded=encoder._cross_sectional_zscore(frame.copy(),features)
        encoded=encoder.prepare_features(encoded,is_train=True,selection_df=encoded.loc[fit])
        columns=encoder.feature_cols
        matrix=encoded[columns].replace([np.inf,-np.inf],np.nan).fillna(0).to_numpy(dtype=np.float32)
        year={'start':start,'end':end,'fit_rows':int(fit.sum()),'validation_rows':int(val.sum()),
              'encoded_features':columns,
              'test_rows':int(test.sum()),'fit_label_end_max':str(frame.loc[fit,'label_end_date'].max()),
              'validation_label_end_max':str(frame.loc[val,'label_end_date'].max()),'heads':{}}
        if calibration is not None:
            if calibration.sum()<100:raise ValueError('insufficient independent calibration rows')
            year.update(calibration_rows=int(calibration.sum()),
                        calibration_label_end_max=str(frame.loc[calibration,'label_end_date'].max()))
        days_old=(frame.loc[fit,'date'].max()-frame.loc[fit,'date']).dt.days.to_numpy()
        test_frame=frame.loc[test,['date','code']].copy()
        test_frame['net_return']=raw[test]
        for horizon in [3,5]:test_frame[f'net_return_{horizon}']=details.loc[test,f'net_return_{horizon}'].to_numpy()
        calibration_prob=None
        for name,target,objective in [
            ('rank',rank,'reg:squarederror'),('net_return',raw,'reg:squarederror'),
            ('win_probability',(raw>0).astype(float),'binary:logistic'),
            ('downside',np.maximum(-raw,0),'reg:squarederror')]:
            predictions=np.zeros(test.sum())
            cal_predictions=np.zeros(calibration.sum()) if calibration is not None else None
            model_records=[]
            for seed in report['seeds']:
                if name=='rank' and rank_method=='ndcg_top5':
                    # NDCG requires nonnegative relevance grades, not signed
                    # return labels. Fixed deciles are derived within each
                    # completed training/validation day; test labels are unused.
                    grade=np.clip(np.floor((rank+.5)*10),0,9)
                    fit_dates=frame.loc[fit,'date']
                    val_dates=frame.loc[val,'date']
                    train_groups=fit_dates.groupby(fit_dates,sort=False).size().to_numpy()
                    val_groups=val_dates.groupby(val_dates,sort=False).size().to_numpy()
                    group_days=(fit_dates.max()-pd.DatetimeIndex(fit_dates.unique())).days.to_numpy()
                    model=XGBRanker(**common,objective='rank:ndcg',eval_metric='ndcg@5',
                        random_state=seed,lambdarank_pair_method='topk',lambdarank_num_pair_per_sample=8)
                    model.fit(matrix[fit],grade[fit],group=train_groups,
                        sample_weight=MLFactorPredictor.time_weights(group_days),
                        eval_set=[(matrix[val],grade[val])],eval_group=[val_groups],verbose=False)
                else:
                    model=XGBRegressor(**common,objective=objective,random_state=seed)
                    model.fit(matrix[fit],target[fit],sample_weight=MLFactorPredictor.time_weights(days_old),
                              eval_set=[(matrix[val],target[val])],verbose=False)
                predictions+=model.predict(matrix[test])/3
                if cal_predictions is not None:cal_predictions+=model.predict(matrix[calibration])/3
                model_dir=output/'models'/start[:4];model_dir.mkdir(parents=True,exist_ok=True)
                model_path=model_dir/f'{name}_{seed}.json';model.save_model(model_path)
                model_records.append({'seed':seed,'best_iteration':int(model.best_iteration),
                                      'file':str(model_path.relative_to(output)),
                                      'sha256':file_content_sha256(model_path)})
            if name=='downside':predictions=np.maximum(predictions,0)
            test_frame[name+'_prediction']=predictions
            result=summarize(predictions,raw[test],frame.loc[test,'date'].to_numpy(),name!='downside')
            result['models']=model_records
            if name=='win_probability':
                known=np.isfinite(raw[test])
                result['uncalibrated_brier']=float(np.mean((predictions[known]-(raw[test][known]>0))**2))
                result['fit_prior_brier']=float(np.mean(((raw[fit]>0).mean()-(raw[test][known]>0))**2))
                calibration_prob=cal_predictions
            year['heads'][name]=result
            print(start,name,json.dumps(result),flush=True)
        if combinations:
            state=fit_probability_calibration(calibration_prob,(raw[calibration]>0).astype(float))
            test_frame['calibrated_win_probability']=calibrated_probability(
                test_frame.win_probability_prediction.to_numpy(),state)
            year['probability_calibration']=state
            year['recipes']={}
            for recipe in ['rank','rank_then_risk','rank_then_probability']:
                chosen=choose_candidates(test_frame,recipe)
                result=selected_metrics(chosen,int(test_frame.date.nunique()))
                year['recipes'][recipe]=result
                print(start,recipe,json.dumps(result),flush=True)
        test_frame.to_csv(output/(start[:4]+'_predictions.csv'),index=False)
        report['windows'].append(year)
        (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False))
    after={str(path):file_content_sha256(path) for path in input_paths}
    report['input_unchanged']=before==after
    (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False))
    if before!=after:raise RuntimeError('objective probe inputs changed during evaluation')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--stocks',type=int,default=256)
    parser.add_argument('--combinations',action='store_true')
    parser.add_argument('--feature-panel',type=Path)
    parser.add_argument('--feature-columns',nargs='+')
    parser.add_argument('--rank-method',choices=['regression','ndcg_top5'],default='regression')
    args=parser.parse_args();run(args.run,args.output,args.stocks,args.combinations,args.feature_panel,args.feature_columns,args.rank_method)
