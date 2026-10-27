"""Predeclared research recipes; no production position rules are enabled."""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit


def fit_probability_calibration(probabilities, outcomes):
    """Monotone Platt calibration on a separate, already completed interval."""
    p=np.asarray(probabilities,dtype=float);y=np.asarray(outcomes,dtype=float)
    if len(p)!=len(y) or not len(p) or not np.isfinite(p).all() or not np.isfinite(y).all():
        raise ValueError('finite aligned calibration samples required')
    if not np.isin(y,[0.,1.]).all() or ((p<0)|(p>1)).any():
        raise ValueError('invalid probability calibration values')
    prior=float(y.mean())
    if len(np.unique(y))<2:
        return {'method':'constant_prior','prior':prior,'fitted':False,'reason':'single_class'}
    x=np.log(np.clip(p,1e-6,1-1e-6)/np.clip(1-p,1e-6,1-1e-6))
    def loss(parameters):
        z=np.exp(parameters[0])*x+parameters[1]
        return float(np.mean(np.logaddexp(0,z)-y*z)+1e-4*parameters[0]**2)
    fit=minimize(loss,[0.,0.],method='L-BFGS-B',bounds=[(-3.,3.),(-10.,10.)])
    if not fit.success:
        return {'method':'constant_prior','prior':prior,'fitted':False,'reason':'optimizer_failure'}
    return {'method':'monotone_platt','log_slope':float(fit.x[0]),
            'intercept':float(fit.x[1]),'fitted':True,'calibration_rows':len(y)}


def calibrated_probability(probabilities,state):
    p=np.asarray(probabilities,dtype=float)
    if state['method']=='constant_prior':return np.full(len(p),state['prior'])
    x=np.log(np.clip(p,1e-6,1-1e-6)/np.clip(1-p,1e-6,1-1e-6))
    return expit(np.exp(state['log_slope'])*x+state['intercept'])


def choose_candidates(frame,recipe,topk=5,candidate_multiple=4):
    """Selection reads predictions only. Unknown outcomes remain eligible.

    Fixed recipes are declared before evaluation: rank top 20 then lowest risk
    five; or rank top 20 then calibrated probability >= .5, up to five. Empty
    slots are cash. These rules are hypotheses, not selected optimal parameters.
    """
    if recipe not in ['rank','rank_then_risk','rank_then_probability']:
        raise ValueError('unknown combination recipe')
    selected=[]
    for _,group in frame.groupby('date',sort=True):
        candidates=group.sort_values(['rank_prediction','code'],ascending=[False,True],kind='stable')
        if recipe!='rank':candidates=candidates.head(topk*candidate_multiple)
        if recipe=='rank_then_risk':
            candidates=candidates.sort_values(['downside_prediction','rank_prediction','code'],
                                               ascending=[True,False,True],kind='stable')
        if recipe=='rank_then_probability':
            candidates=candidates[candidates['calibrated_win_probability']>=.5]
            candidates=candidates.sort_values(['calibrated_win_probability','rank_prediction','code'],
                                               ascending=[False,False,True],kind='stable')
        selected.append(candidates.head(topk))
    return pd.concat(selected) if selected else frame.iloc[:0]


def selected_metrics(selected,decision_days,topk=5):
    slots=decision_days*topk
    if slots<1:raise ValueError('positive decision slots required')
    out={'decision_days':int(decision_days),'available_slots':int(slots),
         'selected_candidates':len(selected),'selected_entry_unavailable':int(selected.net_return.isna().sum()),
         'filled_candidates':int(selected.net_return.notna().sum()),
         'filled_slot_fraction':float(selected.net_return.notna().sum()/slots),
         'mean_net_return_per_slot':float(selected.net_return.fillna(0).sum()/slots),
         'mean_downside_per_slot':float(np.maximum(-selected.net_return.fillna(0),0).sum()/slots)}
    filled=selected.dropna(subset=['net_return'])
    out['filled_win_rate']=float((filled.net_return>0).mean()) if len(filled) else None
    out['filled_mean_net_return']=float(filled.net_return.mean()) if len(filled) else None
    for horizon in [3,5]:
        column=f'net_return_{horizon}'
        if column in selected:
            values=selected[column].dropna()
            out[f'win_rate_{horizon}']=float((values>0).mean()) if len(values) else None
            out[f'mean_net_return_per_slot_{horizon}']=float(selected[column].fillna(0).sum()/slots)
    return out
