"""Explain selected-signal score changes using the frozen scoring functions.

Diagnostic only: read sealed sources and signals; never modify strategy inputs.
The sequential split holds old forecasts for the fundamental effect, then holds
new fundamentals for the forecast effect. It explains scores, not causal NAV.
"""
import argparse
import ast
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def frozen_function(root, rel, name, namespace):
    tree = ast.parse((root / rel).read_text())
    node = copy.deepcopy(next(n for n in ast.walk(tree)
                              if isinstance(n, ast.FunctionDef) and n.name == name))
    node.decorator_list = []
    node.body = [n for n in node.body if not isinstance(n, ast.ImportFrom)]
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
                 str(root / rel), 'exec'), namespace)
    return namespace[name]


def scores(root):
    ns = {}
    frozen_function(root, 'strategy/core/factor_calculator.py', 'compute_fundamental_score', ns)
    helper = root / 'strategy/core/fundamental_units.py'
    if helper.exists():
        frozen_function(root, 'strategy/core/fundamental_units.py', 'parse_percent_points', ns)
    return frozen_function(root, 'strategy/core/signal_engine.py', '_compute_fund_score_from_row', ns)


def selected_bom(root, keys):
    path = root / 'strategy/rolling_validation_results/backtest_signals.csv'
    target = pd.MultiIndex.from_frame(keys[['date', 'code']])
    hits = []
    with pd.read_csv(path, usecols=['date', 'code', 'bom_quality_score'],
                     dtype={'code': str}, chunksize=100000) as reader:
        for chunk in reader:
            chunk['date'] = pd.to_datetime(chunk['date']).dt.strftime('%Y-%m-%d')
            chunk['code'] = chunk['code'].str.zfill(6)
            hit = chunk.loc[pd.MultiIndex.from_frame(chunk[['date', 'code']]).isin(target)]
            if not hit.empty:
                hits.append(hit)
    result = pd.concat(hits, ignore_index=True)
    if result.duplicated(['date', 'code']).any() or len(result) != len(keys):
        raise ValueError('Selected BOM coverage differs from attribution keys')
    return result


def blend(base, prediction, weight):
    return (1 - weight) * base + weight * prediction if abs(prediction) > .01 else base


def run(args):
    roots = [args.baseline.resolve(), args.candidate.resolve()]
    funcs = [scores(root) for root in roots]
    configs = [yaml.safe_load((root / 'strategy/config/factor_config.yaml').read_text())
               for root in roots]
    weights = [(cfg.get('fundamental_weight', .3), cfg['ml']['blend_weight']) for cfg in configs]
    if weights[0] != weights[1]:
        raise ValueError('Changed blend weights require a different attribution')
    fw, mw = weights[0]
    panel = pd.read_csv(args.panel, dtype={'code': str})
    if not np.allclose(panel.factor_value_baseline, panel.factor_value_candidate,
                       atol=1e-12, rtol=0, equal_nan=True):
        raise ValueError('Raw factors changed; this attribution isolates score and forecast changes')
    assemblies = []
    for root in roots:
        tree = ast.parse((root / 'strategy/core/signal_engine.py').read_text())
        node = next(n for n in ast.walk(tree)
                    if isinstance(n, ast.FunctionDef) and n.name == '_vectorized_score_assembly')
        assemblies.append(ast.dump(node, include_attributes=False))
    if assemblies[0] != assemblies[1]:
        raise ValueError('Score assembly changed; simple cancellation is invalid')
    for side, root in zip(['baseline', 'candidate'], roots):
        bom = selected_bom(root, panel).rename(columns={'bom_quality_score': 'bom_' + side})
        panel = panel.merge(bom, on=['date', 'code'], validate='one_to_one')
    if not np.allclose(panel.bom_baseline, panel.bom_candidate, atol=1e-12, rtol=0):
        raise ValueError('BOM changed; alternative adjustments cannot cancel')
    data = {}
    for row in panel.itertuples():
        if row.code not in data:
            frame = pd.read_csv(roots[1] / 'data/stock_data/fundamental_data' / (row.code + '.csv'))
            frame['数据可用日期_str'] = frame['数据可用日期'].astype(str)
            data[row.code] = frame.sort_values('数据可用日期_str').reset_index(drop=True)
    rows = []
    for row in panel.itertuples():
        frame = data[row.code]
        available = frame[frame['数据可用日期_str'] <= row.date.replace('-', '')]
        fs = [fn(available.iloc[-1]) if len(available) else 0. for fn in funcs]
        factors = [np.clip(row.factor_value_baseline, -10., 10.),
                   np.clip(row.factor_value_candidate, -10., 10.)]
        base = [factor + max(score, 0.) * fw for factor, score in zip(factors, fs)]
        pred = [row.ml_score_baseline, row.ml_score_candidate]
        bom = .7 + .6 * row.bom_candidate
        fundamental_effect = (blend(base[1], pred[0], mw) - blend(base[0], pred[0], mw)) * bom
        model_effect = (blend(base[1], pred[1], mw) - blend(base[1], pred[0], mw)) * bom
        actual = row.score_candidate - row.score_baseline
        rows.append({'date': row.date, 'code': row.code, 'fund_score_old': fs[0],
                     'fund_score_new': fs[1], 'fundamental_effect': fundamental_effect,
                     'model_effect': model_effect, 'actual_delta': actual,
                     'unexplained': actual - fundamental_effect - model_effect,
                     'ml_branch_changed': (abs(pred[0]) > .01) != (abs(pred[1]) > .01)})
    result = pd.DataFrame(rows)
    summary = {'scope': 'union of actual selections only; score arithmetic, not NAV causality',
               'baseline': str(roots[0]), 'candidate': str(roots[1]),
               'fundamental_weight': fw, 'ml_weight': mw, 'keys': len(result),
               'max_abs_unexplained': float(result.unexplained.abs().max()),
               'fully_explained_within_1e_10': bool((result.unexplained.abs() <= 1e-10).all()),
               'ml_branch_changed': int(result.ml_branch_changed.sum()),
               'mean_abs_fundamental_effect': float(result.fundamental_effect.abs().mean()),
               'mean_abs_model_effect': float(result.model_effect.abs().mean()),
               'first': result.iloc[0].to_dict()}
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output_prefix.with_suffix('.csv'), index=False)
    args.output_prefix.with_suffix('.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['baseline', 'candidate', 'panel', 'output-prefix']:
        p.add_argument('--' + name, type=Path, required=True)
    run(p.parse_args())
