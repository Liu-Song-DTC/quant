#!/usr/bin/env python3
"""Audit saved forward predictions without refitting models or tuning thresholds."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata


def probability_metrics(frame, column):
    probabilities = frame[column].to_numpy(dtype=float)
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError('invalid saved probabilities')
    known = frame.net_return.notna().to_numpy()
    p = probabilities[known]
    y = (frame.loc[known, 'net_return'].to_numpy() > 0).astype(float)
    positives, negatives = int(y.sum()), int((1-y).sum())
    auc = None
    if positives and negatives:
        auc = float((rankdata(p)[y == 1].sum() - positives*(positives+1)/2)/(positives*negatives))
    bins = []
    for low in np.arange(0., 1., .1):
        high = low + .1
        mask = (p >= low) & ((p < high) if high < .999 else (p <= 1.))
        if mask.any():
            bins.append({'lower': float(low), 'upper': float(high), 'count': int(mask.sum()),
                         'mean_probability': float(p[mask].mean()), 'observed_win_rate': float(y[mask].mean())})
    ece = sum(row['count']*abs(row['mean_probability']-row['observed_win_rate']) for row in bins)/len(y) if len(y) else None
    return {'rows': len(frame), 'evaluable_rows': len(y),
            'observed_win_rate': float(y.mean()) if len(y) else None,
            'probability_quantiles': dict(zip(['min', 'p10', 'p50', 'p90', 'max'],
                                            map(float, np.quantile(probabilities, [0., .1, .5, .9, 1.])))),
            'fraction_probability_at_least_half': float((probabilities >= .5).mean()),
            'auc': auc, 'brier': float(np.mean((p-y)**2)) if len(y) else None,
            'expected_calibration_error_10_bins': ece, 'bins': bins}


def diagnose(source, output):
    if output.exists():
        raise FileExistsError(output)
    report_path = source / 'report.json'
    report = json.loads(report_path.read_text())
    files = [report_path] + [source / (window['start'][:4] + '_predictions.csv') for window in report['windows']]
    hashes = {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    result = {'input_sha256': hashes, 'windows': [],
              'limitations': ['Post-hoc diagnosis of previously researched history, not a new validation set.',
                              'Overlapping labels and common market moves invalidate independent-row significance claims.',
                              'No threshold or model selection is performed. Entry-unavailable rows are excluded only from outcome diagnostics.']}
    for window in report['windows']:
        frame = pd.read_csv(source / (window['start'][:4] + '_predictions.csv'), dtype={'code': str})
        frame['date'] = pd.to_datetime(frame.date)
        top20 = frame.sort_values(['date', 'rank_prediction', 'code'], ascending=[True, False, True], kind='stable').groupby('date', sort=False).head(20)
        item = {'year': window['start'][:4], 'best_iterations':
                {name: [model['best_iteration'] for model in head['models']] for name, head in window['heads'].items()},
                'all_candidates': {}, 'rank_top20': {}, 'monthly_top20': {}}
        for name, subset in [('all_candidates', frame), ('rank_top20', top20)]:
            for column in ['win_probability_prediction', 'calibrated_win_probability']:
                item[name][column] = probability_metrics(subset, column)
        for month, group in top20.groupby(top20.date.dt.strftime('%Y-%m')):
            item['monthly_top20'][month] = probability_metrics(group, 'calibrated_win_probability')
        result['windows'].append(item)
    if hashes != {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}:
        raise RuntimeError('saved experiment changed during diagnosis')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = diagnose(args.source.resolve(), args.output.resolve())
    for window in result['windows']:
        print(window['year'], json.dumps(window['rank_top20'], ensure_ascii=False))
