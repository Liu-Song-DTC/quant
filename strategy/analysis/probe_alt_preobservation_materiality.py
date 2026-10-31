#!/usr/bin/env python3
"""Read-only first-snapshot proxy signal exposure on archived selections.

This joins *later-observed* supplier snapshots to historical selections only for
risk diagnosis. It cannot reproduce the original run or establish PIT accuracy.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd


def probe(selections, alt_root):
    if not __package__:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from strategy.core.alternative_data import AlternativeDataProvider
    from strategy.core.alt_data_versions import _read_manifest

    alt_root = Path(alt_root)
    selected = pd.read_csv(selections, dtype={'date': str, 'code': str})
    if not {'date', 'code'}.issubset(selected):
        raise ValueError('selections need date/code')
    if selected.duplicated(['date', 'code']).any():
        raise ValueError('duplicate selection date/code')
    selected['code'] = selected.code.str.zfill(6)
    sources = ['northbound_daily', 'margin_daily', 'reduction_records', 'reduction_plans']
    observed = {}
    for stem in sources:
        manifest = alt_root / 'alt_versions' / f'{stem}.json'
        observed[stem] = _read_manifest(manifest)['versions'][0]['observed_on']
    if any(selected.date.ge(first).any() for first in observed.values()):
        raise ValueError('probe expects selections strictly before first observed version')
    # Fail rather than silently downloading or rewriting a mutable supplier cache.
    if os.environ.get('QUANT_ALT_NO_AUTOREFRESH') != '1':
        raise ValueError('set QUANT_ALT_NO_AUTOREFRESH=1 before running this probe')
    provider = AlternativeDataProvider(str(alt_root))
    rows = []
    for day, group in selected.groupby('date', sort=True):
        nb = float(provider.get_northbound_signal(day))
        mg = float(provider.get_margin_signal(day))
        reduction = provider.get_reduction_codes(day)
        plans = provider.get_reduction_plan_codes(day)
        for code in group.code:
            rows.append({'date': day, 'code': code, 'proxy_northbound_signal': nb,
                         'proxy_margin_signal': mg, 'proxy_reduction_excluded': code in reduction,
                         'proxy_reduction_plan_excluded': code in plans,
                         'proxy_reduction_universe_size': len(reduction),
                         'proxy_plan_universe_size': len(plans)})
    detail = pd.DataFrame(rows)
    summary = {
        'first_observed_on': observed,
        'historical_selected_rows': len(detail),
        'northbound_proxy_nonzero': int(detail.proxy_northbound_signal.ne(0).sum()),
        'margin_proxy_nonzero': int(detail.proxy_margin_signal.ne(0).sum()),
        'selected_reduction_proxy_hits': int(detail.proxy_reduction_excluded.sum()),
        'selected_plan_proxy_hits': int(detail.proxy_reduction_plan_excluded.sum()),
        'selected_any_proxy_risk_filter_hit': int((detail.proxy_reduction_excluded |
                                                    detail.proxy_reduction_plan_excluded).sum()),
        'selected_dates_with_nonempty_reduction_universe': int(detail.loc[
            detail.proxy_reduction_universe_size.gt(0), 'date'].nunique()),
        'selected_dates_with_nonempty_plan_universe': int(detail.loc[
            detail.proxy_plan_universe_size.gt(0), 'date'].nunique()),
        'by_year': {str(year): {
            'selected_rows': len(group),
            'northbound_nonzero': int(group.proxy_northbound_signal.ne(0).sum()),
            'margin_nonzero': int(group.proxy_margin_signal.ne(0).sum()),
            'selected_any_risk_filter_hit': int((group.proxy_reduction_excluded |
                                                   group.proxy_reduction_plan_excluded).sum())
        } for year, group in detail.groupby(detail.date.str[:4])},
    }
    return detail, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selections', type=Path, required=True)
    parser.add_argument('--alt-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    # Import-time read-only mode is required by the provider's cache loaders.
    os.environ['QUANT_ALT_NO_AUTOREFRESH'] = '1'
    rows, summary = probe(args.selections, args.alt_root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n')
    rows.to_csv(args.out.with_suffix('.csv'), index=False)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
