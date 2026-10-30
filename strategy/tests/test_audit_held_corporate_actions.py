"""Event-day ownership is determined before the opening auction fills."""
import pytest
import pandas as pd

from strategy.analysis.audit_held_corporate_actions import audit


def test_ex_date_captures_previous_close_holder_before_sell():
    fills = pd.DataFrame([
        {'date': '2026-05-27', 'code': '000001', 'side': 'BUY', 'shares': 100},
        {'date': '2026-05-28', 'code': '000001', 'side': 'SELL', 'shares': 100},
        {'date': '2026-05-28', 'code': '000002', 'side': 'BUY', 'shares': 100},
    ])
    events = pd.DataFrame([
        {'ex_date': '2026-05-28', 'code': '000001', 's': 0.4, 'D': 0.3,
         'mult': 1.4, 'src': 'em'},
        {'ex_date': '2026-05-28', 'code': '000002', 's': float('nan'), 'D': float('nan'),
         'mult': 1.2, 'src': 'sina'},
        {'ex_date': '2026-05-29', 'code': '000002', 's': float('nan'), 'D': float('nan'),
         'mult': 1.2, 'src': 'sina'},
    ])
    detail, summary = audit(fills, events, '2026-05-29')
    assert list(detail.code) == ['000001', '000002']
    assert list(detail.ex_date) == ['2026-05-28', '2026-05-29']
    assert detail.synthetic_qfq_shares_before_open.tolist() == [100, 100]
    assert summary['missing_cash_or_share_terms'] == 1
    assert summary['by_source'] == {'em': 1, 'sina': 1}


def test_invalid_sell_does_not_silently_create_a_position():
    fills = pd.DataFrame([{'date': '2026-05-28', 'code': '000001',
                           'side': 'SELL', 'shares': 100}])
    events = pd.DataFrame(columns=['ex_date', 'code', 's', 'D', 'mult', 'src'])
    with pytest.raises(ValueError, match='negative simulated holding'):
        audit(fills, events, '2026-05-28')
