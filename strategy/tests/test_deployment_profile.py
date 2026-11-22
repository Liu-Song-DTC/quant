import hashlib
import os

import pandas as pd
import pytest

from core.deployment_profile import activate_profile, accepted465_active, ROOT
from core.conditional_candidate_order import ConditionalCandidateOrder


@pytest.fixture
def clean_policy(monkeypatch):
    original = {key:value for key,value in os.environ.items() if key.startswith('QUANT_')}
    for key in tuple(os.environ):
        if key.startswith('QUANT_'):
            monkeypatch.delenv(key)
    yield
    for key in tuple(os.environ):
        if key.startswith('QUANT_'):
            os.environ.pop(key)
    os.environ.update(original)


def test_profile_is_explicit_and_does_not_enable_sealed_cache_replay(clean_policy, monkeypatch):
    assert activate_profile() is None
    assert not accepted465_active()
    monkeypatch.setenv('QUANT_TRADING_PROFILE', 'accepted465')
    profile = activate_profile()
    assert accepted465_active()
    assert os.environ['QUANT_RESEARCH_FRESH_BUY_POINTS'] == '2'
    assert os.environ['QUANT_RESEARCH_ADD_CONFIRMATION'] == '1'
    assert os.environ['QUANT_B2_CONTEXT_PROFIT_GUARD'] == '1'
    assert os.environ['QUANT_RESEARCH_B1_ORIGIN_SCOPE'] == 'close_rebound_not_open_gap'
    assert 'QUANT_SIGNAL_REPLAY' not in os.environ
    path = ROOT / profile['ranking']['path']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == profile['ranking']['sha256']


@pytest.mark.parametrize('key,value', [
    ('QUANT_RESEARCH_FRESH_BUY_POINTS','all'),
    ('QUANT_RESEARCH_B1_BLOCK_CONSENSUS','1'),
    ('QUANT_RESEARCH_SOFT_STRATEGY_UNITY','1'),
    ('QUANT_RESEARCH_PROFIT_GUARD_MA_PERIOD','5'),
    ('QUANT_RAW_EXECUTION','1'),
])
def test_profile_rejects_recipe_drift(clean_policy, monkeypatch, key, value):
    monkeypatch.setenv('QUANT_TRADING_PROFILE','accepted465')
    monkeypatch.setenv(key,value)
    with pytest.raises(ValueError, match=key):
        activate_profile()


def test_latest_date_preserves_original_whole_group_as_in_accepted_account(clean_policy, monkeypatch):
    monkeypatch.setenv('QUANT_TRADING_PROFILE','accepted465')
    profile = activate_profile()
    table = ConditionalCandidateOrder(ROOT/profile['ranking']['path'],profile['ranking']['sha256'],
        preserve_b1_origin=True,b1_origin_scope='close_rebound_not_open_gap')
    candidates = [dict(code='600001',chan_buy_point=1,effective_score=.9),
                  dict(code='600002',chan_buy_point=1,effective_score=.8)]
    ranked, evidence = table.reorder(candidates,pd.Timestamp('2026-11-20'),{})
    assert ranked == candidates
    # Same safe fallback is part of the accepted historical account, not a
    # newly invented current-day ranking or a missing table silently ignored.
    assert evidence['groups'][0]['reason'] == 'incomplete_group_prediction_coverage'


def test_backtest_strategy_and_live_constructor_activate_the_same_policy(clean_policy, monkeypatch):
    monkeypatch.setenv('QUANT_TRADING_PROFILE','accepted465')
    monkeypatch.setenv('QUANT_ALT_NO_AUTOREFRESH','1')
    from core.strategy import Strategy
    from core.portfolio import PortfolioConstructor
    from core.config_loader import load_config
    simulation = Strategy(init_cash=500000).portfolio
    live = PortfolioConstructor(rebalance_interval=int(load_config().get('backtest.rebalance_days',10)))
    assert type(simulation) is type(live)
    for field in ('_research_fresh_buy_priority','_research_fresh_buy_points',
                  '_research_add_confirmation','score_rank_source','_rebalance_interval'):
        assert getattr(simulation,field)==getattr(live,field)
    assert simulation._conditional_candidate_order.sha256==live._conditional_candidate_order.sha256
    assert simulation._conditional_candidate_order.b1_origin_scope==live._conditional_candidate_order.b1_origin_scope
