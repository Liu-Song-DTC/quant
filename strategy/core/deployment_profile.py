"""Explicit shared trading policy for backtest and live entry points.

The accepted465 profile preserves the sealed forecast domain, including its
documented whole-group fallback. It never enables signal-cache replay.
"""
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
PROFILE = ROOT / 'strategy/config/deployment/accepted465.json'
_FIXED = {
    'QUANT_RESEARCH_FRESH_BUY_PRIORITY': '1',
    'QUANT_RESEARCH_FRESH_BUY_POINTS': '2',
    'QUANT_RESEARCH_ADD_CONFIRMATION': '1',
    'QUANT_B2_CONTEXT_PROFIT_GUARD': '1',
    'QUANT_RESEARCH_PRESERVE_B1_ORIGIN': '1',
    'QUANT_RESEARCH_B1_BLOCK_CONSENSUS': '0',
    'QUANT_RESEARCH_B1_ORIGIN_SCOPE': 'close_rebound_not_open_gap',
    'QUANT_INDUSTRY_PIT': '0',
    'QUANT_ALT_MANIFEST_SNAPSHOT': '1',
}


def accepted465_active():
    return os.environ.get('QUANT_TRADING_PROFILE', '') == 'accepted465'


@lru_cache(maxsize=1)
def _load():
    profile = json.loads(PROFILE.read_text())
    if profile.get('name') != 'accepted465' or profile.get('schema') != 1:
        raise ValueError('Unknown accepted trading profile')
    if profile.get('fixed_policy') != _FIXED:
        raise ValueError('accepted465 policy differs from the accepted recipe')
    table = ROOT / profile['ranking']['path']
    with table.open('rb') as stream:
        actual = hashlib.file_digest(stream, 'sha256').hexdigest()
    if actual != profile['ranking']['sha256']:
        raise ValueError('accepted465 ranking artifact changed')
    return profile


def activate_profile():
    """Install one shared, conflict-checked policy, only by explicit selection."""
    selected = os.environ.get('QUANT_TRADING_PROFILE', '')
    if not selected:
        return None
    if selected != 'accepted465':
        raise ValueError(f'Unknown trading profile: {selected}')
    profile = _load()
    config_bytes = (ROOT/'strategy/config/factor_config.yaml').read_bytes()
    config_bytes = re.sub(rb'^  todate:[^\n]*',b'  todate: <runtime>',config_bytes,flags=re.MULTILINE)
    if hashlib.sha256(config_bytes).hexdigest()!=profile['config_sha256_without_todate']:
        raise ValueError('Trading configuration differs from the accepted465 configuration')
    defaults = {
        'QUANT_RESEARCH_SOFT_STRATEGY_UNITY': ('0',),
        'QUANT_RESEARCH_B4_CONFIRMATION': ('',),
        'QUANT_RESEARCH_B4_RANK': ('',),
        'QUANT_ENTRY_ADMISSION': ('off', ''),
        'QUANT_PRESERVE_CARRY_SHARES': ('0',),
        'QUANT_RAW_EXECUTION': ('0',),
        'QUANT_B2_THESIS_EXIT': ('0',),
        'QUANT_RESEARCH_PROFIT_GUARD_MA_PERIOD': ('20',),
        'QUANT_RESEARCH_PROFIT_GUARD_POINTS': ('2',),
    }
    for key, allowed in defaults.items():
        if key in os.environ and os.environ[key] not in allowed:
            raise ValueError(f'{key} is outside accepted465 policy')
    values = dict(_FIXED,
        QUANT_RESEARCH_CONDITIONAL_ORDER=str(ROOT / profile['ranking']['path']),
        QUANT_RESEARCH_CONDITIONAL_ORDER_SHA256=profile['ranking']['sha256'])
    for key, value in values.items():
        existing = os.environ.get(key)
        if existing is not None and existing != value:
            raise ValueError(f'{key} conflicts with accepted465 policy')
        os.environ[key] = value
    # The peer dataset is refreshed separately with the same classification
    # recipe. The caller may supply a validated, date-covering snapshot.
    os.environ.setdefault('QUANT_B2_PROFIT_CONTEXT', str(ROOT / profile['peer_context']))
    os.environ.setdefault('QUANT_ALT_NO_AUTOREFRESH', '1')
    os.environ.setdefault('QUANT_ALT_VERSIONS_REQUIRED', '1')
    return profile
