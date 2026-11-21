"""Observational provenance of the winning primary buy-point branch.

These IDs are categorical metadata, never ordinal scores. Earlier branches
can suppress later ones; an ID does not enumerate all matching patterns.
BI_1/BI_2 remain separate stroke-level confirmations, not primary subtypes.
"""
from enum import IntEnum


class BuyPointOrigin(IntEnum):
    UNKNOWN = 0
    B1_PIVOT_DIVERGENCE = 101
    B1_DRAWDOWN_VOLUME_REBOUND = 102
    B1_CLOSE_REBOUND = 103
    B2_EXPLICIT_B1_ZONE = 201
    B2_IMPLICIT_FRACTAL_MACD_ZONE = 202
    B3_BREAKOUT_PULLBACK = 301
    B3_PIVOT_CONSOLIDATION_BREAKOUT = 302
    B4_TREND_MA20_PULLBACK = 401
    B5_MA20_TURN_AND_CROSS = 501
    B6_MA20_PULLBACK = 601
    B6_MA60_PULLBACK = 602
    B6_MA20_AND_MA60_PULLBACK = 603
    B7_RANGE_AND_NEAR_SUPPORT = 701
    B7_RANGE_ONLY = 702
    B7_NEAR_SUPPORT_ONLY = 703
    B8_CONTRACTION_ONLY = 801
    B8_TIGHT_RANGE_ONLY = 802
    B8_CONTRACTION_AND_TIGHT_RANGE = 803
    B9_LIMIT_UP_PULLBACK = 901


def origin_label(value):
    """Return a stable human-readable label; old caches have unknown origin."""
    try:
        return BuyPointOrigin(int(value)).name
    except (ValueError, TypeError, OverflowError):
        return 'UNKNOWN'
