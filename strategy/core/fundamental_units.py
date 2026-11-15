"""Percentage points to ratios, preserving legacy missing/error semantics."""


def parse_percent_points(value):
    """Divide every successfully parsed percentage by 100; propagate NaN/inf.

    Keep legacy strip('%') and ValueError/TypeError boundaries. Callers with
    broader exception handling retain their existing outer handlers.
    """
    if value is None:
        return None
    try:
        return float(value.strip('%') if isinstance(value, str) else value) / 100.0
    except (ValueError, TypeError):
        return None
