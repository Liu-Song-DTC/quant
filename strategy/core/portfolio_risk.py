"""Project the final account target onto explicit risk limits, retaining cash."""
import math


def constrain_targets(targets, equity, limits, industries=None, scores=None):
    if not math.isfinite(equity) or equity <= 0:
        raise ValueError('account equity must be finite and positive')
    book = {}
    for code, value in targets.items():
        if not math.isfinite(value) or value < 0:
            raise ValueError(f'invalid target value: {code}')
        book[code] = float(value)
    if not limits.get('enabled', False):
        return book
    industries, scores = industries or {}, scores or {}
    gross = float(limits.get('max_gross', .98))
    single = float(limits.get('max_single_weight', 1.0))
    industry = float(limits.get('max_industry_weight', 1.0))
    count = int(limits.get('max_holdings', 0))
    if not all(0 < x <= 1 for x in (gross, single, industry)) or count < 0:
        raise ValueError('invalid final account risk limits')
    if count:
        chosen = set(sorted((c for c in book if book[c] > 0),
                            key=lambda c: (-scores.get(c, 0), c))[:count])
        book = {c: v if c in chosen else 0.0 for c, v in book.items()}
    book = {c: min(v, equity * single) for c, v in book.items()}
    groups = {}
    for code in book:
        groups.setdefault(industries.get(code) or f'unknown:{code}', []).append(code)
    for codes in groups.values():
        total = sum(book[c] for c in codes)
        scale = min(1.0, equity * industry / total) if total else 1.0
        for code in codes:
            book[code] *= scale
    total = sum(book.values())
    scale = min(1.0, equity * gross / total) if total else 1.0
    return {c: v * scale for c, v in book.items()}
