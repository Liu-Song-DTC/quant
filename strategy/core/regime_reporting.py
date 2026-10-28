"""Report executed portfolio exposure separately from target book exposure."""

import numpy as np
import pandas as pd


def regime_state_frame(calendar, targets, executed_exposure, executed_n_pos):
    n = len(calendar)
    if len(targets) != n or len(executed_exposure) != n or len(executed_n_pos) != n:
        raise ValueError('regime report dates, targets and actual positions must align')
    actual = np.asarray(executed_exposure, dtype=float)
    counts = np.asarray(executed_n_pos, dtype=int)
    if not np.isfinite(actual).all() or (actual < -1e-9).any() or (actual > 1 + 1e-6).any() or (counts < 0).any():
        raise ValueError('backtest has invalid executed exposure or positions')
    return pd.DataFrame({
        'date': [pd.Timestamp(day).strftime('%Y-%m-%d') for day in calendar],
        'regime': [record[0] for record in targets],
        'n_pos': counts,
        'exposure': actual,
        'target_n_pos': [record[1] for record in targets],
        'target_exposure': [record[2] for record in targets],
    })
