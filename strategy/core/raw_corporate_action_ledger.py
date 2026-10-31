"""Research-only raw-price corporate-action ledger primitive.

Issuer payment date is not necessarily broker spendable time: callers must
supply a separately verified cash-availability date. Amounts are gross until
an independently verified dividend-tax policy is booked.
"""
from dataclasses import dataclass
from datetime import date
from math import isfinite


def _day(value):
    return date.fromisoformat(str(value)[:10])


@dataclass(frozen=True)
class IssuerAction:
    code: str
    record_date: date
    ex_date: date
    bonus_per_share: float
    cash_per_share: float
    cash_payment_date: date
    cash_available_date: date
    bonus_trading_date: date | None

    def __post_init__(self):
        for field in ('record_date', 'ex_date', 'cash_payment_date', 'cash_available_date'):
            object.__setattr__(self, field, _day(getattr(self, field)))
        if self.bonus_trading_date is not None:
            object.__setattr__(self, 'bonus_trading_date', _day(self.bonus_trading_date))
        if (len(self.code) != 6 or not self.code.isdigit() or
                self.record_date >= self.ex_date or
                self.cash_payment_date < self.ex_date or
                self.cash_available_date < self.cash_payment_date):
            raise ValueError('invalid issuer dates or code')
        if (not all(isfinite(x) and x >= 0 for x in
                    (self.bonus_per_share, self.cash_per_share)) or
                (self.bonus_per_share > 0) != (self.bonus_trading_date is not None) or
                (self.bonus_trading_date is not None and
                 self.bonus_trading_date < self.ex_date)):
            raise ValueError('invalid bonus/cash terms or bonus trading date')


class RawCorporateActionLedger:
    """Single-account cash/shares/receivables; orders use unadjusted prices.

    Call capture_at_record_close after that day's fills, then advance(day)
    before the next day's orders. Cash becomes spendable only on its explicit
    availability date, not automatically on the issuer's payment date.
    """

    def __init__(self, cash):
        if not isfinite(cash) or cash < 0:
            raise ValueError('invalid initial cash')
        self.cash = float(cash)
        self.shares = {}
        self.restricted_bonus = {}
        self.cash_receivable = 0.0
        self._entitlements = {}
        self._last_day = None

    def fill(self, code, side, shares, raw_price, fees=0.0):
        if self._last_day is None:
            raise ValueError('advance to the execution date before booking a fill')
        if (len(code) != 6 or not code.isdigit() or
                not isinstance(shares, int) or shares <= 0 or
                not all(isfinite(x) and x >= 0 for x in (raw_price, fees)) or raw_price <= 0):
            raise ValueError('invalid raw fill')
        if side == 'BUY':
            if shares % 100:
                raise ValueError('buy quantity must be a 100-share lot')
            cost = shares * raw_price + fees
            if cost > self.cash + 1e-8:
                raise ValueError('insufficient cash')
            self.cash -= cost
            self.shares[code] = self.shares.get(code, 0) + shares
        elif side == 'SELL':
            if shares > self.shares.get(code, 0):
                raise ValueError('cannot sell untradable or absent shares')
            if shares * raw_price - fees + self.cash < -1e-8:
                raise ValueError('negative cash after sale')
            self.shares[code] -= shares
            self.cash += shares * raw_price - fees
        else:
            raise ValueError('invalid side')

    def capture_at_record_close(self, action):
        key = (action.code, action.ex_date)
        if key in self._entitlements:
            raise ValueError('duplicate entitlement')
        if self._last_day != action.record_date:
            raise ValueError('record date must be advanced before entitlement')
        held = self.shares.get(action.code, 0)
        bonus = held * action.bonus_per_share
        if abs(bonus - round(bonus)) > 1e-8:
            raise ValueError(f'fractional bonus requires issuer rule: {key}')
        self._entitlements[key] = (action, held, set())

    def advance(self, day):
        """Apply ex-date entitlements and settle explicitly available cash."""
        day = _day(day)
        if self._last_day is not None and day <= self._last_day:
            raise ValueError('dates must advance strictly')
        self._last_day = day
        for key, (action, record_shares, applied) in list(self._entitlements.items()):
            if action.ex_date <= day and 'ex' not in applied:
                bonus = record_shares * action.bonus_per_share
                if abs(bonus - round(bonus)) > 1e-8:
                    raise ValueError(f'fractional bonus requires issuer rule: {key}')
                bonus = int(round(bonus))
                self.restricted_bonus[action.code] = self.restricted_bonus.get(action.code, 0) + bonus
                self.cash_receivable += record_shares * action.cash_per_share
                applied.add('ex')
            if action.bonus_trading_date is not None and action.bonus_trading_date <= day and 'bonus' not in applied:
                if 'ex' not in applied:
                    raise ValueError('bonus cannot trade before ex-date')
                bonus = int(round(record_shares * action.bonus_per_share))
                self.restricted_bonus[action.code] -= bonus
                self.shares[action.code] = self.shares.get(action.code, 0) + bonus
                applied.add('bonus')
            if action.cash_available_date <= day and 'cash' not in applied:
                if 'ex' not in applied:
                    raise ValueError('dividend cannot settle before ex-date')
                cash_due = record_shares * action.cash_per_share
                self.cash_receivable -= cash_due
                self.cash += cash_due
                applied.add('cash')
            self._entitlements[key] = (action, record_shares, applied)

    def equity(self, raw_close):
        value = self.cash + self.cash_receivable
        for code in set(self.shares) | set(self.restricted_bonus):
            count = self.shares.get(code, 0) + self.restricted_bonus.get(code, 0)
            if count:
                price = raw_close.get(code)
                if price is None or not isfinite(price) or price <= 0:
                    raise ValueError(f'missing raw close for held symbol: {code}')
                value += count * price
        return value
