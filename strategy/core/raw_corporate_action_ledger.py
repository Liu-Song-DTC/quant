"""Research-only raw-price corporate-action ledger with FIFO dividend tax."""
from calendar import monthrange
from dataclasses import dataclass
from datetime import date, timedelta
from math import isfinite


def _day(value):
    return date.fromisoformat(str(value)[:10])


def _add_months(value, months):
    year, month = divmod(value.year * 12 + value.month - 1 + months, 12)
    return date(year, month + 1, min(value.day, monthrange(year, month + 1)[1]))


def dividend_tax_rate(acquired, sold):
    held_through = _day(sold) - timedelta(days=1)
    acquired = _day(acquired)
    if held_through < acquired:
        raise ValueError('sale precedes lot acquisition')
    if held_through > _add_months(acquired, 12):
        return 0.0
    if held_through > _add_months(acquired, 1):
        return 0.10
    return 0.20


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


@dataclass
class _TaxLot:
    acquired: date
    shares: int
    dividend_per_share: float = 0.0
    bonus_origin: bool = False


class RawCorporateActionLedger:
    """Cash, tradable shares, receivables, restricted bonus and FIFO tax lots."""

    def __init__(self, cash):
        if not isfinite(cash) or cash < 0:
            raise ValueError('invalid initial cash')
        self.cash = float(cash)
        self.shares = {}
        self.restricted_bonus = {}
        self.cash_receivable = 0.0
        self._entitlements = {}
        self._lots = {}
        self._last_day = None
        self.action_log = []
        self.tax_log = []

    def fill(self, code, side, shares, raw_price, fees=0.0):
        if self._last_day is None:
            raise ValueError('advance to the execution date before booking a fill')
        if (len(code) != 6 or not code.isdigit() or
                not isinstance(shares, int) or shares <= 0 or
                not all(isfinite(x) and x >= 0 for x in (raw_price, fees)) or raw_price <= 0):
            raise ValueError('invalid raw fill')
        dividend_tax = 0.0
        if side == 'BUY':
            from .trading_lots import valid_buy_shares
            if not valid_buy_shares(code, shares):
                raise ValueError('buy quantity violates board minimum or lot size')
            cost = shares * raw_price + fees
            if cost > self.cash + 1e-8:
                raise ValueError('insufficient cash')
            self.cash -= cost
            self.shares[code] = self.shares.get(code, 0) + shares
            self._lots.setdefault(code, []).append(_TaxLot(self._last_day, shares))
        elif side == 'SELL':
            if shares > self.shares.get(code, 0):
                raise ValueError('cannot sell untradable or absent shares')
            lots = self._lots.get(code, [])
            if shares > sum(lot.shares for lot in lots):
                raise ValueError('tradable shares have no FIFO tax lots')
            remaining = shares
            for lot in lots:
                taken = min(remaining, lot.shares)
                if taken:
                    gross_dividend = taken * lot.dividend_per_share
                    rate = dividend_tax_rate(lot.acquired, self._last_day) if gross_dividend else 0.0
                    tax_part = gross_dividend * rate
                    dividend_tax += tax_part
                    if gross_dividend:
                        self.tax_log.append({
                            'date': self._last_day, 'code': code,
                            'acquired': lot.acquired, 'shares': taken,
                            'gross_dividend': gross_dividend,
                            'rate': rate, 'tax': tax_part,
                        })
                    lot.shares -= taken
                    remaining -= taken
                if not remaining:
                    break
            self._lots[code] = [lot for lot in lots if lot.shares]
            proceeds = shares * raw_price - fees - dividend_tax
            if proceeds + self.cash < -1e-8:
                raise ValueError('negative cash after sale and dividend tax')
            self.shares[code] -= shares
            self.cash += proceeds
        else:
            raise ValueError('invalid side')
        return dividend_tax

    def capture_at_record_close(self, action):
        key = (action.code, action.ex_date)
        if key in self._entitlements:
            raise ValueError('duplicate entitlement')
        if self._last_day != action.record_date:
            raise ValueError('record date must be advanced before entitlement')
        lots = self._lots.get(action.code, [])
        held = self.shares.get(action.code, 0)
        if held != sum(lot.shares for lot in lots):
            raise ValueError('share and tax-lot ledgers diverge')
        if held and any(lot.bonus_origin for lot in lots):
            raise ValueError('subsequent entitlement on bonus lot needs verified holding-period rule')
        bonus_lots = []
        for lot in lots:
            bonus = lot.shares * action.bonus_per_share
            if abs(bonus - round(bonus)) > 1e-8:
                raise ValueError(f'fractional bonus requires issuer lot-allocation rule: {key}')
            if bonus:
                bonus_lots.append((lot.acquired, int(round(bonus))))
            lot.dividend_per_share += action.cash_per_share
        bonus = held * action.bonus_per_share
        if abs(bonus - round(bonus)) > 1e-8:
            raise ValueError(f'fractional bonus requires issuer rule: {key}')
        self._entitlements[key] = (action, held, set(), bonus_lots)
        return {'record_shares': held, 'bonus_shares': int(round(bonus)),
                'cash_dividend': held * action.cash_per_share}

    def advance(self, day):
        day = _day(day)
        if self._last_day is not None and day <= self._last_day:
            raise ValueError('dates must advance strictly')
        self._last_day = day
        booked = []
        for key, (action, record_shares, applied, bonus_lots) in list(self._entitlements.items()):
            if action.ex_date <= day and 'ex' not in applied:
                bonus = int(round(record_shares * action.bonus_per_share))
                cash_dividend = record_shares * action.cash_per_share
                self.restricted_bonus[action.code] = self.restricted_bonus.get(action.code, 0) + bonus
                self.cash_receivable += cash_dividend
                applied.add('ex')
                booked.append({'date': day, 'code': action.code, 'event': 'ex_entitlement',
                               'record_shares': record_shares, 'bonus_shares': bonus,
                               'cash_amount': cash_dividend})
            if action.bonus_trading_date is not None and action.bonus_trading_date <= day and 'bonus' not in applied:
                if 'ex' not in applied:
                    raise ValueError('bonus cannot trade before ex-date')
                bonus = int(round(record_shares * action.bonus_per_share))
                self.restricted_bonus[action.code] -= bonus
                self.shares[action.code] = self.shares.get(action.code, 0) + bonus
                self._lots.setdefault(action.code, []).extend(
                    _TaxLot(acquired, shares, bonus_origin=True)
                    for acquired, shares in bonus_lots)
                applied.add('bonus')
                booked.append({'date': day, 'code': action.code, 'event': 'bonus_tradable',
                               'record_shares': record_shares, 'bonus_shares': bonus,
                               'cash_amount': 0.0})
            if action.cash_available_date <= day and 'cash' not in applied:
                if 'ex' not in applied:
                    raise ValueError('dividend cannot settle before ex-date')
                cash_due = record_shares * action.cash_per_share
                self.cash_receivable -= cash_due
                if abs(self.cash_receivable) < 1e-10:
                    self.cash_receivable = 0.0
                self.cash += cash_due
                applied.add('cash')
                booked.append({'date': day, 'code': action.code, 'event': 'cash_available',
                               'record_shares': record_shares, 'bonus_shares': 0,
                               'cash_amount': cash_due})
            self._entitlements[key] = (action, record_shares, applied, bonus_lots)
        self.action_log.extend(booked)
        return booked

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
