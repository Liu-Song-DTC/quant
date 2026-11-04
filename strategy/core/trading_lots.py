"""A-share buy quantity rules shared by order generation and simulation."""


def buy_shares_for_budget(code, amount, price):
    if price <= 0 or amount <= 0:
        return 0
    affordable = int(amount / price)
    if str(code).startswith(('688', '689')):
        return affordable if affordable >= 200 else 0
    return affordable // 100 * 100


def valid_buy_shares(code, shares):
    if not isinstance(shares, int) or shares <= 0:
        return False
    if str(code).startswith(('688', '689')):
        return shares >= 200
    return shares % 100 == 0
