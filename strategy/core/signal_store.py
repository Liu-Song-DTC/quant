# core/signal_store.py
class SignalStore:
    """
    保存“可执行信号”
    """

    def __init__(self):
        self._store = {}  # (code, date) -> Signal

    def set(self, code, date, signal):
        self._store[(code, date)] = signal

    def get(self, code, date):
        return self._store.get((code, date))

