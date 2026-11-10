"""Test Pool + initializer pattern (same as bt_execution.py)"""
import multiprocessing, sys, os, traceback

os.chdir('/mnt/d/quant/strategy')

_WORKER_ENGINE = None

def _init():
    """Same pattern as bt_execution.py _init_worker"""
    global _WORKER_ENGINE
    from core.signal_engine import SignalEngine
    from core.config_loader import load_config
    config = load_config()
    _WORKER_ENGINE = SignalEngine(config)
    sys.stderr.write(f"[INIT] engine created\n"); sys.stderr.flush()

def _worker(args):
    """Same pattern as bt_execution.py _generate_stock_signal_worker"""
    global _WORKER_ENGINE
    code, filepath = args
    code = str(code).zfill(6)
    try:
        from core.signal_store import SignalStore
        import pandas as pd

        store = SignalStore()
        data = pd.read_csv(filepath, parse_dates=['datetime'])
        sys.stderr.write(f"[W] {code}: data={len(data)} rows\n"); sys.stderr.flush()

        _WORKER_ENGINE.generate(code, data, store)
        sys.stderr.write(f"[W] {code}: DONE sigs={len(store._store)}\n"); sys.stderr.flush()
        return (code, len(store._store))
    except Exception as e:
        sys.stderr.write(f"[W] {code}: FAIL {e}\n"); sys.stderr.flush()
        traceback.print_exc(file=sys.stderr); sys.stderr.flush()
        return (code, -1)

if __name__ == '__main__':
    items = [('000001', '/mnt/d/quant/data/stock_data/backtrader_data/000001_qfq.csv')]

    ctx = multiprocessing.get_context('spawn')
    pool = ctx.Pool(processes=1, initializer=_init)

    for result in pool.imap_unordered(_worker, items):
        sys.stderr.write(f"[MAIN] got: {result}\n"); sys.stderr.flush()

    pool.close()
    pool.join()
    sys.stderr.write('Main done\n')
