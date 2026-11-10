"""Minimal test: spawn worker generates signals for 1 stock"""
import multiprocessing, sys, os, traceback

os.chdir('/mnt/d/quant/strategy')

def worker_test():
    try:
        from core.signal_engine import SignalEngine
        from core.config_loader import load_config
        from core.signal_store import SignalStore
        import pandas as pd

        sys.stderr.write(f"[W] Starting...\n"); sys.stderr.flush()
        config = load_config()
        engine = SignalEngine(config)
        store = SignalStore()
        data = pd.read_csv('/mnt/d/quant/data/stock_data/backtrader_data/000001_qfq.csv', parse_dates=['datetime'])
        sys.stderr.write(f"[W] Data loaded: {len(data)} rows\n"); sys.stderr.flush()

        engine.generate('000001', data, store)
        sys.stderr.write(f"[W] DONE: {len(store._store)} signals\n"); sys.stderr.flush()
    except Exception as e:
        sys.stderr.write(f"[W] FAIL: {e}\n"); sys.stderr.flush()
        traceback.print_exc(file=sys.stderr); sys.stderr.flush()

if __name__ == '__main__':
    ctx = multiprocessing.get_context('spawn')
    p = ctx.Process(target=worker_test)
    p.start()
    p.join()
    sys.stderr.write('Main done\n')
