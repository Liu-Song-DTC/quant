import ast
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core.input_identity import write_signal_identity


def extract(name):
    source=Path(__file__).resolve().parents[1]/'bt_execution.py'
    tree=ast.parse(source.read_text())
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    ns={'__file__':str(source),'pd':pd}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),ns)
    return ns[name]


class SignalInputIdentityTests(unittest.TestCase):
    def test_signal_reuse_checks_content_even_with_old_timestamps(self):
        import os
        check=extract('_signals_stale')
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);p=root/'signals.csv'
            p.write_text('date,score\n2026-10-26,1\n')
            (root/'.signal_code_fp').write_text('fixture|0')
            ns=check.__globals__
            ns.update(os=os,_B1V2_WARMUP_DAYS=0,
                      _signal_code_fingerprint=lambda:'fixture',
                      _signal_data_fingerprint=lambda:'inputs_a',
                      _latest_data_date=lambda:'2026-10-26',
                      config=SimpleNamespace(get=lambda *a:''))
            write_signal_identity(p,'inputs_a')
            self.assertFalse(check(str(p)))
            stat=p.stat();p.write_text('date,score\n2026-10-26,2\n')
            os.utime(p,ns=(stat.st_atime_ns,stat.st_mtime_ns))
            self.assertTrue(check(str(p)))

    def test_optional_margin_data_is_tracked_only_when_consumed(self):
        import os
        fingerprint=extract('_signal_data_fingerprint')
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);prices=root/'prices';prices.mkdir()
            (prices/'000001_qfq.csv').write_bytes(b'price_one')
            financial=root/'financial';financial.mkdir()
            alt=root/'data/alternative_data';alt.mkdir(parents=True)
            ns=fingerprint.__globals__
            enabled=False
            ns.update(os=os,DATA_PATH=str(prices),FUNDAMENTAL_PATH=str(financial),
                      _PROJECT_DIR=str(root),_resolve_cmap_hist=lambda:None,
                      config=SimpleNamespace(get=lambda *a:enabled))
            before=fingerprint()
            (alt/'margin_detail.pkl').write_bytes(b'margin_a')
            self.assertEqual(before,fingerprint())
            enabled=True
            first=fingerprint()
            (alt/'margin_detail.pkl').write_bytes(b'margin_b')
            self.assertNotEqual(first,fingerprint())


if __name__=='__main__':
    unittest.main()
