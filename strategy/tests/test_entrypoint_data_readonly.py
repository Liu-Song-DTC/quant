"""Execution entrypoints freeze alternative inputs before core modules load."""
import builtins
import os
import runpy
import unittest
from pathlib import Path
from unittest.mock import patch


STRATEGY = Path(__file__).resolve().parents[1]


class _ReachedCore(Exception):
    pass


class EntrypointDataReadonlyTests(unittest.TestCase):
    def test_main_scripts_disable_implicit_network_refresh_before_core_import(self):
        for script, first_core in (('bt_execution.py', 'core.strategy'),
                                   ('generate_trade_orders.py', 'core.config_loader')):
            with self.subTest(script=script):
                original = builtins.__import__
                observed = []

                def inspect_import(name, *args, **kwargs):
                    if name == first_core:
                        observed.append(os.environ.get('QUANT_ALT_NO_AUTOREFRESH'))
                        raise _ReachedCore
                    return original(name, *args, **kwargs)

                with (patch.dict(os.environ, {'QUANT_ALT_NO_AUTOREFRESH': '0'}),
                      patch('builtins.__import__', side_effect=inspect_import)):
                    with self.assertRaises(_ReachedCore):
                        runpy.run_path(str(STRATEGY / script), run_name='__main__')
                self.assertEqual(observed, ['1'])


if __name__ == '__main__':
    unittest.main()
