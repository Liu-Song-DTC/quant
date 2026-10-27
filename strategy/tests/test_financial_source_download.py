"""Exercise the real download method without importing network/bootstrap code."""
import ast
import contextlib
import io
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import pandas as pd


source = Path(__file__).resolve().parents[2] / 'data/data_manager.py'
tree = ast.parse(source.read_text())
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'StockDataManager')
method = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
              and n.name == 'download_financial_data_by_date')


class FinancialSourceDownloadTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.stock = types.SimpleNamespace(get_all_company_performance=Mock())
        self.ak = types.SimpleNamespace(**{name: Mock() for name in
                                          ['stock_zcfz_em', 'stock_lrb_em', 'stock_xjll_em']})
        for function in [self.stock.get_all_company_performance, *vars(self.ak).values()]:
            function.return_value = pd.DataFrame({'股票代码': ['000001'], 'value': [10.]})
        ns = {'os': os, 'pd': pd, 'Path': Path, 'tempfile': tempfile,
              'time': types.SimpleNamespace(sleep=lambda value: None),
              'ef': types.SimpleNamespace(stock=self.stock), 'ak': self.ak}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), 'exec'), ns)
        self.download = ns['download_financial_data_by_date']
        self.manager = types.SimpleNamespace(fundamental_source_dir=self.root)
        self.output = self.root / '20260630'

    def run_download(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return self.download(self.manager, '20260630')

    def test_one_source_failure_is_not_reported_as_all_success(self):
        self.ak.stock_xjll_em.return_value = None
        self.assertFalse(self.run_download())
        self.assertTrue((self.output / 'yjbb.csv').exists())
        self.assertFalse((self.output / 'xjll.csv').exists())
        self.ak.stock_lrb_em.assert_called_once()

    def test_zero_byte_and_header_only_cache_are_fetched_again(self):
        self.output.mkdir()
        (self.output / 'yjbb.csv').write_text('')
        (self.output / 'zcfz.csv').write_text('股票代码,value\n')
        self.assertTrue(self.run_download())
        self.stock.get_all_company_performance.assert_called_once()
        self.ak.stock_zcfz_em.assert_called_once()
        self.assertEqual(len(pd.read_csv(self.output / 'zcfz.csv')), 1)

    def test_valid_cache_is_unchanged(self):
        self.assertTrue(self.run_download())
        before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in self.output.iterdir()}
        for function in [self.stock.get_all_company_performance, *vars(self.ak).values()]:
            function.reset_mock()
        self.assertTrue(self.run_download())
        after = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in self.output.iterdir()}
        self.assertEqual(before, after)
        self.stock.get_all_company_performance.assert_not_called()

    def test_write_failure_does_not_leave_partial_source_files(self):
        with patch.object(pd.DataFrame, 'to_csv', side_effect=OSError('write failed')):
            self.assertFalse(self.run_download())
        self.assertEqual(list(self.output.iterdir()), [])


if __name__ == '__main__':
    unittest.main()
