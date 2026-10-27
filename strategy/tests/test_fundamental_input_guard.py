"""Missing/invalid financial dates cannot disappear from a successful audit."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

import pandas as pd


spec = importlib.util.spec_from_file_location(
    'fundamental_input_guard', Path(__file__).resolve().parents[1] / 'tools/check_fundamental_pit.py')
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)


class FundamentalInputGuardTests(unittest.TestCase):
    def test_valid_proxy_rows_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            pd.DataFrame({'报告期': ['20260331'], '数据可用日期': ['20260430']}).to_csv(
                Path(directory) / '000001.csv', index=False)
            code, result = guard.audit(directory)
            self.assertEqual(code, 0)
            self.assertEqual(result['rows'], 1)

    def test_invalid_period_or_wrong_availability_is_counted(self):
        with tempfile.TemporaryDirectory() as directory:
            pd.DataFrame({'报告期': ['bad', '20260231', '20260331'],
                          '数据可用日期': ['', '20260430', '20260331']}).to_csv(
                Path(directory) / '000001.csv', index=False)
            code, result = guard.audit(directory)
            self.assertEqual(code, 1)
            self.assertEqual(result['bad_rows'], 3)
            self.assertEqual(result['invalid_period_rows'], 2)

    def test_missing_columns_empty_file_and_empty_directory_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(guard.audit(directory)[0], 2)
            path = Path(directory) / '000001.csv'
            path.write_text('报告期\n20260331\n')
            self.assertEqual(guard.audit(directory)[0], 2)
            path.write_text('报告期,数据可用日期\n')
            self.assertEqual(guard.audit(directory)[0], 2)


if __name__ == '__main__':
    unittest.main()
