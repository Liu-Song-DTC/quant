"""Report calculations happen on availability boundaries, without aliasing days."""
import ast
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


class FundamentalReportMemoTests(unittest.TestCase):
    def test_report_boundary_no_future_use_and_independent_daily_entries(self):
        path=Path(__file__).resolve().parents[1]/'core/factor_preparer.py'
        tree=ast.parse(path.read_text())
        functions={node.name:node for node in tree.body if isinstance(node,ast.FunctionDef)}
        ns={'pd':pd,'np':np,'__package__':'core'}
        imports=[node for node in tree.body if isinstance(node,ast.ImportFrom)
                 and node.module=='fundamental_units']
        exec(compile(ast.Module(body=imports+[functions['_parse_pct']],type_ignores=[]),str(path),'exec'),ns)
        parser=ns['_parse_pct'];calls=[]
        def counted(value):
            calls.append(value)
            return parser(value)
        ns['_parse_pct']=counted
        exec(compile(ast.Module(body=[functions['_preload_fundamental_cache']],type_ignores=[]),str(path),'exec'),ns)
        frame=pd.DataFrame([
            {'数据可用日期':'20260501','净资产收益率':1.,'净利润-同比增长':2.,
             '营业总收入-同比增长':3.,'销售毛利率':4.,'每股收益':1.,'所处行业':'制造'},
            {'数据可用日期':'20260701','净资产收益率':5.,'净利润-同比增长':6.,
             '营业总收入-同比增长':7.,'销售毛利率':8.,'每股收益':2.,'所处行业':'银行'},
        ])
        fd=SimpleNamespace(stock_data={'600000':frame})
        dates=pd.to_datetime(['2026-07-20','2026-05-10','2026-04-30','2026-07-01','2026-05-01'])
        cache,industry=ns['_preload_fundamental_cache'](fd,'600000',dates)
        self.assertNotIn(pd.Timestamp('2026-04-30'),cache)
        self.assertEqual(len(calls),8)
        # Preserve the existing first-observed industry return contract.
        self.assertEqual(industry,'制造')
        self.assertEqual(cache[pd.Timestamp('2026-07-01')]['industry'],'银行')
        self.assertEqual(cache[pd.Timestamp('2026-05-10')],cache[pd.Timestamp('2026-05-01')])
        self.assertEqual(cache[pd.Timestamp('2026-07-20')],cache[pd.Timestamp('2026-07-01')])
        self.assertNotEqual(cache[pd.Timestamp('2026-05-10')]['roe'],cache[pd.Timestamp('2026-07-01')]['roe'])
        cache[pd.Timestamp('2026-05-10')]['roe']='changed'
        self.assertNotEqual(cache[pd.Timestamp('2026-05-01')]['roe'],'changed')


if __name__=='__main__':
    unittest.main()
