"""Actual fills need a known calendar date; planned dates cannot supply it."""
from copy import deepcopy
import unittest
from core.execution_ledger import apply_execution_reports, queue_orders


class ExecutionFillDateTests(unittest.TestCase):
    def setUp(self):
        self.account={'cash':10000.}
        self.book={}
        queue_orders(self.account,[{'stock_code':'600000.SH','action':'open','shares':200,'price':10.}],
            self.book,'2026-11-11',execution_date='2026-11-12')

    def report(self, stamp, **changes):
        return dict(run_id='20261112',seq=1,volume=100,price=10.,fees=.5,
            fill_id='first',order_status='PARTIAL',ts=stamp,**changes)

    def test_invalid_positive_fill_date_rejects_entire_batch(self):
        for stamp in ('09:25:07','','not-a-date','2026-02-30 09:25:07','20261112',None):
            with self.subTest(stamp=stamp):
                account,book=deepcopy(self.account),deepcopy(self.book)
                before=deepcopy(account)
                first=self.report('2026-11-12 09:25:07')
                second=self.report(stamp);second['fill_id']='second'
                with self.assertRaisesRegex(ValueError,'execution.*date|成交日期'):
                    apply_execution_reports(account,book,[first,second])
                self.assertEqual(account,before)
                self.assertEqual(book,{})

    def test_full_timestamp_preserves_actual_fill_day_and_idempotence(self):
        report=self.report('2026-11-13 09:25:07')
        self.assertEqual(apply_execution_reports(self.account,self.book,[report]),1)
        self.assertEqual(self.book['600000']['entry_date'],'2026-11-13')
        before=deepcopy((self.account,self.book))
        self.assertEqual(apply_execution_reports(self.account,self.book,[report]),0)
        self.assertEqual((self.account,self.book),before)

    def test_explicit_calendar_date_is_sufficient(self):
        apply_execution_reports(self.account,self.book,[self.report('2026-11-12')])
        self.assertEqual(self.book['600000']['entry_date'],'2026-11-12')

    def test_timezone_timestamp_uses_shanghai_calendar_day(self):
        apply_execution_reports(self.account,self.book,[self.report('2026-11-11T16:30:00+00:00')])
        self.assertEqual(self.book['600000']['entry_date'],'2026-11-12')

    def test_zero_fill_terminal_report_does_not_need_date(self):
        report=dict(run_id='20261112',seq=1,order_status='SKIPPED_NOT_ARMED')
        self.assertEqual(apply_execution_reports(self.account,self.book,[report]),1)
        self.assertEqual(self.account['cash'],10000.)
        self.assertEqual(self.book,{})

    def test_addition_preserves_first_actual_entry_date(self):
        apply_execution_reports(self.account,self.book,[self.report('2026-11-12 09:25:07')])
        report=self.report('2026-11-13 09:25:07');report['fill_id']='second'
        apply_execution_reports(self.account,self.book,[report])
        self.assertEqual(self.book['600000']['entry_date'],'2026-11-12')
        self.assertEqual(self.book['600000']['shares'],200)


if __name__=='__main__':unittest.main()
