"""Exercise the actual CLI acknowledgement branch with a real temporary ledger."""
import ast
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from core.execution_ledger import acknowledge_orders, apply_execution_reports, queue_orders
from core.live_state import atomic_json, commit_json_files, recover_commit


class LiveAcknowledgementCommitTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.account_path = self.root / 'live_account.json'
        self.positions_path = self.root / 'current_positions.json'
        self.account = {'cash': 10000., 'last_date': '2026-11-11'}
        self.positions = {}
        orders = [{'stock_code': '600000.SH', 'action': 'open', 'shares': 200,
                   'price': 10., 'order_amount': 2000., 'order_shares': 200}]
        queue_orders(self.account, orders, self.positions, '2026-11-11')
        atomic_json(self.account_path, self.account)
        atomic_json(self.positions_path, self.positions)
        self.before = self.disk()

    def disk(self):
        return {p.name: p.read_bytes() for p in self.root.iterdir() if p.is_file()}

    def apply_partial_fill(self):
        apply_execution_reports(self.account, self.positions, [{
            'run_id': '20261111', 'seq': 1, 'order_status': 'PARTIAL',
            'stock_code': '600000.SH', 'order_type': 'BUY', 'volume': 100,
            'price': 10., 'fees': .5, 'fill_id': 'fill-one', 'ts': '2026-11-12 09:31:00',
        }])

    def run_branch(self, dry_run=False, keys='20261111:1', target='2026-11-12', commit=None):
        source = Path(os.environ.get('QUANT_ACK_TEST_SOURCE',
            str(Path(__file__).resolve().parents[1] / 'generate_trade_orders.py')))
        tree = ast.parse(source.read_text())
        main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
        branch = next(n for n in main.body if isinstance(n, ast.If)
                      and ast.unparse(n.test) == 'args.ack_stale_unresolved')
        namespace = dict(args=SimpleNamespace(dry_run=dry_run, ack_stale_unresolved=keys),
            account=self.account, prev_positions=self.positions, target_date=target,
            ACCOUNT_FILE=str(self.account_path), POSITIONS_FILE=str(self.positions_path),
            _PROJECT_DIR=str(self.root), os=os, acknowledge_orders=acknowledge_orders,
            commit_json_files=commit or commit_json_files)
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(ast.Module(body=[branch], type_ignores=[]), str(source), 'exec'), namespace)

    def test_dry_run_acknowledges_in_memory_without_writing(self):
        self.apply_partial_fill()
        self.run_branch(dry_run=True)
        self.assertEqual(self.disk(), self.before)
        self.assertEqual(self.account['pending_orders']['20261111:1']['status'], 'EXPIRED_UNFILLED')

    def test_ack_persists_new_fill_cash_and_positions_together(self):
        self.apply_partial_fill()
        self.run_branch()
        saved_account = json.loads(self.account_path.read_text())
        saved_positions = json.loads(self.positions_path.read_text())
        self.assertEqual(saved_positions, self.positions)
        self.assertEqual(saved_account, self.account)
        self.assertEqual(saved_account['cash'], 8999.5)
        self.assertEqual(saved_positions['600000']['shares'], 100)
        self.assertEqual(saved_account['pending_orders']['20261111:1']['unfilled_shares'], 100)

    def test_past_target_cannot_commit_ack(self):
        with self.assertRaises(RuntimeError):
            self.run_branch(target='2026-11-10')
        self.assertEqual(self.disk(), self.before)

    def test_invalid_ack_key_cannot_commit_partial_batch(self):
        with self.assertRaises(ValueError):
            self.run_branch(keys='20261111:1,unknown:2')
        self.assertEqual(self.disk(), self.before)

    def test_plain_ack_does_not_change_cash_or_positions(self):
        self.run_branch()
        self.assertEqual(json.loads(self.account_path.read_text())['cash'], 10000.)
        self.assertEqual(json.loads(self.positions_path.read_text()), {})

    def test_interrupted_ack_transaction_recovers_both_files(self):
        self.apply_partial_fill()
        def interrupt(updates, journal):
            # The existing journal protocol can resume after only the first member is written.
            entries = [[Path(path).name, copy.deepcopy(value)] for path, value in updates.items()]
            atomic_json(journal, {'version': 1, 'updates': entries})
            atomic_json(self.root / entries[0][0], entries[0][1])
            raise OSError('injected interruption after first state file')
        with self.assertRaises(OSError):
            self.run_branch(commit=interrupt)
        self.assertTrue(recover_commit(self.root / '.live_state_journal.json'))
        self.assertEqual(json.loads(self.account_path.read_text()), self.account)
        self.assertEqual(json.loads(self.positions_path.read_text()), self.positions)


if __name__ == '__main__':
    unittest.main()
