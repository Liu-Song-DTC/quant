"""A partial provider rebuild must never erase an old signal input."""
import pickle
import tempfile
import unittest
from datetime import date
from pathlib import Path

from data.rebuild_dragon_tiger_history import write_hist


class DragonTigerHistoryRefreshGuardTests(unittest.TestCase):
    def test_partial_rebuild_cannot_replace_old_events(self):
        with tempfile.TemporaryDirectory() as d:
            output = Path(d) / 'history.pkl'
            old = {'000001': {date(2024, 1, 2), date(2026, 10, 27)}}
            output.write_bytes(pickle.dumps(old))
            self.assertFalse(write_hist({'000001': {date(2026, 10, 28)}},
                                        3, min_ok=3, out=str(output)))
            self.assertEqual(pickle.loads(output.read_bytes()), old)
            self.assertFalse(write_hist({'000001': {date(2024, 1, 2)}},
                                        2, min_ok=3, out=str(output)))
            self.assertEqual(pickle.loads(output.read_bytes()), old)
            updated = {'000001': old['000001'] | {date(2026, 10, 28)}}
            self.assertTrue(write_hist(updated, 3, min_ok=3, out=str(output)))
            self.assertEqual(pickle.loads(output.read_bytes()), updated)


if __name__ == '__main__':
    unittest.main()
