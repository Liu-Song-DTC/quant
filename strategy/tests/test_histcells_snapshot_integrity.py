"""A failed history scan cannot publish a partial comparison baseline."""
import argparse
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from strategy.tools import histcells


class HistoryCellSnapshotIntegrityTests(unittest.TestCase):
    def test_read_failure_keeps_previous_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'roll_today.pkl'
            target.write_bytes(b'previous complete snapshot')
            with (patch.object(histcells, 'SNAP_DIR', directory),
                  patch.object(histcells, '_scan_all', return_value=({}, ['bad.pkl: unreadable']))):
                self.assertEqual(histcells.cmd_snap(argparse.Namespace(label='roll_today')), 3)
            self.assertEqual(target.read_bytes(), b'previous complete snapshot')

    def test_interrupted_serialization_keeps_previous_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'roll_today.pkl'
            target.write_bytes(b'previous complete snapshot')
            with (patch.object(histcells, 'SNAP_DIR', directory),
                  patch.object(histcells, '_scan_all', return_value=({}, [])),
                  patch.object(histcells.pickle, 'dump', side_effect=OSError('disk full'))):
                with self.assertRaisesRegex(OSError, 'disk full'):
                    histcells.cmd_snap(argparse.Namespace(label='roll_today'))
            self.assertEqual(target.read_bytes(), b'previous complete snapshot')
            self.assertEqual(list(Path(directory).glob('.histcells_*')), [])

    def test_failed_diff_is_not_reported_as_clean(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'roll_today.pkl'
            target.write_bytes(pickle.dumps({'ts': 'today', 'layers': {}, 'git': '', 'cell_hash_version': histcells.CELL_HASH_VERSION}))
            with (patch.object(histcells, 'SNAP_DIR', directory),
                  patch.object(histcells, '_scan_all', return_value=({}, ['bad.pkl: unreadable']))):
                self.assertEqual(histcells.cmd_diff(argparse.Namespace(label='roll_today')), 3)


if __name__ == '__main__':
    unittest.main()
