"""Test failure recovery and resume behavior without touching production data/network."""
import importlib.util
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / 'data/akshare_kline_downloader.py'
CHAIN_PATH = ROOT / 'strategy/run_daily_live.sh'


class KlineStampTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location('isolated_kline_downloader', MODULE_PATH)
        cls.mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.mod)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.meta = root / 'metadata'
        self.raw = root / 'raw'
        self.meta.mkdir()
        self.raw.mkdir()
        self.stamp = self.meta / 'kline_sync_1028.stamp'
        self.patchers = [
            patch.object(self.mod, 'META', str(self.meta)),
            patch.object(self.mod, 'RAW', str(self.raw)),
            patch.object(self.mod, 'last_complete_trading_day', return_value='20261028'),
            patch.object(self.mod, 'install_patch'),
            patch.object(self.mod, 'download_index'),
            patch.object(self.mod, 'get_listed_codes', return_value={'000001', '000002'}),
            patch.object(self.mod, 'incremental', return_value='incr'),
            patch.object(self.mod, 'full_download', side_effect=RuntimeError('retry failed')),
            patch.object(self.mod.time, 'strftime', wraps=self.mod.time.strftime),
            patch.dict(os.environ, {'QUANT_FORCE_FETCH': ''}),
        ]
        for p in self.patchers:
            p.start()
            self.addCleanup(p.stop)
        self.mod.time.strftime.side_effect = lambda fmt, *args: '1028' if fmt == '%m%d' else '2026-10-28 21:00:00'

    def test_sample_cannot_publish_full_market_completion(self):
        self.mod.run(test_n=1)
        self.assertFalse(self.stamp.exists())
        self.mod.run()
        self.assertTrue(self.stamp.exists())
        self.assertEqual(json.loads(self.stamp.read_text())['processed_count'], 2)
        self.mod.install_patch.reset_mock()
        self.mod.run()
        self.mod.install_patch.assert_not_called()

    def test_failed_retry_does_not_leave_or_reuse_stamp(self):
        self.mod.incremental.side_effect = lambda code, end: (_ for _ in ()).throw(RuntimeError('bad')) if code == '000002' else 'incr'
        with self.assertRaises(SystemExit):
            self.mod.run()
        self.assertFalse(self.stamp.exists())
        self.mod.incremental.side_effect = None
        self.mod.incremental.return_value = 'incr'
        self.mod.run()
        self.assertTrue(self.stamp.exists())

    def test_forced_failed_refresh_invalidates_previous_success(self):
        self.mod.run()
        with patch.dict(os.environ, {'QUANT_FORCE_FETCH': '1'}):
            self.mod.incremental.side_effect = RuntimeError('network failure')
            with self.assertRaises(SystemExit):
                self.mod.run()
        self.assertFalse(self.stamp.exists())

    def test_legacy_or_partial_stamp_is_not_accepted(self):
        roster = self.meta / 'listed_codes_1028.csv'
        roster.write_text('000001\n000002\n')
        self.stamp.write_text(json.dumps({'roster_sha256': 'unused', 'synced_through': '20261028'}))
        self.mod.run()
        self.assertTrue(json.loads(self.stamp.read_text())['complete'])


class ChainResumeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        text = CHAIN_PATH.read_text()
        cls.parse = text[text.index('ALLOW_WEEKEND=0;'):text.index('if [ "$ALLOW_WEEKEND" = "1" ];')]
        cls.checkpoint = text[text.index('export REFRESH_LOG=/mnt/d/quant/logs/refresh_${TODAY//-/}.log'):text.index('# 1.2/4 数据每日快照')]

    def run_case(self, args=(), marker=None, force=''):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'logs').mkdir()
            if marker:
                (root / 'logs' / f'.chain_state_20261028_{marker}').write_text('refresh\n')
            script = ('set -e\nTODAY=2026-10-28\n'
                      'TRACE="$PWD/trace"\n'
                      'bash() { printf "%s:%s\\n" "$1" "${QUANT_FORCE_FETCH:-}" >> "$TRACE"; }\n'
                      + self.parse + self.checkpoint)
            result = subprocess.run(['bash', '-c', script, 'chain', *args], cwd=root,
                                    env={**os.environ, 'QUANT_FORCE_FETCH': force},
                                    text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            return (root / 'trace').read_text().splitlines()

    def test_checkpoint_verifies_again_without_fetch(self):
        self.assertEqual(self.run_case(marker='evening'), ['data/refresh_verify.sh:'])

    def test_fresh_fetches_despite_checkpoint(self):
        self.assertEqual(self.run_case(args=['--fresh'], marker='evening'),
                         ['data/refresh_fetch.sh:1', 'data/refresh_verify.sh:1'])

    def test_force_env_fetches_despite_checkpoint(self):
        self.assertEqual(self.run_case(marker='evening', force='true'),
                         ['data/refresh_fetch.sh:true', 'data/refresh_verify.sh:true'])

    def test_preopen_checkpoint_cannot_skip_evening_fetch(self):
        self.assertEqual(self.run_case(marker='preopen'),
                         ['data/refresh_fetch.sh:', 'data/refresh_verify.sh:'])

class ChainLockTests(unittest.TestCase):
    def test_standalone_fetch_and_compat_entry_reject_concurrent_writer(self):
        import fcntl
        for rel, stop in [('data/refresh_fetch.sh', 'PY=/mnt/d/quant'),
                          ('data/refresh_all.sh', 'export REFRESH_LOG=')]:
            with self.subTest(script=rel), tempfile.TemporaryDirectory() as tmp:
                text = (ROOT / rel).read_text()
                start = text.index('cd /mnt/d/quant')
                snippet = text[start:text.index(stop, start)].replace('cd /mnt/d/quant',
                                                                       'cd "$TEST_ROOT"')
                root = Path(tmp)
                with (root / '.daily_chain.lock').open('w') as held:
                    fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    blocked = subprocess.run(['bash', '-c', snippet], env={**os.environ,
                                             'TEST_ROOT': tmp}, text=True, capture_output=True)
                    self.assertEqual(blocked.returncode, 3, blocked.stderr)
                available = subprocess.run(['bash', '-c', snippet], env={**os.environ,
                                            'TEST_ROOT': tmp}, text=True, capture_output=True)
                self.assertEqual(available.returncode, 0, available.stderr)
