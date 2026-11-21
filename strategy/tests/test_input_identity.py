import os
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.input_identity import (factor_input_identity, file_content_sha256,
                                 write_signal_identity, signal_identity_matches)


class InputIdentityTests(unittest.TestCase):
    def test_parallel_hash_keeps_sorted_roles_and_recursive_optional_markers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            financial = root/'financial'; financial.mkdir()
            extras = root/'extras'; extras.mkdir()
            (extras/'nested').mkdir()
            prices = {}
            for number in reversed(range(40)):
                path = root/f'{number:06}.csv'
                path.write_bytes(f'price:{number}'.encode())
                prices[f'{number:06}'] = path
            (financial/'a.csv').write_bytes(b'financial')
            (financial/'ignored.txt').write_bytes(b'ignored')
            (extras/'nested/value').write_bytes(b'payload')
            def digest(path):
                return hashlib.sha256(path.read_bytes()).hexdigest()
            entries = [('price', code, digest(path)) for code, path in sorted(prices.items())]
            entries += [('financial_directory', '', 'present'),
                        ('financial', 'a.csv', digest(financial/'a.csv')),
                        ('extra:0', '', 'directory'),
                        ('extra:0', 'nested/value', digest(extras/'nested/value')),
                        ('extra:1', '', None)]
            expected = hashlib.sha256(json.dumps(entries, ensure_ascii=False,
                                      separators=(',', ':')).encode()).hexdigest()
            self.assertEqual(factor_input_identity(prices, financial, [extras, root/'absent']), expected)

    def test_parallel_required_missing_and_broken_optional_symlink_raise(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(FileNotFoundError):
                factor_input_identity({'000001': root/'missing.csv'})
            (root/'broken').symlink_to(root/'absent')
            with self.assertRaises(FileNotFoundError):
                factor_input_identity({}, extra_paths=[root/'broken'])

    def fixture(self, root):
        root.mkdir()
        (root/'price.csv').write_text('date,close\n20261026,10.0\n')
        (root/'financial').mkdir()
        (root/'financial/000001.csv').write_text('period,roe\n20260630,12\n')
        (root/'concept.pkl').write_bytes(b'concept_one')

    def identity(self, root):
        return factor_input_identity({'000001': root/'price.csv'}, root/'financial',
                                     [root/'concept.pkl', root/'optional.csv'])

    def test_same_length_rewrite_restored_mtime_changes_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)/'original'; self.fixture(root)
            before = self.identity(root)
            path = root/'price.csv'; stamp = path.stat()
            path.write_bytes(path.read_bytes().replace(b'10.0', b'11.0'))
            os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
            self.assertEqual(path.stat().st_size, stamp.st_size)
            self.assertNotEqual(before, self.identity(root))

    def test_identical_bytes_in_different_trees_and_new_mtimes_are_identical(self):
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory)/'a', Path(directory)/'b'; self.fixture(a)
            shutil.copytree(a, b)
            for path in b.rglob('*'):
                os.utime(path, (1000, 1000))
            self.assertEqual(self.identity(a), self.identity(b))
            (b/'financial/ignored.txt').write_text('not consumed')
            self.assertEqual(self.identity(a), self.identity(b))

    def test_logical_security_assignment_and_optional_missing_are_distinguished(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'a'; self.fixture(root)
            base=self.identity(root)
            changed=factor_input_identity({'000002':root/'price.csv'},root/'financial',
                                          [root/'concept.pkl', root/'optional.csv'])
            self.assertNotEqual(base, changed)
            (root/'optional.csv').write_bytes(b'')
            self.assertNotEqual(base, self.identity(root))
            (root/'price.csv').unlink()
            with self.assertRaises(FileNotFoundError):
                self.identity(root)

    def test_read_failure_and_replacement_during_read_are_not_silent_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'input.csv'; path.write_bytes(b'abc')
            with patch.object(Path,'open',side_effect=PermissionError('denied')):
                with self.assertRaises(PermissionError): file_content_sha256(path,optional=True)
            original=Path.stat
            def replaced(p, *args, **kwargs):
                if p == path:
                    replacement=path.with_suffix('.new')
                    replacement.write_bytes(b'xyz')
                    os.replace(replacement,path)
                return original(p,*args,**kwargs)
            with patch.object(Path,'stat',replaced):
                with self.assertRaises(RuntimeError): file_content_sha256(path)

    def test_signal_cache_is_bound_to_csv_bytes_and_input_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'signals.csv';path.write_bytes(b'date,score\n20261026,1\n')
            self.assertFalse(signal_identity_matches(path,'input_a'))
            write_signal_identity(path,'input_a')
            self.assertTrue(signal_identity_matches(path,'input_a'))
            self.assertFalse(signal_identity_matches(path,'input_b'))
            old=path.stat()
            path.write_bytes(path.read_bytes().replace(b',1',b',2'))
            os.utime(path,ns=(old.st_atime_ns,old.st_mtime_ns))
            self.assertFalse(signal_identity_matches(path,'input_a'))


if __name__ == '__main__':
    unittest.main()
