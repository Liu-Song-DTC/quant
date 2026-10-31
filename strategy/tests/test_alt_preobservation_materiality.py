"""Pre-observation materiality probes cannot silently become current/live queries."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from strategy.analysis.probe_alt_preobservation_materiality import probe


class AltPreobservationMaterialityTests(unittest.TestCase):
    def test_rejects_post_first_snapshot_or_auto_refresh(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'alt_versions').mkdir()
            for stem in ('northbound_daily', 'margin_daily', 'reduction_records', 'reduction_plans'):
                (root / 'alt_versions' / f'{stem}.json').write_text(json.dumps({
                    'schema': 1, 'versions': [{'observed_on': '2026-10-29', 'sha256': 'a' * 64}]}))
            selected = root / 'selected.csv'
            pd.DataFrame({'date': ['2026-10-29'], 'code': ['000001']}).to_csv(selected, index=False)
            with patch.dict('os.environ', {'QUANT_ALT_NO_AUTOREFRESH': '1'}):
                with self.assertRaisesRegex(ValueError, 'strictly before'):
                    probe(selected, root)
            pd.DataFrame({'date': ['2026-10-28'], 'code': ['000001']}).to_csv(selected, index=False)
            with patch.dict('os.environ', {'QUANT_ALT_NO_AUTOREFRESH': ''}):
                with self.assertRaisesRegex(ValueError, 'NO_AUTOREFRESH'):
                    probe(selected, root)


if __name__ == '__main__':
    unittest.main()
