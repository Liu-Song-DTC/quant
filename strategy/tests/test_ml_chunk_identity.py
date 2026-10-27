"""ML cache invalidation, artifact integrity and actual ensemble replay."""
import ast
import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core.ml_predictor import MLFactorPredictor


class MLChunkIdentityTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.data=self.root/'data'
        self.prices=self.data/'stock_data/backtrader_data'
        self.prices.mkdir(parents=True)
        self.financial=self.data/'stock_data/fundamental_data'
        self.financial.mkdir()
        (self.financial/'000001.csv').write_text('roe\n10\n')
        self.price=self.prices/'000001_qfq.csv'
        self.price.write_text('datetime,close\n2026-01-01,10.0\n2026-01-02,11.0\n')
        self.history=self.data/'concept_map_history.pkl'
        self.history.write_bytes(b'old-history')
        path=Path(__file__).resolve().parents[1]/'bt_execution.py'
        names={'_ml_chunk_cache_fp','_ml_chunk_cache_aux_fp','_ml_chunk_manifest_path',
               '_ml_chunk_load_manifest','_ml_chunk_save_manifest','_ml_chunk_load','_ml_chunk_save'}
        nodes=[n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in names]
        self.ns={'os':os,'pd':pd,'np':np,'_PROJECT_DIR':str(self.root),'DATA_PATH':str(self.prices),
                 '_ML_CHUNK_CACHE_DIR':str(self.root/'cache'),'_resolve_cmap_hist':lambda:str(self.history),
                 '_signal_code_fingerprint':lambda:'fixture-code'}
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),self.ns)
        self.config=SimpleNamespace(config={'ml':{'use_cross_features':False,'use_regime_features':False}})
        self.fd=SimpleNamespace(data_path=str(self.financial))

    def aux(self):
        return self.ns['_ml_chunk_cache_aux_fp'](self.fd,self.config)

    def test_aux_detects_restored_stat_financial_and_versioned_concept_changes(self):
        before=self.aux()
        for path in [self.financial/'000001.csv',self.history]:
            old=path.stat();payload=path.read_bytes()
            path.write_bytes(payload.replace(b'10',b'20') if b'10' in payload else b'new-history')
            os.utime(path,ns=(old.st_atime_ns,old.st_mtime_ns))
            after=self.aux()
            self.assertNotEqual(before,after)
            before=after
        self.config.config['ml']['blend_weight']=.1
        self.assertNotEqual(before,self.aux())

    def test_price_prefix_is_append_stable_but_identity_header_and_index_sensitive(self):
        fingerprint=self.ns['_ml_chunk_cache_fp']
        boundaries=pd.to_datetime(['2026-01-01','2026-01-02'])
        stock={'000001':str(self.price)}
        before=fingerprint(stock,self.fd,boundaries)
        self.price.write_text(self.price.read_text()+'2026-01-03,12.0\n')
        self.assertEqual(before,fingerprint(stock,self.fd,boundaries))
        self.assertNotEqual(before,fingerprint({'000002':str(self.price)},self.fd,boundaries))
        self.price.write_text(self.price.read_text().replace('close','open'))
        self.assertNotEqual(before,fingerprint(stock,self.fd,boundaries))
        before=fingerprint(stock,self.fd,boundaries)
        (self.prices/'sh000001_qfq.csv').write_text('datetime,close\n2026-01-01,3000\n')
        self.assertNotEqual(before,fingerprint(stock,self.fd,boundaries))
        self.price.unlink()
        with self.assertRaises(FileNotFoundError):fingerprint(stock,self.fd,boundaries)

    def test_identical_input_contents_relocate_without_aux_invalidation(self):
        before=self.aux()
        moved=self.root/'moved'
        shutil.copytree(self.data,moved/'data')
        self.ns['_PROJECT_DIR']=str(moved)
        self.ns['_resolve_cmap_hist']=lambda:str(moved/'data/concept_map_history.pkl')
        self.fd.data_path=str(moved/'data/stock_data/fundamental_data')
        for path in (moved/'data').rglob('*'):
            os.utime(path,(1000,1000))
        self.assertEqual(before,self.aux())

    def save(self,key='fixture',model=None):
        codes=np.array(['000001','000002']);dates=np.array(['2026-01-02']*2,dtype='datetime64[ns]')
        self.ns['_ml_chunk_save'](key,0,'trained_good' if model else 'reuse_prev',.1,model,
                                 codes,dates,np.array([.125,-.25]))

    def test_prediction_checksum_rejects_tampering_and_interrupted_write(self):
        self.save()
        cached=self.ns['_ml_chunk_load']('fixture',self.config)
        np.testing.assert_array_equal(cached[3].ml_pred.to_numpy(),[.125,-.25])
        artifact=self.root/'cache/chunk_fixture_preds.parquet'
        frame=pd.read_parquet(artifact);frame['ml_pred']=[.5,.5];frame.to_parquet(artifact,index=False)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertIsNone(self.ns['_ml_chunk_load']('fixture',self.config))
            with patch('core.live_state.atomic_bytes',side_effect=OSError('interrupted')):
                self.save('interrupted')
        manifest=json.loads((self.root/'cache/manifest.json').read_text())
        self.assertNotIn('interrupted',manifest)
        # Unversioned entries are misses, never silently blessed after loading.
        manifest['fixture'].pop('version')
        self.ns['_ml_chunk_save_manifest'](manifest)
        self.assertIsNone(self.ns['_ml_chunk_load']('fixture',self.config))

    def test_actual_ensemble_roundtrip_and_missing_member_fails_closed(self):
        rng=np.random.default_rng(41)
        frame=pd.DataFrame({'date':np.repeat(pd.bdate_range('2025-01-01',periods=100),12),
            'code':np.tile(np.arange(12),100),'a':rng.normal(size=1200),
            'b':rng.normal(size=1200),'c':rng.normal(size=1200),'future_ret':rng.normal(size=1200)})
        frame['label_end_date']=frame.date+pd.offsets.BDay(5)
        model=MLFactorPredictor(self.config.config)
        model.xgb_params.update(n_estimators=4,n_jobs=1,early_stopping_rounds=2)
        with contextlib.redirect_stdout(io.StringIO()):self.assertIsNotNone(model.train(frame))
        self.save(model=model)
        cached=self.ns['_ml_chunk_load']('fixture',self.config)
        self.assertIsNotNone(cached)
        last=frame[frame.date==frame.date.max()]
        np.testing.assert_array_equal(model.predict(last),cached[0].predict(last))
        (self.root/'cache/chunk_fixture_model.json.1.json').unlink()
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertIsNone(self.ns['_ml_chunk_load']('fixture',self.config))


if __name__=='__main__':unittest.main()
