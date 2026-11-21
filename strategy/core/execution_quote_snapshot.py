"""Byte-preserving snapshots of the existing QFQ execution price loader.

Research replay only; the engine owns calculation. Both adjusted bars and
optional raw closes participate in identity. No mtime-only or signal-cache
identity is sufficient for this execution input.
"""
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from .input_identity import _hash_role_entries, file_content_sha256

FIELDS = ('close_px', 'open_px', 'low_px', 'volume_m', 'amount_m', 'raw_close_m')


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _array_proof(array):
    return dict(shape=list(array.shape), dtype=array.dtype.str,
                c=bool(array.flags.c_contiguous), f=bool(array.flags.f_contiguous),
                sha256=hashlib.sha256(array.tobytes(order='A')).hexdigest())


class ExecutionQuoteSnapshot:
    def __init__(self, directory, data_path, raw_path, stock_codes, calendar,
                 fromdate, todate, source_files):
        self.directory=Path(directory)
        self.data_path=Path(data_path); self.raw_path=Path(raw_path)
        self.codes=list(stock_codes); self.calendar=pd.DatetimeIndex(calendar)
        self.fromdate=str(pd.Timestamp(fromdate)); self.todate=str(pd.Timestamp(todate))
        self.source_files=[Path(p) for p in source_files]
        if not self.codes or self.codes != sorted(set(self.codes)):
            raise ValueError('Snapshot requires a unique sorted source universe')
        if self.calendar.has_duplicates or not self.calendar.is_monotonic_increasing:
            raise ValueError('Snapshot requires a unique ordered calendar')
        self.input_identity=self._inputs()
        # The caller read the index before creating this object. Reconcile that
        # calendar with the data being sealed, rather than accepting a stale one.
        index=pd.read_csv(self.data_path/'sh000001_qfq.csv',usecols=['datetime'],parse_dates=['datetime'])
        dates=index.datetime[index.datetime.between(pd.Timestamp(fromdate),pd.Timestamp(todate))]
        if not self.calendar.equals(pd.DatetimeIndex(sorted(dates.unique()))):
            raise ValueError('Snapshot calendar does not match current index bytes')
        self.identity=dict(version=1,inputs=self.input_identity,
            sources=[file_content_sha256(p) for p in self.source_files],
            numpy=np.__version__,pandas=pd.__version__,fromdate=self.fromdate,todate=self.todate,
            source_codes=self.codes,calendar=[d.isoformat() for d in self.calendar])
        self.key=_digest(self.identity)
        self.path=self.directory/(self.key+'.npz')

    def _inputs(self):
        files=sorted(p.name for p in self.data_path.iterdir()
                     if p.name.endswith('_qfq.csv') and not p.name.startswith('._'))
        expected=sorted([c+'_qfq.csv' for c in self.codes]+['sh000001_qfq.csv'])
        if files != expected:
            raise ValueError('Execution universe changed during snapshot preparation')
        roles=[('qfq',name,self.data_path/name,False) for name in files]
        roles.extend(('raw_close',code,self.raw_path/code/'none.csv',True) for code in self.codes)
        entries=list(_hash_role_entries(roles))
        if files != sorted(p.name for p in self.data_path.iterdir()
                           if p.name.endswith('_qfq.csv') and not p.name.startswith('._')):
            raise ValueError('Execution universe changed during snapshot hashing')
        return _digest(entries)

    def assert_unchanged(self):
        if self._inputs()!=self.input_identity:
            raise ValueError('Execution price bytes changed while using snapshot')
        if [file_content_sha256(p) for p in self.source_files]!=self.identity['sources']:
            raise ValueError('Execution loader source changed while using snapshot')

    def _validate(self, arrays, codes, excluded):
        if sorted(excluded)!=excluded or len(set(excluded))!=len(excluded):
            raise ValueError('Invalid excluded-series list')
        if not set(excluded).issubset(self.codes):
            raise ValueError('Unknown excluded series')
        if codes!=[c for c in self.codes if c not in set(excluded)]:
            raise ValueError('Snapshot stock order or exclusions do not match')
        if set(arrays)!=set(FIELDS):
            raise ValueError('Snapshot price fields do not match')
        for name,array in arrays.items():
            if array.dtype!=np.dtype(np.float32) or array.shape!=(len(self.calendar),len(codes)):
                raise ValueError('Snapshot array shape/dtype mismatch: '+name)
            if not (array.flags.c_contiguous or array.flags.f_contiguous):
                raise ValueError('Snapshot requires original contiguous matrices')

    def load(self):
        if not self.path.exists():
            return None
        with np.load(self.path,allow_pickle=False) as stored:
            if set(stored.files)!=set(FIELDS)|{'manifest'}:
                raise ValueError('Unexpected snapshot archive members')
            metadata=stored['manifest']
            if metadata.dtype!=np.uint8 or metadata.ndim!=1:
                raise ValueError('Invalid snapshot manifest encoding')
            record=json.loads(metadata.tobytes().decode())
            if record['identity']!=self.identity:
                raise ValueError('Snapshot input/producer identity mismatch')
            arrays={name:stored[name] for name in FIELDS}
        self._validate(arrays,record['codes'],record['excluded'])
        if {name:_array_proof(a) for name,a in arrays.items()}!=record['arrays']:
            raise ValueError('Snapshot array byte/layout proof mismatch')
        self.assert_unchanged()
        return arrays,record['codes'],record['excluded']

    def save(self, arrays, codes, excluded):
        codes=list(codes); excluded=list(excluded)
        self._validate(arrays,codes,excluded)
        self.assert_unchanged()
        record=dict(identity=self.identity,codes=codes,excluded=excluded,
                    arrays={name:_array_proof(a) for name,a in arrays.items()})
        metadata=np.frombuffer(json.dumps(record,sort_keys=True,separators=(',', ':')).encode(),dtype=np.uint8)
        self.directory.mkdir(parents=True,exist_ok=True)
        name=None
        try:
            with tempfile.NamedTemporaryFile(dir=self.directory,suffix='.tmp',delete=False) as stream:
                name=stream.name
                np.savez_compressed(stream,manifest=metadata,**arrays)
                stream.flush();os.fsync(stream.fileno())
            os.replace(name,self.path)
        finally:
            if name is not None and os.path.exists(name):os.unlink(name)
