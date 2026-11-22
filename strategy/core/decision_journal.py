"""Buffered observational events for strategy research; never a decision input.

Each process owns a gzip JSONL stream. Memory is bounded by a small text buffer;
all recorded events carry a date, stage/reason, sequence and process identity.
"""
import atexit
from collections import Counter
from datetime import datetime
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import sys


def _clean(value):
    if value is None or isinstance(value,(str,bool,int)):
        return value
    if isinstance(value,float):
        return value if math.isfinite(value) else None
    if isinstance(value,dict):
        return {str(k):_clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):
        return [_clean(v) for v in value]
    if isinstance(value,(set,frozenset)):
        return [_clean(v) for v in sorted(value,key=str)]
    # datetime64[ns].item() yields an integer, so preserve the timestamp first.
    if getattr(getattr(value,'dtype',None),'kind',None)=='M':
        return None if str(value)=='NaT' else str(value)
    if hasattr(value,'item'):
        try:return _clean(value.item())
        except (TypeError,ValueError):pass
    if hasattr(value,'isoformat'):
        return value.isoformat()
    return str(value)


class DecisionJournal:
    """Append-only observation stream with explicit completeness accounting."""

    def __init__(self,directory=None,enabled=None,batch_size=256):
        self.enabled=(os.environ.get('QUANT_DETAIL_TRACE','1')!='0') if enabled is None else bool(enabled)
        self.directory=Path(directory or os.environ.get('QUANT_DETAIL_TRACE_DIR') or
            Path(__file__).resolve().parents[1]/'rolling_validation_results/decision_trace')
        self.batch_size=max(1,int(batch_size))
        self.pid=os.getpid()
        self.sequence=0
        self.counts=Counter()
        self.buffer=[]
        self.stream=None
        self.path=None
        self.digest=hashlib.sha256()
        self.failed_events=0
        self.written_events=0
        self.error=None
        self.closed=False
        self.decision_date=None
        self.started_at=datetime.now().astimezone().isoformat()
        atexit.register(self.close)

    def _open(self):
        if self.path is None:
            self.directory.mkdir(parents=True,exist_ok=True)
            self.path=self.directory/f'events_{self.pid}.jsonl.gz'
            # A process never truncates a previous stream or shares a writer.
            with self.path.open('xb'):
                pass

    def event(self,event,date=None,code=None,**payload):
        if not self.enabled:
            return
        if os.getpid()!=self.pid:
            # No file descriptor is held across a batch, so forked workers can
            # discard the inherited buffer without writing a parent footer.
            self.pid=os.getpid()
            self.sequence=0
            self.counts=Counter()
            self.buffer=[]
            self.path=None
            self.digest=hashlib.sha256()
            self.failed_events=0
            self.written_events=0
            self.error=None
            self.closed=False
            self.started_at=datetime.now().astimezone().isoformat()
        if self.closed or self.error is not None:
            self.failed_events+=1
            return
        try:
            self._open()
            self.sequence+=1
            record=dict(schema_version=1,pid=self.pid,seq=self.sequence,event=event,
                date=_clean(date if date is not None else self.decision_date),
                code=None if code is None else str(code),**_clean(payload))
            line=json.dumps(record,ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n'
            self.digest.update(line.encode('utf-8'))
            self.buffer.append(line)
            self.counts[event]+=1
            if len(self.buffer)>=self.batch_size:self.flush()
        except (OSError,TypeError,ValueError) as error:
            self.failed_events+=1
            self.error=f'{type(error).__name__}: {error}'
            # Observability failure must be visible but cannot alter trades.
            print(f'[DETAIL_TRACE_ERROR] {self.error}',file=sys.stderr,flush=True)

    def flush(self):
        if self.path is not None and self.buffer:
            # Concatenated gzip members are valid JSONL streams. Open/close per
            # bounded batch prevents inherited writers and unflushed descriptors.
            with gzip.open(self.path,'at',encoding='utf-8',compresslevel=1) as stream:
                stream.write(''.join(self.buffer))
            self.written_events+=len(self.buffer)
            self.buffer.clear()

    def receipt(self):
        return dict(schema_version=1,pid=self.pid,enabled=self.enabled,
            stream=str(self.path) if self.path else None,events=self.sequence,
            written_events=self.written_events,unconfirmed_events=self.sequence-self.written_events,
            event_counts=dict(self.counts),failed_events=self.failed_events,error=self.error,
            canonical_uncompressed_sha256=self.digest.hexdigest(),closed=self.closed,
            started_at=self.started_at,finished_at=datetime.now().astimezone().isoformat())

    def close(self):
        if self.closed or not self.enabled:return
        try:
            if self.error is None:self.flush()
            if self.stream is not None:self.stream.close()
        except OSError as error:
            self.error=f'{type(error).__name__}: {error}'
            self.failed_events+=len(self.buffer)
            print(f'[DETAIL_TRACE_ERROR] {self.error}',file=sys.stderr,flush=True)
        self.closed=True
        if self.path is not None:
            try:
                path=self.directory/f'summary_{self.pid}.json'
                temp=path.with_suffix('.tmp')
                temp.write_text(json.dumps(self.receipt(),ensure_ascii=False,indent=2)+'\n')
                temp.replace(path)
            except OSError as error:
                print(f'[DETAIL_TRACE_ERROR] cannot publish completeness receipt: {error}',file=sys.stderr,flush=True)
