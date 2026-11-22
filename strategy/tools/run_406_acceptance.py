"""Run the maintained 406 source with native isolated outputs and sealed data.

No strategy source is copied or patched at launch. The child verifies the same
source bytes as the parent before executing the ordinary bt_execution.py.
"""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
REFERENCE=ROOT/'.review_runs/arm_tv406_clean_rebuild_20261111'
INTEGRATION=ROOT/'logs/mainline_406_integration_20261114.json'
DEFAULT_DATA=Path('/home/liusong/quant_review_native_20261113/inputs/09f50b1cb532d0c6da0785d1471f039c08bc10906b6bf4d34c7f3048bbc026a3/data')
EXPECTED_CHANGED={'strategy/bt_execution.py',*[f'strategy/core/{n}.py' for n in ('fundamental','chan_theory','signal_engine','signal_store','multi_timeframe')]}
ACTIVE_OUTPUT=None


def read(path):return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def save(path,value):
    temp=path.with_suffix(f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');temp.replace(path)


def source_identity(expected=None):
    expected=read(INTEGRATION) if expected is None else expected
    assert expected['baseline_commit']=='c3fc53d64a3253fd7fd1f193965c9b36c2f7f24c'
    baseline=read(REFERENCE/'manifest.json')['source_sha256']
    current={rel:sha(ROOT/rel) for rel in baseline}
    assert current=={rel:expected['source_sha256'][rel] for rel in baseline},'Maintained source changed since 406 integration'
    assert {rel for rel in current if current[rel]!=baseline[rel]}==EXPECTED_CHANGED
    for rel in ('strategy/tools/run_406_acceptance.py','strategy/tools/single_source_runtime.py'):
        current[rel]=sha(ROOT/rel)
    return current


def clean_environment(output):
    env={k:v for k,v in os.environ.items() if not k.startswith(('QUANT_','POOL_'))
         and k not in ('FACTOR_CONFIG_PATH','PYTHONPATH','PYTHONHOME')}
    data=str(ROOT/'data/stock_data/backtrader_data')
    env.update(PYTHONDONTWRITEBYTECODE='1',QUANT_ALT_NO_AUTOREFRESH='1',QUANT_ALT_VERSIONS_REQUIRED='1',
               QUANT_DATA_PATH=data,QUANT_LIQ_DATA_DIR=data,QUANT_QFEAT_DATA_DIR=data,
               NUMBA_CACHE_DIR=str(output/'runtime/numba'),XDG_CACHE_HOME=str(output/'runtime/xdg'))
    return env


def probe(output):
    manifest=read(output/'manifest.json')
    actual=source_identity(manifest)
    assert actual==manifest['source_sha256'],'Child source differs from parent'
    assert Path.cwd()==ROOT/'strategy'
    sys.path.insert(0,str(ROOT/'strategy'))
    from core import chan_theory,signal_engine,signal_store,fundamental
    modules={m.__name__:str(Path(m.__file__).resolve()) for m in (chan_theory,signal_engine,signal_store,fundamental)}
    assert all(Path(p).is_relative_to(ROOT/'strategy/core') for p in modules.values())
    assert hasattr(signal_engine,'_perf_scalar_datetime')
    assert hasattr(chan_theory,'_perf_compiled_quality')
    assert hasattr(chan_theory,'_legacy_analyze_bottom_fractal')
    assert 'to_numpy(copy=False)' in (ROOT/'strategy/core/signal_store.py').read_text()
    assert 'np.flatnonzero' in (ROOT/'strategy/core/fundamental.py').read_text()
    source_device=ROOT.stat().st_dev
    data_device=(ROOT/'data').stat().st_dev
    output_device=output.stat().st_dev
    assert data_device==Path(manifest['native_data']).stat().st_dev
    assert data_device!=source_device and output_device!=source_device,'Native storage mounts not active'
    assert (ROOT/'strategy/rolling_validation_results').stat().st_dev==output_device
    import numpy as np
    n=40;c=np.linspace(10.,11.,n);high=c+.1;low=c-.1;volume=np.full(n,100.)
    bottom=np.zeros(n,dtype=np.bool_);bottom[20]=True
    inputs=(c,high,low,volume,c*.99,c*.98,np.zeros(n),bottom,np.zeros(n,dtype=np.int64))
    optimized=chan_theory.analyze_bottom_fractal(*inputs)
    original=chan_theory._legacy_analyze_bottom_fractal(*inputs)
    assert optimized.keys()==original.keys()
    assert all(optimized[k].tobytes()==original[k].tobytes() for k in original)
    assert chan_theory._perf_compiled_quality.nopython_signatures,'Compiled path was not executed'
    # Component tests establish numeric equivalence. This checks actual child
    # imports and writable native compilation-cache routing, without a backtest.
    receipt=dict(passed=True,pid=os.getpid(),cwd=str(Path.cwd()),source_root=str(ROOT),
        module_paths=modules,source_hashes_match=True,source_copied=False,
        actual_data_path=str(ROOT/'data/stock_data/backtrader_data'),native_data_mount=manifest['native_data'],
        outputs=str(output),numba_cache=os.environ.get('NUMBA_CACHE_DIR'),workers=8,
        compiled_fractal_executed=True,compiled_probe_exact=True,
        source_device=source_device,data_device=data_device,output_device=output_device,
        native_storage_verified=True)
    save(output/'execution_receipt.json',receipt)
    print('ACTUAL_CHILD_SOURCE_VERIFIED',json.dumps(receipt),flush=True)


def compare_raw(left,right):
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    a,b=pq.ParquetFile(left),pq.ParquetFile(right)
    assert a.schema_arrow.remove_metadata()==b.schema_arrow.remove_metadata()
    assert a.metadata.num_rows==b.metadata.num_rows
    for col in a.schema_arrow.names:
        x,y=[p.read(columns=[col],use_threads=False)[col] for p in (a,b)]
        same=pc.or_(pc.fill_null(pc.equal(x,y),False),pc.and_(pc.is_null(x),pc.is_null(y)))
        if pa.types.is_floating(x.type):same=pc.or_(same,pc.fill_null(pc.and_(pc.is_nan(x),pc.is_nan(y)),False))
        assert pc.all(same).as_py(),f'Raw factor differs: {col}'
    return dict(rows=a.metadata.num_rows,columns=len(a.schema_arrow.names),ordered_values_exact=True)


def compare_signals(left,right):
    # Bounded memory and per-code ordering: parallel completion order is not a
    # financial difference, but no field value or within-code row may change.
    import importlib.util
    path=ROOT/'.arms/check_full_signal_identity_20261111.py'
    before=sha(path)
    spec=importlib.util.spec_from_file_location('read_only_signal_comparison',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    a,b=module.canonical_table(left),module.canonical_table(right)
    assert a==b,'Signals differ'
    assert sha(path)==before
    return dict(rows=a['rows'],columns=len(a['column_order']),comparison_tool_sha256=before)


def evaluate(output):
    import pandas as pd
    sys.path.insert(0,str(ROOT/'strategy/analysis'))
    from review_metrics import account_metrics,forward_labels
    metrics=account_metrics(output/'corrected')
    prices=ROOT/'data/stock_data/backtrader_data'
    labels=forward_labels(output/'corrected',prices,entry_basis='fills',horizons=(2,4,6))
    calendar=pd.DatetimeIndex(pd.read_csv(prices/'sh000001_qfq.csv',parse_dates=['datetime']).datetime).sort_values().unique()
    positions=calendar.get_indexer(pd.to_datetime(labels.date));assert (positions>=0).all()
    labels['exit_date']=calendar[positions+labels.horizon.to_numpy()].values
    labels=labels[labels.exit_date<=pd.Timestamp(metrics['end'])].copy()
    horizons={}
    for h,group in labels.groupby('horizon'):
        valid=group.dropna(subset=['net_mtm_return'])
        horizons[str(h-1)]=dict(samples=len(valid),wins=int((valid.net_mtm_return>0).sum()),
            net_win_rate=float((valid.net_mtm_return>0).mean()) if len(valid) else None,
            mean_net_return=float(valid.net_mtm_return.mean()) if len(valid) else None,
            blocked_exits_retained=int((~valid.exit_available).sum()))
    report=dict(account=metrics,full_days_after_buy=horizons,reference=str(REFERENCE),
        label_basis='Actual filled entry; buy day0; full 1/3/5 trading days after buy; net estimated exit marks',
        strict_no_lookahead_certified=False,limitation='Same sealed historical proxies; previously researched period')
    save(output/'account_and_horizons.json',report);labels.to_csv(output/'labels.csv',index=False)
    base=read(REFERENCE/'account_and_horizons.json')
    assert report['account']==base['account'],'Account metrics differ from 406'
    assert report['full_days_after_buy']==base['full_days_after_buy'],'Full 1/3/5 differs from 406'
    return report


def inside(output,probe_only):
    started=time.monotonic();probe(output)
    if probe_only:return
    sys.path.insert(0,str(ROOT/'strategy/analysis'))
    import review_validation as rv
    seal=read(REFERENCE/'data_snapshot.json')
    before,records=rv.input_stamp(root=ROOT)
    assert rv.content_id(records)==seal['sha256'],'Actual mounted inputs differ from 406 seal'
    assert before['adjustment_mode']['state']=='qfq_only'
    save(output/'input_files.json',records)
    manifest=read(output/'manifest.json');manifest.update(input_stamp_algorithm='sha256_file_content_v2',
        effective_data_path=before['data_path'],run_environment=before['environment'],
        run_environment_hashes=before['environment_hashes'],input_precheck_passed=True)
    save(output/'manifest.json',manifest)
    print('FULL_INPUT_SEAL_VERIFIED',len(records),flush=True)
    with (output/'corrected.log').open('x') as log:
        process=subprocess.Popen([sys.executable,'-B','-u',str(ROOT/'strategy/bt_execution.py')],
            cwd=ROOT/'strategy',stdout=log,stderr=subprocess.STDOUT)
        save(output/'engine_process.json',dict(pid=process.pid,started_at=datetime.datetime.now().isoformat(),
            command=[sys.executable,'-B','-u',str(ROOT/'strategy/bt_execution.py')],
            source_root=str(ROOT),performance_verified_in_child=True))
        returncode=process.wait()
    assert returncode==0,f'Engine failed: {returncode}; see {output}/corrected.log'
    after,again=rv.input_stamp(root=ROOT)
    unchanged=before==after and rv.content_id(again)==seal['sha256']
    save(output/'corrected_input_stamp.json',dict(before=before,after=after,unchanged=unchanged))
    assert unchanged,'Inputs changed during run'
    assert source_identity(manifest)==manifest['source_sha256'],'Source changed during run'
    results=output/'strategy/rolling_validation_results';archive=output/'corrected';archive.mkdir()
    for name in ('equity_curve.csv','portfolio_selections.csv','trade_realized.csv','regime_state.csv','execution_fills.csv'):
        shutil.copy2(results/name,archive/name)
    audits=rv.write_execution_audits(ROOT,archive)
    manifest['execution_audits']=audits;save(output/'manifest.json',manifest)
    import pandas as pd
    for name in ('equity_curve.csv','portfolio_selections.csv','trade_realized.csv','execution_fills.csv','regime_state.csv'):
        pd.testing.assert_frame_equal(pd.read_csv(REFERENCE/'corrected'/name,dtype={'code':str}),
            pd.read_csv(archive/name,dtype={'code':str}),check_exact=True)
    report=evaluate(output)
    raw_a=list((REFERENCE/'strategy/cache').glob('factor_df_*.parquet'))
    raw_b=list((output/'strategy/cache').glob('factor_df_*.parquet'))
    assert len(raw_a)==len(raw_b)==1,(raw_a,raw_b)
    raw=compare_raw(raw_a[0],raw_b[0])
    preds='strategy/rolling_validation_results/ml_preds_worker.parquet'
    assert sha(REFERENCE/preds)==sha(output/preds),'Prediction bytes differ'
    signals='strategy/rolling_validation_results/backtest_signals.csv.store.parquet'
    sig=compare_signals(REFERENCE/signals,output/signals)
    gate=dict(passed=True,reference=str(REFERENCE),source_root=str(ROOT),source_copied=False,
        fresh_raw=raw,signals=sig,prediction_bytes_equal=True,all_account_tables_equal=True,
        full_1_3_5_equal=True,workers=8,elapsed_seconds=time.monotonic()-started,
        actual_total_return=report['account']['total_return'],source_sha256=manifest['source_sha256'])
    save(output/'performance_full_gate.json',gate)
    print('MAINLINE_406_FULL_PARITY_PASSED',json.dumps({k:v for k,v in gate.items() if k!='source_sha256'}),flush=True)


def main():
    global ACTIVE_OUTPUT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--data',type=Path,default=DEFAULT_DATA)
    parser.add_argument('--probe-only',action='store_true')
    parser.add_argument('--inside',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args();output=args.output.resolve()
    if args.inside:
        ACTIVE_OUTPUT=output
        try:inside(output,args.probe_only)
        except BaseException as exc:
            save(output/'failure.json',dict(error=repr(exc),at=datetime.datetime.now().isoformat()));raise
        return
    from single_source_runtime import prepare_layout,build_command
    hashes=source_identity()
    assert output.is_relative_to(Path('/home/liusong')),'Acceptance outputs must use native filesystem'
    assert not output.exists(),'Preserve existing run; choose a new output directory'
    assert args.data.resolve()==DEFAULT_DATA.resolve(),'406 acceptance requires the verified sealed mirror'
    # Layout preparation and actual namespace construction are covered by real
    # subprocess tests; source/config/data remain read-only inside the child.
    layout=prepare_layout(ROOT,args.data.resolve(),output)
    manifest=dict(created_at=datetime.datetime.now().isoformat(),source_sha256=hashes,
        comparison_reference=str(REFERENCE),native_data=str(args.data.resolve()),source_root=str(ROOT),
        source_copied=False,baseline_commit=read(INTEGRATION)['baseline_commit'],
        generation_policy=dict(reuse_raw_factors=False,reuse_models=False,reuse_predictions=False,reuse_signals=False),
        strategy_parameters_changed=False,workers=8,probe_only=args.probe_only)
    save(output/'manifest.json',manifest)
    ACTIVE_OUTPUT=output
    command=[sys.executable,'-B','-u',str(Path(__file__).resolve()),'--inside','--output',str(output)]
    if args.probe_only:command.append('--probe-only')
    env=clean_environment(output)
    state=dict(pid=os.getpid(),stage='waiting_for_heavy_lock' if not args.probe_only else 'probing',output=str(output))
    save(output/'state.json',state)
    def execute():
        assert source_identity()==hashes,'Source changed while queued'
        state.update(stage='running_probe' if args.probe_only else 'running_406_full_acceptance',started_at=datetime.datetime.now().isoformat())
        save(output/'state.json',state)
        cmd=build_command(layout,command)
        with (output/'launcher.log').open('x') as log:
            result=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
        state.update(stage='completed' if result.returncode==0 else 'failed',returncode=result.returncode,finished_at=datetime.datetime.now().isoformat())
        save(output/'state.json',state)
        if result.returncode:raise SystemExit(result.returncode)
    if args.probe_only:execute()
    else:
        with (ROOT/'logs/buy_side_serial_supervisor_20261112.lock').open('a') as heavy:
            fcntl.flock(heavy,fcntl.LOCK_EX)
            for path in Path('/proc').iterdir():
                if not path.name.isdigit():continue
                try:text=(path/'cmdline').read_bytes().replace(b'\0',b' ').decode()
                except (OSError,UnicodeError):continue
                assert not (text.startswith(str(ROOT/'.venv/bin/python')) and 'bt_execution.py' in text),'Another account engine is still running'
            execute()


if __name__=='__main__':
    try:
        main()
    except BaseException as exc:
        if ACTIVE_OUTPUT is not None and not (isinstance(exc,SystemExit) and exc.code in (None,0)):
            state_path=ACTIVE_OUTPUT/'state.json'
            state=read(state_path) if state_path.exists() else {}
            state.update(stage='failed',error=repr(exc),finished_at=datetime.datetime.now().isoformat())
            save(state_path,state)
        raise
