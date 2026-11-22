"""Serial cold runs of maintained code; preserve the latest live-run baseline."""
import argparse
import ast
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from single_source_runtime import prepare_layout, build_command

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'logs/live_prepare_20261116'
RUNS=Path('/home/liusong/quant_mainline_runs')


def read(path):return json.loads(Path(path).read_text())


def save(path,value):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    temp.replace(path)


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def identity():
    return {str(p.relative_to(ROOT)):sha(p) for p in [
        *sorted((ROOT/'strategy/core').glob('*.py')),ROOT/'strategy/bt_execution.py',
        ROOT/'strategy/config/factor_config.yaml',Path(__file__),
        ROOT/'strategy/tools/single_source_runtime.py',ROOT/'strategy/analysis/review_metrics.py',
        ROOT/'strategy/tools/run_406_acceptance.py',ROOT/'.arms/check_full_signal_identity_20261111.py']}


def inputs_identity():
    """Execute the exact engine input-role function without importing its job."""
    sys.path.insert(0,str(ROOT/'strategy'))
    from core.config_loader import load_config
    from core.cmap_hist import resolve_hist_path
    source=ROOT/'strategy/bt_execution.py'
    function=next(n for n in ast.parse(source.read_text()).body
                  if isinstance(n,ast.FunctionDef) and n.name=='_signal_data_fingerprint')
    ns=dict(os=os,__file__=str(source),DATA_PATH=str(ROOT/'data/stock_data/backtrader_data'),
        FUNDAMENTAL_PATH=str(ROOT/'data/stock_data/fundamental_data'),_PROJECT_DIR=str(ROOT),
        _resolve_cmap_hist=resolve_hist_path,config=load_config())
    exec(compile(ast.Module(body=[function],type_ignores=[]),str(source),'exec'),ns)
    return ns[function.name]()


def evaluate(out):
    sys.path.insert(0,str(ROOT/'strategy/analysis'))
    import pandas as pd
    from review_metrics import account_metrics,forward_labels,summarize_labels
    result=out/'strategy/rolling_validation_results'
    account=account_metrics(result)
    labels=forward_labels(result,ROOT/'data/stock_data/backtrader_data',
                          entry_basis='fills',horizons=(1,2,3,4,5,6))
    calendar=pd.DatetimeIndex(pd.read_csv(ROOT/'data/stock_data/backtrader_data/sh000001_qfq.csv',
        parse_dates=['datetime']).datetime).sort_values().unique()
    positions=calendar.get_indexer(pd.to_datetime(labels.date));assert (positions>=0).all()
    labels['exit_date']=calendar[positions+labels.horizon.to_numpy()].values
    labels=labels.loc[labels.exit_date.le(pd.Timestamp(account['end']))].copy()
    normal=summarize_labels(labels[labels.horizon.isin((1,3,5))],bootstrap_samples=0)
    full=labels[labels.horizon.isin((2,4,6))].copy();full['horizon']-=1
    report=dict(account=account,entry_day_included_1_3_5=normal,
        full_days_after_buy_1_3_5=summarize_labels(full,bootstrap_samples=0),
        label_basis='Actual filled entry costs; two explicit horizon clocks, net marks, not guaranteed exits.')
    save(out/'account_and_horizons.json',report)
    labels.to_csv(out/'labels.csv',index=False)
    return report


def inside(out):
    manifest=read(out/'manifest.json');assert identity()==manifest['source_sha256']
    assert inputs_identity()==manifest['baseline_inputs_sha256']
    for relative in ('strategy/cache','strategy/models','strategy/rolling_validation_results','strategy/tmp_factor'):
        assert not list((ROOT/relative).iterdir()),f'nonempty derived directory: {relative}'
    with (out/'engine.log').open('x') as log:
        process=subprocess.Popen([sys.executable,'-B','-u',str(ROOT/'strategy/bt_execution.py')],
            cwd=ROOT/'strategy',stdout=log,stderr=subprocess.STDOUT)
        save(out/'engine_process.json',dict(pid=process.pid,started_at=datetime.datetime.now().isoformat()))
        assert process.wait()==0,'Engine failed; see engine.log'
    expected=read(ROOT/'strategy/rolling_validation_results/backtest_signals.csv.inputs.json')
    assert expected['inputs_sha256']==manifest['baseline_inputs_sha256']
    assert inputs_identity()==manifest['baseline_inputs_sha256']
    assert identity()==manifest['source_sha256']
    evaluate(out)
    save(out/'completion.json',dict(source_and_inputs_unchanged=True,
        prior_computed_cache_reused=False,completed_at=datetime.datetime.now().isoformat()))


def environment(out,policy):
    env={k:v for k,v in os.environ.items() if not k.startswith(('QUANT_','POOL_'))
         and k not in ('FACTOR_CONFIG_PATH','PYTHONPATH','PYTHONHOME','PYTHONPYCACHEPREFIX')}
    env.update(QUANT_ALT_NO_AUTOREFRESH='1',QUANT_ALT_VERSIONS_REQUIRED='1',
        NUMBA_CACHE_DIR=str(out/'runtime/numba'),XDG_CACHE_HOME=str(out/'runtime/xdg'),
        PYTHONPYCACHEPREFIX=str(out/'runtime/pycache'),PYTHONDONTWRITEBYTECODE='1')
    env.update(policy)
    return env


def launch(root,name,policy,hashes):
    out=root/name;layout=prepare_layout(ROOT,ROOT/'data',out)
    manifest=dict(source_sha256=hashes,source_root=str(ROOT),source_copied=False,
        financial_baseline=str(BASE/'latest_data_baseline.json'),policy=policy,workers=8,
        baseline_inputs_sha256=read(ROOT/'strategy/rolling_validation_results/backtest_signals.csv.inputs.json')['inputs_sha256'],
        generation_policy='Empty factors/models/predictions/signals/context/pool/Numba; no previous-run cache.',
        scope='engineering_exact: only component-contract fixes; industry_pit: additionally export industry using each decision date.')
    save(out/'manifest.json',manifest)
    env=environment(out,policy)
    # Verify the namespace and empty outputs before the expensive calculation.
    probe=[sys.executable,'-B','-c',
        "from pathlib import Path; assert not list(Path('cache').iterdir()); assert not list(Path('models').iterdir()); print(Path.cwd())"]
    with (out/'namespace_probe.log').open('x') as log:
        assert subprocess.run(build_command(layout,probe),env=env,stdout=log,stderr=subprocess.STDOUT).returncode==0
    save(root/'state.json',dict(stage='running',arm=name,output=str(out),
        started_at=datetime.datetime.now().isoformat(),source_frozen=True))
    argv=[sys.executable,'-B','-u',str(Path(__file__).resolve()),'--inside','--output',str(out)]
    with (out/'launcher.log').open('x') as log:
        result=subprocess.run(build_command(layout,argv),env=env,stdout=log,stderr=subprocess.STDOUT)
    assert result.returncode==0,f'{name} failed; see {out}/launcher.log'
    assert identity()==hashes
    return out


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path)
    p.add_argument('--inside',action='store_true');a=p.parse_args();root=a.output.resolve()
    if a.inside:inside(root);return
    root.mkdir(parents=True,exist_ok=False);hashes=identity()
    save(root/'plan.json',dict(source_sha256=hashes,baseline=str(BASE),
        sequence=['engineering_exact','industry_pit'],production_output_overwrite=False,
        no_cache=True,source_frozen_until_queue_completes=True))
    try:
        with (ROOT/'.daily_chain.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            repaired=launch(root,'engineering_exact',{},hashes)
            import pandas as pd
            for name in ('equity_curve.csv','portfolio_selections.csv','trade_realized.csv','regime_state.csv','execution_fills.csv'):
                pd.testing.assert_frame_equal(pd.read_csv(BASE/'baseline_account'/name,dtype={'code':str}),
                    pd.read_csv(repaired/'strategy/rolling_validation_results'/name,dtype={'code':str}),check_exact=True)
            old=read(BASE/'latest_data_baseline.json');new=read(repaired/'account_and_horizons.json')
            for key in ('account','entry_day_included_1_3_5','full_days_after_buy_1_3_5'):
                assert old[key]==new[key],key
            from run_406_acceptance import compare_signals
            signals=compare_signals(ROOT/'strategy/rolling_validation_results/backtest_signals.csv.store.parquet',
                repaired/'strategy/rolling_validation_results/backtest_signals.csv.store.parquet')
            save(repaired/'acceptance.json',dict(passed=True,five_account_tables_exact=True,
                all_signal_values_exact=signals,account_and_1_3_5_exact=True,profit_improvement_claimed=False))
            case=launch(root,'industry_pit',{'QUANT_INDUSTRY_PIT':'1'},hashes)
            baseline=read(repaired/'account_and_horizons.json');candidate=read(case/'account_and_horizons.json')
            save(root/'comparison.json',dict(baseline=baseline,candidate=candidate,
                return_delta=candidate['account']['total_return']-baseline['account']['total_return'],
                promotion=False,attribution_required=True))
            save(root/'state.json',dict(stage='completed',finished_at=datetime.datetime.now().isoformat(),
                source_frozen=False,attribution_required=True,production_promoted=False))
    except BaseException as error:
        save(root/'state.json',dict(stage='failed',error=repr(error),at=datetime.datetime.now().isoformat(),production_promoted=False))
        raise


if __name__=='__main__':main()
