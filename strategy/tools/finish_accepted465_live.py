"""Continue a freshly generated accepted465 bundle after account acceptance.

Only a reviewed live-output update may bridge the original generation commit
and this release. All primary factor/ML/signal/decision-core hashes and data
must remain identical. No factor, ML or signal regeneration occurs here.
"""
import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'strategy/tools'))
from run_latest_review_experiments import environment, inputs_identity, save, sha
from run_accepted465_live import identity

COMPATIBLE_OUTPUT_PATHS={
    'strategy/generate_trade_orders.py',
    'strategy/analysis/bear_plan_presentation.py',
    'strategy/tools/finish_accepted465_live.py',
    'strategy/tests/test_bear_plan_presentation.py',
}
STATUS=ROOT/'logs/live_accepted465_status.json'


def validate_bundle(generated):
    manifest=json.loads((generated/'manifest.json').read_text())
    receipt=json.loads((generated/'generation_complete.json').read_text())
    assert identity()==manifest['source_sha256'],'Primary/decision source changed'
    assert inputs_identity()==manifest['inputs_sha256'],'Input data changed'
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    changed=set(subprocess.check_output(['git','diff','--name-only',manifest['deployed_commit'],revision],
        cwd=ROOT,text=True).splitlines())
    assert changed.issubset(COMPATIBLE_OUTPUT_PATHS),f'Unreviewed generation-to-output changes: {changed-COMPATIBLE_OUTPUT_PATHS}'
    for path,digest in manifest['protected_sha256'].items():
        if path==str(ROOT/'strategy/generate_trade_orders.py'):continue
        assert sha(Path(path))==digest,f'Protected generation artifact changed: {path}'
    result=subprocess.run(['git','diff','--quiet','HEAD','--',*identity(),*COMPATIBLE_OUTPUT_PATHS],cwd=ROOT)
    assert result.returncode==0,'Runtime source differs from the committed release'
    signals=generated/'strategy/rolling_validation_results/backtest_signals.csv'
    assert sha(signals)==receipt['signal_sha256'],'Generated signal bytes changed'
    return manifest,receipt,revision,changed


def main(generated,execution_date,acceptance):
    # The predecessor controller must be retired after its generation child
    # exits, before this command takes the ordinary daily-chain lock.
    with (ROOT/'.daily_chain.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        check=json.loads(acceptance.read_text())
        assert check['passed'] and check['original_result_exact'] and check['current_account_completed']
        manifest,receipt,revision,changed=validate_bundle(generated)
        assert check['release_commit']==revision,'Account acceptance belongs to another release'
        assert check['generation_commit']==manifest['deployed_commit'],'Account acceptance belongs to another generation'
        day=manifest['signal_date']
        assert execution_date==manifest['execution_date']
        out=generated/'live_output';out.mkdir(exist_ok=False)
        policy=dict(manifest['policy'],
            QUANT_SIGNALS_PATH=str(generated/'strategy/rolling_validation_results/backtest_signals.csv'),
            QUANT_DETAIL_TRACE_DIR=str(out/'decision_trace'))
        save(out/'handoff.json',dict(generation_commit=manifest['deployed_commit'],release_commit=revision,
            reviewed_output_only_paths=sorted(changed),primary_and_decision_source_unchanged=True,
            inputs_unchanged=True,signals_regenerated=False,signal_sha256=receipt['signal_sha256'],
            financial_acceptance=str(acceptance)))
        def stage(name,**fields):
            save(STATUS,dict(stage=name,at=datetime.now().isoformat(),controller_pid=os.getpid(),
                output=str(generated),signal_date=day,execution_date=execution_date,
                deployed_commit=revision,**fields))
        env=environment(generated,policy)
        argv=[sys.executable,'-B','-u',str(ROOT/'strategy/generate_trade_orders.py'),
              '--date',execution_date,'--preopen']
        stage('live_plan_preview',log=str(out/'preview.log'))
        with (out/'preview.log').open('x') as stream:
            subprocess.run([*argv,'--dry-run'],env=env,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
        validate_bundle(generated)
        stage('commit_live_plan',log=str(out/'plan.log'))
        with (out/'plan.log').open('x') as stream:
            subprocess.run(argv,env=env,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
        plan=json.loads((ROOT/'trade_orders.json').read_text())
        assert plan['trading_profile']=='accepted465'
        assert plan['signal_date']==day and plan['execution_date']==execution_date
        assert plan['signal_identity']['signals_sha256']==receipt['signal_sha256']
        assert plan['signal_identity']['inputs_sha256']==manifest['inputs_sha256']
        if plan.get('buy_enabled') is False:
            # The publisher has already removed every positive-delta order.
            assert all(o['action']!='open' and
                (o['action']=='close' or int(o['shares'])<=int(o['prev_shares'])) for o in plan['orders'])
        save(out/'complete.json',dict(completed_at=datetime.now().isoformat(),profile='accepted465',
            orders_file=str(ROOT/'trade_orders.json'),signal_date=day,execution_date=execution_date,
            market_status=plan.get('market_status'),orders=len(plan['orders']),
            watchlist=len(plan.get('watchlist',[])),buy_enabled=plan.get('buy_enabled'),
            source_commit=revision,account_acceptance=str(acceptance),broker_orders_submitted=False))
        stage('live_plan_complete',log=str(out/'plan.log'),orders_file=str(ROOT/'trade_orders.json'),
            orders=len(plan['orders']),watchlist=len(plan.get('watchlist',[])),
            buy_enabled=plan.get('buy_enabled'),broker_orders_submitted=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generated',required=True,type=Path)
    parser.add_argument('--execution-date',required=True)
    parser.add_argument('--account-acceptance',required=True,type=Path)
    args=parser.parse_args()
    try:main(args.generated,args.execution_date,args.account_acceptance)
    except BaseException as error:
        previous=json.loads(STATUS.read_text()) if STATUS.exists() else {}
        previous.update(stage='live_plan_failed_held',at=datetime.now().isoformat(),error=repr(error))
        save(STATUS,previous)
        raise
