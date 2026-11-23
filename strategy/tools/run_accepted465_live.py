"""Cold daily signals and live plan through the shared accepted465 policy.

Run after the explicit data refresh, e.g.:
    .venv/bin/python -u strategy/tools/run_accepted465_live.py --execution-date 2026-11-23

Maintained source/data are read-only during generation; all derived inputs
start empty. Live planning consumes the new signal/context bundle directly.
This command writes a plan, never submits a broker order or sends a message.
"""
import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'strategy/tools'))
from single_source_runtime import prepare_layout, build_command
from run_latest_review_experiments import environment, identity as research_identity, inputs_identity, save, sha

STATUS=ROOT/'logs/live_accepted465_status.json'


def identity():
    # The deployed tree is the committed maintained source. Unimported private
    # research helpers left in core/ are not part of a release manifest.
    tracked=set(subprocess.check_output(['git','ls-files'],cwd=ROOT,text=True).splitlines())
    return {path:digest for path,digest in research_identity().items() if path in tracked}


def generate(out):
    """Use the same primary APIs as the regenerated465 acceptance."""
    manifest=json.loads((out/'manifest.json').read_text())
    assert identity()==manifest['source_sha256']
    assert inputs_identity()==manifest['inputs_sha256']
    sys.path.insert(0,str(ROOT/'strategy'))
    import bt_execution as engine
    for name in ('cache','models','rolling_validation_results','tmp_factor'):
        assert not list((ROOT/'strategy'/name).iterdir()),f'Cold generation requires empty {name}'
    assert os.environ.get('QUANT_SIGNAL_REPLAY')!='1'
    assert engine._DEPLOYMENT_PROFILE['name']=='accepted465'
    membership=engine._load_pool_membership()
    codes=[]
    for name in os.listdir(engine.DATA_PATH):
        if name.startswith('._'):continue
        if name.endswith('_qfq.csv') and name!='sh000001_qfq.csv':codes.append(name[:-8])
        elif name.endswith('_hfq.csv') and name!='sh000001_hfq.csv':codes.append(name[:-8])
    if engine.config.get('stock_pool.enabled',True):
        if membership is not None:pool=set().union(*membership.values())
        else:pool=engine.get_stock_pool(todate=engine._pool_todate(),
                bse_exclude=engine.config.get('stock_pool.bse_exclude',True))
        codes=[c for c in codes if c in pool]
    fundamental=engine.FundamentalData(engine.FUNDAMENTAL_PATH,codes)
    sentiment=None
    if engine.config.get('industry_sentiment.enabled',False) and engine.SENTIMENT_AVAILABLE:
        sentiment=engine.SentimentOrchestrator(engine.config,backtest_mode=True)
    strategy=engine.Strategy(init_cash=engine.CASH,fundamental_data=fundamental,
                             sentiment_orchestrator=sentiment)
    code_before=engine._signal_code_fingerprint()
    data_before=engine._signal_data_fingerprint()
    print(f'[accepted465] cold primary generation; workers={engine.NUM_WORKERS}; '
          f'signal date={manifest["signal_date"]}; no old signals/factors/models reused',flush=True)
    engine.add_data_and_signal(engine.bt.Cerebro(),strategy,fundamental,
                              membership_map=membership,prepare_backtrader_feeds=False)
    assert engine._signal_code_fingerprint()==code_before
    assert engine._signal_data_fingerprint()==data_before
    signals=ROOT/'strategy/rolling_validation_results/backtest_signals.csv'
    engine.save_context(strategy,str(signals))
    from core.input_identity import write_signal_identity
    from core.live_state import atomic_bytes
    write_signal_identity(str(signals),data_before)
    atomic_bytes(signals.parent/'.signal_code_fp',f'{code_before}|{engine._B1V2_WARMUP_DAYS}'.encode())
    latest=strategy.signal_store.max_date()
    assert str(latest)[:10]==manifest['signal_date'],f'Stale generated signals: {latest}'
    save(out/'generation_complete.json',dict(completed_at=datetime.now().isoformat(),
        profile='accepted465',signal_date=str(latest)[:10],signal_sha256=sha(signals),
        inputs_sha256=data_before,source_fingerprint=code_before,caches_reused=False))
    print('[accepted465] primary signals and matching context complete',flush=True)


def main(execution_date):
    import pandas as pd
    sys.path.insert(0,str(ROOT/'strategy'))
    from core.deployment_profile import activate_profile
    os.environ['QUANT_TRADING_PROFILE']='accepted465'
    profile=activate_profile()
    index=ROOT/'data/stock_data/backtrader_data/sh000001_qfq.csv'
    day=pd.read_csv(index,usecols=['datetime']).datetime.max()[:10]
    if pd.Timestamp(execution_date)<=pd.Timestamp(day):
        raise ValueError('Execution date must follow the completed signal session')
    configured_peer=Path(os.environ['QUANT_B2_PROFIT_CONTEXT'])
    out=Path('/mnt/d/quant_research_runs')/('live_accepted465_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    def stage(name,**fields):
        row=dict(stage=name,at=datetime.now().isoformat(),controller_pid=os.getpid(),
                 output=str(out),signal_date=day,execution_date=execution_date,**fields)
        save(STATUS,row);print(json.dumps(row,ensure_ascii=False),flush=True)
    with (ROOT/'.daily_chain.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for script in ('strategy/check_data_freshness.py','data/verify_alt_observations.py'):
            subprocess.run([sys.executable,'-B',str(ROOT/script)],cwd=ROOT,check=True)
        from prepare_accepted465_peers import ensure_peer_context
        peer=ensure_peer_context(root=ROOT,end=day,current_path=configured_peer)
        os.environ['QUANT_B2_PROFIT_CONTEXT']=str(peer)
        if pd.read_csv(index,usecols=['datetime']).datetime.max()[:10]!=day:
            raise RuntimeError('Index calendar changed while preparing the live inputs')
        layout=prepare_layout(ROOT,ROOT/'data',out)
        policy=dict(QUANT_TRADING_PROFILE='accepted465',QUANT_TODATE=day,
            QUANT_B2_PROFIT_CONTEXT=str(peer),QUANT_DETAIL_TRACE='1',
            QUANT_DETAIL_TRACE_DIR=str(out/'generation_trace'),
            OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',STRATEGY_QUIET='0')
        sources=identity();inputs=inputs_identity()
        revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        dirty=subprocess.run(['git','diff','--quiet','HEAD','--',*sources,
            'strategy/generate_trade_orders.py','strategy/tools/run_accepted465_live.py',
            'strategy/analysis/bear_reference_orders.py',
            'strategy/config/deployment/accepted465.json','strategy/check_data_freshness.py'],cwd=ROOT)
        if dirty.returncode:
            raise RuntimeError('Live source differs from the committed release; refusing an ambiguous version')
        protected={str(p):sha(p) for p in (Path(__file__),
            ROOT/'strategy/generate_trade_orders.py',
            ROOT/'strategy/analysis/bear_reference_orders.py',
            ROOT/'strategy/check_data_freshness.py',
            ROOT/'data/stock_data/kline_stale_allowlist.txt',
            ROOT/'strategy/config/deployment/accepted465.json',
            ROOT/profile['ranking']['path'],peer/'manifest.json')}
        def assert_inputs_unchanged():
            assert identity()==sources and inputs_identity()==inputs,'Source or inputs changed'
            assert all(sha(Path(p))==h for p,h in protected.items()),'Deployment artifact changed'
            assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==revision,'Release commit changed'
        save(out/'manifest.json',dict(profile=profile,policy=policy,signal_date=day,
            execution_date=execution_date,source_sha256=sources,inputs_sha256=inputs,
            runner_sha256=sha(Path(__file__)),peer_manifest_sha256=sha(peer/'manifest.json'),
            source_root=str(ROOT),source_copied=False,cold_primary=True,
            ranking_domain_expanded=False,orders_are_plans=True,protected_sha256=protected,
            deployed_commit=revision))
        log=out/'generation.log'
        log_pointer=ROOT/('logs/live_accepted465_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'.log')
        log_pointer.symlink_to(log)
        env=environment(out,policy)
        with log.open('x') as stream:
            process=subprocess.Popen(build_command(layout,[sys.executable,'-B','-u',str(Path(__file__)),
                '--generate',str(out)]),env=env,stdout=stream,stderr=subprocess.STDOUT)
            while process.poll() is None:
                memory={r.split(':')[0]:int(r.split()[1]) for r in Path('/proc/meminfo').read_text().splitlines()}
                stage('cold_signal_generation',engine_pid=process.pid,log=str(log_pointer),
                      available_memory_MiB=memory['MemAvailable']/1024,log_bytes=log.stat().st_size)
                time.sleep(10)
        assert process.returncode==0,f'Signal generation failed ({process.returncode}); {log}'
        assert_inputs_unchanged()
        stage('live_plan_preview')
        order_env=dict(env,QUANT_SIGNALS_PATH=str(out/'strategy/rolling_validation_results/backtest_signals.csv'),
            QUANT_DETAIL_TRACE_DIR=str(out/'live_trace'))
        argv=[sys.executable,'-B','-u',str(ROOT/'strategy/generate_trade_orders.py'),
              '--date',execution_date,'--preopen']
        # Preview validates the exact same CLI/decision code without committing
        # an account/plan. The final call retains the real live-account lock.
        with (out/'live_preview.log').open('x') as stream:
            subprocess.run([*argv,'--dry-run','--bear-reference-report',str(out/'bear_reference_plan.json')],
                           env=dict(order_env,QUANT_DETAIL_TRACE_DIR=str(out/'live_preview_trace')),cwd=ROOT,stdout=stream,
                           stderr=subprocess.STDOUT,check=True)
        assert_inputs_unchanged()
        stage('commit_live_plan',log=str(out/'live_plan.log'))
        with (out/'live_plan.log').open('x') as stream:
            subprocess.run(argv,env=order_env,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
        orders=json.loads((ROOT/'trade_orders.json').read_text())
        assert orders['trading_profile']=='accepted465'
        assert orders['signal_date']==day and orders['execution_date']==execution_date
        generated=json.loads((out/'generation_complete.json').read_text())
        assert orders['signal_identity']['signals_sha256']==generated['signal_sha256']
        assert orders['signal_identity']['inputs_sha256']==inputs
        reference=json.loads((out/'bear_reference_plan.json').read_text())
        if reference.get('status') in ('complete','no_eligible_reference_orders'):
            from analysis.bear_reference_orders import attach_reference_report
            orders=attach_reference_report(ROOT,reference)
        save(out/'live_plan_receipt.json',dict(profile='accepted465',signal_date=day,
            execution_date=execution_date,orders_file=str(ROOT/'trade_orders.json'),
            order_count=len(orders['orders']),broker_orders_submitted=False,
            bear_reference_status=reference.get('status'),
            reference_order_count=len(reference.get('reference_orders',[]))))
        stage('live_plan_complete',log=str(out/'live_plan.log'),orders_file=str(ROOT/'trade_orders.json'),
              orders=len(orders['orders']),broker_orders_submitted=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generate',type=Path)
    parser.add_argument('--execution-date')
    args=parser.parse_args()
    if args.generate:generate(args.generate)
    else:
        if not args.execution_date:parser.error('--execution-date is required')
        try:main(args.execution_date)
        except BaseException as error:
            previous=json.loads(STATUS.read_text()) if STATUS.exists() else {}
            previous.update(stage='failed',at=datetime.now().isoformat(),error=repr(error))
            save(STATUS,previous)
            raise
