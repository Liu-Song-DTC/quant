"""Explicitly requested465 research acceptance on newly generated inputs.

One maintained source tree; private writable outputs. Regenerate primary
factors/models/predictions/signals, then apply the original sealed465 ranking
artifact to those NEW signals. This is not a claim that the secondary ranking
training pipeline has been rebuilt or that historical membership is PIT.
Publication requires a separate, affirmative release review; never auto-push
merely because account PnL matches.
"""
from datetime import datetime
from dataclasses import replace
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from run_latest_review_experiments import environment, identity, inputs_identity, read, save, sha, evaluate
from single_source_runtime import prepare_layout, build_command

REFERENCE = Path('/home/liusong/quant_mainline_runs/B1_origin_scope_accounts_20261121_211912/close_rebound_only')
BASELINE = Path('/home/liusong/quant_mainline_runs/exact330_original_cold_20261120_071747/trial')
POINTER = ROOT/'logs/optimization_20261121/acceptance_465_progress.json'


def generate(out):
    """Use the same public preparation functions and main-entry input contract."""
    manifest = read(out/'manifest.json')
    assert identity() == manifest['source_sha256']
    assert inputs_identity() == manifest['baseline_inputs_sha256']
    for name in ('cache', 'models', 'rolling_validation_results', 'tmp_factor'):
        assert not list((ROOT/'strategy'/name).iterdir()), name
    assert os.environ.get('QUANT_SIGNAL_REPLAY') != '1'
    sys.path.insert(0, str(ROOT/'strategy'))
    import bt_execution as engine
    assert Path(engine.__file__).resolve() == ROOT/'strategy/bt_execution.py'
    assert not engine.config.get('industry_sentiment.enabled', False), 'Mirror sentiment orchestration before enabling this profile'
    membership = engine._load_pool_membership()
    codes = []
    for name in os.listdir(engine.DATA_PATH):
        if name.startswith('._'):
            continue
        if name.endswith('_qfq.csv') and name != 'sh000001_qfq.csv':
            codes.append(name.replace('_qfq.csv', ''))
        elif name.endswith('_hfq.csv') and name != 'sh000001_hfq.csv':
            codes.append(name.replace('_hfq.csv', ''))
    if engine.config.get('stock_pool.enabled', True):
        if membership is not None:
            union = set().union(*membership.values())
        else:
            union = engine.get_stock_pool(todate=engine._pool_todate(),
                bse_exclude=engine.config.get('stock_pool.bse_exclude', True))
        codes = [c for c in codes if c in union]
    fundamental = engine.FundamentalData(engine.FUNDAMENTAL_PATH, codes)
    strategy = engine.Strategy(init_cash=engine.CASH, fundamental_data=fundamental,
                               sentiment_orchestrator=None)
    cerebro = engine.bt.Cerebro()
    before_code = engine._signal_code_fingerprint()
    before_data = engine._signal_data_fingerprint()
    print('[465 ACCEPTANCE] Cold primary generation; no old signals, factors or primary models reused.', flush=True)
    print(f'[465 ACCEPTANCE] Fundamental universe={len(codes)}; canonical source={engine.__file__}', flush=True)
    engine.add_data_and_signal(cerebro, strategy, fundamental,
        membership_map=membership, prepare_backtrader_feeds=False)
    assert engine._signal_code_fingerprint() == before_code
    assert engine._signal_data_fingerprint() == before_data
    signals = ROOT/'strategy/rolling_validation_results/backtest_signals.csv'
    engine.save_context(strategy, str(signals))
    from core.input_identity import write_signal_identity
    write_signal_identity(str(signals), before_data)
    from core.live_state import atomic_bytes
    atomic_bytes(signals.parent/'.signal_code_fp',
        f'{before_code}|{engine._B1V2_WARMUP_DAYS}'.encode())
    save(out/'generation_complete.json', dict(at=datetime.now().isoformat(),
        primary_models_retrained=True, factors_regenerated=True, signals_regenerated=True,
        old_primary_cache_reused=False, source_sha256=identity(), inputs_sha256=before_data,
        signal_sha256=sha(signals), secondary_rank_retrained=False))


def main():
    out = Path('/mnt/d/quant_research_runs')/('acceptance_465_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    layout = prepare_layout(ROOT, ROOT/'data', out)
    def stage(name, **fields):
        record = dict(stage=name, at=datetime.now().isoformat(), controller_pid=os.getpid(),
                      output=str(out), **fields)
        save(POINTER, record); print(json.dumps(record, ensure_ascii=False), flush=True)
    def run(label, command, policy):
        log = out/(label+'.log')
        start = time.monotonic()
        # Some existing tests read canonical research helpers under logs/.
        # Keep that source directory read-only during preflight instead of
        # hiding it behind the empty output directory. Trading jobs still use
        # all normal output overlays.
        job_layout = (replace(layout, writable_dirs=tuple(
            name for name in layout.writable_dirs if name != 'logs'))
            if label == 'preflight_tests' else layout)
        with log.open('x') as stream:
            process = subprocess.Popen(build_command(job_layout, command), env=environment(out, policy),
                                       stdout=stream, stderr=subprocess.STDOUT)
            while process.poll() is None:
                mem = {r.split(':')[0]: int(r.split()[1]) for r in Path('/proc/meminfo').read_text().splitlines()}
                disk = os.statvfs(out)
                stage(label, engine_pid=process.pid, log=str(log), elapsed_seconds=time.monotonic()-start,
                      log_bytes=log.stat().st_size, available_memory_MiB=mem['MemAvailable']/1024,
                      disk_free_GiB=disk.f_bavail*disk.f_frsize/1024**3)
                time.sleep(15)
            assert process.returncode == 0, f'{label} failed ({process.returncode}); see {log}'
    with (ROOT/'.daily_chain.lock').open('a') as lock:
        stage('preflight_lock')
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        original = read(REFERENCE/'manifest.json')
        sources = identity(); data = inputs_identity()
        assert data == read(BASELINE/'manifest.json')['baseline_inputs_sha256'], '465 input data changed'
        policy = dict(original['policy'])
        # The original experiment did not enable daily-industry PIT. Retain
        # that exact experiment scope explicitly, and do not call it PIT-safe.
        policy.update(QUANT_INDUSTRY_PIT='0', QUANT_ALT_MANIFEST_SNAPSHOT='1',
            QUANT_DETAIL_TRACE='1', QUANT_DETAIL_TRACE_DIR=str(ROOT/'strategy/rolling_validation_results/decision_trace'),
            STRATEGY_QUIET='0')
        policy.pop('QUANT_EXECUTION_QUOTE_CACHE', None)  # fresh raw quote loading
        table = Path(policy['QUANT_RESEARCH_CONDITIONAL_ORDER'])
        assert sha(table) == policy['QUANT_RESEARCH_CONDITIONAL_ORDER_SHA256']
        sys.path.insert(0, str(ROOT/'strategy'))
        from core.conditional_candidate_order import ConditionalCandidateOrder
        order = ConditionalCandidateOrder(table, sha(table), preserve_b1_origin=True,
                    b1_origin_scope='close_rebound_not_open_gap')
        del order
        sys.path.insert(0, str(ROOT/'logs/optimization_20261121'))
        from run_detailed_observation_acceptance import peer_context_identity, validate_trace
        peer = Path(policy['QUANT_B2_PROFIT_CONTEXT'])
        peer_hash = peer_context_identity(peer)
        delta = {p: dict(original=h, current=sources.get(p)) for p,h in original['source'].items()
                 if sources.get(p) != h}
        save(out/'manifest.json', dict(source=sources, source_sha256=sources,
            runner_sha256=sha(Path(__file__)), baseline_inputs_sha256=data,
            policy=policy, reference=str(REFERENCE), baseline=str(BASELINE),
            original_source_delta=delta, original_ranking_sha256=sha(table),
            peer_context=peer_hash, original_expected=read(REFERENCE/'comparison.json')['candidate'],
            scope='Explicit465 acceptance: cold primary pipeline then original465 rules on only this job new signals. Current logging/equivalent manifest acceleration and isolated warmup fix retained and disclosed.',
            original_primary_cache_reused=False, secondary_rank_retrained=False,
            original_industry_policy_retained=True, pristine_holdout=False, promoted=False,
            publication_gates_pending=['secondary-ranking reproducible build/provenance',
                                       'industry and other time-boundary audit',
                                       'full financial and trade-table validation']))
        cold_policy = {k:v for k,v in policy.items() if k in (
            'QUANT_INDUSTRY_PIT','QUANT_ALT_MANIFEST_SNAPSHOT','QUANT_DETAIL_TRACE',
            'QUANT_DETAIL_TRACE_DIR','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','STRATEGY_QUIET')}
        # Each isolated phase restarts its PID namespace (main process PID2).
        # Never share PID-named journal streams between generation and account.
        cold_policy['QUANT_DETAIL_TRACE_DIR'] = str(out/'generation_trace')
        tests = ['test_candidate_timing.py','test_added_capital_confirmation.py',
                 'test_profit_guard.py','test_conditional_candidate_order.py',
                 'test_signal_dependency_fingerprint.py','test_factor_warmup_clock.py']
        test_policy = dict(cold_policy, PYTHONPATH=str(ROOT/'strategy'),
                          QUANT_DETAIL_TRACE_DIR=str(out/'preflight_trace'))
        run('preflight_tests',[sys.executable,'-B','-m','pytest','-q','-p','no:cacheprovider',
            *[str(ROOT/'strategy/tests'/name) for name in tests]],test_policy)
        assert identity() == sources
        assert os.statvfs(out).f_bavail*os.statvfs(out).f_frsize > 16*1024**3
        run('primary_generation',[sys.executable,'-B','-u',str(Path(__file__).resolve()),
                                  '--generate',str(out)], cold_policy)
        receipt = read(out/'generation_complete.json')
        assert receipt['source_sha256'] == sources and receipt['inputs_sha256'] == data
        policy['QUANT_SIGNAL_REPLAY'] = '1'
        policy['QUANT_SIGNAL_REPLAY_SHA256'] = receipt['signal_sha256']
        save(out/'executed_policy.json', policy)
        # Reuse only the newly produced signal tape in this acceptance job.
        # The strict ranking-source checks must fail on any incompatible row.
        run('candidate_account',[sys.executable,'-B','-u',str(ROOT/'strategy/bt_execution.py')],policy)
        stage('validating_account')
        assert identity() == sources and inputs_identity() == data
        assert sha(table) == original['policy']['QUANT_RESEARCH_CONDITIONAL_ORDER_SHA256']
        assert peer_context_identity(peer) == peer_hash
        metrics = evaluate(out)
        traces = validate_trace(out)
        from analyze_short_target_full import analyze
        analyze(out, out/'account_diagnostics')
        old = read(REFERENCE/'comparison.json')['candidate']
        observed_return = metrics['account']['total_return']*100
        observed_dd = abs(metrics['account']['max_drawdown'])*100
        reproduction = abs(observed_return-old['return_pct'])<1e-6 and abs(observed_dd-old['drawdown_pct'])<1e-6
        save(out/'acceptance.json', dict(completed_at=datetime.now().isoformat(),
            original_return_pct=old['return_pct'], observed_return_pct=observed_return,
            original_drawdown_pct=old['drawdown_pct'], observed_drawdown_pct=observed_dd,
            financial_reproduction_passed=reproduction, trace_validation=traces,
            full_primary_regeneration=True, secondary_rank_retrained=False,
            source_and_inputs_unchanged=True, publication_passed=False,
            remaining=['Audit/rebuild ranking artifact and review industry temporal behavior before any GitHub promotion.'],
            github_pushed=False))
        stage('account_complete_release_review_pending', financial_reproduction_passed=reproduction,
              report=str(out/'acceptance.json'), github_pushed=False)


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--generate', type=Path)
    args=parser.parse_args()
    if args.generate:
        generate(args.generate)
    else:
        try:
            main()
        except BaseException as error:
            previous=read(POINTER) if POINTER.exists() else {}
            previous.update(stage='failed', at=datetime.now().isoformat(), error=repr(error), github_pushed=False)
            save(POINTER,previous)
            raise
