"""臂运行器 (2026-10-25): 组合层臂跑隔离树 + 复用基线信号 + 严格身份校验。

为什么需要这个: 指纹改内容寻址+realpath后, 生产树与臂树的信号指纹首次一致 ⇒ 臂可以
在隔离树里跑并复用基线信号(省~30min/臂)。但基线的 sidecar 记录的是**旧算法**的指纹
⇒ 臂侧要写自己的指纹。这属于"手工改身份产物", 所以本脚本先证明:
    生成器代码差异**仅限** `_signal_code_fingerprint` 这一个函数 (AST 级比对, 去掉该
    函数后两份代码逐节点相同) ⇒ 该函数只算哈希, 不产生任何信号内容 ⇒ 复用安全。

用法: run_arm_1025.py <臂名> [--env K=V,K=V] [--fresh-signals]
"""
import ast
import os
import shutil
import subprocess
import sys
from pathlib import Path

PROD = Path('/mnt/d/quant')
PY = '/mnt/d/quant/.venv/bin/python'
FINGERPRINT_FN = '_signal_code_fingerprint'


def strip_fn(path, fname=FINGERPRINT_FN):
    tree = ast.parse(Path(path).read_text(encoding='utf-8'))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == fname:
            node.body = [ast.Pass()]
    return ast.dump(tree)


def verify_generator_delta(base_file, arm_file):
    a = strip_fn(base_file)
    b = strip_fn(arm_file)
    if a != b:
        raise SystemExit('拒绝: 生成器代码差异超出指纹函数范围, 不能手工改 sidecar')
    print('[identity] 生成器差异已验证: 仅限 _signal_code_fingerprint (去掉该函数后 AST 逐节点相同)')


def arm_fingerprint(arm_strategy):
    code = ("import sys; sys.path.insert(0,'.');"
            "from bt_execution import _signal_code_fingerprint;"
            "print(_signal_code_fingerprint())")
    out = subprocess.run([PY, '-c', code], cwd=arm_strategy,
                         capture_output=True, text=True)
    lines = [ln.strip() for ln in out.stdout.strip().splitlines() if ln.strip()]
    if not lines:
        raise SystemExit(f'指纹计算失败: {out.stderr[-800:]}')
    return lines[-1]


def main():
    name = sys.argv[1]
    env_args = []
    fresh = False
    for i, a in enumerate(sys.argv):
        if a == '--env':
            env_args = [kv for kv in sys.argv[i + 1].split(',') if kv]
        if a == '--fresh-signals':
            fresh = True
    arm = PROD / '.review_runs' / name
    arm_strategy = arm / 'strategy'

    setup = PROD / '.arms/setup_arm_1025.sh'
    cmd = ['bash', str(setup), name] + (['--fresh-signals'] if fresh else [])
    subprocess.run(cmd, check=True)

    # 复制基线信号产物 + raw 缓存 (源 = 基线树, 不是生产树!
    # 生产树的 backtest_signals.csv 还是 1024 锚点那份, 与新基线不同源 ⇒ 不可比)
    base_tree = PROD / '.review_runs/arm_base_1025/strategy'
    rr = arm_strategy / 'rolling_validation_results'
    rr.mkdir(parents=True, exist_ok=True)
    for f in (() if fresh else ('backtest_signals.csv', 'backtest_signals.csv.store.parquet',
              'backtest_signals.csv.store.json', 'execution_context.pkl',
              'ml_preds_worker.parquet')):
        src = base_tree / 'rolling_validation_results' / f
        dst = rr / f
        if src.exists() and not dst.exists():
            print(f'[arm] 复制基线信号产物 {f} ...')
            # copy2 保留 mtime: execution_context 里记的 signals_stat=(size, mtime_ns)
            # 必须与复制后的 CSV 对得上, 否则 load_context 判失效 → 白跑一次重生成
            shutil.copy2(src, dst)
    cache = arm_strategy / 'cache'
    cache.mkdir(parents=True, exist_ok=True)
    for f in (base_tree / 'cache').glob('factor_df_*.parquet'):
        shutil.copy2(f, cache / f.name)
    for f in (base_tree / 'cache').glob('pool_membership_*.parquet'):
        shutil.copy2(f, cache / f.name)
    # ML chunk 缓存 (模型+逐日预测): 键含代码/数据指纹, 组合层臂与基线同键 ⇒ 命中
    # ⇒ 省掉 ~11min 全量重训, 且保证臂的 ML 预测与基线逐位相同 (干净的单变量臂)
    ml_src = base_tree / 'rolling_validation_results/ml_chunk_cache'
    if ml_src.exists():
        ml_dst = rr / 'ml_chunk_cache'
        if not ml_dst.exists():
            shutil.copytree(ml_src, ml_dst)
            print(f'[arm] 复制 ML chunk 缓存 ({sum(1 for _ in ml_dst.iterdir())} 个文件)')

    # 身份校验 + 写 sidecar
    base_gen = PROD / '.review_runs/arm_base_1025/strategy/bt_execution.py'
    if base_gen.exists():
        verify_generator_delta(base_gen, arm_strategy / 'bt_execution.py')
    fp = arm_fingerprint(arm_strategy)
    sidecar = rr / '.signal_code_fp'
    sidecar.write_text(f'{fp}|0', encoding='utf-8')
    print(f'[arm] sidecar 写入指纹 {fp}|0')

    env = dict(os.environ)
    for kv in env_args:
        k, _, v = kv.partition('=')
        env[k] = v
        print(f'[arm] env {k}={v}')
    log = arm / 'arm.log'
    with open(log, 'w', encoding='utf-8') as stream:
        rc = subprocess.run([PY, '-u', 'bt_execution.py'], cwd=arm_strategy,
                            env=env, stdout=stream, stderr=subprocess.STDOUT).returncode
    print(f'[arm] 完成 rc={rc}, 日志 {log}')
    return rc


if __name__ == '__main__':
    sys.exit(main())
