#!/bin/bash
# 2026-09-12 线程环境电池: 改OMP_NUM_THREADS重跑ML全23chunk, 检验IC序列是否随线程数漂移
# 背景: A(9/11夜) vs B(9/12晨) 在bit级相同输入上IC逐chunk差±0.002。文件通道已全排除,
#       最后可测物理杠杆=训练时的OpenMP线程数(浮点归约顺序随线程数变)。
# 判定: 某线程数下IC序列=IC_A → 机制实锤; 全部=IC_B → 线程通道关闭。
# 串行执行 (勿与其他重任务并行), 每组~8min, 共~25min。
cd /mnt/d/quant/strategy || exit 1
for t in 1 8 20; do
  echo "=== OMP_NUM_THREADS=$t @ $(date +%H:%M:%S) ==="
  OMP_NUM_THREADS=$t python analysis/probe_ml_determinism_v2_0912.py \
    > logs/probe_threads_omp${t}_0912.log 2>&1
  grep -E "^\[cmp\]|^\[check\]" logs/probe_threads_omp${t}_0912.log
done
echo "=== battery done @ $(date +%H:%M:%S) ==="
