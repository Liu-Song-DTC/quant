#!/usr/bin/env bash
# armguard — 臂产物 / 生产产物 的命名空间隔离 (2026-10-19)
#
# ── 为什么 ────────────────────────────────────────────────────────────────
# 2026-10-19 事故: `.arms/run_snap_c15.sh` 把整棵 strategy 树 `git show ded5dfd:`
# 回退到 10-15 旧代码跑了一臂, 产物**直接留在** `strategy/rolling_validation_results/`
# —— 它只额外 `cp` 了一份 `.snap_c15` 副本, **没有恢复原件**。
# 结果那个 670,382 被当成"当前生产净值"上报, 还进了 navlaw 台账。
#
# 根因不是"忘了恢复", 是**两套产物共用一个命名空间**: 谁最后写谁就是"生产"。
# 隔离的正解是让它们在物理上不同时存在。
#
# ── 怎么做 ────────────────────────────────────────────────────────────────
# 用 `mv`(同文件系统 = O(1) rename, 不拷 3GB):
#   begin  → 生产产物**搬走**, RVD 里不存在 ⇒ 臂只能写自己的产物
#   end    → 臂产物搬去 .arms/armout/<label>/, 生产产物**搬回原位**
#
# 关键性质: 臂运行期间生产产物**不在场** ⇒ 任何"误把臂产物当生产"的读法
# 都会**大声失败**(文件不存在), 而不是静默拿到一个错的数。
# 这比"跑完记得恢复"强, 因为后者依赖纪律, 前者依赖物理。
#
# ── 用法 ──────────────────────────────────────────────────────────────────
#   source /mnt/d/quant/.arms/lib/armguard.sh
#   arm_begin mylabel            # 或 armguard.sh begin mylabel
#   trap arm_end EXIT INT TERM   # ← 必须; 崩溃/被杀也能恢复
#   ... 跑臂 ...
#
# 命令行:
#   armguard.sh begin <label>    # 搬走生产产物, 记 git 态
#   armguard.sh end   <label>    # 归档臂产物 + 还原生产产物 + 逐位校验
#   armguard.sh status           # 当前是否有臂在飞 / 生产产物是否在场
#   armguard.sh rescue           # 强制还原最近一次 begin 的生产产物
#   armguard.sh ls               # 列出归档的臂产物
#
# ⚠ 备份一律在 /mnt/d 下, **永不放 /tmp**(tmpfs, 已实际 OOM 掉一次回测)。

ARM_ROOT=/mnt/d/quant
ARM_RVD="$ARM_ROOT/strategy/rolling_validation_results"
ARM_PRODGUARD="$ARM_ROOT/.arms/prodguard"
ARM_ARMOUT="$ARM_ROOT/.arms/armout"

# 受保护的生产产物。臂会原地覆盖的就是这些 —— 每加一个输出点就要加进来。
ARM_GUARD_FILES=(
  equity_curve.csv
  portfolio_selections.csv
  trade_realized.csv
  backtest_report.json
  backtest_signals.csv
  regime_state.csv
  yaogu_watchlist.csv
  anchor.json
)

_arm_die() { echo "✗ [armguard] $*" >&2; return 1; }

_arm_gitstate() {
  local head dirty
  head=$(git -C "$ARM_ROOT" rev-parse --short HEAD 2>/dev/null || echo '?')
  dirty=$(git -C "$ARM_ROOT" status --porcelain -- strategy/ 2>/dev/null | wc -l)
  echo "$head+$dirty"
}

# ── begin ────────────────────────────────────────────────────────────────
arm_begin() {
  local label="$1"
  [ -z "$label" ] && { _arm_die "用法: arm_begin <label>"; return 1; }
  export ARM_LABEL="$label"
  local pg="$ARM_PRODGUARD/$label"
  if [ -d "$pg" ]; then
    _arm_die "prodguard/$label 已存在 —— 该臂已在飞, 或上次没 end。先 armguard.sh rescue"
    return 1
  fi
  mkdir -p "$pg"
  local n=0
  for f in "${ARM_GUARD_FILES[@]}"; do
    if [ -e "$ARM_RVD/$f" ]; then
      mv "$ARM_RVD/$f" "$pg/$f" || { _arm_die "搬走 $f 失败"; return 1; }
      n=$((n + 1))
    fi
  done
  {
    echo "label=$label"
    echo "ts=$(date '+%Y-%m-%d %H:%M:%S')"
    echo "git=$(_arm_gitstate)"
    echo "moved=$n"
  } > "$pg/META.txt"
  # 生产产物的 md5 台账 —— end 用它逐位校验"还原回去的是不是原来那份"
  ( cd "$pg" && md5sum "${ARM_GUARD_FILES[@]}" 2>/dev/null ) > "$pg/PROD_MD5.txt"
  echo "▶ [armguard] begin $label: 搬走 $n 个生产产物 → prodguard/$label  [git $(_arm_gitstate)]"
  echo "  RVD 现在是空的 ⇒ 臂产物无法伪装成生产"
  return 0
}

# ── end ──────────────────────────────────────────────────────────────────
arm_end() {
  local label="${1:-$ARM_LABEL}"
  [ -z "$label" ] && { _arm_die "arm_end 需要 label (ARM_LABEL 未设)"; return 1; }
  local pg="$ARM_PRODGUARD/$label" out="$ARM_ARMOUT/$label"
  [ -d "$pg" ] || { _arm_die "prodguard/$label 不存在 —— 没 begin 过?"; return 1; }

  # 1. 臂产物归档 (先算 md5 再搬, 便于事后对账)
  mkdir -p "$out"
  local n=0
  : > "$out/MD5.txt"
  for f in "${ARM_GUARD_FILES[@]}"; do
    if [ -e "$ARM_RVD/$f" ]; then
      md5sum "$ARM_RVD/$f" | awk -v OFS='  ' '{print $1, $2}' >> "$out/MD5.txt"
      mv "$ARM_RVD/$f" "$out/$f" || { _arm_die "归档 $f 失败"; return 1; }
      n=$((n + 1))
    fi
  done
  echo "label=$label" > "$out/META.txt"
  echo "ts=$(date '+%Y-%m-%d %H:%M:%S')" >> "$out/META.txt"
  echo "git=$(_arm_gitstate)" >> "$out/META.txt"
  echo "n_products=$n" >> "$out/META.txt"

  # 2. 生产产物搬回原位 + 逐位校验
  local m=0 bad=0
  local bn
  for f in "$pg"/*; do
    [ -f "$f" ] || continue
    bn=$(basename "$f")
    case "$bn" in META.txt|PROD_MD5.txt) continue ;; esac
    mv "$f" "$ARM_RVD/$bn" || { _arm_die "还原 $bn 失败"; return 1; }
    m=$((m + 1))
  done
  while read -r want path; do
    [ -e "$ARM_RVD/$path" ] || { bad=$((bad + 1)); echo "  ✗ $path 没还原回来"; continue; }
    got=$(md5sum "$ARM_RVD/$path" | cut -d' ' -f1)
    [ "$got" = "$want" ] || { bad=$((bad + 1)); echo "  ✗ $path md5 变了: $want → $got"; }
  done < "$pg/PROD_MD5.txt"

  # 3. 结论
  if [ "$bad" -eq 0 ]; then
    echo "✓ [armguard] end $label: 臂产物 $n 个 → armout/$label; 生产产物 $m 个已还原, md5 逐位一致"
  else
    echo "✗ [armguard] end $label: 生产产物还原后 **$bad 个 md5 不符** —— 立刻查!" >&2
    return 2
  fi
  if [ "$m" -eq 0 ]; then
    echo "  ⚠ 本次 begin 时 RVD 是空的(之前就没有生产产物) —— 别把 armout 当成生产"
  fi
  rm -f "$pg/META.txt" "$pg/PROD_MD5.txt"
  rmdir "$pg" 2>/dev/null || echo "  ⚠ prodguard/$label 未清空, 手工看一下"
  unset ARM_LABEL
  return 0
}

# ── status / rescue / ls ─────────────────────────────────────────────────
arm_status() {
  local flying=() present=0 missing=0 n_prod
  # 只有**真的扣着生产产物**的目录才算"臂在飞"; 空壳目录(收尾没删干净)不算
  for d in "$ARM_PRODGUARD"/*/; do
    [ -d "$d" ] || continue
    n_prod=$(find "$d" -maxdepth 1 -type f \
             ! -name META.txt ! -name PROD_MD5.txt 2>/dev/null | wc -l)
    [ "$n_prod" -gt 0 ] && flying+=("$(basename "$d")")
  done
  for f in "${ARM_GUARD_FILES[@]}"; do
    if [ -e "$ARM_RVD/$f" ]; then present=$((present + 1)); else missing=$((missing + 1)); fi
  done
  echo "RVD: $ARM_RVD"
  echo "  在场 $present / 受保护 ${#ARM_GUARD_FILES[@]}"
  if [ "${#flying[@]}" -gt 0 ]; then
    echo "  ⚠ 有 ${#flying[@]} 个臂在飞(或崩了没收尾): ${flying[*]}"
    echo "    ⇒ 生产产物已被搬走 $missing 个; 收尾: armguard.sh rescue"
    return 1
  fi
  echo "  ✓ 无臂在飞"
  if [ "$present" -eq 0 ]; then
    echo "  ⚠ 但一个生产产物都没有 —— RVD 是空的, 先跑一次生产回测"
    return 1
  fi
  return 0
}

arm_rescue() {
  local d
  d=$(ls -1dt "$ARM_PRODGUARD"/*/ 2>/dev/null | head -1)
  [ -n "$d" ] || { _arm_die "没有 prodguard 目录可还原"; return 1; }
  local label; label=$(basename "$d")
  echo "▶ [armguard] rescue: 从 prodguard/$label 还原"
  local m=0
  for f in "$d"*; do
    [ -f "$f" ] || continue
    case "$(basename "$f")" in META.txt) continue ;; esac
    mv "$f" "$ARM_RVD/$(basename "$f")" && m=$((m + 1))
  done
  echo "✓ 还原 $m 个"
  rm -f "$d/META.txt"
  rmdir "$d" 2>/dev/null || echo "  ⚠ $d 未清空 (还有残留文件), 手工看一下"
}

arm_ls() {
  [ -d "$ARM_ARMOUT" ] || { echo "(无归档臂产物)"; return 0; }
  printf '%-28s %-21s %-10s %s\n' label 归档时间 git 产物数
  for d in "$ARM_ARMOUT"/*/; do
    [ -d "$d" ] || continue
    printf '%-28s %-21s %-10s %s\n' "$(basename "$d")" \
      "$(grep -m1 '^ts=' "$d/META.txt" 2>/dev/null | cut -d= -f2-)" \
      "$(grep -m1 '^git=' "$d/META.txt" 2>/dev/null | cut -d= -f2-)" \
      "$(grep -m1 '^n_products=' "$d/META.txt" 2>/dev/null | cut -d= -f2-)"
  done
}

# ── CLI ──────────────────────────────────────────────────────────────────
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
  cmd="${1:-status}"; shift || true
  case "$cmd" in
    begin)  arm_begin "$1" ;;
    end)    arm_end "$1" ;;
    status) arm_status ;;
    rescue) arm_rescue ;;
    ls)     arm_ls ;;
    *) echo "用法: armguard.sh {begin <label>|end <label>|status|rescue|ls}" >&2; exit 2 ;;
  esac
fi
