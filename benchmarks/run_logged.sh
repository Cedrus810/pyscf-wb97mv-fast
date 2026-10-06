#!/usr/bin/env bash
# Usage: benchmarks/run_logged.sh <log file> <command...>
# Runs the command, tees stdout+stderr to the log, and stamps a START line
# (host, CPU, OMP_NUM_THREADS, GPU, python) and an END line (rc, wall time).
# The pthreads-OpenBLAS "Detect OpenMP Loop" warning (tens of thousands of
# lines per run in a conda pthreads-OpenBLAS env) is dropped from the stream; one
# summary line with its count is printed at the end instead.
log=$1; shift
set -o pipefail   # exit with the command's rc, not tee's
{
  echo "[$(date '+%F %T')] START $* | host $(hostname) | $(lscpu | sed -n 's/Model name: *//p') | OMP_NUM_THREADS=${OMP_NUM_THREADS:-unset} | OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-unset} | $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null) | exe=$(command -v "$1")"
  t0=$SECONDS
  "$@"
  rc=$?
  echo "[$(date '+%F %T')] END rc=$rc wall=$((SECONDS-t0))s"
  exit $rc
} 2>&1 | awk '/^OpenBLAS Warning : Detect OpenMP Loop/ {n++; next}
              {print; fflush()}
              END {print "[run_logged] suppressed " n+0 " OpenBLAS \"Detect OpenMP Loop\" warnings"}' \
  | tee "$log"
