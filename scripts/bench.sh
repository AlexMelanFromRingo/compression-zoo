#!/usr/bin/env bash
# bench.sh — measure compression ratio + time for each codec on a single
# input file. Outputs a markdown table.
#
# Usage:
#   scripts/bench.sh <input-file>
#
# Codecs:
#   xz     -9e            (LZMA2 baseline)
#   zstd   --ultra -22    (modern fast baseline)
#   zpaq   level 1, 5     (this repo)
#   bsc    level 1, 5, 9  (this repo)
#   cmix   default        (this repo, only on tiny inputs)

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <input-file>" >&2
  exit 1
fi

INPUT="$1"
SIZE=$(wc -c < "$INPUT")
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

ZPAQ_CLI="$ROOT/plugins/zpaq/tests/zpaq_cli"
BSC_CLI="$ROOT/plugins/bsc/tests/bsc_cli"
CMIX_CLI="$ROOT/plugins/cmix/tests/test_linux"

if [[ ! -x "$ZPAQ_CLI" ]] || [[ ! -x "$BSC_CLI" ]] || [[ ! -x "$CMIX_CLI" ]]; then
  echo "build the per-plugin Linux test binaries first" >&2
  exit 1
fi

# format: "label cmd_to_compress" — cmd reads stdin, writes stdout.
run_one() {
  local label="$1" cmd="$2"
  local out_path="$TMP/${label// /_}.bin"
  local t0 t1
  t0=$(date +%s.%N)
  bash -c "$cmd" < "$INPUT" > "$out_path" 2>/dev/null || { echo "FAIL $label"; return; }
  t1=$(date +%s.%N)
  local out_size
  out_size=$(wc -c < "$out_path")
  local ratio
  if [[ "$SIZE" -gt 0 ]]; then
    ratio=$(awk "BEGIN { printf \"%.2f\", 100.0 * $out_size / $SIZE }")
  else
    ratio="—"
  fi
  local elapsed
  elapsed=$(awk "BEGIN { printf \"%.2f\", $t1 - $t0 }")
  printf "| %-22s | %10d | %6s%% | %7s s |\n" "$label" "$out_size" "$ratio" "$elapsed"
}

echo
echo "## Benchmark: \`$INPUT\` ($SIZE bytes)"
echo
echo "| codec                  |   out_size |  ratio |    time |"
echo "|------------------------|-----------:|-------:|--------:|"

run_one "xz -9e (LZMA2)"      "xz -9e -c"
run_one "zstd --ultra -22"    "zstd -q --ultra -22 -c"
run_one "zpaq level 1"        "$ZPAQ_CLI e 1"
run_one "zpaq level 3"        "$ZPAQ_CLI e 3"
run_one "zpaq level 5"        "$ZPAQ_CLI e 5"
run_one "bsc level 1"         "$BSC_CLI e 1 1048576"
run_one "bsc level 5"         "$BSC_CLI e 5 1048576"
run_one "bsc level 9"         "$BSC_CLI e 9 1048576"
if [[ "$SIZE" -le 4096 ]]; then
  run_one "cmix"              "$CMIX_CLI encode"
else
  echo "| cmix                   |       (skipped on >4 KiB inputs) | | |"
fi

echo
