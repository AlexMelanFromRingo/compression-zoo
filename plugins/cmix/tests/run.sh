#!/usr/bin/env bash
# run.sh — drives test_linux as two separate processes (one encodes, one
# decodes) so each gets a fresh set of CMIX globals.  Compares the round
# trip against the original input.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$ROOT/tests/test_linux"

if [[ ! -x "$BIN" ]]; then
  echo "build $BIN first" >&2
  exit 1
fi

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

case "${1:-tiny}" in
  tiny)
    printf 'ABCDEFGH' > "$TMP/in.bin"
    ;;
  small)
    # 64 bytes with mild structure (avoid `yes | head` which fails under -e)
    python3 -c 'import sys; sys.stdout.write(("ABCD\n" * 13)[:64])' > "$TMP/in.bin"
    ;;
  *)
    echo "usage: $0 [tiny|small]" >&2
    exit 2
    ;;
esac

# Encode in one process.
"$BIN" encode < "$TMP/in.bin" > "$TMP/enc.bin"
# Decode in a fresh process — globals zero-initialised again.
"$BIN" decode < "$TMP/enc.bin" > "$TMP/out.bin"

if cmp -s "$TMP/in.bin" "$TMP/out.bin"; then
  echo "[cmix] PASS  ($(wc -c <"$TMP/in.bin") bytes -> $(wc -c <"$TMP/enc.bin") bytes -> $(wc -c <"$TMP/out.bin") bytes)"
else
  echo "[cmix] FAIL  round-trip mismatch" >&2
  exit 1
fi
