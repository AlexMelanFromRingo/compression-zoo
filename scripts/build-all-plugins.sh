#!/usr/bin/env bash
# build-all-plugins.sh — convenience wrapper that runs each plugin's Makefile
# in turn. Skips plugins whose Makefile doesn't yet exist.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PLUGINS=(zpaq bsc cmix)

for p in "${PLUGINS[@]}"; do
  dir="$REPO_ROOT/plugins/$p"
  if [[ -f "$dir/Makefile" ]]; then
    echo "==> building $p"
    make -C "$dir" "$@"
  else
    echo "(skipping $p — no Makefile yet)"
  fi
done
