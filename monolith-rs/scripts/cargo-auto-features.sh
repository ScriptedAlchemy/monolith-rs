#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

features=()

if [[ "$(uname -s)" == "Darwin" ]]; then
  # Metal is the preferred GPU backend on macOS.
  features+=("metal")
elif command -v nvcc >/dev/null 2>&1; then
  # CUDA is available when nvcc is on PATH.
  features+=("cuda")
fi

if [[ ${#features[@]} -gt 0 ]]; then
  cargo "$@" --features "$(IFS=,; echo "${features[*]}")"
else
  cargo "$@"
fi

