#!/usr/bin/env bash
# Run against this checkout's package and bundled diffusion library.
set -euo pipefail
ORGACAST_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ORGACAST_ROOT}/vendor/diffusers/src:${ORGACAST_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
ORGACAST_WORKDIR="${ORGACAST_ROOT}"
if [[ "${1:-}" == "--cwd" ]]; then
    ORGACAST_WORKDIR="${ORGACAST_ROOT}/${2:?Missing relative working directory}"
    shift 2
fi
if [[ $# -eq 0 ]]; then
    echo "Usage: scripts/run.sh [--cwd relative-directory] command [arguments...]" >&2
    exit 2
fi
cd -- "${ORGACAST_WORKDIR}"
exec "$@"
