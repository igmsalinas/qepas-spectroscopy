#!/usr/bin/env bash
# Train all QEPAS spectroscopy models with optional resume support.
#
# Usage:
#   ./train_all.sh              # fresh run
#   ./train_all.sh --resume     # resume from previous tuner/fold checkpoints
#   ./train_all.sh --skip-deep  # skip deep learning models
#
# The script runs the `qepas-train` CLI through `uv run`. If uv is not
# installed, it falls back to the first `qepas-train` found on PATH.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${PROJECT_ROOT}/outputs/logs"
mkdir -p "${LOG_DIR}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/train_all_${TIMESTAMP}.log"

RESUME_FLAG=""
SKIP_DEEP_FLAG=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --resume)
            RESUME_FLAG="--resume"
            shift
            ;;
        --skip-deep)
            SKIP_DEEP_FLAG="--skip-deep"
            shift
            ;;
        --skip-xgb-tune)
            EXTRA_ARGS+=("--skip-xgb-tune")
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--resume] [--skip-deep] [--skip-xgb-tune] [extra qepas-train args...]"
            exit 0
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

if command -v uv >/dev/null 2>&1; then
    echo "uv found at $(command -v uv); running qepas-train with uv run"
    CLI=(uv run qepas-train)
elif command -v qepas-train >/dev/null 2>&1; then
    CLI=(qepas-train)
else
    echo "Error: neither uv nor qepas-train found on PATH." >&2
    echo "Install uv (https://docs.astral.sh/uv/) or install this package first." >&2
    exit 1
fi

echo "Starting full training pipeline at ${TIMESTAMP}"
echo "Resume:      $([[ -n "${RESUME_FLAG}" ]] && echo yes || echo no)"
echo "Skip deep:   $([[ -n "${SKIP_DEEP_FLAG}" ]] && echo yes || echo no)"
echo "Log file:    ${LOG_FILE}"
echo "Command:     ${CLI[*]} ${RESUME_FLAG} ${SKIP_DEEP_FLAG} ${EXTRA_ARGS[*]:-}"

exec > >(tee -a "${LOG_FILE}")
exec 2>&1

cd "${PROJECT_ROOT}"

"${CLI[@]}" \
    --deep-tuner-trials 20 \
    --deep-epochs 150 \
    --deep-batch-size 256 \
    --deep-early-stopping-patience 5 \
    --signal-length 8192 \
    --tensorboard-dir outputs/tensorboard \
    ${RESUME_FLAG} \
    ${SKIP_DEEP_FLAG} \
    "${EXTRA_ARGS[@]}"

echo "Training completed successfully. Outputs are in ${PROJECT_ROOT}/outputs"
