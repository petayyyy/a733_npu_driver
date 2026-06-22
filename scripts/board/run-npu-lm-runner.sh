#!/usr/bin/env bash
set -eu

usage() {
    cat >&2 <<'USAGE'
Usage: run-npu-lm-runner.sh [options]

Runs the persistent tiny-LM VIPLite runner and stores a reproducible log.

Options:
  --model-dir DIR   Tiny LM model directory.
                    Default: /home/radxa/a733_npu_driver/models/tiny_lm_gather_int16
  --runner FILE     Runner binary. Default: build/npu_lm_runner
  --steps N         Generated token count. Default: 8
  --prompt "IDS"    Initial token IDs. Default: 1 5 9 2
  --seq-len N       Fixed token window length. Default: 4
  --vocab N         Tiny LM vocabulary size. Default: 16
  --log-root DIR    Log root. Default: logs/board
  --label LABEL     Label used in the log directory name.
USAGE
    exit 2
}

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_DIR="$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd)"
MODEL_DIR="${A733_MODEL_DIR:-/home/radxa/a733_npu_driver/models/tiny_lm_gather_int16}"
RUNNER="${A733_NPU_LM_RUNNER:-${REPO_DIR}/build/npu_lm_runner}"
STEPS="${A733_STEPS:-8}"
PROMPT="${A733_PROMPT:-1 5 9 2}"
SEQ_LEN="${A733_SEQ_LEN:-4}"
VOCAB="${A733_VOCAB:-16}"
LOG_ROOT="${A733_LOG_ROOT:-${REPO_DIR}/logs/board}"
LABEL="${A733_LOG_LABEL:-persistent}"
SDK_DIR="${A733_AI_SDK_DIR:-/home/radxa/ai-sdk}"

while [ "$#" -gt 0 ]; do
    case "$1" in
        --model-dir)
            MODEL_DIR="$2"
            shift 2
            ;;
        --runner)
            RUNNER="$2"
            shift 2
            ;;
        --steps)
            STEPS="$2"
            shift 2
            ;;
        --prompt)
            PROMPT="$2"
            shift 2
            ;;
        --seq-len)
            SEQ_LEN="$2"
            shift 2
            ;;
        --vocab)
            VOCAB="$2"
            shift 2
            ;;
        --log-root)
            LOG_ROOT="$2"
            shift 2
            ;;
        --label)
            LABEL="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage
            ;;
    esac
done

VIP_LIB="${SDK_DIR}/viplite-tina/lib/aarch64-none-linux-gnu/v2.0"
export LD_LIBRARY_PATH="${VIP_LIB}:${LD_LIBRARY_PATH:-}"

if [ ! -x "${RUNNER}" ]; then
    echo "Runner is not executable: ${RUNNER}" >&2
    echo "Build it first with scripts/board/build-npu-lm-runner.sh" >&2
    exit 1
fi
if [ ! -d "${MODEL_DIR}" ]; then
    echo "Model directory not found: ${MODEL_DIR}" >&2
    exit 1
fi

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
HOST="$(hostname 2>/dev/null || echo board)"
OUT_DIR="${LOG_ROOT}/t1-persistent-runner-${LABEL}-${HOST}-${STAMP}"
mkdir -p "${OUT_DIR}"
LOG="${OUT_DIR}/run.log"

{
    echo "runner=${RUNNER}"
    echo "model_dir=${MODEL_DIR}"
    echo "steps=${STEPS}"
    echo "prompt=${PROMPT}"
    echo "seq_len=${SEQ_LEN}"
    echo "vocab=${VOCAB}"
    echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"
    "${RUNNER}" \
        --model-dir "${MODEL_DIR}" \
        --steps "${STEPS}" \
        --prompt "${PROMPT}" \
        --seq-len "${SEQ_LEN}" \
        --vocab "${VOCAB}"
} 2>&1 | tee "${LOG}"
status="${PIPESTATUS[0]}"

grep -E '^(final_tokens|mean_wall_us|mean_profile_us|mean_tok_s|create_network_us|prepare_network_us|cid|nbg_loaded_once)=' "${LOG}" > "${OUT_DIR}/summary.env" || true
echo "logs=${OUT_DIR}" | tee -a "${LOG}"
exit "${status}"
