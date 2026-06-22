#!/usr/bin/env bash
set -eu

usage() {
    cat >&2 <<'USAGE'
Usage: run-npu-lm-runner-rss.sh [options]

Runs the persistent LM runner while sampling /proc/<pid>/status for peak VmRSS.

Options:
  --model-dir DIR   Model directory. Required.
  --runner FILE     Runner binary. Default: build/npu_lm_runner
  --steps N         Generated token count. Default: 16
  --prompt IDS      Initial token IDs, comma or space separated. Required.
  --seq-len N       Fixed token window length. Default: 32
  --vocab N         Vocabulary size. Default: 49152
  --log-dir DIR     Output directory. Required.
USAGE
    exit 2
}

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_DIR="$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd)"
MODEL_DIR=""
RUNNER="${A733_NPU_LM_RUNNER:-${REPO_DIR}/build/npu_lm_runner}"
STEPS="${A733_STEPS:-16}"
PROMPT=""
SEQ_LEN="${A733_SEQ_LEN:-32}"
VOCAB="${A733_VOCAB:-49152}"
LOG_DIR=""
SDK_DIR="${A733_AI_SDK_DIR:-/home/radxa/ai-sdk}"

while [ "$#" -gt 0 ]; do
    case "$1" in
        --model-dir) MODEL_DIR="$2"; shift 2 ;;
        --runner) RUNNER="$2"; shift 2 ;;
        --steps) STEPS="$2"; shift 2 ;;
        --prompt) PROMPT="$2"; shift 2 ;;
        --seq-len) SEQ_LEN="$2"; shift 2 ;;
        --vocab) VOCAB="$2"; shift 2 ;;
        --log-dir) LOG_DIR="$2"; shift 2 ;;
        -h|--help) usage ;;
        *) echo "Unknown argument: $1" >&2; usage ;;
    esac
done

[ -n "${MODEL_DIR}" ] || usage
[ -n "${PROMPT}" ] || usage
[ -n "${LOG_DIR}" ] || usage

VIP_LIB="${SDK_DIR}/viplite-tina/lib/aarch64-none-linux-gnu/v2.0"
export LD_LIBRARY_PATH="${VIP_LIB}:${LD_LIBRARY_PATH:-}"

mkdir -p "${LOG_DIR}"
RUN_LOG="${LOG_DIR}/run.log"
RSS_LOG="${LOG_DIR}/rss.env"

"${RUNNER}" \
    --model-dir "${MODEL_DIR}" \
    --steps "${STEPS}" \
    --prompt "${PROMPT}" \
    --seq-len "${SEQ_LEN}" \
    --vocab "${VOCAB}" \
    > "${RUN_LOG}" 2>&1 &
pid=$!

peak_rss_kb=0
while kill -0 "${pid}" 2>/dev/null; do
    rss_kb=$(awk '/VmRSS/ { print $2 }' "/proc/${pid}/status" 2>/dev/null || true)
    rss_kb=${rss_kb:-0}
    if [ "${rss_kb}" -gt "${peak_rss_kb}" ]; then
        peak_rss_kb=${rss_kb}
    fi
    sleep 0.02
done

status=0
wait "${pid}" || status=$?

{
    echo "peak_rss_kb=${peak_rss_kb}"
    echo "status=${status}"
} > "${RSS_LOG}"

cat "${RUN_LOG}"
cat "${RSS_LOG}"
exit "${status}"
