#!/usr/bin/env bash
set -eu

REPO_DIR="${A733_REPO_DIR:-/home/orangepi/a733_npu_driver}"
STEPS="${A733_B1B_STEPS:-16}"
TOKENIZER="${A733_B1B_TOKENIZER:-${REPO_DIR}/work/models/smollm2-135m-instruct/tokenizer.json}"
RUNNER="${A733_B1B_RUNNER:-${REPO_DIR}/build/npu_lm_runner}"
VIP_LIB="${A733_B1B_VIP_LIB:-/home/orangepi/lib}"
LOG_DIR="${A733_B1B_LOG_DIR:-${REPO_DIR}/logs/board/b1b}"
PROMPT="${A733_B1B_PROMPT:-The capital of France is}"
PAD_TOKEN="${A733_B1B_PAD_TOKEN:-0}"

preflight() {
    busy="$(
        ps -eo pid,user,comm,args |
            awk '/npu_lm_runner|vpm_run|chat_shell.py|monitor_command.py|llama|cmake|ninja/ && $0 !~ /awk/ {print}'
    )"
    if [ -n "${busy}" ]; then
        echo "[preflight] busy process detected" >&2
        echo "${busy}" >&2
        exit 77
    fi
    if fuser -s /dev/vipcore 2>/dev/null; then
        echo "[preflight] /dev/vipcore busy" >&2
        fuser -v /dev/vipcore >&2 || true
        exit 78
    fi
}

run_case() {
    label="$1"
    window="$2"
    nbg="${REPO_DIR}/models/${label}_int16/network_binary.nb"
    out="${LOG_DIR}/${label}-board.json"

    echo "== ${label} W=${window} =="
    preflight
    python3 "${REPO_DIR}/scripts/board/b1b_benchmark_smollm2.py" \
        --nbg "${nbg}" \
        --tokenizer "${TOKENIZER}" \
        --runner "${RUNNER}" \
        --vip-lib "${VIP_LIB}" \
        --window "${window}" \
        --steps "${STEPS}" \
        --pad-token "${PAD_TOKEN}" \
        --prompt "${PROMPT}" \
        --output-json "${out}"
    preflight
}

mkdir -p "${LOG_DIR}"
cd "${REPO_DIR}"

run_case b1_smollm2_135m_w32 32
run_case b1_smollm2_135m_w64 64
run_case b1_smollm2_135m_w128 128
run_case b1_smollm2_135m_w256 256
run_case b1_smollm2_360m_w32 32
run_case b1_smollm2_360m_w64 64
run_case b1_smollm2_360m_w128 128
run_case b1_smollm2_360m_w256 256
