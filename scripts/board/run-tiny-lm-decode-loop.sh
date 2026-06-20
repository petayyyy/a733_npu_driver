#!/usr/bin/env bash
set -eu

usage() {
    cat >&2 <<'USAGE'
Usage: run-tiny-lm-decode-loop.sh --model-dir <dir> [options]

Runs a fixed-window tiny LM autoregressive loop where every model-layer forward
pass is executed by vpm_run on the A733 NPU. CPU work is limited to writing the
next token-id input window and choosing the next token from NPU logits.

Options:
  --model-dir DIR    Directory with network_binary.nb, sample.txt, input_0.dat.
  --prompt "IDS"     Initial token IDs, space or comma separated. Default: 1 5 9 2.
  --steps N          Number of generated tokens. Default: 8.
  --seq-len N        Fixed token window length. Default: 4.
  --vocab N          Tiny LM vocabulary size. Default: 16.
  --sample FILE      vpm_run sample file. Default: sample.txt.
  --device N         vpm_run device index. Default: 0.

Environment:
  A733_VPM_RUN=/path/to/vpm_run      Optional explicit vpm_run path.
  A733_VIP_LIB_DIR=/path/to/libs     Optional VIPLite library directory.
  A733_LOG_ROOT=logs/board           Optional log root.
  A733_LOG_LABEL=name                Optional label appended to log directory.
USAGE
    exit 2
}

find_command() {
    cmd="$1"
    if command -v "${cmd}" >/dev/null 2>&1; then
        command -v "${cmd}"
        return 0
    fi
    find /home/radxa /usr /opt /root "$PWD" -type f -name "${cmd}" 2>/dev/null | head -n 1
}

MODEL_DIR="${A733_MODEL_DIR:-}"
PROMPT="${A733_PROMPT:-1 5 9 2}"
STEPS="${A733_STEPS:-8}"
SEQ_LEN="${A733_SEQ_LEN:-4}"
VOCAB="${A733_VOCAB:-16}"
SAMPLE="${A733_SAMPLE:-sample.txt}"
DEVICE="${A733_DEVICE:-0}"

while [ "$#" -gt 0 ]; do
    case "$1" in
        --model-dir)
            MODEL_DIR="$2"
            shift 2
            ;;
        --prompt)
            PROMPT="$2"
            shift 2
            ;;
        --steps)
            STEPS="$2"
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
        --sample)
            SAMPLE="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
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

if [ -z "${MODEL_DIR}" ]; then
    echo "--model-dir is required" >&2
    usage
fi

if [ ! -d "${MODEL_DIR}" ]; then
    echo "Model directory not found: ${MODEL_DIR}" >&2
    exit 1
fi

VPM="${A733_VPM_RUN:-}"
if [ -z "${VPM}" ]; then
    VPM="$(find_command vpm_run || true)"
fi

if [ -z "${VPM}" ] || [ ! -x "${VPM}" ]; then
    echo "Could not find executable vpm_run. Set A733_VPM_RUN." >&2
    exit 1
fi

if [ -n "${A733_VIP_LIB_DIR:-}" ]; then
    export LD_LIBRARY_PATH="${A733_VIP_LIB_DIR}:${LD_LIBRARY_PATH:-}"
else
    VIP_LIB_FILE="$(find /home/radxa/lib /usr /opt /lib /root "$PWD" -type f \( -name 'libVIPhal.so*' -o -name 'libNBGlinker.so*' \) 2>/dev/null | head -n 1)"
    if [ -n "${VIP_LIB_FILE}" ]; then
        VIP_LIB_DIR="$(dirname "${VIP_LIB_FILE}")"
        export LD_LIBRARY_PATH="${VIP_LIB_DIR}:${LD_LIBRARY_PATH:-}"
    fi
fi

ROOT_DIR="${A733_LOG_ROOT:-logs/board}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
HOST="$(hostname 2>/dev/null || echo board)"
if [ -n "${A733_LOG_LABEL:-}" ]; then
    OUT_DIR="${ROOT_DIR}/${HOST}-tiny-lm-decode-${A733_LOG_LABEL}-${STAMP}"
else
    OUT_DIR="${ROOT_DIR}/${HOST}-tiny-lm-decode-${STAMP}"
fi
mkdir -p "${OUT_DIR}"
OUT_DIR="$(cd "${OUT_DIR}" && pwd)"
LOG="${OUT_DIR}/run.log"
SUMMARY="${OUT_DIR}/steps.tsv"

echo "vpm_run=${VPM}" | tee "${LOG}"
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}" | tee -a "${LOG}"
echo "model_dir=${MODEL_DIR}" | tee -a "${LOG}"
echo "sample=${SAMPLE}" | tee -a "${LOG}"
echo "prompt=${PROMPT}" | tee -a "${LOG}"
echo "steps=${STEPS}" | tee -a "${LOG}"
echo "seq_len=${SEQ_LEN}" | tee -a "${LOG}"
echo "vocab=${VOCAB}" | tee -a "${LOG}"

TOKENS="$(
    python3 - "${PROMPT}" "${SEQ_LEN}" "${VOCAB}" <<'PY'
import sys

prompt = [int(part) for part in sys.argv[1].replace(",", " ").split()]
seq_len = int(sys.argv[2])
vocab = int(sys.argv[3])
if not prompt:
    raise SystemExit("prompt must contain at least one token")
for token in prompt:
    if token < 0 or token >= vocab:
        raise SystemExit(f"token {token} outside vocab 0..{vocab - 1}")
if len(prompt) < seq_len:
    prompt = [0] * (seq_len - len(prompt)) + prompt
print(" ".join(str(token) for token in prompt))
PY
)"

cd "${MODEL_DIR}"
if [ ! -f "${SAMPLE}" ]; then
    echo "Sample file not found: ${MODEL_DIR}/${SAMPLE}" >&2
    exit 1
fi

printf 'step\twindow\tnext_token\tlast_position_top5\tprofile\n' > "${SUMMARY}"

step=0
while [ "${step}" -lt "${STEPS}" ]; do
    WINDOW="$(
        python3 - "${TOKENS}" "${SEQ_LEN}" <<'PY'
import sys

tokens = [int(part) for part in sys.argv[1].split()]
seq_len = int(sys.argv[2])
print(" ".join(str(token) for token in tokens[-seq_len:]))
PY
    )"

    python3 - "${WINDOW}" "input_0.dat" <<'PY'
import struct
import sys

tokens = [int(part) for part in sys.argv[1].split()]
with open(sys.argv[2], "wb") as f:
    for token in tokens:
        f.write(struct.pack("<i", token))
PY

    RUN_OUT="${OUT_DIR}/step_${step}_vpm_run.out"
    RUN_ERR="${OUT_DIR}/step_${step}_vpm_run.err"
    echo "step=${step} window=${WINDOW}" | tee -a "${LOG}"
    set +e
    "${VPM}" -s "${SAMPLE}" -l 1 -d "${DEVICE}" -b 0 --show_top5 1 --save_txt 1 > "${RUN_OUT}" 2> "${RUN_ERR}"
    status=$?
    set -e
    cat "${RUN_OUT}" >> "${LOG}"
    cat "${RUN_ERR}" >> "${LOG}"
    echo "step=${step} exit=${status}" | tee -a "${LOG}"
    if [ "${status}" -ne 0 ]; then
        exit "${status}"
    fi

    STEP_OUTPUT="${OUT_DIR}/step_${step}_output_0.txt"
    if [ ! -f output_0.txt ]; then
        echo "vpm_run did not create output_0.txt" >&2
        exit 1
    fi
    cp output_0.txt "${STEP_OUTPUT}"

    NEXT_LINE="$(
        python3 - "${STEP_OUTPUT}" "${VOCAB}" <<'PY'
import sys

values = [float(line.strip()) for line in open(sys.argv[1], "r", encoding="ascii") if line.strip()]
vocab = int(sys.argv[2])
if len(values) < vocab or len(values) % vocab != 0:
    raise SystemExit(f"unexpected logits length {len(values)} for vocab {vocab}")
last = values[-vocab:]
order = sorted(range(vocab), key=lambda i: last[i], reverse=True)
top = ",".join(f"{i}:{last[i]:.6f}" for i in order[:5])
print(f"{order[0]}|{top}")
PY
    )"
    NEXT_TOKEN="${NEXT_LINE%%|*}"
    TOP5="${NEXT_LINE#*|}"
    PROFILE="$(grep -E 'profile inference time=' "${RUN_OUT}" | tail -n 1 || true)"
    printf '%s\t%s\t%s\t%s\t%s\n' "${step}" "${WINDOW}" "${NEXT_TOKEN}" "${TOP5}" "${PROFILE}" >> "${SUMMARY}"
    echo "step=${step} next=${NEXT_TOKEN} last_position_top5=${TOP5}" | tee -a "${LOG}"
    TOKENS="${TOKENS} ${NEXT_TOKEN}"
    step=$((step + 1))
done

echo "${TOKENS}" > "${OUT_DIR}/tokens.txt"
echo "final_tokens=${TOKENS}" | tee -a "${LOG}"

if grep -Eiq 'cid=0x1000003b|0x1000003b|VIPLite|vipcore' "${OUT_DIR}"/step_*_vpm_run.out "${OUT_DIR}"/step_*_vpm_run.err; then
    echo "VIP identity detected" | tee -a "${LOG}"
else
    echo "VIP identity not detected in logs" | tee -a "${LOG}"
fi

echo "logs=${OUT_DIR}" | tee -a "${LOG}"
