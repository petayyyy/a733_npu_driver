#!/usr/bin/env bash
set -eu

ROOT_DIR="${A733_LOG_ROOT:-logs/board}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
HOST="$(hostname 2>/dev/null || echo board)"
if [ -n "${A733_LOG_LABEL:-}" ]; then
    OUT_DIR="${ROOT_DIR}/${HOST}-vpm-${A733_LOG_LABEL}-${STAMP}"
else
    OUT_DIR="${ROOT_DIR}/${HOST}-vpm-${STAMP}"
fi
mkdir -p "${OUT_DIR}"

if [ "$#" -eq 0 ]; then
    cat >&2 <<'USAGE'
Usage: run-vpm.sh <vpm_run arguments>

Environment:
  A733_VPM_RUN=/path/to/vpm_run      Optional explicit vpm_run path.
  A733_VIP_LIB_DIR=/path/to/libs     Optional VIPLite library directory.
  A733_VPM_CWD=/path/to/model-dir    Optional working directory for vpm_run.
  A733_LOG_LABEL=name                Optional label appended to log directory.
  A733_LOG_ROOT=logs/board           Optional log root.
USAGE
    exit 2
fi

find_command() {
    cmd="$1"
    if command -v "${cmd}" >/dev/null 2>&1; then
        command -v "${cmd}"
        return 0
    fi
    find /usr /opt /root "$PWD" -type f -name "${cmd}" 2>/dev/null | head -n 1
}

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

echo "vpm_run=${VPM}" | tee "${OUT_DIR}/run.log"
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}" | tee -a "${OUT_DIR}/run.log"
if [ -n "${A733_VPM_CWD:-}" ]; then
    echo "cwd=${A733_VPM_CWD}" | tee -a "${OUT_DIR}/run.log"
    cd "${A733_VPM_CWD}"
fi
echo "+ ${VPM} $*" | tee -a "${OUT_DIR}/run.log"

"${VPM}" "$@" > "${OUT_DIR}/vpm_run.out" 2> "${OUT_DIR}/vpm_run.err"
status=$?

cat "${OUT_DIR}/vpm_run.out" >> "${OUT_DIR}/run.log"
cat "${OUT_DIR}/vpm_run.err" >> "${OUT_DIR}/run.log"
echo "exit=${status}" | tee -a "${OUT_DIR}/run.log"

if grep -Eiq 'cid=0x1000003b|0x1000003b|VIPLite|vipcore' "${OUT_DIR}/vpm_run.out" "${OUT_DIR}/vpm_run.err"; then
    echo "VIP identity detected" | tee -a "${OUT_DIR}/run.log"
else
    echo "VIP identity not detected in logs" | tee -a "${OUT_DIR}/run.log"
fi

exit "${status}"
