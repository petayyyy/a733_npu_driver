#!/usr/bin/env bash
set -eu

usage() {
    cat >&2 <<'USAGE'
Usage: build-npu-lm-runner.sh [options]

Builds scripts/board/npu_lm_runner.c on the A733 board against the installed
VIPLite 2.0 SDK.

Options:
  --sdk-dir DIR   ai-sdk checkout. Default: /home/radxa/ai-sdk
  --vip-inc DIR   VIPLite include directory. Overrides --sdk-dir derived path.
  --vip-lib DIR   VIPLite library directory. Overrides --sdk-dir derived path.
  --src FILE      Runner C source. Default: scripts/board/npu_lm_runner.c
  --out FILE      Output binary. Default: build/npu_lm_runner
USAGE
    exit 2
}

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_DIR="$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd)"
SDK_DIR="${A733_AI_SDK_DIR:-/home/radxa/ai-sdk}"
VIP_INC_OVERRIDE="${A733_VIP_INC:-}"
VIP_LIB_OVERRIDE="${A733_VIP_LIB:-}"
SRC="${SCRIPT_DIR}/npu_lm_runner.c"
OUT="${REPO_DIR}/build/npu_lm_runner"

while [ "$#" -gt 0 ]; do
    case "$1" in
        --sdk-dir)
            SDK_DIR="$2"
            shift 2
            ;;
        --vip-inc)
            VIP_INC_OVERRIDE="$2"
            shift 2
            ;;
        --vip-lib)
            VIP_LIB_OVERRIDE="$2"
            shift 2
            ;;
        --src)
            SRC="$2"
            shift 2
            ;;
        --out)
            OUT="$2"
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

VIP_DIR="${SDK_DIR}/viplite-tina/lib/aarch64-none-linux-gnu/v2.0"
VIP_INC="${VIP_INC_OVERRIDE:-${VIP_DIR}/inc}"
VIP_LIB="${VIP_LIB_OVERRIDE:-${VIP_DIR}}"

if [ ! -f "${SRC}" ]; then
    echo "Runner source not found: ${SRC}" >&2
    exit 1
fi
if [ ! -f "${VIP_INC}/vip_lite.h" ]; then
    echo "VIPLite header not found: ${VIP_INC}/vip_lite.h" >&2
    exit 1
fi
if [ ! -f "${VIP_LIB}/libVIPhal.so" ] && [ ! -f "${VIP_LIB}/libVIPhal.so.2" ]; then
    echo "VIPLite library not found under: ${VIP_LIB}" >&2
    exit 1
fi

mkdir -p "$(dirname "${OUT}")"

EXTRA_CFLAGS=""
if grep -q "VIP_NETWORK_PROP_SET_DEVICE_ID" "${VIP_INC}/vip_lite.h" &&
    ! grep -q "VIP_NETWORK_PROP_SET_DEVICE_INDEX" "${VIP_INC}/vip_lite.h"; then
    EXTRA_CFLAGS="${EXTRA_CFLAGS} -DA733_VIP_LEGACY_DEVICE_ID"
fi
if ! grep -q "VIP_NETWORK_PROP_SET_CORE_INDEX" "${VIP_INC}/vip_lite.h"; then
    EXTRA_CFLAGS="${EXTRA_CFLAGS} -DA733_VIP_NO_CORE_INDEX"
fi

# EXTRA_CFLAGS is assembled from fixed tokens above.
# shellcheck disable=SC2086
cc -std=c11 -O2 -Wall -Wextra -DNPU_SW_VERSION=2 \
    ${EXTRA_CFLAGS} \
    -I"${VIP_INC}" \
    -o "${OUT}" "${SRC}" \
    -L"${VIP_LIB}" -Wl,-rpath-link,"${VIP_LIB}" -Wl,-rpath,"${VIP_LIB}" \
    -lNBGlinker -lVIPhal -lm

echo "built=${OUT}"
echo "vip_inc=${VIP_INC}"
echo "vip_lib=${VIP_LIB}"
echo "extra_cflags=${EXTRA_CFLAGS}"
