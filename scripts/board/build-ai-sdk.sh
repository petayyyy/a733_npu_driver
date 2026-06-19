#!/usr/bin/env bash
set -eu

SDK_DIR=""
JOBS="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)"
PLATFORM="${AI_SDK_PLATFORM:-a733}"
NPU_VERSION="${NPU_SW_VERSION:-v2.0}"

usage() {
    cat <<'USAGE'
Usage: build-ai-sdk.sh --sdk-dir <path> [--jobs N]

Build an already cloned Allwinner/Radxa ai-sdk tree for A733 VIPLite 2.0.
This script does not clone repositories or install packages.
USAGE
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --sdk-dir)
            SDK_DIR="$2"
            shift 2
            ;;
        --jobs)
            JOBS="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [ -z "${SDK_DIR}" ]; then
    echo "Missing --sdk-dir" >&2
    usage >&2
    exit 2
fi

if [ ! -d "${SDK_DIR}" ]; then
    echo "SDK directory does not exist: ${SDK_DIR}" >&2
    exit 1
fi

if [ ! -f "${SDK_DIR}/Makefile" ]; then
    echo "No Makefile found in ${SDK_DIR}" >&2
    exit 1
fi

cd "${SDK_DIR}"

echo "Building ai-sdk"
echo "SDK_DIR=${SDK_DIR}"
echo "AI_SDK_PLATFORM=${PLATFORM}"
echo "NPU_SW_VERSION=${NPU_VERSION}"
echo "JOBS=${JOBS}"

make "AI_SDK_PLATFORM=${PLATFORM}" "NPU_SW_VERSION=${NPU_VERSION}" -j"${JOBS}"

echo ""
echo "Potential runtime libraries:"
find "${SDK_DIR}" -type f \( -name 'libVIPhal.so*' -o -name 'libNBGlinker.so*' -o -name 'vpm_run' \) 2>/dev/null | sort
