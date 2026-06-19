#!/usr/bin/env bash
set -u

ROOT_DIR="${A733_LOG_ROOT:-logs/board}"
SEARCH_DIRS="${A733_SEARCH_DIRS:-/home/radxa/lib /home/radxa/yolo_shm /home/radxa /usr /opt /lib /root $PWD}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
HOST="$(hostname 2>/dev/null || echo board)"
OUT_DIR="${ROOT_DIR}/${HOST}-${STAMP}"
SUMMARY="${OUT_DIR}/summary.env"

mkdir -p "${OUT_DIR}"

pass_count=0
fail_count=0
warn_count=0

log() {
    printf '%s\n' "$*" | tee -a "${OUT_DIR}/smoke.log"
}

record() {
    printf '%s=%s\n' "$1" "$2" >> "${SUMMARY}"
}

run_capture() {
    name="$1"
    shift
    log ""
    log "### ${name}"
    log "+ $*"
    "$@" > "${OUT_DIR}/${name}.out" 2> "${OUT_DIR}/${name}.err"
    status=$?
    cat "${OUT_DIR}/${name}.out" >> "${OUT_DIR}/smoke.log"
    cat "${OUT_DIR}/${name}.err" >> "${OUT_DIR}/smoke.log"
    log "exit=${status}"
    return "${status}"
}

check_pass() {
    pass_count=$((pass_count + 1))
    record "$1" "pass"
    log "PASS: $2"
}

check_warn() {
    warn_count=$((warn_count + 1))
    record "$1" "warn"
    log "WARN: $2"
}

check_fail() {
    fail_count=$((fail_count + 1))
    record "$1" "fail"
    log "FAIL: $2"
}

find_command() {
    cmd="$1"
    if command -v "${cmd}" >/dev/null 2>&1; then
        command -v "${cmd}"
        return 0
    fi
    # shellcheck disable=SC2086
    find ${SEARCH_DIRS} -type f -name "${cmd}" 2>/dev/null | head -n 1
}

: > "${SUMMARY}"
record "timestamp_utc" "${STAMP}"
record "host" "${HOST}"
log "A733 G0/G1 smoke test"
log "Log directory: ${OUT_DIR}"

run_capture uname uname -a || true

if [ -r /etc/os-release ]; then
    run_capture os_release cat /etc/os-release || true
else
    check_warn os_release "no /etc/os-release"
fi

if command -v nproc >/dev/null 2>&1; then
    cores="$(nproc 2>/dev/null || echo 0)"
else
    cores="$(grep -c '^processor' /proc/cpuinfo 2>/dev/null || echo 0)"
fi
record "cpu_cores" "${cores}"
if [ "${cores}" -ge 8 ] 2>/dev/null; then
    check_pass g0_cpu_cores "detected ${cores} CPU cores"
else
    check_fail g0_cpu_cores "expected 8 CPU cores, detected ${cores}"
fi

run_capture cpuinfo sh -c "cat /proc/cpuinfo" || true

if [ -d /sys/class/thermal ]; then
    run_capture thermals sh -c "for z in /sys/class/thermal/thermal_zone*; do [ -e \"\$z/temp\" ] && printf '%s ' \"\$z\" && cat \"\$z/type\" \"\$z/temp\" 2>/dev/null; done" || true
    check_pass g0_thermals "thermal sysfs is readable"
else
    check_warn g0_thermals "thermal sysfs not found"
fi

if [ -d /sys/devices/system/cpu ]; then
    run_capture governors sh -c "find /sys/devices/system/cpu -path '*/cpufreq/scaling_governor' -print -exec cat {} \;" || true
fi

if [ -e /dev/vipcore ]; then
    run_capture vipcore ls -l /dev/vipcore || true
    check_pass g1_vipcore "found /dev/vipcore"
else
    check_fail g1_vipcore "missing /dev/vipcore"
fi

if command -v ldconfig >/dev/null 2>&1; then
    run_capture viplite_ldconfig sh -c "ldconfig -p 2>/dev/null | grep -Ei 'VIP|NBG|vsi|awnn|OpenVX' || true" || true
fi

run_capture viplite_find sh -c "find ${SEARCH_DIRS} -type f \( -name 'libVIPhal.so*' -o -name 'libNBGlinker.so*' -o -name 'libVIP*.so*' -o -name 'libawnn*' -o -name 'vpm_run' -o -name '*.nb' -o -name '*.nbg' \) 2>/dev/null | sort" || true

# shellcheck disable=SC2086
vip_lib_file="$(find ${SEARCH_DIRS} -type f \( -name 'libVIPhal.so*' -o -name 'libNBGlinker.so*' \) 2>/dev/null | head -n 1)"
if [ -n "${vip_lib_file}" ]; then
    vip_lib_dir="$(dirname "${vip_lib_file}")"
    export LD_LIBRARY_PATH="${vip_lib_dir}:${LD_LIBRARY_PATH:-}"
    record "vip_libraries_dir" "${vip_lib_dir}"
    check_pass g1_viplite_libs "found VIPLite libraries in ${vip_lib_dir}"
else
    check_warn g1_viplite_libs "VIPLite libraries not found in common paths"
fi

vpm_run_path="${A733_VPM_RUN:-}"
if [ -z "${vpm_run_path}" ]; then
    vpm_run_path="$(find_command vpm_run || true)"
fi

if [ -n "${vpm_run_path}" ] && [ -x "${vpm_run_path}" ]; then
    record "vpm_run" "${vpm_run_path}"
    check_pass g1_vpm_run "found executable vpm_run at ${vpm_run_path}"
    run_capture vpm_run_probe "${vpm_run_path}" --help || true
else
    check_warn g1_vpm_run "vpm_run not found or not executable"
fi

if [ -n "${A733_VPM_RUN_ARGS:-}" ] && [ -n "${vpm_run_path}" ] && [ -x "${vpm_run_path}" ]; then
    log ""
    log "### vpm_run_model"
    log "+ ${vpm_run_path} ${A733_VPM_RUN_ARGS}"
    # shellcheck disable=SC2086
    "${vpm_run_path}" ${A733_VPM_RUN_ARGS} > "${OUT_DIR}/vpm_run_model.out" 2> "${OUT_DIR}/vpm_run_model.err"
    status=$?
    cat "${OUT_DIR}/vpm_run_model.out" >> "${OUT_DIR}/smoke.log"
    cat "${OUT_DIR}/vpm_run_model.err" >> "${OUT_DIR}/smoke.log"
    log "exit=${status}"
    if [ "${status}" -eq 0 ]; then
        check_pass g1_vpm_inference "vpm_run model command completed"
    else
        check_fail g1_vpm_inference "vpm_run model command failed with exit ${status}"
    fi
    if grep -Eiq 'cid=0x1000003b|0x1000003b|VIPLite|vipcore' "${OUT_DIR}/vpm_run_model.out" "${OUT_DIR}/vpm_run_model.err"; then
        check_pass g1_vip_banner "VIPLite/A733 identity found in vpm_run logs"
    else
        check_warn g1_vip_banner "VIPLite/A733 identity not found in vpm_run logs"
    fi
else
    check_warn g1_vpm_inference "set A733_VPM_RUN_ARGS to run a real NBG inference"
fi

if [ -n "${A733_NPU_RUN_CMD:-}" ]; then
    log ""
    log "### npu_run_cmd"
    log "+ ${A733_NPU_RUN_CMD}"
    sh -c "${A733_NPU_RUN_CMD}" > "${OUT_DIR}/npu_run_cmd.out" 2> "${OUT_DIR}/npu_run_cmd.err"
    status=$?
    cat "${OUT_DIR}/npu_run_cmd.out" >> "${OUT_DIR}/smoke.log"
    cat "${OUT_DIR}/npu_run_cmd.err" >> "${OUT_DIR}/smoke.log"
    log "exit=${status}"
    if [ "${status}" -eq 0 ]; then
        check_pass g1_npu_inference "custom NPU command completed"
    else
        check_fail g1_npu_inference "custom NPU command failed with exit ${status}"
    fi
    if grep -Eiq 'VIPLite|vipcore|detection num|create network|prepare network|0x1000003b|cid=0x1000003b' "${OUT_DIR}/npu_run_cmd.out" "${OUT_DIR}/npu_run_cmd.err"; then
        check_pass g1_npu_banner "NPU/VIPLite evidence found in custom command logs"
    else
        check_warn g1_npu_banner "NPU/VIPLite evidence not found in custom command logs"
    fi
fi

if command -v dmesg >/dev/null 2>&1; then
    run_capture dmesg_npu sh -c "dmesg 2>/dev/null | grep -Ei 'vip|vivante|npu|galcore|vsi' | tail -n 200 || true" || true
fi

record "pass_count" "${pass_count}"
record "warn_count" "${warn_count}"
record "fail_count" "${fail_count}"

log ""
log "Summary: pass=${pass_count} warn=${warn_count} fail=${fail_count}"
log "Summary file: ${SUMMARY}"

if [ "${fail_count}" -gt 0 ]; then
    exit 1
fi

exit 0
