#!/usr/bin/env bash
set -u

usage() {
    cat >&2 <<'USAGE'
Usage: run-b3-vpm-package.sh --model-dir DIR --log-dir DIR [options]

Options:
  --vpm-run FILE    vpm_run binary, default: /opt/vpm_run/vpm_run
  --vip-lib DIR     VIPLite library directory, default: /home/orangepi/lib
  --loops N         vpm_run loop count, default: 5
  --device N        VIP device index, default: 0
USAGE
    exit 2
}

MODEL_DIR=
LOG_DIR=
VPM_RUN=/opt/vpm_run/vpm_run
VIP_LIB=/home/orangepi/lib
LOOPS=5
DEVICE=0

while [ "$#" -gt 0 ]; do
    case "$1" in
        --model-dir)
            MODEL_DIR=$2
            shift 2
            ;;
        --log-dir)
            LOG_DIR=$2
            shift 2
            ;;
        --vpm-run)
            VPM_RUN=$2
            shift 2
            ;;
        --vip-lib)
            VIP_LIB=$2
            shift 2
            ;;
        --loops)
            LOOPS=$2
            shift 2
            ;;
        --device)
            DEVICE=$2
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

[ -n "$MODEL_DIR" ] || usage
[ -n "$LOG_DIR" ] || usage
[ -d "$MODEL_DIR" ] || { echo "model dir not found: $MODEL_DIR" >&2; exit 1; }
[ -x "$VPM_RUN" ] || { echo "vpm_run not executable: $VPM_RUN" >&2; exit 1; }

mkdir -p "$LOG_DIR"
cd "$MODEL_DIR" || exit 1
rm -f output_*.txt output_*.dat

export LD_LIBRARY_PATH="${VIP_LIB}:${LD_LIBRARY_PATH:-}"

{
    echo "RUN_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "model_dir=$MODEL_DIR"
    echo "log_dir=$LOG_DIR"
    echo "vpm_run=$VPM_RUN"
    echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
    echo "command=$VPM_RUN -s sample.txt -l $LOOPS -d $DEVICE -b 0 --show_top5 1 --save_txt 1"
} > "$LOG_DIR/run.log"

"$VPM_RUN" -s sample.txt -l "$LOOPS" -d "$DEVICE" \
    -b 0 --show_top5 1 --save_txt 1 \
    > "$LOG_DIR/vpm_run.out" 2> "$LOG_DIR/vpm_run.err" &
pid=$!
peak_rss_kb=0
peak_hwm_kb=0
while kill -0 "$pid" 2>/dev/null; do
    if [ -r "/proc/$pid/status" ]; then
        rss_kb=0
        hwm_kb=0
        while IFS=" :" read -r key value _rest; do
            case "$key" in
                VmRSS)
                    rss_kb=$value
                    ;;
                VmHWM)
                    hwm_kb=$value
                    ;;
            esac
        done < "/proc/$pid/status" 2>/dev/null || true
        rss_kb=${rss_kb:-0}
        hwm_kb=${hwm_kb:-0}
        if [ "$rss_kb" -gt "$peak_rss_kb" ]; then
            peak_rss_kb=$rss_kb
        fi
        if [ "$hwm_kb" -gt "$peak_hwm_kb" ]; then
            peak_hwm_kb=$hwm_kb
        fi
    fi
    sleep 0.01
done
wait "$pid"
status=$?

{
    echo "exit=$status"
    echo "peak_rss_kb=$peak_rss_kb"
    echo "peak_hwm_kb=$peak_hwm_kb"
    cat "$LOG_DIR/vpm_run.out"
    cat "$LOG_DIR/vpm_run.err"
} >> "$LOG_DIR/run.log"

if [ -f output_0.txt ]; then
    cp output_0.txt "$LOG_DIR/output_0.txt"
fi
if [ -f output_0.dat ]; then
    cp output_0.dat "$LOG_DIR/output_0.dat"
fi

exit "$status"
