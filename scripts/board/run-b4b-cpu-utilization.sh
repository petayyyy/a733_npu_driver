#!/usr/bin/env bash
set -euo pipefail

ROOT="${A733_ROOT:-$HOME/a733_npu_driver}"
LLAMA_BIN_DIR="${A733_LLAMA_BIN_DIR:-$HOME/llama.cpp/build/bin}"
MODEL_DIR="${A733_GGUF_DIR:-$ROOT/models/gguf}"
LOG_DIR="${A733_B4B_LOG_DIR:-$ROOT/logs/board/b4b-cpu-utilization}"

LLAMA_COMPLETION="$LLAMA_BIN_DIR/llama-completion"
Q8_MODEL="${A733_QWEN_Q8_MODEL:-$MODEL_DIR/qwen2.5-0.5b-instruct-q8_0.gguf}"
CTX="${A733_B4B_CTX:-2048}"
TOKENS="${A733_B4B_TOKENS:-128}"
PROMPT_FILE="$LOG_DIR/prompt.txt"

export LD_LIBRARY_PATH="$LLAMA_BIN_DIR:${LD_LIBRARY_PATH:-}"

for path in "$LLAMA_COMPLETION" "$Q8_MODEL"; do
  if [ ! -e "$path" ]; then
    echo "missing required path: $path" >&2
    exit 1
  fi
done

for tool in pidstat mpstat taskset awk; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "missing required tool: $tool" >&2
    exit 1
  fi
done

mkdir -p "$LOG_DIR"

cat > "$PROMPT_FILE" <<'PROMPT'
In one concise technical note, explain why CPU utilization must be measured when benchmarking a small language model on an embedded board. Mention throughput, thermal headroom, scheduling impact, and why robotics workloads care about spare cores.
PROMPT

{
  echo "date_utc=$(date -u -Iseconds)"
  echo "hostname=$(hostname)"
  echo "kernel=$(uname -a)"
  echo "nproc=$(getconf _NPROCESSORS_ONLN)"
  echo "llama_completion=$LLAMA_COMPLETION"
  echo "q8_model=$Q8_MODEL"
  echo "ctx=$CTX"
  echo "tokens=$TOKENS"
  echo "prompt_file=$PROMPT_FILE"
  echo "thread_cpuset_map=t2:6,7 t4:4-7 t6:2-7 t8:0-7"
} > "$LOG_DIR/env.txt"

lscpu > "$LOG_DIR/lscpu.txt" 2>&1 || true
free -h > "$LOG_DIR/free-before.txt" 2>&1 || true
df -h / /home > "$LOG_DIR/df-before.txt" 2>&1 || true
"$LLAMA_COMPLETION" --version > "$LOG_DIR/llama-completion-version.txt" 2>&1 || true
sha256sum "$Q8_MODEL" > "$LOG_DIR/model-sha256.txt" 2>&1 || true

capture_thermal() {
  local out="$1"
  {
    date -u -Iseconds
    for zone in /sys/class/thermal/thermal_zone*/temp; do
      if [ -r "$zone" ]; then
        printf "%s " "$zone"
        cat "$zone"
      fi
    done
  } > "$out" 2>&1 || true
}

sample_rss() {
  local pid="$1"
  local out="$2"
  echo "timestamp_ms rss_kb" > "$out"
  while kill -0 "$pid" >/dev/null 2>&1; do
    local rss="0"
    if [ -r "/proc/$pid/status" ]; then
      rss="$(awk '/^VmRSS:/ {print $2}' "/proc/$pid/status")"
      [ -n "$rss" ] || rss="0"
    fi
    printf "%s %s\n" "$(date +%s%3N)" "$rss" >> "$out"
    sleep 0.2
  done
}

run_one() {
  local threads="$1"
  local cpuset="$2"
  local name="q8-ctx${CTX}-t${threads}-cpu${cpuset//,/-}"
  name="${name//\//-}"

  echo "RUN $name"
  capture_thermal "$LOG_DIR/$name.thermal-before.txt"

  taskset -c "$cpuset" "$LLAMA_COMPLETION" \
    -m "$Q8_MODEL" \
    -f "$PROMPT_FILE" \
    -c "$CTX" \
    -n "$TOKENS" \
    -t "$threads" \
    -tb "$threads" \
    -ngl 0 \
    --temp 0 \
    --seed 42 \
    --no-warmup \
    --no-display-prompt \
    -no-cnv \
    --simple-io \
    --color off \
    > "$LOG_DIR/$name.stdout.log" \
    2> "$LOG_DIR/$name.stderr.log" &
  local pid="$!"
  echo "$pid" > "$LOG_DIR/$name.pid"
  sleep 0.5

  {
    date -u -Iseconds
    taskset -cp "$pid" || true
    if [ -r "/proc/$pid/status" ]; then
      grep -E '^(Cpus_allowed_list|Mems_allowed_list|VmRSS|VmHWM):' "/proc/$pid/status" || true
    fi
  } > "$LOG_DIR/$name.affinity.txt" 2>&1

  pidstat -p "$pid" -u -r 1 > "$LOG_DIR/$name.pidstat.log" 2>&1 &
  local pidstat_pid="$!"
  mpstat -P ALL 1 > "$LOG_DIR/$name.mpstat.log" 2>&1 &
  local mpstat_pid="$!"
  sample_rss "$pid" "$LOG_DIR/$name.rss.log" &
  local rss_pid="$!"

  local rc=0
  wait "$pid" || rc="$?"

  kill "$pidstat_pid" "$mpstat_pid" "$rss_pid" >/dev/null 2>&1 || true
  wait "$pidstat_pid" "$mpstat_pid" "$rss_pid" >/dev/null 2>&1 || true

  echo "$rc" > "$LOG_DIR/$name.exitcode"
  capture_thermal "$LOG_DIR/$name.thermal-after.txt"
  echo "DONE $name rc=$rc"
  return "$rc"
}

run_one 2 "6,7"
run_one 4 "4-7"
run_one 6 "2-7"
run_one 8 "0-7"

capture_thermal "$LOG_DIR/thermal-final.txt"
free -h > "$LOG_DIR/free-after.txt" 2>&1 || true
df -h / /home > "$LOG_DIR/df-after.txt" 2>&1 || true
echo "logs=$LOG_DIR"
