#!/bin/bash
# B5 Benchmark for already-downloaded Qwen2.5 models
set -euo pipefail

LLAMA_BIN="/home/orangepi/llama.cpp/build/bin/llama-completion"
MODEL_DIR="/home/orangepi/a733_npu_driver/models"
LOG_DIR="/home/orangepi/a733_npu_driver/logs/b5-sweep"
mkdir -p "$LOG_DIR"

PROMPT="In one concise technical note, explain why CPU utilization must be measured when benchmarking a small language model on an embedded board with heterogeneous CPU cores such as Cortex-A55 and Cortex-A76."

log_msg() { echo "[$(date -u +%H:%M:%SZ)] $*" | tee -a "$LOG_DIR/sweep.log" >&2; }

CONFIGS=(
  "1xA55__c0 0 1"
  "2xA55__c01 0,1 2"
  "3xA55__c012 0,1,2 3"
  "4xA55__c0123 0,1,2,3 4"
  "1xA76__c6 6 1"
  "2xA76__c67 6,7 2"
  "4mixed__c0167 0,1,6,7 4"
  "6mixed__c012367 0,1,2,3,6,7 6"
  "8all__c0to7 0-7 8"
)

RESULTS="$LOG_DIR/results.dat"
if [ ! -f "$RESULTS" ] || [ ! -s "$RESULTS" ]; then
  echo "label|config|prefill_tok_s|decode_tok_s|avg_cpu_pct|peak_cpu_pct|pct_of_8|peak_rss_mib|temp_max_c|exit_code" > "$RESULTS"
fi

run_one() {
  local label=$1 mpath=$2 cname=$3 cores=$4 threads=$5
  local rid="${label}_${cname}"
  local rlog="$LOG_DIR/${rid}_run.log"
  local plog="$LOG_DIR/${rid}_pidstat.log"
  local mlog="$LOG_DIR/${rid}_mpstat.log"
  local tlog="$LOG_DIR/${rid}_temps.txt"

  log_msg "BENCH: $rid cores=$cores t=$threads"

  # Temps before
  > "$tlog"
  for z in /sys/class/thermal/thermal_zone*/temp; do
    local tv=$(cat "$z" 2>/dev/null) && echo "B $z $tv" >> "$tlog"
  done

  # Start mpstat
  mpstat -P ALL 1 > "$mlog" 2>&1 &
  local mp_pid=$!

  # Start llama in background to get PID
  taskset -c "$cores" "$LLAMA_BIN" \
    -m "$mpath" -p "$PROMPT" \
    -c 2048 -n 128 -t "$threads" --temp 0 --seed 42 -ngl 0 \
    --no-warmup --no-display-prompt -no-cnv --simple-io \
    > "$rlog" 2>&1 &
  local ll_pid=$!

  > "$plog"
  local peak_rss=0
  local sum_cpu=0 cnt_cpu=0 peak_cpu_val=0

  # Sample pidstat and RSS while running
  while kill -0 $ll_pid 2>/dev/null; do
    # RSS
    local rss=$(awk '/VmRSS:/{print $2}' /proc/$ll_pid/status 2>/dev/null || echo 0)
    if [ "$rss" -gt "$peak_rss" ] 2>/dev/null; then peak_rss=$rss; fi

    # pidstat
    local line=$(pidstat -p $ll_pid 1 1 2>/dev/null | grep -E '^[0-9].* [0-9]+ .*[0-9]+' | tail -1)
    if [ -n "$line" ]; then
      echo "$line" >> "$plog"
      local cpu_v=$(echo "$line" | awk '{print $9}')
      if [ -n "$cpu_v" ] && [ "$cpu_v" != "-" ] && [ "$cpu_v" != "CPU" ] && [ "$cpu_v" != "%CPU" ]; then
        sum_cpu=$(echo "$sum_cpu+$cpu_v" | bc 2>/dev/null || echo "$sum_cpu")
        cnt_cpu=$((cnt_cpu+1))
        if (( $(echo "$cpu_v > $peak_cpu_val" | bc -l 2>/dev/null || echo 0) )); then
          peak_cpu_val=$cpu_v
        fi
      fi
    fi
    sleep 1
  done

  wait $ll_pid 2>/dev/null || true
  local ret=$?

  # Kill mpstat
  kill $mp_pid 2>/dev/null || true
  wait $mp_pid 2>/dev/null || true

  # Temps after
  for z in /sys/class/thermal/thermal_zone*/temp; do
    local tv=$(cat "$z" 2>/dev/null) && echo "A $z $tv" >> "$tlog"
  done

  # === PARSE METRICS ===
  local prefill="N/A" decode="N/A"

  # Prefill tok/s
  local pe_line=$(grep "prompt eval time" "$rlog" 2>/dev/null | tail -1)
  if [ -n "$pe_line" ]; then
    local pe_ms=$(echo "$pe_line" | sed -n 's/.*prompt eval time =\s*\([0-9.]*\).*/\1/p')
    local pe_tok=$(echo "$pe_line" | sed -n 's/.*\/\s*\([0-9]*\)\s*tokens.*/\1/p')
    if [ -n "$pe_ms" ] && [ -n "$pe_tok" ] && [ "$pe_tok" != "0" ]; then
      prefill=$(echo "scale=2; $pe_tok/($pe_ms/1000)" | bc)
    fi
  fi

  # Decode tok/s
  local ev_line=$(grep "eval time" "$rlog" 2>/dev/null | grep -v prompt | tail -1)
  if [ -n "$ev_line" ]; then
    local ev_ms=$(echo "$ev_line" | sed -n 's/.*eval time =\s*\([0-9.]*\).*/\1/p')
    local ev_tok=$(echo "$ev_line" | sed -n 's/.*\/\s*\([0-9]*\)\s*runs.*/\1/p')
    if [ -n "$ev_ms" ] && [ -n "$ev_tok" ] && [ "$ev_tok" != "0" ]; then
      decode=$(echo "scale=2; $ev_tok/($ev_ms/1000)" | bc)
    fi
  fi

  # CPU%
  local avg_cpu="N/A" peak_cpu="N/A" pct8="N/A"
  if [ $cnt_cpu -gt 0 ]; then
    avg_cpu=$(echo "scale=1; $sum_cpu/$cnt_cpu" | bc)
    peak_cpu=$peak_cpu_val
    pct8=$(echo "scale=0; $avg_cpu/8" | bc)
  fi

  # RSS in MiB
  local rss_mib="N/A"
  if [ "$peak_rss" -gt 0 ] 2>/dev/null; then
    rss_mib=$(echo "scale=0; $peak_rss/1024" | bc)
  fi

  # Max temp
  local max_temp="N/A"
  local all_t=$(grep -oP '[0-9]+' "$tlog" 2>/dev/null | sort -rn | head -1)
  if [ -n "$all_t" ]; then
    max_temp=$(echo "scale=1; $all_t/1000" | bc)
  fi

  echo "$label|$cname|$prefill|$decode|$avg_cpu|$peak_cpu|$pct8|$rss_mib|$max_temp|$ret" >> "$RESULTS"
  log_msg "DONE: $rid pf=$prefill dk=$decode cpu=$avg_cpu% pcp=$peak_cpu% p8=${pct8}% rss=${rss_mib}MiB t=${max_temp}C"
}

# === FIT TEST ===
fit_test() {
  local label=$1 mpath=$2
  log_msg "FIT: $label"
  free -h > "$LOG_DIR/${label}_free_before.txt"

  "$LLAMA_BIN" \
    -m "$mpath" -p "$PROMPT" -c 2048 -n 1 -t 2 --temp 0 --seed 42 -ngl 0 \
    --no-warmup --no-display-prompt -no-cnv --simple-io \
    > "$LOG_DIR/${label}_fit.log" 2>&1 &
  local ll_pid=$!

  local peak_kb=0
  while kill -0 $ll_pid 2>/dev/null; do
    local rss=$(awk '/VmRSS:/{print $2}' /proc/$ll_pid/status 2>/dev/null || echo 0)
    if [ "$rss" -gt "$peak_kb" ] 2>/dev/null; then peak_kb=$rss; fi
    sleep 0.5
  done
  wait $ll_pid 2>/dev/null || true
  local ret=$?

  free -h > "$LOG_DIR/${label}_free_after.txt"

  if grep -q "eval time" "$LOG_DIR/${label}_fit.log" 2>/dev/null; then
    local peak_mib=$(echo "scale=0; $peak_kb/1024" | bc)
    log_msg "FIT_OK: $label (peak_rss=${peak_mib}MiB)"
    echo "FIT"
  elif [ $ret -eq 137 ] || [ $ret -eq 124 ]; then
    local peak_mib=$(echo "scale=0; $peak_kb/1024" | bc)
    log_msg "FIT_OOM: $label (exit=$ret, peak_rss=${peak_mib}MiB)"
    echo "OOM"
  else
    log_msg "FIT_FAIL: $label (exit=$ret)"
    echo "FAIL"
  fi
}

# === MAIN ===

MODELS=(
  "qwen25_15b_q4km:/home/orangepi/a733_npu_driver/models/qwen2.5-1.5b-instruct-q4_k_m.gguf"
  "qwen25_15b_q8:/home/orangepi/a733_npu_driver/models/qwen2.5-1.5b-instruct-q8_0.gguf"
  "qwen25_3b_q4km:/home/orangepi/a733_npu_driver/models/qwen2.5-3b-instruct-q4_k_m.gguf"
  "qwen25_7b_q4km:/home/orangepi/a733_npu_driver/models/qwen2.5-7b-instruct-q4_k_m.gguf"
)

for mentry in "${MODELS[@]}"; do
  ml="${mentry%%:*}"
  mp="${mentry##*:}"
  [ ! -f "$mp" ] && { log_msg "SKIP $ml: file missing"; continue; }

  log_msg "========== $ml =========="

  # Skip if already done
  if grep -q "^${ml}|8all" "$RESULTS" 2>/dev/null; then
    log_msg "SKIP $ml: already complete in results"
    continue
  fi

  ft=$(fit_test "$ml" "$mp")
  if [ "$ft" != "FIT" ]; then
    log_msg "SKIP $ml: $ft"
    echo "$ml|FIT_TEST|N/A|N/A|N/A|N/A|N/A|N/A|N/A|$ft" >> "$RESULTS"
    continue
  fi

  for c_entry in "${CONFIGS[@]}"; do
    read -r cn cc ct <<< "$c_entry"
    sleep 2
    run_one "$ml" "$mp" "$cn" "$cc" "$ct"
  done
  free -h >> "$LOG_DIR/sweep.log"
done

log_msg "========== ALL DONE =========="
