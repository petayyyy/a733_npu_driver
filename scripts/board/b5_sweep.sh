#!/bin/bash
# B5 Qwen2.5 Size Sweep for Orange Pi Zero 3W
set -euo pipefail

LLAMA_BIN="/home/orangepi/llama.cpp/build/bin/llama-completion"
MODEL_DIR="/home/orangepi/a733_npu_driver/models"
LOG_DIR="/home/orangepi/a733_npu_driver/logs/b5-sweep"
mkdir -p "$MODEL_DIR" "$LOG_DIR"

PROMPT="In one concise technical note, explain why CPU utilization must be measured when benchmarking a small language model on an embedded board with heterogeneous CPU cores such as Cortex-A55 and Cortex-A76."
WARMUP="Hello"

log_msg() {
  echo "[$(date -u +%H:%M:%SZ)] $*" | tee -a "$LOG_DIR/sweep.log"
}

# === DOWNLOAD MODEL ===
download_model() {
  local fname=$1
  local url=$2
  local path="$MODEL_DIR/$fname"
  if [ -f "$path" ]; then
    log_msg "Already exists: $fname ($(ls -lh "$path" | awk '{print $5}'))"
    return
  fi
  log_msg "Downloading $fname ..."
  wget -q --show-progress -O "$path.tmp" "$url"
  mv "$path.tmp" "$path"
  log_msg "Downloaded: $(ls -lh "$path" | awk '{print $5}')"
}

# === FIT TEST: does model load at ctx=2048? ===
fit_test() {
  local label=$1
  local mpath=$2
  log_msg "FIT_TEST: $label"

  free -h > "$LOG_DIR/${label}_free_before.txt"

  set +e
  taskset -c 0,1 timeout 60 "$LLAMA_BIN" \
    -m "$mpath" -p "$PROMPT" \
    -c 2048 -n 1 -t 2 --temp 0 --seed 42 -ngl 0 \
    --no-warmup --no-display-prompt -no-cnv --simple-io \
    > "$LOG_DIR/${label}_fit.log" 2>&1
  local ret=$?
  set -e

  free -h > "$LOG_DIR/${label}_free_after.txt"

  # Check peak RSS from log
  local rss_line=$(grep "peak_rss" "$LOG_DIR/${label}_fit.log" 2>/dev/null || echo "")
  if [ -n "$rss_line" ]; then
    log_msg "FIT: $label LOADS, $rss_line"
    echo "FIT"
  elif [ $ret -eq 137 ] || [ $ret -eq 124 ]; then
    log_msg "FIT: $label OOM/KILLED (exit=$ret)"
    echo "OOM"
  else
    log_msg "FIT: $label FAILED (exit=$ret)"
    echo "FAIL"
  fi
}

# === RUN SINGLE CONFIG ===
run_config() {
  local label=$1
  local mpath=$2
  local cname=$3
  local cores=$4
  local threads=$5
  local run_id="${label}_${cname}"

  log_msg "RUN: $run_id cores=$cores t=$threads"

  # Record temps before
  echo "BEFORE" > "$LOG_DIR/${run_id}_temps.txt"
  for z in /sys/class/thermal/thermal_zone*/temp; do
    local tv=$(cat "$z" 2>/dev/null) && echo "$z $tv" >> "$LOG_DIR/${run_id}_temps.txt"
  done

  # Start mpstat
  mpstat -P ALL 1 > "$LOG_DIR/${run_id}_mpstat.log" 2>&1 &
  local mp_pid=$!

  # Run llama-completion
  set +e
  taskset -c "$cores" timeout 300 "$LLAMA_BIN" \
    -m "$mpath" -p "$PROMPT" \
    -c 2048 -n 128 -t "$threads" --temp 0 --seed 42 -ngl 0 \
    --no-warmup --no-display-prompt -no-cnv --simple-io \
    > "$LOG_DIR/${run_id}_run.log" 2>&1 &
  local ll_pid=$!
  set -e

  # Sample RSS and pidstat while running
  local peak_rss=0
  > "$LOG_DIR/${run_id}_rss.log"
  > "$LOG_DIR/${run_id}_pidstat.log"
  sleep 2

  while kill -0 $ll_pid 2>/dev/null; do
    local rss=$(ps -o rss= -p $ll_pid 2>/dev/null || echo 0)
    echo "$rss" >> "$LOG_DIR/${run_id}_rss.log"
    if [ "$rss" -gt "$peak_rss" ] 2>/dev/null; then peak_rss=$rss; fi
    pidstat -p $ll_pid 1 1 >> "$LOG_DIR/${run_id}_pidstat.log" 2>/dev/null || true
    sleep 0.5
  done

  wait $ll_pid 2>/dev/null || true
  local ret=$?

  # Kill mpstat
  kill $mp_pid 2>/dev/null || true
  wait $mp_pid 2>/dev/null || true

  # Temps after
  echo "AFTER" >> "$LOG_DIR/${run_id}_temps.txt"
  for z in /sys/class/thermal/thermal_zone*/temp; do
    local tv=$(cat "$z" 2>/dev/null) && echo "$z $tv" >> "$LOG_DIR/${run_id}_temps.txt"
  done

  # === PARSE METRICS ===
  local prefill="N/A"
  local decode="N/A"
  local avg_cpu="N/A"
  local peak_cpu="N/A"
  local pct8="N/A"
  local rss_mib="N/A"
  local max_temp="N/A"

  # Prefill tok/s from llama output
  local pe_line=$(grep "prompt eval time" "$LOG_DIR/${run_id}_run.log" 2>/dev/null | tail -1)
  if [ -n "$pe_line" ]; then
    local pe_ms=$(echo "$pe_line" | sed -n 's/.*prompt eval time =\s*\([0-9.]*\).*/\1/p')
    local pe_tok=$(echo "$pe_line" | sed -n 's/.*\/\s*\([0-9]*\)\s*tokens.*/\1/p')
    if [ -n "$pe_ms" ] && [ -n "$pe_tok" ] && [ "$pe_tok" != "0" ]; then
      local pe_sec=$(echo "scale=6; $pe_ms/1000" | bc)
      prefill=$(echo "scale=2; $pe_tok/$pe_sec" | bc)
    fi
  fi

  # Decode tok/s from llama output
  local ev_line=$(grep "eval time" "$LOG_DIR/${run_id}_run.log" 2>/dev/null | grep -v prompt | tail -1)
  if [ -n "$ev_line" ]; then
    local ev_ms=$(echo "$ev_line" | sed -n 's/.*eval time =\s*\([0-9.]*\).*/\1/p')
    local ev_tok=$(echo "$ev_line" | sed -n 's/.*\/\s*\([0-9]*\)\s*tokens.*/\1/p')
    if [ -n "$ev_ms" ] && [ -n "$ev_tok" ] && [ "$ev_tok" != "0" ]; then
      local ev_sec=$(echo "scale=6; $ev_ms/1000" | bc)
      decode=$(echo "scale=2; $ev_tok/$ev_sec" | bc)
    fi
  fi

  # CPU% from pidstat
  local cpu_vals=$(grep -E '^[0-9]' "$LOG_DIR/${run_id}_pidstat.log" 2>/dev/null | awk '{print $8}' | grep -v '^$' | grep -v CPU || echo "")
  if [ -n "$cpu_vals" ]; then
    peak_cpu=$(echo "$cpu_vals" | sort -rn | head -1)
    local sum=0; local cnt=0
    for v in $cpu_vals; do
      sum=$(echo "$sum+$v" | bc)
      cnt=$((cnt+1))
    done
    if [ $cnt -gt 0 ]; then
      avg_cpu=$(echo "scale=1; $sum/$cnt" | bc)
      pct8=$(echo "scale=0; $avg_cpu/8" | bc)
    fi
  fi

  # RSS in MiB
  if [ "$peak_rss" -gt 0 ] 2>/dev/null; then
    rss_mib=$(echo "scale=0; $peak_rss/1024" | bc)
  fi

  # Max temp
  local all_temps=$(grep -oP '[0-9]+' "$LOG_DIR/${run_id}_temps.txt" 2>/dev/null | sort -rn || echo "")
  if [ -n "$all_temps" ]; then
    local max_raw=$(echo "$all_temps" | head -1)
    max_temp=$(echo "scale=1; $max_raw/1000" | bc)
  fi

  echo "$label|$cname|$prefill|$decode|$avg_cpu|$peak_cpu|$pct8|$rss_mib|$max_temp|$ret" >> "$LOG_DIR/results.dat"

  log_msg "DONE: $run_id pf=$prefill dk=$decode cpu=$avg_cpu% pcp=$peak_cpu% p8=$pct8% rss=${rss_mib}MiB t=${max_temp}C"
}

# === MAIN ===
echo "label|config|prefill_tok_s|decode_tok_s|avg_cpu_pct|peak_cpu_pct|pct_of_8|peak_rss_mib|temp_max_c|exit_code" > "$LOG_DIR/results.dat"

# Model list: label|filename|url
download_model "qwen2.5-0.5b-instruct-q4_k_m.gguf" \
  "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf"

download_model "qwen2.5-0.5b-instruct-q8_0.gguf" \
  "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q8_0.gguf"

download_model "qwen2.5-1.5b-instruct-q4_k_m.gguf" \
  "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf"

download_model "qwen2.5-1.5b-instruct-q8_0.gguf" \
  "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q8_0.gguf"

download_model "qwen2.5-3b-instruct-q4_k_m.gguf" \
  "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf"

download_model "qwen2.5-7b-instruct-q4_k_m.gguf" \
  "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/main/qwen2.5-7b-instruct-q4_k_m.gguf"

# Core configs: label cores threads
CONFIGS=(
  "1xA55__c0:0:1"
  "2xA55__c01:0,1:2"
  "3xA55__c012:0,1,2:3"
  "4xA55__c0123:0,1,2,3:4"
  "1xA76__c6:6:1"
  "2xA76__c67:6,7:2"
  "4mixed__c0167:0,1,6,7:4"
  "6mixed__c012367:0,1,2,3,6,7:6"
  "8all__c0to7:0-7:8"
)

MODELS=(
  "qwen25_05b_q4km:qwen2.5-0.5b-instruct-q4_k_m.gguf"
  "qwen25_05b_q8:qwen2.5-0.5b-instruct-q8_0.gguf"
  "qwen25_15b_q4km:qwen2.5-1.5b-instruct-q4_k_m.gguf"
  "qwen25_15b_q8:qwen2.5-1.5b-instruct-q8_0.gguf"
  "qwen25_3b_q4km:qwen2.5-3b-instruct-q4_k_m.gguf"
  "qwen25_7b_q4km:qwen2.5-7b-instruct-q4_k_m.gguf"
)

for m_entry in "${MODELS[@]}"; do
  m_label="${m_entry%%:*}"
  m_file="${m_entry##*:}"
  m_path="$MODEL_DIR/$m_file"

  log_msg "========== $m_label =========="

  # Fit test
  ft=$(fit_test "$m_label" "$m_path")

  if [ "$ft" != "FIT" ]; then
    log_msg "SKIP $m_label: $ft"
    echo "$m_label|FIT_TEST|N/A|N/A|N/A|N/A|N/A|N/A|N/A|$ft" >> "$LOG_DIR/results.dat"
    continue
  fi

  # Core sweep
  for c_entry in "${CONFIGS[@]}"; do
    c_name="${c_entry%%:*}"
    rest="${c_entry#*:}"
    c_cores="${rest%%:*}"
    c_threads="${rest##*:}"
    sleep 3
    run_config "$m_label" "$m_path" "$c_name" "$c_cores" "$c_threads"
  done

  free -h >> "$LOG_DIR/sweep.log"
done

log_msg "========== ALL DONE =========="
log_msg "Results: $LOG_DIR/results.dat"
free -h >> "$LOG_DIR/sweep.log"
