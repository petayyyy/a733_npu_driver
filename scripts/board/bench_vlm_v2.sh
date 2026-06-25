#!/bin/bash
# V1 VLM benchmark - one-shot image+text test
set -euo pipefail

MODEL_DIR=/home/orangepi/a733_npu_driver/models/vlm
LLAMA=/home/orangepi/llama.cpp/build/bin/llama-cli
IMAGE_DIR=/home/orangepi/a733_npu_driver/test_images
LOG_DIR=/home/orangepi/a733_npu_driver/logs/v1-vlm
mkdir -p "$LOG_DIR"

MODEL_BASE="$1"    # e.g. SmolVLM-256M-Instruct
IMAGE_NAME="$2"    # e.g. dog.jpg
PROMPT="$3"        # e.g. "Describe this image."
THREADS="${4:-2}"

MODEL="$MODEL_DIR/${MODEL_BASE}-Q8_0.gguf"
MMPROJ="$MODEL_DIR/mmproj-${MODEL_BASE}-Q8_0.gguf"
IMAGE="$IMAGE_DIR/${IMAGE_NAME}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG="$LOG_DIR/${MODEL_BASE}_${IMAGE_NAME%.*}_${TIMESTAMP}.log"
RSS_LOG="$LOG_DIR/${MODEL_BASE}_${IMAGE_NAME%.*}_${TIMESTAMP}_rss.csv"

echo "V1 VLM Benchmark" | tee "$LOG"
echo "Model: $MODEL_BASE" | tee -a "$LOG"
echo "Image: $IMAGE_NAME" | tee -a "$LOG"
echo "Prompt: $PROMPT" | tee -a "$LOG"
echo "Threads: $THREADS (A76 cores 6,7)" | tee -a "$LOG"
echo "Started: $(date -Iseconds)" | tee -a "$LOG"
echo "---" | tee -a "$LOG"

# Kill any previous runs
pkill -9 llama-cli 2>/dev/null || true
sleep 2

# Start RSS monitor in background
(
  echo "timestamp,pid,rss_kb"
  while true; do
    P=$(pgrep -f llama-cli | head -1)
    if [ -z "$P" ]; then
      sleep 2
      P=$(pgrep -f llama-cli | head -1)
      if [ -z "$P" ]; then break; fi
    fi
    R=$(awk '/^VmRSS:/{print $2}' /proc/$P/status 2>/dev/null)
    echo "$(date +%s.%N),${P:-0},${R:-0}"
    sleep 0.2
  done
) > "$RSS_LOG" &
RSS_PID=$!

# Get free memory before
FREE_BEFORE=$(free -m | grep Mem | awk '{print $7}')
echo "Free memory before: ${FREE_BEFORE} MB" | tee -a "$LOG"

# Run VLM in one-shot mode:
# Uses printf to send prompt + /exit command to quit after response
START_EPOCH=$(date +%s.%N)
printf '%s\n/exit\n' "$PROMPT" | taskset -c 6,7 "$LLAMA" \
  -m "$MODEL" \
  --mmproj "$MMPROJ" \
  --image "$IMAGE" \
  -n 128 \
  --chat-template smolvlm \
  --temp 0.0 \
  -t "$THREADS" \
  --simple-io \
  --no-perf \
  --log-disable \
  >> "$LOG" 2>&1
RC=$?
END_EPOCH=$(date +%s.%N)

# Kill RSS monitor
kill $RSS_PID 2>/dev/null || true
sleep 2

ELAPSED=$(python3 -c "print(round($END_EPOCH - $START_EPOCH, 2))")
FREE_AFTER=$(free -m | grep Mem | awk '{print $7}')

# Get peak RSS
PEAK_RSS=0
if [ -f "$RSS_LOG" ]; then
  while IFS=, read -r ts pid rss; do
    [ "$rss" = "rss_kb" ] && continue
    [ -n "$rss" ] && [ "$rss" -gt "$PEAK_RSS" ] 2>/dev/null && PEAK_RSS=$rss
  done < "$RSS_LOG"
fi

echo "---" | tee -a "$LOG"
echo "Exit code: $RC" | tee -a "$LOG"
echo "Elapsed: $ELAPSED sec" | tee -a "$LOG"
echo "Peak RSS: $PEAK_RSS KB ($(python3 -c "print(round($PEAK_RSS/1024,1))") MB)" | tee -a "$LOG"
echo "Free memory after: ${FREE_AFTER} MB" | tee -a "$LOG"
echo "Log: $LOG" | tee -a "$LOG"
echo "RSS log: $RSS_LOG" | tee -a "$LOG"
