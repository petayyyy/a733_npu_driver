#!/bin/bash
# Simple VLM benchmark - runs model, saves output
MODEL_DIR=/home/orangepi/a733_npu_driver/models/vlm
LLAMA=/home/orangepi/llama.cpp/build/bin/llama-cli
IMAGE_DIR=/home/orangepi/a733_npu_driver/test_images
LOG_DIR=/home/orangepi/a733_npu_driver/logs/v1-vlm
mkdir -p "$LOG_DIR"

MODEL="$MODEL_DIR/SmolVLM-256M-Instruct-Q8_0.gguf"
MMPROJ="$MODEL_DIR/mmproj-SmolVLM-256M-Instruct-Q8_0.gguf"

echo "=== Starting V1 benchmarks at $(date) ==="

# Run on all 3 images
for IMG in dog.jpg cat.jpg test-1.jpeg; do
  for PROMPT in "Describe this image." "What animal is in this image?"; do
    LOG="$LOG_DIR/smolvlm256_${IMG%.*}_$(echo $PROMPT | tr ' ' '_' | tr -d '?.')_$(date +%H%M%S).log"
    echo "--- Running: $IMG / $PROMPT ---" | tee "$LOG"
    
    # Capture RSS: launch in background, measure RSS frequently
    (
      printf "/image $IMAGE_DIR/$IMG\n$PROMPT\n/exit\n" | \
        taskset -c 6,7 "$LLAMA" \
        -m "$MODEL" --mmproj "$MMPROJ" \
        -n 128 --temp 0.0 -t 2 \
        --simple-io --no-perf --log-disable
    ) >> "$LOG" 2>&1 &
    PID=$!
    
    # Monitor RSS every 0.2s
    PEAK=0
    while kill -0 $PID 2>/dev/null; do
      R=$(awk '/^VmRSS:/{print $2}' /proc/$PID/status 2>/dev/null || echo 0)
      if [ "$R" -gt "$PEAK" ] 2>/dev/null; then PEAK=$R; fi
      sleep 0.2
    done
    wait $PID
    
    echo "Peak RSS: ${PEAK} KB" | tee -a "$LOG"
    echo "Done: $LOG"
  done
done

echo "=== All benchmarks complete ==="
