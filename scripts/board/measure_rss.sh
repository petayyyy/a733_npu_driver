#!/bin/bash
# Measure peak RSS of llama-cli VLM run

MODEL=/home/orangepi/a733_npu_driver/models/vlm/SmolVLM-256M-Instruct-Q8_0.gguf
MMPROJ=/home/orangepi/a733_npu_driver/models/vlm/mmproj-SmolVLM-256M-Instruct-Q8_0.gguf
IMAGE=/home/orangepi/a733_npu_driver/test_images/dog.jpg
LLAMA=/home/orangepi/llama.cpp/build/bin/llama-cli

pkill -9 llama-cli 2>/dev/null || true
sleep 1

echo "Free before: $(free -m | grep Mem | awk '{print $7}') MB"

# Launch llama-cli in background with proper PID tracking
printf "/image %s\nDescribe this image.\n/exit\n" "$IMAGE" | \
  taskset -c 6,7 "$LLAMA" \
    -m "$MODEL" --mmproj "$MMPROJ" \
    -n 128 --temp 0.0 -t 2 \
    --simple-io --no-perf --log-disable > /tmp/rss_out.txt 2>&1 &
LLAMA_PID=$!

# Wait for process to start
sleep 0.5

# Monitor RSS
PEAK=0
while kill -0 $LLAMA_PID 2>/dev/null; do
  R=$(awk '/^VmRSS:/{print $2}' /proc/$LLAMA_PID/status 2>/dev/null)
  R=${R:-0}
  if [ "$R" -gt "$PEAK" ] 2>/dev/null; then
    PEAK=$R
  fi
  sleep 0.2
done
wait $LLAMA_PID
RC=$?

echo "Exit code: $RC"
echo "Peak RSS: $PEAK KB"
echo "Peak RSS: $(python3 -c "print(round($PEAK/1024,1))") MB"
echo "Free after: $(free -m | grep Mem | awk '{print $7}') MB"
cat /tmp/rss_out.txt
