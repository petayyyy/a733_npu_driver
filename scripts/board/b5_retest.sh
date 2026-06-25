#!/bin/bash
taskset -c 6,7 /home/orangepi/llama.cpp/build/bin/llama-completion \
  -m /home/orangepi/a733_npu_driver/models/qwen2.5-1.5b-instruct-q8_0.gguf \
  -p "In one concise technical note, explain why CPU utilization must be measured when benchmarking a small language model on an embedded board with heterogeneous CPU cores such as Cortex-A55 and Cortex-A76." \
  -c 2048 -n 128 -t 2 --temp 0 --seed 42 -ngl 0 \
  --no-warmup --no-display-prompt -no-cnv --simple-io \
  > /home/orangepi/a733_npu_driver/logs/b5-sweep/retest.log 2>&1
echo "EXIT=$?" >> /home/orangepi/a733_npu_driver/logs/b5-sweep/retest.log
