TASK B4b-cpu-utilization: B4 measured Qwen2.5-0.5B CPU tok/s but did NOT record thread
count (-t) or CPU utilization. Re-measure the best CPU config and capture both.

DO (on the Orange Pi Zero 3W, 192.168.31.225, ROS2 stopped so the CPU is free):
1. Run Qwen2.5-0.5B Q8_0 at ctx=2k via llama-cli on a fixed prompt generating ~128 tokens,
   for thread counts -t = 2, 4, 6, 8 (2 = A76 pair; 8 = all cores). One run each.
2. For each run, capture:
   - decode tok/s and prefill tok/s (from llama output),
   - average AND peak CPU% over the run, sampled with `pidstat -p <pid> 1` (or `top -b -d 1`)
     for the llama process — report both the per-core% (can exceed 100%) and the normalized
     %-of-8-cores,
   - which cores were actually used (taskset/affinity; check with `taskset -cp <pid>` and
     per-core load via `mpstat -P ALL 1`),
   - peak RSS.
3. Note thermals (`cat /sys/class/thermal/thermal_zone*/temp`) at the end of the longest run.

DELIVERABLE: reports/b4b-cpu-utilization.md with a table (rows = -t 2/4/6/8; columns = decode
tok/s, prefill tok/s, avg CPU%, peak CPU%, cores used, peak RSS, temp), and a one-line
recommendation: the -t that gives best tok/s and how many cores Qwen actually occupies (so you
know how much CPU is consumed while ROS2 is paused). All numbers verified/measured.

SUCCESS GATE: measured tok/s + CPU utilization for each thread count on the board. Committed.

START FROM: the B4 llama.cpp build and Qwen2.5-0.5B Q8_0 GGUF; the Orange Pi at 192.168.31.225.