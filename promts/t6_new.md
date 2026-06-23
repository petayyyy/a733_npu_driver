TASK T6-port: The final board is an Orange Pi Zero 3W with 6 GB LPDDR5.
This removes the two Radxa-specific blockers: int16 Qwen no longer has to fit
in about 1 GB RAM, and the broken int8/ACUITY-hybrid path is no longer needed
for the first useful Qwen result.

Goal: run Qwen2.5-0.5B-Instruct in int16, the known-coherent path, on the
Orange Pi Zero 3W NPU.

PART A - Port the working stack to Orange Pi Zero 3W:
1. Flash the Orange Pi Zero 3W official image 1.0.4 from orangepi.org
   (the A733 Zero 3W, not the H618 Zero 3). Confirm boot, kernel 6.1.31,
   8 cores, and `free -h` showing about 6 GB RAM.
2. Verify the NPU stack on this BSP: `/dev/vipcore`, VIPLite userspace
   libraries, and the Orange Pi BSP kernel NPU module. The kernel module must
   come from the Orange Pi BSP, not from the Radxa kernel.
3. Recompile the T1 persistent runner (`scripts/board/npu_lm_runner.c`) on
   Bookworm/glibc 2.36. Do not reuse the Bullseye binary.
4. Smoke-test the port: copy the working SmolLM2-135M W=32 or W=64 int16 NBG
   unchanged and run it through the rebuilt runner. Confirm it still produces
   a coherent "The capital of France is Paris..." answer. This proves the port
   before touching Qwen.

PART B - Qwen2.5-0.5B int16 on the Orange Pi NPU:
1. Build or reuse the fixed-window Qwen2.5-0.5B-Instruct ONNX at W=32 with the
   T3 last-logits slice. The full-Qwen int16 graph already exported on host in
   T4/T5; reuse that path where possible. Watch `vocab_size=151936`, but RAM
   should be sufficient on the 6 GB board.
2. Convert to int16 NBG with `scripts/host/convert_onnx_to_nbg.sh`
   (`--quant int16`, not `pcq`).
3. Upload to the Orange Pi and run with the persistent runner. Verify coherent
   text against an FP oracle, and record tok/s, peak RSS, NBG size, create time,
   first-token latency, and NPU profile time.
4. If W=32 is coherent, try W=64 for more usable context.

SUCCESS GATE: Qwen2.5-0.5B-Instruct produces coherent text with every model
layer running on the Orange Pi Zero 3W NPU in int16, with tok/s and RSS
recorded. Bonus gate: W=64 coherent too.

START FROM: the working SmolLM2 int16 NBG, the T1 runner, the full-Qwen int16
export path from T4/T5, and the Orange Pi Zero 3W 6 GB image.
