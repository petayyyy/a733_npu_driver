# Status

## 2026-06-20

- Read `task.md` and selected the practical first target: G0/G1 hardware
  bring-up and CNN proof of principle.
- Added project structure for docs, board scripts, host scripts, and reports.
- Added SSH launch helper for copying board scripts and starting G0/G1 remotely.
- Verified host preparation script locally. Docker is installed, but the daemon
  is not running and `ubuntu-npu:v2.0.10` is not present locally.
- Verified bash syntax for board scripts with Git Bash.
- Connected to Radxa Cubie A7Z at `192.168.31.76`.
- Gate G0 passed: Debian 11, kernel `5.15.147-21-a733`, 8 cores, thermals
  readable.
- Gate G1 passed: `/dev/vipcore` present; VIPLite 2.0.3.2 loaded
  `yolov8n_6_uint8_a733.nb`; single-image YOLO inference on `dog.jpg` produced
  bicycle/dog/car detections.
- Built standard SDK `examples/vpm_run` on the board from `ZIFENG278/ai-sdk`;
  `operator/v3/network_binary.nb` runs with `cid=0x1000003b` and
  `profile inference time=2807us`.
- Gate G2 passed for SDK LeNet: ACUITY Docker `ubuntu-npu:v2.0.10.1` generated
  uint8 and int16 NBG files, both validated on the A733 through `vpm_run`.
- Gate G2 extension passed for ONNX Inception v1: ACUITY generated uint8 and
  int16 NBG files, both validated on the A733 through `vpm_run`.
  - uint8: `1x3x224x224`, `profile inference time` about `14.36ms`, top-1
    class index `885`, `vpm run ret=0`.
  - int16: `1x3x224x224`, `profile inference time` about `20.85ms`, top-1
    class index `885`, ONNX/non-quantized top-5 preserved, `vpm run ret=0`.
- Phase 3a probe started: `hf-internal-testing/tiny-random-CLIPModel`
  `onnx/vision_model.onnx` was fixed to `1x3x30x30`, converted to int16 NBG,
  and validated on the A733 through `vpm_run`.
  - NBG size: `720,824` bytes.
  - Operators covered include MatMul, Softmax, LayerNorm pattern, Gather, Conv,
    and MLP blocks.
  - Runtime: `profile inference time` about `2.17ms`, output shape `1x64`,
    `vpm run ret=0`.
- Phase 3a real encoder subgate passed: `Xenova/mobileclip_s0`
  `onnx/vision_model.onnx` was fixed to `1x3x256x256`, converted to int16 NBG,
  and validated on the A733 through `vpm_run`.
  - NBG size: `19,376,840` bytes.
  - Output: `1x512` int16 image embedding.
  - Runtime: `profile inference time` about `22.6ms`, `vpm run ret=0`.
  - ACUITY int16 vs NPU int16 output comparison: top-5 indices match, max abs
    diff `0.002471924`, mean abs diff `0.000398278`, cosine `0.999884700`.
- Historical CPU baseline recorded: llama.cpp built on the Radxa board at
  commit `f449e0553708b895adbd94a301431cef691f632d`, and
  `SmolLM2-135M-Instruct-Q4_K_M.gguf` ran through CPU-only GGUF inference.
  This is no longer considered a project gate or deliverable because the active
  requirement is NPU-only LLM/VLM model-layer compute.
  - Model: `134.52M` params, `98.87 MiB` in llama-bench, Q4_K_M.
  - llama-bench, CPU-only: best decode for this model was `56.74 tok/s` at
    2 threads; best prompt throughput was `122.57 tok/s` at 8 threads.
  - llama-simple chat prompt smoke: prompt eval `46.93 tok/s`, decode eval
    `29.92 tok/s`, total `2515.07 ms / 64 tokens`.
- Active requirement correction from user: all LLM/VLM model-layer compute must
  run on the A733 NPU. CPU decode is not acceptable as the target path.
- NPU-only decoder-block subgate passed: generated a deterministic tiny
  fixed-shape transformer decoder block ONNX, exported it through ACUITY
  `ubuntu-npu:v2.0.10.1` to an int16 A733 NBG, and validated it on the Radxa
  through `vpm_run`.
  - NBG size: `85,144` bytes.
  - Input/output: `1x4x8` float16 embedding tensor to `1x4x16` logits tensor.
  - Runtime: `profile inference time` between `59us` and `68us`,
    `vpm run ret=0`.
  - ACUITY int16 vs NPU int16 output comparison: top-5 indices match, max abs
    diff `0.000549316`, mean abs diff `0.000133514`, cosine `0.999999919`.
- NPU-only tiny language-model subgate passed: generated a deterministic
  fixed-shape LM ONNX with int32 token IDs, ONNX `Gather` token embeddings,
  position embeddings, decoder compute, and logits, then exported it through
  ACUITY to an int16 A733 NBG and validated it on the Radxa through `vpm_run`.
  - NBG size: `87,016` bytes.
  - Input/output: `1x4` int32 tokens (`1 5 9 2`) to `1x4x16` logits.
  - Runtime: `profile inference time` between `62us` and `71us`,
    `vpm run ret=0`.
  - ACUITY int16 vs NPU int16 output comparison: top-5 indices match, max abs
    diff `0.000610352`, mean abs diff `0.000153542`, cosine `0.999999929`.
- NPU-only tiny VLM bridge subgate passed: generated a deterministic
  fixed-shape bridge ONNX with a MobileCLIP-S0-style `1x512` image embedding
  input, NPU projector/adapter, int32 token IDs, ONNX `Gather`, image/text
  concat, decoder compute, and logits, then exported it through ACUITY to an
  int16 A733 NBG and validated it on the Radxa through `vpm_run`.
  - NBG size: `94,656` bytes.
  - Input/output: `1x512` image embedding plus `1x4` int32 tokens (`1 5 9 2`)
    to `1x5x16` logits.
  - Runtime: `profile inference time` between `63us` and `72us`,
    `vpm run ret=0`.
  - ACUITY int16 vs NPU int16 output comparison: top-5 indices match, max abs
    diff `0.001159668`, mean abs diff `0.000180054`, cosine `0.999999827`.
- NPU-only fixed-window tiny LM decode-loop subgate passed: ran 8 repeated tiny
  LM NBG forward passes on the Radxa, with CPU limited to writing the next
  `1x4` int32 token window and selecting argmax from NPU logits.
  - Initial prompt: `1 5 9 2`.
  - Generated tiny-token sequence: `1 5 9 2 1 8 4 5 8 4 8 4`.
  - Per-step NPU profile: min `68us`, max `138us`, mean `93.375us`.
  - Every step logged `cid=0x1000003b` and `vpm run ret=0`.

## 2026-06-22

- T0 done: added reusable ACUITY conversion infrastructure.
  - `scripts/host/convert_onnx_to_nbg.sh` runs ONNX import, quantization,
    ACUITY host inference, and A733 NBG export inside Docker image
    `ubuntu-npu:v2.0.10.1` for target
    `VIP9000NANODI_PLUS_PID0X1000003B`.
  - `scripts/host/compare_outputs.py` compares ACUITY host golden output with
    board `output_0.txt` and reports top-5 index match, max/mean abs diff,
    RMSE, and cosine.
  - Fixed a reproducibility issue in generated ACUITY input metadata for `.npy`
    tensor datasets: verified `tiny_lm_gather` token IDs stay `1 5 9 2`
    instead of being reversed as image channels.
- T0 verified on hardware: regenerated `tiny_lm_gather` int16 package with one
  host command, uploaded it to the Radxa board, and ran it through
  `/home/radxa/ai-sdk/examples/vpm_run/vpm_run`.
  - ACUITY export ended with `Error(0),Warning(0)`.
  - Package path: `work/model-packages/tiny_lm_gather/int16/`.
  - Board path: `/home/radxa/a733_npu_driver/models/tiny_lm_gather_t0_int16`.
  - Board run logged VIPLite `2.0.3.2-AW-2024-08-30`, `cid=0x1000003b`, and
    `vpm run ret=0`.
  - `compare_outputs.py` result vs board `output_0.txt`: top-5 index match
    `yes`, max abs diff `0.000610352`, mean abs diff `0.000153542`, RMSE
    `0.000204006`, cosine `0.999999929`.
- T0 report written: `reports/t0-acuity-flow.md`.

- Task T1 passed on the Radxa Cubie A7Z: added a persistent VIPLite C runner
  for the existing tiny LM NBG.
  - Source/build/run helpers:
    `scripts/board/npu_lm_runner.c`,
    `scripts/board/build-npu-lm-runner.sh`,
    `scripts/board/run-npu-lm-runner.sh`.
  - Verified on board with VIPLite `2.0.3.2-AW-2024-08-30` and
    `cid=0x1000003b`.
  - Verified the NBG is loaded/prepared once:
    `create_network_us=457`, `prepare_network_us=218`, `nbg_loaded_once=1`.
  - Verified generated sequence matches the prior reload loop:
    `1 5 9 2 1 8 4 5 8 4 8 4`.
  - Verified persistent mean per-token wall time: `146.375us`; mean NPU
    profile time: `61.375us`; mean runner throughput: `6831.768 tok/s`.
  - Verified old reload-loop baseline on the same board/model/prompt:
    same token sequence, mean `vpm_run` create+prepare+read+run component sum
    `1,019.875us/token`, external shell-loop wall `1,236,942.860us/token`.
  - Result: persistent runner is about `6.97x` faster than the old
    SDK-visible per-token reload component sum, and much faster than the full
    shell loop that also included process launch, Python/file I/O, and logging.
- Report added: `reports/t1-persistent-runner.md`.

- Task T2 passed on the Radxa Cubie A7Z: added an architecturally faithful
  tiny fixed-shape decoder LM probe for the real small-model operator set.
  - Verified generator: `scripts/host/make_tiny_faithful_block_onnx.py`.
  - Verified model shape: `1x16` int32 token IDs to `1x16x256` logits,
    `dim=64`, `2` layers, `4` attention heads, `2` KV heads, `W=16`.
  - Verified ONNX ops include RMSNorm components (`ReduceMean`, `Sqrt`,
    `Reciprocal`), RoPE (`Slice`, `Neg`, `Concat`), GQA repeat (`Reshape`,
    `Tile`), batched attention `MatMul`/`Softmax`, SwiGLU `Sigmoid`/`Mul`,
    token `Gather`, and logits `MatMul`.
  - Verified ACUITY int16 export completed for target
    `VIP9000NANODI_PLUS_PID0X1000003B`; final export ended with
    `Error(0),Warning(0)`.
  - Verified NBG package path:
    `work/model-packages/tiny_faithful_block/int16/`; `network_binary.nb`
    size `409,136` bytes.
  - Verified board path:
    `/home/radxa/a733_npu_driver/models/tiny_faithful_block_t2_int16`.
  - Verified board run logged VIPLite `2.0.3.2-AW-2024-08-30`,
    `cid=0x1000003b`, output `dfp=13`, and `vpm run ret=0`.
  - Verified board profile inference times for three loops:
    `186us`, `251us`, `245us`.
  - Verified `compare_outputs.py` result vs board `output_0.txt`: length
    `4096`, max abs diff `0.073730469`, mean abs diff `0.003069133`, RMSE
    `0.006297570`, cosine `0.999967503`.
  - Verified no model-op fallback or unsupported-op blocker appeared in the
    host or board logs.
- Report added: `reports/t2-faithful-block.md`.

- Task T3 passed on the Radxa Cubie A7Z: added logits slicing and validated a
  sliced-logits `pcq` int8 package for the faithful tiny decoder block.
  - Updated generator: `scripts/host/make_tiny_faithful_block_onnx.py` now
    supports `--logits full|last`, `--seed`, and `--tokens`.
  - Updated comparison helper: `scripts/host/compare_outputs.py` now supports
    `--golden-tail` and `--board-tail` for full-logits tail comparisons.
  - Verified sliced graph shape: full final hidden `1x16x64` is sliced to
    `1x1x64` before the final logits `MatMul`, producing `1x1x256` logits.
  - Verified conversion packages:
    `work/model-packages/tiny_faithful_block_t3_tokensA_full/int16/` and
    `work/model-packages/tiny_faithful_block_t3_tokensA_last_logits/pcq/`.
  - Verified board paths:
    `/home/radxa/a733_npu_driver/models/tiny_faithful_block_t3_tokensA_full_int16`
    and
    `/home/radxa/a733_npu_driver/models/tiny_faithful_block_t3_tokensA_last_pcq`.
  - Verified board runs logged VIPLite `2.0.3.2-AW-2024-08-30`,
    `cid=0x1000003b`, and `vpm run ret=0` for both int16 and pcq packages.
  - Verified last-position argmax is unchanged: full int16 last-position local
    vocab argmax `250`; sliced pcq argmax `250`.
  - Verified board profile times:
    full int16 `182us`, `181us`, `211us`, `217us`, `212us`;
    sliced pcq `160us`, `150us`, `150us`, `151us`, `150us`.
  - Verified measured profile speedup: mean `200.6us` to `152.2us`,
    `1.318x` faster, `24.13%` lower latency.
  - Verified `vpm_run` reported `memory pool size=0byte` for both tiny graphs;
    NBG size dropped from `409,136` bytes to `285,440` bytes.
- Report added: `reports/t3-slice-int8.md`.

- Task T4 started: added the real-model fixed-window ONNX generator and built
  the first full-depth SmolLM2-135M-Instruct graph at `W=32`.
  - Added `scripts/host/make_real_llm_onnx.py`.
  - Verified HF source files under
    `work/models/smollm2-135m-instruct/`: `config.json`,
    `model.safetensors`, `tokenizer.json`, `tokenizer_config.json`, and
    `generation_config.json`.
  - Verified SmolLM2 config: `30` layers, `hidden_size=576`,
    `intermediate_size=1536`, `9` attention heads, `3` KV heads,
    `head_dim=64`, `vocab_size=49152`, `rope_theta=100000`, tied
    embeddings/logits.
  - Verified generated ONNX:
    `work/generated/smollm2_135m_w32/real_llm.onnx`, size
    `651,500,555` bytes.
  - Verified graph output is sliced last-token logits with shape
    `1x1x49152`.
  - Verified ACUITY `pcq` conversion completed for the full W=32 graph.
    - ONNX import: `SUCCESS`.
    - Quantization: `Error(0),Warning(61)`.
    - Final NBG export: `Error(0),Warning(0)`.
    - Package path: `work/model-packages/smollm2_135m_w32/pcq/`.
    - `network_binary.nb` size: `153,990,896` bytes.
    - NBG metadata: int32 input `1x32`, int8 asymmetric affine output
      `1x1x49152`, scale `0.1845247447490692`, zero point `-55`.
    - ACUITY export simulator timing: create network `21.508s`, verify graph
      `44.701s`, one run `7.74266s`.
    - Verified no `unsupported`, `not support`, or `fallback` blocker appeared
      in the conversion logs.
  - Next: upload the W=32 `pcq` package to the Radxa and run it with the T1
    persistent runner.
  - Verified original one-sample `pcq` package runs on the A733 through the T1
    persistent runner, but fails coherence:
    `... assistant the the the  the the ** ...`; mean wall `34.243ms/token`,
    mean NPU profile `25.632ms/token`, `29.203 tok/s`.
  - Added a 12-window representative calibration dataset and rebuilt `pcq`.
    Verified package path:
    `work/model-packages/smollm2_135m_w32_calib/pcq/`; NBG size
    `153,984,304` bytes; final export `Error(0),Warning(0)`.
  - Verified calibrated `pcq` package runs on the A733 through the persistent
    runner, but still fails coherence:
    `... assistant the  the$ interspers strugg ...`; mean wall
    `30.187ms/token`, mean NPU profile `25.541ms/token`, `33.127 tok/s`.
  - Verified calibrated `pcq` exact-sample mismatch:
    - CPU FP fixed-window oracle first token for `The capital of France is`:
      token `253` (`" a"`).
    - ACUITY host `pcq` top-1: token `37353`.
    - A733 NPU `pcq` top-1: token `2581`.
    - Board-vs-host cosine for that sample: `0.992959037`, but top-5 index
      match `no`.
  - Added an int16 correctness-control export for the same W=32 graph.
    - Package path: `work/model-packages/smollm2_135m_w32_int16/int16/`.
    - `network_binary.nb` size: `280,882,632` bytes.
    - Final NBG export: `Error(0),Warning(0)`.
    - Output metadata: int16 dynamic fixed point, `fl=10`, shape
      `1x1x49152`.
  - Verified int16 SmolLM2 W=32 runs on the A733 through the T1 persistent
    runner and produces coherent text:
    `The capital of France is Paris, located in the northern part of the country.`
  - Verified CPU oracle for the same prompt starts:
    `The capital of France is Paris. Paris is a city located in the northern part`.
    The first six generated tokens match exactly:
    `504 3575 282 4649 314 7042` (`The capital of France is Paris`).
  - Verified int16 benchmark with RSS sampler:
    create network `296.038ms`, prepare `7.281ms`, first-step wall
    `46.046ms`, first-step NPU profile `41.052ms`, mean wall
    `46.905ms/token`, mean NPU profile `41.883ms/token`, `21.320 tok/s`,
    peak RSS `278,176 KB`.
  - Verified usable context window is currently fixed `W=32`; the graph
    recomputes the full window each decode step and does not use a KV cache.
  - Result: SmolLM2-135M-Instruct passes the NPU-only coherent-text gate with
    `int16`; the requested `pcq` int8 path is a precise quality blocker, not an
    op-support or NBG-size blocker.
  - Verified SmolLM2-135M-Instruct `W=64` int16 build/conversion/run:
    - ONNX: `work/generated/smollm2_135m_w64/real_llm.onnx`, size
      `651,529,233` bytes.
    - Package path: `work/model-packages/smollm2_135m_w64_int16/int16/`.
    - `network_binary.nb` size: `282,310,408` bytes.
    - Final NBG export: `Error(0),Warning(0)`.
    - Board path: `/home/radxa/a733_npu_driver/models/smollm2_135m_w64_int16`.
    - Runtime metadata: int32 input `1x64`, int16 output `1x1x49152`,
      `dfp=10`, `memory_pool_bytes=345088`.
    - CPU oracle output:
      `The capital of France is Paris. It is a city that has a rich history`.
    - A733 NPU output:
      `The capital of France is Paris, a city that is known for its rich history`.
    - Benchmark with RSS sampler: create network `770.561ms`, prepare
      `27.492ms`, first-step wall `71.560ms`, first-step NPU profile
      `64.861ms`, mean wall `69.656ms/token`, mean NPU profile
      `64.892ms/token`, `14.356 tok/s`, peak RSS `280,904 KB`.
  - Verified usable context window on the working int16 path is now `W=64`.
- Reports/scripts added for T4:
  - `reports/t4-real-model.md`
  - `scripts/host/make_real_llm_onnx.py`
  - `scripts/host/smollm2_tokenizer.py`
  - `scripts/host/smollm2_numpy_reference.py`
  - `scripts/host/make_smollm2_calibration.py`
  - `scripts/board/run-npu-lm-runner-rss.sh`
- Task T5 started: investigated ACUITY hybrid/weight-only style `pcq` as the
  first int8 quality-fix attempt for SmolLM2-135M W=32.
  - Verified `pegasus quantize --help` in `ubuntu-npu:v2.0.10.1` exposes
    `--hybrid`.
  - Updated `scripts/host/convert_onnx_to_nbg.sh` with a `--hybrid` flag.
  - Verified direct hybrid quantize without an existing `.quantize` table fails
    before inference/export with:
    `quantize file 'smollm2_135m_w32_hybrid_pcq_pcq.quantize' does not exist`.
    Logs:
    `logs/host/t5-smollm2-w32-hybrid-pcq-convert.log` and
    `logs/host/t5-smollm2-w32-hybrid-pcq-convert.err.log`.
  - Updated the hybrid flow to seed the normal `pcq` quantize table first, then
    run `pegasus_quantize_hybird.sh`.
  - Verified seeded run imported the SmolLM2 W=32 graph and reached
    `End quantization...` / `Dump net quantize tensor table`, but the quantize
    table remained `0` bytes while the T5 Docker container was still CPU-active.
  - Stopped only the T5 Docker container to avoid interfering with the
    separately running Qwen2.5-0.5B conversion container from another chat.
    Verified Qwen container remained running.
  - No T5 hybrid NBG package or board run yet; this is paused due to parallel
    Qwen work, not a hybrid quality result.
- Report added: `reports/t5-quant.md`.
- Task T4 Qwen continuation resumed per user request:
  - Downloaded Qwen2.5-0.5B-Instruct model files under
    `work/models/qwen25-0.5b-instruct/`; `model.safetensors` size is
    `988,097,824` bytes.
  - Updated `scripts/host/make_real_llm_onnx.py` for Qwen q/k/v projection
    biases and tied `lm_head` via `Transpose(token_embed)` instead of a
    duplicated embedding initializer.
  - Added Qwen host helpers:
    `scripts/host/qwen2_tokenizer.py` and
    `scripts/host/make_qwen2_calibration.py`.
  - Verified one-layer diagnostic Qwen W=32 ONNX generation:
    `work/generated/qwen25_05b_w32_layer1/real_llm.onnx`, size
    `604,219,825` bytes.
  - Verified full Qwen2.5-0.5B-Instruct W=32 ONNX generation:
    `work/generated/qwen25_05b_w32/real_llm.onnx`, size
    `1,976,297,294` bytes; 24 layers, hidden size 896, 14 attention heads,
    2 KV heads, vocab size 151,936.
  - Created Qwen W=32 calibration dataset with 12 token windows:
    `work/generated/qwen25_05b_w32_calib/dataset.txt`.
  - Started ACUITY `pcq` conversion in Docker container `tender_buck`; ONNX
    import succeeded and quantization reached `End quantization`, but the
    active `pegasus.py quantize` process stayed at about 100% CPU and 5.0 GiB
    RSS with unchanged IO counters and `qwen25_05b_w32_pcq.quantize` still
    0 bytes. Stopped `tender_buck` as a stuck full-Qwen `pcq` conversion; no
    full Qwen `pcq` NBG was exported.
  - Verified Qwen-shaped W=32 one-layer `pcq` diagnostic export:
    `work/model-packages/qwen25_05b_w32_layer1/pcq/network_binary.nb`, size
    `274,904,704` bytes, final export `Error(0),Warning(0)`, output
    `1x1x151936` int8 asymmetric affine.
  - Verified full Qwen2.5-0.5B-Instruct W=32 `int16` control export:
    `work/model-packages/qwen25_05b_w32_int16/int16/network_binary.nb`, size
    `1,064,540,800` bytes, final export `Error(0),Warning(0)`, output
    `1x1x151936` int16 dynamic fixed point with `fl=11`.
  - Board upload/run is currently blocked by host-to-board network access from
    this environment: Paramiko reports `WinError 10013`, `ping 192.168.31.76`
    reports `General failure`, and OpenSSH reports `Permission denied` while
    connecting to port 22.
  - Network access later recovered; verified Radxa at `192.168.31.76`, host
    `radxa-cubie-a7z`, `/home/radxa` had `23G` free before upload.
  - Uploaded full Qwen W=32 `int16` package to
    `/home/radxa/a733_npu_driver/models/qwen25_05b_w32_int16`.
  - Verified full Qwen W=32 `int16` board smoke is blocked by board RAM:
    runner exited `137`, `run.log` stayed empty, peak RSS reached
    `641,340 KB`, and board memory after the kill was `959Mi` total with
    `641Mi` available. The NBG is `1,064,540,800` bytes, so it does not fit
    this 1GiB board configuration.
  - Uploaded and ran Qwen W=32 one-layer `pcq` diagnostic package on A733 NPU:
    `/home/radxa/a733_npu_driver/models/qwen25_05b_w32_layer1_pcq`.
    - `network_binary.nb` size: `274,904,704` bytes.
    - Runtime: VIPLite `2.0.3.2-AW-2024-08-30`, `cid=0x1000003b`,
      int32 input `1x32`, int8 asymmetric output `1x1x151936`,
      `memory_pool_bytes=214016`, `nbg_loaded_once=1`, `status=0`.
    - Timing: create network `583.531ms`, prepare `0.744ms`, first-step wall
      `46.465ms`, first-step NPU profile `19.217ms`, mean wall
      `45.572ms/token`, mean NPU profile `19.197ms/token`, `21.943 tok/s`,
      peak RSS `270,220 KB`.
    - Generated diagnostic layer1 tokens: `56446 56446 56446 732`
      (`forgettableforgettableforgettable im`). This is a Qwen-shaped NPU
      execution control, not a coherence pass, because it has only one decoder
      layer.

## Next Gate

T4 Qwen resume point: full Qwen `int16` exports on host but is too large for the
1GiB Radxa board; full Qwen `pcq` is the viable memory target but is currently
blocked in ACUITY quantize-table serialization/rebuild. Next step is to unblock
or bisect the full Qwen `pcq` conversion; the one-layer Qwen `pcq` NBG is the
passing hardware diagnostic control.
