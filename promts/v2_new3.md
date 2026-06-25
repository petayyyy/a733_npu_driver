TASK V2d-vlm-npu-mmproj-glue: V2c proved the SmolVLM-256M SigLIP vision encoder runs on the NPU
with int16 cosine 0.9914 vs FP32, and a C injector feeds NPU embeddings into llama.cpp via
batch.embd — but the decoder produces a degenerate token because SmolVLM's GGUF is a plain LLaMA
model with no mmproj/mtmd multimodal context (no image-token setup). Close the last gap: make the
hybrid (NPU vision → CPU LLM) produce CORRECT answers. This is software integration, not research.

DO:
1. Use the cleaner approach: load the SmolVLM-256M mmproj GGUF alongside the text model so
   llama.cpp's mtmd layer establishes the full multimodal context (special image tokens, token
   slots), THEN replace ONLY the mmproj's vision-encoder output with the NPU-produced int16
   embedding (1x64x576). The CPU mmproj does the image-token plumbing; the NPU does the heavy
   SigLIP compute. (Fallback if that proves awkward: patch llama-cli to accept an
   --image-embeddings-file and inject after the multimodal context is built.)
2. Confirm the embedding layout/scale alignment: the 64 image tokens x 576 dim from the NPU must
   land in exactly the slots mtmd expects (same order/normalization as mmproj output). Verify by
   comparing, for one image, the NPU embedding to the mmproj's own vision output (cosine) before
   the LLM step.
3. END-TO-END GATE on the Orange Pi (192.168.31.225), the 3 V1 test images (dog, cat, moon-landing
   newspaper), question "Describe this image.": answers must be ACCURATE and match V1 CPU-only
   quality (correct objects; newspaper still readable). If wrong, diagnose whether it's embedding
   layout/scale (step 2) or the multimodal-context wiring (step 1), and report which.
4. Measure: NPU vision latency, end-to-end answer latency, decode tok/s, peak RSS, and A76 cores
   freed vs V1 (which loads 2 cores fully for vision).

ON SUCCESS: the hybrid VLM with NPU vision offload is a PROVEN deliverable — SmolVLM image chat
where vision runs on the NPU (both A76 cores free for ROS2) and the LLM runs on CPU.

DOCS UPDATE (mandatory, either outcome): update docs/blockers.md (move SmolVLM-vision-on-NPU to
RESOLVED if accurate, else keep the precise remaining gap), docs/RESULTS.md (add the proven hybrid
row: e2e accuracy, latencies, tok/s, RSS, CPU freed), docs/configurations.md (if working, make
"image chat with NPU vision offload" a recommended config with measured numbers), docs/05-run-vlm-
npu.md (add the reproducible hybrid run steps), README + roadmap (final state, no contradictions).

DELIVERABLE: reports/v2d-vlm-npu-mmproj-glue.md with the embedding-alignment cosine, the 3-image
e2e accuracy vs V1, latencies, and CPU freed — plus the doc updates committed together. Mark each
result verified.

SUCCESS GATE: SmolVLM-256M image chat runs end-to-end on the Orange Pi with vision on the NPU,
answers accurate (match V1), CPU offload quantified — OR a precisely diagnosed remaining gap with
V1 CPU-only confirmed as the deliverable. Docs reconciled either way.

DO NOT: re-quantize with noise calibration; reintroduce the Conv (use the V2b/V2c Conv→MatMul NBG);
break the working V1 CPU path.

START FROM: scripts/board/inject_embeds.c, scripts/host/run_v2c_e2e.py (V2c bridge); the real-image
NBG at work/model-packages/smolvlm_256m_vision_v2c/int16/ (cosine 0.9914); the V1 CPU SmolVLM
llama.cpp setup + mmproj GGUF + 3 test images; the T10b/B3 runner + VIPLite layout on the Orange Pi.