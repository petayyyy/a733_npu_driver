TASK V2c-vlm-npu-e2e-closeout: V2b broke the toolchain wall — SmolVLM-256M's SigLIP vision encoder
(Conv→Reshape+MatMul rewrite) now converts to a 271 MB int16 NBG and runs on the Orange Pi NPU
(vpm run ret=0, 1x64x576 output, cosine 1.0 vs PyTorch at the ONNX level). But it is "PARTIALLY
RESOLVED": end-to-end accuracy is NOT yet validated, and the NBG was calibrated on RANDOM NOISE
(randn ~[-5,5]) instead of real SigLIP-normalized images (~[-1,1]), which likely degrades int16
quality on real photos. Close this out so the hybrid (NPU vision + CPU SmolVLM LLM) is a PROVEN,
not partial, result.

=== STEP 1: rebuild the NBG with REAL-IMAGE calibration ===
1. Build a calibration set of 8-16 REAL images preprocessed exactly as Idefics3/SigLIP does:
   resize to 512x512, normalize with mean=0.5,std=0.5 (range ~[-1,1]). Use varied content (include
   the V1 test images: dog, cat, the moon-landing newspaper, + a few more diverse photos).
2. Re-run the ACUITY int16 conversion of the V2b Conv→MatMul encoder ONNX with THIS calibration
   set (not noise). Confirm Error(0)/Warning(0), and check the input quantization now uses the real
   image range (fl should reflect ~[-1,1], not the noise ~[-5,5]).
3. HOST GATE: compare the int16 NBG host-sim embedding (1x64x576) vs the PyTorch encoder output for
   2-3 held-out real images. GATE: cosine > 0.95. If poor (int16-outlier degradation like the LLMs
   had), record it — the encoder may need int16-sensitive handling; report the per-image cosine.

=== STEP 2: wire the NPU embedding into the CPU SmolVLM decoder ===
1. Write the glue: image → CPU preprocess (resize/normalize) → quantize to the NBG's int16 DFP
   input format (.dat) → NPU vision NBG → 1x64x576 embedding → dequantize → feed into the llama.cpp
   SmolVLM-256M decoder IN PLACE of its mmproj-produced image embedding. Keep tokenize/KV-cache/
   sampling/LLM on CPU.
2. Confirm the embedding tensor layout/scale matches what llama.cpp's SmolVLM expects (this is the
   fiddly part — the 64 image tokens × 576 dim must align with the decoder's image-token slots).

=== STEP 3: end-to-end accuracy validation on the Orange Pi (the actual gate) ===
1. Run the full hybrid on the 3 V1 test images (dog, cat, moon-landing newspaper) with a fixed
   question ("Describe this image.").
2. Compare answers to V1's CPU-only SmolVLM answers (V1 read the newspaper correctly, identified
   dog/cat). GATE: NPU-vision answers are ACCURATE and match V1's quality (same objects identified,
   newspaper still readable). If answers are wrong/degraded, diagnose whether it's the int16
   calibration (Step 1 cosine) or the embedding injection (Step 2 layout/scale), and report which.
3. Measure: NPU vision latency, end-to-end answer latency, decode tok/s, peak RSS, and CPU freed
   (A76 cores not running the encoder vs V1's 2 fully loaded).

=== IF IT WORKS ===
The hybrid VLM with NPU vision offload is a proven deliverable: SmolVLM image chat where the vision
encoder runs on the NPU (freeing the A76 cores) and the LLM runs on CPU.

=== IF IT FAILS (after honest diagnosis) ===
Record exactly where it broke (int16 calibration cosine, or embedding layout/scale). V1 CPU-only
(SmolVLM-256M Q8_0, 52.6 tok/s) stays the deliverable. Note the NPU is ~6 s/image vs ~1-2 s CPU
anyway, so CPU-only remains reasonable; document the offload trade honestly.

=== DOCS UPDATE (MANDATORY, after the run, either outcome) ===
Update the docs to the FINAL outcome, keeping them self-consistent:
1. docs/blockers.md: change "Blocker 4 — SmolVLM SigLIP (PARTIALLY RESOLVED)" to its final state —
   RESOLVED (move to works) if e2e accurate, or PARTIALLY RESOLVED with the exact remaining gap if
   not.
2. docs/RESULTS.md: add the final VLM-on-NPU hybrid row (real-image cosine, e2e accuracy, vision
   latency, decode tok/s, RSS, CPU freed) — or note CPU-only remains the VLM result.
3. docs/configurations.md: if it works, update "VLM vision offload (SmolVLM)" to a recommended
   hybrid config with the measured e2e numbers; if not, keep SmolVLM-256M CPU-only as the
   recommended image-chat config and note the offload status.
4. docs/05-run-vlm-npu.md: add the end-to-end hybrid run instructions (or the calibration/injection
   procedure) so a reader can reproduce it.
5. docs/roadmap.md + README "what works/doesn't" box + Project Summary: update so the final V2
   outcome is reflected with zero contradictions. Mark the research phase fully complete.
6. If failed, add/refresh the V2 vendor packet in docs/vendor-tickets.md.

DELIVERABLE: reports/v2c-vlm-npu-e2e.md with Step 1 real-image cosine, Step 2 injection notes,
Step 3 on-board accuracy vs V1 + latencies + CPU freed; AND all docs updated to the final outcome.
Commit report + doc updates together. Mark each result verified.

SUCCESS GATE: SmolVLM-256M image chat runs end-to-end on the Orange Pi with the vision encoder on
the NPU, answers accurate (match V1 quality), CPU offload quantified — OR a precisely diagnosed
remaining gap with V1 CPU-only confirmed as the deliverable; docs fully reconciled either way.

DO NOT: re-quantize with noise calibration; reintroduce the original Conv (use the V2b Conv→MatMul
ONNX); break the working V1 CPU path.

START FROM: the V2b Conv→MatMul encoder ONNX + 271 MB NBG (work/generated/smolvlm_256m_vision_
encoder/); convert_onnx_to_nbg.sh (int16, real-image calibration); the V1 CPU SmolVLM llama.cpp
setup + its 3 test images; the T10b/B3 runner + VIPLite layout on the Orange Pi (192.168.31.225);
the docs/ structure DOC3 finalized.