TASK V2b-smolvlm-vision-npu-retry: V2 got SmolVLM-256M's SigLIP vision encoder to ONNX (cosine
1.0 vs PyTorch) but ACUITY 6.30.22 could not convert it: after a clean NonZero workaround, import
crashes at Conv shape inference -- IndexError in _conv_shape (smart_toolkit.py:1571) on the SigLIP
patch-embedding Conv (kernel=16, stride=16, in=3, out=768, input 1x3x512x512). This is a SPECIFIC
ACUITY bug, NOT a fundamental wall. Goal: get the vision encoder onto the NPU so the hybrid (NPU
vision + CPU SmolVLM LLM) offloads the A76 cores. CPU-only V1 (SmolVLM-256M Q8_0, 52.6 tok/s) is
the fallback deliverable if all attempts fail.

Try in order, cheapest first; STOP at the first that exports to int16 NBG with good quality.

ATTEMPT 1 -- rewrite patch-embedding Conv as Reshape+MatMul (highest promise, cheapest):
A Conv2d with kernel=stride=16, no padding == split the 512x512 image into 32x32=1024 non-
overlapping 16x16x3 patches, flatten each to 768, MatMul by the reshaped conv weight (768x768) +
bias. ACUITY handles Reshape/Transpose/MatMul (the LLM used them); this sidesteps _conv_shape.
1. In export_smolvlm_vision_onnx.py, replace the patch-embed Conv with patchify (Reshape/Transpose
   /Slice -> 1x1024x768) -> MatMul(weight 768x768) -> Add(bias), then the transformer unchanged.
   Keep the NonZero->Constant fix.
2. Verify rewritten ONNX vs PyTorch (onnxruntime cosine > 0.9999 -- patchify must be index-correct).
3. Convert to int16 NBG; validate the encoder output on host vs PyTorch (cosine gate). Watch for
   int16-outlier quality loss like the LLMs had.
   GATE: exports to int16 NBG AND host int16 cosine vs PyTorch > 0.95.

ATTEMPT 2 -- export bare SigLIP, bypassing the Idefics3 wrapper (V2 noted "not done"):
1. Load SigLIP weights from the SmolVLM safetensors; export ONLY the SigLIP tower (patch embed ->
   transformer -> 1x1024x768), fixed shapes, no Idefics3 dynamics; apply Attempt-1's Conv->MatMul
   rewrite if the raw Conv still crashes.
2. Run the Idefics3 connector (1024->64 pool, 768->576) on CPU as a cheap post-step.
3. Convert SigLIP ONNX to int16 NBG; validate 1x1024x768 host cosine.
   GATE: SigLIP NBG exports AND host cosine > 0.95; full pipeline still gives accurate answers.

ATTEMPT 3 -- swap to a VLM with an ACUITY-friendly CNN/ViT encoder (fallback):
1. Identify a small (~0.5-1B) VLM whose vision encoder ACUITY can convert (ResNet/MobileCLIP/
   Inception-class all worked). Note MobileCLIP-S0 (512-dim/1 token) does NOT match SmolVLM's
   576-dim/64-token LLM without a trained adapter -- so this needs a VLM built around a convertible
   encoder. Document candidates and whether any is realistically wireable.
   GATE: a VLM whose encoder exports to int16 NBG and runs end-to-end with a CPU LLM, accurate.

ON SUCCESS (any attempt): wire the full hybrid on the Orange Pi (192.168.31.225): image -> CPU
preprocess -> NPU vision NBG -> connector -> CPU SmolVLM LLM (llama.cpp). Use the V1 test images
(dog, cat, moon-landing newspaper); confirm answers stay ACCURATE vs V1. Measure vision latency on
NPU, end-to-end latency, decode tok/s, peak RSS, CPU freed (A76 no longer running the encoder).

IF ALL THREE FAIL: record the exact ACUITY error per attempt (op, error, log) as a vendor packet;
confirm V1 CPU-only as the final VLM deliverable. A documented Conv->MatMul failure is itself a
finding (the blocker is deeper than the Conv shape bug).

DELIVERABLE (report): reports/v2b-smolvlm-vision-npu-retry.md with, per attempt: ONNX-rewrite
correctness, ACUITY import/export result (+ exact error if failed), host int16 cosine, and on
success the on-board end-to-end accuracy + latencies + CPU offload. Mark each result verified.

=== DOCS UPDATE (MANDATORY -- do this AFTER the experiment, regardless of outcome) ===
DOC3 already reconciled all documentation to the prior state; now update it to reflect THIS V2
result so docs stay consistent:
1. docs/blockers.md: if V2b FAILED, update the SmolVLM-vision-on-NPU entry with the new exact
   error(s) from Attempts 1-3 and mark it confirmed-vendor-gated (note Conv->MatMul was tried). If
   V2b SUCCEEDED, REMOVE it from blockers and move it to the "works" set.
2. docs/RESULTS.md: if SUCCEEDED, add the VLM-on-NPU hybrid row (vision latency on NPU, end-to-end
   latency, decode tok/s, RSS, CPU freed). If FAILED, leave V1 CPU-only as the VLM result and note
   the NPU-offload attempt outcome.
3. docs/configurations.md: if SUCCEEDED, add/update the "image chat with NPU vision offload"
   configuration as the recommended VLM path (freeing A76 cores) with the measured numbers; if
   FAILED, keep SmolVLM-256M CPU-only as the recommended image-chat config.
4. docs/vendor-tickets.md: if FAILED, add the V2b SigLIP/Conv->MatMul vendor packet (exact op,
   error, shapes, ACUITY 6.30.22).
5. docs/roadmap.md and README "what works/doesn't" box: update to reflect the final V2 outcome so
   nothing contradicts. Keep the verified/assumption convention.
Ensure every number in the docs matches this report; fix any link broken by the change. Commit the
report and the doc updates together.

SUCCESS GATE: SmolVLM (or alt VLM) runs end-to-end on the Orange Pi with vision on NPU, accurate,
CPU offload quantified -- OR all three attempts documented as failing with V1 CPU-only as the
deliverable; AND all documentation updated to match the outcome with zero contradictions. Committed.

DO NOT retry: the original SigLIP Conv as-is (crashes _conv_shape); bolting MobileCLIP-S0 onto
SmolVLM's LLM (dimension mismatch).

START FROM: export_smolvlm_vision_onnx.py, remove_nonzero.py, fix_onnx.py; the V2 NonZero-removed
ONNX at work/generated/smolvlm_256m_vision_encoder/; convert_onnx_to_nbg.sh; host-oracle tooling;
the V1 CPU SmolVLM setup; the T10b/B3 runner + VIPLite layout on the Orange Pi; the docs/ structure
DOC3 finalized.