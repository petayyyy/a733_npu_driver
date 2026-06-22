## T5

TASK T5-quant (int8 quality fix): SmolLM2-135M int16 already runs coherently on the
NPU (verified). The pcq int8 path runs on NPU but produces garbage, and ACUITY's OWN
host pcq output is already wrong vs the FP oracle — so this is a quantization-method
problem, not an NPU/op/size problem. Recover int8 coherence so the model fits RAM for
scaling to Qwen2.5-0.5B.

DO, in order, stopping at the first that yields coherent text:
1. w8a16: configure ACUITY to quantize WEIGHTS to int8 but keep ACTIVATIONS in int16
   (weight-only / hybrid quant if available). Activation outliers are the likely cause.
2. Mixed precision: keep token embedding, final RMSNorm, and lm_head/logits in int16;
   quantize only the 30 transformer linear layers to int8.
3. Combine 1+2.
For each attempt: re-run on the board via the T1 persistent runner, compare first ~6
generated tokens to scripts/host/smollm2_numpy_reference.py FP oracle for
"The capital of France is", and record tok/s + RSS + NBG size.

SUCCESS GATE: an int8/hybrid SmolLM2-135M whose first 6 tokens match the FP oracle and
text is coherent, with NBG and RSS smaller than the int16 build. If all three fail with
the exact divergence logged, escalate to T6 vendor with the reproducer.

START FROM: work/model-packages/smollm2_135m_w32_int16 (the working int16 build) and the
failing pcq packages; scripts/host/convert_onnx_to_nbg.sh quant options.