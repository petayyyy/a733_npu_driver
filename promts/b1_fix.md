TASK B1b-matrix-fix: B1's int16 host gate is BROKEN as a board-run filter -- it failed
SmolLM2-135M W=32 at host cosine 0.778 (some configs NEGATIVE), yet that exact config is PROVEN
coherent on the Orange Pi (B2 chatted with it at ~21 tok/s). ACUITY pegasus inference is NOT a
faithful predictor of on-board int16 coherence. Redo the matrix gating on ON-BOARD coherence.

Keep from B1 (don't redo): all 12 configs pass the FP ORACLE gate (builder correct at every
size). SmolLM2-1.7B FAILS NBG export at every window (gen_nbg segfault / 0-byte NBG, 6.85 GB
ONNX) -> 1.7B is export-blocked; EXCLUDE it.

DO (Orange Pi Zero 3W, 192.168.31.225, via the B2/T10b persistent runner, one NPU job at a time):
1. For every config whose int16 NBG already exports -- SmolLM2-135M at W=32/64/128/256 and
   SmolLM2-360M at W=32/64/128/256 -- deploy and RUN it on the board.
2. Gate on REAL coherence: generate from the fixed prompt, compare the board's generated token
   stream + decoded text to the FP-oracle continuation; judge coherent y/n on board. Do NOT use
   pegasus-inference cosine.
3. Record decode tok/s, prefill tok/s, first-token ms, peak RSS, NBG MB for each. For configs
   incoherent ON BOARD, record that as the real datapoint (where int16 genuinely breaks).

DELIVERABLE: reports/b1b-benchmark-matrix.md, ONE table (135M/360M x W=32/64/128/256), all
numbers verified on board, with notes on where int16 stops being coherent on board and where
tok/s drops below ~3. Note the corrected methodology (FP-oracle for builder; on-board for the
real filter).

SUCCESS GATE: measured on-board tok/s + coherence for every exporting 135M/360M window; 1.7B
documented export-blocked. Committed.

START FROM: the exporting B1 NBGs (135M + 360M, all windows); the B2/T10b runner; the FP-oracle
continuations from B1.