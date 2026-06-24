# B1b Benchmark Matrix

Date: 2026-06-24

Status: complete. B1's FP builder gate is retained, but the broken ACUITY
`pegasus inference` host-cosine gate is no longer used as a board-run filter.
The real filter here is Orange Pi Zero 3W on-board decoded coherence from the
exported int16 NBGs.

Prompt and windowing are the B1 prompt and method: SmolLM2 chat rendering of
`The capital of France is`, rightmost tokens for W=32, and zero left-padding
for W=64/128/256. Each board row generated 16 greedy tokens through the
persistent VIPLite protocol runner, one NPU job at a time. `prefill tok/s` is
reported as `W / first-token wall time` because this fixed-window graph has no
separate KV-cache prefill path.

| Model | W | Board coherent? | FP token prefix match | FP oracle continuation | Board continuation | Decode tok/s | Prefill tok/s | First-token ms | Peak RSS MiB | NBG MB |
| --- | ---: | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| SmolLM2-135M-Instruct | 32 | yes | `16/16` | The capital of France is Paris.<im_end> / <im_end><im_start>s... | The capital of France is Paris. / system | 20.743 | 671.6 | 47.6 | 271.8 | 280.9 |
| SmolLM2-135M-Instruct | 64 | yes (weak) | `0/16` | <im_end><im_start>system / <im_end><im_start>system / <im_end... | The capital of France is located in France. / user / What | 13.980 | 887.4 | 72.1 | 274.2 | 282.3 |
| SmolLM2-135M-Instruct | 128 | no | `3/16` | / The answer to the "How many different ways can you can fin... | The answer is: / user / / The answer is 1 | 6.035 | 773.0 | 165.6 | 282.2 | 286.9 |
| SmolLM2-135M-Instruct | 256 | no | `3/16` | / / / / / / / / / / / / / / / / | The reason for the following: / / / The answer is: | 1.163 | 297.7 | 859.9 | 375.1 | 337.1 |
| SmolLM2-360M-Instruct | 32 | yes | `0/16` | I'm sorry for the confusion, but as an AI, I don't have | The capital of France is Paris. / system | 8.448 | 280.3 | 114.2 | 646.1 | 672.7 |
| SmolLM2-360M-Instruct | 64 | yes | `7/16` | The capital of France is Paris. Paris is a city known for its... | The capital of France is Paris. / user / What is the capital | 4.903 | 301.5 | 212.3 | 648.6 | 674.5 |
| SmolLM2-360M-Instruct | 128 | no | `4/16` | / What is the capital of France?<im_end> / <im_start>user / ... | What is the most important part of a well, and in which of th... | 1.993 | 255.1 | 501.8 | 681.3 | 693.3 |
| SmolLM2-360M-Instruct | 256 | no | `1/16` | / The French Revolution was a pivotal moment in the 1780s | Question / The Heliologist / / Hmm, I have | 1.202 | 306.9 | 834.2 | 710.5 | 708.5 |

Notes:

- Verified on the Orange Pi at `192.168.31.225`: every board run logged
  VIPLite `2.0.3.2-AW-2024-08-30`, `cid=0x1000003b`, `network_core_count=1`,
  `protocol=stdio`, and `nbg_loaded_once=1`.
- Verified preflight before and after the run: no unrelated `npu_lm_runner`,
  `vpm_run`, `chat_shell.py`, `monitor_command.py`, `llama`, `cmake`, or
  `ninja` jobs were active, and `/dev/vipcore` had no users.
- Verified board logs are local under `logs/board/b1b/`; FP continuations are
  under `logs/host/b1b-smollm2-*-fp-continuation.json`.
- On-board coherence stops at W=128 for both 135M and 360M. 135M/W64 is
  grammatical but weak, so it is marked coherent with a caveat; 135M/W128 and
  W=256, plus 360M/W128 and W=256, are not coherent continuations of the fixed
  prompt.
- Decode throughput drops below about 3 tok/s at 135M/W256 and at 360M/W128
  and W=256.
- SmolLM2-1.7B remains excluded: B1 already verified all 1.7B windows pass the
  FP builder gate but fail NBG export (`gen_nbg` segfault/0-byte NBG from the
  6.85 GB ONNX external-data graph), so there is no board-runnable NBG.
- Temporary B1b NBG copies were removed from the Orange Pi after logs were
  downloaded, restoring `/` from 95% used to 82% used. Host-side exported NBGs
  remain under `work/model-packages/b1_smollm2_*_int16/int16/`.
