TASK V4b-cli-tools: Replace any web UI with simple, robust COMMAND-LINE terminal scripts for
running VLM image-chat and LLM text-chat on the Orange Pi Zero 3W, plus an install/usage guide.
TERMINAL ONLY — no web server, no browser UI. Wrap the EXISTING proven pieces; do not reinvent.

REMOVE/IGNORE any web app: if app/ contains a FastAPI/Gradio/Flask web interface, replace it with
CLI scripts (keep the underlying logic, drop the server/HTML). The deliverables are terminal tools.

=== TOOL 1: VLM image chat (CLI) — app/vlm_chat.py ===
A terminal program: pass an image and ask questions, get answers streamed in the terminal.
- Usage: `python3 vlm_chat.py --image dog.jpg` then type questions at a prompt; or
  `python3 vlm_chat.py --image dog.jpg --question "Describe this image."` for one-shot.
- Backends (flag --backend): 
  * "cpu" (default): SmolVLM-256M Q8_0 via llama.cpp (V1/V3 winner, ~52 tok/s, accurate).
  * "npu": NPU vision offload (V2d, PROVEN WORKING) — runs the SigLIP encoder on the NPU via the
    Conv->MatMul int16 NBG + the A733_NPU_EMBEDDINGS injection into llama.cpp, freeing both A76
    cores. Slower vision (~6 s/img) but offloads CPU.
- Model selector (flag --model): smolvlm-256m (default), smolvlm-500m (more detail, ~18 tok/s).
  Do NOT offer >1B VLMs (InternVL3.5-1B loads but is too slow + starves RAM — V3 verified).
- Stream tokens to stdout; print backend, model, gen tok/s, and peak RSS at the end. Multi-turn:
  keep the image loaded, accept follow-up questions until the user types /exit.

=== TOOL 2: LLM text chat (CLI) — app/llm_chat.py ===
A terminal chat with a text LLM on CPU via llama.cpp.
- Usage: `python3 llm_chat.py` -> interactive REPL; `--model qwen-1.5b` etc.
- Model selector (flag --model), based on B5 measured fit/speed:
  * qwen-0.5b (fast, ~18 tok/s)
  * qwen-1.5b (balanced, recommended, ~8.5 tok/s)  [default]
  * qwen-3b ONLY if B5b confirmed it fits with headroom (else omit it / mark experimental).
- Pin to 2x A76 by default (taskset -c 6,7 — B5 showed it's fastest); flag --cores to override.
- Stream tokens; show tok/s and a token/context counter; /reset to clear, /exit to quit.
- Apply the model's chat template so replies are coherent.

=== Shared requirements ===
- Pure CLI, minimal deps, runs over SSH in a plain terminal. Sane defaults so `python3 vlm_chat.py
  --image x.jpg` and `python3 llm_chat.py` "just work" with the recommended models.
- A --cpu-only / graceful fallback: if --backend npu is requested but /dev/vipcore or the NBG is
  missing, print a clear message and fall back to CPU.
- Print a one-line resource note (RAM used, cores used) so the user knows ROS2 headroom.

=== Install & usage guide: docs/09-cli-tools.md (replace the web-app doc 09 if present) ===
1. From a fresh Orange Pi Zero 3W: system deps, building llama.cpp with multimodal (mmproj)
   support, the exact GGUF + mmproj downloads (SmolVLM-256M/500M; Qwen2.5-0.5B/1.5B — HF
   links/filenames already used in the repo), and the NPU bring-up only if the user wants
   --backend npu (link docs/02-board-bringup.md).
2. Exact run commands for both tools (one-shot and interactive), the flags, and example sessions.
3. Troubleshooting: OOM (pick a smaller model), NPU not present (CPU fallback), bad image path,
   wrong embedding alignment for --backend npu.
4. A "hardware requirements / expectations" box from the measured numbers: SmolVLM-256M 52 tok/s
   /634 MB; Qwen-1.5B 8.5 tok/s/~2 GB; what's left for ROS2/picoclaw; what NOT to run (>1B VLM,
   3B+ LLM if tight).
5. "For other A733 boards": CPU path is board-agnostic (any aarch64 with RAM); NPU path needs
   /dev/vipcore + glibc-matched VIPLite .so (link the porting guide).

DELIVERABLE: app/vlm_chat.py, app/llm_chat.py (CLI, no web), app/README.md (quick usage),
docs/09-cli-tools.md (install + run + troubleshoot), updated README "Start here" and
docs/configurations.md to point at the CLI tools. Remove the web-app references. Commit code + docs.

SUCCESS GATE: on the Orange Pi, `python3 app/vlm_chat.py --image <img> --question "..."` returns an
accurate answer in the terminal (CPU backend; and NPU backend if --backend npu), and
`python3 app/llm_chat.py` gives an interactive terminal chat with Qwen-1.5B. The install guide is
copy-pasteable on a fresh board. No web server anywhere. Committed.

DO NOT: build any web/HTTP/browser interface; offer >1B VLMs or OOM-ing LLM sizes as defaults;
break the V1 CPU path or the working V2d NPU-offload path.

START FROM: the existing app/ files (strip any web server, keep the logic); the V1 CPU SmolVLM
llama.cpp setup + GGUF/mmproj paths; the V2d NPU encoder + injector (A733_NPU_EMBEDDINGS, proven);
B5/B5b for the LLM model/speed table; the docs/ structure; the Orange Pi at 192.168.31.225.