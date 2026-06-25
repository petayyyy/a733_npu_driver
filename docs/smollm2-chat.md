# SmolLM2 Chat — Superseded

This document is superseded by:
- **[04-chat-shell.md](04-chat-shell.md)** — Interactive chat shell on Orange Pi
- **[03-run-llm-npu.md](03-run-llm-npu.md)** — Convert and run SmolLM2 on NPU
- **[configurations.md](configurations.md)** — All configs at a glance

The `scripts/board/smollm2_chat.py` script was the earlier Radxa A7Z chat
wrapper. The current recommended chat entry point is `chat_shell.py` on the
Orange Pi Zero 3W (see [04-chat-shell.md](04-chat-shell.md)).

For Radxa Cubie A7Z users, the old `smollm2_chat.py` still works with the
W=64 int16 NBG at `/home/radxa/a733_npu_driver/models/smollm2_135m_w64_int16/`.
