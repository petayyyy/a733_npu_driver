#!/usr/bin/env python3
"""Run a command while recording stdout/stderr, TTFT, wall time, and peak RSS."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import threading
import time
from typing import BinaryIO


def read_vm_rss_kb(pid: int) -> int:
    try:
        with open(f"/proc/{pid}/status", "r", encoding="utf-8") as status:
            for line in status:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return int(parts[1])
    except FileNotFoundError:
        return 0
    return 0


def copy_stream(
    stream: BinaryIO,
    out_path: Path,
    start_ns: int,
    first_seen: dict[str, float | None],
    key: str,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as out:
        while True:
            chunk = os.read(stream.fileno(), 4096)
            if not chunk:
                return
            if first_seen[key] is None:
                first_seen[key] = (time.monotonic_ns() - start_ns) / 1_000_000.0
            out.write(chunk)
            out.flush()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-json", required=True)
    parser.add_argument("--stdout", required=True)
    parser.add_argument("--stderr", required=True)
    parser.add_argument("--poll-interval", type=float, default=0.05)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise SystemExit("missing command after --")

    stdout_path = Path(args.stdout)
    stderr_path = Path(args.stderr)
    metrics_path = Path(args.metrics_json)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    started_ns = time.monotonic_ns()
    proc = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=subprocess.DEVNULL,
        close_fds=True,
    )
    assert proc.stdout is not None
    assert proc.stderr is not None

    first_seen: dict[str, float | None] = {"stdout_ms": None, "stderr_ms": None}
    stdout_thread = threading.Thread(
        target=copy_stream,
        args=(proc.stdout, stdout_path, started_ns, first_seen, "stdout_ms"),
        daemon=True,
    )
    stderr_thread = threading.Thread(
        target=copy_stream,
        args=(proc.stderr, stderr_path, started_ns, first_seen, "stderr_ms"),
        daemon=True,
    )
    stdout_thread.start()
    stderr_thread.start()

    peak_rss_kb = 0
    rss_samples = 0
    while proc.poll() is None:
        rss_kb = read_vm_rss_kb(proc.pid)
        if rss_kb > peak_rss_kb:
            peak_rss_kb = rss_kb
        rss_samples += 1
        time.sleep(args.poll_interval)

    returncode = proc.wait()
    stdout_thread.join(timeout=5)
    stderr_thread.join(timeout=5)
    ended_ns = time.monotonic_ns()
    rss_kb = read_vm_rss_kb(proc.pid)
    if rss_kb > peak_rss_kb:
        peak_rss_kb = rss_kb

    metrics = {
        "command": command,
        "cwd": os.getcwd(),
        "returncode": returncode,
        "wall_ms": (ended_ns - started_ns) / 1_000_000.0,
        "first_stdout_ms": first_seen["stdout_ms"],
        "first_stderr_ms": first_seen["stderr_ms"],
        "peak_rss_kb": peak_rss_kb,
        "rss_samples": rss_samples,
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
    }
    with metrics_path.open("w", encoding="utf-8") as out:
        json.dump(metrics, out, indent=2, sort_keys=True)
        out.write("\n")
    print(json.dumps(metrics, sort_keys=True))
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
