#!/usr/bin/env python3
"""Download the SmolLM2 checkpoint files needed by the B1 matrix."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from urllib.parse import quote

import requests


MODELS = {
    "360m": ("HuggingFaceTB/SmolLM2-360M-Instruct", "smollm2-360m-instruct"),
    "1.7b": ("HuggingFaceTB/SmolLM2-1.7B-Instruct", "smollm2-1.7b-instruct"),
}

METADATA_FILES = {
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
}


def sibling_names(repo_id: str) -> set[str]:
    url = f"https://huggingface.co/api/models/{repo_id}"
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    payload = response.json()
    return {str(sibling["rfilename"]) for sibling in payload.get("siblings", [])}


def files_to_download(repo_id: str) -> list[str]:
    names = sibling_names(repo_id)
    files = sorted(name for name in METADATA_FILES if name in names)
    safetensors = sorted(
        name
        for name in names
        if name == "model.safetensors"
        or name == "model.safetensors.index.json"
        or (name.startswith("model-") and name.endswith(".safetensors"))
    )
    if not safetensors:
        raise SystemExit(f"{repo_id}: no model safetensors found")
    return files + safetensors


def download_file(repo_id: str, filename: str, target: Path) -> None:
    url = f"https://huggingface.co/{repo_id}/resolve/main/{quote(filename, safe='/')}"
    part = target.with_name(target.name + ".part")
    resume_at = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={resume_at}-"} if resume_at else {}

    with requests.get(url, headers=headers, stream=True, timeout=60, allow_redirects=True) as response:
        if response.status_code == 200 and resume_at:
            resume_at = 0
            part.unlink()
        elif response.status_code not in (200, 206):
            response.raise_for_status()

        mode = "ab" if resume_at else "wb"
        with part.open(mode + "") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)

    os.replace(part, target)


def download_model(repo_id: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    filenames = files_to_download(repo_id)
    print(f"{repo_id}: downloading {len(filenames)} files to {output_dir}")
    for filename in filenames:
        target = output_dir / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.stat().st_size > 0:
            print(f"  {filename} already exists ({target.stat().st_size} bytes)")
            continue
        download_file(repo_id, filename, target)
        print(f"  {filename} -> {target}")

    shards = [name for name in filenames if name.startswith("model-") and name.endswith(".safetensors")]
    if shards:
        print(f"{repo_id}: downloaded sharded safetensors ({len(shards)} shards)")
    elif "model.safetensors" in filenames:
        print(f"{repo_id}: downloaded unsharded model.safetensors")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-root", type=Path, default=Path("work/models"))
    parser.add_argument("--model", choices=sorted(MODELS), action="append")
    args = parser.parse_args()

    selected = args.model or sorted(MODELS)
    for key in selected:
        repo_id, dirname = MODELS[key]
        download_model(repo_id, args.models_root / dirname)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
