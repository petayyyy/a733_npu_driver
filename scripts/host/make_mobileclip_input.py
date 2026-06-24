#!/usr/bin/env python3
"""Prepare a MobileCLIP-S0 pixel_values tensor from a real image."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def resize_shortest_edge(image: Image.Image, shortest_edge: int, resample: int) -> Image.Image:
    width, height = image.size
    if width <= 0 or height <= 0:
        raise SystemExit(f"invalid image size: {image.size}")
    scale = shortest_edge / min(width, height)
    new_width = int(round(width * scale))
    new_height = int(round(height * scale))
    return image.resize((new_width, new_height), resample=resample)


def center_crop(image: Image.Image, crop_width: int, crop_height: int) -> Image.Image:
    width, height = image.size
    if width < crop_width or height < crop_height:
        raise SystemExit(f"image too small for crop after resize: {image.size}")
    left = (width - crop_width) // 2
    top = (height - crop_height) // 2
    return image.crop((left, top, left + crop_width, top + crop_height))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--preprocessor-config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    cfg = json.loads(args.preprocessor_config.read_text(encoding="utf-8"))
    crop_size = cfg.get("crop_size", {})
    size_cfg = cfg.get("size", {})
    crop_height = int(crop_size.get("height", 256))
    crop_width = int(crop_size.get("width", 256))
    shortest_edge = int(size_cfg.get("shortest_edge", crop_height))
    rescale_factor = float(cfg.get("rescale_factor", 1.0 / 255.0))
    resample = int(cfg.get("resample", Image.Resampling.BILINEAR))

    image = Image.open(args.image)
    if cfg.get("do_convert_rgb", True):
        image = image.convert("RGB")
    if cfg.get("do_resize", True):
        image = resize_shortest_edge(image, shortest_edge, resample)
    if cfg.get("do_center_crop", True):
        image = center_crop(image, crop_width, crop_height)

    array = np.asarray(image, dtype=np.float32)
    if cfg.get("do_rescale", True):
        array *= rescale_factor
    if cfg.get("do_normalize", False):
        mean = np.asarray(cfg["image_mean"], dtype=np.float32).reshape(1, 1, 3)
        std = np.asarray(cfg["image_std"], dtype=np.float32).reshape(1, 1, 3)
        array = (array - mean) / std

    pixel_values = np.transpose(array, (2, 0, 1))[None, :, :, :].astype(np.float32)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "pixel_values.npy", pixel_values)
    with (args.output_dir / "dataset.txt").open("w", encoding="ascii", newline="\n") as handle:
        handle.write("pixel_values.npy\n")
    with (args.output_dir / "inputs_outputs.txt").open("w", encoding="ascii", newline="\n") as handle:
        handle.write("--inputs pixel_values --input-size-list '3,256,256' --outputs image_embeds\n")
    with (args.output_dir / "source_image.txt").open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(str(args.image) + "\n")
    print(f"wrote {args.output_dir / 'pixel_values.npy'} shape={pixel_values.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
