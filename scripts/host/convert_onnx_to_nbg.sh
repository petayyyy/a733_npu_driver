#!/usr/bin/env bash
set -euo pipefail

PATH="/usr/bin:/bin:$PATH"

usage() {
    cat <<'EOF'
Usage:
  scripts/host/convert_onnx_to_nbg.sh \
    --name NAME \
    --onnx PATH \
    --dataset PATH \
    --quant uint8|int16|bf16|fp16|pcq \
    --inputs NAMES \
    --input-size-list SIZES \
    --outputs NAMES

Optional:
  --image IMAGE        Docker image, default: ubuntu-npu:v2.0.10.1
  --target TARGET      ACUITY target id, default: VIP9000NANODI_PLUS_PID0X1000003B
  --package-root DIR   Output package root, default: work/model-packages
  --hybrid             Use ACUITY hybrid quantization for the quantize step
  --seed-quantize PATH
                       Existing .quantize file; skip quantize and export only
  --hybrid-seed-quantize PATH
                       Existing .quantize file to seed --hybrid and skip rebuild

Environment:
  DOCKER_RUN_ARGS      Extra arguments inserted after `docker run --rm`, for
                       example: --cpus 10 --memory 24g
EOF
}

die() {
    echo "error: $*" >&2
    exit 1
}

find_python() {
    for exe in python python3 py; do
        if command -v "$exe" >/dev/null 2>&1 && "$exe" -c "import sys" >/dev/null 2>&1; then
            echo "$exe"
            return 0
        fi
    done
    return 1
}

abs_existing() {
    local path=$1
    local dir
    local base
    dir=$(dirname -- "$path")
    base=$(basename -- "$path")
    (cd "$dir" >/dev/null 2>&1 && printf '%s/%s\n' "$(pwd -P)" "$base")
}

copy_dataset_payloads() {
    local dataset_file=$1
    local model_dir=$2
    local dataset_dir
    local line
    local item
    local src
    local dst

    dataset_dir=$(dirname -- "$dataset_file")
    while IFS= read -r line || [ -n "$line" ]; do
        line=${line%%#*}
        for item in $line; do
            [ -n "$item" ] || continue
            case "$item" in
                /*|[A-Za-z]:*)
                    die "dataset entries must be relative to the dataset file: $item"
                    ;;
            esac
            src="$dataset_dir/$item"
            [ -f "$src" ] || die "dataset payload not found: $src"
            dst="$model_dir/$item"
            mkdir -p "$(dirname -- "$dst")"
            cp "$src" "$dst"
        done
    done < "$dataset_file"
}

quote_for_inputs_outputs() {
    case "$1" in
        *"'"*) die "single quotes are not supported in ACUITY argument values: $1" ;;
    esac
    printf "'%s'" "$1"
}

NAME=
ONNX=
DATASET=
QUANT=
INPUTS=
INPUT_SIZE_LIST=
OUTPUTS=
IMAGE=ubuntu-npu:v2.0.10.1
TARGET=VIP9000NANODI_PLUS_PID0X1000003B
PACKAGE_ROOT=work/model-packages
AI_SDK_MODELS=work/ai-sdk/ZIFENG278-ai-sdk/models
ACUITY_PATH=/root/acuity-toolkit-whl-6.30.22/bin
VIV_SDK=/root/Vivante_IDE/VivanteIDE5.11.0/cmdtools
HYBRID=0
SEED_QUANTIZE=
HYBRID_SEED_QUANTIZE=

while [ "$#" -gt 0 ]; do
    case "$1" in
        --name) NAME=${2:-}; shift 2 ;;
        --onnx) ONNX=${2:-}; shift 2 ;;
        --dataset) DATASET=${2:-}; shift 2 ;;
        --quant) QUANT=${2:-}; shift 2 ;;
        --inputs) INPUTS=${2:-}; shift 2 ;;
        --input-size-list) INPUT_SIZE_LIST=${2:-}; shift 2 ;;
        --outputs) OUTPUTS=${2:-}; shift 2 ;;
        --image) IMAGE=${2:-}; shift 2 ;;
        --target) TARGET=${2:-}; shift 2 ;;
        --package-root) PACKAGE_ROOT=${2:-}; shift 2 ;;
        --hybrid) HYBRID=1; shift ;;
        --seed-quantize) SEED_QUANTIZE=${2:-}; shift 2 ;;
        --hybrid-seed-quantize) HYBRID_SEED_QUANTIZE=${2:-}; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown argument: $1" ;;
    esac
done

[ -n "$NAME" ] || die "--name is required"
[ -n "$ONNX" ] || die "--onnx is required"
[ -n "$DATASET" ] || die "--dataset is required"
[ -n "$QUANT" ] || die "--quant is required"
[ -n "$INPUTS" ] || die "--inputs is required"
[ -n "$INPUT_SIZE_LIST" ] || die "--input-size-list is required"
[ -n "$OUTPUTS" ] || die "--outputs is required"

case "$NAME" in
    *[!A-Za-z0-9_.-]*|"") die "--name must contain only letters, digits, dot, underscore, or dash" ;;
esac
case "$QUANT" in
    uint8|int16|bf16|fp16|pcq|perchannel_int16) ;;
    *) die "--quant must be one of: uint8, int16, bf16, fp16, pcq, perchannel_int16" ;;
esac

PYTHON=$(find_python) || die "python3 or python is required"

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd -P)
ONNX_ABS=$(abs_existing "$ONNX") || die "ONNX path not found: $ONNX"
DATASET_ABS=$(abs_existing "$DATASET") || die "dataset path not found: $DATASET"
[ -f "$ONNX_ABS" ] || die "ONNX path is not a file: $ONNX_ABS"
[ -f "$DATASET_ABS" ] || die "dataset path is not a file: $DATASET_ABS"
if [ -n "$HYBRID_SEED_QUANTIZE" ] && [ "$HYBRID" != "1" ]; then
    die "--hybrid-seed-quantize requires --hybrid"
fi
if [ -n "$SEED_QUANTIZE" ] && [ "$HYBRID" = "1" ]; then
    die "--seed-quantize skips all quantize passes; use --hybrid-seed-quantize for --hybrid"
fi
if [ -n "$SEED_QUANTIZE" ] && [ -n "$HYBRID_SEED_QUANTIZE" ]; then
    die "--seed-quantize and --hybrid-seed-quantize are mutually exclusive"
fi
if [ "$QUANT" = "fp16" ] && [ "$HYBRID" = "1" ]; then
    die "--hybrid is not supported with --quant fp16"
fi
SEED_QUANTIZE_ABS=
if [ -n "$SEED_QUANTIZE" ]; then
    SEED_QUANTIZE_ABS=$(abs_existing "$SEED_QUANTIZE") || die "seed quantize path not found: $SEED_QUANTIZE"
    [ -f "$SEED_QUANTIZE_ABS" ] || die "seed quantize path is not a file: $SEED_QUANTIZE_ABS"
fi
HYBRID_SEED_QUANTIZE_ABS=
if [ -n "$HYBRID_SEED_QUANTIZE" ]; then
    HYBRID_SEED_QUANTIZE_ABS=$(abs_existing "$HYBRID_SEED_QUANTIZE") || die "seed quantize path not found: $HYBRID_SEED_QUANTIZE"
    [ -f "$HYBRID_SEED_QUANTIZE_ABS" ] || die "seed quantize path is not a file: $HYBRID_SEED_QUANTIZE_ABS"
fi

cd "$REPO_ROOT"
SDK_SCRIPT_DIR="$AI_SDK_MODELS/../scripts"
MODEL_DIR="$AI_SDK_MODELS/$NAME"
PACKAGE_DIR="$PACKAGE_ROOT/$NAME/$QUANT"

[ -d "$SDK_SCRIPT_DIR" ] || die "AI SDK scripts directory not found: $SDK_SCRIPT_DIR"
[ -f "$SDK_SCRIPT_DIR/pegasus_setup.sh" ] || die "missing AI SDK pegasus_setup.sh in: $SDK_SCRIPT_DIR"
cp "$SDK_SCRIPT_DIR"/pegasus_*.sh "$AI_SDK_MODELS/"
cp "$SDK_SCRIPT_DIR/pegasus_setup.sh" "$AI_SDK_MODELS/env.sh"
if [ -f "$SDK_SCRIPT_DIR/awnet_normalize.py" ]; then
    cp "$SDK_SCRIPT_DIR/awnet_normalize.py" "$AI_SDK_MODELS/"
fi

rm -rf "$MODEL_DIR"
mkdir -p "$MODEL_DIR"
cp "$ONNX_ABS" "$MODEL_DIR/$NAME.onnx"
if [ -f "$ONNX_ABS.data" ]; then
    cp "$ONNX_ABS.data" "$MODEL_DIR/$(basename -- "$ONNX_ABS").data"
fi
cp "$DATASET_ABS" "$MODEL_DIR/dataset.txt"
for extra_dataset in "$(dirname -- "$DATASET_ABS")"/dataset[0-9]*.txt; do
    if [ -f "$extra_dataset" ]; then
        cp "$extra_dataset" "$MODEL_DIR/$(basename -- "$extra_dataset")"
    fi
done
copy_dataset_payloads "$DATASET_ABS" "$MODEL_DIR"
if [ -f "$(dirname -- "$DATASET_ABS")/tokens.txt" ]; then
    cp "$(dirname -- "$DATASET_ABS")/tokens.txt" "$MODEL_DIR/tokens.txt"
fi
if [ -n "$SEED_QUANTIZE_ABS" ]; then
    cp "$SEED_QUANTIZE_ABS" "$MODEL_DIR/${NAME}_${QUANT}.quantize"
fi
if [ -n "$HYBRID_SEED_QUANTIZE_ABS" ]; then
    cp "$HYBRID_SEED_QUANTIZE_ABS" "$MODEL_DIR/${NAME}_${QUANT}.quantize"
fi
printf -- "--inputs %s --input-size-list %s --outputs %s\n" \
    "$(quote_for_inputs_outputs "$INPUTS")" \
    "$(quote_for_inputs_outputs "$INPUT_SIZE_LIST")" \
    "$(quote_for_inputs_outputs "$OUTPUTS")" \
    > "$MODEL_DIR/inputs_outputs.txt"

DOCKER_REPO_ROOT=$REPO_ROOT
if command -v cygpath >/dev/null 2>&1; then
    DOCKER_REPO_ROOT=$(cygpath -w "$REPO_ROOT")
fi

CONTAINER_SCRIPT=$(cat <<EOF
set -euo pipefail
export ACUITY_PATH=$ACUITY_PATH
export VIV_SDK=$VIV_SDK
source env.sh v3
bash pegasus_import.sh "$NAME"
python3 - "$NAME" <<'PY'
from pathlib import Path
import sys

name = sys.argv[1]
model_dir = Path(name)
dataset = model_dir / "dataset.txt"
items = []
for line in dataset.read_text(encoding="ascii").splitlines():
    line = line.split("#", 1)[0].strip()
    if line:
        items.extend(line.split())

if items and all(item.lower().endswith(".npy") for item in items):
    inputmeta = model_dir / f"{name}_inputmeta.yml"
    text = inputmeta.read_text(encoding="ascii")
    text = text.replace("category: image", "category: undefined")
    text = text.replace("reverse_channel: true", "reverse_channel: false")
    inputmeta.write_text(text, encoding="ascii")
    print(f"patched tensor inputmeta for {name}: {inputmeta}")
PY
if [ "$QUANT" = "fp16" ]; then
    pushd "$NAME"
    PEGASUS=$ACUITY_PATH/pegasus
    if [ ! -e "\$PEGASUS" ]; then
        PEGASUS="python3 \$PEGASUS.py"
    fi
    if [ -n "$SEED_QUANTIZE" ]; then
        echo "using seeded quantize table: $NAME/${NAME}_fp16.quantize"
    else
        cmd="\$PEGASUS quantize \
            --model         ${NAME}.json \
            --model-data    ${NAME}.data \
            --device        CPU \
            --with-input-meta ${NAME}_inputmeta.yml \
            --compute-entropy \
            --rebuild \
            --model-quantize ${NAME}_fp16.quantize \
            --quantizer float16 \
            --qtype float16"
        echo "\$cmd"
        eval "\$cmd"
    fi
    cmd="\$PEGASUS inference \
        --model         ${NAME}.json \
        --model-data    ${NAME}.data \
        --dtype         quantized \
        --model-quantize ${NAME}_fp16.quantize \
        --iterations    1 \
        --device        CPU \
        --output-dir    ./inf/${NAME}_fp16 \
        --postprocess-file ${NAME}_postprocess_file.yml \
        --with-input-meta ${NAME}_inputmeta.yml"
    echo "\$cmd"
    eval "\$cmd"
    cmd="\$PEGASUS export ovxlib \
        --model                 ${NAME}.json \
        --model-data            ${NAME}.data \
        --dtype                 quantized \
        --model-quantize        ${NAME}_fp16.quantize \
        --target-ide-project    'linux64' \
        --with-input-meta       ${NAME}_inputmeta.yml \
        --postprocess-file      ${NAME}_postprocess_file.yml \
        --pack-nbg-unify \
        --optimize              ${TARGET} \
        --viv-sdk               ${VIV_SDK} \
        --output-path           ./wksp/${NAME}_fp16/${NAME}_fp16"
    echo "\$cmd"
    eval "\$cmd"
    popd
elif [ "$QUANT" = "perchannel_int16" ]; then
    pushd "$NAME"
    PEGASUS=$ACUITY_PATH/pegasus
    if [ ! -e "\$PEGASUS" ]; then
        PEGASUS="python3 \$PEGASUS.py"
    fi
    cmd="\$PEGASUS quantize \
        --model         ${NAME}.json \
        --model-data    ${NAME}.data \
        --device        CPU \
        --with-input-meta ${NAME}_inputmeta.yml \
        --compute-entropy \
        --rebuild \
        --model-quantize ${NAME}_int16.quantize \
        --quantizer perchannel_symmetric_affine \
        --qtype int16"
    echo "\$cmd"
    eval "\$cmd"
    popd
    bash pegasus_inference.sh "$NAME" "int16"
    bash pegasus_export_ovx_nbg.sh "$NAME" "int16" "$TARGET" "$VIV_SDK"
else
    if [ "$HYBRID" = "1" ]; then
        if [ -z "$HYBRID_SEED_QUANTIZE" ]; then
            bash pegasus_quantize.sh "$NAME" "$QUANT"
        fi
        bash pegasus_quantize_hybird.sh "$NAME" "$QUANT"
    elif [ -n "$SEED_QUANTIZE" ]; then
        echo "using seeded quantize table: $NAME/${NAME}_${QUANT}.quantize"
    else
        bash pegasus_quantize.sh "$NAME" "$QUANT"
    fi
    bash pegasus_inference.sh "$NAME" "$QUANT"
    bash pegasus_export_ovx_nbg.sh "$NAME" "$QUANT" "$TARGET" "$VIV_SDK"
fi
EOF
)

MSYS_NO_PATHCONV=1 docker run --rm ${DOCKER_RUN_ARGS:-} \
    -v "$DOCKER_REPO_ROOT:/workspace" \
    -w "/workspace/$AI_SDK_MODELS" \
    "$IMAGE" \
    bash -lc "$CONTAINER_SCRIPT"

PACKAGE_QUANT="$QUANT"
if [ "$QUANT" = "perchannel_int16" ]; then
    PACKAGE_QUANT="int16"
fi
"$PYTHON" - "$MODEL_DIR" "$PACKAGE_DIR" "$PACKAGE_QUANT" <<'PY'
from __future__ import annotations

import json
import math
from pathlib import Path
import re
import shutil
import struct
import sys


def sanitize(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")


def read_numbers(path: Path) -> list[float]:
    values: list[float] = []
    with path.open("r", encoding="ascii", errors="ignore") as handle:
        for line in handle:
            if ":" in line:
                line = line.split(":", 1)[1]
            for part in line.replace(",", " ").split():
                try:
                    values.append(float(part))
                except ValueError:
                    pass
    return values


def product(shape: list[int]) -> int:
    total = 1
    for dim in shape:
        total *= int(dim)
    return total


def candidates_for(inf_dir: Path, key: str, info: dict, prefer_quantized: bool) -> list[Path]:
    needles = [sanitize(key)]
    if info.get("name"):
        needles.append(sanitize(str(info["name"])))
    files = sorted(inf_dir.glob("iter_0_*.tensor"))
    matches: list[Path] = []
    for needle in needles:
        if not needle:
            continue
        matches.extend(path for path in files if needle in path.name)
    unique = list(dict.fromkeys(matches))
    if prefer_quantized:
        unique.sort(key=lambda path: (not path.name.endswith(".qnt.tensor"), path.name))
    else:
        unique.sort(key=lambda path: (path.name.endswith(".qnt.tensor"), path.name))
    return unique


def write_packed(path: Path, values: list[float], info: dict) -> None:
    quant = info.get("quantize")
    dtype = str(info.get("dtype", "")).lower()

    if quant:
        qtype = str(quant.get("qtype", "")).lower()
        if qtype in {"i16", "int16"}:
            writer = lambda handle, value: handle.write(struct.pack("<h", max(-32768, min(32767, int(round(value))))))
        elif qtype in {"u8", "uint8"}:
            writer = lambda handle, value: handle.write(struct.pack("<B", max(0, min(255, int(round(value))))))
        elif qtype in {"i8", "int8"}:
            writer = lambda handle, value: handle.write(struct.pack("<b", max(-128, min(127, int(round(value))))))
        else:
            raise SystemExit(f"unsupported quantized input qtype for {path.name}: {qtype}")
    elif dtype == "int32":
        writer = lambda handle, value: handle.write(struct.pack("<i", int(round(value))))
    elif dtype == "float16":
        writer = lambda handle, value: handle.write(struct.pack("<e", float(value)))
    elif dtype in {"float", "float32", ""}:
        writer = lambda handle, value: handle.write(struct.pack("<f", float(value)))
    else:
        raise SystemExit(f"unsupported input dtype for {path.name}: {dtype}")

    with path.open("wb") as handle:
        for value in values:
            writer(handle, value)


def write_float_text(path: Path, values: list[float], info: dict, source_is_quantized: bool) -> None:
    quant = info.get("quantize") if source_is_quantized else None
    if quant and str(quant.get("qtype", "")).lower() in {"i16", "int16", "i8", "int8"}:
        scale = 2.0 ** int(quant["fl"])
        values = [value / scale for value in values]
    elif quant and str(quant.get("qtype", "")).lower() in {"u8", "uint8"}:
        scale = float(quant["scale"])
        zero_point = float(quant["zero_point"])
        values = [(value - zero_point) * scale for value in values]

    with path.open("w", encoding="ascii", newline="\n") as handle:
        for value in values:
            handle.write(f"{value:.16f}\n")


model_dir = Path(sys.argv[1])
package_dir = Path(sys.argv[2])
quant = sys.argv[3]
name = model_dir.name
export_dir = model_dir / "wksp" / f"{name}_{quant}_nbg_unify"
inf_dir = model_dir / "inf" / f"{name}_{quant}"

if not export_dir.is_dir():
    raise SystemExit(f"missing export directory: {export_dir}")
if not inf_dir.is_dir():
    raise SystemExit(f"missing inference directory: {inf_dir}")

package_dir.mkdir(parents=True, exist_ok=True)
for old in package_dir.glob("*"):
    if old.is_dir():
        shutil.rmtree(old)
    else:
        old.unlink()

shutil.copy2(export_dir / "network_binary.nb", package_dir / "network_binary.nb")
shutil.copy2(export_dir / "nbg_meta.json", package_dir / "nbg_meta.json")
meta = json.loads((export_dir / "nbg_meta.json").read_text(encoding="ascii"))

sample_lines = ["[network]", "./network_binary.nb", "[input]"]
for index, (key, info) in enumerate(meta.get("Inputs", {}).items()):
    tensor_candidates = candidates_for(inf_dir, key, info, prefer_quantized=bool(info.get("quantize")))
    if not tensor_candidates:
        raise SystemExit(f"no ACUITY inference tensor found for input {key}")
    tensor_path = tensor_candidates[0]
    values = read_numbers(tensor_path)
    expected = product(info.get("shape", []))
    if expected and len(values) != expected:
        raise SystemExit(f"input {key} expected {expected} values, found {len(values)} in {tensor_path.name}")
    input_path = package_dir / f"input_{index}.dat"
    write_packed(input_path, values, info)
    sample_lines.append(f"./input_{index}.dat")

with (package_dir / "sample.txt").open("w", encoding="ascii", newline="\n") as handle:
    handle.write("\n".join(sample_lines) + "\n")

for index, (key, info) in enumerate(meta.get("Outputs", {}).items()):
    tensor_candidates = candidates_for(inf_dir, key, info, prefer_quantized=False)
    if not tensor_candidates:
        raise SystemExit(f"no ACUITY inference tensor found for output {key}")
    tensor_path = tensor_candidates[0]
    values = read_numbers(tensor_path)
    expected = product(info.get("shape", []))
    if expected and len(values) != expected:
        raise SystemExit(f"output {key} expected {expected} values, found {len(values)} in {tensor_path.name}")
    shutil.copy2(tensor_path, package_dir / f"host_output_{index}.raw.tensor")
    write_float_text(package_dir / f"host_output_{index}.txt", values, info, tensor_path.name.endswith(".qnt.tensor"))

tokens = model_dir / "tokens.txt"
if tokens.exists():
    shutil.copy2(tokens, package_dir / "tokens.txt")

print(f"wrote package: {package_dir}")
PY

echo "done: $PACKAGE_DIR"
