# Run VLM with NPU Vision Offload (Hybrid)

Reproducible steps to run SmolVLM-256M image chat with vision on the NPU and
LLM on CPU.

## Prerequisites (Orange Pi Zero 3W)

```bash
# Models (already deployed)
/home/orangepi/a733_npu_driver/models/vlm/SmolVLM-256M-Instruct-Q8_0.gguf
/home/orangepi/a733_npu_driver/models/vlm/mmproj-SmolVLM-256M-Instruct-Q8_0.gguf
/home/orangepi/a733_npu_driver/models/smolvlm_256m_vision_v2d_int16/network_binary.nb

# Tools
/opt/vpm_run/vpm_run  # VIPLite runner
/home/orangepi/llama.cpp/build/bin/llama-cli  # with V2c mtmd patch
```

## Step 1: Preprocess Image

```bash
python3 -c "
from PIL import Image
import numpy as np

img = Image.open('/path/to/image.jpg').convert('RGB')
img = img.resize((512, 512), Image.BICUBIC)
arr = np.array(img, dtype=np.float32) / 255.0
arr = (arr - 0.5) / 0.5                    # normalize to [-1,1]
arr = arr.transpose(2, 0, 1).reshape(1, 3, 512, 512)
scale = 2.0 ** 15
int16_arr = np.clip(np.round(arr * scale), -32768, 32767).astype(np.int16)
int16_arr.tofile('input.dat')
"
```

## Step 2: Run NPU Vision Encoder

```bash
cd /home/orangepi/a733_npu_driver/models/smolvlm_256m_vision_v2d_int16
printf '[network]\n./network_binary.nb\n[input]\n./input.dat\n' > sample.txt
export LD_LIBRARY_PATH=/home/orangepi/lib
/opt/vpm_run/vpm_run -s sample.txt -b 0 --save_txt 1
# Output: output_0.txt (36864 float32 values = 64x576)
```

## Step 3: Convert to Float32 Binary

```bash
python3 -c "
import struct
vals = []
with open('output_0.txt') as f:
    for line in f:
        line = line.strip()
        if not line: continue
        try: vals.append(float(line))
        except: pass
with open('embeddings.bin', 'wb') as f:
    for v in vals:
        f.write(struct.pack('<f', float(v)))
"
```

## Step 4: Run LLM with NPU Embeddings

```bash
export A733_NPU_EMBEDDINGS=embeddings.bin
export LD_LIBRARY_PATH=/home/orangepi/llama.cpp/build/bin

printf '/exit\n' | /home/orangepi/llama.cpp/build/bin/llama-cli \
    -m /home/orangepi/a733_npu_driver/models/vlm/SmolVLM-256M-Instruct-Q8_0.gguf \
    --mmproj /home/orangepi/a733_npu_driver/models/vlm/mmproj-SmolVLM-256M-Instruct-Q8_0.gguf \
    --image /path/to/image.jpg \
    -p '<image>Describe this image.' \
    -n 100 -t 2 --temp 0.0 --simple-io --no-perf --log-disable
```

## One-liner for dog.jpg

```bash
cd /home/orangepi/a733_npu_driver/models/smolvlm_256m_vision_v2d_int16 && \
printf '[network]\n./network_binary.nb\n[input]\n./dog_input.dat\n' > s.txt && \
LD_LIBRARY_PATH=/home/orangepi/lib /opt/vpm_run/vpm_run -s s.txt -b 0 --save_txt 1 && \
python3 -c "import struct;f=open('output_0.txt');v=[float(l.strip())for l in f if l.strip()];f.close();open('/tmp/e.bin','wb').write(b''.join(struct.pack('<f',x)for x in v))" && \
A733_NPU_EMBEDDINGS=/tmp/e.bin LD_LIBRARY_PATH=/home/orangepi/llama.cpp/build/bin \
printf '/exit\n' | /home/orangepi/llama.cpp/build/bin/llama-cli \
  -m /home/orangepi/a733_npu_driver/models/vlm/SmolVLM-256M-Instruct-Q8_0.gguf \
  --mmproj /home/orangepi/a733_npu_driver/models/vlm/mmproj-SmolVLM-256M-Instruct-Q8_0.gguf \
  --image /home/orangepi/a733_npu_driver/test_images/dog.jpg \
  -p '<image>What animal is in this image?' -n 80 -t 2 --temp 0.0 --simple-io --no-perf --log-disable
```

## Expected Timings

| Step | Time |
|------|------|
| Image preprocessing | <1 sec |
| NPU create network | 237 ms |
| NPU prepare network | 12.4 sec (first time) |
| NPU vision inference | 5.95 sec |
| LLM prompt processing | ~174 t/s |
| LLM generation | ~46.5 t/s |

## Known Issues

- `llama-cli --no-conversation` is not supported with `--image`. Use pipe `/exit`
  through stdin instead.
- SmolVLM chat template in this llama.cpp version doesn't insert `<image>` in
  generated prompt. Use raw `<image>` in prompt text.
- PIL (Pillow) must be installed for image preprocessing. If unavailable,
  preprocess images on host and upload pre-made `*_input.dat` files.
