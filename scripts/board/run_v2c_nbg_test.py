#!/usr/bin/env python3
"""Run V2c NBG on Orange Pi with real image, compare with ONNX Runtime reference."""
import os, sys, subprocess, struct
import numpy as np
import cv2

MODEL_DIR = "/home/orangepi/a733_npu_driver/models/smolvlm_256m_vision_v2c_int16"
VPM_RUN = "/opt/vpm_run/vpm_run"
VPM_LIB = "/home/orangepi/lib"
TEST_IMAGES = "/home/orangepi/a733_npu_driver/test_images"

def preprocess(image_path, fl=15):
    """Preprocess image to NBG int16 DFP input."""
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
    pixels = img.astype(np.float32) / 127.5 - 1.0  # [-1, 1]
    pixels = np.transpose(pixels, (2, 0, 1))  # CHW
    pixels = pixels[np.newaxis, np.newaxis, :, :, :]  # [1,1,3,512,512]
    scale = 2.0 ** fl
    quantized = np.round(pixels * scale).clip(-32768, 32767).astype(np.int16)
    return quantized, pixels.astype(np.float32)

def run_nbg(input_dat_path, output_dir):
    """Run NBG via vpm_run, return float32 output."""
    # Write sample.txt
    sample_txt = os.path.join(output_dir, "sample_run.txt")
    with open(sample_txt, "w") as f:
        f.write(f"[network]\n./network_binary.nb\n[input]\n{input_dat_path}\n")
    
    # Remove old output
    old_out = os.path.join(output_dir, "output_0.txt")
    if os.path.exists(old_out):
        os.remove(old_out)
    
    # Run vpm_run
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = VPM_LIB
    result = subprocess.run(
        [VPM_RUN, "-s", sample_txt, "-l", "1", "-b", "0", "--save_txt", "1"],
        cwd=output_dir,
        capture_output=True,
        text=True,
        timeout=600,
        env=env
    )
    
    # Parse output
    if "vpm run ret=0" not in result.stdout:
        print(f"VPM_RUN FAILED:\n{result.stdout[-1000:]}")
        print(f"STDERR:\n{result.stderr[-500:]}")
        return None
    
    # Read output_0.txt
    out_path = os.path.join(output_dir, "output_0.txt")
    if not os.path.exists(out_path):
        print("No output_0.txt generated!")
        return None
    
    float_vals = np.loadtxt(out_path, dtype=np.float32)
    
    # Parse profile time
    for line in result.stdout.split("\n"):
        if "profile inference time" in line:
            print(f"  NPU: {line.strip()}")
        if "create network" in line:
            print(f"  {line.strip()}")
        if "prepare network" in line:
            print(f"  {line.strip()}")
    
    return float_vals

def main():
    # Preprocess dog.jpg
    dog_path = os.path.join(TEST_IMAGES, "dog.jpg")
    print(f"=== Preprocessing {dog_path} ===")
    quantized, float_input = preprocess(dog_path)
    
    # Save quantized input
    input_dat = os.path.join(MODEL_DIR, "dog_real_input.dat")
    quantized.tofile(input_dat)
    print(f"Input saved: {input_dat} ({os.path.getsize(input_dat)} bytes)")
    print(f"Quantized range: [{quantized.min()}, {quantized.max()}]")
    
    # Run NPU
    print(f"\n=== Running NBG on NPU ===")
    npu_output = run_nbg(input_dat, MODEL_DIR)
    if npu_output is None:
        print("NPU run failed!")
        sys.exit(1)
    
    npu_embeddings = npu_output.reshape(1, 64, 576)
    print(f"NPU output shape: {npu_embeddings.shape}")
    print(f"NPU output range: [{npu_embeddings.min():.3f}, {npu_embeddings.max():.3f}]")
    
    # Compare with host golden if available
    golden_path = "/home/orangepi/a733_npu_driver/work/generated/smolvlm_256m_v2c/golden_output.npy"
    if os.path.exists(golden_path):
        golden = np.load(golden_path)
        print(f"\nGolden shape: {golden.shape}")
        cos = np.dot(npu_embeddings.flatten(), golden.flatten()) / (
            np.linalg.norm(npu_embeddings.flatten()) * np.linalg.norm(golden.flatten())
        )
        print(f"NPU vs Golden cosine: {cos:.6f}")
    
    print("\nDone!")

if __name__ == "__main__":
    main()
