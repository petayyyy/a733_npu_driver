#!/usr/bin/env python3
"""V2c E2E: Run NPU vision + CPU LLM hybrid on all 3 V1 test images."""
import sys, os, time
sys.path.insert(0, r'C:\Users\ilyah\Documents\Work\a733_npu_driver\work\pydeps')
import paramiko
import numpy as np

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('192.168.31.225', username='orangepi', password='orangepi', timeout=30)

MODEL = '/home/orangepi/a733_npu_driver/models/vlm/SmolVLM-256M-Instruct-Q8_0.gguf'
MMPROJ = '/home/orangepi/a733_npu_driver/models/vlm/mmproj-SmolVLM-256M-Instruct-Q8_0.gguf'
LLAMA = '/home/orangepi/llama.cpp/build/bin/llama-cli'
IMG_DIR = '/home/orangepi/a733_npu_driver/test_images'
NPU_DIR = '/home/orangepi/a733_npu_driver/models/smolvlm_256m_vision_v2c_int16'
VPM = '/opt/vpm_run/vpm_run'
VPM_LIB = '/home/orangepi/lib'

test_images = [
    ('dog.jpg', 'What animal is in this image?'),
    ('cat.jpg', 'What animal is in this image?'),
    ('test-1.jpeg', 'Describe this image.'),
]

results = []

for img_name, prompt in test_images:
    print(f"\n{'='*60}")
    print(f"IMAGE: {img_name} | PROMPT: {prompt}")
    print(f"{'='*60}")
    
    # Step 1: Preprocess image and run NPU
    print("  [1/4] Preprocessing image...")
    stdin, stdout, stderr = client.exec_command(
        f'python3 -c "'
        f'import cv2, numpy as np; '
        f'img = cv2.imread(\"{IMG_DIR}/{img_name}\"); '
        f'img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB); '
        f'img = cv2.resize(img, (512,512)); '
        f'pixels = img.astype(np.float32) / 127.5 - 1.0; '
        f'pixels = np.transpose(pixels, (2,0,1)); '
        f'pixels = pixels[np.newaxis, np.newaxis, :, :, :]; '
        f'q = np.round(pixels * 32768).clip(-32768, 32767).astype(np.int16); '
        f'q.tofile(\"/tmp/v2c_input.dat\")"',
        timeout=30)
    stdin.close()
    stdout.read()
    
    # Step 2: Run NPU
    print("  [2/4] Running NPU vision encoder...")
    stdin, stdout, stderr = client.exec_command(
        f'cp /tmp/v2c_input.dat {NPU_DIR}/test_input.dat && '
        f'printf "[network]\\n./network_binary.nb\\n[input]\\n./test_input.dat\\n" > {NPU_DIR}/sample_test.txt && '
        f'cd {NPU_DIR} && LD_LIBRARY_PATH={VPM_LIB} {VPM} -s sample_test.txt -l 1 -b 0 --save_txt 1 2>&1',
        timeout=600)
    stdin.close()
    vpm_out = stdout.read().decode('utf-8', errors='replace')
    
    # Parse NPU timing
    npu_time = 0
    for line in vpm_out.split('\n'):
        if 'profile inference time' in line:
            npu_time = int(line.split('=')[1].replace('us','').strip()) / 1000
    print(f"  NPU done: {npu_time:.1f} ms")
    
    # Step 3: Convert NPU output to binary embeddings
    print("  [3/4] Converting NPU output to embeddings...")
    stdin, stdout, stderr = client.exec_command(
        f'python3 -c "'
        f'import numpy as np; '
        f'e = np.loadtxt(\"{NPU_DIR}/output_0.txt\", dtype=np.float32); '
        f'e = e.reshape(1, 64, 576); '
        f'e.astype(np.float32).tofile(\"/tmp/v2c_emb.bin\")"',
        timeout=30)
    stdin.close()
    stdout.read()
    
    # Step 4: Run llama-cli with NPU embeddings
    print("  [4/4] Running VLM decoder with NPU embeddings...")
    t0 = time.time()
    stdin, stdout, stderr = client.exec_command(
        f'A733_NPU_EMBEDDINGS=/tmp/v2c_emb.bin taskset -c 6,7 {LLAMA} '
        f'-m {MODEL} --mmproj {MMPROJ} --image {IMG_DIR}/{img_name} '
        f'-p "{prompt}" -n 128 --temp 0.0 -t 2 '
        f'--simple-io --no-perf --log-disable 2>/dev/null',
        timeout=600)
    stdin.close()
    answer = stdout.read().decode('utf-8', errors='replace')
    elapsed = time.time() - t0
    
    # Parse answer (skip spinner/load messages)
    answer_lines = []
    in_answer = False
    for line in answer.split('\n'):
        if 'Loaded media' in line:
            in_answer = True
            continue
        if in_answer and line.strip() and '> ' not in line:
            answer_lines.append(line.strip())
    
    clean_answer = ' '.join(answer_lines)
    
    print(f"\n  ANSWER: {clean_answer[:300]}")
    print(f"  Time: {elapsed:.1f}s + NPU {npu_time:.1f}ms")
    
    results.append({
        'image': img_name,
        'prompt': prompt,
        'answer': clean_answer,
        'npu_ms': npu_time,
        'llm_sec': elapsed,
    })

# Summary
print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
for r in results:
    print(f"\n{r['image']}: \"{r['prompt']}\"")
    print(f"  {r['answer'][:200]}")
    print(f"  NPU: {r['npu_ms']:.0f}ms | LLM: {r['llm_sec']:.1f}s")

client.close()
