#!/usr/bin/env python3
"""V2d: E2E hybrid VLM runner — NPU vision + CPU LLM via mmproj."""
import sys, os, struct, time, tempfile, hashlib
sys.path.insert(0, r'C:\Users\ilyah\Documents\Work\a733_npu_driver\work\pydeps')

import numpy as np
from PIL import Image
from pathlib import Path
import paramiko
import json

REPO = Path(r'C:\Users\ilyah\Documents\Work\a733_npu_driver')

class OrangePi:
    def __init__(self):
        self.client = paramiko.SSHClient()
        self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.client.connect('192.168.31.225', username='orangepi', password='orangepi', timeout=30)
    
    def run(self, cmd, timeout=120):
        stdin, stdout, stderr = self.client.exec_command(cmd, timeout=timeout)
        stdin.close()
        out = stdout.read().decode('utf-8', errors='replace')
        err = stderr.read().decode('utf-8', errors='replace')
        status = stdout.channel.recv_exit_status()
        return status, out, err
    
    def put(self, local, remote):
        sftp = self.client.open_sftp()
        try: sftp.stat(str(Path(remote).parent))
        except: 
            parts = str(Path(remote).parent).split('/')
            cur = ''
            for p in parts:
                if not p: continue
                cur += '/' + p
                try: sftp.stat(cur)
                except: sftp.mkdir(cur)
        sftp.put(str(local), remote)
        sftp.close()
    
    def get(self, remote, local):
        sftp = self.client.open_sftp()
        Path(local).parent.mkdir(parents=True, exist_ok=True)
        sftp.get(remote, str(local))
        sftp.close()
    
    def close(self):
        self.client.close()

def prep_image(img_path, fl=15):
    """Preprocess image for SmolVLM NBG input (int16 DFP)."""
    img = Image.open(img_path).convert('RGB')
    img = img.resize((512, 512), Image.BICUBIC)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    arr = arr.transpose(2, 0, 1).reshape(1, 3, 512, 512)
    scale = 2.0 ** fl
    int16_arr = np.clip(np.round(arr * scale), -32768, 32767).astype(np.int16)
    return int16_arr

def npu_output_to_embeddings(output_txt_path):
    """Convert vpm_run output_0.txt to float32 binary embeddings."""
    vals = []
    with open(output_txt_path) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try: vals.append(float(line))
            except: pass
    return np.array(vals, dtype=np.float32)

def run_vision(opi, int16_arr, tag):
    """Run NPU vision encoder on board, return embeddings array."""
    NBG_DIR = '/home/orangepi/a733_npu_driver/models/smolvlm_256m_vision_v2d_int16'
    
    # Upload input
    local_input = tempfile.mktemp(suffix='.dat')
    int16_arr.tofile(local_input)
    opi.put(local_input, f'{NBG_DIR}/{tag}_input.dat')
    os.unlink(local_input)
    
    # Create sample.txt
    sample_content = '[network]\n./network_binary.nb\n[input]\n./' + tag + '_input.dat\n'
    local_sample = tempfile.mktemp(suffix='.txt')
    with open(local_sample, 'w', newline='\n') as f:
        f.write(sample_content)
    opi.put(local_sample, f'{NBG_DIR}/sample_{tag}.txt')
    os.unlink(local_sample)
    
    # Run vpm_run
    t0 = time.time()
    status, out, err = opi.run(
        f'cd {NBG_DIR} && export LD_LIBRARY_PATH=/home/orangepi/lib && '
        f'/opt/vpm_run/vpm_run -s sample_{tag}.txt -b 0 --save_txt 1 2>&1',
        timeout=300
    )
    t1 = time.time()
    
    if status != 0:
        raise RuntimeError(f'vpm_run failed: {out}\n{err}')
    
    # Parse timing from output
    create_us = prepare_us = profile_us = None
    for line in out.split('\n'):
        if 'create network' in line:
            try: create_us = int(line.split()[-2])
            except: pass
        elif 'prepare network' in line:
            try: prepare_us = int(line.split()[-2])
            except: pass
        elif 'profile inference time' in line:
            try: profile_us = int(line.strip().split('=')[1].split('us')[0])
            except: pass
    
    # Download output
    local_output = str(REPO / f'work/generated/smolvlm_256m_v2d/npu_output_{tag}.txt')
    opi.get(f'{NBG_DIR}/output_0.txt', local_output)
    
    # Parse embeddings
    embeddings = npu_output_to_embeddings(local_output)
    
    return {
        'embeddings': embeddings,
        'wall_ms': (t1 - t0) * 1000,
        'create_ms': create_us / 1000 if create_us else None,
        'prepare_ms': prepare_us / 1000 if prepare_us else None,
        'profile_ms': profile_us / 1000 if profile_us else None,
        'raw_log': out,
    }

def run_llm_decode(opi, image_path, embeddings, prompt, n_gen=128):
    """Run llama-cli with NPU embeddings injected via env var."""
    # Upload embeddings
    local_emb = tempfile.mktemp(suffix='.bin')
    with open(local_emb, 'wb') as f:
        for v in embeddings.flatten():
            f.write(struct.pack('<f', float(v)))
    emb_remote = f'/tmp/v2d_emb_{os.getpid()}.bin'
    opi.put(local_emb, emb_remote)
    os.unlink(local_emb)
    
    VLM_DIR = '/home/orangepi/a733_npu_driver/models/vlm'
    MODEL = f'{VLM_DIR}/SmolVLM-256M-Instruct-Q8_0.gguf'
    MMPROJ = f'{VLM_DIR}/mmproj-SmolVLM-256M-Instruct-Q8_0.gguf'
    LLAMA_CLI = '/home/orangepi/llama.cpp/build/bin/llama-cli'
    IMG_REMOTE = '/home/orangepi/a733_npu_driver/test_images/' + os.path.basename(image_path)
    
    cmd = (
        f'export A733_NPU_EMBEDDINGS="{emb_remote}" && '
        f'export LD_LIBRARY_PATH="/home/orangepi/llama.cpp/build/bin" && '
        f'{LLAMA_CLI} '
        f'-m "{MODEL}" --mmproj "{MMPROJ}" --image "{IMG_REMOTE}" '
        f'-p "{prompt}" --chat-template smolvlm '
        f'-n {n_gen} -t 2 --temp 0.0 --no-conversation '
        f'2>/tmp/v2d_llama_err_{os.getpid()}.log'
    )
    
    t0 = time.time()
    status, out, err = opi.run(cmd, timeout=600)
    t1 = time.time()
    
    # Cleanup remote temp
    opi.run(f'rm -f {emb_remote} /tmp/v2d_llama_err_*.log', timeout=5)
    
    return {
        'status': status,
        'answer': out.strip(),
        'stderr': err.strip(),
        'wall_ms': (t1 - t0) * 1000,
    }

def main():
    images = [
        ('dog.jpg', 'Describe this image.'),
        ('cat.jpg', 'Describe this image.'),
        ('test-1.jpeg', 'Describe this image.'),
    ]
    
    opi = OrangePi()
    results = {}
    
    try:
        for img_name, prompt in images:
            print(f'\n{"="*60}')
            print(f'=== Testing: {img_name} ===')
            print(f'{"="*60}')
            
            img_path = str(REPO / f'work/generated/smolvlm_256m_v2d/test_images/{img_name}')
            
            # Preprocess
            print('Step 1: Preprocessing image...')
            int16_arr = prep_image(img_path)
            print(f'  Input shape: {int16_arr.shape}, range [{int16_arr.min()}, {int16_arr.max()}]')
            
            # Run vision
            print('Step 2: Running NPU vision encoder...')
            vision = run_vision(opi, int16_arr, img_name.replace('.', '_'))
            print(f'  Wall: {vision["wall_ms"]:.0f}ms, Profile: {vision["profile_ms"]:.0f}ms')
            print(f'  Create: {vision["create_ms"]:.0f}ms, Prepare: {vision["prepare_ms"]:.0f}ms')
            print(f'  Embeddings: {vision["embeddings"].shape}, range [{vision["embeddings"].min():.4f}, {vision["embeddings"].max():.4f}]')
            
            # Verify embeddings vs golden
            golden_path = REPO / f'work/generated/smolvlm_256m_v2d/golden_dog.npy'
            if golden_path.exists():
                golden = np.load(str(golden_path))
                cos = np.dot(vision['embeddings'].flatten(), golden.flatten()) / (
                    np.linalg.norm(vision['embeddings'].flatten()) * np.linalg.norm(golden.flatten())
                )
                print(f'  Cosine vs golden: {cos:.6f}')
            
            # Run LLM
            print('Step 3: Running LLM decode...')
            llm = run_llm_decode(opi, img_name, vision['embeddings'], prompt)
            print(f'  Wall: {llm["wall_ms"]:.0f}ms, Status: {llm["status"]}')
            print(f'  Answer: {llm["answer"][:500]}')
            
            if llm['stderr']:
                # Filter spinner chars
                stderr_clean = llm['stderr']
                for ch in ['\xe2\x96\x8c', '\xe2\x96\x8f', '\xe2\x96\x8e']:
                    stderr_clean = stderr_clean.replace(ch.encode('utf-8', errors='ignore').decode('utf-8', errors='ignore'), '')
                print(f'  Stderr: {stderr_clean[:300]}')
            
            results[img_name] = {
                'vision': vision,
                'llm': llm,
            }
    
    finally:
        opi.close()
    
    # Print summary
    print(f'\n{"="*60}')
    print('=== V2d E2E Results Summary ===')
    for img_name, r in results.items():
        print(f'\n--- {img_name} ---')
        print(f'  Vision: {r["vision"]["wall_ms"]:.0f}ms wall, {r["vision"]["profile_ms"]:.0f}ms profile')
        print(f'  LLM:    {r["llm"]["wall_ms"]:.0f}ms wall')
        print(f'  Answer: {r["llm"]["answer"][:200]}')
    
    # Save results
    import json
    summary = {}
    for img_name, r in results.items():
        summary[img_name] = {
            'vision_wall_ms': r['vision']['wall_ms'],
            'vision_profile_ms': r['vision']['profile_ms'],
            'llm_wall_ms': r['llm']['wall_ms'],
            'answer': r['llm']['answer'],
        }
    out_path = REPO / 'work/generated/smolvlm_256m_v2d/v2d_e2e_results.json'
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\nResults saved: {out_path}')

if __name__ == '__main__':
    main()
