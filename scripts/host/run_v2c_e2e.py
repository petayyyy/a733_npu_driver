#!/usr/bin/env python3
"""Run V2c embedding injection e2e test on Orange Pi."""
import sys, os
sys.path.insert(0, r'C:\Users\ilyah\Documents\Work\a733_npu_driver\work\pydeps')
import paramiko
import numpy as np

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('192.168.31.225', username='orangepi', password='orangepi', timeout=30)

# Step 1: Download NPU output
sftp = client.open_sftp()
sftp.get('/home/orangepi/a733_npu_driver/models/smolvlm_256m_vision_v2c_int16/output_0.txt',
         'work/generated/smolvlm_256m_v2c/npu_dog_output_e2e.txt')
sftp.close()

# Step 2: Convert to binary
embeddings = np.loadtxt('work/generated/smolvlm_256m_v2c/npu_dog_output_e2e.txt', dtype=np.float32)
embeddings = embeddings.reshape(1, 64, 576)
print(f'Embeddings: shape={embeddings.shape}, range=[{embeddings.min():.3f}, {embeddings.max():.3f}]')
bin_path = 'work/generated/smolvlm_256m_v2c/npu_dog_embeddings.bin'
embeddings.astype(np.float32).tofile(bin_path)
print(f'Binary: {os.path.getsize(bin_path)} bytes')

# Step 3: Upload binary
sftp = client.open_sftp()
sftp.put(bin_path, '/tmp/npu_dog_embeddings.bin')
sftp.close()

# Step 4: Run injector
MODEL = '/home/orangepi/a733_npu_driver/models/vlm/SmolVLM-256M-Instruct-Q8_0.gguf'
INJECTOR = '/tmp/inject_embeds'
PROMPT = 'Describe this image.'
cmd = f'{INJECTOR} {MODEL} /tmp/npu_dog_embeddings.bin "{PROMPT}" 64 576 64'
print(f'Running: {cmd[:100]}...')

stdin, stdout, stderr = client.exec_command(cmd, timeout=600)
stdin.close()
out = stdout.read().decode('utf-8', errors='replace')
err = stderr.read().decode('utf-8', errors='replace')
print('=== V2c E2E RESULT ===')
print(out)
if err:
    print('STDERR:', err[:500])

client.close()
