#!/usr/bin/env python3
"""Run V2c injector and capture output safely."""
import sys, os
sys.path.insert(0, r'C:\Users\ilyah\Documents\Work\a733_npu_driver\work\pydeps')
import paramiko

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('192.168.31.225', username='orangepi', password='orangepi', timeout=30)

# Upload latest C source
sftp = client.open_sftp()
sftp.put(r'C:\Users\ilyah\Documents\Work\a733_npu_driver\scripts\board\inject_embeds.c', '/tmp/inject_embeds.c')
sftp.close()

# Compile
cmd = ('gcc -O2 -o /tmp/inject_embeds /tmp/inject_embeds.c '
       '-I/home/orangepi/llama.cpp/include -I/home/orangepi/llama.cpp/ggml/include '
       '-L/home/orangepi/llama.cpp/build/bin -lllama -lggml -lggml-base -lggml-cpu '
       '-lm -lpthread -fopenmp -Wl,-rpath,/home/orangepi/llama.cpp/build/bin 2>&1')
stdin, stdout, stderr = client.exec_command(cmd, timeout=120)
stdin.close()
err = stderr.read().decode()
if 'error:' in err + stdout.read().decode():
    print('COMPILE ERROR:', err[:500])
    client.close()
    sys.exit(1)

# Run and save to file
MODEL = '/home/orangepi/a733_npu_driver/models/vlm/SmolVLM-256M-Instruct-Q8_0.gguf'
EMB = '/tmp/npu_dog_embeddings.bin'
run_cmd = f'{"/tmp/inject_embeds"} {MODEL} {EMB} "Describe this image." 64 576 128 > /tmp/v2c_out.txt 2>&1'
stdin, stdout, stderr = client.exec_command(run_cmd, timeout=600)
stdin.close()
stdout.read()  # wait for completion

# Download output
sftp = client.open_sftp()
sftp.get('/tmp/v2c_out.txt', 'work/generated/smolvlm_256m_v2c/v2c_e2e_out.txt')
sftp.close()

# Print safely
with open('work/generated/smolvlm_256m_v2c/v2c_e2e_out.txt', encoding='utf-8', errors='replace') as f:
    content = f.read()
print(content)

client.close()
