#!/usr/bin/env python3
"""V2c E2E - final version. Runs VLM with timeout, reads answer from file."""
import sys, os, time
sys.path.insert(0, r'C:\Users\ilyah\Documents\Work\a733_npu_driver\work\pydeps')
import paramiko

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('192.168.31.225', username='orangepi', password='orangepi', timeout=30)

def test_image(img_name, prompt, emb_bin='/tmp/v2c_emb.bin'):
    out_file = f'/home/orangepi/v2c_{img_name}_out.txt'
    
    # Run with timeout - kill after 120s (answer generated in first 5-10s)
    cmd = (
        f'A733_NPU_EMBEDDINGS={emb_bin} timeout 120 '
        f'/home/orangepi/llama.cpp/build/bin/llama-cli '
        f'-m /home/orangepi/a733_npu_driver/models/vlm/SmolVLM-256M-Instruct-Q8_0.gguf '
        f'--mmproj /home/orangepi/a733_npu_driver/models/vlm/mmproj-SmolVLM-256M-Instruct-Q8_0.gguf '
        f'--image /home/orangepi/a733_npu_driver/test_images/{img_name} '
        f'-p "{prompt}" -n 64 --temp 0.0 -t 2 '
        f'> {out_file} 2>/dev/null; true'
    )
    
    stdin, stdout, stderr = client.exec_command(cmd, timeout=180)
    stdin.close()
    stdout.read()  # wait for completion (may be killed by timeout)
    
    # Read answer file - use tail to avoid processing huge spinner text
    stdin2, stdout2, stderr2 = client.exec_command(
        f'tail -c 50000 {out_file} 2>/dev/null | '
        f'grep -v "^$" | grep -v "?" | grep -v "|" | '
        f'grep -v "Loading" | grep -v "build" | grep -v "model:" | '
        f'grep -v "modalities" | grep -v "available" | grep -v "^/" | '
        f'grep -v "^>" | grep -E "^[A-Z]|^ [A-Z]" | head -10',
        timeout=15)
    stdin2.close()
    answer = stdout2.read().decode('utf-8', errors='replace').strip()
    
    # Cleanup
    stdin3, stdout3, stderr3 = client.exec_command(f'rm -f {out_file}', timeout=5)
    stdin3.close()
    
    return answer

# Test all 3 images
tests = [
    ('dog.jpg', 'What animal is in this image?'),
    ('cat.jpg', 'What animal is in this image?'),
    ('test-1.jpeg', 'Describe this image.'),
]

for img, prompt in tests:
    print(f'\n=== {img} ===')
    print(f'Q: {prompt}')
    answer = test_image(img, prompt)
    if answer:
        print(f'A: {answer[:500]}')
    else:
        print('A: [no answer generated]')

client.close()
