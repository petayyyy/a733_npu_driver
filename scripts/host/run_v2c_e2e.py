#!/usr/bin/env python3
"""V2c E2E: Run NPU + llama-cli hybrid, save results to file."""
import sys, os
sys.path.insert(0, r'C:\Users\ilyah\Documents\Work\a733_npu_driver\work\pydeps')
import paramiko

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('192.168.31.225', username='orangepi', password='orangepi', timeout=30)

# Upload shell scripts
sftp = client.open_sftp()
sftp.put(r'C:\Users\ilyah\AppData\Local\Temp\opencode\v2c_e2e.sh', '/tmp/v2c_e2e.sh')
sftp.close()

# Run on dog.jpg
results = {}
for img, prompt in [('dog.jpg', 'What animal is in this image?'), 
                     ('cat.jpg', 'What animal is in this image?'),
                     ('test-1.jpeg', 'Describe this image.')]:
    print(f'\n=== {img} ===')
    out_file = f'/tmp/v2c_{img}_result.txt'
    cmd = f'bash /tmp/v2c_e2e.sh {img} "{prompt}" > {out_file} 2>&1'
    stdin, stdout, stderr = client.exec_command(cmd, timeout=600)
    stdin.close()
    stdout.read()  # wait
    
    # Read result  
    stdin2, stdout2, stderr2 = client.exec_command(f'cat {out_file}', timeout=10)
    stdin2.close()
    content = stdout2.read().decode('utf-8', errors='replace')
    
    # Extract non-spinner lines
    answer_lines = []
    for line in content.split('\n'):
        line = line.strip()
        if len(line) > 20 and '?' not in line[:8] and '|' not in line[:5] and 'Loading' not in line and 'build' not in line and 'model' not in line and 'modalities' not in line and 'available' not in line and '/' not in line[:3]:
            answer_lines.append(line)
    
    answer = ' '.join(answer_lines)
    results[img] = answer
    print(f'  Answer: {answer[:300]}')

# Save all results
with open('work/generated/smolvlm_256m_v2c/v2c_e2e_results.txt', 'w', encoding='utf-8') as f:
    for img, answer in results.items():
        f.write(f'\n=== {img} ===\n{answer}\n')

print('\nResults saved to v2c_e2e_results.txt')
client.close()
