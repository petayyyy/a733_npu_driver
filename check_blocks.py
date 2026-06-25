import json
for b in [0,1,2]:
    p = f'work/ai-sdk/ZIFENG278-ai-sdk/models/qwen25_05b_w32_block{b}/wksp/qwen25_05b_w32_block{b}_int16_nbg_unify/nbg_meta.json'
    try:
        d = json.load(open(p))
        for k,v in d.get('Outputs',{}).items():
            print(f'block{b}: {k} shape={v.get("shape")}')
    except Exception as e:
        print(f'block{b}: {e}')
