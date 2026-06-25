import json
for n in ['embed','block0','final']:
    p = f'work/ai-sdk/ZIFENG278-ai-sdk/models/qwen25_05b_w32_{n}/wksp/qwen25_05b_w32_{n}_int16_nbg_unify/nbg_meta.json'
    try:
        d = json.load(open(p))
        for k,v in d.get('Outputs',{}).items():
            shape = v.get('shape',[])
            q = v.get('quantize',{})
            print(f'{n}: {k} shape={shape} qtype={q.get("qtype")} fl={q.get("fl")}')
    except Exception as e:
        print(f'{n}: {e}')
