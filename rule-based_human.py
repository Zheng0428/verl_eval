import json
import pandas as pd

def load_jsonl(file_path):
    with open(file_path, 'r') as file:
        return [json.loads(line) for line in file]

def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)
    
input_path = "/Users/bytedance/Desktop/verl/genrm_train/preli_exp/reorg_gpt_labeled_data_new/DeepSeek-R1-Distill-Qwen-32B-new/rollout_deepscaler/size_1000_val_deepseek-r1_-1_seed0_t1.0_s0_e-1_generate_response_-1_reorg.jsonl"
input_path = "/Users/bytedance/Desktop/verl/genrm_train/preli_exp/reorg_gpt_labeled_data_new/DeepSeek-R1-Distill-Qwen-32B-new/rollout_skywork/size_1000_val_deepseek-r1_-1_seed0_t1.0_s0_e-1_generate_response_-1_reorg.jsonl"

data = load_jsonl(input_path)


for item in data:
    gt = item['answer']
    question = item['question']
    for gpt_judge, rule_based_judge, pred in zip(item['gpt_judge'], item['score'], item['pred']):
        if rule_based_judge:
            # if len(gpt_judge) != 0:
                
            print(f"question: {question}")
            print(f"gpt_judge: {gpt_judge}, rule_based_judge: {rule_based_judge}, pred: {pred}, gt: {gt}")
            print("-"*100)
    
            
