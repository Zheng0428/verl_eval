import json

def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data
    
    
def write_jsonl(data, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in data:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')
        
        
def create_rollout_prompt(original_problem, ground_truth, extracted_answer):
    """Create a prompt for rollout."""
    prompt = '''A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. User: 
Your task is to determine if the **Extracted Answer** is mathematically equivalent to the **Ground Truth Answer**.
**Ground Truth Answer:**
{Ground Truth Answer}
**Extracted Answer:**
{Extracted Answer}
- If **Extracted Answer** and **Ground Truth Answer** are mathematically equivalent, respond with \\boxed{1}
- If they are not mathematically equivalent, or if the **Extracted Answer** is nonsensical (e.g., a random string), respond with \\boxed{0}

Assistant:<think>
'''

    # Replace the placeholders with actual values
    prompt = prompt.replace("{Ground Truth Answer}", ground_truth)
    prompt = prompt.replace("{Extracted Answer}", extracted_answer)
    
    return prompt


def main():
    input_file = "genrm_train/data/skywork/validation_data.json"
    output_file = "genrm_train/data/skywork/validation_data_processed_wo_q.jsonl"
    
    data = read_json(input_file)
    
    for item in data: 
        extra_info = item.get("extra_info", {})
        original_problem = extra_info.get("raw_question", "")
        ground_truth = extra_info.get("Ground Truth Answer", "")
        extracted_answer = extra_info.get("Extracted Answer", "")
        prompt = create_rollout_prompt(original_problem, ground_truth, extracted_answer)
        item["problem"] = prompt
        item['answer'] = extra_info['4o Reward Score']
        del item['extra_info']
        del item['genrm_prompt']
        del item['rollout_prompt']
        print(item)
    write_jsonl(data, output_file)

if __name__ == "__main__":
    main()
    

   