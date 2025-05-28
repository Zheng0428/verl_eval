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
        
        
def create_prompt(original_problem, ground_truth, extracted_answer):
    """Create a prompt for mathematical equivalence checking."""
    prompt = '''Your task is to determine if the **Extracted Answer** is mathematically equivalent to the **Ground Truth Answer**.
**Question**
{Original Problem}
**Ground Truth Answer:**
{Ground Truth Answer}
**Extracted Answer:**
{Extracted Answer}
Please follow these steps clearly:
1. **Review the Question and Ground Truth Answer carefully.**
2. **Compare the Extracted Answer with the Ground Truth Answer.**
3. **Explain step-by-step** whether or not they express the same meaning or information.
4. **Provide your final decision clearly** at the end:
   - Respond with \\boxed{1} if the answers are equivalent.
   - Respond with \\boxed{0} if the answers are **not** equivalent.'''
    
    # Replace the placeholders with actual values
    prompt = prompt.replace("{Original Problem}", original_problem)
    prompt = prompt.replace("{Ground Truth Answer}", ground_truth)
    prompt = prompt.replace("{Extracted Answer}", extracted_answer)
    
    return prompt


def main():
    input_file = "genrm_train/data/skywork/validation_data.json"
    output_file = "genrm_train/data/skywork/validation_data_processed.jsonl"
    
    data = read_json(input_file)
    
    for item in data: 
        extra_info = item.get("extra_info", {})
        original_problem = extra_info.get("raw_question", "")
        ground_truth = extra_info.get("Ground Truth Answer", "")
        extracted_answer = extra_info.get("Extracted Answer", "")
        prompt = create_prompt(original_problem, ground_truth, extracted_answer)
        item["problem"] = prompt
        item['answer'] = extra_info['4o Reward Score']
        del item['extra_info']
        del item['genrm_prompt']
        del item['rollout_prompt']
        print(item)
    write_jsonl(data, output_file)

if __name__ == "__main__":
    main()
    

   