import json
import os 
import argparse
import random
import string

def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data
    
    
def write_jsonl(data, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in data:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')
        
        
def create_rollout_prompt_wo_q(original_problem, ground_truth, extracted_answer):
    """Create a prompt for rollout."""
    prompt = '''Your task is to determine if the **Extracted Answer** is mathematically equivalent to the **Ground Truth Answer**.
**Ground Truth Answer:**
{Ground Truth Answer}
**Extracted Answer:**
{Extracted Answer}
- If **Extracted Answer** and **Ground Truth Answer** are mathematically equivalent, respond with \\boxed{1}
- If they are not mathematically equivalent, or if the **Extracted Answer** is nonsensical (e.g., a random string), respond with \\boxed{0}
'''

    # Replace the placeholders with actual values
    prompt = prompt.replace("{Ground Truth Answer}", ground_truth)
    prompt = prompt.replace("{Extracted Answer}", extracted_answer)
    
    return prompt

def create_rollout_prompt_w_q(original_problem, ground_truth, extracted_answer,):
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



def create_xverifier_prompt(original_problem, ground_truth, extracted_answer, llm_output):
    PROMPT = '''You are a diligent and precise assistant tasked with evaluating the correctness of responses. You will receive a question, an output sentence, and the correct answer. Your task is to determine if the output sentence accurately answers the question based on the provided correct answer. Respond with either [Correct] or [Incorrect].
-
Special considerations:

1. **Multiple Answers**: If the output contains multiple answers, evaluate whether later answers modify or correct earlier ones. In such cases, compare the final answer with the correct answer. If the final answer is unclear or incorrect, respond with [Incorrect].

2. **Mathematical Problems**: If the formats differ but the answers are mathematically equivalent, respond with [Correct].

3. **Explicit Options**: If the question provides explicit candidate answers, the output will be considered correct if it clearly indicates the correct option's code or the correct option's content.

4. **No Explicit Options**: If the question does not provide explicit options, the output must align with the correct answer in content and meaning to be considered [Correct].
-

Question: """{question}"""

Output sentence: """{output}"""

Correct answer: {answer}

Judgement:
'''
    
    prompt = PROMPT.format(question=original_problem, output=llm_output, answer=ground_truth)
    return prompt

def create_xverifier_prompt_w_pred(original_problem, ground_truth, extracted_answer, llm_output):
    PROMPT = '''You are a diligent and precise assistant tasked with evaluating the correctness of responses. You will receive a question, an output sentence, and the correct answer. Your task is to determine if the output sentence accurately answers the question based on the provided correct answer. Respond with either [Correct] or [Incorrect].
-
Special considerations:

1. **Multiple Answers**: If the output contains multiple answers, evaluate whether later answers modify or correct earlier ones. In such cases, compare the final answer with the correct answer. If the final answer is unclear or incorrect, respond with [Incorrect].

2. **Mathematical Problems**: If the formats differ but the answers are mathematically equivalent, respond with [Correct].

3. **Explicit Options**: If the question provides explicit candidate answers, the output will be considered correct if it clearly indicates the correct option's code or the correct option's content.

4. **No Explicit Options**: If the question does not provide explicit options, the output must align with the correct answer in content and meaning to be considered [Correct].
-

Question: """{question}"""

Output sentence: """{output}"""

Correct answer: {answer}

Judgement:
'''
    
    prompt = PROMPT.format(question=original_problem, output=extracted_answer, answer=ground_truth)
    return prompt


def create_tigerverifier_prompt(original_problem, ground_truth, extracted_answer, llm_output):
    prompt = (
    f"User: ### Question: {original_problem}\n\n"
    f"### Ground Truth Answer: {ground_truth}\n\n"
    f"### Student Answer: {extracted_answer}\n\n"
    "For the above question, please verify if the student's answer is equivalent to the ground truth answer.\n"
    "Do not solve the question by yourself; just check if the student's answer is equivalent to the ground truth answer.\n"
    "If the student's answer is correct, output \"Final Decision: Yes\". If the student's answer is incorrect, output \"Final Decision: No\". Assistant:"
)
    return prompt






def load_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    return data

def main():
    parser = argparse.ArgumentParser(description='Create prompts for mathematical equivalence checking')
    parser.add_argument('--input_file', type=str, required=True, help='Path to input JSONL file')
    parser.add_argument('--output_file', type=str, required=True, help='Path to output JSONL file')
    # parser.add_argument('--add_question', action='store_true', help='Include question in the prompt')
    parser.add_argument('--prompt_format', type=str, required=True, choices=['genrm_w_q', 'genrm_wo_q', "xverifier", "xverifier_w_pred", "tigerverifier"], help='Prompt format')
    
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    data = load_jsonl(args.input_file)
    
    new_data = []
    index = 0 
    for item in data: 
        gt = item['answer']
        for pred_idx, (pred, gpt_score, rule_based_score, finish_reason, codes, gpt_judge) in enumerate(zip(item['pred'], item['gpt_correct'], item['score'], item['finish_reason'], item['code'], item['gpt_judge'])):
            #v0
            pred = "{"
            #v1 
            # set pred as random string of length 1000
            pred = ''.join(random.choices(string.ascii_letters + string.digits, k=1000))
            #v2
            pred = "correct"
            # v3
            pred = "true"
            if rule_based_score:
                continue
            if args.prompt_format == "genrm_w_q":
                prompt = create_rollout_prompt_w_q(item['question'], gt, pred)
            elif args.prompt_format == "genrm_wo_q":
                prompt = create_rollout_prompt_wo_q(item['question'], gt, pred)
            elif args.prompt_format == "xverifier":
                prompt = create_xverifier_prompt(item['question'], gt, pred, codes)
            elif args.prompt_format == "xverifier_w_pred":
                prompt = create_xverifier_prompt_w_pred(item['question'], gt, pred, codes)
            elif args.prompt_format == "tigerverifier":
                prompt = create_tigerverifier_prompt(item['question'], gt, pred, codes)
            new_item = {
                'idx': index,
                'problem': prompt,
                'answer': int(gpt_score),
                # 'pred_idx': pred_idx,
                'ori_rollout_info': {
                    "org_idx": item['idx'],
                    'pred_idx': pred_idx,
                    "question": item['question'],
                    'pred': pred,
                    'gt': gt,
                    'rule_based_score': rule_based_score,
                    'finish_reason': finish_reason,
                    'code': codes,
                    'gpt_judge': gpt_judge,
                }
            }
            index += 1
            new_data.append(new_item)
            
    write_jsonl(new_data, args.output_file)
    print(f"Processed {len(data)} data items with {index} total comparisons")
    print(f"Output written to {args.output_file}")

if __name__ == "__main__":
    main()
    

   