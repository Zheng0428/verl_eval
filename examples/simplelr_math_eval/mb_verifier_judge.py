from vllm import LLM, SamplingParams
import os
import json
import argparse
import random
import time
import logging
import re
import torch
import numpy as np
from tqdm import tqdm
from verl.utils.reward_score.qwen_math_eval_toolkit.parser import extract_answer as qwen_extract_answer

def load_jsonl(file_path):
    if not os.path.exists(file_path):
        return []
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]


def save_jsonl(data, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')


def construct_md_prompt(policy_output_data, judge_template):
    """
    Construct prompts for the verifier model based on policy outputs and the judge template.
    
    Args:
        policy_output_data: List of dictionaries containing policy outputs
        judge_template: Template type to use for the verifier prompts
        
    Returns:
        List of dictionaries containing the original data and verifier prompts
    """
    md_data = []
    for item in policy_output_data:
        prompt = item["question"]
        ground_truth = item["answer"]
        
        item_with_prompts = {
            "original_item": item,
            "prompts": [],
            "pred_indices": []
        }
        
        for i, pred in enumerate(item["pred"]):
            extracted_answer = pred
            
            # Create verifier prompt using the template from genRM.py
            if judge_template == "tiger-verifier":
                verifier_prompt = (
                    f"User: ### Question: {prompt}\n\n"
                    f"### Ground Truth Answer: {ground_truth}\n\n"
                    f"### Student Answer: {extracted_answer}\n\n"
                    "For the above question, please verify if the student's answer is equivalent to the ground truth answer.\n"
                    "Do not solve the question by yourself; just check if the student's answer is equivalent to the ground truth answer.\n"
                    "If the student's answer is correct, output \"Final Decision: Yes\". If the student's answer is incorrect, output \"Final Decision: No\". Assistant:"
                )
            elif judge_template == "r1_wo_question":
                model_eval_prompt = '''
Your task is to determine if the **Extracted Answer** is mathematically equivalent to the **Ground Truth Answer**.
**Ground Truth Answer:**
{ground_truth}
**Extracted Answer:**
{extracted_answer}
- If **Extracted Answer** and **Ground Truth Answer** are mathematically equivalent, respond with \\boxed{{1}}
- If they are not mathematically equivalent, or if the **Extracted Answer** is nonsensical (e.g., a random string), respond with \\boxed{{0}}
'''
                deepseek_system_prompt = 'A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. User: {input}\nAssistant:<think>\n'
                verifier_prompt = deepseek_system_prompt.format(
                    input=model_eval_prompt.format(
                        ground_truth=ground_truth, 
                        extracted_answer=extracted_answer
                    )
                )
            elif judge_template == "r1_with_question":
                model_eval_prompt = '''
Your task is to determine if the **Extracted Answer** is mathematically equivalent to the **Ground Truth Answer**.
**Question**
{original_problem}
**Ground Truth Answer:**
{ground_truth}
**Extracted Answer:**
{extracted_answer}
Please follow these steps clearly:
1. **Review the Question and Ground Truth Answer carefully.**
2. **Compare the Extracted Answer with the Ground Truth Answer.**
3. **Explain step-by-step** whether or not they express the same meaning or information.
4. **Provide your final decision clearly** at the end:
   - Respond with \\boxed{{1}} if the answers are equivalent.
   - Respond with \\boxed{{0}} if the answers are **not** equivalent.
'''
                deepseek_system_prompt = 'A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. User: {input}\nAssistant:<think>\n'
                verifier_prompt = deepseek_system_prompt.format(
                    input=model_eval_prompt.format(
                        original_problem=prompt,
                        ground_truth=ground_truth, 
                        extracted_answer=extracted_answer
                    )
                )
            elif judge_template == "qwen-boxed_wo_question":
                model_eval_prompt = '''
Your task is to determine if the **Extracted Answer** is mathematically equivalent to the **Ground Truth Answer**.
**Ground Truth Answer:**
{ground_truth}
**Extracted Answer:**
{extracted_answer}
- If **Extracted Answer** and **Ground Truth Answer** are mathematically equivalent, respond with \\boxed{{1}}
- If they are not mathematically equivalent, or if the **Extracted Answer** is nonsensical (e.g., a random string), respond with \\boxed{{0}}
'''
                qwen_instruct_template = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n" \
                "<|im_start|>user\n{input}\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n" \
                "<|im_start|>assistant\n"
                
                verifier_prompt = qwen_instruct_template.format(
                    input=model_eval_prompt.format(
                        ground_truth=ground_truth, 
                        extracted_answer=extracted_answer
                    )
                )
            elif judge_template == "qwen-boxed_with_question":
                model_eval_prompt = '''
Your task is to determine if the **Extracted Answer** is mathematically equivalent to the **Ground Truth Answer**.
**Question**
{original_problem}
**Ground Truth Answer:**
{ground_truth}
**Extracted Answer:**
{extracted_answer}
Please follow these steps clearly:
1. **Review the Question and Ground Truth Answer carefully.**
2. **Compare the Extracted Answer with the Ground Truth Answer.**
3. **Explain step-by-step** whether or not they express the same meaning or information.
4. **Provide your final decision clearly** at the end:
   - Respond with \\boxed{{1}} if the answers are equivalent.
   - Respond with \\boxed{{0}} if the answers are **not** equivalent.
'''
                qwen_instruct_template = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n" \
                "<|im_start|>user\n{input}\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n" \
                "<|im_start|>assistant\n"
                
                verifier_prompt = qwen_instruct_template.format(
                    input=model_eval_prompt.format(
                        original_problem=prompt,
                        ground_truth=ground_truth, 
                        extracted_answer=extracted_answer
                    )
                )
            else:
                raise ValueError(f"Unknown judge template: {judge_template}")
            
            item_with_prompts["prompts"].append(verifier_prompt)
            item_with_prompts["pred_indices"].append(i)
            
        md_data.append(item_with_prompts)
    
    return md_data


def extract_judgment(response, prompt_type):
    """
    Extract judgment from verifier response based on the prompt type.
    
    Args:
        response: The response text from the verifier model
        prompt_type: The type of prompt used for the verifier
        
    Returns:
        A dict with score (0 or 1) and extracted_answer (the relevant part of the response)
    """
    if prompt_type == "tiger-verifier":
        ext_re = r"Final Decision:\s*(yes|no|true|false)"
        match = re.search(ext_re, response, re.IGNORECASE)
        if match:
            extracted_answer = match.group(1).strip().lower()
            if extracted_answer.lower() in ["yes", "true"]:
                score = 1.0
            elif extracted_answer.lower() in ["no", "false"]:
                score = 0.0
            else:
                score = 0.0
        else:
            extracted_answer = ""
            score = 0.0
    elif prompt_type in ["r1_wo_question", "r1_with_question", "qwen-boxed_wo_question", "qwen-boxed_with_question"]:
        # For boxed answer formats, look for \boxed{0} or \boxed{1}
        score = 0.0
        extracted_answer = qwen_extract_answer(response, data_name="math")
        if extracted_answer.strip() == '1':
            score = 1.0
        elif extracted_answer.strip() == '0':
            score = 0.0
        else:
            score = 0.0
    else:
        raise ValueError(f"Unknown prompt type: {prompt_type}")
    
    return {"score": score, "extracted_answer": extracted_answer, "full_response": response}


def run_verifier(md_data, args):
    """
    Run the verifier model on the prompts.
    
    Args:
        md_data: List of dictionaries containing the original data and verifier prompts
        args: Command-line arguments
        
    Returns:
        Updated data with verifier judgments
    """
    # Flatten all prompts to process them in one batch
    all_prompts = []
    prompt_map = []  # Maps each prompt back to its original data item and pred index
    
    for i, item in enumerate(md_data):
        for j, prompt in enumerate(item["prompts"]):
            all_prompts.append(prompt)
            pred_idx = item["pred_indices"][j]
            prompt_map.append((i, j, pred_idx))
    
    # Initialize the LLM
    available_gpus = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    # Initialize the LLM
    llm = LLM(model=args.judge_model_path,
              tensor_parallel_size=len(available_gpus) // 1,
              pipeline_parallel_size=1,
              trust_remote_code=True,
              max_seq_len_to_capture= args.judge_max_tokens + 2000
              #max_seq_len_to_capture= 32000
              )
    
    # Generate responses for all prompts
    logging.info(f"Running verifier on {len(all_prompts)} prompts")
    stop_words = ["</s>", "<|im_end|>", "<|endoftext|>"]

    responses = llm.generate(
        all_prompts,
        SamplingParams(
                    temperature=args.judge_temperature,
                    top_p=args.judge_top_p,
                    max_tokens=args.judge_max_tokens,
                    n=1,
                    stop=stop_words,
                )
    )
    
    # Process the responses and add them to the original data
    for i, output in enumerate(responses):
        item_idx, prompt_idx, pred_idx = prompt_map[i]
        response_text = output.outputs[0].text
        
        # Extract judgment from the response
        judgment = extract_judgment(response_text, args.judge_template)
        
        # Add the judgment to the original item
        if "genrm_response" not in md_data[item_idx]:
            md_data[item_idx]['original_item']["genrm_response"] = [None] * len(md_data[item_idx]['original_item']["pred"])
            md_data[item_idx]['original_item']["genrm_reward_score"] = [None] * len(md_data[item_idx]['original_item']["pred"])
        
        md_data[item_idx]['original_item']["genrm_response"][pred_idx] = judgment["full_response"]
        md_data[item_idx]['original_item']["genrm_reward_score"][pred_idx] = judgment["score"]
    
    # Return the updated data (only the original items with added genrm_response)
    return [item['original_item'] for item in md_data]


def load_metrics(metrics_path):
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    return metrics

def update_metrics(updated_data, metrics_path):
    metrics = load_metrics(metrics_path)
    
    rule_based_reward_list = []
    hybrid_reward_list = []
    model_only_reward_list = []
    for item in updated_data:
        for rule, model in zip(item["score"], item["genrm_reward_score"]):
            hybrid = model > 0.0 or rule 
            rule_based_reward_list.append(rule)
            hybrid_reward_list.append(hybrid)
            model_only_reward_list.append(model > 0.0 )
        
        
    metrics["rule_based_reward"] = np.mean(rule_based_reward_list)
    metrics["hybrid_reward"] = np.mean(hybrid_reward_list)
    metrics['model_only_reward'] = np.mean(model_only_reward_list)
    metrics['rule_and_hybrid_gap'] = np.mean(hybrid_reward_list) - np.mean(rule_based_reward_list)
    return metrics

def save_metrics(metrics, metrics_path):
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_template", type=str, required=True)
    parser.add_argument("--policy_temperature", type=float, default=0.0)
    parser.add_argument("--policy_start", type=int, default=0)
    parser.add_argument("--policy_end", type=int, default=-1)
    parser.add_argument("--policy_seed", type=int, default=0)
    parser.add_argument("--policy_num_test_sample", type=int, default=1)
    
    parser.add_argument("--judge_model_path", type=str, required=True)
    parser.add_argument("--judge_template", type=str, required=True, 
                      help="Template for judge prompts: tiger-verifier, r1_wo_question, r1_with_question, etc.")
    parser.add_argument("--judge_temperature", type=float, default=0.0)
    parser.add_argument("--judge_top_p", type=float, default=1.0)
    parser.add_argument("--judge_max_tokens", type=int, default=2048)
    
    parser.add_argument("--data_names", default="gsm8k,math", type=str)
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    # parser.add_argument("--n_sampling", type=int, default=1)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--seed", type=int, default=0)
    # parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--log_file", type=str, default="verifier_judge.log")
    parser.add_argument("--cal_metrics_only", type=bool, default=False)
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(args.log_file),
            logging.StreamHandler()
        ]
    )
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Process each dataset
    data_names = args.data_names.split(",")
    for data_name in data_names:
        out_file_prefix = f"{args.split}_{args.policy_template}_{args.policy_num_test_sample}_seed{args.policy_seed}_t{args.policy_temperature}_s{args.policy_start}_e{args.policy_end}.jsonl"
        out_metrics_prefix = f"{args.split}_{args.policy_template}_{args.policy_num_test_sample}_seed{args.policy_seed}_t{args.policy_temperature}_s{args.policy_start}_e{args.policy_end}_metrics.json"
        policy_output_path = os.path.join(args.input_dir, f"{data_name}/{out_file_prefix}")
        policy_output_metrics_path = os.path.join(args.input_dir, f"{data_name}/{out_metrics_prefix}")
        logging.info(f"Processing {data_name} from {policy_output_path}")
        
        # Load policy output data
        policy_output_data = load_jsonl(policy_output_path)
        if len(policy_output_data) == 0:
            logging.warning(f"No policy output data found for {data_name}, skip it.")
            continue
        verifier_name = args.judge_model_path.split("/")[-1]
        new_out_file_prefix = out_file_prefix.replace(".jsonl", f"_verifier_{verifier_name}_t{args.judge_temperature}_p{args.judge_top_p}_m{args.judge_max_tokens}.jsonl")
        output_path = os.path.join(args.output_dir, f"{data_name}/{new_out_file_prefix}")
        new_out_metrics_path = output_path.replace(".jsonl", "_metrics.json")
        if os.path.exists(new_out_metrics_path):
            logging.warning(f"Verifier results already exist for {data_name}, skip it.")
            continue
        # Construct prompts for the verifier
        md_data = construct_md_prompt(policy_output_data, args.judge_template)
        
        # Run the verifier model
        # updated_data = run_verifier(md_data, args)
        if not args.cal_metrics_only:
            updated_data = run_verifier(md_data, args)
            save_jsonl(updated_data, output_path)
        else:
            updated_data = load_jsonl(output_path)
        
        metrics = update_metrics(updated_data, policy_output_metrics_path)
        # Save the results
        save_metrics(metrics, new_out_metrics_path)
        logging.info(f"Saved verifier results to {output_path}")
        
        # # Calculate and log some statistics
        # correct_count = 0
        # total_count = 0
        
        # for item in updated_data:
        #     for response in item.get("genrm_response", []):
        #         if response:  # Skip None values
        #             total_count += 1
        #             if response["score"] == 1.0:
        #                 correct_count += 1
        
        # if total_count > 0:
        #     accuracy = correct_count / total_count
        #     logging.info(f"Verification results for {data_name}: {correct_count}/{total_count} correct ({accuracy:.2%})")


if __name__ == "__main__":
    main()
