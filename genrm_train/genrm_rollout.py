import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any
import time
import torch
from tqdm import tqdm
from vllm import LLM, SamplingParams

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
import re
def extract_last_boxed(text):
    """
    提取 LaTeX 文本中最后一个 \boxed 命令中的内容
    
    返回:
    - str: 最后一个 \boxed 中的内容。如果没有找到则返回 None
    """
    pattern = r'\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}'
    
    # 找到所有匹配
    matches = list(re.finditer(pattern, text))
    
    # 如果找到匹配，返回最后一个的内容
    if matches:
        return matches[-1].group(0)
    return None


def read_json(file_path):
    """Read data from a JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def write_json(data, output_file):
    """Write data to a JSON file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)

def batch_data(data, batch_size):
    """Divide data into batches of specified size."""
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]

def rollout_with_vllm(
   args
):
    """
    Generate completions for prompts using vLLM.
    
    Args:
        model_path: Path to the LLM model
        input_file: Path to input JSON file with prompts
        output_file: Path to save outputs
        batch_size: Number of prompts to process at once
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        max_tokens: Maximum number of tokens to generate
        num_gpus: Number of GPUs to use (None for all available)
        gpu_memory_utilization: GPU memory utilization target (0.0 to 1.0)
    """
    model_path = args.model
    input_file = args.input
    output_file = args.output
    batch_size = args.batch_size
    temperature = args.temperature
    top_p = args.top_p
    max_tokens = args.max_tokens
    # num_gpus = args.num_gpus
    gpu_memory_utilization = args.gpu_memory_utilization
    num_rollouts = args.num_rollouts
    
    logging.info(f"Loading data from {input_file}")
    data = read_json(input_file)
    data = data[:args.num_examples]
    logging.info(f"Loaded {len(data)} examples")
    
    # Create output directory if it doesn't exist
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Initialize vLLM model
    logging.info(f"Loading model from {model_path}")
    start_time = time.time()
    
    # Set tensor parallel size if num_gpus is specified
    # tensor_parallel_size = num_gpus if num_gpus is not None else 1
    
    llm = LLM(
        model=model_path,
        # tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=True,
    )
    
    logging.info(f"Model loaded in {time.time() - start_time:.2f} seconds")
    
    # Set sampling parameters
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        n=num_rollouts,
    )
    
    total_processed = 0
    processed_data = []
    
    # Process data in batches
    for batch_idx, batch in enumerate(batch_data(data, batch_size)):
        batch_prompts = [item["rollout_prompt"] for item in batch]
        
        logging.info(f"Processing batch {batch_idx+1}, examples {total_processed+1}-{total_processed+len(batch)}")
        
        # Generate completions
        start_time = time.time()
        outputs = llm.generate(batch_prompts, sampling_params)
        end_time = time.time()
        
        # Process and save results
        for i, output in enumerate(outputs):
            item_idx = batch_idx * batch_size + i
            if item_idx < len(data):  # Safety check
                item = data[item_idx].copy()
                item["completion"] = [
                    {
                        'text': output.outputs[j].text,
                        'GenRM reward_score': extract_last_boxed(output.outputs[j].text) if extract_last_boxed(output.outputs[j].text) is not None else ""
                    }
                    for j in range(num_rollouts)
                ]
                processed_data.append(item)
        
        total_processed += len(batch)
        logging.info(f"Batch completed in {end_time - start_time:.2f} seconds")
        
        # Optionally save intermediate results
        if (batch_idx + 1) % 10 == 0:
            temp_output = output_file.replace(".json", f"_temp_{batch_idx+1}.json")
            write_json(processed_data, temp_output)
            logging.info(f"Saved intermediate results to {temp_output}")
    
    # Save final results
    write_json(processed_data, output_file)
    logging.info(f"Processed {total_processed} examples")
    logging.info(f"Results saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Generate completions with vLLM")
    
    # Required arguments
    parser.add_argument("--model", type=str, required=True, help="Path or name of the LLM model")
    parser.add_argument("--input", type=str, required=True, help="Path to input JSON file")
    parser.add_argument("--output", type=str, required=True, help="Path to output JSON file")
    
    # Optional generation parameters
    parser.add_argument("--batch_size", type=int, default=100000, help="Batch size for generation")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p sampling parameter")
    parser.add_argument("--max_tokens", type=int, default=1024, help="Maximum tokens to generate")
    parser.add_argument("--num_rollouts", type=int, default=16, help="Number of rollouts")
    
    # Hardware parameters
    # parser.add_argument("--num_gpus", type=int, default=None, help="Number of GPUs to use (default: all available)")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85, help="GPU memory utilization (0.0 to 1.0)")
    parser.add_argument("--num_examples", type=int, default=100000000, help="Number of examples to generate")
    
    args = parser.parse_args()
    
    rollout_with_vllm(
        args
    )

if __name__ == "__main__":
    main()
