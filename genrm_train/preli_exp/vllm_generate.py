import argparse
import json
import os
import time
from pathlib import Path
from typing import List, Dict, Any

import torch
from vllm import LLM, SamplingParams
from tqdm import tqdm

from utils import load_jsonl, save_jsonl, set_seed, construct_prompt


def parse_args():
    parser = argparse.ArgumentParser(description="Generate responses using vLLM")
    
    # Model parameters
    parser.add_argument("--model", type=str, required=True, help="Model name or path")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Number of GPUs for tensor parallelism")
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "half", "float16", "bfloat16", "float", "float32"], 
                        help="Data type for model weights")
    
    # Input/output parameters
    parser.add_argument("--input_file", type=str, required=True, help="Path to the preprocessed JSONL data file")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the generated responses")
    
    # Sampling parameters
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling parameter")
    parser.add_argument("--max_tokens", type=int, default=1024, help="Maximum number of tokens to generate")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of samples to generate per prompt")
    
    # Prompt parameters
    parser.add_argument("--prompt_type", type=str, default="cot", help="Type of prompt to use")
    parser.add_argument("--num_query", type=int, default=10000000000, help="Number of query to use")
    parser.add_argument("--num_shots", type=int, default=0, help="Number of few-shot examples")
    parser.add_argument("--adapt_few_shot", action="store_true", help="Adapt few-shot examples based on data")
    
    # Batch size
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for generation")
    
    # Other
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    return parser.parse_args()


def prepare_prompts(data: List[Dict[Any, Any]], args):
    prompts = []
    for example in data:
        # Convert to the format expected by construct_prompt
        formatted_example = {
            "question": example["problem"],
            "gt_ans": example.get("gt_ans", "")  # May not exist in all datasets
        }
        
        # Determine data_name from extra_info
        extra_info = example.get("extra_info", {})
        data_name = extra_info.get("source", "math")  # Default to "math" if not specified
        
        prompt = construct_prompt(formatted_example, data_name, args)
        prompts.append(prompt)
    
    return prompts


def main():
    args = parse_args()
    set_seed(args.seed)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    
    print(f"Loading data from {args.input_file}")
    data = list(load_jsonl(args.input_file))
    print(f"Loaded {len(data)} examples")
    
    print(f"Initializing vLLM with model {args.model}")
    model = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,
        trust_remote_code=True,
    )
    
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        n=args.num_samples,
    )
    
    print("Preparing prompts")
    prompts = prepare_prompts(data, args)
    prompts = prompts[:args.num_query]
    # Generate responses in batches
    all_outputs = []
    num_batches = (len(prompts) + args.batch_size - 1) // args.batch_size
    
    print(f"Generating responses for {len(prompts)} prompts in {num_batches} batches")
    start_time = time.time()
    
    for i in tqdm(range(0, len(prompts), args.batch_size)):
        batch_prompts = prompts[i:i + args.batch_size]
        batch_indices = list(range(i, min(i + args.batch_size, len(prompts))))
        
        # Generate responses for the batch
        outputs = model.generate(batch_prompts, sampling_params)
        
        # Process the outputs
        for idx, output in zip(batch_indices, outputs):
            generated_texts = [o.text for o in output.outputs]
            
            # Create result with original data and generated responses
            result = data[idx].copy()
            result["prompt"] = prompts[idx]
            result["generations"] = generated_texts
            
            all_outputs.append(result)
    
    end_time = time.time()
    print(f"Generation completed in {end_time - start_time:.2f} seconds")
    
    # Save results
    print(f"Saving results to {args.output_file}")
    save_jsonl(all_outputs, args.output_file)
    print(f"Saved {len(all_outputs)} results")


if __name__ == "__main__":
    main()
