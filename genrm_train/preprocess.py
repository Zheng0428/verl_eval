import json
import argparse
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def read_json(file_path):
    """Read a JSONL file and return a list of JSON objects."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def create_prompt(original_problem, ground_truth, extracted_answer):
    """Create a prompt for mathematical equivalence checking."""
    prompt = '''A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. User: 
Your task is to determine if the **Extracted Answer** is mathematically equivalent to the **Ground Truth Answer**.
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
   - Respond with \\boxed{0} if the answers are **not** equivalent.

Assistant:<think>
'''
    
    # Replace the placeholders with actual values
    prompt = prompt.replace("{Original Problem}", original_problem)
    prompt = prompt.replace("{Ground Truth Answer}", ground_truth)
    prompt = prompt.replace("{Extracted Answer}", extracted_answer)
    
    return prompt


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


def extract_raw_question(formatted_question):
    """Extract the raw question from a formatted question string."""
    try:
        # Find the part between user tag and the instruction
        user_start = formatted_question.find("<|im_start|>user\n")
        if user_start == -1:
            logging.warning("Could not find user start tag in question")
            return formatted_question
            
        # Get the content after the user tag
        content_start = user_start + len("<|im_start|>user\n")
        
        # Find where the instruction starts
        instruction_start = formatted_question.find("\nPlease reason step by step", content_start)
        if instruction_start == -1:
            # If no instruction found, look for the end tag
            instruction_start = formatted_question.find("<|im_end|>", content_start)
            if instruction_start == -1:
                # If no end tag either, return the rest of the string
                return formatted_question[content_start:]
        
        # Extract the raw question
        raw_question = formatted_question[content_start:instruction_start]
        return raw_question.strip()
    except Exception as e:
        logging.error(f"Error extracting raw question: {e}")
        return formatted_question

def preprocess_data(input_file, output_file):
    """Preprocess the data from input JSONL file and write to output JSONL file."""
    logging.info(f"Reading data from {input_file}")
    data = read_json(input_file)
    
    processed_data = []
    for i, item in enumerate(data):
        try:
            original_problem = item.get("Original Problem", "")
            
            # Extract the raw question from the formatted original problem
            raw_question = extract_raw_question(original_problem)
            
            ground_truth = item.get("Ground Truth Answer", "")
            extracted_answer = item.get("Extracted Answer", "")
            
            if not all([original_problem, ground_truth, extracted_answer]):
                logging.warning(f"Missing required fields in item {i+1}")
                continue
            
            # Use the raw question instead of the original problem
            prompt = create_prompt(raw_question, ground_truth, extracted_answer)
            rollout_prompt = create_rollout_prompt(raw_question, ground_truth, extracted_answer)
            item["raw_question"] = raw_question
            processed_item = {
                "genrm_prompt": prompt,
                "rollout_prompt": rollout_prompt,
                "extra_info": item
            }
            
            processed_data.append(processed_item)
            
        except Exception as e:
            logging.error(f"Error processing item {i+1}: {e}")
    
    logging.info(f"Processed {len(processed_data)} items")
    
    # Create directory if it doesn't exist
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write to output file json
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, indent=4)
    
    logging.info(f"Preprocessed data saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Preprocess data for LLM generation")
    parser.add_argument("--input", type=str, required=True, help="Path to input JSONL file")
    parser.add_argument("--output", type=str, required=True, help="Path to output JSONL file")
    
    args = parser.parse_args()
    
    preprocess_data(args.input, args.output)

if __name__ == "__main__":
    main()
