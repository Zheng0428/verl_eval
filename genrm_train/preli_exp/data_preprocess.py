import json
import os
import pandas as pd

import argparse 
from datasets import load_dataset
import random

def load_json(data_path):
    with open(data_path, "r") as f:
        data = json.load(f)
    return data

def load_jsonl(data_path):
    with open(data_path, "r") as f:
        data = [json.loads(line) for line in f]
    return data

def load_orz_data(data_path):
    data = load_json(data_path)
    file_name = os.path.basename(data_path)
    processed_data = []
    for idx, item in enumerate(data):
        human = item[0]['value']
        assistant = item[1]['ground_truth']['value']
        processed_data.append({
            "idx": idx,
            "problem": human,
            "answer": assistant,
            "extra_info": {
                "source": f"orz",
                "file_name": file_name
            }
        })
    return processed_data

def load_math_data(data_path):
    data = load_jsonl(data_path)
    file_name = os.path.basename(data_path)
    processed_data = []
    for idx, item in enumerate(data):
        problem = item['problem']
        answer = item['answer']
        processed_data.append({
            "idx": idx,
            "problem": problem,
            "answer": answer,
            "extra_info": {
                "source": f"math",
                "level": item['level'],
                "solution": item['solution'],
                "subject": item['subject'],
                "unique_id": item['unique_id'],
                "file_name": file_name
            }
        })
    return processed_data

def load_skywork_data(data_path):
    data = load_dataset(data_path, split="math")
    file_name = os.path.basename(data_path)
    processed_data = []
    for idx, item in enumerate(data):
        problem = item['prompt'][0]['content']
        answer = item['reward_model']['ground_truth']
        answer= json.loads(answer)[0]
        
        processed_data.append({
            "idx": idx,
            "problem": problem,
            "answer": answer,
            "extra_info": {
                "source": f"skywork",
                "data_source": item['data_source'],
                "file_name": file_name
            }
        })
    return processed_data

def load_deepscaler_data(data_path):
    data = load_dataset(data_path, split="train")
    file_name = os.path.basename(data_path)
    processed_data = []
    for idx, item in enumerate(data):
        problem = item['problem']
        answer = item['answer']
        processed_data.append({
            "idx": idx,
            "problem": problem,
            "answer": answer,
            "extra_info": {
                "source": f"deepscaler",
                "file_name": file_name,
                "official_solution": item['solution'],
            }
        })
    return processed_data
        
def load_data(data_path):
    if "orz" in data_path.lower():
        data = load_orz_data(data_path)
    elif "math" in data_path.lower():
        data = load_math_data(data_path=data_path)
    elif "skywork" in data_path.lower():
        data = load_skywork_data(data_path)
    elif "deepscaler" in data_path.lower():
        data = load_deepscaler_data(data_path)
    else:
        raise NotImplementedError(f"Data {data_path} is not supported")
    return data

def save_jsonl(data, save_path):
    with open(save_path, "w") as f:
        for item in data:
            json.dump(item, f)
            f.write("\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--train_size", type=int, default=None)
    parser.add_argument("--val_size", type=int, default=None)
    args = parser.parse_args()
    data = load_data(args.data_path)  
    save_jsonl(data, args.save_path)
    random.shuffle(data)
    if args.train_size is not None:
        train_data = data[:args.train_size]
    if args.val_size is not None:
        val_data = data[args.train_size:args.train_size+args.val_size]
    save_jsonl(train_data, args.save_path.replace(".jsonl", "_size_{}_train.jsonl".format(args.train_size)))
    save_jsonl(val_data, args.save_path.replace(".jsonl", "_size_{}_val.jsonl".format(args.val_size)))

if __name__ == "__main__":
    main()