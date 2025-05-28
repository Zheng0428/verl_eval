# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the ORZ dataset to parquet format
"""

import os
from datasets import Dataset, load_dataset
import json
from functools import partial
import argparse


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        data = f.readlines()  
    return [json.loads(line) for line in data]

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def process_fn_qwen(example, idx, split='train', data_source='simplelr'):
    for key in ["question", "problem", "Question", "input"]:
        if key in example:
            question_raw = example[key]
            break
    answer = example['answer']
    
    question = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n" + "<|im_start|>user\n"+ question_raw + "\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n" + "<|im_start|>assistant\n"
    # answer = example.pop('ground_truth_answer')
    question_level = 1
    data = {
        "data_source": data_source,
        "prompt": [{
            "role": "user",
            "content": question,
        }],
        "ability": "math",
        "reward_model": {
            "style": "rule",
            "ground_truth": answer
        },
        "extra_info": {
            'split': split,
            'index': idx,
            'answer': answer,
            "question": question_raw,
            'level': question_level
        }
    }
    return data

def process_fn(example, idx, split='train', data_source='simplelr'):
    # assert len(example) == 2
    # human_message = example[0]
    # assistant_message = example[1]
    
    # assert human_message['from'].lower() == 'human'
    # question_raw = human_message['value']
    # assert assistant_message['from'].lower() == 'assistant'
    # answer = assistant_message['ground_truth']['value']
    assert len(example['prompt']) == 1
    question_raw = example['prompt'][0]['content']
    
    example['extra_info']['org_data_source'] = example['data_source']

    question = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n" + "<|im_start|>user\n"+question_raw + "\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n" + "<|im_start|>assistant\n"
    
    # Handle the case where ground_truth is a string representation of a JSON list
    ground_truth = example['reward_model']['ground_truth']
    ground_truth_list = json.loads(ground_truth)
    assert isinstance(ground_truth_list, list) and len(ground_truth_list) > 0
    answer = ground_truth_list[0]
    # if isinstance(ground_truth, str):
    #     try:
    #         # Try to parse the string as JSON
    #         ground_truth_list = json.loads(ground_truth)
    #         if isinstance(ground_truth_list, list) and len(ground_truth_list) > 0:
    #             answer = ground_truth_list[0]
    #         else:
    #             # If not a non-empty list after parsing, use the original string
    #             answer = ground_truth
    #     except json.JSONDecodeError:
    #         # If it's not valid JSON, use the original string
    #         answer = ground_truth
    # elif isinstance(ground_truth, list) and len(ground_truth) > 0:
    #     # If it's already a list, use the first element
    #     answer = ground_truth[0]
    # else:
    #     # Fallback to the original value
    #     answer = ground_truth
    
    question_level = 1
    data = {
        "data_source": data_source,
        "prompt": [{
            "role": "user",
            "content": question,
        }],
        "ability": "math",
        "reward_model": {
            "style": "rule",
            "ground_truth": answer,
            
        },
        "extra_info": {
            'split': split,
            'index': example['extra_info']['index'],
            'answer': answer,
            "question": question_raw,
            'level': question_level,
            "model_difficulty": example['extra_info']['model_difficulty']
        }
    }
    return data
    
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default='simplelr_math_35')
    parser.add_argument('--train_data_path', default=None)
    parser.add_argument('--test_data_paths', default=None )

    args = parser.parse_args()

    output_dir = args.output_dir
    save_dir = os.path.join(output_dir, "simplelr_skywork")
    # if args.train_data_path is not None:
    train_data =load_dataset("Skywork/Skywork-OR1-RL-Data", split='math')
    # train_dataset = Dataset.from_list(train_data)
    make_map_fn_train = partial(process_fn, data_source="simplelr_skywork", split='train')
    train_dataset  = [ make_map_fn_train(example, idx) for idx, example in enumerate(train_data)]
    train_dataset = Dataset.from_list(train_dataset)
    train_dataset.to_parquet(os.path.join(save_dir, 'train.parquet'))
    
            
    # if args.test_data_paths is not None:
    #     test_data_paths = args.test_data_paths.split(',')
    #     test_data = []
    #     for test_data_path in test_data_paths:
    #         if test_data_path.endswith('.json'):
    #             data = load_json(test_data_path)
    #         else:
    #             data = load_jsonl(test_data_path)
    #         make_map_fn_test = partial(process_fn_qwen, data_source="simplelr_"+test_data_path.split('/')[-1].split('.')[0], split='test')
    #         test_data.extend([make_map_fn_test(example, idx) for idx, example in enumerate(data)])
    #     test_dataset = Dataset.from_list(test_data)
    #     test_dataset.to_parquet(os.path.join(save_dir, 'test.parquet'))

