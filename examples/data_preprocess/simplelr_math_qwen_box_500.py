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
Preprocess the GSM8k dataset to parquet format
"""

import re
import os
import datasets
from datasets import Dataset
import json
from verl.utils.hdfs_io import copy, makedirs
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='simplelr_math_35')
    parser.add_argument('--train_data_path', default='math3to5_train.json')
    parser.add_argument('--test_data_path', default='math500_test.json')

    args = parser.parse_args()

    data_source = 'qwen_box_math_35'

    def gen_from_jsonl(path):
        return json.loads(open(path, "r", encoding="utf-8").read())
    
    train_dataset = Dataset.from_generator(gen_from_jsonl, gen_kwargs={'path': args.train_data_path})
    #test_dataset = Dataset.from_generator(gen_from_jsonl, gen_kwargs={'path': args.test_data_path})
    # 修改测试数据加载部分为：
    test_data = gen_from_jsonl(args.test_data_path)
    test_dataset = Dataset.from_list(test_data)
    test_dataset = test_dataset.shuffle().select(range(500)) 

    #instruction_following = "Please reason step by step, and put your final answer within \\boxed{}."

    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            question_raw = example.pop('question')
            question = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n" + "<|im_start|>user\n"+question_raw + "\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n" + "<|im_start|>assistant\n"
            answer = example.pop('answer')
            question_level = example.pop('level')
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

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    local_dir = args.local_dir
    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))
