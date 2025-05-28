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

from verl.utils.hdfs_io import copy, makedirs
import argparse


python_pattern = r"```python[ \t]*[\r\n]+(.*?)[ \t]*[\r\n]+```"
python_re = re.compile(python_pattern, re.DOTALL | re.IGNORECASE)

def python_extract(text: str) -> str:
    match = python_re.search(text)
    if match:
        return match.group(1)
    else:
        return ""


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='~/data/opc_sft_stage2')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--test_size', type=int, default=400, help='Number of samples to use for test set')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for splitting')
    parser.add_argument('--cot', action='store_true', help='Whether to use the CoT prompt')

    args = parser.parse_args()

    data_source = 'OpenCoder-LLM/opc-sft-stage2'

    dataset = datasets.load_dataset(data_source, 'educational_instruct')

    full_dataset = dataset['train']
    split_dataset = full_dataset.train_test_split(test_size=args.test_size, seed=args.seed)
    train_dataset = split_dataset['train'] 
    test_dataset = split_dataset['test']

    if args.cot:
        data_source += '_cot'

    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            question_raw = example.pop('instruction')
            entry_point = example.pop('entry_point')

            instruction_following = f'You need to generate a python function with entry point {entry_point} in a self-contained code snippet in a markdown code block.'
            if args.cot:
                instruction_following += '\nYou need first write a step-by-step outline and then write the code.'

            question = question_raw + ' ' + instruction_following

            answer_raw = example.pop('output')
            solution = example.pop('code')
            test_cases = "\n".join(example.pop('testcase'))
            data = {
                "data_source": data_source,
                "prompt": [
                    # TODO: add system prompt
                    # {
                    #     "role": "system",
                    #     "content": SYSTEM_PROMPT
                    # },
                    {
                    "role": "user",
                    "content": question,
                    }
                ],
                "ability": "code",
                "reward_model": {
                    "style": "rule",
                    "canonical_solution": solution,
                    "ground_truth": test_cases
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'answer': answer_raw,
                    "question": question_raw,
                }
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir
    if args.cot:
        local_dir += '_cot'
        if hdfs_dir is not None:
            hdfs_dir += '_cot'

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
