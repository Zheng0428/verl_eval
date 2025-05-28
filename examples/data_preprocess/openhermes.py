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
Preprocess the OpenHermes dataset to parquet format
"""

import os
import argparse
from datasets import load_dataset, Dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='openhermes')
    args = parser.parse_args()

    data_source = 'openhermes'
    dataset = load_dataset('teknium/OpenHermes-2.5')
    
    # Split dataset into train and test if not already split
    if 'test' not in dataset:
        dataset = dataset['train'].train_test_split(test_size=0.1)
    
    train_dataset = dataset['train']
    test_dataset = dataset['test'] if 'test' in dataset else dataset['validation']

    def make_map_fn(split):
        def process_fn(example, idx):
            # Extract the conversation
            conversation = example.get('conversations', [])
            
            if not conversation:
                return None
            
            # Extract question from the first human message
            human_messages = [msg for msg in conversation if msg.get('from') == 'human']
            if not human_messages:
                return None
                
            question_raw = human_messages[0].get('value', '')
            
            # Extract answer from the first assistant message
            assistant_messages = [msg for msg in conversation if msg.get('from') == 'gpt']
            if not assistant_messages:
                return None
                
            answer = assistant_messages[0].get('value', '')
            
            data = {
                "data_source": data_source,
                "question": question_raw,
                "answer": answer,
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
                }
            }
            return data

        return process_fn

    # Apply the processing function and filter out None values
    train_processed = train_dataset.map(
        function=make_map_fn('train'), 
        with_indices=True, 
        remove_columns=train_dataset.column_names
    )
    train_processed = train_processed.filter(lambda x: x is not None)
    
    test_processed = test_dataset.map(
        function=make_map_fn('test'), 
        with_indices=True,
        remove_columns=test_dataset.column_names
    )
    test_processed = test_processed.filter(lambda x: x is not None)

    # Create the local directory if it doesn't exist
    os.makedirs(args.local_dir, exist_ok=True)
    
    # Save the processed datasets as parquet files
    train_processed.to_parquet(os.path.join(args.local_dir, 'train.parquet'))
    test_processed.to_parquet(os.path.join(args.local_dir, 'test.parquet'))
    
    print(f"Processed {len(train_processed)} training examples and {len(test_processed)} test examples")
    print(f"Saved to {args.local_dir}")