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

execution_prompt = """\
Answer the given coding question. You must conduct reasoning inside <think> and </think> first before you can finally output the final program. During the thinking, you can test your program by writing it inside <python> and </python> tags. The code will be executed, and the terminal output (standard output and standard error) will be returned between <output> and </output>. Each program between <python> and </python> tags are independent program. You can run Python code as many times as you want. If you find no further code execution needed, you can then give the final program in a markdown code block like this: ```python\nyour code here\n```. The final program will be evaluated against the test cases. If the final program passes all the test cases, you will get a reward. If the final program fails any of the test cases, you will get a penalty.
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='~/data/acecoder')
    parser.add_argument('--hdfs_dir', default=None)

    args = parser.parse_args()

    data_source = 'CodeDPO/AceCoderV2-mini-processed'

    dataset = datasets.load_dataset(data_source, split='train')

    # 500 examples for testing
    
    dataset = dataset.train_test_split(test_size=500, seed=42)
    train_dataset = dataset['train']
    test_dataset = dataset['test']

    instruction_following = "Let's think step by step and generate the final program in a markdown code block like this: ```python\nyour code here\n```."

    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            question_raw = example.pop('question')

            # question = question_raw + ' ' + instruction_following
            question = question_raw + '\n\n' + execution_prompt
            
            tests = example.pop('tests')
            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": question,
                }],
                "ability": "code",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": tests
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'id': example['id'],
                    "question": question_raw,
                }
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
