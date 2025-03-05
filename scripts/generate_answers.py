import datasets

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

import os
import datasets
import fire
from verl.utils.hdfs_io import copy, makedirs
import argparse

from verl.utils.reward_score.math import remove_boxed, last_boxed_only_string
from llm_engines import LLMEngine
from verl.utils.reward_score import prime_math

def extract_solution(solution_str):
    return remove_boxed(last_boxed_only_string(solution_str))

def main(
    data_source='DigitalLearningGmbH/MATH-lighteval',
    n=2,
    temperature=0.6,
    max_tokens=2048,
    model_name='Qwen/Qwen2.5-0.5B-Instruct',
    n_gpu=1,
    local_dir='~/data/math_ct',
    hdfs_dir=None,
):
    
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    dataset = datasets.load_dataset(data_source, trust_remote_code=True)
    train_dataset = dataset['train'].select(range(1000))
    test_dataset = dataset['test'].select(range(200))
    train_dataset = train_dataset
    test_dataset = test_dataset
    llm = LLMEngine()
    llm.load_model(
        model_name=model_name,
        num_workers=n_gpu,
        num_gpu_per_worker=1,
        engine='vllm',
    )
    instruction_following = "Let's think step by step and output the final answer within \\boxed{}."
    
    train_messages = [x['problem'] + ' ' + instruction_following for x in train_dataset]
    test_messages = [x['problem'] + ' ' + instruction_following for x in test_dataset]
    train_outputs = llm.batch_call_model(model_name=model_name, batch_messages=train_messages, n=n, temperature=temperature, max_tokens=max_tokens)
    test_outputs = llm.batch_call_model(model_name=model_name, batch_messages=test_messages, n=n, temperature=temperature, max_tokens=max_tokens)
    train_dataset = train_dataset.add_column('output', train_outputs)
    test_dataset = test_dataset.add_column('output', test_outputs)
    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            question = example.pop('problem')
            answer = example.pop('solution')
            solution = extract_solution(answer)
            final_data = []
            for output in example['output']:
                
                ct_question = "Give the following problem and an answer, please judge whether the answer is correct or not: \nQuestion: " + question \
                    + "\nAnswer: " + output + "\nLet's analyze the answer step by step and output the final judgement within \\boxed{} with 'Correct' or 'Incorrect'."

                ground_truth = prime_math.compute_score(output, solution)[0]
                data = {
                    "data_source": data_source,
                    "prompt": [{
                        "role": "user",
                        "content": ct_question
                    }],
                    "ability": "math",
                    "reward_model": {
                        "style": "rule",
                        "ground_truth": "Correct" if ground_truth else "Incorrect"
                        # "ground_truth": ground_truth
                    },
                    "extra_info": {
                        'split': split,
                        'index': idx,
                        'output': output,
                        'solution': solution,
                    }
                }
                final_data.append(data)
            return {
                "batch": final_data
            }

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)
    
    flatten_train_dataset = []
    flatten_test_dataset = []
    for x in train_dataset:
        flatten_train_dataset += x['batch']
    for x in test_dataset:
        flatten_test_dataset += x['batch']
    train_dataset = datasets.Dataset.from_list(flatten_train_dataset)
    test_dataset = datasets.Dataset.from_list(flatten_test_dataset)
    

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))
    
    train_dataset.to_json("./train.jsonl")

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
    

if __name__ == '__main__':
    fire.Fire(main)