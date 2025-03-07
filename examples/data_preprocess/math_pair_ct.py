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
import random
from verl.utils.reward_score.math import remove_boxed, last_boxed_only_string
from llm_engines import LLMEngine
from verl.utils.reward_score import prime_math

def extract_solution(solution_str):
    return remove_boxed(last_boxed_only_string(solution_str))


pair_prompt="""Please provide a detailed comparative analysis of Answer 1 and Answer 2 based on the following criteria:

1. Mathematical/logical accuracy: Identify any calculation errors, logical flaws, or incorrect assumptions
2. Methodology: Evaluate the approach used in each answer
3. Completeness: Assess whether all aspects of the problem were addressed
4. Clarity: Consider how clearly the reasoning is presented

For each criterion, explain:
- The strengths and weaknesses of both answers
- Specific examples from each answer to support your analysis
- How the differences impact the overall correctness

After your detailed analysis, provide your final verdict as either:
\\boxed{Answer 1 is correct} or \\boxed{Answer 2 is correct}

If both answers contain errors or if they're both correct but through different approaches, explain this nuance before providing your verdict on which is ultimately more accurate or complete.

Important: Your analysis must go beyond simply repeating or paraphrasing the answers. Focus on meaningful comparison and evaluation.
"""
def main(
    data_source='SynthLabsAI/Big-Math-RL-Verified',
    n=4,
    temperature=0.6,
    max_tokens=2048,
    model_name='Qwen/Qwen2.5-0.5B-Instruct',
    n_gpu=1,
    local_dir='~/data/big_math_ct',
    hdfs_dir=None,
    seed=42,
    upload_hf_repo="DongfuJiang/Big-Math-RL-Verified-CT",
    debug=False,
):
    random.seed(seed)  
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    dataset = datasets.load_dataset(data_source, trust_remote_code=True)
    dataset = dataset['train'].train_test_split(test_size=1000, seed=seed)
    train_dataset = dataset['train']
    test_dataset = dataset['test']
    if debug:
        train_dataset = train_dataset.select(range(10000))
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
            solution = example['answer']
            output_scores = [int(prime_math.compute_score(output, solution)[0]) for output in example['output']]
            final_data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": question
                }],
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
                    # "ground_truth": ground_truth
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'outputs': example['output'],
                    'outputs_scores': output_scores,
                }
            }
            
            final_pair_data = []
            if all([x == output_scores[0] for x in output_scores]):
                pass
            else:
                correct_outputs = [output for output, score in zip(example['output'], output_scores) if score]
                incorrect_outputs = [output for output, score in zip(example['output'], output_scores) if not score]
                # construct pair from correct and incorrect outputs
                for correct_output in correct_outputs:
                    for incorrect_output in incorrect_outputs:
                        
                        if random.random() > 0.5:
                            answer_1 = correct_output
                            answer_2 = incorrect_output
                            ground_truth = "Answer 1 is correct"
                        else:
                            answer_1 = incorrect_output
                            answer_2 = correct_output
                            ground_truth = "Answer 2 is correct"
                        ct_question = "Give the following problem and two answers, please judge which answer is correct and which answer is incorrect: \nQuestion: " + question \
                            + "\nAnswer 1: " + answer_1 + "\nAnswer 2: " + answer_2 + "\n" + pair_prompt
                            
                        data = {
                            "data_source": "math_ct",
                            "prompt": [{
                                "role": "user",
                                "content": ct_question
                            }],
                            "ability": "math",
                            "reward_model": {
                                "style": "rule",
                                "ground_truth": ground_truth
                            },
                            "extra_info": {
                                'split': split,
                                'index': idx,
                                'solution': solution,
                            }
                        }
                        final_pair_data.append(data)
            return {
                "batch_pair_data": final_pair_data,
                "data": final_data
            }
            
        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)
    
    flatten_train_dataset = []
    flatten_test_dataset = []
    flatten_train_pair_dataset = []
    flatten_test_pair_dataset = []
    for x in train_dataset:
        flatten_train_dataset.append(x['data'])
        flatten_train_pair_dataset += x['batch_pair_data']
    for x in test_dataset:
        flatten_test_dataset.append(x['data'])
        flatten_test_pair_dataset += x['batch_pair_data']
    
    train_dataset = datasets.Dataset.from_list(flatten_train_dataset)
    test_dataset = datasets.Dataset.from_list(flatten_test_dataset)
    train_pair_dataset = datasets.Dataset.from_list(flatten_train_pair_dataset)
    test_pair_dataset = datasets.Dataset.from_list(flatten_test_pair_dataset)
    
    # # # upload to hugging face
    # train_dataset.push_to_hub(upload_hf_repo, config_name='main', split='train')
    # test_dataset.push_to_hub(upload_hf_repo, config_name='main', split='test')
    # train_pair_dataset.push_to_hub(upload_hf_repo, config_name='pair', split='train')
    # test_pair_dataset.push_to_hub(upload_hf_repo, config_name='pair', split='test')
    

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))
    train_pair_dataset.to_parquet(os.path.join(local_dir, 'train_pair.parquet'))
    test_pair_dataset.to_parquet(os.path.join(local_dir, 'test_pair.parquet'))
    
    # train_dataset.to_json("./train.jsonl")

    # if hdfs_dir is not None:
    #     makedirs(hdfs_dir)

    #     copy(src=local_dir, dst=hdfs_dir)
    

if __name__ == '__main__':
    fire.Fire(main)
    
"""
Installations:
```
pip install torch
pip install verl
pip instlal llm-engines
pip install flash-attn --no-build-isolation
pip install datasets
```
python math_pair_ct.py --upload_hf_repo "DongfuJiang/Big-Math-RL-Verified-CT" --debug True
# official run
python math_pair_ct.py --upload_hf_repo "DongfuJiang/Big-Math-RL-Verified-CT" --model_name 'Qwen/Qwen2.5-7B-Instruct' --n_gpu 1 -n 4 --temperature 0.6 --max_tokens 2048 --debug True
"""