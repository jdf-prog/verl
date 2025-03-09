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
import random
import regex as re
from string import Template
from verl.utils.reward_score.prime_math import match_answer
from transformers import AutoTokenizer


# prompt_template = """
# Give the following problem and two answers, please judge which answer is correct and which answer is incorrect. 

# Provide a detailed comparative analysis of Answer 1 and Answer 2 based on the following criteria:

# 1. Mathematical/logical accuracy: Identify any calculation errors, logical flaws, or incorrect assumptions
# 2. Methodology: Evaluate the approach used in each answer
# 3. Completeness: Assess whether all aspects of the problem were addressed
# 4. Clarity: Consider how clearly the reasoning is presented

# For each criterion, explain:
# - The strengths and weaknesses of both answers
# - Specific examples from each answer to support your analysis
# - How the differences impact the overall correctness

# After your detailed analysis, provide your final verdict as either:
# \\boxed{Answer 1 is correct} or \\boxed{Answer 2 is correct}

# If both answers contain errors or if they're both correct but through different approaches, explain this nuance before providing your verdict on which is ultimately more accurate or complete.

# Important: Your analysis must go beyond simply repeating or paraphrasing the answers. Focus on meaningful comparison and evaluation.

# <begin_of_question>
# {question}
# <end_of_question>

# <begin_of_answer_1>
# {answer_1}
# <end_of_answer_1>

# <begin_of_answer_2>
# {answer_2}
# <end_of_answer_2>
# """

prompt_template = """
For the given problem and two answers below, determine which is correct with a detailed analysis:

Evaluate both answers based on:
1. Mathematical/logical accuracy: Identify errors, flaws, or incorrect assumptions
2. Methodology: Assess the approach used
3. Completeness: Check if all aspects were addressed
4. Clarity: Evaluate how clearly the reasoning is presented

For each criterion:
- Compare strengths and weaknesses with specific examples
- Explain how differences affect correctness

Conclude with your final verdict: \\boxed{Answer 1 is correct} or \\boxed{Answer 2 is correct}

If both answers contain errors or use different yet valid approaches, explain this before your verdict.

<begin_of_question>
${question}
<end_of_question>

<begin_of_answer_1>
${answer_1}
<end_of_answer_1>

<begin_of_answer_2>
${answer_2}
<end_of_answer_2>
"""
def main(
    data_source='RLRM/Big-Math-RL-Verified-CT',
    num_shards=32,
    seed=42,
    local_dir='~/data/big_math_rl_pair_ct',
    remove_thinking=False,
    remove_clipped_output=False
):
    random.seed(seed)  
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    all_subsets = [f"main_shard_{i}_of_{num_shards}" for i in range(num_shards)]
    exsiting_configs = datasets.get_dataset_config_names(data_source)
    
    train_datasets = []
    test_datasets = []
    for subset in all_subsets:
        if subset in exsiting_configs:
            dataset = datasets.load_dataset(data_source, subset)
            train_datasets.append(dataset['train'])
            test_datasets.append(dataset['test'])
    train_dataset = datasets.concatenate_datasets(train_datasets)
    test_dataset = datasets.concatenate_datasets(test_datasets)
    print(f"Loaded {len(train_dataset)} training examples and {len(test_dataset)} testing examples", flush=True)
    
    if remove_thinking:
        print("Will remove <think>...</think> from the outputs", flush=True)
    if remove_clipped_output:
        print("Will remove clipped outputs", flush=True)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            question = example['prompt'][0]['content']
            solution = example['reward_model']['ground_truth']
            outputs = example['extra_info']['outputs']
            output_scores = example['extra_info']['outputs_scores']
            
            num_clipped_output = 0
            if remove_clipped_output:
                matched_outputs = [match_answer(x) for x in outputs]
                is_matched = [x[0] for x in matched_outputs]
                has_think_end = ["</think>" in x for x in outputs]
                matched_answers = [x[1] for x in matched_outputs]
                outputs = [outputs[i] for i in range(len(outputs)) if is_matched[i] and has_think_end[i]]
                output_scores = [output_scores[i] for i in range(len(output_scores)) if is_matched[i] and has_think_end[i]]
                num_clipped_output = len(example['extra_info']['outputs_scores']) - sum(output_scores)
                
            if remove_thinking:
                outputs = [re.sub(r"(<think>)?(.|\n)*?</think>", "", x) for x in outputs]
            
            final_pair_data = []
            if all([x == output_scores[0] for x in output_scores]):
                pass
            else:
                correct_outputs = [output for output, score in zip(outputs, output_scores) if score]
                incorrect_outputs = [output for output, score in zip(outputs, output_scores) if not score]
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
                        ct_question = Template(prompt_template).substitute(question=question, answer_1=answer_1, answer_2=answer_2).strip('\n')
                        ct_question_num_token = len(tokenizer(ct_question)['input_ids'])
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
                                'question_num_token': ct_question_num_token
                            }
                        }
                        final_pair_data.append(data)
            return {
                "batch": final_pair_data,
                "num_clipped_output": num_clipped_output,
                "total_num": len(example['extra_info']['outputs'])
            }
            
        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True, num_proc=16)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True, num_proc=16)
    
    total_clipped_output = sum([x['num_clipped_output'] for x in train_dataset]) + sum([x['num_clipped_output'] for x in test_dataset])
    total_num = sum([x['total_num'] for x in train_dataset]) + sum([x['total_num'] for x in test_dataset])
    
    if remove_clipped_output:
        print(f"Removed {total_clipped_output}/{total_num}(={total_clipped_output/total_num*100:.2f}%) clipped outputs", flush=True)
    
    flatten_train_dataset = []
    flatten_test_dataset = []
    for x in train_dataset:
        flatten_train_dataset.extend(x['batch'])
    for x in test_dataset:
        flatten_test_dataset.extend(x['batch'])
    
    train_dataset = datasets.Dataset.from_list(flatten_train_dataset)
    test_dataset = datasets.Dataset.from_list(flatten_test_dataset)
    
    print(f"Dataset's question has an average of {sum([x['extra_info']['question_num_token'] for x in train_dataset]) / len(train_dataset):.2f} tokens", flush=True)
    print(f"Saving {len(train_dataset)} examples to {local_dir}/train.parquet", flush=True)
    print(f"Saving {len(test_dataset)} examples to {local_dir}/test.parquet", flush=True)

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))
    train_dataset.select(range(100)).to_json(os.path.join(local_dir, 'train.sample.json'))
    test_dataset.select(range(100)).to_json(os.path.join(local_dir, 'test.sample.json'))
    print(f"Saved the train and test datasets to {local_dir}", flush=True)
    print(f"See the first 100 examples in the train and test datasets in file \n- {local_dir}/train.sample.json\n- {local_dir}/test.sample.json", flush=True)
    print(f"Set training and testing data to the parquet format when training as \n- {local_dir}/train.parquet\n- {local_dir}/test.parquet", flush=True)

if __name__ == '__main__':
    fire.Fire(main)
    
"""
python examples/data_preprocess/math_rl_pair_ct.py --data_source RLRM/Big-Math-RL-Verified-CT --num_shards 32 --seed 42 --local_dir ~/data/big_math_rl_pair_ct --remove_thinking True --remove_clipped_output True
python examples/data_preprocess/math_rl_pair_ct.py --data_source RLRM/Big-Math-RL-Verified-CT --num_shards 32 --seed 42 --local_dir ~/data/big_math_rl_pair_ct_with_thinking --remove_thinking False --remove_clipped_output True
"""