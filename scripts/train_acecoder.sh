dataset_name=AceCoderV2-150K-processed
train_data=data/acecoder/$dataset_name/train.parquet
val_data=data/acecoder/$dataset_name/test.parquet
# model_name=Qwen/Qwen2.5-0.5B-Instruct
model_name=Qwen/Qwen2.5-7B-Instruct
rl_alg=grpo # gae(ppo) or grpo, if grpo, then better set n>1 otherwise the group norm can not be effective
n_gpus_per_node=8
n_nodes=1
n=8
batch_size=512
ppo_mini_batch_size=64
temperature=1.2

model_pretty_name=$(echo $model_name | tr '/' '_' | tr '[:upper:]' '[:lower:]')
run_name="${model_pretty_name}-${rl_alg}-n${n}-b${batch_size}-t${temperature}"
export VERL_RUN_ID=$run_name # used for saving intermediate results to temp_verl_results/$VERL_RUN_ID
# https://verl.readthedocs.io/en/latest/README_vllm0.7.html#use-vllm-v1-engine
export VLLM_USE_V1=0 # use V1 version of VLLM, which is faster, might have bugs, but should be fine for this; if bugs, set to 0

mkdir -p temp_verl_results/${run_name}

# to accelerate, try increase ppo_micro_batch_size_per_gpu, log_prob_micro_batch_size_per_gpu

# export VLLM_USE_V1=1
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=$rl_alg \
    data.train_files=$train_data \
    data.val_files=$val_data \
    data.train_batch_size=$batch_size \
    data.max_prompt_length=2048 \
    data.max_response_length=4096 \
    +data.max_start_length=2048 \
    +data.max_obs_length=512 \
    +max_turns=10 \
    +do_execute=False \
    reward_model.reward_manager=acecoder \
    actor_rollout_ref.model.path=$model_name \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.temperature=$temperature \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.n=$n \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    critic.optim.lr=1e-5 \
    critic.model.path=$model_name \
    critic.ppo_micro_batch_size_per_gpu=4 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='acecoder' \
    trainer.experiment_name=$run_name \
    +trainer.val_before_train=False \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=$n_gpus_per_node \
    trainer.nnodes=$n_nodes \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.total_epochs=5 2>&1 | tee verl_demo.log


#  +actor_rollout_ref.rollout.stop="'</python>'" \
