

set -x


# =============== env and training settings ===============


export VLLM_USE_V1=1
export HF_DATASETS_DISABLE_PROGRESS_BARS=1


NNODES=16
GPUS_PER_NODE=8


max_turns=8
max_prompt_length=2048
max_response_length=16384
max_token_length=$max_response_length
actor_lr=1e-6

train_batch_size=128
ppo_mini_batch_size=64

infer_tp=4 # vllm
train_sp=8 # train

n_resp_per_prompt=16
n_resp_per_prompt_val=32




# =============== log settings ===============

experiment_name=qwen2.5-32b_drbot
project_name=retool_rl


# =============== path settings ===============
ROOT_PATH=${1:-$PWD}

model_path=checkpoints/retool_sft/qwen-2.5-32b-instruct/global_step_372_merge
dapo_math_17k=${ROOT_PATH}/datasets/BytedTsinghua-SIA/DAPO-Math-17k
aime_2024=${ROOT_PATH}/datasets/Maxwell-Jia/AIME_2024
aime_2025=${ROOT_PATH}/datasets/yentinglin/aime_2025

train_files="['$dapo_math_17k']"
test_files="['$aime_2024','$aime_2025']"

tool_config_path=${ROOT_PATH}/recipe/retool/sandbox_fusion_tool_config.yaml

default_local_dir=$ROOT_PATH/checkpoints/$project_name/$experiment_name
mkdir -p ${default_local_dir}



# =============== self-imitation learning settings ===============
enable_trajectory_replay=False
trajectory_buffer_size=2048
advantage_threshold=1
trajectory_tolerate_steps=5
replay_loss_coef=1
max_replay_loss_ascending_steps=300

loss_mode="vanilla"
clip_cov_ratio_replay=0.02 
clip_cov_lb_replay=1 
clip_cov_ub_replay=40.0
kl_cov_ratio_replay=0.02 


## re-estimate advantage 
weight_decay_trajectory_replay=-1 # use p50 change to estimate the advantage
baseline_buffer_size=10240


# =============== intrinsic reward settings ===============
use_toolcall_reward="none"
max_toolcall_steps=200


# ================ Dr.BoT settings ================

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0

clip_ratio_low=0.2
clip_ratio_high=0.28

norm_adv_by_std_in_grpo=False

loss_agg_mode="seq-mean-token-sum-norm"

filter_overlong_responses=True
filter_incomplete_responses=True
filter_repetitive_responses=True
filter_unreadable_responses=True

rollout_filter_ratio=0.75
rollout_filter_type="std"


# ================= other settings =================
offload=True
actor_max_token_len_per_gpu=$(( (max_prompt_length + max_response_length) * 1 ))
log_prob_max_token_len_per_gpu=$(( actor_max_token_len_per_gpu * 4 ))
val_before_train=True



python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=$use_kl_in_reward \
    algorithm.kl_ctrl.kl_coef=$kl_coef \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.return_raw_chat=True \
    data.train_batch_size=$train_batch_size \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.max_length=$max_token_length \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.custom_cls.path=${ROOT_PATH}/recipe/retool/retool.py \
    data.custom_cls.name=CustomRLHFDataset \
    reward_model.reward_manager=agentConcurrent \
    custom_reward_function.path=${ROOT_PATH}/recipe/retool/reward.py \
    custom_reward_function.name=default_compute_score_enforce_toolcall_posneg_decay \
    actor_rollout_ref.model.path=$model_path \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.use_kl_loss=$use_kl_loss \
    actor_rollout_ref.actor.kl_loss_coef=$kl_loss_coef \
    actor_rollout_ref.actor.clip_ratio_low=$clip_ratio_low \
    actor_rollout_ref.actor.clip_ratio_high=$clip_ratio_high \
    actor_rollout_ref.actor.policy_loss.loss_mode=${loss_mode} \
    actor_rollout_ref.actor.policy_loss.clip_cov_ratio_replay=${clip_cov_ratio_replay} \
    actor_rollout_ref.actor.policy_loss.clip_cov_lb_replay=${clip_cov_lb_replay} \
    actor_rollout_ref.actor.policy_loss.clip_cov_ub_replay=${clip_cov_ub_replay} \
    actor_rollout_ref.actor.policy_loss.kl_cov_ratio_replay=${kl_cov_ratio_replay} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.optim.lr=$actor_lr \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$actor_max_token_len_per_gpu \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$train_sp \
    actor_rollout_ref.actor.fsdp_config.param_offload=$offload \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$offload \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$log_prob_max_token_len_per_gpu \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$infer_tp \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=$max_turns \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=$max_turns \
    actor_rollout_ref.rollout.multi_turn.tool_config_path=$tool_config_path \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
    actor_rollout_ref.rollout.n=$n_resp_per_prompt \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.6 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.n=$n_resp_per_prompt_val \
    actor_rollout_ref.rollout.rollout_filter_type=${rollout_filter_type} \
    actor_rollout_ref.rollout.rollout_filter_ratio=${rollout_filter_ratio} \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.enable_trajectory_replay=${enable_trajectory_replay} \
    actor_rollout_ref.actor.replay_loss_coef=${replay_loss_coef} \
    actor_rollout_ref.actor.max_replay_loss_steps=${max_replay_loss_ascending_steps} \
    actor_rollout_ref.actor.weight_decay_trajectory_replay=${weight_decay_trajectory_replay} \
    actor_rollout_ref.actor.trajectory_buffer_size=${trajectory_buffer_size} \
    actor_rollout_ref.actor.trajectory_score_threshold=${advantage_threshold} \
    actor_rollout_ref.actor.trajectory_tolerate_steps=${trajectory_tolerate_steps} \
    actor_rollout_ref.actor.baseline_buffer_size=${baseline_buffer_size} \
    algorithm.norm_adv_by_std_in_grpo=${norm_adv_by_std_in_grpo} \
    algorithm.filter_overlong_responses=${filter_overlong_responses} \
    algorithm.filter_incomplete_responses=${filter_incomplete_responses} \
    algorithm.filter_repetitive_responses=${filter_repetitive_responses} \
    algorithm.filter_unreadable_responses=${filter_unreadable_responses} \
    algorithm.use_toolcall_reward=${use_toolcall_reward} \
    algorithm.max_toolcall_steps=${max_toolcall_steps} \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node=$GPUS_PER_NODE \
    trainer.val_before_train=${val_before_train} \
    trainer.log_val_generations=4 \
    trainer.nnodes=$NNODES \
    trainer.save_freq=5 \
    trainer.default_local_dir=$default_local_dir \
    trainer.rollout_data_dir=$default_local_dir/rollout \
    trainer.validation_data_dir=$default_local_dir/validation \
    trainer.test_freq=5 \
    trainer.total_epochs=1 
# $@
