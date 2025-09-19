set -x

# =============== env and training settings ===============
max_steps=15
ENGINE=vllm
export VLLM_ATTENTION_BACKEND=XFORMERS
num_cpus_per_env_worker=0.1 # The CPU resource allocated for each environment worker. If you want to use less CPU resources, you can decrease this value.
export HF_DATASETS_DISABLE_PROGRESS_BARS=1


train_data_size=4
val_data_size=128
group_size=8
ppo_mini_batch_size=128
ppo_micro_batch_size_per_gpu=1
log_prob_micro_batch_size_per_gpu=1

N_NODES=1
N_GPUS=2


ROOT_PATH=${1:-$PWD}
MODEL_PATH=model/Qwen/Qwen2.5-1.5B-Instruct
PROJECT_NAME="verl_agent_alfworld"
EXP_NAME="grpo_qwen2.5_1.5b-sal"
LOCAL_DIR=${ROOT_PATH}/checkpoint/${PROJECT_NAME}/${EXP_NAME}
ROLLOUT_DIR=${LOCAL_DIR}/rollout
VALIDATION_DIR=${LOCAL_DIR}/validation
mkdir -p $LOCAL_DIR
mkdir -p $ROLLOUT_DIR
mkdir -p $VALIDATION_DIR


# =============== self-imitation learning settings ===============
enable_trajectory_replay=True
TRAIN_BUFFERSIZE=2048
advantage_threshold=1
tolerate_steps=5
replay_loss_coef=1
max_replay_loss_ascending_steps=100

loss_mode="clip_cov"
clip_cov_ratio_replay=0.02 
clip_cov_lb_replay=2 
clip_cov_ub_replay=60.0 
kl_cov_ratio_replay=0.02 


## re-estimate advantage 
weight_decay_trajectory_replay=-1 # use p50 change to estimate the advantage
baseline_buffer_size=10240


# =============== intrinsic reward settings ===============
use_toolcall_reward="cosine"
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



## We only use data preparation to indicate the modality and the data size.
python3 -m examples.data_preprocess.prepare \
    --mode 'text' \
    --train_data_size $train_data_size \
    --val_data_size $val_data_size









python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$ROOT_PATH/data/verl-agent/text/train.parquet \
    data.val_files=$ROOT_PATH/data/verl-agent/text/test.parquet \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=2048 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.actor.policy_loss.loss_mode=${loss_mode} \
    actor_rollout_ref.actor.policy_loss.clip_cov_ratio_replay=${clip_cov_ratio_replay} \
    actor_rollout_ref.actor.policy_loss.clip_cov_lb_replay=${clip_cov_lb_replay} \
    actor_rollout_ref.actor.policy_loss.clip_cov_ub_replay=${clip_cov_ub_replay} \
    actor_rollout_ref.actor.policy_loss.kl_cov_ratio_replay=${kl_cov_ratio_replay} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${ppo_micro_batch_size_per_gpu} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.clip_ratio_low=$clip_ratio_low \
    actor_rollout_ref.actor.clip_ratio_high=$clip_ratio_high \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${log_prob_micro_batch_size_per_gpu} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${log_prob_micro_batch_size_per_gpu} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    actor_rollout_ref.actor.enable_trajectory_replay=${enable_trajectory_replay} \
    actor_rollout_ref.actor.replay_loss_coef=${replay_loss_coef} \
    actor_rollout_ref.actor.weight_decay_trajectory_replay=${weight_decay_trajectory_replay} \
    actor_rollout_ref.actor.trajectory_buffer_size=${TRAIN_BUFFERSIZE} \
    actor_rollout_ref.actor.baseline_buffer_size=${baseline_buffer_size} \
    actor_rollout_ref.actor.trajectory_score_threshold=${advantage_threshold} \
    actor_rollout_ref.actor.trajectory_tolerate_steps=${tolerate_steps} \
    actor_rollout_ref.rollout.rollout_filter_type=${rollout_filter_type} \
    actor_rollout_ref.rollout.rollout_filter_ratio=${rollout_filter_ratio} \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    algorithm.norm_adv_by_std_in_grpo=${norm_adv_by_std_in_grpo} \
    algorithm.filter_overlong_responses=${filter_overlong_responses} \
    algorithm.filter_incomplete_responses=${filter_incomplete_responses} \
    algorithm.filter_repetitive_responses=${filter_repetitive_responses} \
    algorithm.filter_unreadable_responses=${filter_unreadable_responses} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.use_toolcall_reward=${use_toolcall_reward} \
    algorithm.max_toolcall_steps=${max_toolcall_steps} \
    env.env_name=alfworld/AlfredTWEnv \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    env.seed=0 \
    env.max_steps=${max_steps} \
    env.rollout.n=$group_size \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXP_NAME} \
    trainer.default_local_dir=${LOCAL_DIR} \
    trainer.rollout_data_dir=${ROLLOUT_DIR} \
    trainer.validation_data_dir=${VALIDATION_DIR} \
    trainer.n_gpus_per_node=${N_GPUS} \
    trainer.nnodes=${N_NODES} \
    trainer.save_freq=10 \
    trainer.test_freq=5 \
    trainer.total_epochs=200 \
    trainer.val_before_train=False
    # actor_rollout_ref.rollout.val_kwargs.n=3
    
    
# $@

