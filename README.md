<div align="center">
  <img src="./imgs/spear-agent.png" width="400"/>
</div>
<p align="center">
  <a href="xxxx">
    <img src="https://img.shields.io/badge/arXiv-Paper-red?style=flat-square&logo=arxiv" alt="arXiv Paper"></a>
  &nbsp;
  <a href="xxxx">
    <img src="https://img.shields.io/badge/GitHub-Project-181717?style=flat-square&logo=github" alt="GitHub Project"></a>
  &nbsp;
  <a href="xxxx">
    <img src="https://img.shields.io/badge/HuggingFace-Models-yellow?style=flat-square&logo=huggingface" alt="HuggingFace Models"></a>
  &nbsp;
</p>

SPEAR <img src="./imgs/spear-logo-in-line.png" alt="spear-logo-in-line" height="15px"/> is a curriculum-based self-imitation learning (SIL) framework for training agentic LLMs on long-horizon, sparse-reward tasks. It balances exploration and exploitation by first leveraging auxiliary tool-use rewards to encourage broad skill-level exploration, and later strengthening self-imitation to exploit successful trajectories from replayed experiences. This adaptive curriculum stabilizes training and improves efficiency while maintaining well-controlled entropy.


# News



# Contents

- [Results](#Results)  

- [Training Configuration](#training-configuration)
  - [Self-imitation Learning](#self-imitation-learning)
  - [Advantage Recalibration](#advantage-recalibration-for-off-policy-estimation)
  - [Regularization for Entropy Control](#regularization-for-entropy-control-clip-cov-loss)
  - [Intrinsic Reward](#intrinsic-reward)
  - [Dr.BoT](##drbot-settings)

- [Reproduce results](#reproduce-results)
  - [Math](#math)
  - [verl-agent](#verl-agent)

# Results

Results using Qwen2.5-1.5B-Instruct on ALFWorld and WebShop:

| Method        | ALFWorld    | WebShop(SR) |
| :------------ | ----------- | ----------- |
| GRPO          | 72.8        | 56.8        |
| +SPEAR(ours)   | 88.9(+16.1) | 77.5(+20.7) |
| Dr.BoT(GRPO)  | 79.1        | 62.9        |
| +SPEAR(ours)   | 87.7(+8.6)  | 76.8(+13.9) |
| GiGPO         | 86.1        | 67.4        |
| +SPEAR(ours)   | 91.2(+5.1)  | 79.3(+11.8) |
| Dr.BoT(GiGPO) | 90.6        | 68.8        |
| +SPEAR(ours)   | 93.2(+2.6)  | 81.1(+81.1) |

Results using Qwen2.5-32B-Instruct and Qwen3-32B-Instruct on AIME24 and AIME25:

| Method       | Model                | AIME24 | AIME25 |
| ------------ | -------------------- | ------ | ------ |
| PPO          | Qwen2.5-32B-Instruct | -      | 55.0   |
| GRPO         | Qwen2.5-32B-Instruct | -      | 60.0   |
| Dr.BoT(GRPO) | Qwen2.5-32B-Instruct | 64.7   | 54.0   |
| +SPEAR(ours)  | Qwen2.5-32B-Instruct | 66.3(+1.6)   | 60.1(+6.1)   |
| Dr.BoT(GRPO) | Qwen3-32B-Instruct   | 82.5   | 77.3   |
| +SPEAR(ours)  | Qwen3-32B-Instruct   | 85.6(+3.1)   | 80.5(+3.2)   |





# Training Configuration

## Self-imitation Learning

```yaml
actor_rollout_ref:
  actor:
    # Whether to enable self-imitation loss
    enable_trajectory_replay: False
    
    # Maximum number of trajectories stored in the self-imitation buffer
    trajectory_buffer_size: 2048 
    
    # Only trajectories with an advantage larger than this threshold will be saved
    trajectory_score_threshold: 1
    
    # Only trajectories with a step delay less than this tolerance will be remained
    trajectory_tolerate_steps: 10
    
    # PPO loss coefficient for self-imitation learning
    replay_loss_coef: 1
    
    # Number of steps for increasing the PPO loss coefficient using a cosine scheduler
    max_replay_loss_steps: 200
```



## Advantage Recalibration for Off-Policy Estimation

```yaml
actor_rollout_ref:
  actor:
    # How the advantage of trajectories in the replay buffer is re-estimated
    weight_decay_trajectory_replay: -1
    
    # Number of trajectories' rewards used to calculate the 50th percentile baseline
    baseline_buffer_size: 10240 
```

- ``weight_decay_trajectory_replay`` controls how the advantage of trajectories in the replay buffer is recalibrated.

- If ``weight_decay_trajectory_replay`` is  -1, the 50th percentile baseline will be used to re-estimate the advantage.

- If ``weight_decay_trajectory_replay`` is in (0, 1], the advantage will decay as: $$ \text{advantage} = \text{old advantage} \times \text{weightDecayTrajectoryReplay} $$

  
  
## Regularization for Entropy Control (Clip-cov Loss)

```yaml
actor_rollout_ref:
  actor:
    policy_loss:
      # Loss mode for regularization. Options: (see https://arxiv.org/abs/2505.22617)
      #   - vanilla
      #   - clip-cov, default = clip-cov for Dr.BoT
      #   - kl-cov
      #   - gpg 
      loss_mode: "vanilla"

      # ================== Hyperparameters for On-policy RL Loss ==================
      # Ratio of tokens to be clipped for clip-cov loss
      clip_cov_ratio: 0.02
      
      # Lower bound for clip-cov loss
      clip_cov_lb: 1.0
      
      # Upper bound for clip-cov loss
      clip_cov_ub: 40.0
      
      # Ratio of tokens to apply KL penalty for kl-cov loss
      kl_cov_ratio: 0.02

      # ================== Hyperparameters for SIL Loss ==========================
      # [Replay Only] Ratio of tokens to be clipped for clip-cov loss
      clip_cov_ratio_replay: 0.02
      
      # [Replay Only] Lower bound for clip-cov loss
      clip_cov_lb_replay: 1.0
      
      # [Replay Only] Upper bound for clip-cov loss
      clip_cov_ub_replay: 40.0
      
      # [Replay Only] Ratio of tokens to apply KL penalty for kl-cov loss
      kl_cov_ratio_replay: 0.02
```



## Intrinsic Reward

```yaml
algorithm:
  # Tool-call reward mode:
  #   - "none"     : Do not use tool-call reward
  #   - "constant" : Use a fixed tool-call reward coefficient (1) during training 
  #   - "cosine"   : Decay the tool-call reward coefficient with a cosine scheduler
  use_toolcall_reward: "cosine"  

  # Maximum number of steps for the cosine scheduler
  max_toolcall_steps: 100
```





## Dr.BoT Settings

Removing KL divergence to the reference model: 

```yaml
actor_rollout_ref:
  actor:
    # Whether to use KL loss against the reference model
    use_kl_loss: False
    
    # Coefficient for KL loss (set to 0.0 if disabled)
    kl_loss_coef: 0.0
    
    # KL loss type (e.g., "low_var_kl" for GRPO)
    kl_loss_type: low_var_kl
```



Clip higher: 

```yaml
actor_rollout_ref:
  actor:
    # Lower bound of the clipping ratio
    clip_ratio_low: 0.2
    
    # Upper bound of the clipping ratio
    clip_ratio_high: 0.28
```



Removing  intra-group normalization:

```yaml
algorithm:
  # Whether to normalize advantages by group standard deviation in GRPO
  norm_adv_by_std_in_grpo: False
```



Removing length bias:

```yaml
actor_rollout_ref:
  actor:
    # Aggregation mode for loss:
    #   - "token-mean" (DAPO)
    #   - "seq-mean-token-sum"
    #   - "seq-mean-token-mean"
    #   - "seq-mean-token-sum-norm" (Dr.GRPO)
    loss_agg_mode: "seq-mean-token-sum-norm"
```



Filtering low-quality samples:

```yaml
algorithm:
  # Filter out overlong responses, default = True for Dr.BoT
  filter_overlong_responses: True
  
  # Filter out incomplete responses (void-turn), default = True for Dr.BoT
  filter_incomplete_responses: True
  
  # Filter out repetitive responses, default = True for Dr.BoT
  filter_repetitive_responses: True
  
  # Filter out unreadable responses, default = True for Dr.BoT
  filter_unreadable_responses: True
```



Filtering low-variance groups:

```yaml
actor_rollout_ref:
  rollout:
    # Rollout filtering ratio by standard deviation. We use 0.75 in Dr.BoT
    rollout_filter_ratio: 0.75
    
    # Rollout filter type: "std" (standard deviation)
    rollout_filter_type: std
```



# Reproduce results

## Math

### 1. Install

We follow the installation instructions in [verl documentation](https://verl.readthedocs.io/en/latest/start/install.html#install-from-custom-environment) to install the nessary environment.

#### a) Install CUDA, cuDNN and Apex (Optional)

Install CUDA>=12.4

```bash
# change directory to anywher you like, in verl source code directory is not recommended
mkdir tmp
cd tmp
wget https://developer.download.nvidia.com/compute/cuda/12.4.1/local_installers/cuda-repo-ubuntu2204-12-4-local_12.4.1-550.54.15-1_amd64.deb
dpkg -i cuda-repo-ubuntu2204-12-4-local_12.4.1-550.54.15-1_amd64.deb
cp /var/cuda-repo-ubuntu2204-12-4-local/cuda-*-keyring.gpg /usr/share/keyrings/
apt-get update
apt-get -y install cuda-toolkit-12-4
update-alternatives --set cuda /usr/local/cuda-12.4
```

Install cuDNN>9.8.0

```bash
# change directory to anywher you like, in verl source code directory is not recommended
mkdir tmp
cd tmp
wget https://developer.download.nvidia.com/compute/cudnn/9.8.0/local_installers/cudnn-local-repo-ubuntu2204-9.8.0_1.0-1_amd64.deb
dpkg -i cudnn-local-repo-ubuntu2204-9.8.0_1.0-1_amd64.deb
cp /var/cudnn-local-repo-ubuntu2204-9.8.0/cudnn-*-keyring.gpg /usr/share/keyrings/
apt-get update
apt-get -y install cudnn-cuda-12
```

Install NVIDIA Apex, you can change the ``MAX_JOBS`` to accelerate the installation process, but do not set it too large in case of memory issues.

```bash
# change directory to anywher you like, in verl source code directory is not recommended
mkdir tmp
cd tmp
git clone https://github.com/NVIDIA/apex.git && \
cd apex && \
MAX_JOB=32 pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
```

#### b) Install dependencies 

Create a new environment:

```bash
conda create -n verl python==3.10 -y
conda activate verl
```

Then, execute the `install.sh` script provided in verl:

```bash
cd repo_root/verl
USE_MEGATRON=0 USE_SGLANG=0 bash scripts/install_vllm_sglang_mcore.sh
```

#### c)  Install verl

```bash
cd repo_root/verl
pip install --no-deps -e .
```

### 2. Preparing data and cold-start model
1. Preparing data:
```bash
python3 recipe/spear/sft_preprocess.py
```

2. Getting the cold-start model:
```bash
bash recipe/spear/run_qwen2-32b_sft.sh
```

3. Transform to HuggingFace format:
```bash
python -m verl.model_merger merge --backend fsdp \
    --local_dir <SFT_SAVE_PATH>/global_step_372/actor \
    --target_dir <SFT_SAVE_PATH>/global_step_372_merge
```



### 3. Training 


Training with GRPO baseline:

```bash
bash recipe/spear/run_qwen2-32b.sh
```

Training with Dr.BoT:

```bash
bash recipe/spear/run_qwen2-32b_drbot.sh
```

Training with SPEAR:

```bash
bash recipe/spear/run_qwen2-32b_spear.sh
```


## verl-agent

### 1. Install

We follow the installation instructions in ``verl-agent``[documentation](https://github.com/langfengQ/verl-agent?tab=readme-ov-file#install-supported-environments) to install the nessary environment.

Unzip the environments:
```
cd verl-agent/agent_system/
tar -xvf environments.tar
```

> Due to potential package version conflicts, we recommend setting independent conda environments for different agent environments.

#### ALFWorld

Install verl and ALFWorld dependencies

```bash
## Install verl dependencies 
conda create -n verl-agent-alfworld python==3.12 -y
conda activate verl-agent-alfworld
pip3 install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip3 install flash-attn==2.7.4.post1 --no-build-isolation 
pip3 install -r requirements.txt
pip3 install -e . 
pip3 install vllm==0.8.5 



## Install ALFWorld dependencies
pip3 install gymnasium==0.29.1 
pip3 install stable-baselines3==2.6.0 
pip install alfworld 
pip install vllm==0.8.5 
```

Download PDDL & Game files and pre-trained MaskRCNN detector (will be stored in `~/.cache/alfworld/`):

```bash
alfworld-download -f
```

Use `--extra` to download pre-trained checkpoints and seq2seq data.

Play a Textworld game:

```bash
alfworld-play-tw
```

#### WebShop

WebShop requires Python <=3.10, so begin by creating a new `verl-agent-webshop` environment

```bash
conda create -n verl-agent-webshop python==3.10 -y
conda activate verl-agent-webshop
```

Install WebShop

```bash
cd ./agent_system/environments/env_package/webshop/webshop
./setup.sh -d all
```

Note: If you encounter issues with gdown, you may need to visit `https://drive.google.com/`, get your Google Drive cookie, and paste it into `~/.cache/gdown/cookies.txt`. Or you may need to manually download the files.

After WebShop is installed, return to the root directory of the repository and install the verl package in `verl-agent`:

```bash
cd repo_root/verl-agent
pip3 install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip3 install flash-attn==2.7.4.post1 --no-build-isolation
pip3 install -e .
pip3 install vllm==0.8.2 --no-deps
pip3 install -r requirements-vllm-0.8.2.txt --no-deps
# vllm 0.8.2 requires mistral_common[opencv]>=1.5.4, which is not installed.
# spacy 3.7.2 requires typer<0.10.0,>=0.3.0, but you have typer 0.15.2 which is incompatible.
# weasel 0.3.4 requires typer<0.10.0,>=0.3.0, but you have typer 0.15.2 which is incompatible.
# The warnings can be safely ignored.
```

Installing ``mistral_common`` will update ``numpy`` and cause errors in the following training. So we don't install it here.



### 2. Training 

Training with GRPO:

```bash
# ALFWorld
bash examples/grpo_trainer/run_alfworld.sh # GRPO baseline
bash examples/grpo_trainer/run_alfworld_drbot.sh # Dr.BoT
bash examples/grpo_trainer/run_alfworld_spear.sh # SPEAR

# WebShop
bash examples/grpo_trainer/run_webshop.sh # GRPO baseline
bash examples/grpo_trainer/run_webshop_drbot.sh # Dr.BoT
bash examples/grpo_trainer/run_webshop_spear.sh # SPEAR
```

Training with GiGPO:

```bash
# ALFWorld
bash examples/gigpo_trainer/run_alfworld.sh # GRPO baseline
bash examples/gigpo_trainer/run_alfworld_drbot.sh # Dr.BoT
bash examples/gigpo_trainer/run_alfworld_spear.sh # SPEAR

# WebShop
bash examples/gigpo_trainer/run_webshop.sh # GRPO baseline
bash examples/gigpo_trainer/run_webshop_drbot.sh # Dr.BoT
bash examples/gigpo_trainer/run_webshop_spear.sh # SPEAR
```

## Acknowledgement

Our codebase is bulit upon [verl](https://github.com/volcengine/verl) and [verl-agent](https://github.com/langfengQ/verl-agent). We greatly appreciate their awesome work and the dedication of the contributors who made these projects available to the community. 
