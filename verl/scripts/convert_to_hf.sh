

local_dir=""

target_dir=${local_dir}_hf
mkdir -p $target_dir
python3 -m verl.model_merger merge --backend fsdp --local_dir ${local_dir} --target_dir ${target_dir} --use_cpu_initialization



