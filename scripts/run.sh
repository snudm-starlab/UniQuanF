#! /usr/bin/env bash
GPU=0
METHOD_NAME=UniQuanF
bcq=True

export PYTHONPATH=$(pwd)
model_name_or_path='meta-llama/Meta-Llama-3-8B'

# Settings
dataset=c4
seed=0
n_bits_w=4
n_bits_a=16 # weight-only quantization
quantization_dataset=train
channel_wise=True
torch_dtype=bfloat16
recon_dtype=float32
save_model=False
seq_len=2048
num_samples=128
group_size=-1 # channel-wise quantization
symmetric=False
cache_dir=./cache # Your directory for caching

# Hyperparameters
u_lr=5e-3
b_lr=5e-4
EPOCHS=20
batch_size_quant=1
period=2 # p
alternating_update_iters=15 # T
grid_search_iters=1 # G
update_z=True # update zero-point or not
MAPPING=lpmapping # local and periodic mapping

iters_w=$(expr $num_samples \* $EPOCHS \/ $batch_size_quant)
FILE_NAME=${n_bits_w}_${seed}

output_dir=outputs/${METHOD_NAME}/${model_name_or_path}/${FILE_NAME}

CUDA_VISIBLE_DEVICES=$GPU python src/main.py \
    --model_name_or_path $model_name_or_path  \
    --use_bcq=$bcq \
    --group_size $group_size \
    --block_size $seq_len \
    --dataset_name $dataset \
    --per_device_train_batch_size $batch_size_quant \
    --u_lr $u_lr \
    --b_lr $b_lr \
    --output_dir $output_dir \
    --n_bits_w $n_bits_w \
    --n_bits_a $n_bits_a \
    --num_samples $num_samples \
    --iters_w $iters_w \
    --channel_wise $channel_wise \
    --symmetric $symmetric \
    --quantization_dataset $quantization_dataset \
    --save_model $save_model \
    --torch_dtype $torch_dtype \
    --recon_dtype $recon_dtype \
    --cache_dir $cache_dir \
    --period ${period} \
    --seed $seed \
    --alternating_update_iters $alternating_update_iters \
    --grid_search_iters $grid_search_iters \
    --update_z $update_z \
    --mapping $MAPPING
