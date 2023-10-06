#!/bin/bash

# change below to your own path
data_dir=$1
model=$2


if [ "$model" = "alpaca" ]
then
  model_path=../out_7b
elif [ "$model_type" = "bioalpaca" ]
then
  model_path=../bio_alpaca_7b
fi
input_file=generation_model_boolean_to_title.json


echo "$model_path"
echo "$data_dir"/"$input_file"

torchrun --nproc_per_node=3 --master_port=12345 train.py \
    --model_name_or_path "$model_path" \
    --data_path "$data_dir"/"$input_file" \
    --bf16 True \
    --output_dir "$data_dir"/bool_title/"$model"/ \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 2 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 768 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --tf32 True