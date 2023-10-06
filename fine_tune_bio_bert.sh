#!/bin/bash

data_folder=$1
type=$2

python3  model_train_predict.py \
--output_dir "$data_folder"/"$type"/model_bio_bert/ \
--model_name_or_path dmis-lab/biobert-v1.1 \
--do_train \
--save_steps 5000 \
--cache_dir ./cache/ \
--train_dir "$data_folder"/"$type"/train/model_bio_bert/ \
--max_len 512 \
--overwrite_output_dir \
--fp16 \
--train_group_size 10 \
--per_device_train_batch_size 3 \
--warmup_ratio 0.1 \
--weight_decay 0.01 \
--learning_rate 1e-5 \
--num_train_epochs 100 \
--dataloader_num_workers 10

