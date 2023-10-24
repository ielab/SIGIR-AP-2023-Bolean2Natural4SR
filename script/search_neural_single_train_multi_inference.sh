#!/bin/bash

input_folder=$1
type=$2

for model_type in "model_bio_bert"
do

  output_folder=output/"$input_folder"/"$type"_single_multi/"$model_type"
  mkdir -p "$output_folder"
  if [ "$model_type" = "model_bert" ]
  then
    model_tokenizer=bert-base-uncased
  elif [ "$model_type" = "model_bio_bert" ]
  then
    model_tokenizer=dmis-lab/biobert-v1.1
  elif [ "$model_type" = "model_luyu" ]
  then
    model_tokenizer=bert-base-uncased
  fi

  python3 search_neural.py \
    --output_dir $output_folder \
    --model_name_or_path "$input_folder"/models_single/"$type"/"$model_type"/ \
    --tokenizer_name $model_tokenizer \
    --do_predict \
    --max_len 512 \
    --fp16 \
    --per_device_eval_batch_size 96 \
    --dataloader_num_workers 4 \
    --pred_path "$input_folder"/neural_inputs/"$type"_multi/"$model_tokenizer"/run.jsonl \
    --pred_id_file  "$input_folder"/neural_inputs/"$type"_multi/"$model_tokenizer"/run.tsv \
    --cache_dir ./cache/
done