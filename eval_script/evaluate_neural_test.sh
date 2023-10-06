#!/bin/bash

for collection in "clef-tar/CLEF-2017/testing" "clef-tar/CLEF-2018/testing" "clef-tar/CLEF-2019-dta/testing" "clef-tar/CLEF-2019-intervention/testing" "sysrev-seed-collection/testing"
do
  for model in "model_bio_bert"
  do
    for prompt_type in "title" "title_alpaca" #"title_alpaca_multi"
    #"title_openai" "title_alpaca" "title_bioalpaca"
    #"title_alpaca" "title_bioalpaca"
    #"title" "abstract" "boolean"
    do

      echo "$collection"_"$model"_"$prompt_type"
#      python3 fuse_mean.py --DATA_DIR combined_output/data/"$collection"/"$prompt_type"/"$model"/
#      python3 fuse_max.py --DATA_DIR combined_output/data/"$collection"/"$prompt_type"/"$model"/

      python3 evaluation.py \
      --DATA_DIR combined_output/data/"$collection"/"$prompt_type"/"$model"/
      

#      python3 evaluation.py  \
#      --DATA_DIR combined_output/data/"$collection"/"$prompt_type"/"$model"/fusion_max
#
#
#      python3 evaluation.py  \
#      --DATA_DIR combined_output/data/"$collection"/"$prompt_type"/"$model"/fusion_mean
#
    done
#    echo "$collection"_"$model"_"$prompt_type"_title
#    python3 evaluation.py --qrel_file data/$collection/data.qrels \
#    --DATA_DIR combined_output/data/"$collection"/title/"$model"/
  done
done

#for model in "model_bio_bert"
#do
#  for prompt_type in "title_alpaca_multi"
#  #"title_openai" "title_alpaca" "title_bioalpaca"
#  #"title_alpaca" "title_bioalpaca"
#  #"title" "abstract" "boolean"
#  do
#    echo sysrev-seed-collection_"$model"_"$prompt_type"
#
##    python3 fuse_max.py --DATA_DIR combined_output/data/sysrev-seed-collection/"$prompt_type"/"$model"/
##    python3 fuse_mean.py --DATA_DIR combined_output/data/sysrev-seed-collection/"$prompt_type"/"$model"/
#
#    python3 evaluation.py  \
#    --DATA_DIR combined_output/data/sysrev-seed-collection/"$prompt_type"/"$model"/
#
#
#    python3 evaluation.py  \
#    --DATA_DIR combined_output/data/sysrev-seed-collection/"$prompt_type"/"$model"/fusion_max
#
#
#    python3 evaluation.py  \
#    --DATA_DIR combined_output/data/sysrev-seed-collection/"$prompt_type"/"$model"/fusion_mean
#
#
#
#  done
#
#done