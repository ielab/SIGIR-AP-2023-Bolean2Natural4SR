#!/bin/bash

for method in "BM25" "QLM"
  do
#  for collection in "CLEF-2017/testing" "CLEF-2018/testing" "CLEF-2019-dta/testing" "CLEF-2019-intervention/testing"
#  do
#    for type in "title_seed" "seed_fuse"
#    do
#      python3 search_traditional_prompt.py --DATA_DIR data/clef-tar/$collection \
#      --collection_file data/clef-tar/all.jsonl --METHOD $method --type $type --prompt_type biogpt
#    done
##    python3 search_traditional.py --DATA_DIR data/clef-tar/$collection \
##    --collection_file data/clef-tar/all.jsonl --METHOD $method --type title
#  done

  for type in "title_seed" "seed_fuse" "title"
  do
#    python3 search_traditional.py --DATA_DIR data/sysrev-seed-collection \
#    --collection_file data/sysrev-seed-collection/all.jsonl --METHOD $method --type $type
    if [ "$type" = "title_seed" ] || [ "$type" = "seed_fuse" ]
    then
      python3 search_traditional_prompt.py --DATA_DIR data/sysrev-seed-collection \
      --collection_file data/sysrev-seed-collection/all.jsonl --METHOD $method --type $type --prompt_type real_seed
    fi
  done



done

