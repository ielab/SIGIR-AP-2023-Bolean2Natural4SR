

for collection in "clef-tar/CLEF-2017/testing" "clef-tar/CLEF-2018/testing" "clef-tar/CLEF-2019-dta/testing" "clef-tar/CLEF-2019-intervention/testing" "sysrev-seed-collection/testing"
do
  python3 interpolate_two.py --DATA_DIR_1 combined_output/data/"$collection"/boolean/model_bio_bert/ --DATA_DIR_2 combined_output/data/"$collection"/boolean_alpaca_tuned_query/model_bio_bert/
  python3 evaluation.py --DATA_DIR combined_output/data/"$collection"/interpolate_boolean_boolean_alpaca_tuned_query/model_bio_bert
done