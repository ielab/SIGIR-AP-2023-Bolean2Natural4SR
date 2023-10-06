import json
import argparse
import os
import glob
from transformers import AutoTokenizer
from tqdm import tqdm
#from nltk.corpus import stopwords
#import string
#cachedStopwords = set(tok.lower() for tok in stopwords.words("english"))
#string_set =set(string.punctuation)


parser = argparse.ArgumentParser()

parser.add_argument("--DATA_DIR", type=str, default="data/sysrev-seed-collection")

args = parser.parse_args()



input_files = glob.glob(os.path.join(args.DATA_DIR, "*.trec"))

out_folder = os.path.join(args.DATA_DIR, "fusion_mean")
if not os.path.exists(out_folder):
    os.makedirs(out_folder)

result_dict = {}

for input_file in tqdm(input_files):
    if "_" not in input_file.split('/')[-1]:
        continue
    with open(input_file) as f:
        for line in f:
            qid, _, pid, rank, score, _ = line.strip().split()
            qid_original = qid.split('_')[0]
            if qid_original not in result_dict:
                result_dict[qid_original] = {}
            if pid not in result_dict[qid_original]:
                result_dict[qid_original][pid] = float(score)
            else:
                result_dict[qid_original][pid] += float(score)

for qid in tqdm(result_dict):
    result_dict[qid] = sorted(result_dict[qid].items(), key=lambda x: x[1], reverse=True)
    with open(os.path.join(out_folder, qid + ".trec"), 'w') as f:
        for i, (pid, score) in enumerate(result_dict[qid]):
            f.write(f"{qid} Q0 {pid} {i+1} {score} fuse\n")



