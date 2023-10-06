import argparse
import os
import glob
from subprocess import Popen, PIPE, STDOUT
from tqdm import tqdm
parser = argparse.ArgumentParser()
parser.add_argument("--DATA_DIR", type=str, default="data/sysrev-seed-collection/title/BM25")
args = parser.parse_args()

data_dir = args.DATA_DIR
input_file = os.path.join(data_dir, "qrel_abs_train_loo.txt" )
out_folder = os.path.join(data_dir, "train_qrel_loo" )



if not os.path.exists(out_folder):
    os.makedirs(out_folder)

result_dict = {}
with open(input_file) as f:
    for line in f:
        qid, _, _, _ = line.strip().split()
        if qid not in result_dict:
            result_dict[qid] = [line]
        else:
            result_dict[qid].append(line)



for qid in tqdm(result_dict):
    out_file = os.path.join(out_folder, qid + ".qrels")
    with open(out_file, 'w') as f:
        for line in result_dict[qid]:
            f.write(line)

input_file = os.path.join(data_dir, "testing", "data.qrels" )
if not os.path.exists(input_file):
    input_file = os.path.join(data_dir, "data.qrels" )

out_folder = os.path.join(data_dir, "test_qrel" )


if not os.path.exists(out_folder):
    os.makedirs(out_folder)
result_dict = {}
with open(input_file) as f:
    for line in f:
        qid, _, _, _ = line.strip().split()
        if qid not in result_dict:
            result_dict[qid] = [line]
        else:
            result_dict[qid].append(line)



for qid in tqdm(result_dict):
    out_file = os.path.join(out_folder, qid + ".qrels")
    with open(out_file, 'w') as f:
        for line in result_dict[qid]:
            f.write(line)
