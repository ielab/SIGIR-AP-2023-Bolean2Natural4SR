import argparse
import glob
import os
from tqdm import tqdm


parser = argparse.ArgumentParser()

parser.add_argument("--DATA_DIR_1", type=str, default="data/sysrev-seed-collection")
parser.add_argument("--DATA_DIR_2", type=str, default="data/sysrev-seed-collection")

args = parser.parse_args()


files_1 = glob.glob(os.path.join(args.DATA_DIR_1, "*.trec"))
files_2 = glob.glob(os.path.join(args.DATA_DIR_2, "*.trec"))

method_1 = args.DATA_DIR_1.split('/')[-2]

if ("fusion_mean" in args.DATA_DIR_1):
    method_1 = args.DATA_DIR_1.split('/')[-3]

method_2 = args.DATA_DIR_2.split('/')[-2]
if ("fusion_mean" in args.DATA_DIR_2):
    method_2 = args.DATA_DIR_2.split('/')[-3]

base_dir = args.DATA_DIR_1.split("/testing")[0] + "/testing"

out_folder = os.path.join(base_dir, f"interpolate_{method_1}_{method_2}", "model_bio_bert")
print(out_folder)
if not os.path.exists(out_folder):
    os.makedirs(out_folder)

result_dict_1 = {}
for file_1 in tqdm(files_1):
    qid = file_1.split('/')[-1].split('.')[0]
    original_qid = qid.split('_')[0]
    with open(file_1) as f:
        for line in f:
            qid, _, pid, rank, score, _ = line.strip().split()

            if qid not in result_dict_1:
                result_dict_1[qid] = {}
            result_dict_1[qid][pid] = float(score)

result_dict_2 = {}

for file_2 in tqdm(files_2):
    qid = file_2.split('/')[-1].split('.')[0]
    with open(file_2) as f:
        for line in f:
            qid, _, pid, rank, score, _ = line.strip().split()
            if qid not in result_dict_2:
                result_dict_2[qid] = {}

            result_dict_2[qid][pid] = float(score)


for qid in tqdm(result_dict_1):
    current_dict = {}
    max_1 = max(result_dict_1[qid].values())
    max_2 = max(result_dict_2[qid].values())
    min_1 = min(result_dict_1[qid].values())
    min_2 = min(result_dict_2[qid].values())

    for pid in result_dict_1[qid]:
        norm_1 = (result_dict_1[qid][pid] - min_1) / (max_1 - min_1)
        norm_2 = (result_dict_2[qid][pid] - min_2) / (max_2 - min_2)
        current_dict[pid] = (norm_1 + norm_2) / 2

    current_dict = sorted(current_dict.items(), key=lambda x: x[1], reverse=True)
    with open(os.path.join(out_folder, qid + ".trec"), 'w') as f:
        for i, (pid, score) in enumerate(current_dict):
            f.write(f"{qid} Q0 {pid} {i+1} {score} fuse\n")




