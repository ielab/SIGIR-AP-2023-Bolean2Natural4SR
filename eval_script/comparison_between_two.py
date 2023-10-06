import json
import argparse
import os
import glob
from transformers import AutoTokenizer
from tqdm import tqdm
from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import pearsonr

## this file is to compare the results of two different runs, and generate gain-loss plot for each matrics

parser = argparse.ArgumentParser()
parser.add_argument("--DATA_DIR_1", type=str, default="data/sysrev-seed-collection")
parser.add_argument("--DATA_DIR_2", type=str, default="data/sysrev-seed-collection")

parser.add_argument("--cor_file", type=str, default="output/cor.txt")
args = parser.parse_args()


input_files_1 = glob.glob(os.path.join(args.DATA_DIR_1, "*.res"))
input_files_2 = glob.glob(os.path.join(args.DATA_DIR_2, "*.res"))

result_dict_1 = {}
result_dict_2 = {}

cor_dict = {}
with open(args.cor_file) as f:
    for line in f:
        qid, rel, total, ratio = line.strip().split()
        if "rel" not in cor_dict:
            cor_dict["rel"] = {}
        if "total" not in cor_dict:
            cor_dict["total"] = {}
        if "ratio" not in cor_dict:
            cor_dict["ratio"] = {}
        cor_dict["rel"][qid] = int(rel)
        cor_dict["total"][qid] = int(total)
        cor_dict["ratio"][qid] = float(ratio)

for input_file in tqdm(input_files_1):
    metric_name = input_file.split('/')[-1].split('.')[0]
    result_dict_1[metric_name] = {}

    with open(input_file) as f:
        for line in f:
            qid, score = line.strip().split()
            if qid=="all":
                continue
            result_dict_1[metric_name][qid] = float(score)

for input_file in tqdm(input_files_2):
    metric_name = input_file.split('/')[-1].split('.')[0]
    result_dict_2[metric_name] = {}

    with open(input_file) as f:
        for line in f:
            qid, score = line.strip().split()
            if qid=="all":
                continue
            result_dict_2[metric_name][qid] = float(score)


metrics = ["ap",  "wss_100"]

## draw gain-loss plot for each metric
x_axis_order = []
for metric_name in metrics:
    plt.figure()
    result_diff = {}
    for qid in result_dict_1[metric_name]:
        result_diff[qid] = result_dict_1[metric_name][qid] - result_dict_2[metric_name][qid]
    ## if metric name is ap then sort the result_diff by its value, else use the order in x_axis_order
    if metric_name =="ap":
        result_diff = sorted(result_diff.items(), key=lambda x: x[1], reverse=True)
        x_axis_order = [qid for qid, _ in result_diff]
    else:
        result_diff = [(qid, result_diff[qid]) for qid in x_axis_order]
    ##print average
    print(len(result_diff))
    print(f"average difference for {metric_name} is {sum([score for _, score in result_diff])/len(result_diff)}")

    # bar plot for gain-loss
    plt.bar(x_axis_order, [score for _, score in result_diff])
    #x_axis vertical
    plt.xticks(rotation=90)

    #plt.plot([i for i in range(len(result_diff))], [score for _, score in result_diff])
    ##
    plt.xlabel('pid')
    plt.ylabel('score')
    plt.title(metric_name)
    plt.tight_layout()

    plt.savefig(os.path.join("gain_loss", metric_name + ".png"))

    plt.close()


    ## if metric is app, give coorelation between difference and the ratio of relevant documents
    diff = [score for _, score in result_diff]
    file_2_results = [result_dict_2[metric_name][qid] for qid, _ in result_diff]
    file_1_results = [result_dict_1[metric_name][qid] for qid, _ in result_diff]
    ratio = [cor_dict["ratio"][qid] for qid, _ in result_diff]
    rel_num = [cor_dict["rel"][qid] for qid, _ in result_diff]
    total_num = [cor_dict["total"][qid] for qid, _ in result_diff]
    #draw a scatter plot
    plt.figure()
    plt.scatter(diff, ratio)
    ## draw correlation line that is most fit
    x = np.array(diff)
    y = np.array(ratio)
    a, b = np.polyfit(x, y, 1)
    plt.plot(x, a*x + b)

    plt.xlabel("ratio of relevant documents")
    plt.ylabel("difference between two runs")
    plt.title(metric_name)
    plt.tight_layout()
    plt.savefig(os.path.join("gain_loss", metric_name + "cor.png"))
    plt.close()

    print("current metric", metric_name)
    print(f"pearson correlation between first run and ratio ", pearsonr(file_1_results, ratio))
    print(f"pearson correlation between second run and ratio: ", pearsonr(file_2_results, ratio))
    print(f"pearson correlation between difference and ratio of relevant documents: ", pearsonr(diff, ratio))
    print(f"pearson correlation between difference and number of relevant documents: ", pearsonr(diff, rel_num))
    print(f"pearson correlation between difference and number of total documents: ", pearsonr(diff, total_num))









