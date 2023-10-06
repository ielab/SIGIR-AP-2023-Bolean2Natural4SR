import glob
import os
from matplotlib import pyplot as plt
from tqdm import tqdm
import matplotlib.lines as mlines
import numpy as np

metrics = ["ap"]
collections = ["clef-tar/CLEF-2017/testing", "clef-tar/CLEF-2018/testing", "clef-tar/CLEF-2019-dta/testing", "clef-tar/CLEF-2019-intervention/testing", "sysrev-seed-collection/testing"]
#collections = ["clef-tar/CLEF-2017/training", "clef-tar/CLEF-2018/training", "clef-tar/CLEF-2019-dta/training", "clef-tar/CLEF-2019-intervention/training", "sysrev-seed-collection/training"]
#collections = ["clef-tar/CLEF-2018/training"]

collection_dict = {
    "clef-tar/CLEF-2017/testing": "CLEF-2017",
    "clef-tar/CLEF-2018/testing": "CLEF-2018",
    "clef-tar/CLEF-2019-dta/testing": "CLEF-2019-dta",
    "clef-tar/CLEF-2019-intervention/testing": "CLEF-2019-intervention",
    "sysrev-seed-collection/testing": "Seed-Collection"}



types = ["boolean_openai", "boolean_alpaca_tuned_query"]#, "title_bioalpaca"]

type_dict = {
    "boolean_openai": "ChatGPT",
    "boolean_alpaca_tuned_query": "Alpaca",
   }

#types = ["alternativetitle_openai"]
    #"title_openai", "title_alpaca", "title_bioalpaca"]

a = 0
b = 0

type_2 = "_multi"

fig, axs = plt.subplots(len(types), len(collections), figsize=(38.5,15), sharey=True)


outfile = "combined_output/graph/combined_graph.pdf"

for j, collection in enumerate(collections):
    metric = "ap"
    title_eval_file = os.path.join("combined_output/eval", collection, "title", "model_bio_bert",
                                   metric + ".res")
    title_dict = {}
    with open(title_eval_file) as f:
        for line in f:
            qid, score = line.strip().split()
            if qid != "all":
                title_dict[qid] = float(score)
    title_mean = sum([title_dict[k] for k in title_dict])/len(title_dict)

    boolean_eval_file = os.path.join("combined_output/eval", collection, "boolean", "model_bio_bert",
                                   metric + ".res")
    boolean_dict = {}
    with open(boolean_eval_file) as f:
        for line in f:
            qid, score = line.strip().split()
            if qid != "all":
                boolean_dict[qid] = float(score)
    boolean_mean = sum([boolean_dict[k] for k in boolean_dict]) / len(boolean_dict)

    for i, type in enumerate(types):
        out_folder = os.path.join("combined_output/graph", collection, type+ type_2, "model_bio_bert")
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)
        #out_file = os.path.join(out_folder, metric+ ".pdf")

        result_dict_single = {}
        eval_file = os.path.join("combined_output/eval", collection, type, "model_bio_bert",
                                 metric + ".res")
        with open(eval_file) as f:
            for line in f:
                qid, score = line.strip().split()
                if qid != "all":
                    result_dict_single[qid] = float(score)


        result_dict = {}
        eval_file = os.path.join("combined_output/eval", collection, type + type_2, "model_bio_bert", metric + ".res")
        with open(eval_file) as f:
            for line in f:
                qid, score = line.strip().split()
                if "_" not in qid:
                    continue
                else:
                    original_qid = qid.split('_')[0]
                    if original_qid not in result_dict:
                        result_dict[original_qid] = [float(score)]
                    else:
                        result_dict[original_qid].append(float(score))

        result_dict_mean = {}
        eval_file_fuse = os.path.join("combined_output/eval", collection, type + type_2, "model_bio_bert", "fusion_mean",
                                 metric + ".res")
        with open(eval_file_fuse) as f:
            for line in f:
                qid, score = line.strip().split()
                if qid != "all":
                    result_dict_mean[qid] = float(score)


        sorted_dict = {k: v for k, v in sorted(result_dict.items(), key=lambda item: sum(item[1])/len(item[1]))}




        result_single_mean = sum(result_dict_single.values())/len(result_dict_single)
        result_fuse_mean = sum(result_dict_mean.values())/len(result_dict_mean)

        if metric == "last_rel":
            max_metric_qids = [k for k, x in sorted_dict.items() if min(x) < title_dict[k]]
            max_metric_value = sum([min(x) for k, x in sorted_dict.items()]) / len(sorted_dict)
        else:
            max_metric_qids = [k for k, x in sorted_dict.items() if max(x) > title_dict[k]]
            max_metric_value = sum([max(x) for k, x in sorted_dict.items()])/len(sorted_dict)

        # calculate variability varaince
        va_list = []
        for qid in sorted_dict:
            va_list.append(np.var(sorted_dict[qid]))


        if "openai" in type:
            for qid in result_dict_mean:
                if result_dict_mean[qid] > result_dict_single[qid]:
                    a += 1
                b += 1

        #title_mean = sum([title_dict[k] for k in sorted_dict])/len(sorted_dict)

        print(collection, type, metric, len(max_metric_qids), max_metric_value, title_mean, sum(va_list))

        #plt.figure(figsize=(6, 6))
        #plt.ylim(0, 1)
        axs[i, j].boxplot(sorted_dict.values(), vert=True, patch_artist=True,
                                   positions=range(len(sorted_dict)))

        axs[i, j].plot([boolean_mean] * len(sorted_dict), label='Average Boolean Performance', color='red')
        #plt.plot([title_mean] * len(sorted_dict), label='Average Title Performance')
        axs[i, j].plot([result_single_mean] * len(sorted_dict), label='Average Single Generation  Performance', color='green')
        axs[i, j].plot([result_fuse_mean] * len(sorted_dict), label='Average Multi-Generations Fusion Performance', color='orange')
        axs[i, j].plot([max_metric_value] * len(sorted_dict), label='Oracle Multi-Generations Performance', color='blue')
        if i == 0:
            axs[i, j].set_title(collection_dict[collection], size=35)

        if j == 0:
            axs[i, j].set_ylabel(type_dict[type]  + "     " + "MAP", size=28)
            # y from 0 to 1, font size 30
            axs[i, j].tick_params(axis='y', labelsize=24)
            axs[i, j].set_ylim(0, 1)

        if i == 1:
            if j == len(types):
                axs[i, j].set_xlabel("Systematic Review Topics", size=28)



        # set y ticks size

        axs[i, j].tick_params(axis='x', labelsize=13)
        axs[i, j].set_xticklabels([], rotation=90)
        #remove tick qid but keep the label
        axs[i, j].set_xticks(range(len(sorted_dict)))
plt.subplots_adjust(wspace=0.1, hspace=0.12)

line1 = mlines.Line2D([], [], color='red', label='Boolean')
line2 = mlines.Line2D([], [], color='green', label='Single-Generation')
line3 = mlines.Line2D([], [], color='orange', label='Multi-Generations Fusion')
line4 = mlines.Line2D([], [], color='blue', label='Multi-Generations Oracle')
    #line5 = mlines.Line2D([], [], color='black', label='Average Title Performance')

    # Add a legend to the figure with the custom lines
    #legend location

fig.legend(handles=[line1, line2, line3, line4], loc='lower center', ncol=4, fontsize=28)

#plt.tight_layout()
plt.savefig(outfile)

print(a/b)




