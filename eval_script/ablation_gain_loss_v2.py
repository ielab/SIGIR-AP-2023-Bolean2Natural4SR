import glob
import os
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
import matplotlib.patches as mpatches

metrics = ["ap"]
collections = ["clef-tar/CLEF-2017/testing", "clef-tar/CLEF-2018/testing", "clef-tar/CLEF-2019-dta/testing",
               "clef-tar/CLEF-2019-intervention/testing", "sysrev-seed-collection/testing"]
# collections = ["clef-tar/CLEF-2017/training", "clef-tar/CLEF-2018/training", "clef-tar/CLEF-2019-dta/training", "clef-tar/CLEF-2019-intervention/training", "sysrev-seed-collection/training"]
# collections = ["clef-tar/CLEF-2018/training"]

collection_dict = {
    "clef-tar/CLEF-2017/testing": "CLEF-2017",
    "clef-tar/CLEF-2018/testing": "CLEF-2018",
    "clef-tar/CLEF-2019-dta/testing": "CLEF-2019-dta",
    "clef-tar/CLEF-2019-intervention/testing": "CLEF-2019-int",
    "sysrev-seed-collection/testing": "Seed-Collection"}

types = ["boolean_openai", "boolean_alpaca_tuned_query"]  # , "title_bioalpaca"]

type_dict = {
    "boolean_openai": "ChatGPT",
    "boolean_alpaca_tuned_query": "Alpaca",
}

# types = ["alternativetitle_openai"]
# "title_openai", "title_alpaca", "title_bioalpaca"]

a = 0
b = 0

type_2 = "_multi"


final_dict = {}

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
    title_mean = sum([title_dict[k] for k in title_dict]) / len(title_dict)

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

        result_dict_single = {}
        eval_file = os.path.join("combined_output/eval", collection, "interpolate_boolean_"+type, "model_bio_bert",
                                 metric + ".res")

        with open(eval_file) as f:
            for line in f:
                qid, score = line.strip().split()
                if qid != "all":
                    result_dict_single[qid] = float(score)


        result_dict_query_single = {}
        eval_file_query_single = os.path.join("combined_output/eval", collection, type,
                                      "model_bio_bert",
                                      metric + ".res")

        with open(eval_file_query_single) as f:
            for line in f:
                qid, score = line.strip().split()
                if qid != "all":
                    result_dict_query_single[qid] = float(score)

        result_single_mean = sum(result_dict_single.values()) / len(result_dict_single)
        result_query_single_mean = sum(result_dict_query_single.values()) / len(result_dict_query_single)


        if type not in final_dict:
            final_dict[type] = {}
        final_dict[type][collection] = [boolean_mean-result_single_mean, result_query_single_mean-result_single_mean]

        #final_list = [boolean_mean-result_single_mean, result_query_single_mean-result_single_mean]


        #bars = axs[i].bar(range(len(final_list)), final_list, color=['red', 'green'])
        #axs[i].set_xticks(range(len(final_list)))



outfile = "combined_output/graph/ablation_gain_loss.pdf"


fig, axs = plt.subplots(2, figsize=(19, 10))
colors = {"Boolean": "darkgray", "NLQ": "dimgray"}
type_color_dict = dict(zip(types, colors))

for ax, (model_name, results) in zip(axs, final_dict.items()):
    collections = list(results.keys())
    values = list(results.values())

    bar_width = 0.2
    index = np.arange(len(collections))*0.6

    rects1 = ax.bar(index - bar_width / 2, [v[0] for v in values], bar_width, color='darkgray')
    rects2 = ax.bar(index + bar_width / 2, [v[1] for v in values], bar_width, color='dimgray')
    ax.axhline(0, color='black', linestyle='--')
    ax.tick_params(axis='y', labelsize=14)
    ax.set_ylim(-0.065, 0.04)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    if model_name == "boolean_openai":
        labels = (collection_dict[collection] for collection in collections)
        ax.set_xticklabels(labels, fontsize=24)

    else:
        ax.set_xticklabels([])

    ax.set_ylabel('\u0394MAP', fontsize=26)

    ax.set_xticks(index)

    ax.text(0.5, 0.95, type_dict[model_name], ha='center', va='top', transform=ax.transAxes, fontsize=24)
plt.subplots_adjust(hspace=0.1)
patches = [mpatches.Patch(color=color, label=type) for type, color in colors.items()]
plt.subplots_adjust(bottom=0.09)
fig.legend(handles=patches, loc='lower center', ncol=2, fontsize=24)

#fig.tight_layout()
plt.show()

plt.savefig(outfile)
