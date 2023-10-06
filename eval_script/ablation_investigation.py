import os
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerPatch

from matplotlib.legend_handler import HandlerPatch

# custom legend handler
class HandlerHatch(HandlerPatch):
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        # the position of the legend marker
        xy = (0, 0 * height - 0.5 * ydescent)
        p = mpatches.Rectangle(xy=xy, width=width, height=height,
                               hatch=orig_handle.get_hatch(),
                               fill=True, facecolor=orig_handle.get_facecolor())
        self.update_prop(p, orig_handle, legend)
        p.set_transform(trans)
        return [p]


metrics = ["ap"]
collections = ["clef-tar/CLEF-2017/testing", "clef-tar/CLEF-2018/testing", "clef-tar/CLEF-2019-dta/testing", "clef-tar/CLEF-2019-intervention/testing", "sysrev-seed-collection/testing"]

collection_dict = {
    "clef-tar/CLEF-2017/testing": "CLEF-2017",
    "clef-tar/CLEF-2018/testing": "CLEF-2018",
    "clef-tar/CLEF-2019-dta/testing": "CLEF-2019-dta",
    "clef-tar/CLEF-2019-intervention/testing": "CLEF-2019-int",
    "sysrev-seed-collection/testing": "Seed-Collection"}


types = ["boolean_openai", "boolean_openai_multi", "boolean_openai_multi_train", "boolean_openai_multi_multi_train"]#, "title_bioalpaca"]

type_dict = {
    "boolean_openai": "single_train_single_inference",
    "boolean_openai_multi": "single_train_multi_inference",
    "boolean_openai_multi_train": "multi_train_single_inference",
    "boolean_openai_multi_multi_train": "multi_train_multi_inference",
   }
a = 0
b = 0
#types = ["alternativetitle_openai"]
    #"title_openai", "title_alpaca", "title_bioalpaca"]
colors = ['silver', 'darkgray', 'gray', 'dimgray']
hatches = ['/', '.', '|', '+']

type_color_dict = dict(zip(types, colors))
hatches = dict(zip(types, hatches))
fig, axs = plt.subplots(1, len(collections), figsize=(30,10), sharey=True)
axs[0].set_yticks(np.arange(0, 0.6, 0.1))
#y from 0 to 1, font size 30
axs[0].tick_params(axis='y', labelsize=32)
axs[0].set_ylim([0, 0.5])

outfile = "combined_output/graph/combined_graph_ablation.pdf"

for j, collection in enumerate(collections):
    metric = "ap"
    result_dict = {}
    result_individual = {}
    for i, type in enumerate(types):
        if (type =="boolean_openai") or (type=="boolean_openai_multi_train"):
        #out_file = os.path.join(out_folder, metric+ ".pdf")
            eval_file = os.path.join("combined_output/eval", collection, type, "model_bio_bert",
                                 metric + ".res")
            with open(eval_file) as f:
                for line in f:
                    qid, score = line.strip().split()
                    if qid == "all":
                        result_dict[type] = float(score)
                    else:
                        if qid not in result_individual:
                            result_individual[qid] = {}
                        result_individual[qid][type] = float(score)
        else:
            eval_file = os.path.join("combined_output/eval", collection, type, "model_bio_bert", "fusion_mean",
                                 metric + ".res")
            with open(eval_file) as f:
                for line in f:
                    qid, score = line.strip().split()
                    if qid == "all":
                        result_dict[type] = float(score)
                    else:
                        if qid not in result_individual:
                            result_individual[qid] = {}
                        result_individual[qid][type] = float(score)

    for qid in result_individual:
        if result_individual[qid]["boolean_openai_multi"] >= result_individual[qid]["boolean_openai"]:
            a += 1
        b += 1

    #draw a bar chart
    bars = axs[j].bar(range(len(result_dict)), list(result_dict.values()), align='center',
                      color=[type_color_dict[t] for t in types], hatch=[hatches[t] for t in types])
    # draw the maximum line, with the same color as the maximum bar
    max_value = max(result_dict.values())
    max_index = list(result_dict.values()).index(max_value)
    axs[j].axhline(y=max_value, color=colors[max_index], linestyle='--', linewidth=3)


    axs[j].set_title(collection_dict[collection], fontsize=38)


    axs[j].set_xticks([])

    if j == 0:
        axs[j].set_ylabel("MAP", fontsize=38)


plt.subplots_adjust(wspace=0.05)

patches = [mpatches.Patch(facecolor=color, label=type_dict[type], hatch=hatches[type]) for type, color in type_color_dict.items()]
plt.subplots_adjust(bottom=0.25)
handler_map = {mpatches.Patch: HandlerHatch()}

legend = fig.legend(handles=patches,handler_map=handler_map, loc='lower center', ncol=len(types)/2, fontsize=38)


# Update hatches for the patches in the legend


print("aaaa")


#plt.tight_layout()
plt.savefig(outfile)

print(a/b)

