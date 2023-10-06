import argparse
import os
import glob
from subprocess import Popen, PIPE, STDOUT
from tqdm import tqdm
parser = argparse.ArgumentParser()
parser.add_argument("--DATA_DIR", type=str, default="output/data/sysrev-seed-collection/title/BM25")
parser.add_argument("--trec_eval", type = str, default="trec_eval/trec_eval")
parser.add_argument("--tar_eval", type = str, default="tar/scripts/tar_eval_2018.py 2 ")
#parser.add_argument("--type", type = str, default="train")
args = parser.parse_args()

input_files = glob.glob(args.DATA_DIR+"/*.trec")


trec_eval = args.trec_eval
tar_eval = args.tar_eval

out_folder = args.DATA_DIR.replace("output/data", "output/eval")
if not os.path.exists(out_folder):
    os.makedirs(out_folder)

eval_dict = {}
eval_dict["recall_10"] = out_folder + "/recall_10.res"
eval_dict["recall_100"] = out_folder + "/recall_100.res"
eval_dict["recall_1000"] = out_folder + "/recall_1000.res"
eval_dict["P_10"] = out_folder + "/P_10.res"
eval_dict["P_100"] = out_folder + "/P_100.res"
eval_dict["P_1000"] = out_folder + "/P_1000.res"
eval_dict["map"] = out_folder + "/map.res"
eval_dict["ndcg_cut_10"] = out_folder + "/ndcg_cut_10.res"
eval_dict["ndcg_cut_100"] = out_folder + "/ndcg_cut_100.res"
eval_dict["ndcg_cut_1000"] = out_folder + "/ndcg_cut_1000.res"
eval_tar_dict = {}
eval_tar_dict["last_rel"] = out_folder + "/last_rel.res"
eval_tar_dict["ap"] = out_folder + "/ap.res"
eval_tar_dict["recall1.0"] = out_folder + "/recall1.res"
eval_tar_dict["recall5.0"] = out_folder + "/recall5.res"
eval_tar_dict["recall10.0"] = out_folder + "/recall10.res"
eval_tar_dict["recall20.0"] = out_folder + "/recall20.res"
eval_tar_dict["wss_95"] = out_folder + "/wss95.res"
eval_tar_dict["wss_100"] = out_folder + "/wss100.res"





# below is to evaluate the trec results using trec_eval
# for eval_r in tqdm(eval_dict):
#     result_dict = {}
#     for input_file in input_files:
#         id = input_file.split("/")[-1].split(".")[0]
#         command = trec_eval + " -m " + eval_r.replace("_1", ".1") + " " + qrel_file + " " + input_file
#
#         results = Popen(command, stdout=PIPE, stderr=PIPE, universal_newlines=True, shell=True).stdout.readlines()
#         for result in results:
#             items = result.split()
#             if (len(items) == 3) and (items[1]=="all"):
#                 if id not in result_dict:
#                     result_dict[id] = items[-1]
#     rev_o = open(eval_dict[eval_r], 'w')
#     sum_value = 0
#     for key in result_dict:
#         rev_o.write(key + "\t" + result_dict[key] + "\n")
#         sum_value += float(result_dict[key])
#
#     rev_o.write("all" + "\t" + str(sum_value/len(result_dict)) + "\n")
#     print(eval_r, str(sum_value/len(result_dict)))
#     rev_o.close()

#below is for tar evaluation
result_dict = {}

for input_file in tqdm(input_files):
    id = input_file.split("/")[-1].split(".")[0]

    if "train" in args.DATA_DIR.replace("multi_train", ""):
        qrel_path = os.path.join(args.DATA_DIR.split("/training")[0].split("output/")[1], "train_qrel_loo", id.split('_')[0] + '.qrels')
        command_tar = "python3 "  + tar_eval + " " + qrel_path + " " + input_file
    else:
        qrel_path = os.path.join(args.DATA_DIR.split("/testing")[0].split("output/")[1], "test_qrel", id.split('_')[0] + '.qrels')
        command_tar = "python3 "  + tar_eval + " " + qrel_path + " " + input_file

    #print(command_tar)

    results_tar = Popen(command_tar, stdout=PIPE, stderr=PIPE, universal_newlines=True, shell=True).stdout.readlines()
    for result in results_tar:
        #print(result)

        items = result.split()
        if (len(items) == 3) and (items[0] == "ALL"):
            i = items[1].replace("@", '').replace("%", '')
            if i not in result_dict:
                result_dict[i] = {}
            result_dict[i][id] = float(items[-1])


for eval_r in eval_tar_dict:
    max_dict = {}
    rev_o = open(os.path.join(out_folder, eval_r + '.res'), 'w')
    sum_value = 0
    len_sum = 0

    for key in result_dict[eval_r]:
        rev_o.write(key + "\t" + str(result_dict[eval_r][key]) + "\n")
        if "_" not in key:
            sum_value += result_dict[eval_r][key]
            len_sum +=1
        else:
            if key.split("_")[0] not in max_dict:
                max_dict[key.split("_")[0]] = float(result_dict[eval_r][key])
            else:
                if float(result_dict[eval_r][key]) > max_dict[key.split("_")[0]]:
                    max_dict[key.split("_")[0]] = float(result_dict[eval_r][key])

    if len_sum==0:
        final_value = str(sum(max_dict.values())/len(max_dict))
    else:
        final_value = str(sum_value/len_sum)
    rev_o.write("all" + "\t" + final_value + "\n")

    if eval_r=="ap":
        print(eval_r, final_value)
    rev_o.close()



