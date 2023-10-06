import argparse

import os
from matplotlib import pyplot as plt
from tqdm import tqdm
import json

Instruction = 'Construct a high-quality natural language query for a systematic review topic: {title}. The effectiveness of the query will be determined by its capability to retrieve relevant documents when searching on a semantic-based search engine.'

#metrics = ["last_rel", "ap", "recall1.0", "recall5.0", "recall10.0", "recall20.0", "wss_95", "wss_100"]
collections = ["clef-tar/CLEF-2017/testing", "clef-tar/CLEF-2018/testing"]
    #, "clef-tar/CLEF-2019-dta/training", "clef-tar/CLEF-2019-intervention/training", "sysrev-seed-collection/training"]

#types = ["title_openai", "title_alpaca", "title_bioalpaca"]
types = ["title_alpaca", "titles_alpaca"]


prompt_multi_dict = {}
prompt_single_dict = {}
title_dict = {}

for type in types:

    prompt_file = os.path.join("data", f'generated_from_{type}.jsonl')

    with open(prompt_file) as f:
        for line in f:
            current_dict = json.loads(line)
            generated_queries = current_dict["generated_query"]
            if isinstance(generated_queries, list):
                for index, query in enumerate(generated_queries):
                    prompt_multi_dict[f'{current_dict["id"]}_{index}'] = query
            else:
                prompt_multi_dict[f'{current_dict["id"]}'] = generated_queries

            title_dict[current_dict["id"]] = current_dict["title"]



for collection in tqdm(collections):
    metric = "ap"
    title_eval_file = os.path.join("output_combined/eval", collection, "title", "model_bio_bert",
                                   metric + ".res")
    title_score_dict = {}
    with open(title_eval_file) as f:
        for line in f:
            qid, score = line.strip().split()
            if qid != "all":
                title_score_dict[qid] = float(score)

    result_dict = {}
    out_file = os.path.join("data", collection.replace("/training", ""), f'generation_model_training_{type}.json')
    for type in types:
        #out_folder = os.path.join("output/graph", collection, type+ "_single_multi", "model_bio_bert")
        if type.split("_")[0].endswith("s"):
            refined_type = type.split("_")[0][:-1] + "_" + type.split("_")[1] + "_multi"
            eval_file = os.path.join("output_combined/eval", collection, refined_type , "model_bio_bert", metric + ".res")
        else:
            eval_file = os.path.join("output_combined/eval", collection, type , "model_bio_bert", metric + ".res")
        print(eval_file)
        with open(eval_file) as f:
            for line in f:
                qid, score = line.strip().split()
                if "_" not in qid:
                    continue
                else:
                    original_qid = qid.split('_')[0]
                    if original_qid not in result_dict:
                        result_dict[original_qid] = (qid, float(score))
                    else:

                        if float(score) > result_dict[original_qid][1]:
                            result_dict[original_qid] = (qid, float(score))

    overall_list = []
    with open(out_file, 'w') as f:
        for original_qid in result_dict:
            current_dict = {}
            current_dict["instruction"] = Instruction.format(title=title_dict[original_qid])
            current_dict["input"] = ""
            if result_dict[original_qid][1] >= title_score_dict[original_qid]:
                current_dict["output"] = prompt_multi_dict[result_dict[original_qid][0]]

            else:
                current_dict["output"] = title_dict[original_qid]
            overall_list.append(current_dict)
        json.dump(overall_list, f)