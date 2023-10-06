import argparse

import os
from matplotlib import pyplot as plt
from tqdm import tqdm
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer



tokenizer = LlamaTokenizer.from_pretrained("out_7b/")

Apaca_PROMPT = 'Below is an instruction that describes a task, paired with an input that provides further context. ' \
               'Write a response that appropriately completes the request.\n\n### ' \
               'Instruction:\n Construct a high-quality natural language query providing the systematic review boolean query.' \
               'The effectiveness of the query will be determined by its capability to retrieve relevant documents when searching on a semantic-based search engine. ' \
               'The generated query can also be the same as the original topic, as long as it can achieve high effectiveness.' \
               '\n\n ### Input:\n{boolean}\n\n ' \
               '### Response:\n'


Instruction = 'Construct a natural language query using the systematic review boolean query provided.'
               # 'The effectiveness of the query will be determined by its capability to retrieve relevant documents when searching on a semantic-based search engine. ' \
               # 'The generated query can also be the same as the original topic, as long as it can achieve high effectiveness.' \

#metrics = ["last_rel", "ap", "recall1.0", "recall5.0", "recall10.0", "recall20.0", "wss_95", "wss_100"]
collections = ["clef-tar/CLEF-2017", "clef-tar/CLEF-2018", "clef-tar/CLEF-2019-dta", "clef-tar/CLEF-2019-intervention", "sysrev-seed-collection"]
    #, "clef-tar/CLEF-2019-dta/training", "clef-tar/CLEF-2019-intervention/training", "sysrev-seed-collection/training"]

#types = ["title_openai", "title_alpaca", "title_bioalpaca"]


init_dict = {"instruction": Instruction}
for collection in collections:
    qrel_file = os.path.join("data", collection, "qrel_abs_train_loo.txt")
    qrel_dict = {}
    with open(qrel_file) as f:
        for line in f:
            qid, _, docid, score = line.strip().split()
            if qid not in qrel_dict:
                qrel_dict[qid] = {}
            qrel_dict[qid][docid] = score

    preprocessed_file = os.path.join("data", "preprocessed_with_abstract.jsonl")

    out_file = os.path.join("data", collection, f'generation_model_boolean_to_title.json')
    with open(out_file, "w") as fw:
        overall_list = []
        with open(preprocessed_file) as f:
            for line in f:
                current_dict = json.loads(line)
                dump_dict = init_dict.copy()
                #prompt_len = len(tokenizer.tokenize(Apaca_PROMPT.format(boolean=current_dict["query"])))
                query = current_dict["query"]
                title = current_dict["title"]
                id = current_dict["id"]
                if id in qrel_dict:
                    dump_dict["input"] = query
                    dump_dict["output"] = title
                    overall_list.append(dump_dict)
        print(len(
            overall_list))
        json.dump(overall_list, fw, indent=4)












