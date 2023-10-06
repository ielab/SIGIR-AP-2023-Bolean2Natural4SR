import json
import argparse
import os
import glob
from transformers import AutoTokenizer
from tqdm import tqdm


parser = argparse.ArgumentParser()

parser.add_argument("--train_input_qrel", type=str, default="qrel_eval/2017/qrel_abs_train.txt")
parser.add_argument("--type", type=str, default="tiab", help="title, abstract, boolean or title_alpaca or titles_alpaca")
parser.add_argument("--DATA_DIR", type=str, default="title/2019/intervention")
parser.add_argument("--tokeniser", type=str, default="dmis-lab/biobert-v1.1")
parser.add_argument("--collection_index", type=str, default="collection/model_bio_bert.jsonl")
parser.add_argument("--cache_dir", type=str, default="./cache")
parser.add_argument("--prompt_file", type=str, default="data/preprocessed_with_abstract.jsonl")

args = parser.parse_args()
prompt_file = args.prompt_file

tokenizer = AutoTokenizer.from_pretrained(args.tokeniser, use_fast=True, cache_dir=args.cache_dir)

type=args.type


positive_doc_dict = {}
negative_doc_dict = {}


    #out_dir = os.path.join(args.DATA_DIR, type.split("_")[-1] + "_"+ prompt_type + "_multi", "train", args.collection_index.split("/")[-1].split(".")[0])
if type.split("_")[0].endswith("s"):
    redefined_type = type.replace("titles", "title").replace("abstracts", "abstract").replace("booleans", "boolean")

    out_inference_dir = os.path.join(args.DATA_DIR.split('/testing')[0], "training", "inference", redefined_type + "_multi",  args.collection_index.split("/")[-1].split(".")[0])
else:
    out_inference_dir = os.path.join(args.DATA_DIR.split('/testing')[0], "training", "inference",
                                     type , args.collection_index.split("/")[-1].split(".")[0])

#
# if not os.path.exists(out_dir):
#     os.makedirs(out_dir)
if not os.path.exists(out_inference_dir):
    os.makedirs(out_inference_dir)

#out_file = os.path.join(out_dir, "train.tsv")
out_inference_input = os.path.join(out_inference_dir, "run.jsonl")
out_inference_tsv_input = os.path.join(out_inference_dir, "run.tsv")

collection_index = args.collection_index

passage_title_dict = {}
passage_dict = {}
with open(collection_index) as f:
    for line in tqdm(f):
        current_dict = json.loads(line)
        pid = current_dict["pmid"]
        title = current_dict["title"]
        t_ab = current_dict["title_abstract"]
        if len(t_ab) == 0:
            continue
        passage_title_dict[pid] = title
        passage_dict[pid] = t_ab

with open(args.train_input_qrel) as f:
    for line in tqdm(f):
        qid, _, pid, rel = line.split()
        if qid not in negative_doc_dict:
            negative_doc_dict[qid] = []
        if qid not in positive_doc_dict:
            positive_doc_dict[qid] = []
        if int(rel)==1:
            positive_doc_dict[qid].append(pid)
        elif int(rel)==0:
            negative_doc_dict[qid].append(pid)
        else:
            print(rel)


title_abstract_dict = {}

if (type=="title") or (type=="abstract") or (type=="boolean"):
    with open(prompt_file) as f:
        for line in tqdm(f):
            current_dict = json.loads(line)
            qid = current_dict["id"]
            title = current_dict["title"]
            abstract = current_dict["abstract"]
            bool_query = current_dict["query"]
            title_abstract_dict[qid] = [title ,abstract, bool_query]
else:
    with open(prompt_file) as f:
        for line in tqdm(f):
            current_dict = json.loads(line)
            qid = current_dict["id"]
            generated_query = current_dict["generated_query"]
            if isinstance(generated_query, list):
                title_abstract_dict[qid] = generated_query
            else:
                title_abstract_dict[qid] = [generated_query]



out_runs = []
out_runs_tsv = []

for qid in tqdm(positive_doc_dict):
    positive_ids = positive_doc_dict[qid]
    negative_ids = negative_doc_dict[qid]
    query_list = []
    if (type=="title") or (type=="abstract") or (type=="boolean"):
        if type=="title":
            query = title_abstract_dict[qid][0]
            query_tokenized = tokenizer.encode(
                query,
                add_special_tokens=False,
                max_length=64,
                truncation=True,
                padding="max_length")
        elif type=="abstract":
            query = title_abstract_dict[qid][1]
            query_tokenized = tokenizer.encode(
                query,
                add_special_tokens=False,
                max_length=256,
                truncation=True,
                padding="max_length")
        else:
            query = title_abstract_dict[qid][2]
            query_tokenized = tokenizer.encode(
                query,
                add_special_tokens=False,
                max_length=256,
                truncation=True,
                padding="max_length")

        current_dict = {}
        current_dict["qid"] = qid
        current_dict["qry"] = query_tokenized
        for pid in negative_ids:
            if pid not in passage_dict:
                continue
            current_dict["pid"] = pid
            current_dict["psg"] = passage_dict[pid]
            out_runs.append(json.dumps(current_dict) + "\n")
            out_runs_tsv.append(f"{qid}\t{pid}\n")
        for pid in positive_ids:
            if pid not in passage_dict:
                continue
            current_dict["pid"] = pid
            current_dict["psg"] = passage_dict[pid]
            out_runs.append(json.dumps(current_dict) + "\n")
            out_runs_tsv.append(f"{qid}\t{pid}\n")

    else:
        generated_queries = title_abstract_dict[qid]

        
        for index, query in enumerate(generated_queries):
            query_tokenized = tokenizer.encode(
                query,
                add_special_tokens=False,
                max_length=64,
                truncation=True,
                padding="max_length")
            current_dict = {}
            current_dict["qid"] = qid + "_" + str(index)
            current_dict["qry"] = query_tokenized
            for pid in negative_ids:
                if pid not in passage_dict:
                    continue
                current_dict["pid"] = pid
                current_dict["psg"] = passage_dict[pid]
                out_runs.append(json.dumps(current_dict) + "\n")
                out_runs_tsv.append(f"{qid}_{index}\t{pid}\n")
            for pid in positive_ids:
                if pid not in passage_dict:
                    continue
                current_dict["pid"] = pid
                current_dict["psg"] = passage_dict[pid]
                out_runs.append(json.dumps(current_dict) + "\n")
                out_runs_tsv.append(f"{qid}_{index}\t{pid}\n")


with open(out_inference_input, "w") as out_rerank, open(out_inference_tsv_input, "w") as out_rerank_tsv:
    out_rerank_tsv.writelines(out_runs_tsv)
    out_rerank.writelines(out_runs)


