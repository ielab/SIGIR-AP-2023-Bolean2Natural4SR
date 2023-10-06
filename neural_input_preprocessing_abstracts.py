import json
import argparse
import os
from transformers import AutoTokenizer
from tqdm import tqdm


parser = argparse.ArgumentParser()

parser.add_argument("--DATA_DIR", type=str, default="data/sysrev-seed-collection")
parser.add_argument("--prompt_type", type=str, default='openai', help="openai or alpaca or biogpt or real_seed")
parser.add_argument("--tokeniser", type=str, default="dmis-lab/biobert-v1.1")
parser.add_argument("--collection_file", type=str, default="data/sysrev-seed-collection/all.jsonl")
parser.add_argument("--cache_dir", type=str, default="./cache")

args = parser.parse_args()



input_file = os.path.join(args.DATA_DIR, "preprocessed_input.jsonl")
collection_file =  args.collection_file
collection_out_folder = os.path.join(args.collection_file.replace('/all.jsonl', ""), "neural_collections")

collection_index = os.path.join(collection_out_folder, args.tokeniser.replace("/", "_") + ".jsonl")

if not os.path.exists(collection_out_folder):
    os.makedirs(collection_out_folder)

passage_title_dict = {}
passage_abstract_dict = {}
passage_tiab_dict = {}


tokenizer = AutoTokenizer.from_pretrained(args.tokeniser, use_fast=True, cache_dir=args.cache_dir)

# below is for creating collection index
if not os.path.exists(collection_index):
    with open(collection_index, 'w') as fw:
        with open(collection_file) as f:
            for line in tqdm(f):
                current_dict = json.loads(line)
                pid = current_dict["pmid"]
                title = current_dict["title"]
                abstract = current_dict["abstract"]
                title_tokenized = tokenizer.encode(
                    title,
                    add_special_tokens=False,
                    max_length=64,
                    truncation=True,
                    padding="max_length"
                )
                t_ab_tokenized = tokenizer.encode(
                    title + "[SEP]" + abstract,
                    add_special_tokens=False,
                    max_length=448,
                    truncation=True,
                    padding="max_length")
                ab_tokenized = tokenizer.encode(
                    abstract,
                    add_special_tokens=False,
                )

                current_dict["title_tokenized"] = title_tokenized
                current_dict["title_abstract_tokenized"] = t_ab_tokenized
                current_dict["abstract_tokenized"] = ab_tokenized


                fw.write(json.dumps(current_dict) + "\n")
                passage_title_dict[pid] = title_tokenized
                passage_abstract_dict[pid] = ab_tokenized
                passage_tiab_dict[pid] = t_ab_tokenized
else:
    with open(collection_index) as f:
        for line in tqdm(f):
            current_dict = json.loads(line)
            pid = current_dict["pmid"]
            title = current_dict["title"] # this is tokenized
            if "abstract" in current_dict:
                abstract = current_dict["abstract"] # this is not tokenized
                passage_abstract_dict[pid] = abstract
            tiab = current_dict["title_abstract"]
            passage_title_dict[pid] = title
            passage_tiab_dict[pid] = tiab




prompt_dict = {}

if args.prompt_type == "real_seed":
    seed_file = os.path.join(args.DATA_DIR, f'preprocessed_input.jsonl')
    with open(seed_file) as f:
        for line in tqdm(f):
            current_dict = json.loads(line)
            qid = current_dict["id"]
            title = current_dict["title"]
            seed_studies = current_dict["seed_studies"]

            prompt_dict[qid] = {}
            if qid not in prompt_dict:
                prompt_dict[qid] = {}
            for index, s_id in enumerate(seed_studies):
                if s_id not in passage_tiab_dict:
                    continue
                abstract = passage_abstract_dict[s_id]
                query = title + " " + abstract
                prompt_dict[qid][index] = query

else:
    input_type = args.prompt_type.split("_")[0]
    prompt_seed_file = os.path.join(args.DATA_DIR, f'generated_from_{args.prompt_type}.jsonl')
    with open(prompt_seed_file) as f:
        for line in tqdm(f):
            current_dict = json.loads(line)
            qid = current_dict["id"]
            if "title" in current_dict:
                original_info = current_dict["title"]

            elif "abstract" in current_dict:
                original_info = current_dict["abstract"]
            else:
                original_info = current_dict["boolean"]
            generated_query = current_dict["generated_query"]
            prompt_dict[qid] = generated_query


# below is for creating query index and read inputfile
query_dict = {}
passage_mapping = {}

with open(input_file) as f:
    for line in tqdm(f):
        current_dict = json.loads(line)
        qid = current_dict["id"]
        input_type = args.prompt_type.split("_")[0]
        query_list = prompt_dict[qid]
        query_dict[qid] = {}
        if "title" in input_type:
            for index, query in enumerate(query_list):
                query_dict[qid][index] = tokenizer.encode(
                    query,
                    add_special_tokens=False,
                    max_length=64,
                    truncation=True,
                    padding="max_length")
            search_docs = current_dict["search_docs"]
            passage_mapping[qid] = search_docs
        else:
            for index, query in enumerate(query_list):
                query_dict[qid][index] = tokenizer.encode(
                    query,
                    add_special_tokens=False,
                    max_length=256,
                    truncation=True,
                    padding="max_length")
            search_docs = current_dict["search_docs"]
            passage_mapping[qid] = search_docs

# bwlow is for output

if args.prompt_type.split("_")[0].endswith("s"):
    redefined_prompt = args.prompt_type.split("_", 1)[0][:-1] + "_"  + args.prompt_type.split("_", 1)[1]
    out_folder = os.path.join(args.DATA_DIR, "neural_inputs", redefined_prompt + "_multi", args.tokeniser)

if not os.path.exists(out_folder):
    os.makedirs(out_folder)


out_run_file = os.path.join(out_folder, 'run.jsonl')
out_run_tsv_file = os.path.join(out_folder, 'run.tsv')


out_runs = []
out_runs_tsv = []
for qid, q_dict in tqdm(query_dict.items()):
    for index, query in q_dict.items():
        current_dict = {}
        current_dict["qid"] = qid + "_" + str(index)
        current_dict["qry"] = query
        if qid in passage_mapping:
            for pid in passage_mapping[qid]:
                if pid not in passage_tiab_dict:
                    continue
                current_dict["pid"] = pid
                current_dict["psg"] = passage_tiab_dict[pid]
                out_runs.append(json.dumps(current_dict) + "\n")
                out_runs_tsv.append(f"{qid}_{index}\t{pid}\n")

with open(out_run_file, "w") as out_rerank, open(out_run_tsv_file, "w") as out_rerank_tsv:
    out_rerank_tsv.writelines(out_runs_tsv)
    out_rerank.writelines(out_runs)