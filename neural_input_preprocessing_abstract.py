import json
import argparse
import os
from transformers import AutoTokenizer
from tqdm import tqdm


parser = argparse.ArgumentParser()

parser.add_argument("--DATA_DIR", type=str, default="data/sysrev-seed-collection")
parser.add_argument("--prompt_type", type=str, default='openai', help="title_alpaca, title or abstract or tiab or boolean")
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
            title = current_dict["title"]  # this is tokenized\
            if "abstract" in current_dict:
                abstract = current_dict["abstract"]  # this is not tokenized
                passage_abstract_dict[pid] = abstract
            tiab = current_dict["title_abstract"]
            passage_title_dict[pid] = title
            passage_tiab_dict[pid] = tiab




prompt_dict = {}

if (args.prompt_type == "title") or (args.prompt_type =="abstract") or (args.prompt_type =="boolean") or (args.prompt_type =="search_name"):
    prompt_seed_file = os.path.join(args.DATA_DIR, f'preprocessed_input_abstract.jsonl')
    with open(prompt_seed_file) as f:
        for line in tqdm(f):
            current_dict = json.loads(line)
            qid = current_dict["id"]
            title = current_dict["title"]
            abstract = current_dict["abstract"]
            bool_query = current_dict["query"]
            prompt_dict[qid] = [title, abstract, bool_query]
            if args.prompt_type == "search_name":
                search_name = current_dict["search_name"]
                prompt_dict[qid] = [title, abstract, bool_query, search_name]


else:
    input_type = args.prompt_type.split("_")[0]
    prompt_seed_file = os.path.join(args.DATA_DIR, f'generated_from_{args.prompt_type}.jsonl')
    print(prompt_seed_file)
    with open(prompt_seed_file) as f:
        for line in tqdm(f):
            current_dict = json.loads(line)
            qid = current_dict["id"]
            if input_type == "title":
                original_info = current_dict["title"]
            elif input_type == "abstract":
                original_info = current_dict["abstract"]
            else:
                original_info = current_dict["boolean"]
            generated_query = current_dict["generated_query"]
            prompt_dict[qid] = [original_info, generated_query]

# below is for creating query index and read inputfile
query_dict = {}
passage_mapping = {}
with open(input_file) as f:

    for line in tqdm(f):
        current_dict = json.loads(line)
        qid = current_dict["id"]
        if args.prompt_type == "title":
            query_dict[qid] = tokenizer.encode(
                prompt_dict[qid][0],
                add_special_tokens=False,
                max_length=64,
                truncation=True,
                padding="max_length"
            )
        elif args.prompt_type == "search_name":
            query_dict[qid] = tokenizer.encode(
                prompt_dict[qid][3],
                add_special_tokens=False,
                max_length=64,
                truncation=True,
                padding="max_length"
            )
        elif args.prompt_type == "abstract":
            query_dict[qid] = tokenizer.encode(
                prompt_dict[qid][1],
                add_special_tokens=False,
                max_length=256,
                truncation=True,
                padding="max_length"
            )
        elif args.prompt_type == "boolean":
            query_dict[qid] = tokenizer.encode(
                prompt_dict[qid][2],
                add_special_tokens=False,
                max_length=256,
                truncation=True,
                padding="max_length"
            )
        else:
            input_type = args.prompt_type.split("_")[0]
            if "title" in input_type:

                query_dict[qid] = tokenizer.encode(
                    prompt_dict[qid][1],
                    add_special_tokens=False,
                    max_length=64,
                    truncation=True,
                    padding="max_length")
            else:
                query_dict[qid] = tokenizer.encode(
                    prompt_dict[qid][1],
                    add_special_tokens=False,
                    max_length=256,
                    truncation=True,
                    padding="max_length")


        search_docs = current_dict["search_docs"]
        passage_mapping[qid] = search_docs

# bwlow is for output

out_folder = os.path.join(args.DATA_DIR, "neural_inputs", args.prompt_type, args.tokeniser)
print(out_folder)
if not os.path.exists(out_folder):
    os.makedirs(out_folder)


out_run_file = os.path.join(out_folder, 'run.jsonl')
out_run_tsv_file = os.path.join(out_folder, 'run.tsv')


out_runs = []
out_runs_tsv = []
for qid, query in tqdm(query_dict.items()):
    if qid in passage_mapping:
            for pid in passage_mapping[qid]:
                if pid not in passage_tiab_dict:
                    continue
                current_dict = {}
                current_dict["qid"] = qid
                current_dict["pid"] = pid
                current_dict["qry"] = query
                current_dict["psg"] = passage_tiab_dict[pid]
                out_runs.append(json.dumps(current_dict) + "\n")
                out_runs_tsv.append(f"{qid}\t{pid}\n")

with open(out_run_file, "w") as out_rerank, open(out_run_tsv_file, "w") as out_rerank_tsv:
    out_rerank_tsv.writelines(out_runs_tsv)
    out_rerank.writelines(out_runs)