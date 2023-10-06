from gensim.summarization import bm25
from baseline_methods.qlm import QLM
import numpy
import argparse
import os
import json
from tqdm import tqdm
from nltk.corpus import stopwords
import string
import nltk
nltk.download('stopwords')
cachedStopwords = set(tok.lower() for tok in stopwords.words("english"))
string_set =set(string.punctuation)

# if you vector file is in binary format, change to binary=True

def COMBSUM(pid_scores_dict):
    for pid in pid_scores_dict:
        pid_scores_dict[pid] = sum(pid_scores_dict[pid])
    #next sort pid_scores_dict based on the score
    sorted_pid_scores = sorted(pid_scores_dict.items(), key=lambda kv: kv[1], reverse=True)
    sorted_pid_scores_dict = {}
    for i in range(len(sorted_pid_scores)):
        sorted_pid_scores_dict[sorted_pid_scores[i][0]] = [sorted_pid_scores[i][1]]
    return sorted_pid_scores_dict


def tokenise_collection(collection_file, collection_output_file):
    doc_dict = {}
    with open(collection_output_file, 'w') as fw:
        with open(collection_file) as f:
            for line in tqdm(f):
                datalist = json.loads(line)
                id = datalist["pmid"]
                title = datalist["title"]
                abstract = datalist["abstract"]
                for i in string_set:
                    title = title.replace(i, '')
                for i in string_set:
                    abstract = abstract.replace(i, '')
                title_tokenized = [tok for tok in title.split() if tok.lower() not in cachedStopwords]
                abstract_tokenized = [tok for tok in abstract.split() if tok.lower() not in cachedStopwords]
                fw.write(json.dumps({"pmid": id, "title": title_tokenized, "abstract": abstract_tokenized})+"\n")
                doc_dict[id] = [title_tokenized, abstract_tokenized]
    return doc_dict


def get_collection(collection_file):
    doc_dict = {}
    with open(collection_file) as f:
        for line in tqdm(f):
            datalist = json.loads(line)
            id = datalist["pmid"]
            title = datalist["title"]
            abstract = datalist["abstract"]
            doc_dict[id] = [title, abstract]
    return doc_dict

def build_corpus_model(overall_dic):
    df = {}
    corpus_size = 0
    for key in overall_dic:
        list_tokens = overall_dic[key]
        for word in list_tokens:
            if word not in df:
                df[word] = 0
            df[word] += 1
            corpus_size += 1

    return df, corpus_size


def build_bm25_model(result_dic):
    corpus = []
    pids = []
    for key in result_dic:
        pids.append(key)
        list_tokens = result_dic[key]
        corpus.append(list_tokens)
    #print(corpus)
    model = bm25.BM25(corpus)
    average_idf = sum(float(val) for val in model.idf.values()) / len(model.idf)
    return pids, model, average_idf

def build_qlm_model(result_dic, query_dic):
    corpus = []
    pids = []
    for key in result_dic:
        pids.append(key)
        list_tokens = result_dic[key]
        corpus.append(list_tokens)
    model = QLM(corpus, query_dic)
    return pids, model


def bm25_rerank_results(query_dict, passage_dict, map_q_p, out_folder, fuse_type):
    for qid in tqdm(query_dict):
        map_list = map_q_p[qid]
        output_file = os.path.join(out_folder, qid + '.trec')
        output = open(output_file, 'w')
        result_dict = {}
        for pid in map_list:
            if pid in passage_dict:
                result_dict[pid] = passage_dict[pid]
        #print(len(result_dict))
        queries = query_dict[qid]
        pid_scores_dict = {}
        for query in queries:
            pids, model, average_idf = build_bm25_model(result_dict)
            scores = model.get_scores(query, average_idf)
            indices = sorted(range(len(scores)), key=lambda k: scores[k])[::-1]
            diff = max(scores)-min(scores)
            min_s = min(scores)
            for i in indices:
                if pids[i] not in pid_scores_dict:
                    pid_scores_dict[pids[i]] = []
                if diff==0:
                    pid_scores_dict[pids[i]].append(1)
                else:
                    pid_scores_dict[pids[i]].append((scores[i]-min_s)/diff)
        if len(queries)>1:
            if fuse_type == 'COMBSUM':
                pid_scores_dict = COMBSUM(pid_scores_dict)
            elif fuse_type == 'COMBMNZ':
                pid_scores_dict = COMBMNZ(pid_scores_dict)
            elif fuse_type == 'BORDA':
                pid_scores_dict = BORDA(pid_scores_dict)
        order_index = 1
        write_lines = []
        for pid_ordered in pid_scores_dict:
            write_lines.append(qid + ' 0 ' + pid_ordered + ' ' + str(order_index) + ' ' + str(pid_scores_dict[pid_ordered][0]) + ' BM25\n')
            order_index = order_index + 1
        output.writelines(write_lines)

def qlm_rerank_results(query_dict, passage_dict, map_q_p, out_folder, fuse_type):
    df_dic, corpus_size = build_corpus_model(passage_dict)
    for qid in tqdm(query_dict):
        map_list = map_q_p[qid]
        output_file = os.path.join(out_folder, qid + '.trec')
        output = open(output_file, 'w')
        result_dict = {}
        for pid in map_list:
            if pid in passage_dict:
                result_dict[pid] = passage_dict[pid]
        queries = query_dict[qid]
        pid_scores_dict = {}
        for query in queries:
            query_dictionary = {}
            for word in query:
                if word not in query_dictionary:
                    query_dictionary[word] = 0
                query_dictionary[word] += 1
            pids, model = build_qlm_model(result_dict, query_dictionary)
            scores = model.get_scores(query_dictionary, df_dic, corpus_size)
            indices = sorted(range(len(scores)), key=lambda k: scores[k])[::-1]
            diff = max(scores) - min(scores)
            min_s = min(scores)
            for i in indices:
                if pids[i] not in pid_scores_dict:
                    pid_scores_dict[pids[i]] = []
                if diff == 0:
                    pid_scores_dict[pids[i]].append(1)
                else:
                    pid_scores_dict[pids[i]].append((scores[i] - min_s) / diff)
        if len(queries) > 1:
            if fuse_type == 'COMBSUM':
                pid_scores_dict = COMBSUM(pid_scores_dict)
            elif fuse_type == 'COMBMNZ':
                pid_scores_dict = COMBMNZ(pid_scores_dict)
            elif fuse_type == 'BORDA':
                pid_scores_dict = BORDA(pid_scores_dict)
        order_index = 1
        write_lines = []
        for pid_ordered in pid_scores_dict:
            write_lines.append(qid + ' 0 ' + pid_ordered + ' ' + str(order_index) + ' ' + str(
                pid_scores_dict[pid_ordered][0]) + ' BM25\n')
            order_index = order_index + 1
        output.writelines(write_lines)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--DATA_DIR", type=str, default="data/sysrev-seed-collection")
    parser.add_argument("--collection_file", type=str, default="data/sysrev-seed-collection/all.jsonl")
    parser.add_argument("--METHOD", type=str, required=True)
    parser.add_argument("--prompt_type", type=str, default='openai', help="openai or alpaca, real_seed or biogpt")
    parser.add_argument("--type", type=str, default='title_seed', help="title_seed or seed_fuse")
    parser.add_argument("--fuse_type", type=str, default='COMBSUM', help="COMBSUM or COMNMNZ or BORDA")
    parser.add_argument("--OUT_DIR", type=str, default="output")

    args = parser.parse_args()

    DATA_DIR = args.DATA_DIR
    method = args.METHOD
    type = args.type
    if args.type=="seed_fuse":
        out_folder = os.path.join(args.OUT_DIR, "/".join(args.DATA_DIR.split("/")), type, f'{method}_{args.prompt_type}', args.fuse_type)
    else:
        out_folder = os.path.join(args.OUT_DIR, "/".join(args.DATA_DIR.split("/")), type, f'{method}_{args.prompt_type}')

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    run_file = os.path.join(DATA_DIR, "preprocessed_input.jsonl")
    prompt_file = os.path.join(DATA_DIR, f"generated_seed_{args.prompt_type}.jsonl")
    desired_collection_file = "/".join(args.collection_file.split("/")[:2]) + "/tokenised_collection.jsonl"

    if os.path.exists(desired_collection_file):
        collection_dict = get_collection(desired_collection_file)
    else:
        collection_dict = tokenise_collection(args.collection_file, desired_collection_file)

    seed_dict = {}
    if args.prompt_type != 'real_seed':
        with open(prompt_file) as f:
            for line in f:
                data_dict = json.loads(line)
                qid = data_dict['id']
                seeds = data_dict['generated_seeds']
                for title in seeds:
                    for i in string_set:
                        title = title.replace(i, '')
                    title_tokenized = [tok for tok in title.split() if tok.lower() not in cachedStopwords]
                    if qid not in seed_dict:
                        seed_dict[qid] = []
                    seed_dict[qid].append(title_tokenized)

    query_dict = {}
    passage_dict = {}
    map_q_p = {}
    doc_set = set()
    for pid in collection_dict:
        passage_dict[pid] = collection_dict[pid][0] + collection_dict[pid][1]
    with open(run_file) as f:
        for line in f:
            data_dict = json.loads(line)
            qid = data_dict['id']
            title = data_dict['title']
            if args.prompt_type == 'real_seed':
                if qid not in seed_dict:
                    seed_dict[qid] = []
                seed_studies = data_dict['seed_studies']
                for i in seed_studies:
                    seed_dict[qid].append(collection_dict[i.strip()][0])
            generated_eed_studies = seed_dict[qid]
            searched_docs = data_dict['search_docs']
            seed_studies = seed_dict[qid]
            for i in string_set:
                title = title.replace(i, '')
            title_tokenized = [tok for tok in title.split() if tok.lower() not in cachedStopwords]
            if type == 'title_seed':
                query_dict[qid] = [title_tokenized]
                for i in seed_studies:

                    query_dict[qid][0] += i
            elif type == 'seed_fuse':
                query_dict[qid] = [title_tokenized]
                for i in seed_studies:
                    query_dict[qid].append(i)
            else:
                print("Wrong type")
                exit()
            for i in searched_docs:
                doc_set.add(i)
            map_q_p[qid] = searched_docs
    not_in_collection = []
    in_collection = []
    for i in doc_set:
        if i not in passage_dict:
            not_in_collection.append(i)
        else:
            in_collection.append(i)
    print(len(collection_dict))
    print("In collection: " + str(len(in_collection)))
    print("Not in collection: " + str(len(not_in_collection)))
    print(not_in_collection)

    print(run_file)
    print("Start processing " + method)
    print(query_dict)

    if method=="BM25":
        bm25_rerank_results(query_dict, passage_dict, map_q_p, out_folder, fuse_type=args.fuse_type)

    if method=="QLM":
        qlm_rerank_results(query_dict, passage_dict, map_q_p, out_folder, args.fuse_type)