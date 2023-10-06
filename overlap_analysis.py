# this program is to analyze the overlap of generated seeds to their original queries.

import argparse
import json
import os
from tqdm import tqdm
import nltk
from nltk.corpus import stopwords
import string
from glob import glob
cachedStopwords = set(tok.lower() for tok in stopwords.words("english"))

def get_overlap(input_file):
    overlap_dict = {}
    with open(input_file, "r") as f:
        for line in f:
            data = json.loads(line)
            id = data["id"]
            title = data["title"]
            generated_seeds = data["generated_seeds"]
            # get all terms in title and generated seeds, remove stop words
            title_terms = set([t for t in title.split() if t not in cachedStopwords])
            seed_terms = set()
            for seed in generated_seeds:
                seed_terms.update([t for t in seed.split() if t not in cachedStopwords])
            overall_terms = title_terms.union(seed_terms)
            ratio = len(title_terms)/len(overall_terms)
            overlap_dict[id] = ratio
    return overlap_dict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--DATA_DIR", type=str, default="data/clef-tar/CLEF-2017/testing/")
    args = parser.parse_args()

    input_folder = args.DATA_DIR
    input_files = glob(os.path.join(input_folder, "generated_seed_*.jsonl"))
    for input_file in input_files:
        overlap_dict = get_overlap(input_file)
        print(f'For Input File {input_file.split("_")[-1].split(".")[0]}: ', sum(overlap_dict.values())/len(overlap_dict))

if __name__ == "__main__":
    main()
