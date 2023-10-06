import openai
import torch
import argparse
import json
from tqdm import tqdm
from transformers import LlamaForCausalLM, LlamaTokenizer, GenerationConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
from peft import PeftModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

openai.api_key = ""

Apaca_PROMPT = 'Below is an instruction that describes a task, paired with an input that provides further context. ' \
               'Write a response that appropriately completes the request.\n\n### ' \
               'Instruction:\n Construct a natural language query using the systematic review boolean query provided.' \
               '\n\n ### Input:\n{boolean}\n\n ' \
               '### Response:'

Chatgpt_PROMPT = 'Construct a high-quality natural language query for the boolean query of a systematic review: {boolean}. ' \
                 'The effectiveness of the query will be determined by its capability to retrieve relevant documents when searching on a semantic-based search engine.' \
                 'The generated query can also be the same as the original topic, as long as it can achieve high effectiveness.'

def get_openai_response(prompt, boolean, n):
    print(boolean)
    generated_query = []
    while len(generated_query) < n:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt.format(boolean=boolean)},
            ],
            n=n-len(generated_query),
            temperature=1,
            max_tokens=256,
            top_p=0.9
        )
        responses = response["choices"]
        for item in responses:
            generated_query.append(item["message"]["content"])

    return list(set(generated_query))

def get_alpaca_response(boolean, model, tokenizer, n):
    input_sentence = Apaca_PROMPT.format(boolean=boolean)
    print(input_sentence)
    input_ids = tokenizer.encode(input_sentence, return_tensors="pt").to(device)
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=device)
    generated_query = []
    while len(generated_query) < n:
        output_ids = model.generate(
            inputs=input_ids,
            attention_mask=attention_mask,
            temperature=1,
            top_p=0.9,
            do_sample=True,
            num_beams=1,
            num_return_sequences=n,
            max_new_tokens=256,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        for i in range(len(output_ids)):
            output_sentence = tokenizer.decode(output_ids[i], skip_special_tokens=True)
            output_sentence = output_sentence.split("### Response:")[1].strip()
            generated_query.append(output_sentence)
            print(output_sentence)

        # output_sentence = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        # output_sentence = output_sentence.split("### Response:")[1].strip()
        # generated_query.add(output_sentence)
        # print(output_sentence)

    return list(set(generated_query))



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", type=str, required=True)
    parser.add_argument("--input_model", type=str, default="../reproduced_alpaca_models/")
    parser.add_argument("--input_tuned", type=str, default="")
    parser.add_argument("--type", type=str, help="openai or alpaca or alpaca_tuned",
                        required=True)

    parser.add_argument("--n", type=int, default=10)
    arg = parser.parse_args()

    print(device)

    input_file = os.path.join(arg.input_folder, "preprocessed_input_abstract.jsonl")
    input_dict = {}
    out_file = os.path.join(arg.input_folder, f'generated_from_booleans_{arg.type}_query.jsonl')

    n = arg.n

    with open(input_file, "r") as f:
        for line in f:
            data = json.loads(line)
            id = data["id"]
            boolean = data["query"]
            if "seed_studies" in data:
                seed_studies = data["seed_studies"]

                input_dict[id] = [boolean, seed_studies]
            else:
                input_dict[id] = [boolean, []]

    if "alpaca" in arg.type:
        if "tuned" in arg.type:

            tokenizer = LlamaTokenizer.from_pretrained(arg.input_tuned)
            model = LlamaForCausalLM.from_pretrained(arg.input_tuned).to(device)

        else:
            tokenizer = LlamaTokenizer.from_pretrained(arg.input_model)
            model = LlamaForCausalLM.from_pretrained(arg.input_model).to(device)

    if os.path.exists(out_file):
        with open(out_file, "r") as f:
            for line in f:
                data = json.loads(line)
                id = data["id"]
                if id in input_dict:
                    del input_dict[id]
        fw = open(out_file, "a+")
    else:
        fw = open(out_file, "w")

    for id, boolean_seeds in tqdm(input_dict.items()):
        #try:
        if arg.type == "openai":
            boolean = boolean_seeds[0]
            generated_query = get_openai_response(Chatgpt_PROMPT, boolean, n)
            data = {"id": id, "boolean": boolean, "generated_query": generated_query}
        elif ("alpaca" in arg.type):
            boolean = boolean_seeds[0]
            generated_query = get_alpaca_response(boolean, model, tokenizer, n)
            data = {"id": id, "boolean": boolean, "generated_query": generated_query}
        else:
            raise ValueError("Wrong type")

        fw.write(json.dumps(data) + "\n")



if __name__ == "__main__":
    main()