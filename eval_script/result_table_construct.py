import os
import glob
import json
import scipy.stats as stats

#below are for collections
collections = ["clef-tar/CLEF-2017/testing", "clef-tar/CLEF-2018/testing", "clef-tar/CLEF-2019-dta/testing", "clef-tar/CLEF-2019-intervention/testing", "sysrev-seed-collection"]

#below are for combination types

prompt_types = [

                "search_name_title",
                "title",

                #"boolean_BM25",
                #"boolean_QLM",
                #"boolean"
                #"boolean_openai",
                #"boolean_openai_title_title",
                # "interpolate_boolean_boolean_openai",

                #"boolean_openai_multi",
                #"boolean_alpaca_tuned_query",
                #"boolean_alpaca_tuned_title",

                # "interpolate_boolean_boolean_alpaca_tuned_query",
                #"boolean_alpaca_tuned_query_multi"

                ]

#below are for generation methods


#below are for retrieval methods
#traditional_methods = ["BM25", "QLM"]
neural_methods= ["model_bio_bert"]

#below are for metrics
metrics = ["ap",
           "last_rel", "recall1.0", "recall5.0", "recall10.0", "recall20.0",
           "wss_95", "wss_100"]
result_dict = {}

# CREATE first table that compares systev_seed_collection when only title is used or real seed is used for neural methods.
collection = "sysrev-seed-collection/testing"


title_dict = {}
qid_set = set()
combination_type = "title"
for method in neural_methods:
    input_folder = os.path.join("combined_output/eval", collection, combination_type, method)
    if method not in title_dict:
        title_dict[method] = {}
    for metric in metrics:
        input_file = os.path.join(input_folder, metric + ".res")
        if metric not in title_dict[method]:
            title_dict[method][metric] = {}
        with open(input_file, 'r') as f:
            for line in f:
                qid, score = line.strip().split()
                title_dict[method][metric][qid] = float(score)
                if qid!="all":
                    qid_set.add(qid)


prompt_dict = {}
for prompt_type in prompt_types:
    for method in neural_methods:
        input_folder = os.path.join("combined_output/eval", collection, prompt_type, method)
        if method not in prompt_dict:
            prompt_dict[method] = {}
        if prompt_type not in prompt_dict[method]:
            prompt_dict[method][prompt_type] = {}
        qid_ordering_dict = {}

        for metric in metrics:
            input_file = os.path.join(input_folder, metric + ".res")
            if metric not in prompt_dict[method][prompt_type]:
                prompt_dict[method][prompt_type][metric] = {}
            with open(input_file, 'r') as f:
                for line in f:
                    qid, score = line.strip().split()
                    original_qid = qid.split("_")[0]
                    if metric == "ap":
                        if original_qid not in prompt_dict[method][prompt_type][metric]:
                            prompt_dict[method][prompt_type][metric][original_qid] = float(score)
                            qid_ordering_dict[original_qid] = qid
                        else:
                            if float(score) > prompt_dict[method][prompt_type][metric][original_qid]:
                                prompt_dict[method][prompt_type][metric][original_qid] = float(score)
                                qid_ordering_dict[original_qid] = qid
                    else:
                        if original_qid in qid_ordering_dict:
                            if qid_ordering_dict[original_qid] == qid:
                                prompt_dict[method][prompt_type][metric][original_qid] = float(score)


qid_list = list(qid_set)
for method in neural_methods:
    title_line = collection.replace("/testing", "") + " & "  + " boolean"
    for metric in metrics:
        current_title_scores = [title_dict[method][metric][qid] for qid in qid_list]
        average_title_score = sum(current_title_scores) / len(current_title_scores)
        title_line += " & " + f'{average_title_score:.3f}'
    print(title_line + " \\\\")


    for prompt_type in prompt_types:
        line = collection + " & " + prompt_type.replace("_query", "")
        if prompt_type.endswith("_multi"):
            prompt_type_w = prompt_type.replace("_query", "") + "_oracle"
            line = collection  + " & " + prompt_type_w


        for metric in metrics:
            current_title_scores = [title_dict[method][metric][qid] for qid in qid_list]
            current_prompt_scores = [prompt_dict[method][prompt_type][metric][qid] for qid in qid_list]
            average_title_score = sum(current_title_scores) / len(current_title_scores)
            average_prompt_score = sum(current_prompt_scores) / len(current_prompt_scores)
            # perform t-test
            t, p = stats.ttest_rel(current_title_scores, current_prompt_scores)
            if p < 0.05:
                line += " & " + f'{average_prompt_score:.3f}' + "*"
            else:
                line += " & " + f'{average_prompt_score:.3f}'
        print(line + " \\\\")
    print("\\midrule")


# below is for the second table that compares the performance of different methods on different collections
prompt_types = [

    # "search_name_title",

    # "boolean_BM25",
    # "boolean_QLM",
     "boolean",
     "title",
"harry_result",
    #"boolean_openai",
    #"boolean_openai_title_title",
    # "interpolate_boolean_boolean_openai",

    # "boolean_openai_multi",
    "boolean_alpaca_tuned_query",
    "boolean_alpaca_tuned_title",

    # "interpolate_boolean_boolean_alpaca_tuned_query",
    # "boolean_alpaca_tuned_query_multi"

]

collections = ["clef-tar/CLEF-2017/testing", "clef-tar/CLEF-2018/testing", "clef-tar/CLEF-2019-dta/testing", "clef-tar/CLEF-2019-intervention/testing"]




print("next table")
max_dict = {}
for collection in collections:
    if collection not in max_dict:
        max_dict[collection] = {}
    title_dict = {}
    qid_set = set()
    harry_result = {}
    for metric in metrics:
        harry_result[metric] = {}
    if (collection == "clef-tar/CLEF-2017/testing"):
        harry_file = os.path.join("../data", collection, "clf+weighting+pubmed+qe_2017.run.eval.json")
    if (collection == "clef-tar/CLEF-2018/testing"):
        harry_file = os.path.join("../data", collection, "clf+weighting+pubmed+qe_2018.run.eval.json")
    if (collection=="clef-tar/CLEF-2017/testing") or (collection=="clef-tar/CLEF-2018/testing"):
        #harry_file = os.path.join("data", collection, "clf+weighting+pubmed+qe_2018.run.eval.json")
        with open(harry_file, 'r') as f:
            harry_temp_result = json.load(f)
            for qid in harry_temp_result:
                harry_result["ap"][qid] = harry_temp_result[qid]["ap"]
                harry_result["last_rel"][qid] = harry_temp_result[qid]["last_rel"]
                if "recall@1.0%" in  harry_temp_result[qid]:
                    harry_result["recall1.0"][qid] = harry_temp_result[qid]["recall@1.0%"]
                    harry_result["recall5.0"][qid] = harry_temp_result[qid]["recall@5.0%"]
                    harry_result["recall10.0"][qid] = harry_temp_result[qid]["recall@10.0%"]
                    harry_result["recall20.0"][qid] = harry_temp_result[qid]["recall@20.0%"]
                harry_result["wss_95"][qid] = harry_temp_result[qid]["wss_95"]
                harry_result["wss_100"][qid] = harry_temp_result[qid]["wss_100"]

    for method in neural_methods:
        target_type = "boolean"
        input_folder = os.path.join("combined_output/eval", collection ,target_type, method)
        if method not in title_dict:
            title_dict[method] = {}
        for metric in metrics:
            input_file = os.path.join(input_folder, metric + ".res")
            if metric not in title_dict[method]:
                title_dict[method][metric] = {}
            with open(input_file, 'r') as f:
                for line in f:
                    qid, score = line.strip().split()
                    title_dict[method][metric][qid] = float(score)
                    if qid != "all":
                        qid_set.add(qid)
                    else:
                        if metric not in max_dict[collection]:
                            max_dict[collection][metric] = float(score)
                        elif float(score) > max_dict[collection][metric]:
                            max_dict[collection][metric] = float(score)


    prompt_dict = {}
    for prompt_type in prompt_types:
        for method in neural_methods:
            input_folder = os.path.join("combined_output/eval", collection, prompt_type, method)
            if method not in prompt_dict:
                prompt_dict[method] = {}
            if prompt_type not in prompt_dict[method]:
                prompt_dict[method][prompt_type] = {}
            qid_ordering_dict = {}
            for metric in metrics:
                input_file = os.path.join(input_folder, metric + ".res")
                if metric not in prompt_dict[method][prompt_type]:
                    prompt_dict[method][prompt_type][metric] = {}
                with open(input_file, 'r') as f:
                    for line in f:
                        qid, score = line.strip().split()
                        original_qid = qid.split("_")[0]
                        if metric == "ap":
                            if original_qid not in prompt_dict[method][prompt_type][metric]:
                                prompt_dict[method][prompt_type][metric][original_qid] = float(score)
                                qid_ordering_dict[original_qid] = qid
                            else:
                                if float(score) > prompt_dict[method][prompt_type][metric][original_qid]:
                                    prompt_dict[method][prompt_type][metric][original_qid] = float(score)
                                    qid_ordering_dict[original_qid] = qid
                        else:
                            if original_qid in qid_ordering_dict:
                                if qid_ordering_dict[original_qid] == qid:
                                    prompt_dict[method][prompt_type][metric][original_qid] = float(score)


    qid_list = list(qid_set)

    ## print out harry result


    for method in neural_methods:
        title_line = collection.replace("/", "-").replace("-testing", "").replace("clef-tar-", "")  + " & boolean"
        for metric in metrics:
            current_title_scores = [title_dict[method][metric][qid] for qid in qid_list]
            average_title_score = sum(current_title_scores) / len(current_title_scores)
            title_line += " & " + f'{average_title_score:.3f}'
        print(title_line + " \\\\")

        if (collection == "clef-tar/CLEF-2017/testing") or (collection == "clef-tar/CLEF-2018/testing"):

            harry_line = collection.replace("/", "-").replace("-testing", "").replace("clef-tar-", "") + " & harry"

            for metric in metrics:
                if harry_result[metric] != {}:
                    # perform t-test
                    current_prompt_scores = [harry_result[metric][qid] for qid in qid_list]
                    average_prompt_score = sum(current_prompt_scores) / len(current_prompt_scores)
                    # perform t-test
                    t, p = stats.ttest_rel(current_title_scores, current_prompt_scores)
                    if p < 0.05:
                        harry_line += " & " + f'{average_prompt_score:.3f}' + "*"
                    else:
                        harry_line += " & " + f'{average_prompt_score:.3f}'
                else:
                    harry_line += " & -"
            print(harry_line + " \\\\")


        for prompt_type in prompt_types:
            line = collection.replace("/", "-").replace("-testing", "").replace("clef-tar-", "") + " & " + prompt_type.replace("_query", "")
            if prompt_type.endswith("_multi"):
                prompt_type_w = prompt_type.replace("_query", "") + "_oracle"
                line = collection.replace("/", "-").replace("-testing", "").replace("clef-tar-",
                                                                                    "") + " & " + prompt_type_w

            for metric in metrics:
                current_title_scores = [title_dict[method][metric][qid] for qid in qid_list]
                current_prompt_scores = [prompt_dict[method][prompt_type][metric][qid] for qid in qid_list]
                average_title_score = sum(current_title_scores) / len(current_title_scores)
                average_prompt_score = sum(current_prompt_scores) / len(current_prompt_scores)
                # perform t-test
                t, p = stats.ttest_rel(current_title_scores, current_prompt_scores)
                if p < 0.05:
                    line += " & " + f'{average_prompt_score:.3f}' + "*"
                else:
                    line += " & " + f'{average_prompt_score:.3f}'
            print(line + " \\\\")
        print("\\midrule")
    print("\\midrule")