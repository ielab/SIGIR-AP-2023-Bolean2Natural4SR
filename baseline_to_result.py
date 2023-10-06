import os
import tqdm

baseline_file_location = ["/tar/2017-TAR/participant-runs/sheffield_4.txt", "tar/2018-TAR/participant-runs/sheffield_general.txt", "tar/2019-TAR/Runs/UniversityofSheffield/DTA/DTA_sheffield-Odds_Ratio", "tar/2019-TAR/Runs/UniversityofSheffield/Intervention/Intervention_sheffield-Log_likelihood"]
destination = ["combined_output/data/clef-tar/CLEF-2017/testing/baseline/model_bio_bert/", "combined_output/data/clef-tar/CLEF-2018/testing/baseline/model_bio_bert/", "combined_output/data/clef-tar/CLEF-2019-dta/testing/baseline/model_bio_bert/", "combined_output/data/clef-tar/CLEF-2019-intervention/testing/baseline/model_bio_bert/"]


for i in range(len(baseline_file_location)):
    baseline_file = baseline_file_location[i]
    destination_file = destination[i]
    if not os.path.exists(destination_file):
        os.makedirs(destination_file)
    qid_dict = {}
    with open(baseline_file) as f:
        for line in f:
            qid, _, pid, rank, score, _ = line.split()
            if qid not in qid_dict:
                qid_dict[qid] = []
            qid_dict[qid].append(line)
    for qid in qid_dict:
        qid_dict[qid].sort(key=lambda x: x[3])
        with open(os.path.join(destination_file, qid + ".trec"), "w") as f:
            for line in qid_dict[qid]:
                f.write(line)


