# Copyright 2021 Reranker Author. All rights reserved.
# Code structure inspired by HuggingFace run_glue.py in the transformers library.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from reranker import Reranker, RerankerDC
from reranker import RerankerTrainer, RerankerDCTrainer
from reranker.data import GroupedTrainDataset, PredictionDataset, GroupCollator
from reranker.arguments import ModelArguments, DataArguments, \
    RerankerTrainingArguments as TrainingArguments

from transformers import AutoConfig, AutoTokenizer
from transformers import (
    HfArgumentParser,
    set_seed,
)
from tqdm import tqdm

logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args: ModelArguments
    data_args: DataArguments
    training_args: TrainingArguments

    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("Model parameters %s", model_args)
    logger.info("Data parameters %s", data_args)

    # Set seed
    set_seed(training_args.seed)

    num_labels = 1

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=False,
    )

    _model_class = RerankerDC if training_args.distance_cache else Reranker

    model = _model_class.from_pretrained(
        model_args, data_args, training_args,
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )


    # Get datasets
    if training_args.do_train:
        train_dataset = GroupedTrainDataset(
            data_args, data_args.train_path, tokenizer=tokenizer, train_args=training_args
        )
    else:
        train_dataset = None


    # Initialize our Trainer
    _trainer_class = RerankerDCTrainer if training_args.distance_cache else RerankerTrainer
    trainer = _trainer_class(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=GroupCollator(tokenizer),
    )

    # Training
    if training_args.do_train:
        trainer.train(
            model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
        )
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_process_zero():
            tokenizer.save_pretrained(training_args.output_dir)

    if training_args.do_eval:
        trainer.evaluate()

    if training_args.do_predict:
        logging.info("*** Prediction ***")
        print(training_args.output_dir)


        test_dataset = PredictionDataset(
            data_args.pred_path, tokenizer=tokenizer,
            max_len=data_args.max_len
        )
        assert data_args.pred_id_file is not None

        pred_qids = []
        pred_pids = []
        with open(data_args.pred_id_file) as f:
            for l in f:
                q, p = l.split()
                pred_qids.append(q)
                pred_pids.append(p)


        pred_scores = trainer.predict(test_dataset=test_dataset).predictions
        result_dict = {}
        if trainer.is_world_process_zero():
            assert len(pred_qids) == len(pred_scores)

            for qid, pid, score in zip(pred_qids, pred_pids, pred_scores):
                if qid not in result_dict:
                    result_dict[qid] = {pid: float(score)}
                else:
                    result_dict[qid] = {**result_dict[qid], pid: float(score)}

        ## sort scores and write to file
        for qid in tqdm(result_dict):
            result_dict[qid] = sorted(result_dict[qid].items(), key=lambda x: x[1], reverse=True)



        for qid in tqdm(result_dict):
            output = open(os.path.join(training_args.output_dir, qid + '.trec'), 'w')
            output_lines = []
            index = 1
            for result in result_dict[qid]:
                pid = result[0]
                score = result[1]
                output_lines.append(f'{qid} Q0 {pid} {index} {score} rerank\n')
                index += 1
            output.writelines(output_lines)

if __name__ == "__main__":
    main()