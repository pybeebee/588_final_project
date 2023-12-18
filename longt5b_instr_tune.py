### INSTRUCTION TUNE LONGT5-BASE USING THE CURATED INSTRUCTION DATA

import wandb
import argparse
import numpy as np
import os
import torch
from datetime import datetime
import transformers
from datasets import load_dataset
from transformers.optimization import Adafactor, AdafactorSchedule
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, LongT5Config, LongT5ForConditionalGeneration
from evaluate import load
import nltk
import pandas as pd
from datasets import Dataset, DatasetDict


def main(args):

    ### Setup wandb logging
    wandb.login()
    run = wandb.init(
        project=args.wandb_project_name,
        )

    ### Set up output dirs for model and checkpoints
    data_type = args.json_dir[args.json_dir.index("data_splits")+12:]
    current_date_time = str(datetime.now().strftime("%Y-%m-%d_hr%H-min%M"))
    output_dir = os.path.join(args.output_dir, data_type, current_date_time)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # checkpoint dir 
    checkpoint_dir = os.path.join(output_dir,"checkpoints")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # final model dir:
    model_save_dir = os.path.join(output_dir,"final_model")
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)


    def preprocess_function(examples):
        inputs = [doc for doc in examples['instruction']]
        outputs = [ans for ans in examples['answer']]

        # tokenize document instances
        model_inputs = tokenizer(inputs, 
                                max_length=4096, 
                                padding="max_length",  # CHECK: NEED THIS?
                                truncation=True)#, return_tensors="pt")

        labels = tokenizer(
            text_target=outputs, 
            max_length=512, 
            padding="max_length",
            truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    # use EM and F1 score since instruction examples follow QA format for the most part
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        
        em_metric = load("exact_match") #https://huggingface.co/spaces/evaluate-metric/exact_match
        f1_metric = load("f1") # https://huggingface.co/spaces/evaluate-metric/f1

        predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        em_result = em_metric.compute(predictions=decoded_preds,
                                   references=decoded_labels, 
                                   ignore_case=True,
                                   ignore_punctuation=True,
                                   ignore_numbers=False,
        )
        predictions = list(predictions.flatten())
        labels = list(labels.flatten())
        if len(predictions) < len(labels):
            predictions += [0] * int(len(labels)-len(predictions))
        if len(predictions) > len(labels):
            labels += [0]*(len(predictions)-len(labels))
        f1_result = f1_metric.compute(predictions=predictions, # requires numerical input!!
                                      references=labels,
                                      average="micro",
                                      )
        results = {"exact_match": round(em_result["exact_match"], 4),
                   "f1": round(f1_result["f1"], 4)
                   }
        
        # Add mean generated length
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
        results["mean_gen_len"] = np.mean(prediction_lens)
        results["min_gen_len"] = min(prediction_lens)
        results["max_gen_len"] = max(prediction_lens)
        
        wandb.log(results)

        return results

    ### Load model
    tokenizer = AutoTokenizer.from_pretrained("google/long-t5-tglobal-base")
    config = LongT5Config.from_pretrained("google/long-t5-tglobal-base")
    model = LongT5ForConditionalGeneration.from_pretrained("google/long-t5-tglobal-base", config=config)

    ### Prepare dataset
    train_path = args.json_dir+'/train.json'
    valid_path = args.json_dir+'/valid.json'
    test_path = args.json_dir+'/test.json'

    train_df = pd.read_json(train_path, lines=True)
    train_df['answer'] = train_df['answer'].astype(str) # convert to string to avoid json error (have some ints sprinkled in...)
    valid_df = pd.read_json(valid_path, lines=True)
    valid_df['answer'] = valid_df['answer'].astype(str) # convert to string to avoid json error (have some ints sprinkled in...)
    test_df = pd.read_json(test_path, lines=True) 
    test_df['answer'] = test_df['answer'].astype(str) # convert to string to avoid json error (have some ints sprinkled in...)
    train_ds = Dataset.from_pandas(train_df)
    valid_ds = Dataset.from_pandas(valid_df)
    test_ds = Dataset.from_pandas(test_df)
    dataset = DatasetDict()

    dataset["train"] = train_ds
    dataset["validation"] = valid_ds
    dataset["test"] = test_ds

    # tell # samples directly if needed
    if args.train_num_samples:
        dataset["train"] = dataset["train"].shuffle(seed=42).select(range(args.train_num_samples))
    if args.valid_num_samples:
        dataset["validation"] = dataset["validation"].shuffle(seed=42).select(range(args.valid_num_samples))
    if args.test_num_samples:
        dataset["test"] = dataset["test"].shuffle(seed=42).select(range(args.test_num_samples))

    tokenized_dataset = dataset.map(preprocess_function, batch_size=128, batched=True)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    ### Training parameters -- same as in LongT5 paper
    optimizer = Adafactor(model.parameters(),
            lr=1e-3,
            eps=(1e-30, 1e-3),
            clip_threshold=1.0,
            decay_rate=-0.8,
            beta1=None,
            weight_decay=0.0,
            relative_step=False,
            scale_parameter=False,
            warmup_init=False)
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=checkpoint_dir,
        evaluation_strategy="steps",
        save_strategy="steps",
        learning_rate=args.lr,
        lr_scheduler_type=args.lr_scheduler_type,
        per_device_train_batch_size=args.per_batch_size,
        per_device_eval_batch_size=args.per_batch_size, 
        gradient_accumulation_steps=args.grad_acc_steps,
        optim="adafactor",
        adafactor=True,
        max_steps=120,  # don't use if use num_train_epochs
        # num_train_epochs=args.num_train_epochs,
        save_steps=args.save_steps, 
        eval_steps=args.eval_steps, 
        warmup_steps=args.warmup_steps,
        logging_steps=5,
        # save_total_limit=5,
        # load_best_model_at_end=True,
        predict_with_generate=True,
        generation_config="google/long-t5-tglobal-base",
        generation_max_length=512,
        generation_num_beams=1,
        fp16=args.fp16,
        push_to_hub=False,
        report_to="wandb",
        resume_from_checkpoint=args.resume_from_checkpoint,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        optimizers=(optimizer, None)
    )

    trainer.train()

    trainer.save_model(model_save_dir)
    print("model saved to:",model_save_dir)

    test_results = trainer.evaluate(
        tokenized_dataset["test"], 
        metric_key_prefix="test",
        max_length=512,
        num_beams=1, # use greedy decoding instead of beam search
    )
    wandb.log(test_results)

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # name of path to appropriate data_json folder from ./generated_instructions folder
    parser.add_argument("--json_dir", type=str) 
    parser.add_argument("--wandb_project_name", 
                        type=str, 
                        default="longt5b_instr_tune",
                        ) 
    parser.add_argument("--output_dir", 
                        type=str,
                        ) 

    # training args
    parser.add_argument("--lr", 
                        type=float, 
                        default=0.001,
                        ) 
    parser.add_argument("--lr_scheduler_type", 
                        type=str, 
                        default="constant",
                        ) 
    parser.add_argument("--warmup_steps", 
                        type=int, 
                        default=0,
                        ) 
    parser.add_argument("--per_batch_size", type=int) 
    parser.add_argument("--grad_acc_steps", type=int) 
    parser.add_argument("--save_steps", type=int) 
    parser.add_argument("--eval_steps", type=int)
    parser.add_argument("--num_train_epochs", type=int)
    parser.add_argument("--fp16", 
                        action="store_true",
                        default=False, 
                        ) 
    parser.add_argument("--resume_from_checkpoint", 
                        action="store_true",
                        default=False, 
                        ) 

    # dataset preparation args
    parser.add_argument("--train_num_samples", 
                        type=int, 
                        default=None,
                        ) 
    parser.add_argument("--valid_num_samples", 
                        type=int, 
                        default=None,
                        ) 
    parser.add_argument("--test_num_samples", 
                        type=int, 
                        default=None,
                        ) 
    
    args = parser.parse_args()

    main(args)


### TODO BEFORE RUN: CREATE DESIRED CHECKPOINT DIR