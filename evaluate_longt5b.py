### FILE TO EVALUATE INSTRUCTION-TUNED MODEL FOR ANALYSIS AND REPORTING

import argparse
import numpy as np
from datasets import load_dataset
from transformers.optimization import Adafactor, AdafactorSchedule
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, LongT5Config, LongT5ForConditionalGeneration
from evaluate import load
import nltk
import pandas as pd
from datasets import Dataset, DatasetDict


def main(args):

    # generic preprocessing function for zeroscrolls dataset
    def preprocess_function(examples):
        inputs = [doc for doc in examples['input']]
        outputs = [ans for ans in examples['output']]

        model_inputs = tokenizer(inputs,max_length=4096,padding="max_length",truncation=True)#, return_tensors="pt")

        labels = tokenizer(text_target=outputs, max_length=512, padding="max_length", truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    # preprocessing function for use with CNN/DM dataset
    # difference solely in the keys for each data sample ('articles', 'highlights' here)
    def preprocess_function_cnndm(examples):
        inputs = [doc for doc in examples['article']]
        outputs = [ans for ans in examples['highlights']]

        model_inputs = tokenizer(inputs,max_length=4096,padding="max_length",truncation=True)#, return_tensors="pt")

        labels = tokenizer(text_target=outputs, max_length=512, padding="max_length", truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    # preprocessing function for use with MultiNews dataset
    def preprocess_function_multinews(examples):
        inputs = [doc for doc in examples['document']]
        outputs = [ans for ans in examples['summary']]

        model_inputs = tokenizer(inputs,max_length=4096,padding="max_length",truncation=True)#, return_tensors="pt")

        labels = tokenizer(text_target=outputs, max_length=512, padding="max_length", truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    # preprocessing function for use with HotpotQA dataset
    def preprocess_function_hqa(examples):
        inputs = [doc for doc in examples['question']]
        outputs = [ans for ans in examples['answer']]

        model_inputs = tokenizer(inputs,max_length=4096,padding="max_length",truncation=True)#, return_tensors="pt")

        labels = tokenizer(text_target=outputs, max_length=512, padding="max_length", truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    # function to compute EM and F1 score during evaluation
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        results = {}
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
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
        
        # ensure inputs are same length
        if len(predictions) < len(labels):
            predictions += [0] * int(len(labels)-len(predictions))
        if len(predictions) > len(labels):
            labels += [0]*(len(predictions)-len(labels))
        f1_result = f1_metric.compute(predictions=predictions, # requires numerical input!!
                                      references=labels,
                                      average="micro",
                                      )
        
        # record scores
        results["exact_match"] = round(em_result["exact_match"], 3)
        results["f1"] = round(f1_result["f1"], 3)
        
        # Add mean, min, max generated length
        results["mean_gen_len"] = np.mean(prediction_lens)
        results["min_gen_len"] = min(prediction_lens)
        results["max_gen_len"] = max(prediction_lens)
        
        return results

    # function to compute rouge1 score during evaluation
    def compute_metrics_r1(eval_pred):
        rouge_metric = load("rouge")
        predictions, labels = eval_pred
        predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Rouge expects a newline after each sentence
        decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
        decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
        
        # record scores
        result = rouge_metric.compute(predictions=decoded_preds,
                                    references=decoded_labels, 
                                    use_stemmer=True)
        results = {k: round(v, 4) for k, v in result.items()}
        
        # Add mean, min, max generated length
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
        results["mean_gen_len"] = np.mean(prediction_lens)
        results["min_gen_len"] = min(prediction_lens)
        results["max_gen_len"] = max(prediction_lens)
        
        return results
    
    # Load model checkpoint as specified
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    config = LongT5Config.from_pretrained(args.model_dir)
    model = LongT5ForConditionalGeneration.from_pretrained(args.model_dir, config=config)

    ### Prepare zeroscrolls dataset
    # Choices: qasper narrative_qa quality summ_screen_fd gov_report qmsum
    dataset = load_dataset("tau/zeroscrolls", "quality", split="validation")
    tokenized_dataset = dataset.map(preprocess_function, batch_size=128, batched=True)
    # compute_metrics = compute_metrics_r1 # use r1 for SSFD, GR, QMSM

    ### Prepare CNN/DM dataset
    # dataset = load_dataset("cnn_dailymail", "1.0.0",split="test") 
    # tokenized_dataset = dataset.map(preprocess_function_cnndm, batch_size=128, batched=True)
    # compute_metrics = compute_metrics_r1
    
    ### Prepare MultiNews dataset
    # dataset = load_dataset("multi_news", split="test")
    # tokenized_dataset = dataset.map(preprocess_function_multinews, batch_size=128, batched=True)
    # compute_metrics = compute_metrics_r1

    ### Prepare HotpotQA dataset
    # dataset = load_dataset("hotpot_qa", "distractor", split="validation")
    # tokenized_dataset = dataset.map(preprocess_function_hqa, batch_size=128, batched=True)
    # compute_metrics = compute_metrics # use f1

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    ### Training parameters (unused)
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

    # set up trainer to call .evaluate() later
    training_args = Seq2SeqTrainingArguments(
        output_dir="./",
        max_steps=120,  
        logging_steps=5,
        per_device_eval_batch_size=2, 
        predict_with_generate=True,
        generation_config="google/long-t5-tglobal-base",
        generation_max_length=512,
        generation_num_beams=1,
        push_to_hub=False,
        report_to="none",
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        optimizers=(optimizer, None),
    )

    # obtain predictions
    test_results = trainer.evaluate(
        tokenized_dataset, 
        max_length=512,
        num_beams=1, # use greedy decoding instead of beam search
    )
    print("RESULTS:", test_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # path to model checkpoint to evaluate upon
    parser.add_argument("--model_dir", 
                        type=str,
                        ) 

    args = parser.parse_args()

    main(args)

