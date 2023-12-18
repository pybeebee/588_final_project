### FILE TO CONVERT CURATED DATA TO .JSON FILES FOR INPUT TO HUGGINGFACE DATASET LOADER
import argparse
from transformers import pipeline
from tqdm import tqdm
import pandas as pd
import re
import random
import os
from datetime import datetime
import torch
from transformers import AutoTokenizer, AutoModel
import json

LENGTH_ENHANCEMENTS = ["\nRespond in 2 sentences.", "\nRespond briefly."]
SPLIT_JSON_NAMES = ["train.json", "valid.json", "test.json"]

# get string representation of enhancement status for use in output directory path
def get_enhancement_descrip(args):
    descrip = "enhance_"
    if args.use_enhancement_1:
        descrip += "1"
    if args.use_enhancement_2:
        descrip += "2"
    if args.use_enhancement_3:
        descrip += "3"
    if args.use_enhancement_4:
        descrip += "4"
    if args.use_enhancement_5:
        descrip += "5"
    if args.use_enhancement_6:
        descrip += "6"
    if args.use_enhancement_7:
        descrip += "7"
    if args.use_enhancement_8:
        descrip += "8"
    if args.use_enhancement_9:
        descrip += "9"
    if not args.use_enhancement_1 and not args.use_enhancement_2 and not args.use_enhancement_3 and not args.use_enhancement_4 and not args.use_enhancement_5 and not args.use_enhancement_6 and not args.use_enhancement_7 and not args.use_enhancement_8 and not args.use_enhancement_9:
        descrip += "0"
    return descrip
  
# for any rows with the same string in "answer", keep only the one with the highest "score" entry. if all have the same score then just keep the first one and discard the others from the df.
def keep_highest_score(group):
    max_score_index = group['score'].idxmax()
    return group.loc[max_score_index]

def main(args):
    
    # function to add enhancement to a given instruction
    def enhance(instr):
        addition = ""

        if args.use_enhancement_1:
            if random.random() < 1/3: # 33% of the time
                instr += random.choice(LENGTH_ENHANCEMENTS)
        if args.use_enhancement_2:
            if random.random() < 0.5: # 50% of the time
                addition = "\nRead the prompt again: "+instr[instr.rfind('\n')+1:]
        elif args.use_enhancement_3:
            if random.random() < 0.5:
                addition = "\nYou will need to look for the answer across multiple locations in the input. Do not ignore the middle of the input context. Do not prioritize only the beginning and end."
        elif args.use_enhancement_4: 
            if random.random() < 0.5:
                addition = "\nTo come up with your response, first read once through all the documents in detail. Then, do a second pass to consider each sentence and determine whether it is relevant to the query as you construct each response."
        elif args.use_enhancement_5: 
            if random.random() < 0.5:
                addition = "\nWhen responding, first consider the topic sentences in the documents and then do a second read through the documents to generate your response."
        elif args.use_enhancement_6: 
            if random.random() < 0.5:
                addition = "\nSkim the text before providing a response."
        elif args.use_enhancement_7: 
            if random.random() < 0.5:
                addition = "\nAs you come up with your response, consider carefully whether each piece of information in the source documents is useful toward successfully responding to the query or not."
        elif args.use_enhancement_8: 
            if random.random() < 0.5:
                addition = "\nWhen coming up with the answer, did you consider information across all positions in the input documents? If not, do so and revise your answer before responding."
        elif args.use_enhancement_9: 
            if random.random() < 0.5:
                addition = "\nBefore providing your response, read through the provided documents again. Is there anything you would change about your response? If so, make such change before giving your final answer."
        return instr+addition

    
    # function to deduplicate examples in the dataframe and select a specified number of samples
    def process_df(data_df, num_to_choose):
        # deduplicate ones with same answer (keep only first/highest scoring such instruction)
        deduped_df = data_df.groupby('answer').apply(keep_highest_score)
        deduped_df = deduped_df.reset_index(drop=True) # reset index col to default

        # select num_to_choose of them randomly
        filtered_df = deduped_df.sample(n=min(num_to_choose,deduped_df.shape[0]), random_state=0)

        # apply enhancement function to each instruction
        filtered_df['instruction'] = filtered_df['instruction'].apply(enhance)

        return filtered_df
    
    # same as above but processing specific to E2 and E5
    def process_df_E2_E5(data_df, num_to_choose):
        # deduplicate ones with same instruction (keep only first/highest scoring such instruction)
        deduped_df = data_df.groupby('instruction').apply(keep_highest_score)
        deduped_df = deduped_df.reset_index(drop=True) # reset index col to default

        # select num_to_choose of them randomly
        filtered_df = deduped_df.sample(n=min(num_to_choose, deduped_df.shape[0]), random_state=0)

        # apply enhancement function to each instruction
        filtered_df['instruction'] = filtered_df['instruction'].apply(enhance)

        return filtered_df
    
    # create output directory
    enhancement_descrip = get_enhancement_descrip(args)
    output_dir = os.path.join("./data_splits", args.instr_type_proportions_id,"thresh_"+str(args.instr_thresh_num), enhancement_descrip,str(args.total_instr_num))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # address one data split at a time (train/test/val)
    for split_json in SPLIT_JSON_NAMES:
        print("##############################")
        print("Working on:",split_json)
        dfs_to_concatenate = []

        if args.use_ABDE6 != 0:
            num_to_choose = args.total_instr_num * args.use_ABDE6

            if split_json != "train.json":
                num_to_choose *= 0.10
            num_to_choose = round(num_to_choose)

            # read all of the json files of the form specified at file call
            A_0_df = None
            if args.A_0_json_path!="": # path to train.json, valid.json, test.json for A_0 (take from output of self_curation_revised2.py)
                A_0_json = os.path.join(args.A_0_json_path, "train.json")
                A_0_df_full = pd.read_json(A_0_json, lines=True)
                num_rows = A_0_df_full.shape[0]
                test_rows = round(0.2*num_rows)
                train_rows = round(0.6*num_rows)
                if split_json=="train.json":
                    A_0_df = A_0_df_full.iloc[:train_rows]
                elif split_json=="valid.json":
                    A_0_df = A_0_df_full.iloc[train_rows:train_rows+test_rows]
                elif split_json=="test.json":
                    A_0_df = A_0_df_full.iloc[train_rows+test_rows:]
                    
            A_1_0_df = None
            if args.A_1_0_json_path!="":
                A_1_0_json = os.path.join(args.A_1_0_json_path, "train.json")
                A_1_0_df_full = pd.read_json(A_1_0_json, lines=True)
                num_rows = A_1_0_df_full.shape[0]
                test_rows = round(0.2*num_rows)
                train_rows = round(0.6*num_rows)
                if split_json=="train.json":
                    A_1_0_df = A_1_0_df_full.iloc[:train_rows]
                elif split_json=="valid.json":
                    A_1_0_df = A_1_0_df_full.iloc[train_rows:train_rows+test_rows]
                elif split_json=="test.json":
                    A_1_0_df = A_1_0_df_full.iloc[train_rows+test_rows:]

            B_0_3_df = None
            if args.B_0_3_json_path!="":
                B_0_3_json = os.path.join(args.B_0_3_json_path, "train.json")
                B_0_3_df_full = pd.read_json(B_0_3_json, lines=True)
                num_rows = B_0_3_df_full.shape[0]
                test_rows = round(0.2*num_rows)
                train_rows = round(0.6*num_rows)
                if split_json=="train.json":
                    B_0_3_df = B_0_3_df_full.iloc[:train_rows]
                elif split_json=="valid.json":
                    B_0_3_df = B_0_3_df_full.iloc[train_rows:train_rows+test_rows]
                elif split_json=="test.json":
                    B_0_3_df = B_0_3_df_full.iloc[train_rows+test_rows:]

            B_1_3_df = None
            if args.B_1_3_json_path!="":
                B_1_3_json = os.path.join(args.B_1_3_json_path, "train.json")
                B_1_3_df_full = pd.read_json(B_1_3_json, lines=True)
                num_rows = B_1_3_df_full.shape[0]
                test_rows = round(0.2*num_rows)
                train_rows = round(0.6*num_rows)
                if split_json=="train.json":
                    B_1_3_df = B_1_3_df_full.iloc[:train_rows]
                elif split_json=="valid.json":
                    B_1_3_df = B_1_3_df_full.iloc[train_rows:train_rows+test_rows]
                elif split_json=="test.json":
                    B_1_3_df = B_1_3_df_full.iloc[train_rows+test_rows:]

            data_df = pd.concat([A_0_df, A_1_0_df, B_0_3_df, B_1_3_df], ignore_index=True)

            processed_df = process_df(data_df, num_to_choose)
            dfs_to_concatenate.append(processed_df)
            print("Total instruction num (use_ABDE6): %d"%(processed_df.shape[0]))

        if args.use_E2 != 0:
            num_to_choose = args.total_instr_num * args.use_E2
            
            if split_json != "train.json":
                num_to_choose *= 0.10
            num_to_choose = round(num_to_choose)
            
            E_2_json_path = args.E_2_json_path
            E_2_json = os.path.join(E_2_json_path, split_json)
            E_2_df = pd.read_json(E_2_json, lines=True)
            
            processed_df = process_df_E2_E5(E_2_df, num_to_choose)
            dfs_to_concatenate.append(processed_df)
            print("Total instruction num (use_E2): %d"%(processed_df.shape[0]))

        if args.use_E5 != 0:
            num_to_choose = args.total_instr_num * args.use_E5

            if split_json != "train.json":
                num_to_choose *= 0.10
            num_to_choose = round(num_to_choose)

            E_5_json_path = args.E_5_json_path
            E_5_json = os.path.join(E_5_json_path, split_json)
            E_5_df = pd.read_json(E_5_json, lines=True)

            processed_df = process_df_E2_E5(E_5_df, num_to_choose)
            dfs_to_concatenate.append(processed_df)
            print("Total instruction num (use_E5): %d"%(processed_df.shape[0]))

        split_data_df = pd.concat(dfs_to_concatenate, ignore_index=True)
        split_data_file_path = os.path.join(output_dir, split_json)
        split_data_df.to_json(split_data_file_path, orient='records', lines=True)
        print("################################")
        print("Finished saving instructions for %s to %s"%(split_json, split_data_file_path))
        print("Total number of instructions (%s):"%(split_json), split_data_df.shape[0])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--total_instr_num",
        type=int,
        default=25000,
    )

    parser.add_argument(
        "--instr_type_proportions_id", # id describing the types of instructions, their relative proportions, and selection thresholds; keep tabs of this in personal notes file for reference.
        type=str,
        default="",
    )
    parser.add_argument(
        "--instr_thresh_num", # id describing the threshold used to select instructions
        type=int,
        default=3,
    )
    
    parser.add_argument(
        "--use_ABDE6",
        type=float, # e.g., 0.33 = 33% of data is of this instruction type
        default=0,
    )
    parser.add_argument(
        "--use_E2",
        type=float,
        default=0,
    )
    parser.add_argument(
        "--use_E5",
        type=float,
        default=0,
    )
    parser.add_argument(
        "--use_E3",
        type=float,
        default=0,
    )
    parser.add_argument(
        "--use_E4",
        type=float,
        default=0,
    )

    parser.add_argument(
        "--A_0_json_path",
        type=str,
        default="",
    )
    parser.add_argument(
        "--A_1_0_json_path",
        type=str,
        default="",
    )
    parser.add_argument(
        "--B_0_3_json_path",
        type=str,
        default="",
    )
    parser.add_argument(
        "--B_1_3_json_path",
        type=str,
        default="",
    )
    parser.add_argument(
        "--B_1_4_json_path",
        type=str,
        default="",
    )
    parser.add_argument(
        "--D_3_json_path",
        type=str,
        default="",
    )
    parser.add_argument(
        "--D_4_json_path",
        type=str,
        default="",
    )
    parser.add_argument(
        "--E_6_json_path",
        type=str,
        default="",
    )
    parser.add_argument(
        "--E_2_json_path",
        type=str,
        default="",
    )
    parser.add_argument(
        "--E_5_json_path",
        type=str,
        default="",
    )
    parser.add_argument(
        "--E_3_json_path",
        type=str,
        default="",
    )
    parser.add_argument(
        "--E_4_json_path",
        type=str,
        default="",
    )


    # enhance instructions? 
    parser.add_argument(
        '--use_enhancement_1', 
        action='store_true',
        default=False,
    )
    parser.add_argument(
        '--use_enhancement_2', 
        action='store_true',
        default=False,
    )
    parser.add_argument(
        '--use_enhancement_3', 
        action='store_true',
        default=False,
    )
    parser.add_argument(
        '--use_enhancement_4', 
        action='store_true',
        default=False,
    )
    parser.add_argument(
        '--use_enhancement_5', 
        action='store_true',
        default=False,
    )
    parser.add_argument(
        '--use_enhancement_6', 
        action='store_true',
        default=False,
    )
    parser.add_argument(
        '--use_enhancement_7', 
        action='store_true',
        default=False,
    )
    parser.add_argument(
        '--use_enhancement_8', 
        action='store_true',
        default=False,
    )
    parser.add_argument(
        '--use_enhancement_9', 
        action='store_true',
        default=False,
    )

    args = parser.parse_args()

    main(args)








