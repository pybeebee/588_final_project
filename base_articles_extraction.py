### ADAPTED FROM: https://github.com/allenai/PRIMER
### Code to extract downloaded NewSHead articles into local directory as .pt files

import torch
import os
import pdb
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer
import random
import argparse
from multiprocessing import Pool
import numpy as np
from tqdm import tqdm
import re


def process_all_newshead(
    input_dir,
    max_length_input,
    max_length_output,
    mask_ratio,
    output_dir,
):

    for data_type in ["train", "valid", "test"]:
        # loop over all the files for each split
        all_files = [
            f
            for f in os.listdir(os.path.join(input_dir, data_type))
            if f.endswith(".pt")
        ]
        all_files = all_files[::-1]
        out_path = os.path.join(output_dir, data_type)
        print(out_path)

        # Set up counts to track progress throughout the loop
        cluster_count = 0
        article_count = 0
        clustered_data = []
        for i_f, f in enumerate(tqdm(all_files)):
            
            # if the file's already been addressed, skip it
            if os.path.exists(os.path.join(out_path, f)):
                continue
            
            # otherwise save the file as a list of the articles in the cluster
            all_data = torch.load(os.path.join(input_dir, data_type, f))
            for cluster_idx, cluster in enumerate(all_data):
                cluster_count += 1
                cluster_articles = [doc["text"] for doc in cluster["articles"]]
                article_count += len(cluster_articles)
                clustered_data.append(cluster_articles)
            

        torch.save(clustered_data, out_path+".pt")
        print("Finished saving %s data"%(data_type))
        print("Final cluster count for %s: %d"%(data_type, cluster_count))
        print("Final article count for %s: %d"%(data_type, article_count))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # General
    parser.add_argument("--max_length_input", default=4096, type=int)
    parser.add_argument("--max_length_output", default=512, type=int)
    parser.add_argument("--mask_ratio", default=0.3, type=float)
    parser.add_argument("--non_mask_ratio", default=0.5, type=float)
    parser.add_argument("--primer_path", default="./PRIMER", type=str)

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./base_articles/", # just the articles saved as clusters, each cluster is saved as a list
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="./newshead/",
    )
    
    args = parser.parse_args()
    print(args)

    process_all_newshead(
        args.input_dir,
        args.max_length_input,
        args.max_length_output,
        args.mask_ratio,
        args.output_dir,
    )