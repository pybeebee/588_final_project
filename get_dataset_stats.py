### HELPER FILE TO GET STATISTICS FOR DIFFERENT INSTRUCITON TYPES UNDER DIFFERENT CURATION SELECTION THRESHOLDS

import json
import os
import pandas as pd

print("##########################################")
print("############# THRESHOLD 3/4/5 ############")
print("##########################################")
A_0_json = os.path.join("", "train.json")  # fill in appropriate filepath for local use
A_0_df_full = pd.read_json(A_0_json, lines=True)
print("A_0_df_full")
print(A_0_df_full.shape[0])

A_1_0_json = os.path.join("", "train.json") # fill in appropriate filepath for local use
A_1_0_df_full = pd.read_json(A_1_0_json, lines=True)
print("A_1_0_df_full")
print(A_1_0_df_full.shape[0])

B_0_3_json = os.path.join("", "train.json") # fill in appropriate filepath for local use
B_0_3_df_full = pd.read_json(B_0_3_json, lines=True)
print("B_0_3_df_full")
print(B_0_3_df_full.shape[0])

B_1_3_json = os.path.join("", "train.json") # fill in appropriate filepath for local use
B_1_3_df_full = pd.read_json(B_1_3_json, lines=True)
print("B_1_3_df_full")
print(B_1_3_df_full.shape[0])
